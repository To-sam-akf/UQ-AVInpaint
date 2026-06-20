"""UQ-AV Latent Diffusion (P3) — training wrapper.

K=1 conditional AV latent diffusion for Mel inpainting.
Uses a frozen Mel autoencoder to operate in latent space.

Training flow:
  1. Encode clean/corrupted Mel → z_target / z_context
  2. Downsample mask and boundary_map → mask_z / boundary_map_z
  3. Encode video → video_tokens
  4. Sample timestep t, noise epsilon
  5. Forward diffusion (masked) → z_t
  6. UNet predicts epsilon_pred (conditioned on video_tokens)
  7. Diffusion loss only over missing latent region
  8. Optional boundary + sync auxiliary losses

Inference (K=1 DDIM):
  1. Encode context → z_context
  2. Start from z_T = mask_z * N(0,I) + (1-mask_z) * z_context
  3. Iteratively denoise with known-region clamp at each step
  4. Decode → mel_pred
  5. Compose with known Mel bins
"""

import copy
import os
import warnings
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.uq.mel_autoencoder import MelAutoencoder, time_gradient
from networks.uq.latent_diffusion_unet import LatentDiffusionUNet
from networks.uq.video_evidence_encoder import (
    VideoEvidenceEncoderP3,
    VideoConditionDummy,
)
from networks.uq.diffusion_schedule import (
    DiffusionSchedule,
    downsample_mask_2d,
    downsample_boundary_map,
    compose_known_region,
    compute_diffusion_loss,
)


P2_CONDITIONING_MODES = (
    "audio_video",
    "drop_video",
    "partial_audio_video",
    "wrong_video",
    "shuffled_video",
    "drop_audio",
)

UQ_PREDICTION_TYPES = ("epsilon", "x0", "v")


class PatchGANMelTeacher(nn.Module):
    """Frozen VIAI-A/PatchGAN generator used as a Mel-space teacher."""

    def __init__(self, hparams, checkpoint_path, device):
        super().__init__()
        from networks import Inpainting_Networks, New_Inpainting_Networks

        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "Mel_Encoder" not in checkpoint or "Mel_Decoder" not in checkpoint:
            raise RuntimeError(
                "PatchGAN teacher checkpoint must contain Mel_Encoder and "
                f"Mel_Decoder keys. Found: {list(checkpoint.keys())[:10]}"
            )
        if "netD" not in checkpoint:
            warnings.warn(
                "PatchGAN teacher checkpoint does not contain netD; using the "
                "generator as a reconstruction teacher only.",
                RuntimeWarning,
            )

        self.encoder = Inpainting_Networks.MelEncoder(hparams=hparams).to(device)
        self.decoder = New_Inpainting_Networks.MelDecoder(hparams=hparams).to(device)
        self.encoder.load_state_dict(checkpoint["Mel_Encoder"])
        self.decoder.load_state_dict(checkpoint["Mel_Decoder"])
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, mel_corrupted, missing_mask):
        mel_input = mel_corrupted[:, 0] if mel_corrupted.dim() == 4 else mel_corrupted
        pred = self.decoder(self.encoder(mel_input), mel_corrupted.size())
        return compose_known_region(pred, mel_corrupted, missing_mask)


class UQAVDiffusionModel:
    """Training / inference wrapper for K=1 AV latent diffusion inpainting."""

    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")

        # --- AE (frozen) ---
        ae_latent_dim = int(getattr(hparams, "ae_latent_dim", 8))
        ae_base_channels = int(getattr(hparams, "ae_base_channels", 32))
        ae_norm_type = getattr(hparams, "ae_norm_type", "batch")

        self.ae = MelAutoencoder(
            latent_dim=ae_latent_dim,
            base_channels=ae_base_channels,
            norm_type=ae_norm_type,
        ).to(self.device)
        # AE is frozen by default in P3
        for p in self.ae.parameters():
            p.requires_grad = False

        # Latent normalisation stats (set via load_ae_checkpoint / set_latent_stats)
        self.latent_mean = None
        self.latent_std = None
        self.latent_is_normalised = False

        # --- Diffusion schedule ---
        diff_timesteps = int(
            getattr(
                hparams, "uq_diffusion_timesteps",
                getattr(hparams, "diff_timesteps", 1000),
            )
        )
        diff_beta_start = float(
            getattr(
                hparams, "uq_beta_start",
                getattr(hparams, "diff_beta_start", 1e-4),
            )
        )
        diff_beta_end = float(
            getattr(
                hparams, "uq_beta_end",
                getattr(hparams, "diff_beta_end", 0.02),
            )
        )
        diff_schedule = getattr(
            hparams, "uq_beta_schedule",
            getattr(
                hparams, "uq_schedule_type",
                getattr(hparams, "diff_schedule", "linear"),
            ),
        )
        self.diffusion = DiffusionSchedule(
            timesteps=diff_timesteps,
            beta_start=diff_beta_start,
            beta_end=diff_beta_end,
            schedule=diff_schedule,
        )
        if self.device.type != "cpu":
            self.diffusion.to_device(self.device)
        self.prediction_type = str(
            getattr(hparams, "uq_prediction_type", "epsilon")
        )
        if self.prediction_type not in UQ_PREDICTION_TYPES:
            raise ValueError(
                "uq_prediction_type must be one of: "
                + ", ".join(UQ_PREDICTION_TYPES)
            )
        self.latent_clip_value = float(
            getattr(hparams, "uq_latent_clip_value", 4.0)
        )

        # --- Video encoder ---
        video_dim = int(getattr(hparams, "uq_video_dim", 256))
        self.use_video = not bool(getattr(hparams, "uq_no_video", False))
        if self.use_video:
            self.video_encoder = VideoEvidenceEncoderP3(
                video_dim=video_dim,
                image_size=int(getattr(hparams, "image_size", 256)),
            ).to(self.device)
        else:
            self.video_encoder = VideoConditionDummy(
                video_dim=video_dim,
            ).to(self.device)

        # --- Diffusion U-Net ---
        unet_base_channels = int(getattr(hparams, "uq_unet_base_channels", 64))
        unet_in_channels = 2 * ae_latent_dim + 3  # z_t + z_ctx + mask + bdy
        self.unet = LatentDiffusionUNet(
            in_channels=unet_in_channels,
            out_channels=ae_latent_dim,
            base_channels=unet_base_channels,
            time_emb_dim=int(getattr(hparams, "uq_time_emb_dim", 256)),
            video_dim=video_dim,
            num_heads=int(getattr(hparams, "uq_attn_heads", 4)),
            norm_type=ae_norm_type,
        ).to(self.device)

        # --- Loss weights ---
        self.lambda_boundary = float(
            getattr(hparams, "uq_lambda_boundary", 0.1)
        )
        self.lambda_sync = float(getattr(hparams, "uq_lambda_sync", 0.0))
        self.lambda_video_margin = float(
            getattr(hparams, "uq_lambda_video_margin", 0.0)
        )
        self.lambda_distill = float(getattr(hparams, "uq_lambda_distill", 0.0))
        self.teacher_type = str(getattr(hparams, "uq_teacher_type", "none"))
        if self.teacher_type not in {"none", "patchgan", "audio_only_diffusion"}:
            raise ValueError(
                "uq_teacher_type must be one of: none, patchgan, "
                "audio_only_diffusion"
            )
        self.video_margin = float(getattr(hparams, "uq_video_margin", 0.02))
        self.video_margin_negative = str(
            getattr(hparams, "uq_video_margin_negative", "cycle")
        )
        if self.video_margin_negative not in {
            "cycle", "batch_shuffle", "temporal_shuffle", "no_video",
        }:
            raise ValueError(
                "uq_video_margin_negative must be one of: cycle, "
                "batch_shuffle, temporal_shuffle, no_video"
            )

        # --- Optimizer ---
        lr = float(getattr(hparams, "uq_lr", None) or getattr(hparams, "lr", 1e-4))
        beta1 = float(getattr(hparams, "beta1", 0.5))
        beta2 = float(getattr(hparams, "beta2", 0.999))
        self.optimizer = torch.optim.Adam(
            list(self.unet.parameters()) + list(self.video_encoder.parameters()),
            lr=lr, betas=(beta1, beta2),
        )
        self.current_lr = lr
        self.use_ema = bool(getattr(hparams, "uq_use_ema", False))
        self.ema_decay = float(getattr(hparams, "uq_ema_decay", 0.999))
        self.ema_start_step = int(getattr(hparams, "uq_ema_start_step", 0))
        self.ema_state = None
        if self.use_ema:
            self._sync_ema_from_model()

        # --- Loss items ---
        self.loss_total_item = 0.0
        self.loss_diff_item = 0.0
        self.loss_boundary_item = 0.0
        self.loss_sync_item = 0.0
        self.loss_video_margin_item = 0.0
        self.loss_distill_item = 0.0
        self.loss_distill = torch.tensor(0.0, device=self.device)
        self.video_margin_l_original_item = 0.0
        self.video_margin_l_wrong_item = 0.0
        self.video_margin_negative_mode = "disabled"
        self.video_gate_mean_item = 0.0
        self.video_attn_norm_item = 0.0
        self.video_token_norm_item = 0.0
        self.video_gate_mean = torch.tensor(0.0, device=self.device)
        self.video_attn_norm = torch.tensor(0.0, device=self.device)
        self.video_token_norm = torch.tensor(0.0, device=self.device)
        self.condition_counts = {}
        self.condition_loss_sums = {}
        self.condition_loss_items = {}
        self.condition_ratio_items = {}

        # --- Placeholders for current batch ---
        self.mel_target = None
        self.mel_corrupted = None
        self.missing_mask = None
        self.boundary_map = None
        self.video = None
        self.flow = None
        self.video_original = None
        self.flow_original = None
        self._current_batch = None

        # Outputs
        self.mel_pred = None
        self.z_target = None
        self.z_context = None
        self.z_pred = None
        self.teacher = None
        self._setup_distill_teacher()

    # ------------------------------------------------------------------
    # AE checkpoint
    # ------------------------------------------------------------------
    def load_ae_checkpoint(self, checkpoint_path):
        """Load frozen AE weights and latent statistics."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "encoder" in ckpt and "decoder" in ckpt:
            self.ae.encoder.load_state_dict(ckpt["encoder"])
            self.ae.decoder.load_state_dict(ckpt["decoder"])
        else:
            raise RuntimeError(
                "AE checkpoint must contain 'encoder' and 'decoder' keys. "
                f"Found: {list(ckpt.keys())[:10]}"
            )
        self.latent_mean = ckpt.get("latent_mean")
        self.latent_std = ckpt.get("latent_std")
        if self.latent_mean is not None and self.latent_std is not None:
            self.latent_mean = self.latent_mean.to(self.device)
            self.latent_std = self.latent_std.to(self.device)
            self.latent_is_normalised = True
            print(
                "[UQ-AV] Loaded AE latent stats: "
                f"mean={self.latent_mean.tolist()[:3]}... "
                f"std={self.latent_std.tolist()[:3]}..."
            )
        else:
            self.latent_is_normalised = False
            print("[UQ-AV] No latent stats in AE checkpoint — "
                  "latents will NOT be normalised.")
            if bool(getattr(self.hparams, "uq_require_latent_stats", False)):
                raise RuntimeError(
                    "--uq_require_latent_stats was set, but the AE checkpoint "
                    "does not contain latent_mean and latent_std."
                )
        print(f"[UQ-AV] Loaded frozen AE from {checkpoint_path}")

    def set_latent_stats(self, mean, std):
        self.latent_mean = mean.to(self.device)
        self.latent_std = std.to(self.device)
        self.latent_is_normalised = True

    # ------------------------------------------------------------------
    # Data ingestion
    # ------------------------------------------------------------------
    def set_input(self, batch):
        """Accept a UQ AV batch dict and move tensors to device."""
        self._current_batch = batch
        self.sample_ids = batch["sample_id"]
        self.mask_specs = batch["mask_spec"]
        self.video_conditions = batch["video_condition"]
        self.conditioning_modes = batch.get(
            "conditioning_mode",
            ["audio_video"] * len(self.sample_ids),
        )

        self.mel_target = batch["mel_target"].float().to(self.device)
        self.mel_corrupted = batch["mel_corrupted"].float().to(self.device)
        self.missing_mask = batch["missing_mask"].float().to(self.device)
        self.boundary_map = batch["boundary_map"].float().to(self.device)
        self.video = batch["video"].float().to(self.device)
        self.flow = batch["flow"].float().to(self.device)
        self.video_original = batch.get("video_original", batch["video"])
        self.flow_original = batch.get("flow_original", batch["flow"])
        self.video_original = self.video_original.float().to(self.device)
        self.flow_original = self.flow_original.float().to(self.device)

        if "audio_target" in batch:
            self.audio_target = batch["audio_target"].float().to(self.device)

    # ------------------------------------------------------------------
    # Teacher / EMA helpers
    # ------------------------------------------------------------------
    def _distill_enabled(self):
        return self.teacher_type != "none" and self.lambda_distill > 0.0

    def _clone_hparams_for_audio_teacher(self):
        if hasattr(self.hparams, "__dict__"):
            values = copy.deepcopy(vars(self.hparams))
        else:
            values = {}
        values.update({
            "uq_teacher_type": "none",
            "uq_lambda_distill": 0.0,
            "uq_no_video": True,
            "uq_use_ema": False,
            "uq_ema_eval": False,
            "save_optimizer_state": False,
        })
        return SimpleNamespace(**values)

    def _setup_distill_teacher(self):
        if not self._distill_enabled():
            return
        checkpoint_path = getattr(self.hparams, "uq_teacher_checkpoint", None)
        if not checkpoint_path:
            raise RuntimeError(
                "--uq_teacher_checkpoint is required when "
                "--uq_teacher_type is not none and --uq_lambda_distill > 0."
            )

        if self.teacher_type == "patchgan":
            self.teacher = PatchGANMelTeacher(
                self.hparams, checkpoint_path, self.device,
            )
            print(f"[UQ-AV] Loaded PatchGAN teacher from {checkpoint_path}")
            return

        if self.teacher_type == "audio_only_diffusion":
            teacher_hparams = self._clone_hparams_for_audio_teacher()
            teacher = UQAVDiffusionModel(teacher_hparams, device=self.device)
            ae_checkpoint = (
                getattr(self.hparams, "uq_teacher_ae_checkpoint", None)
                or getattr(self.hparams, "ae_checkpoint", None)
            )
            if not ae_checkpoint:
                raise RuntimeError(
                    "Audio-only diffusion teacher requires "
                    "--uq_teacher_ae_checkpoint or --ae_checkpoint."
                )
            teacher.load_ae_checkpoint(ae_checkpoint)
            teacher.load_checkpoint(checkpoint_path, reset_optimizer=True)
            teacher.unet.eval()
            teacher.video_encoder.eval()
            teacher.ae.eval()
            for module in (teacher.unet, teacher.video_encoder, teacher.ae):
                for param in module.parameters():
                    param.requires_grad = False
            self.teacher = teacher
            print(
                "[UQ-AV] Loaded audio-only diffusion teacher from "
                f"{checkpoint_path}"
            )

    @torch.no_grad()
    def _teacher_completed_mel(self):
        if not self._distill_enabled() or self.teacher is None:
            return None
        if self.teacher_type == "patchgan":
            return self.teacher(
                self.mel_corrupted, self.missing_mask,
            ).detach()
        if self.teacher_type == "audio_only_diffusion":
            devices = []
            if self.device.type == "cuda":
                devices = [
                    self.device.index
                    if self.device.index is not None
                    else torch.cuda.current_device()
                ]
            with torch.random.fork_rng(devices=devices):
                result = self.teacher.sample(
                    self._current_batch,
                    num_candidates=1,
                    inference_steps=int(
                        getattr(self.hparams, "uq_teacher_inference_steps", 50)
                    ),
                    ddim_eta=float(
                        getattr(self.hparams, "uq_teacher_ddim_eta", 0.0)
                    ),
                    seed=getattr(self.hparams, "eval_seed", None),
                )
            return result["completed_mels"][:, 0].detach()
        raise RuntimeError(f"Unsupported teacher type: {self.teacher_type}")

    def _distill_loss(self, mel_pred):
        teacher_completed = self._teacher_completed_mel()
        if teacher_completed is None:
            return torch.tensor(0.0, device=self.device)
        teacher_completed = teacher_completed.to(
            device=self.device, dtype=mel_pred.dtype,
        )
        diff = torch.abs(mel_pred - teacher_completed) * self.missing_mask.float()
        return diff.sum() / self.missing_mask.float().sum().clamp(min=1.0)

    def _ema_modules(self):
        return {
            "unet": self.unet,
            "video_encoder": self.video_encoder,
        }

    def _clone_state_dict(self, module):
        return {
            key: value.detach().clone()
            for key, value in module.state_dict().items()
        }

    def _sync_ema_from_model(self):
        self.ema_state = {
            name: self._clone_state_dict(module)
            for name, module in self._ema_modules().items()
        }

    def _update_ema(self, global_step):
        if not self.use_ema:
            return
        if self.ema_state is None or global_step <= self.ema_start_step:
            self._sync_ema_from_model()
            return
        decay = float(self.ema_decay)
        with torch.no_grad():
            for name, module in self._ema_modules().items():
                current = module.state_dict()
                shadow = self.ema_state[name]
                for key, value in current.items():
                    if torch.is_floating_point(value):
                        shadow[key].mul_(decay).add_(
                            value.detach(), alpha=1.0 - decay,
                        )
                    else:
                        shadow[key].copy_(value.detach())

    def apply_ema_weights(self):
        if self.ema_state is None:
            raise RuntimeError("EMA weights are not available in this checkpoint.")
        for name, module in self._ema_modules().items():
            if name == "video_encoder" and not self.use_video:
                continue
            state = self.ema_state.get(name)
            if state is not None:
                module.load_state_dict(state)
        print("[UQ-AV] Applied EMA weights for evaluation.")

    def _apply_video_conditioning_modes(self, video_out):
        modes = list(getattr(self, "conditioning_modes", []))
        if not modes or "drop_video" not in modes:
            return video_out

        drop_mask = torch.tensor(
            [mode == "drop_video" for mode in modes],
            device=self.device,
            dtype=torch.bool,
        )
        if not bool(drop_mask.any()):
            return video_out

        output = {}
        for key, value in video_out.items():
            if not torch.is_tensor(value) or value.size(0) != drop_mask.size(0):
                output[key] = value
                continue
            value = value.clone()
            value[drop_mask] = 0.0
            output[key] = value
        return output

    def _diffusion_loss_per_sample(self, prediction, target, mask_z):
        diff = (prediction - target) ** 2
        masked_diff = diff * mask_z.float()
        numer = masked_diff.sum(dim=(1, 2, 3))
        denom = mask_z.float().sum(dim=(1, 2, 3)).clamp(min=1.0)
        return numer / denom

    def _record_condition_losses(self, per_sample_losses):
        modes = list(getattr(self, "conditioning_modes", []))
        self.condition_counts = {}
        self.condition_loss_sums = {}
        self.condition_loss_items = {}
        self.condition_ratio_items = {}
        if not modes:
            return
        losses = per_sample_losses.detach()
        total_count = len(modes)
        for mode in P2_CONDITIONING_MODES:
            indices = [
                index for index, value in enumerate(modes)
                if value == mode
            ]
            if not indices:
                continue
            index_tensor = torch.tensor(
                indices, device=losses.device, dtype=torch.long,
            )
            selected = losses.index_select(0, index_tensor)
            self.condition_counts[mode] = len(indices)
            self.condition_loss_sums[mode] = float(selected.sum().cpu().item())
            self.condition_loss_items[mode] = float(selected.mean().cpu().item())
            self.condition_ratio_items[mode] = len(indices) / max(1, total_count)

    def _video_margin_enabled(self):
        return self.use_video and self.lambda_video_margin > 0.0

    def _resolve_video_margin_negative_mode(self, global_step):
        mode = self.video_margin_negative
        if mode != "cycle":
            return mode
        modes = ("batch_shuffle", "temporal_shuffle", "no_video")
        return modes[int(global_step) % len(modes)]

    def _negative_video_pair(self, video, flow, global_step):
        mode = self._resolve_video_margin_negative_mode(global_step)
        if mode == "batch_shuffle":
            if video.size(0) > 1:
                return video.roll(shifts=1, dims=0), flow.roll(shifts=1, dims=0), mode
            mode = "no_video"
        if mode == "temporal_shuffle":
            if video.size(1) > 1:
                shift = int(global_step) % (video.size(1) - 1) + 1
                return (
                    video.roll(shifts=shift, dims=1),
                    flow.roll(shifts=shift, dims=1),
                    mode,
                )
            mode = "no_video"
        if mode == "no_video":
            return torch.zeros_like(video), torch.zeros_like(flow), mode
        raise ValueError(f"Unsupported video margin negative mode: {mode}")

    def _record_unet_video_diagnostics(self):
        self.video_gate_mean = getattr(
            self.unet,
            "video_gate_mean",
            torch.tensor(0.0, device=self.device),
        )
        self.video_attn_norm = getattr(
            self.unet,
            "video_attn_norm",
            torch.tensor(0.0, device=self.device),
        )
        self.video_token_norm = getattr(
            self.unet,
            "video_token_norm",
            torch.tensor(0.0, device=self.device),
        )

    def _extract_alpha_terms(self, t, ref):
        alpha_bar = self.diffusion.alphas_cumprod[t].to(ref.device)[
            :, None, None, None
        ]
        sqrt_alpha = torch.sqrt(alpha_bar)
        sqrt_one_minus = torch.sqrt((1.0 - alpha_bar).clamp(min=1e-12))
        return sqrt_alpha, sqrt_one_minus

    def _clip_predicted_x0(self, z_0_pred):
        if self.latent_clip_value > 0.0:
            clip = float(self.latent_clip_value)
            return torch.clamp(z_0_pred, -clip, clip)
        return z_0_pred

    def _diffusion_target(self, z_0, epsilon, t):
        if self.prediction_type == "epsilon":
            return epsilon
        if self.prediction_type == "x0":
            return z_0
        sqrt_alpha, sqrt_one_minus = self._extract_alpha_terms(t, z_0)
        return sqrt_alpha * epsilon - sqrt_one_minus * z_0

    def _epsilon_from_model_output(self, z_t, model_output, t):
        if self.prediction_type == "epsilon":
            return model_output
        sqrt_alpha, sqrt_one_minus = self._extract_alpha_terms(t, z_t)
        if self.prediction_type == "x0":
            return (z_t - sqrt_alpha * model_output) / sqrt_one_minus
        if self.prediction_type == "v":
            return sqrt_alpha * model_output + sqrt_one_minus * z_t
        raise RuntimeError(f"Unsupported prediction type: {self.prediction_type}")

    def _z0_from_model_output(self, z_t, model_output, t):
        sqrt_alpha, sqrt_one_minus = self._extract_alpha_terms(t, z_t)
        if self.prediction_type == "epsilon":
            z_0_pred = (z_t - sqrt_one_minus * model_output) / sqrt_alpha
        elif self.prediction_type == "x0":
            z_0_pred = model_output
        elif self.prediction_type == "v":
            z_0_pred = sqrt_alpha * z_t - sqrt_one_minus * model_output
        else:
            raise RuntimeError(
                f"Unsupported prediction type: {self.prediction_type}"
            )
        return self._clip_predicted_x0(z_0_pred)

    def _predict_model_output(self, z_t, t, video_tokens,
                              record_diagnostics=True):
        unet_input = torch.cat(
            [z_t, self.z_context, self.mask_z, self.boundary_map_z], dim=1,
        )
        model_output = self.unet(unet_input, t, video_tokens=video_tokens)
        if record_diagnostics:
            self._record_unet_video_diagnostics()
        return model_output

    def _z0_from_epsilon(self, z_t, epsilon_pred, t):
        sqrt_alpha, sqrt_one_minus = self._extract_alpha_terms(t, z_t)
        return self._clip_predicted_x0(
            (z_t - sqrt_one_minus * epsilon_pred) / sqrt_alpha
        )

    def _missing_mel_l1_per_sample(self, mel_pred):
        diff = (mel_pred - self.mel_target).abs() * self.missing_mask.float()
        numer = diff.sum(dim=(1, 2, 3))
        denom = self.missing_mask.float().sum(dim=(1, 2, 3)).clamp(min=1.0)
        return numer / denom

    def _video_margin_loss(self, z_t, t, global_step):
        if not self._video_margin_enabled():
            self.video_margin_negative_mode = "disabled"
            zero = torch.tensor(0.0, device=self.device)
            return zero, zero.detach(), zero.detach()

        original_out = self.video_encoder(self.video_original, self.flow_original)
        original_tokens = original_out["video_tokens"]
        negative_video, negative_flow, negative_mode = self._negative_video_pair(
            self.video_original, self.flow_original, global_step,
        )
        negative_out = self.video_encoder(negative_video, negative_flow)
        negative_tokens = negative_out["video_tokens"]

        original_pred = self._predict_model_output(
            z_t, t, original_tokens, record_diagnostics=False,
        )
        negative_pred = self._predict_model_output(
            z_t, t, negative_tokens, record_diagnostics=False,
        )
        mel_original = self._decode_latent(
            self._z0_from_model_output(z_t, original_pred, t)
        )
        mel_negative = self._decode_latent(
            self._z0_from_model_output(z_t, negative_pred, t)
        )
        l_original = self._missing_mel_l1_per_sample(mel_original)
        l_wrong = self._missing_mel_l1_per_sample(mel_negative)
        loss = F.relu(self.video_margin + l_original - l_wrong).mean()
        self.video_margin_negative_mode = negative_mode
        return loss, l_original.mean(), l_wrong.mean()

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------
    def _encode_mel(self, mel):
        """Encode Mel → latent (with optional normalisation)."""
        z = self.ae.encode(mel)
        if self.latent_is_normalised:
            mean = self.latent_mean[None, :, None, None]
            std = self.latent_std[None, :, None, None]
            z = (z - mean) / std
        return z

    def _decode_latent(self, z):
        """Decode latent → Mel (with optional denormalisation)."""
        if self.latent_is_normalised:
            mean = self.latent_mean[None, :, None, None]
            std = self.latent_std[None, :, None, None]
            z = z * std + mean
        return self.ae.decode(z)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def optimize_parameters(self, global_step):
        self.unet.train()
        self.video_encoder.train()
        self.ae.eval()
        # Clear stale mel_pred from a previous test()/val call — it may have a
        # different batch size than the current training batch (e.g. last val
        # batch had 4 samples but current train batch has 8).
        self.mel_pred = None

        # Encode to latent space (frozen AE)
        with torch.no_grad():
            self.z_target = self._encode_mel(self.mel_target)
            self.z_context = self._encode_mel(self.mel_corrupted)

            # Downsample mask / boundary map to latent resolution
            self.mask_z = downsample_mask_2d(
                self.missing_mask, target_size=(10, 50)
            ).to(self.device)
            self.boundary_map_z = downsample_boundary_map(
                self.boundary_map, target_size=(10, 50)
            ).to(self.device)

        # Video conditioning is trainable in P3; keep it outside no_grad.
        video_out = self.video_encoder(self.video, self.flow)
        video_out = self._apply_video_conditioning_modes(video_out)
        self.video_tokens = video_out["video_tokens"]

        # Sample random timesteps
        B = self.z_target.size(0)
        t = torch.randint(
            0, self.diffusion.timesteps, (B,),
            device=self.device, dtype=torch.long,
        )

        # Sample noise
        epsilon = torch.randn_like(self.z_target)

        # Masked forward diffusion
        z_t = self.diffusion.masked_q_sample(
            self.z_target, self.z_context, self.mask_z, t, noise=epsilon,
        )

        # Predict the configured diffusion target (epsilon, x0, or v).
        model_output = self._predict_model_output(
            z_t, t, video_tokens=self.video_tokens,
        )
        diffusion_target = self._diffusion_target(self.z_target, epsilon, t)

        # Diffusion loss (masked)
        self.loss_diff = compute_diffusion_loss(
            model_output, diffusion_target, self.mask_z,
        )
        self.loss_diff_per_sample = self._diffusion_loss_per_sample(
            model_output, diffusion_target, self.mask_z,
        )
        self._record_condition_losses(self.loss_diff_per_sample)

        # Boundary/distill losses keep gradient through the frozen decoder.
        z_0_pred = self._z0_from_model_output(z_t, model_output, t)
        mel_0_pred = self._decode_latent(z_0_pred)

        self.loss_boundary = self._boundary_loss(
            mel_0_pred, self.mel_target, self.missing_mask,
        )
        self.loss_distill = self._distill_loss(mel_0_pred)

        # Sync loss (P3: minimal — just cosine similarity on global features)
        if self.lambda_sync > 0:
            self.loss_sync = self._sync_loss(video_out)
        else:
            self.loss_sync = torch.tensor(0.0, device=self.device)

        (
            self.loss_video_margin,
            self.video_margin_l_original,
            self.video_margin_l_wrong,
        ) = self._video_margin_loss(z_t, t, global_step)

        # Total loss
        self.loss_total = (
            self.loss_diff
            + self.lambda_boundary * self.loss_boundary
            + self.lambda_sync * self.loss_sync
            + self.lambda_video_margin * self.loss_video_margin
            + self.lambda_distill * self.loss_distill
        )

        # Backward
        self.optimizer.zero_grad()
        self.loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.unet.parameters())
            + list(self.video_encoder.parameters()),
            max_norm=float(getattr(self.hparams, "uq_grad_clip", 1.0)),
        )
        self.optimizer.step()
        self._update_ema(global_step)
        self.current_lr = self.optimizer.param_groups[0]["lr"]

    def _boundary_loss(self, mel_pred, mel_target, missing_mask):
        """Gradient L1 around mask boundaries."""
        grad_pred = time_gradient(mel_pred)
        grad_target = time_gradient(mel_target)

        # Focus on boundary region: expand mask by a few frames
        mask = missing_mask[:, :1, :, :]  # [B, 1, 80, 200]
        # Simple: weight by mask (focus on missing region + nearby)
        weight = F.max_pool2d(
            mask, kernel_size=(1, 5), stride=1, padding=(0, 2),
        )
        weight = weight.clamp(0, 1)  # [B, 1, 80, 200]

        # Trim weight to match gradient length
        weight = weight[:, :, :, :grad_pred.size(-1)]

        diff = F.l1_loss(
            grad_pred * weight, grad_target * weight,
            reduction="sum",
        ) / weight.sum().clamp(min=1.0)
        return diff

    def _sync_loss(self, video_out):
        """Simple cosine-similarity based sync signal (placeholder for P5)."""
        # Use global average of video tokens as sync target
        vid_global = video_out["video_tokens"].mean(dim=1)  # [B, D]
        # Use audio context global feature
        aud_global = self.z_context.mean(dim=(2, 3))  # [B, C]
        # Project to same dim if needed
        if aud_global.size(-1) != vid_global.size(-1):
            return torch.tensor(0.0, device=self.device)
        cos_sim = F.cosine_similarity(aud_global, vid_global, dim=-1)
        return (1.0 - cos_sim).mean()

    # ------------------------------------------------------------------
    # Inference (K=1 DDIM)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, batch, num_candidates=1, inference_steps=50,
               ddim_eta=0.0, seed=None):
        """K-candidate DDIM sampling.  P3 defaults to K=1.

        Args:
            batch: UQ AV batch dict
            num_candidates: K (P3: always 1)
            inference_steps: DDIM steps (default 50)
            ddim_eta: 0=deterministic, 1=stochastic
            seed: optional int seed per candidate
        Returns:
            dict with candidate_mels, completed_mels, etc.
        """
        self.unet.eval()
        self.video_encoder.eval()
        self.ae.eval()

        self.set_input(batch)

        # Encode
        self.z_context = self._encode_mel(self.mel_corrupted)
        self.mask_z = downsample_mask_2d(self.missing_mask).to(self.device)
        self.boundary_map_z = downsample_boundary_map(self.boundary_map).to(
            self.device
        )
        video_out = self.video_encoder(self.video, self.flow)
        video_out = self._apply_video_conditioning_modes(video_out)
        self.video_tokens = video_out["video_tokens"]

        B = self.z_context.size(0)
        K = int(num_candidates)
        if K != 1:
            raise ValueError(
                f"P3 only supports K=1 sampling; got num_candidates={K}."
            )

        candidate_mels = []
        candidate_latents = []

        for k in range(K):
            if seed is not None:
                torch.manual_seed(int(seed) + k)

            # Start from noise in missing region
            z_t = torch.randn_like(self.z_context)
            z_t = compose_known_region(z_t, self.z_context, self.mask_z)

            # DDIM sampling loop
            ddim_ts = self.diffusion.get_ddim_timesteps(inference_steps)
            for i in range(len(ddim_ts) - 1):
                t = ddim_ts[i].to(self.device)
                t_next = ddim_ts[i + 1].to(self.device)

                t_batch = t.expand(B).long()
                t_next_batch = t_next.expand(B).long()

                model_output = self._predict_model_output(
                    z_t, t_batch, self.video_tokens,
                )
                epsilon_pred = self._epsilon_from_model_output(
                    z_t, model_output, t_batch,
                )

                z_t = self.diffusion.ddim_step(
                    z_t, epsilon_pred, t_batch, t_next_batch,
                    eta=ddim_eta,
                    clamp_mask=self.mask_z,
                    z_context=self.z_context,
                    x0_clip_value=self.latent_clip_value,
                )

            # Decode final latent
            mel_pred = self._decode_latent(z_t)

            # Compose with known Mel bins
            mel_completed = compose_known_region(
                mel_pred, self.mel_corrupted, self.missing_mask,
            )

            candidate_mels.append(mel_pred)
            candidate_latents.append(z_t)

        # Stack: [B, K, C, H, W] for mels
        candidate_mels = torch.stack(candidate_mels, dim=1)  # [B, K, 1, 80, 200]
        candidate_latents = torch.stack(candidate_latents, dim=1)  # [B, K, 8, 10, 50]

        # completed mels with known-region compose
        completed_mels = torch.stack(
            [
                compose_known_region(
                    candidate_mels[:, i],
                    self.mel_corrupted,
                    self.missing_mask,
                )
                for i in range(K)
            ],
            dim=1,
        )

        return {
            "candidate_mels": candidate_mels,
            "completed_mels": completed_mels,
            "candidate_latents": candidate_latents,
            "candidate_scores": None,
            "uncertainty": None,
            "visual_evidence": None,
        }

    # ------------------------------------------------------------------
    # Test (single forward without backward)
    # ------------------------------------------------------------------
    def test(self, global_step=0):
        self.unet.eval()
        self.video_encoder.eval()
        self.ae.eval()

        with torch.no_grad():
            self.z_target = self._encode_mel(self.mel_target)
            self.z_context = self._encode_mel(self.mel_corrupted)
            self.mask_z = downsample_mask_2d(self.missing_mask).to(self.device)
            self.boundary_map_z = downsample_boundary_map(
                self.boundary_map
            ).to(self.device)

            video_out = self.video_encoder(self.video, self.flow)
            video_out = self._apply_video_conditioning_modes(video_out)
            self.video_tokens = video_out["video_tokens"]

            B = self.z_target.size(0)
            t = torch.zeros(B, device=self.device, dtype=torch.long)
            epsilon = torch.randn_like(self.z_target)
            z_t = self.diffusion.masked_q_sample(
                self.z_target, self.z_context, self.mask_z, t,
                noise=epsilon,
            )

            model_output = self._predict_model_output(
                z_t, t, self.video_tokens,
            )
            diffusion_target = self._diffusion_target(
                self.z_target, epsilon, t,
            )
            self.loss_diff = compute_diffusion_loss(
                model_output, diffusion_target, self.mask_z,
            )
            self.loss_diff_per_sample = self._diffusion_loss_per_sample(
                model_output, diffusion_target, self.mask_z,
            )
            self._record_condition_losses(self.loss_diff_per_sample)

            # Decode predicted z_0
            z_0_pred = self._z0_from_model_output(z_t, model_output, t)
            mel_0_pred = self._decode_latent(z_0_pred)
            self.mel_pred = compose_known_region(
                mel_0_pred, self.mel_corrupted, self.missing_mask,
            )

            self.loss_boundary = self._boundary_loss(
                mel_0_pred, self.mel_target, self.missing_mask,
            )
            self.loss_distill = self._distill_loss(mel_0_pred)
            self.loss_sync = (
                self._sync_loss(video_out)
                if self.lambda_sync > 0
                else torch.tensor(0.0, device=self.device)
            )
            self.loss_video_margin = torch.tensor(0.0, device=self.device)
            self.video_margin_l_original = torch.tensor(0.0, device=self.device)
            self.video_margin_l_wrong = torch.tensor(0.0, device=self.device)
            self.video_margin_negative_mode = "disabled"

            self.loss_total = (
                self.loss_diff
                + self.lambda_boundary * self.loss_boundary
                + self.lambda_sync * self.loss_sync
                + self.lambda_video_margin * self.loss_video_margin
                + self.lambda_distill * self.loss_distill
            )

    # ------------------------------------------------------------------
    # Loss items & logging
    # ------------------------------------------------------------------
    def get_loss_items(self):
        self.loss_total_item = float(self.loss_total.detach().cpu().item())
        self.loss_diff_item = float(self.loss_diff.detach().cpu().item())
        self.loss_boundary_item = float(
            self.loss_boundary.detach().cpu().item()
        )
        self.loss_sync_item = float(self.loss_sync.detach().cpu().item())
        self.loss_video_margin_item = float(
            self.loss_video_margin.detach().cpu().item()
        )
        self.loss_distill_item = float(
            self.loss_distill.detach().cpu().item()
        )
        self.video_margin_l_original_item = float(
            self.video_margin_l_original.detach().cpu().item()
        )
        self.video_margin_l_wrong_item = float(
            self.video_margin_l_wrong.detach().cpu().item()
        )
        self.video_gate_mean_item = float(
            self.video_gate_mean.detach().cpu().item()
        )
        self.video_attn_norm_item = float(
            self.video_attn_norm.detach().cpu().item()
        )
        self.video_token_norm_item = float(
            self.video_token_norm.detach().cpu().item()
        )
        self.condition_loss_items = dict(getattr(self, "condition_loss_items", {}))
        self.condition_ratio_items = dict(getattr(self, "condition_ratio_items", {}))

    def get_current_errors(self):
        errors = {
            "loss_total": self.loss_total_item,
            "loss_diff": self.loss_diff_item,
            "loss_boundary": self.loss_boundary_item,
            "loss_sync": self.loss_sync_item,
            "loss_video_margin": self.loss_video_margin_item,
            "loss_distill": self.loss_distill_item,
            "video_margin_l_original": self.video_margin_l_original_item,
            "video_margin_l_wrong": self.video_margin_l_wrong_item,
            "video_margin_negative_mode": self.video_margin_negative_mode,
            "video_gate_mean": self.video_gate_mean_item,
            "video_attn_norm": self.video_attn_norm_item,
            "video_token_norm": self.video_token_norm_item,
            "lr": self.current_lr,
        }
        for mode, value in self.condition_loss_items.items():
            errors[f"cond_{mode}_loss_diff"] = value
        for mode, value in self.condition_ratio_items.items():
            errors[f"cond_{mode}_ratio"] = value
        return errors

    def TF_writer(self, writer, step, prefix="train"):
        if writer is None:
            return
        writer.add_scalar(f"{prefix}/loss_total", self.loss_total_item, step)
        writer.add_scalar(f"{prefix}/loss_diff", self.loss_diff_item, step)
        writer.add_scalar(
            f"{prefix}/loss_boundary", self.loss_boundary_item, step,
        )
        writer.add_scalar(
            f"{prefix}/loss_sync", self.loss_sync_item, step,
        )
        writer.add_scalar(
            f"{prefix}/loss_video_margin",
            self.loss_video_margin_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/loss_distill", self.loss_distill_item, step,
        )
        writer.add_scalar(
            f"{prefix}/video_margin_l_original",
            self.video_margin_l_original_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/video_margin_l_wrong",
            self.video_margin_l_wrong_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/video_gate_mean",
            self.video_gate_mean_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/video_attn_norm",
            self.video_attn_norm_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/video_token_norm",
            self.video_token_norm_item,
            step,
        )
        if hasattr(writer, "add_text"):
            writer.add_text(
                f"{prefix}/video_margin_negative_mode",
                str(self.video_margin_negative_mode),
                step,
            )
        for mode, value in self.condition_loss_items.items():
            writer.add_scalar(
                f"{prefix}/cond_{mode}_loss_diff", value, step,
            )
        for mode, value in self.condition_ratio_items.items():
            writer.add_scalar(
                f"{prefix}/cond_{mode}_ratio", value, step,
            )
        writer.add_scalar(f"{prefix}/lr", self.current_lr, step)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def _state_to_cpu(self, state):
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in state.items()
        }

    def _ema_state_to_cpu(self):
        if self.ema_state is None:
            return None
        return {
            name: self._state_to_cpu(state)
            for name, state in self.ema_state.items()
        }

    def _ema_filename(self, filename):
        if filename.endswith(".pth.tar"):
            return filename[:-8] + "_ema.pth.tar"
        root, ext = os.path.splitext(filename)
        return root + "_ema" + ext

    def save_checkpoint(
        self, global_step, global_epoch, checkpoint_dir, filename=None,
    ):
        os.makedirs(checkpoint_dir, exist_ok=True)
        if filename is None:
            filename = f"UQ-AV_checkpoint_step{global_step:09d}.pth.tar"
        checkpoint_path = os.path.join(
            checkpoint_dir,
            filename,
        )
        checkpoint = {
            "unet": self.unet.state_dict(),
            "video_encoder": self.video_encoder.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if getattr(self.hparams, "save_optimizer_state", True)
            else None,
            "global_step": global_step,
            "global_epoch": global_epoch,
            "hparams": {
                "ae_latent_dim": int(
                    getattr(self.hparams, "ae_latent_dim", 8)
                ),
                "uq_video_dim": int(
                    getattr(self.hparams, "uq_video_dim", 256)
                ),
                "uq_prediction_type": self.prediction_type,
                "uq_beta_schedule": getattr(
                    self.diffusion, "schedule", "linear",
                ),
                "uq_latent_clip_value": self.latent_clip_value,
            },
            "latent_mean": self.latent_mean.cpu()
            if self.latent_mean is not None else None,
            "latent_std": self.latent_std.cpu()
            if self.latent_std is not None else None,
            "ema": self._ema_state_to_cpu(),
            "ema_decay": self.ema_decay,
            "ema_start_step": self.ema_start_step,
            "is_ema_checkpoint": False,
        }
        torch.save(checkpoint, checkpoint_path)
        print("[UQ-AV] Saved checkpoint:", checkpoint_path)
        if self.use_ema and self.ema_state is not None:
            ema_filename = self._ema_filename(filename)
            ema_path = os.path.join(checkpoint_dir, ema_filename)
            ema_checkpoint = dict(checkpoint)
            ema_checkpoint["unet"] = checkpoint["ema"]["unet"]
            ema_checkpoint["video_encoder"] = checkpoint["ema"]["video_encoder"]
            ema_checkpoint["optimizer"] = None
            ema_checkpoint["is_ema_checkpoint"] = True
            torch.save(ema_checkpoint, ema_path)
            print("[UQ-AV] Saved EMA checkpoint:", ema_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, reset_optimizer=False,
                        use_ema=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        requested_ema = (
            bool(getattr(self.hparams, "uq_ema_eval", False))
            if use_ema is None else bool(use_ema)
        )
        is_ema_checkpoint = bool(checkpoint.get("is_ema_checkpoint", False))
        self.ema_state = checkpoint.get("ema")
        if self.ema_state is not None:
            # map_location already moved tensors, but clone to detach from ckpt.
            self.ema_state = {
                name: {
                    key: value.detach().clone()
                    if torch.is_tensor(value) else value
                    for key, value in state.items()
                }
                for name, state in self.ema_state.items()
            }

        state_unet = checkpoint["unet"]
        state_video = checkpoint["video_encoder"]
        if requested_ema:
            if self.ema_state is None and not is_ema_checkpoint:
                raise RuntimeError(
                    "--uq_ema_eval was requested, but checkpoint does not "
                    "contain EMA weights. Use an EMA checkpoint or a raw "
                    "checkpoint saved with --uq_use_ema."
                )
            if self.ema_state is not None:
                state_unet = self.ema_state["unet"]
                state_video = self.ema_state["video_encoder"]

        self.unet.load_state_dict(state_unet)
        if self.use_video:
            self.video_encoder.load_state_dict(state_video)
        else:
            print(
                "[UQ-AV] Skipping video_encoder weights because "
                "--uq_no_video uses VideoConditionDummy."
            )
        if (
            not requested_ema
            and not reset_optimizer
            and checkpoint.get("optimizer") is not None
        ):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except ValueError as exc:
                print(
                    "[UQ-AV] Skipping optimizer state due to parameter "
                    f"mismatch: {exc}"
                )
        self.latent_mean = checkpoint.get("latent_mean")
        self.latent_std = checkpoint.get("latent_std")
        if self.latent_mean is not None and self.latent_std is not None:
            self.latent_mean = self.latent_mean.to(self.device)
            self.latent_std = self.latent_std.to(self.device)
            self.latent_is_normalised = True
        if self.use_ema and self.ema_state is None:
            self._sync_ema_from_model()
        if requested_ema:
            print("[UQ-AV] Loaded EMA weights for evaluation.")
        global_step = int(checkpoint.get("global_step", 0))
        global_epoch = int(checkpoint.get("global_epoch", 0))
        self._loaded_step = global_step
        self._loaded_epoch = global_epoch
        return global_step, global_epoch
