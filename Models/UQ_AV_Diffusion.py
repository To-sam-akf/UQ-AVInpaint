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

import os
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

        # --- Optimizer ---
        lr = float(getattr(hparams, "uq_lr", None) or getattr(hparams, "lr", 1e-4))
        beta1 = float(getattr(hparams, "beta1", 0.5))
        beta2 = float(getattr(hparams, "beta2", 0.999))
        self.optimizer = torch.optim.Adam(
            list(self.unet.parameters()) + list(self.video_encoder.parameters()),
            lr=lr, betas=(beta1, beta2),
        )
        self.current_lr = lr

        # --- Loss items ---
        self.loss_total_item = 0.0
        self.loss_diff_item = 0.0
        self.loss_boundary_item = 0.0
        self.loss_sync_item = 0.0

        # --- Placeholders for current batch ---
        self.mel_target = None
        self.mel_corrupted = None
        self.missing_mask = None
        self.boundary_map = None
        self.video = None
        self.flow = None

        # Outputs
        self.mel_pred = None
        self.z_target = None
        self.z_context = None
        self.z_pred = None

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
            print("[UQ-AV] No latent stats in AE checkpoint — "
                  "latents will NOT be normalised.")
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
        self.sample_ids = batch["sample_id"]
        self.mask_specs = batch["mask_spec"]
        self.video_conditions = batch["video_condition"]

        self.mel_target = batch["mel_target"].float().to(self.device)
        self.mel_corrupted = batch["mel_corrupted"].float().to(self.device)
        self.missing_mask = batch["missing_mask"].float().to(self.device)
        self.boundary_map = batch["boundary_map"].float().to(self.device)
        self.video = batch["video"].float().to(self.device)
        self.flow = batch["flow"].float().to(self.device)

        if "audio_target" in batch:
            self.audio_target = batch["audio_target"].float().to(self.device)

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

        # U-Net input: concat [z_t, z_context, mask_z, boundary_map_z]
        unet_input = torch.cat(
            [z_t, self.z_context, self.mask_z, self.boundary_map_z], dim=1,
        )

        # Predict epsilon
        epsilon_pred = self.unet(unet_input, t, video_tokens=self.video_tokens)

        # Diffusion loss (masked)
        self.loss_diff = compute_diffusion_loss(
            epsilon_pred, epsilon, self.mask_z,
        )

        # Boundary loss: keep gradient through epsilon_pred and frozen decoder.
        alpha_bar_t = self.diffusion.alphas_cumprod[t].to(self.device)[
            :, None, None, None
        ]
        z_0_pred = (
            z_t - torch.sqrt(1.0 - alpha_bar_t) * epsilon_pred
        ) / torch.sqrt(alpha_bar_t)
        mel_0_pred = self._decode_latent(z_0_pred)

        self.loss_boundary = self._boundary_loss(
            mel_0_pred, self.mel_target, self.missing_mask,
        )

        # Sync loss (P3: minimal — just cosine similarity on global features)
        if self.lambda_sync > 0:
            self.loss_sync = self._sync_loss(video_out)
        else:
            self.loss_sync = torch.tensor(0.0, device=self.device)

        # Total loss
        self.loss_total = (
            self.loss_diff
            + self.lambda_boundary * self.loss_boundary
            + self.lambda_sync * self.loss_sync
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

                unet_input = torch.cat(
                    [z_t, self.z_context, self.mask_z,
                     self.boundary_map_z], dim=1,
                )
                epsilon_pred = self.unet(
                    unet_input, t_batch, video_tokens=self.video_tokens,
                )

                z_t = self.diffusion.ddim_step(
                    z_t, epsilon_pred, t_batch, t_next_batch,
                    eta=ddim_eta,
                    clamp_mask=self.mask_z,
                    z_context=self.z_context,
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
            self.video_tokens = video_out["video_tokens"]

            B = self.z_target.size(0)
            t = torch.zeros(B, device=self.device, dtype=torch.long)
            epsilon = torch.randn_like(self.z_target)
            z_t = self.diffusion.masked_q_sample(
                self.z_target, self.z_context, self.mask_z, t,
                noise=epsilon,
            )

            unet_input = torch.cat(
                [z_t, self.z_context, self.mask_z, self.boundary_map_z],
                dim=1,
            )
            epsilon_pred = self.unet(
                unet_input, t, video_tokens=self.video_tokens,
            )
            self.loss_diff = compute_diffusion_loss(
                epsilon_pred, epsilon, self.mask_z,
            )

            # Decode predicted z_0
            alpha_bar_0 = self.diffusion.alphas_cumprod[0].to(self.device)
            z_0_pred = (
                z_t - torch.sqrt(1.0 - alpha_bar_0) * epsilon_pred
            ) / torch.sqrt(alpha_bar_0)
            mel_0_pred = self._decode_latent(z_0_pred)
            self.mel_pred = compose_known_region(
                mel_0_pred, self.mel_corrupted, self.missing_mask,
            )

            self.loss_boundary = self._boundary_loss(
                mel_0_pred, self.mel_target, self.missing_mask,
            )
            self.loss_sync = (
                self._sync_loss(video_out)
                if self.lambda_sync > 0
                else torch.tensor(0.0, device=self.device)
            )

            self.loss_total = (
                self.loss_diff
                + self.lambda_boundary * self.loss_boundary
                + self.lambda_sync * self.loss_sync
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

    def get_current_errors(self):
        return {
            "loss_total": self.loss_total_item,
            "loss_diff": self.loss_diff_item,
            "loss_boundary": self.loss_boundary_item,
            "loss_sync": self.loss_sync_item,
            "lr": self.current_lr,
        }

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
        writer.add_scalar(f"{prefix}/lr", self.current_lr, step)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self, global_step, global_epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"UQ-AV_checkpoint_step{global_step:09d}.pth.tar",
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
            },
            "latent_mean": self.latent_mean.cpu()
            if self.latent_mean is not None else None,
            "latent_std": self.latent_std.cpu()
            if self.latent_std is not None else None,
        }
        torch.save(checkpoint, checkpoint_path)
        print("[UQ-AV] Saved checkpoint:", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path, reset_optimizer=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.unet.load_state_dict(checkpoint["unet"])
        if self.use_video:
            self.video_encoder.load_state_dict(checkpoint["video_encoder"])
        else:
            print(
                "[UQ-AV] Skipping video_encoder weights because "
                "--uq_no_video uses VideoConditionDummy."
            )
        if not reset_optimizer and checkpoint.get("optimizer") is not None:
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
        global_step = int(checkpoint.get("global_step", 0))
        global_epoch = int(checkpoint.get("global_epoch", 0))
        self._loaded_step = global_step
        self._loaded_epoch = global_epoch
        return global_step, global_epoch
