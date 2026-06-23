import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from Data_loaders import mel_loader
from loss_functions import GANLoss, L2ContrastiveLoss
from networks import Discriminator_Networks
from networks import EC_VIAI_Modules
from networks import Image_Embedding
from networks import Inpainting_Networks
from networks import New_Inpainting_Networks
from utils import util


def _copy_matching_state(module, source_state, label):
    target_state = module.state_dict()
    copied = 0
    skipped = 0
    for name, value in source_state.items():
        if isinstance(value, nn.Parameter):
            value = value.data
        if name not in target_state:
            continue
        if target_state[name].shape != value.shape:
            skipped += 1
            continue
        target_state[name].copy_(value)
        copied += 1
    print(f"[VIAI-AV] loaded {copied} tensors into {label}; skipped_shape={skipped}")
    return copied


class VIAIAVModel(object):
    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")
        self.use_gan = bool(getattr(hparams, "use_gan", False))
        self.enable_sync_loss = not bool(getattr(hparams, "disable_sync_loss", False))
        self.enable_probe_loss = not bool(getattr(hparams, "disable_probe_loss", False))
        self.enable_ec_viai_av = bool(getattr(hparams, "enable_ec_viai_av", False))
        self.use_deterministic_adapter = self.enable_ec_viai_av and bool(
            getattr(hparams, "deterministic_adapter", False)
        )
        self.use_stochastic_adapter = self.enable_ec_viai_av and bool(
            getattr(hparams, "stochastic_adapter", False)
        )
        self.use_evidence_gate = self.enable_ec_viai_av and bool(
            getattr(hparams, "enable_evidence_gate", False)
        )
        self.use_candidate_scorer = self.enable_ec_viai_av and bool(
            getattr(hparams, "enable_candidate_scorer", False)
        )
        self.freeze_gate_evidence_backbone = bool(
            getattr(hparams, "freeze_gate_evidence_backbone", False)
        )
        self.enable_evidence_scaled_sigma = bool(
            getattr(hparams, "enable_evidence_scaled_sigma", False)
        )
        self.enable_visual_evidence_aug = bool(
            getattr(hparams, "enable_visual_evidence_aug", False)
        )
        self.visual_evidence_aug_modes = (
            EC_VIAI_Modules.normalize_visual_evidence_aug_modes(
                getattr(
                    hparams,
                    "visual_evidence_aug_modes",
                    "flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow",
                )
            )
        )
        if self.use_deterministic_adapter and self.use_stochastic_adapter:
            raise ValueError(
                "--stochastic_adapter and --deterministic_adapter cannot both be enabled."
            )
        if self.use_candidate_scorer and not self.use_stochastic_adapter:
            raise ValueError(
                "--enable_candidate_scorer requires "
                "--enable_ec_viai_av --stochastic_adapter."
            )
        lambda_calib = float(getattr(hparams, "lambda_calib", 0.0))
        if lambda_calib > 0.0 and not self.use_candidate_scorer:
            raise ValueError("--lambda_calib > 0 requires --enable_candidate_scorer.")
        if float(getattr(hparams, "calib_error_tau", 0.1)) <= 0.0:
            raise ValueError("--calib_error_tau must be > 0.")
        self.train_num_candidates = int(getattr(hparams, "num_candidates", 1))
        self.test_num_candidates = int(
            getattr(hparams, "test_num_candidates", self.train_num_candidates)
        )
        if self.train_num_candidates < 1 or self.test_num_candidates < 1:
            raise ValueError("num_candidates and test_num_candidates must be >= 1.")
        if not self.use_stochastic_adapter and (
            self.train_num_candidates > 1 or self.test_num_candidates > 1
        ):
            raise ValueError(
                "num_candidates > 1 or test_num_candidates > 1 requires "
                "--enable_ec_viai_av --stochastic_adapter."
            )
        lambda_diversity = float(getattr(hparams, "lambda_diversity", 0.0))
        if lambda_diversity > 0.0 and (
            self.train_num_candidates < 2 or self.test_num_candidates < 2
        ):
            raise ValueError(
                "--lambda_diversity > 0 requires num_candidates and "
                "test_num_candidates to be >= 2."
            )
        self.current_num_candidates = self.train_num_candidates
        self.use_bottleneck_adapter = (
            self.use_deterministic_adapter or self.use_stochastic_adapter
        )

        self.Mel_Encoder = Inpainting_Networks.MelEncoder(hparams=hparams).to(self.device)
        self.VideoEncoder = Image_Embedding.ImageEmbedding(hparams=hparams).to(self.device)
        self.Mel_Decoder = New_Inpainting_Networks.MelDecoderImage(hparams=hparams).to(self.device)
        self.EvidenceEstimator = EC_VIAI_Modules.VisualEvidenceEstimator().to(self.device)
        self.netD = None
        if self.use_gan:
            self.netD = Discriminator_Networks.MelDiscriminator().to(self.device)
        self.BottleneckAdapter = None
        if self.use_bottleneck_adapter:
            self.BottleneckAdapter = EC_VIAI_Modules.BottleneckAdapter().to(self.device)
        self.EvidenceFusionGate = None
        if self.use_evidence_gate:
            self.EvidenceFusionGate = EC_VIAI_Modules.EvidenceFusionGate().to(self.device)
        self.CandidateScorer = None
        self.UncertaintyHead = None
        if self.use_candidate_scorer:
            self.CandidateScorer = EC_VIAI_Modules.CandidateScorer().to(self.device)
            self.UncertaintyHead = EC_VIAI_Modules.UncertaintyHead().to(self.device)
        if self.freeze_gate_evidence_backbone:
            self._set_module_requires_grad(self.Mel_Encoder, False)
            self._set_module_requires_grad(self.VideoEncoder, False)

        self.criterion_l1 = nn.L1Loss()
        self.criterion_gan = GANLoss(use_lsgan=False, device=self.device) if self.use_gan else None
        self.criterion_sync = L2ContrastiveLoss(
            margin=getattr(hparams, "sync_margin", 1.0),
            max_violation=False,
        )
        self.criterion_gate_evidence = nn.SmoothL1Loss()
        self.criterion_uncertainty_calib = nn.SmoothL1Loss()

        generator_params = []
        if not self.freeze_gate_evidence_backbone:
            generator_params += list(self.Mel_Encoder.parameters())
            generator_params += list(self.VideoEncoder.parameters())
        generator_params += list(self.Mel_Decoder.parameters())
        if self.use_bottleneck_adapter:
            generator_params += list(self.BottleneckAdapter.parameters())
        if self.use_evidence_gate:
            generator_params += list(self.EvidenceFusionGate.parameters())
        if self.use_candidate_scorer:
            generator_params += list(self.CandidateScorer.parameters())
            generator_params += list(self.UncertaintyHead.parameters())
        self.optimizer_G = torch.optim.Adam(
            generator_params,
            lr=hparams.lr,
            betas=(hparams.beta1, hparams.beta2),
        )
        self.optimizer_D = None
        if self.use_gan:
            self.optimizer_D = torch.optim.Adam(
                self.netD.parameters(),
                lr=hparams.lr,
                betas=(hparams.beta1, hparams.beta2),
            )
        self.current_lr = hparams.lr
        self.blank_length = getattr(hparams, "min_blank_frames", 20)

        self.loss_total_item = 0.0
        self.loss_av_gen_item = 0.0
        self.loss_recon_item = 0.0
        self.loss_full_l1_item = 0.0
        self.loss_missing_l1_item = 0.0
        self.loss_G_GAN_item = 0.0
        self.weighted_loss_recon_item = 0.0
        self.weighted_loss_gan_item = 0.0
        self.loss_sync_item = 0.0
        self.loss_probe_gen_item = 0.0
        self.loss_probe_recon_item = 0.0
        self.loss_probe_full_l1_item = 0.0
        self.loss_probe_missing_l1_item = 0.0
        self.loss_probe_G_GAN_item = 0.0
        self.weighted_loss_probe_gen_item = 0.0
        self.loss_anchor_item = 0.0
        self.loss_min_k_item = 0.0
        self.loss_mean_k_item = 0.0
        self.loss_boundary_item = 0.0
        self.loss_evidence_div_item = 0.0
        self.loss_gate_evidence_item = 0.0
        self.loss_multi_candidate_item = 0.0
        self.weighted_loss_min_k_item = 0.0
        self.weighted_loss_mean_k_item = 0.0
        self.weighted_loss_boundary_item = 0.0
        self.weighted_loss_evidence_div_item = 0.0
        self.weighted_loss_gate_evidence_item = 0.0
        self.loss_candidate_scorer_item = 0.0
        self.loss_uncertainty_calib_item = 0.0
        self.loss_calib_item = 0.0
        self.weighted_loss_calib_item = 0.0
        self.best_of_k_missing_l1_item = 0.0
        self.mean_k_missing_l1_item = 0.0
        self.top1_missing_l1_item = 0.0
        self.candidate0_missing_l1_item = 0.0
        self.random_expected_missing_l1_item = 0.0
        self.oracle_gain_item = 0.0
        self.uncertainty_mean_item = 0.0
        self.uncertainty_min_item = 0.0
        self.uncertainty_max_item = 0.0
        self.candidate_top1_index_mean_item = 0.0
        self.candidate_pi_entropy_item = 0.0
        self.candidate_pi_max_item = 0.0
        self.loss_D_item = 0.0
        self.loss_D_real_item = 0.0
        self.loss_D_fake_item = 0.0
        self.eta1_item = 0.0
        self.eta2_item = 0.0
        self.evidence_score = torch.zeros(1, 1, device=self.device)
        self.evidence_mean_item = 0.0
        self.evidence_min_item = 0.0
        self.evidence_max_item = 0.0
        self.gate_value = torch.ones(1, 1, 1, 1, device=self.device)
        self.gate_target = torch.ones(1, 1, 1, 1, device=self.device)
        self.gate_mean_item = 1.0
        self.gate_min_item = 1.0
        self.gate_max_item = 1.0
        self.gate_target_mean_item = 1.0
        self.gate_target_gap_item = 0.0
        self.visual_evidence_aug_mode = "none"
        self.visual_evidence_aug_applied_item = 0.0
        self.visual_evidence_aug_mode_items = {
            "none": 1.0,
            "flow_75": 0.0,
            "flow_50": 0.0,
            "flow_25": 0.0,
            "flow_zero": 0.0,
            "static_video_zero_flow": 0.0,
        }
        self.audio_prior_feature = torch.zeros(1, 256, 1, 1, device=self.device)
        self.calibrated_video_feature = torch.zeros(1, 256, 1, 1, device=self.device)
        self.adapter_residual = torch.zeros(1, 256, 1, 1, device=self.device)
        self.adapter_scale_item = 0.0
        self.adapter_stochastic_scale_item = 0.0
        self.adapter_residual_l1_item = 0.0
        self.adapter_logvar = torch.zeros(1, 256, 1, 1, device=self.device)
        self.adapter_sigma = torch.zeros(1, 256, 1, 1, device=self.device)
        self.adapter_sigma_scale = torch.ones(1, 1, 1, 1, device=self.device)
        self.adapter_effective_sigma = torch.zeros(1, 256, 1, 1, device=self.device)
        self.adapter_logvar_mean_item = 0.0
        self.adapter_sigma_mean_item = 0.0
        self.adapter_sigma_scale_mean_item = 1.0
        self.adapter_sigma_scale_min_item = 1.0
        self.adapter_sigma_scale_max_item = 1.0
        self.adapter_effective_sigma_mean_item = 0.0
        self.candidate_pairwise_distance_per_sample = torch.zeros(1, device=self.device)
        self.candidate_pairwise_distance = torch.zeros((), device=self.device)
        self.candidate_pairwise_distance_item = 0.0
        self.evidence_diversity_target = torch.zeros(1, 1, device=self.device)
        self.evidence_diversity_gap = torch.zeros((), device=self.device)
        self.evidence_diversity_gap_item = 0.0
        self.candidate_pairwise_l1 = torch.zeros((), device=self.device)
        self.candidate_pairwise_l1_item = 0.0
        self.candidate_logits = torch.zeros(1, 1, device=self.device)
        self.candidate_pi = torch.ones(1, 1, device=self.device)
        self.candidate_top1_index = torch.zeros(1, device=self.device, dtype=torch.long)
        self.candidate_missing_l1 = torch.zeros(1, 1, device=self.device)
        self.top1_missing_l1 = torch.zeros((), device=self.device)
        self.candidate0_missing_l1 = torch.zeros((), device=self.device)
        self.random_expected_missing_l1 = torch.zeros((), device=self.device)
        self.oracle_gain = torch.zeros((), device=self.device)
        self.top1_missing_l1_per_sample = torch.zeros(1, device=self.device)
        self.best_of_k_missing_l1_per_sample = torch.zeros(1, device=self.device)
        self.candidate0_missing_l1_per_sample = torch.zeros(1, device=self.device)
        self.random_expected_missing_l1_per_sample = torch.zeros(1, device=self.device)
        self.candidate_pi_entropy = torch.zeros((), device=self.device)
        self.candidate_pi_max = torch.ones((), device=self.device)
        self.uncertainty_score = torch.zeros(1, 1, device=self.device)
        self.loss_candidate_scorer = torch.zeros((), device=self.device)
        self.loss_uncertainty_calib = torch.zeros((), device=self.device)
        self.loss_calib = torch.zeros((), device=self.device)
        self.weighted_loss_calib = torch.zeros((), device=self.device)
        mel_height = int(getattr(hparams, "cin_channels", 80))
        mel_width = int(getattr(hparams, "max_mel_lengths", 200))
        self.mel_candidates = torch.zeros(1, 1, 1, mel_height, mel_width, device=self.device)
        self.mel_completed_candidates = torch.zeros_like(self.mel_candidates)
        self.mel_completed_pred = torch.zeros(1, 1, mel_height, mel_width, device=self.device)
        self.loaded_stage = self._stage_name()
        setattr(self.hparams, "loaded_stage", self.loaded_stage)
        self._print_loss_configuration()

    def _set_module_requires_grad(self, module, requires_grad):
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _set_generator_train_modes(self):
        if self.freeze_gate_evidence_backbone:
            self.Mel_Encoder.eval()
            self.VideoEncoder.eval()
        else:
            self.Mel_Encoder.train()
            self.VideoEncoder.train()
        self.Mel_Decoder.train()
        if self.use_bottleneck_adapter:
            self.BottleneckAdapter.train()
        if self.use_evidence_gate:
            self.EvidenceFusionGate.train()
        if getattr(self, "use_candidate_scorer", False):
            self.CandidateScorer.train()
            self.UncertaintyHead.train()

    def _print_loss_configuration(self):
        lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
        lambda_recon = getattr(self.hparams, "lambda_recon", 1.0)
        lambda_sync = getattr(self.hparams, "lambda_sync", 1.0)
        lambda_probe = getattr(self.hparams, "lambda_probe", 1.0)
        lambda_calib = getattr(self.hparams, "lambda_calib", 0.0)
        print(
            "[VIAI-AV] loss weights: "
            f"use_gan={self.use_gan} "
            f"lambda_gan={lambda_gan} "
            f"lambda_recon={lambda_recon} "
            f"lambda_sync={lambda_sync} "
            f"lambda_probe={lambda_probe} "
            f"lambda_calib={lambda_calib} "
            f"enable_sync_loss={self.enable_sync_loss} "
            f"enable_probe_loss={self.enable_probe_loss}"
        )
        print(
            "[VIAI-AV] EC modules: "
            f"enable_ec_viai_av={self.enable_ec_viai_av} "
            f"use_deterministic_adapter={self.use_deterministic_adapter} "
            f"use_stochastic_adapter={self.use_stochastic_adapter} "
            f"use_bottleneck_adapter={self.use_bottleneck_adapter} "
            f"use_evidence_gate={self.use_evidence_gate} "
            f"use_candidate_scorer={self.use_candidate_scorer} "
            f"freeze_gate_evidence_backbone={self.freeze_gate_evidence_backbone} "
            f"enable_evidence_scaled_sigma={self.enable_evidence_scaled_sigma} "
            f"enable_visual_evidence_aug={self.enable_visual_evidence_aug} "
            f"num_candidates={self.train_num_candidates} "
            f"test_num_candidates={self.test_num_candidates}"
        )
        if self.use_gan:
            print(
                "[VIAI-AV] generator formula: "
                "loss_av_gen = lambda_gan * loss_g_gan + lambda_recon * loss_recon"
            )
        else:
            print("[VIAI-AV] generator formula: loss_av_gen = lambda_recon * loss_recon")
        if self.enable_sync_loss or self.enable_probe_loss:
            print(
                "[VIAI-AV] total formula: "
                "loss_total = loss_av_gen + lambda_sync * loss_sync "
                "+ lambda_probe * eta2 * loss_probe_gen + loss_multi_candidate "
                "+ weighted_loss_gate_evidence + weighted_loss_calib"
            )

    def _eta(self, step, base, interval, floor):
        if interval <= 0:
            return floor
        return max(floor, base ** (float(step) / float(interval)))

    def _to_device(self, tensor):
        if tensor is None:
            return None
        return tensor.float().to(self.device)

    def get_blank_space_length(self, global_step):
        min_blank = max(1, int(getattr(self.hparams, "min_blank_frames", 20)))
        max_blank = max(min_blank, int(getattr(self.hparams, "max_blank_frames", 50)))
        self.blank_length = random.randint(min_blank, max_blank)
        return self.blank_length

    def set_inputs(self, data):
        if len(data) < 8:
            raise ValueError("Expected 8-tuple batch from audio_loader.collate_fn")
        (
            video_batch,
            flow_batch,
            c_batch,
            x_batch,
            y_batch,
            g_batch,
            input_lengths,
            path_batch,
        ) = data
        self.video_batch = self._to_device(video_batch)
        self.flow_batch = self._to_device(flow_batch)
        self.mel_target = self._to_device(c_batch)
        self.audio_input = self._to_device(x_batch)
        self.audio_target = self._to_device(y_batch)
        self.g_batch = g_batch.to(self.device) if g_batch is not None else None
        self.input_lengths = input_lengths.to(self.device)
        self.path_batch = path_batch

        if self.mel_target is None:
            raise ValueError("Local conditioning Mel target is required for VIAI-AV training")
        self.mel_input, self.missing_mask, self.missing_span = mel_loader.corrupt_mel_spectrogram(
            self.mel_target,
            self.blank_length,
        )
        self.mel_target_4d = self.mel_target.unsqueeze(1)
        self.mel_input_4d = self.mel_input.unsqueeze(1)
        self.missing_mask = self.missing_mask.to(self.device)
        self.visual_evidence_aug_mode = "none"

    def _set_visual_evidence_aug_mode(self, mode):
        self.visual_evidence_aug_mode = mode
        self.visual_evidence_aug_mode_items = {
            "none": 0.0,
            "flow_75": 0.0,
            "flow_50": 0.0,
            "flow_25": 0.0,
            "flow_zero": 0.0,
            "static_video_zero_flow": 0.0,
        }
        self.visual_evidence_aug_mode_items[mode] = 1.0
        self.visual_evidence_aug_applied_item = 0.0 if mode == "none" else 1.0

    def _maybe_apply_visual_evidence_aug(self):
        self._set_visual_evidence_aug_mode("none")
        if not self.enable_visual_evidence_aug:
            return
        prob = float(getattr(self.hparams, "visual_evidence_aug_prob", 0.5))
        if random.random() >= prob:
            return
        mode = random.choice(self.visual_evidence_aug_modes)
        self.video_batch, self.flow_batch = EC_VIAI_Modules.apply_visual_evidence_augmentation(
            self.video_batch,
            self.flow_batch,
            mode,
        )
        self._set_visual_evidence_aug_mode(mode)

    def _gate_target_from_evidence(self, evidence):
        low = float(getattr(self.hparams, "evidence_gate_low", 0.24))
        high = float(getattr(self.hparams, "evidence_gate_high", 0.34))
        target = (evidence - low) / (high - low)
        return torch.clamp(target, min=0.0, max=1.0)

    def _sigma_scale_from_gate_target(self, gate_target):
        min_scale = float(getattr(self.hparams, "evidence_sigma_scale_min", 0.5))
        max_scale = float(getattr(self.hparams, "evidence_sigma_scale_max", 2.0))
        uncertainty = 1.0 - gate_target.detach()
        return min_scale + (max_scale - min_scale) * uncertainty

    def _expand_features_for_candidates(self, features, num_candidates):
        expanded_features = []
        for feature in features:
            batch_size, channels, height, width = feature.shape
            expanded = feature.unsqueeze(1).expand(
                batch_size,
                num_candidates,
                channels,
                height,
                width,
            )
            expanded_features.append(
                expanded.reshape(batch_size * num_candidates, channels, height, width)
            )
        return expanded_features

    def _expand_feature_for_candidates(self, feature, num_candidates):
        batch_size, channels, height, width = feature.shape
        expanded = feature.unsqueeze(1).expand(
            batch_size,
            num_candidates,
            channels,
            height,
            width,
        )
        return expanded.reshape(batch_size * num_candidates, channels, height, width)

    def _compose_candidate_mels(self, mel_candidates):
        mask = self.missing_mask.unsqueeze(1).to(
            device=mel_candidates.device,
            dtype=mel_candidates.dtype,
        )
        mel_input = self.mel_input_4d.unsqueeze(1).to(
            device=mel_candidates.device,
            dtype=mel_candidates.dtype,
        )
        return mel_input * (1.0 - mask) + mel_candidates * mask

    def _candidate_pairwise_distance_per_sample(self, mel_candidates):
        batch_size = mel_candidates.size(0)
        num_candidates = mel_candidates.size(1)
        if num_candidates < 2:
            return torch.zeros(batch_size, device=mel_candidates.device, dtype=mel_candidates.dtype)
        mask = self.missing_mask.unsqueeze(1).unsqueeze(1).to(
            device=mel_candidates.device,
            dtype=mel_candidates.dtype,
        )
        pairwise_l1 = torch.abs(
            mel_candidates.unsqueeze(2) - mel_candidates.unsqueeze(1)
        )
        pairwise_l1 = (pairwise_l1 * mask).sum(dim=(3, 4, 5)) / torch.clamp(
            mask.sum(dim=(3, 4, 5)),
            min=1.0,
        )
        pair_mask = torch.triu(
            torch.ones(
                num_candidates,
                num_candidates,
                device=mel_candidates.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return pairwise_l1[:, pair_mask].mean(dim=1)

    def _update_candidate_pairwise_metrics(self):
        self.candidate_pairwise_distance_per_sample = (
            self._candidate_pairwise_distance_per_sample(self.mel_candidates)
        )
        self.candidate_pairwise_distance = self.candidate_pairwise_distance_per_sample.mean()
        self.candidate_pairwise_l1 = self.candidate_pairwise_distance

    def _candidate_missing_input_l1_proxy(self):
        mask = self.missing_mask.unsqueeze(1).to(
            device=self.mel_candidates.device,
            dtype=self.mel_candidates.dtype,
        )
        mel_input = self.mel_input_4d.unsqueeze(1).to(
            device=self.mel_candidates.device,
            dtype=self.mel_candidates.dtype,
        )
        proxy_abs = torch.abs(self.mel_candidates - mel_input) * mask
        proxy_den = torch.clamp(mask.sum(dim=(2, 3, 4)), min=1.0)
        return proxy_abs.sum(dim=(2, 3, 4)) / proxy_den

    def _candidate_boundary_jump_proxy(self):
        start, end = self.missing_span
        start = int(start)
        end = int(end)
        time_steps = int(self.mel_completed_candidates.size(-1))
        jumps = []
        if 0 < start < time_steps:
            left_jump = torch.abs(
                self.mel_completed_candidates[..., start]
                - self.mel_completed_candidates[..., start - 1]
            ).mean(dim=(2, 3))
            jumps.append(left_jump)
        if 0 < end < time_steps:
            right_jump = torch.abs(
                self.mel_completed_candidates[..., end]
                - self.mel_completed_candidates[..., end - 1]
            ).mean(dim=(2, 3))
            jumps.append(right_jump)
        if not jumps:
            return torch.zeros(
                self.mel_completed_candidates.size(0),
                self.mel_completed_candidates.size(1),
                device=self.mel_completed_candidates.device,
                dtype=self.mel_completed_candidates.dtype,
            )
        return sum(jumps) / len(jumps)

    def _candidate_sync_score_proxy(self, decoder_video_feature):
        batch_size, num_candidates = self.mel_completed_candidates.shape[:2]
        previous_training = self.Mel_Encoder.training
        self.Mel_Encoder.eval()
        with torch.no_grad():
            candidate_features = self.Mel_Encoder(
                self.mel_completed_candidates.detach().reshape(
                    batch_size * num_candidates,
                    self.mel_completed_candidates.size(2),
                    self.mel_completed_candidates.size(3),
                    self.mel_completed_candidates.size(4),
                )
            )[-1].flatten(1)
            video_embedding = decoder_video_feature.detach().flatten(1)
            video_embedding = video_embedding.unsqueeze(1).expand(
                batch_size,
                num_candidates,
                video_embedding.size(1),
            )
            video_embedding = video_embedding.reshape(batch_size * num_candidates, -1)
            candidate_embedding = F.normalize(candidate_features, p=2, dim=1)
            video_embedding = F.normalize(video_embedding, p=2, dim=1)
            distance = torch.norm(
                candidate_embedding - video_embedding,
                p=2,
                dim=1,
            ).reshape(batch_size, num_candidates)
            sync_score = 1.0 - torch.clamp(distance / 2.0, min=0.0, max=1.0)
        if previous_training:
            self.Mel_Encoder.train()
        return sync_score.to(dtype=self.mel_candidates.dtype)

    def _candidate_scorer_stats(self, decoder_video_feature):
        missing_proxy = self._candidate_missing_input_l1_proxy()
        boundary_proxy = self._candidate_boundary_jump_proxy()
        sync_score = self._candidate_sync_score_proxy(decoder_video_feature)
        return torch.stack([missing_proxy, boundary_proxy, sync_score], dim=2)

    def _gather_candidate(self, candidates, indices):
        gather_index = indices.view(-1, 1, 1, 1, 1).expand(
            -1,
            1,
            candidates.size(2),
            candidates.size(3),
            candidates.size(4),
        )
        return torch.gather(candidates, dim=1, index=gather_index).squeeze(1)

    def _default_candidate_scores(self, batch_size, num_candidates, reference):
        self.candidate_logits = torch.zeros(
            batch_size,
            num_candidates,
            device=reference.device,
            dtype=reference.dtype,
        )
        self.candidate_pi = torch.zeros_like(self.candidate_logits)
        self.candidate_pi[:, 0] = 1.0
        self.candidate_top1_index = torch.zeros(
            batch_size,
            device=reference.device,
            dtype=torch.long,
        )
        self.uncertainty_score = torch.zeros(
            batch_size,
            1,
            device=reference.device,
            dtype=reference.dtype,
        )
        self.candidate_pi_entropy = torch.zeros((), device=reference.device, dtype=reference.dtype)
        self.candidate_pi_max = torch.ones((), device=reference.device, dtype=reference.dtype)

    def _score_and_select_candidates(self, decoder_video_feature):
        self._update_candidate_pairwise_metrics()
        batch_size, num_candidates = self.mel_candidates.shape[:2]
        if not getattr(self, "use_candidate_scorer", False):
            self._default_candidate_scores(batch_size, num_candidates, self.mel_candidates)
        else:
            candidate_stats = self._candidate_scorer_stats(decoder_video_feature)
            self.candidate_logits, self.candidate_pi = self.CandidateScorer(
                candidate_stats.detach(),
                self.mel_features[-1].detach(),
                decoder_video_feature.detach(),
                self.evidence_score.detach(),
            )
            self.candidate_top1_index = torch.argmax(self.candidate_pi, dim=1)
            eps = 1e-8
            if num_candidates > 1:
                entropy_norm = torch.log(
                    torch.tensor(
                        float(num_candidates),
                        device=self.candidate_pi.device,
                        dtype=self.candidate_pi.dtype,
                    )
                )
                entropy = -(
                    self.candidate_pi * torch.log(torch.clamp(self.candidate_pi, min=eps))
                ).sum(dim=1, keepdim=True) / entropy_norm
            else:
                entropy = torch.zeros(
                    batch_size,
                    1,
                    device=self.candidate_pi.device,
                    dtype=self.candidate_pi.dtype,
                )
            max_pi = self.candidate_pi.max(dim=1, keepdim=True).values
            top1_proxy = torch.gather(
                candidate_stats[:, :, 0].detach(),
                dim=1,
                index=self.candidate_top1_index.view(-1, 1),
            )
            evidence = self.evidence_score.to(
                device=self.mel_candidates.device,
                dtype=self.mel_candidates.dtype,
            ).reshape(batch_size, -1)
            if evidence.size(1) != 1:
                evidence = evidence.mean(dim=1, keepdim=True)
            gate = self.gate_value.reshape(batch_size, -1).to(
                device=self.mel_candidates.device,
                dtype=self.mel_candidates.dtype,
            )
            if gate.size(1) != 1:
                gate = gate.mean(dim=1, keepdim=True)
            sigma_scale = self.adapter_sigma_scale.reshape(batch_size, -1).to(
                device=self.mel_candidates.device,
                dtype=self.mel_candidates.dtype,
            )
            if sigma_scale.size(1) != 1:
                sigma_scale = sigma_scale.mean(dim=1, keepdim=True)
            uncertainty_stats = torch.cat(
                [
                    entropy.detach(),
                    max_pi.detach(),
                    top1_proxy.detach(),
                    self.candidate_pairwise_distance_per_sample.detach().view(batch_size, 1),
                    evidence.detach(),
                    gate.detach(),
                    sigma_scale.detach(),
                ],
                dim=1,
            )
            self.uncertainty_score = self.UncertaintyHead(
                self.mel_features[-1].detach(),
                decoder_video_feature.detach(),
                uncertainty_stats,
            )
            self.candidate_pi_entropy = entropy.mean()
            self.candidate_pi_max = max_pi.mean()

        self.mel_pred = self._gather_candidate(self.mel_candidates, self.candidate_top1_index)
        self.mel_completed_pred = self._gather_candidate(
            self.mel_completed_candidates,
            self.candidate_top1_index,
        )

    def _forward_inpainter(self):
        self.mel_features = self.Mel_Encoder(self.mel_input)
        self.video_feature = self.VideoEncoder(self.video_batch, self.flow_batch)
        self.mel_target_features = self.Mel_Encoder(self.mel_target)
        self.mel_target_feature_flat = self.mel_target_features[-1].flatten(1)
        self.video_feature_flat = self.video_feature.flatten(1)
        self.mel_net_norm = util.l2_norm(self.mel_target_feature_flat.detach())
        self.video_net_norm = util.l2_norm(self.video_feature_flat)
        with torch.no_grad():
            self.evidence_score = self.EvidenceEstimator(
                self.video_feature,
                self.flow_batch,
                self.mel_target_feature_flat,
                self.video_feature_flat,
            )
        self.gate_target = self._gate_target_from_evidence(self.evidence_score).view(
            self.video_feature.size(0),
            1,
            1,
            1,
        )
        decoder_video_feature = self.video_feature
        self.calibrated_video_feature = self.video_feature
        self.audio_prior_feature = torch.zeros_like(self.video_feature)
        self.gate_value = torch.ones(
            self.video_feature.size(0),
            1,
            1,
            1,
            device=self.video_feature.device,
            dtype=self.video_feature.dtype,
        )
        if self.use_evidence_gate:
            (
                self.calibrated_video_feature,
                self.gate_value,
                self.audio_prior_feature,
            ) = self.EvidenceFusionGate(
                self.mel_features[-1],
                self.video_feature,
                self.evidence_score,
            )
            decoder_video_feature = self.calibrated_video_feature
        decoder_features = self.mel_features
        if self.use_stochastic_adapter:
            num_candidates = int(self.current_num_candidates)
            self.adapter_sigma_scale = torch.ones(
                self.video_feature.size(0),
                1,
                1,
                1,
                device=self.video_feature.device,
                dtype=self.video_feature.dtype,
            )
            sigma_scale_arg = None
            if self.enable_evidence_scaled_sigma:
                self.adapter_sigma_scale = self._sigma_scale_from_gate_target(
                    self.gate_target
                ).to(device=self.video_feature.device, dtype=self.video_feature.dtype)
                sigma_scale_arg = self.adapter_sigma_scale
            (
                adapter_residuals,
                _adapter_mu,
                self.adapter_logvar,
                self.adapter_sigma,
            ) = self.BottleneckAdapter.sample_residuals(
                self.mel_features[-1],
                decoder_video_feature,
                num_candidates=num_candidates,
                sigma_min=getattr(self.hparams, "sigma_min", 0.0),
                sigma_max=getattr(self.hparams, "sigma_max", 1.0),
                sigma_scale=sigma_scale_arg,
            )
            self.adapter_effective_sigma = self.adapter_sigma
            batch_size = self.mel_features[-1].size(0)
            self.adapter_residual = adapter_residuals[:, 0]
            decoder_features = self._expand_features_for_candidates(
                self.mel_features,
                num_candidates,
            )
            decoder_features[-1] = decoder_features[-1] + adapter_residuals.reshape(
                batch_size * num_candidates,
                adapter_residuals.size(2),
                adapter_residuals.size(3),
                adapter_residuals.size(4),
            )
            decoder_video_feature_flat = self._expand_feature_for_candidates(
                decoder_video_feature,
                num_candidates,
            )
            mel_candidates_flat = self.Mel_Decoder(
                decoder_features,
                self.mel_input_4d.size(),
                decoder_video_feature_flat,
            )
            self.mel_candidates = mel_candidates_flat.reshape(
                batch_size,
                num_candidates,
                mel_candidates_flat.size(1),
                mel_candidates_flat.size(2),
                mel_candidates_flat.size(3),
            )
            self.mel_completed_candidates = self._compose_candidate_mels(self.mel_candidates)
            self.mel_pred = self.mel_candidates[:, 0]
            self.mel_completed_pred = self.mel_completed_candidates[:, 0]
            self._update_candidate_pairwise_metrics()
        elif self.use_deterministic_adapter:
            self.adapter_residual = self.BottleneckAdapter(
                self.mel_features[-1],
                decoder_video_feature,
            )
            decoder_features = list(self.mel_features)
            decoder_features[-1] = decoder_features[-1] + self.adapter_residual
            self.adapter_logvar = torch.zeros_like(self.mel_features[-1])
            self.adapter_sigma = torch.zeros_like(self.mel_features[-1])
            self.adapter_sigma_scale = torch.ones(
                self.mel_features[-1].size(0),
                1,
                1,
                1,
                device=self.mel_features[-1].device,
                dtype=self.mel_features[-1].dtype,
            )
            self.adapter_effective_sigma = torch.zeros_like(self.mel_features[-1])
            self.mel_pred = self.Mel_Decoder(
                decoder_features,
                self.mel_input_4d.size(),
                decoder_video_feature,
            )
            self.mel_candidates = self.mel_pred.unsqueeze(1)
            self.mel_completed_candidates = self._compose_candidate_mels(self.mel_candidates)
            self.mel_completed_pred = self.mel_completed_candidates[:, 0]
            self._update_candidate_pairwise_metrics()
        else:
            self.adapter_residual = torch.zeros_like(self.mel_features[-1])
            self.adapter_logvar = torch.zeros_like(self.mel_features[-1])
            self.adapter_sigma = torch.zeros_like(self.mel_features[-1])
            self.adapter_sigma_scale = torch.ones(
                self.mel_features[-1].size(0),
                1,
                1,
                1,
                device=self.mel_features[-1].device,
                dtype=self.mel_features[-1].dtype,
            )
            self.adapter_effective_sigma = torch.zeros_like(self.mel_features[-1])
            self.mel_pred = self.Mel_Decoder(
                decoder_features,
                self.mel_input_4d.size(),
                decoder_video_feature,
            )
            self.mel_candidates = self.mel_pred.unsqueeze(1)
            self.mel_completed_candidates = self._compose_candidate_mels(self.mel_candidates)
            self.mel_completed_pred = self.mel_completed_candidates[:, 0]
            self._update_candidate_pairwise_metrics()
        self._score_and_select_candidates(decoder_video_feature)
        if self.enable_probe_loss:
            self.mel_probe_pred = self.Mel_Decoder(
                self.mel_features,
                self.mel_input_4d.size(),
                self.mel_target_features[-1],
            )
        else:
            self.mel_probe_pred = None
        return self.mel_pred

    def _zero_loss_like(self, reference):
        return torch.zeros((), device=self.device, dtype=reference.dtype)

    def _reconstruction_losses(self, prediction):
        loss_full_l1 = self.criterion_l1(prediction, self.mel_target_4d)
        masked_abs = torch.abs(prediction - self.mel_target_4d) * self.missing_mask
        loss_missing_l1 = masked_abs.sum() / torch.clamp(self.missing_mask.sum(), min=1.0)
        loss_recon = self.eta1 * loss_full_l1 + loss_missing_l1
        return loss_recon, loss_full_l1, loss_missing_l1

    def _boundary_loss(self, mel_completed_candidates):
        start, end = self.missing_span
        start = int(start)
        end = int(end)
        time_steps = int(self.mel_target_4d.size(-1))
        boundary_losses = []

        if 0 < start < time_steps:
            pred_delta = (
                mel_completed_candidates[..., start]
                - mel_completed_candidates[..., start - 1]
            )
            target_delta = (
                self.mel_target_4d[..., start]
                - self.mel_target_4d[..., start - 1]
            ).unsqueeze(1)
            boundary_losses.append(torch.abs(pred_delta - target_delta).mean())

        if 0 < end < time_steps:
            pred_delta = (
                mel_completed_candidates[..., end]
                - mel_completed_candidates[..., end - 1]
            )
            target_delta = (
                self.mel_target_4d[..., end]
                - self.mel_target_4d[..., end - 1]
            ).unsqueeze(1)
            boundary_losses.append(torch.abs(pred_delta - target_delta).mean())

        if not boundary_losses:
            return self._zero_loss_like(mel_completed_candidates)
        return sum(boundary_losses) / len(boundary_losses)

    def _multi_candidate_losses(self):
        mel_candidates = self.mel_candidates
        target = self.mel_target_4d.unsqueeze(1).to(
            device=mel_candidates.device,
            dtype=mel_candidates.dtype,
        )
        mask = self.missing_mask.unsqueeze(1).to(
            device=mel_candidates.device,
            dtype=mel_candidates.dtype,
        )
        self._update_candidate_pairwise_metrics()
        if not hasattr(self, "candidate_top1_index") or self.candidate_top1_index.numel() != mel_candidates.size(0):
            self.candidate_top1_index = torch.zeros(
                mel_candidates.size(0),
                device=mel_candidates.device,
                dtype=torch.long,
            )

        self.loss_anchor, _, _ = self._reconstruction_losses(mel_candidates[:, 0])
        candidate_abs = torch.abs(mel_candidates - target) * mask
        candidate_missing_den = torch.clamp(mask.sum(dim=(2, 3, 4)), min=1.0)
        self.candidate_missing_l1 = (
            candidate_abs.sum(dim=(2, 3, 4)) / candidate_missing_den
        )

        best_per_sample = self.candidate_missing_l1.min(dim=1).values
        top1_per_sample = torch.gather(
            self.candidate_missing_l1,
            dim=1,
            index=self.candidate_top1_index.view(-1, 1),
        ).squeeze(1)
        candidate0_per_sample = self.candidate_missing_l1[:, 0]
        random_expected_per_sample = self.candidate_missing_l1.mean(dim=1)
        self.best_of_k_missing_l1_per_sample = best_per_sample
        self.top1_missing_l1_per_sample = top1_per_sample
        self.candidate0_missing_l1_per_sample = candidate0_per_sample
        self.random_expected_missing_l1_per_sample = random_expected_per_sample

        self.loss_min_k = best_per_sample.mean()
        self.loss_mean_k = self.candidate_missing_l1.mean()
        self.loss_boundary = self._boundary_loss(self.mel_completed_candidates)
        self.best_of_k_missing_l1 = self.loss_min_k
        self.mean_k_missing_l1 = self.loss_mean_k
        self.top1_missing_l1 = top1_per_sample.mean()
        self.candidate0_missing_l1 = candidate0_per_sample.mean()
        self.random_expected_missing_l1 = random_expected_per_sample.mean()
        self.oracle_gain = self.top1_missing_l1 - self.best_of_k_missing_l1

        lambda_min_k = getattr(self.hparams, "lambda_min_k", 0.0)
        lambda_mean_k = getattr(self.hparams, "lambda_mean_k", 0.0)
        lambda_boundary = getattr(self.hparams, "lambda_boundary", 0.0)
        lambda_diversity = getattr(self.hparams, "lambda_diversity", 0.0)
        d_min = getattr(self.hparams, "evidence_diversity_d_min", 0.02)
        alpha = getattr(self.hparams, "evidence_diversity_alpha", 0.08)
        if lambda_diversity > 0.0 and mel_candidates.size(1) < 2:
            raise ValueError("--lambda_diversity > 0 requires at least 2 candidates.")

        evidence = self.evidence_score.to(
            device=mel_candidates.device,
            dtype=mel_candidates.dtype,
        ).reshape(mel_candidates.size(0), -1)
        if evidence.size(1) != 1:
            evidence = evidence.mean(dim=1, keepdim=True)
        self.evidence_diversity_target = d_min + alpha * (1.0 - evidence.detach())
        diversity_gap_per_sample = (
            self.candidate_pairwise_distance_per_sample
            - self.evidence_diversity_target.squeeze(1)
        )
        self.evidence_diversity_gap = diversity_gap_per_sample.mean()
        self.loss_evidence_div = torch.abs(diversity_gap_per_sample).mean()
        self.weighted_loss_min_k = lambda_min_k * self.loss_min_k
        self.weighted_loss_mean_k = lambda_mean_k * self.loss_mean_k
        self.weighted_loss_boundary = lambda_boundary * self.loss_boundary
        self.weighted_loss_evidence_div = lambda_diversity * self.loss_evidence_div
        self.loss_multi_candidate = (
            self.weighted_loss_min_k
            + self.weighted_loss_mean_k
            + self.weighted_loss_boundary
            + self.weighted_loss_evidence_div
        )
        return self.loss_multi_candidate

    def _calibration_losses(self):
        if not getattr(self, "use_candidate_scorer", False):
            zero = self._zero_loss_like(self.loss_av_gen)
            self.loss_candidate_scorer = zero
            self.loss_uncertainty_calib = zero
            self.loss_calib = zero
            self.weighted_loss_calib = zero
            return self.weighted_loss_calib

        best_idx = torch.argmin(self.candidate_missing_l1.detach(), dim=1)
        self.loss_candidate_scorer = F.cross_entropy(self.candidate_logits, best_idx)
        best_error = torch.gather(
            self.candidate_missing_l1.detach(),
            dim=1,
            index=best_idx.view(-1, 1),
        )
        tau = float(getattr(self.hparams, "calib_error_tau", 0.1))
        difficulty = 1.0 - torch.exp(-best_error / tau)
        self.loss_uncertainty_calib = self.criterion_uncertainty_calib(
            self.uncertainty_score,
            difficulty,
        )
        self.loss_calib = self.loss_candidate_scorer + self.loss_uncertainty_calib
        lambda_calib = float(getattr(self.hparams, "lambda_calib", 0.0))
        self.weighted_loss_calib = lambda_calib * self.loss_calib
        return self.weighted_loss_calib

    def _gate_evidence_loss(self):
        if not self.use_evidence_gate:
            self.loss_gate_evidence = self._zero_loss_like(self.loss_av_gen)
            self.weighted_loss_gate_evidence = self._zero_loss_like(self.loss_av_gen)
            return self.weighted_loss_gate_evidence
        self.loss_gate_evidence = self.criterion_gate_evidence(
            self.gate_value,
            self.gate_target.detach(),
        )
        lambda_gate_evidence = getattr(self.hparams, "lambda_gate_evidence", 0.0)
        self.weighted_loss_gate_evidence = lambda_gate_evidence * self.loss_gate_evidence
        return self.weighted_loss_gate_evidence

    def _compute_losses(self, global_step):
        self.eta1 = self._eta(
            global_step,
            getattr(self.hparams, "recon_decay_base", 0.9),
            getattr(self.hparams, "recon_decay_interval", 1000.0),
            getattr(self.hparams, "recon_decay_floor", 0.1),
        )
        self.eta2 = self._eta(
            global_step,
            getattr(self.hparams, "probe_decay_base", getattr(self.hparams, "sync_decay_base", 0.9)),
            getattr(
                self.hparams,
                "probe_decay_interval",
                getattr(self.hparams, "sync_decay_interval", 1000.0),
            ),
            getattr(self.hparams, "probe_decay_floor", getattr(self.hparams, "sync_decay_floor", 0.1)),
        )
        self.loss_recon, self.loss_full_l1, self.loss_missing_l1 = self._reconstruction_losses(
            self.mel_pred
        )
        lambda_recon = getattr(self.hparams, "lambda_recon", 1.0)
        self.weighted_loss_recon = lambda_recon * self.loss_recon

        if self.use_gan:
            pred_fake = self.netD(self.mel_pred)
            self.loss_G_GAN = self.criterion_gan(pred_fake, True)
            lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
            self.weighted_loss_gan = lambda_gan * self.loss_G_GAN
        else:
            self.loss_G_GAN = self._zero_loss_like(self.loss_recon)
            self.weighted_loss_gan = self._zero_loss_like(self.loss_recon)
        self.loss_av_gen = self.weighted_loss_gan + self.weighted_loss_recon

        if self.enable_sync_loss:
            self.loss_sync = self.criterion_sync(self.mel_net_norm, self.video_net_norm)
        else:
            self.loss_sync = self._zero_loss_like(self.loss_av_gen)

        if self.enable_probe_loss and self.mel_probe_pred is not None:
            (
                self.loss_probe_recon,
                self.loss_probe_full_l1,
                self.loss_probe_missing_l1,
            ) = self._reconstruction_losses(self.mel_probe_pred)
            if self.use_gan:
                pred_probe_fake = self.netD(self.mel_probe_pred)
                self.loss_probe_G_GAN = self.criterion_gan(pred_probe_fake, True)
                lambda_gan = getattr(self.hparams, "lambda_gan", 1.0)
                self.loss_probe_gen = (
                    lambda_gan * self.loss_probe_G_GAN
                    + lambda_recon * self.loss_probe_recon
                )
            else:
                self.loss_probe_G_GAN = self._zero_loss_like(self.loss_probe_recon)
                self.loss_probe_gen = lambda_recon * self.loss_probe_recon
        else:
            self.loss_probe_recon = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_full_l1 = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_missing_l1 = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_G_GAN = self._zero_loss_like(self.loss_av_gen)
            self.loss_probe_gen = self._zero_loss_like(self.loss_av_gen)

        lambda_sync = getattr(self.hparams, "lambda_sync", 1.0)
        lambda_probe = getattr(self.hparams, "lambda_probe", 1.0)
        self.weighted_loss_probe_gen = self.eta2 * self.loss_probe_gen
        self._multi_candidate_losses()
        self._gate_evidence_loss()
        self._calibration_losses()
        baseline_loss_total = (
            self.loss_av_gen
            + lambda_sync * self.loss_sync
            + lambda_probe * self.weighted_loss_probe_gen
        )
        self.loss_total = (
            baseline_loss_total
            + self.loss_multi_candidate
            + self.weighted_loss_gate_evidence
            + self.weighted_loss_calib
        )
        if not self.use_gan:
            self.loss_D_real = self._zero_loss_like(self.loss_av_gen)
            self.loss_D_fake = self._zero_loss_like(self.loss_av_gen)
            self.loss_D = self._zero_loss_like(self.loss_av_gen)

    def _compute_discriminator_loss(self):
        if not self.use_gan:
            self.loss_D_real = self._zero_loss_like(self.loss_av_gen)
            self.loss_D_fake = self._zero_loss_like(self.loss_av_gen)
            self.loss_D = self._zero_loss_like(self.loss_av_gen)
            return
        pred_real = self.netD(self.mel_target_4d)
        fake_losses = [self.criterion_gan(self.netD(self.mel_pred.detach()), False, softlabel=True)]
        if self.enable_probe_loss and self.mel_probe_pred is not None:
            fake_losses.append(
                self.criterion_gan(self.netD(self.mel_probe_pred.detach()), False, softlabel=True)
            )
        self.loss_D_real = self.criterion_gan(pred_real, True, softlabel=True)
        self.loss_D_fake = sum(fake_losses) / len(fake_losses)
        self.loss_D = 0.5 * (self.loss_D_real + self.loss_D_fake)

    def optimize_parameters(self, global_step):
        self.current_num_candidates = self.train_num_candidates
        self._set_generator_train_modes()
        if self.use_gan:
            self.netD.eval()
            for parameter in self.netD.parameters():
                parameter.requires_grad = False

        self._maybe_apply_visual_evidence_aug()
        self._forward_inpainter()
        self._compute_losses(global_step)
        self.optimizer_G.zero_grad()
        self.loss_total.backward()
        self.optimizer_G.step()

        if self.use_gan:
            self.netD.train()
            for parameter in self.netD.parameters():
                parameter.requires_grad = True
            self.optimizer_D.zero_grad()
            self._compute_discriminator_loss()
            self.loss_D.backward()
            self.optimizer_D.step()
        self.current_lr = self.optimizer_G.param_groups[0]["lr"]

    def test(self, global_step=0):
        self.current_num_candidates = self.test_num_candidates
        self.Mel_Encoder.eval()
        self.VideoEncoder.eval()
        self.Mel_Decoder.eval()
        if self.use_bottleneck_adapter:
            self.BottleneckAdapter.eval()
        if self.use_evidence_gate:
            self.EvidenceFusionGate.eval()
        if getattr(self, "use_candidate_scorer", False):
            self.CandidateScorer.eval()
            self.UncertaintyHead.eval()
        if self.use_gan:
            self.netD.eval()
        with torch.no_grad():
            self._forward_inpainter()
            self._compute_losses(global_step=global_step)
            self._compute_discriminator_loss()

    def get_loss_items(self):
        self.loss_total_item = float(self.loss_total.detach().cpu().item())
        self.loss_av_gen_item = float(self.loss_av_gen.detach().cpu().item())
        self.loss_recon_item = float(self.loss_recon.detach().cpu().item())
        self.loss_full_l1_item = float(self.loss_full_l1.detach().cpu().item())
        self.loss_missing_l1_item = float(self.loss_missing_l1.detach().cpu().item())
        self.loss_G_GAN_item = float(self.loss_G_GAN.detach().cpu().item())
        self.weighted_loss_recon_item = float(self.weighted_loss_recon.detach().cpu().item())
        self.weighted_loss_gan_item = float(self.weighted_loss_gan.detach().cpu().item())
        self.loss_sync_item = float(self.loss_sync.detach().cpu().item())
        self.loss_probe_gen_item = float(self.loss_probe_gen.detach().cpu().item())
        self.loss_probe_recon_item = float(self.loss_probe_recon.detach().cpu().item())
        self.loss_probe_full_l1_item = float(self.loss_probe_full_l1.detach().cpu().item())
        self.loss_probe_missing_l1_item = float(self.loss_probe_missing_l1.detach().cpu().item())
        self.loss_probe_G_GAN_item = float(self.loss_probe_G_GAN.detach().cpu().item())
        self.weighted_loss_probe_gen_item = float(
            self.weighted_loss_probe_gen.detach().cpu().item()
        )
        self.loss_anchor_item = float(self.loss_anchor.detach().cpu().item())
        self.loss_min_k_item = float(self.loss_min_k.detach().cpu().item())
        self.loss_mean_k_item = float(self.loss_mean_k.detach().cpu().item())
        self.loss_boundary_item = float(self.loss_boundary.detach().cpu().item())
        self.loss_evidence_div_item = float(
            self.loss_evidence_div.detach().cpu().item()
        )
        self.loss_gate_evidence_item = float(
            self.loss_gate_evidence.detach().cpu().item()
        )
        self.loss_candidate_scorer_item = float(
            self.loss_candidate_scorer.detach().cpu().item()
        )
        self.loss_uncertainty_calib_item = float(
            self.loss_uncertainty_calib.detach().cpu().item()
        )
        self.loss_calib_item = float(self.loss_calib.detach().cpu().item())
        self.loss_multi_candidate_item = float(
            self.loss_multi_candidate.detach().cpu().item()
        )
        self.weighted_loss_min_k_item = float(
            self.weighted_loss_min_k.detach().cpu().item()
        )
        self.weighted_loss_mean_k_item = float(
            self.weighted_loss_mean_k.detach().cpu().item()
        )
        self.weighted_loss_boundary_item = float(
            self.weighted_loss_boundary.detach().cpu().item()
        )
        self.weighted_loss_evidence_div_item = float(
            self.weighted_loss_evidence_div.detach().cpu().item()
        )
        self.weighted_loss_gate_evidence_item = float(
            self.weighted_loss_gate_evidence.detach().cpu().item()
        )
        self.weighted_loss_calib_item = float(
            self.weighted_loss_calib.detach().cpu().item()
        )
        self.best_of_k_missing_l1_item = float(
            self.best_of_k_missing_l1.detach().cpu().item()
        )
        self.mean_k_missing_l1_item = float(
            self.mean_k_missing_l1.detach().cpu().item()
        )
        self.top1_missing_l1_item = float(self.top1_missing_l1.detach().cpu().item())
        self.candidate0_missing_l1_item = float(
            self.candidate0_missing_l1.detach().cpu().item()
        )
        self.random_expected_missing_l1_item = float(
            self.random_expected_missing_l1.detach().cpu().item()
        )
        self.oracle_gain_item = float(self.oracle_gain.detach().cpu().item())
        self.uncertainty_mean_item = float(
            self.uncertainty_score.detach().mean().cpu().item()
        )
        self.uncertainty_min_item = float(
            self.uncertainty_score.detach().min().cpu().item()
        )
        self.uncertainty_max_item = float(
            self.uncertainty_score.detach().max().cpu().item()
        )
        self.candidate_top1_index_mean_item = float(
            self.candidate_top1_index.detach().float().mean().cpu().item()
        )
        self.candidate_pi_entropy_item = float(
            self.candidate_pi_entropy.detach().cpu().item()
        )
        self.candidate_pi_max_item = float(self.candidate_pi_max.detach().cpu().item())
        self.loss_D_item = float(self.loss_D.detach().cpu().item())
        self.loss_D_real_item = float(self.loss_D_real.detach().cpu().item())
        self.loss_D_fake_item = float(self.loss_D_fake.detach().cpu().item())
        self.eta1_item = float(self.eta1)
        self.eta2_item = float(self.eta2)
        self.evidence_mean_item = float(self.evidence_score.detach().mean().cpu().item())
        self.evidence_min_item = float(self.evidence_score.detach().min().cpu().item())
        self.evidence_max_item = float(self.evidence_score.detach().max().cpu().item())
        self.gate_mean_item = float(self.gate_value.detach().mean().cpu().item())
        self.gate_min_item = float(self.gate_value.detach().min().cpu().item())
        self.gate_max_item = float(self.gate_value.detach().max().cpu().item())
        self.gate_target_mean_item = float(
            self.gate_target.detach().mean().cpu().item()
        )
        self.gate_target_gap_item = self.gate_mean_item - self.gate_target_mean_item
        if self.BottleneckAdapter is None:
            self.adapter_scale_item = 0.0
            self.adapter_stochastic_scale_item = 0.0
        else:
            self.adapter_scale_item = float(
                self.BottleneckAdapter.residual_scale.detach().cpu().item()
            )
            self.adapter_stochastic_scale_item = float(
                self.BottleneckAdapter.stochastic_residual_scale.detach().cpu().item()
            )
        self.adapter_residual_l1_item = float(
            self.adapter_residual.detach().abs().mean().cpu().item()
        )
        self.adapter_logvar_mean_item = float(
            self.adapter_logvar.detach().mean().cpu().item()
        )
        self.adapter_sigma_mean_item = float(self.adapter_sigma.detach().mean().cpu().item())
        self.adapter_sigma_scale_mean_item = float(
            self.adapter_sigma_scale.detach().mean().cpu().item()
        )
        self.adapter_sigma_scale_min_item = float(
            self.adapter_sigma_scale.detach().min().cpu().item()
        )
        self.adapter_sigma_scale_max_item = float(
            self.adapter_sigma_scale.detach().max().cpu().item()
        )
        self.adapter_effective_sigma_mean_item = float(
            self.adapter_effective_sigma.detach().mean().cpu().item()
        )
        self.candidate_pairwise_distance_item = float(
            self.candidate_pairwise_distance.detach().cpu().item()
        )
        self.evidence_diversity_gap_item = float(
            self.evidence_diversity_gap.detach().cpu().item()
        )
        self.candidate_pairwise_l1_item = float(
            self.candidate_pairwise_l1.detach().cpu().item()
        )

    def TF_writer(self, writer, step, prefix="train"):
        if writer is None:
            return
        writer.add_scalar(f"{prefix}/loss_total", self.loss_total_item, step)
        writer.add_scalar(f"{prefix}/loss_av_gen", self.loss_av_gen_item, step)
        writer.add_scalar(f"{prefix}/loss_recon", self.loss_recon_item, step)
        writer.add_scalar(f"{prefix}/loss_full_l1", self.loss_full_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_missing_l1", self.loss_missing_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_g_gan", self.loss_G_GAN_item, step)
        writer.add_scalar(f"{prefix}/weighted_loss_recon", self.weighted_loss_recon_item, step)
        writer.add_scalar(f"{prefix}/weighted_loss_gan", self.weighted_loss_gan_item, step)
        writer.add_scalar(f"{prefix}/loss_sync", self.loss_sync_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_gen", self.loss_probe_gen_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_recon", self.loss_probe_recon_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_full_l1", self.loss_probe_full_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_missing_l1", self.loss_probe_missing_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_probe_g_gan", self.loss_probe_G_GAN_item, step)
        if self.enable_probe_loss:
            writer.add_scalar(
                f"{prefix}/weighted_loss_probe_gen",
                self.weighted_loss_probe_gen_item,
                step,
            )
        writer.add_scalar(f"{prefix}/loss_anchor", self.loss_anchor_item, step)
        writer.add_scalar(f"{prefix}/loss_min_k", self.loss_min_k_item, step)
        writer.add_scalar(f"{prefix}/loss_mean_k", self.loss_mean_k_item, step)
        writer.add_scalar(f"{prefix}/loss_boundary", self.loss_boundary_item, step)
        writer.add_scalar(f"{prefix}/loss_evidence_div", self.loss_evidence_div_item, step)
        writer.add_scalar(f"{prefix}/loss_gate_evidence", self.loss_gate_evidence_item, step)
        writer.add_scalar(f"{prefix}/loss_candidate_scorer", self.loss_candidate_scorer_item, step)
        writer.add_scalar(
            f"{prefix}/loss_uncertainty_calib",
            self.loss_uncertainty_calib_item,
            step,
        )
        writer.add_scalar(f"{prefix}/loss_calib", self.loss_calib_item, step)
        writer.add_scalar(
            f"{prefix}/loss_multi_candidate",
            self.loss_multi_candidate_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/weighted_loss_min_k",
            self.weighted_loss_min_k_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/weighted_loss_mean_k",
            self.weighted_loss_mean_k_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/weighted_loss_boundary",
            self.weighted_loss_boundary_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/weighted_loss_evidence_div",
            self.weighted_loss_evidence_div_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/weighted_loss_gate_evidence",
            self.weighted_loss_gate_evidence_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/weighted_loss_calib",
            self.weighted_loss_calib_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/candidate/best_of_k_missing_l1",
            self.best_of_k_missing_l1_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/candidate/mean_k_missing_l1",
            self.mean_k_missing_l1_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/candidate/top1_missing_l1",
            self.top1_missing_l1_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/candidate/candidate0_missing_l1",
            self.candidate0_missing_l1_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/candidate/random_expected_missing_l1",
            self.random_expected_missing_l1_item,
            step,
        )
        writer.add_scalar(f"{prefix}/candidate/oracle_gain", self.oracle_gain_item, step)
        writer.add_scalar(
            f"{prefix}/candidate/top1_index_mean",
            self.candidate_top1_index_mean_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/candidate/pi_entropy",
            self.candidate_pi_entropy_item,
            step,
        )
        writer.add_scalar(f"{prefix}/candidate/pi_max", self.candidate_pi_max_item, step)
        writer.add_scalar(f"{prefix}/uncertainty/mean", self.uncertainty_mean_item, step)
        writer.add_scalar(f"{prefix}/uncertainty/min", self.uncertainty_min_item, step)
        writer.add_scalar(f"{prefix}/uncertainty/max", self.uncertainty_max_item, step)
        writer.add_scalar(f"{prefix}/loss_d", self.loss_D_item, step)
        writer.add_scalar(f"{prefix}/loss_d_real", self.loss_D_real_item, step)
        writer.add_scalar(f"{prefix}/loss_d_fake", self.loss_D_fake_item, step)
        writer.add_scalar(f"{prefix}/eta1", self.eta1_item, step)
        writer.add_scalar(f"{prefix}/eta2", self.eta2_item, step)
        writer.add_scalar(f"{prefix}/evidence/mean", self.evidence_mean_item, step)
        writer.add_scalar(f"{prefix}/evidence/min", self.evidence_min_item, step)
        writer.add_scalar(f"{prefix}/evidence/max", self.evidence_max_item, step)
        writer.add_scalar(f"{prefix}/gate/mean", self.gate_mean_item, step)
        writer.add_scalar(f"{prefix}/gate/min", self.gate_min_item, step)
        writer.add_scalar(f"{prefix}/gate/max", self.gate_max_item, step)
        writer.add_scalar(f"{prefix}/gate/target_mean", self.gate_target_mean_item, step)
        writer.add_scalar(f"{prefix}/gate/target_gap", self.gate_target_gap_item, step)
        writer.add_scalar(
            f"{prefix}/visual_evidence_aug/applied",
            self.visual_evidence_aug_applied_item,
            step,
        )
        for mode, value in sorted(self.visual_evidence_aug_mode_items.items()):
            writer.add_scalar(f"{prefix}/visual_evidence_aug/{mode}", value, step)
        writer.add_scalar(
            f"{prefix}/candidate/pairwise_distance",
            self.candidate_pairwise_distance_item,
            step,
        )
        writer.add_scalar(
            f"{prefix}/evidence_diversity_gap",
            self.evidence_diversity_gap_item,
            step,
        )
        if self.use_bottleneck_adapter:
            writer.add_scalar(f"{prefix}/adapter/scale", self.adapter_scale_item, step)
            writer.add_scalar(
                f"{prefix}/adapter/stochastic_scale",
                self.adapter_stochastic_scale_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/residual_l1",
                self.adapter_residual_l1_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/logvar_mean",
                self.adapter_logvar_mean_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/sigma_mean",
                self.adapter_sigma_mean_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/sigma_scale_mean",
                self.adapter_sigma_scale_mean_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/sigma_scale_min",
                self.adapter_sigma_scale_min_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/sigma_scale_max",
                self.adapter_sigma_scale_max_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/adapter/effective_sigma_mean",
                self.adapter_effective_sigma_mean_item,
                step,
            )
            writer.add_scalar(
                f"{prefix}/candidate/pairwise_l1",
                self.candidate_pairwise_l1_item,
                step,
            )

    def save_checkpoint(self, global_step, global_epoch, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"{self.hparams.name}_checkpoint_step{global_step:09d}.pth.tar",
        )
        checkpoint = {
            "Mel_Encoder": self.Mel_Encoder.state_dict(),
            "VideoEncoder": self.VideoEncoder.state_dict(),
            "Mel_Decoder": self.Mel_Decoder.state_dict(),
            "EvidenceEstimator": self.EvidenceEstimator.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict()
            if self.hparams.save_optimizer_state
            else None,
            "global_step": global_step,
            "global_epoch": global_epoch,
            "use_gan": self.use_gan,
            "stage": self._stage_name(),
            "enable_sync_loss": self.enable_sync_loss,
            "enable_probe_loss": self.enable_probe_loss,
            "enable_ec_viai_av": self.enable_ec_viai_av,
            "deterministic_adapter": bool(getattr(self.hparams, "deterministic_adapter", False)),
            "stochastic_adapter": bool(getattr(self.hparams, "stochastic_adapter", False)),
            "enable_evidence_gate": bool(getattr(self.hparams, "enable_evidence_gate", False)),
            "enable_candidate_scorer": bool(
                getattr(self.hparams, "enable_candidate_scorer", False)
            ),
            "calib_error_tau": float(getattr(self.hparams, "calib_error_tau", 0.1)),
            "lambda_calib": float(getattr(self.hparams, "lambda_calib", 0.0)),
            "freeze_gate_evidence_backbone": bool(
                getattr(self.hparams, "freeze_gate_evidence_backbone", False)
            ),
            "enable_evidence_scaled_sigma": bool(
                getattr(self.hparams, "enable_evidence_scaled_sigma", False)
            ),
            "evidence_sigma_scale_min": float(
                getattr(self.hparams, "evidence_sigma_scale_min", 0.5)
            ),
            "evidence_sigma_scale_max": float(
                getattr(self.hparams, "evidence_sigma_scale_max", 2.0)
            ),
            "num_candidates": int(getattr(self.hparams, "num_candidates", 1)),
            "test_num_candidates": int(
                getattr(self.hparams, "test_num_candidates", self.train_num_candidates)
            ),
            "evidence_diversity_d_min": float(
                getattr(self.hparams, "evidence_diversity_d_min", 0.02)
            ),
            "evidence_diversity_alpha": float(
                getattr(self.hparams, "evidence_diversity_alpha", 0.08)
            ),
            "lambda_gate_evidence": float(
                getattr(self.hparams, "lambda_gate_evidence", 0.0)
            ),
            "evidence_gate_low": float(getattr(self.hparams, "evidence_gate_low", 0.24)),
            "evidence_gate_high": float(getattr(self.hparams, "evidence_gate_high", 0.34)),
            "enable_visual_evidence_aug": bool(
                getattr(self.hparams, "enable_visual_evidence_aug", False)
            ),
            "visual_evidence_aug_prob": float(
                getattr(self.hparams, "visual_evidence_aug_prob", 0.5)
            ),
            "visual_evidence_aug_modes": getattr(
                self.hparams,
                "visual_evidence_aug_modes",
                "flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow",
            ),
        }
        if self.use_bottleneck_adapter:
            checkpoint["BottleneckAdapter"] = self.BottleneckAdapter.state_dict()
        if self.use_evidence_gate:
            checkpoint["EvidenceFusionGate"] = self.EvidenceFusionGate.state_dict()
        if self.use_candidate_scorer:
            checkpoint["CandidateScorer"] = self.CandidateScorer.state_dict()
            checkpoint["UncertaintyHead"] = self.UncertaintyHead.state_dict()
        if self.use_gan:
            checkpoint["netD"] = self.netD.state_dict()
            checkpoint["optimizer_D"] = (
                self.optimizer_D.state_dict()
                if self.hparams.save_optimizer_state
                else None
            )
        torch.save(checkpoint, checkpoint_path)
        print("Saved VIAI-AV checkpoint:", checkpoint_path)
        return checkpoint_path

    def _stage_name(self):
        if self.use_candidate_scorer:
            return "EC-VIAI-AV-stage8-candidate-scorer-calib"
        if self.use_evidence_gate and self.enable_evidence_scaled_sigma:
            return "EC-VIAI-AV-stage7d-evidence-scaled-sigma"
        if self.use_evidence_gate and self.freeze_gate_evidence_backbone:
            return "EC-VIAI-AV-stage7c-frozen-evidence-gate"
        if self.use_evidence_gate and (
            self.enable_visual_evidence_aug
            or float(getattr(self.hparams, "lambda_gate_evidence", 0.0)) > 0.0
        ):
            return "EC-VIAI-AV-stage7b-controlled-evidence-gate"
        if self.use_evidence_gate:
            return "EC-VIAI-AV-stage7-evidence-gate"
        if self.use_stochastic_adapter:
            return "EC-VIAI-AV-stage5-stochastic-adapter"
        if self.use_deterministic_adapter:
            return "EC-VIAI-AV-stage4-deterministic-adapter"
        return "VIAI-AV-stage4-sync-probe"

    def _checkpoint_use_gan(self, checkpoint):
        if "use_gan" in checkpoint:
            return bool(checkpoint["use_gan"])
        return "netD" in checkpoint

    def _assert_checkpoint_gan_mode(self, checkpoint, checkpoint_path):
        checkpoint_use_gan = self._checkpoint_use_gan(checkpoint)
        if checkpoint_use_gan == self.use_gan:
            return
        if checkpoint_use_gan:
            hint = "This checkpoint includes PatchGAN weights; rerun with --use_gan."
        else:
            hint = "This checkpoint was trained without PatchGAN; rerun without --use_gan."
        raise RuntimeError(
            "VIAI-AV checkpoint GAN mode mismatch: "
            f"checkpoint use_gan={checkpoint_use_gan}, current use_gan={self.use_gan}. "
            f"{hint} checkpoint={checkpoint_path}"
        )

    def _load_optional_module_state(self, module, checkpoint, key, label):
        if key not in checkpoint:
            print(f"[VIAI-AV] checkpoint has no {key}; {label} keeps current initialization")
            return False
        try:
            missing, unexpected = module.load_state_dict(checkpoint[key], strict=False)
        except RuntimeError as exc:
            print(f"[VIAI-AV] skipped {label} state because it is incompatible: {exc}")
            return False
        if missing or unexpected:
            print(
                f"[VIAI-AV] loaded {label} with missing={list(missing)} "
                f"unexpected={list(unexpected)}"
            )
        else:
            print(f"[VIAI-AV] loaded {label}")
        return True

    def _load_optimizer_state(self, optimizer, state_dict, label):
        if state_dict is None:
            return False
        try:
            optimizer.load_state_dict(state_dict)
        except (ValueError, RuntimeError) as exc:
            print(f"[VIAI-AV] skipped {label} state because it is incompatible: {exc}")
            return False
        print(f"[VIAI-AV] loaded {label} state")
        return True

    # 导入共有的结构权重，主要是Mel_Encoder和Mel_Decoder的权重，VideoEncoder不导入，因为VIAI-A没有视频编码器；如果有netD且当前模型使用GAN，则导入netD权重，否则不导入netD权重
    def load_checkpoint(self, checkpoint_path, reset_optimizer=False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self._assert_checkpoint_gan_mode(checkpoint, checkpoint_path)
        self.Mel_Encoder.load_state_dict(checkpoint["Mel_Encoder"])
        self.Mel_Decoder.load_state_dict(checkpoint["Mel_Decoder"])
        if self.use_gan:
            self.netD.load_state_dict(checkpoint["netD"])
        if "VideoEncoder" not in checkpoint:
            raise RuntimeError(
                "This VIAI-AV checkpoint does not contain VideoEncoder. "
                "Use a stage3 checkpoint saved by train-viai-av."
            )
        self.VideoEncoder.load_state_dict(checkpoint["VideoEncoder"])
        self._load_optional_module_state(
            self.EvidenceEstimator,
            checkpoint,
            "EvidenceEstimator",
            "EvidenceEstimator",
        )
        if self.BottleneckAdapter is None:
            if "BottleneckAdapter" in checkpoint:
                print(
                    "[VIAI-AV] checkpoint has BottleneckAdapter, but current "
                    "run does not enable it; skipped adapter weights"
                )
        else:
            self._load_optional_module_state(
                self.BottleneckAdapter,
                checkpoint,
                "BottleneckAdapter",
                "BottleneckAdapter",
            )
        if self.EvidenceFusionGate is None:
            if "EvidenceFusionGate" in checkpoint:
                print(
                    "[VIAI-AV] checkpoint has EvidenceFusionGate, but current "
                    "run does not enable it; skipped gate weights"
                )
        else:
            self._load_optional_module_state(
                self.EvidenceFusionGate,
                checkpoint,
                "EvidenceFusionGate",
                "EvidenceFusionGate",
            )
        if self.CandidateScorer is None:
            if "CandidateScorer" in checkpoint or "UncertaintyHead" in checkpoint:
                print(
                    "[VIAI-AV] checkpoint has candidate scorer weights, but current "
                    "run does not enable it; skipped scorer weights"
                )
        else:
            self._load_optional_module_state(
                self.CandidateScorer,
                checkpoint,
                "CandidateScorer",
                "CandidateScorer",
            )
            self._load_optional_module_state(
                self.UncertaintyHead,
                checkpoint,
                "UncertaintyHead",
                "UncertaintyHead",
            )
        checkpoint_stage = checkpoint.get("stage", "unknown")
        self.loaded_stage = self._stage_name()
        setattr(self.hparams, "loaded_stage", self.loaded_stage)
        if checkpoint_stage != self.loaded_stage:
            print(
                f"[VIAI-AV] checkpoint stage={checkpoint_stage}; "
                f"current model stage={self.loaded_stage}"
            )
        if not reset_optimizer:
            self._load_optimizer_state(
                self.optimizer_G,
                checkpoint.get("optimizer_G"),
                "optimizer_G",
            )
            if self.use_gan:
                self._load_optimizer_state(
                    self.optimizer_D,
                    checkpoint.get("optimizer_D"),
                    "optimizer_D",
                )
        return int(checkpoint.get("global_step", 0)), int(checkpoint.get("global_epoch", 0))

    def load_viai_a_checkpoint(self, checkpoint_path):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise RuntimeError(f"VIAI-A initialization checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "Mel_Encoder" not in checkpoint or "Mel_Decoder" not in checkpoint:
            raise RuntimeError(
                "VIAI-A initialization checkpoint must contain Mel_Encoder and Mel_Decoder."
            )
        print(f"[VIAI-AV] initializing audio branch from: {checkpoint_path}")
        _copy_matching_state(self.Mel_Encoder, checkpoint["Mel_Encoder"], "Mel_Encoder")
        _copy_matching_state(self.Mel_Decoder, checkpoint["Mel_Decoder"], "MelDecoderImage")
        if hasattr(self.Mel_Decoder, "init_deconv_1_1_1"):
            self.Mel_Decoder.init_deconv_1_1_1()
            print("[VIAI-AV] initialized MelDecoderImage fusion stem from audio decoder stem")
        if self.use_gan and "netD" in checkpoint:
            _copy_matching_state(self.netD, checkpoint["netD"], "MelDiscriminator")
        elif self.use_gan:
            print(
                "[VIAI-AV] source checkpoint has no netD; "
                "MelDiscriminator remains randomly initialized and will be trained by VIAI-AV"
            )
        return int(checkpoint.get("global_step", 0)), int(checkpoint.get("global_epoch", 0))
