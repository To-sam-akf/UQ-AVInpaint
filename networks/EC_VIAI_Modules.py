import torch
import torch.nn as nn
import torch.nn.functional as F


VISUAL_EVIDENCE_AUG_MODES = (
    "flow_75",
    "flow_50",
    "flow_25",
    "flow_zero",
    "static_video_zero_flow",
)


def normalize_visual_evidence_aug_modes(modes):
    if isinstance(modes, str):
        modes = [mode.strip() for mode in modes.split(",") if mode.strip()]
    normalized = []
    for mode in modes:
        if mode not in VISUAL_EVIDENCE_AUG_MODES:
            raise ValueError(
                f"Unknown visual evidence augmentation mode: {mode}. "
                f"Supported modes: {', '.join(VISUAL_EVIDENCE_AUG_MODES)}"
            )
        normalized.append(mode)
    return normalized


def apply_visual_evidence_augmentation(video_batch, flow_batch, mode):
    if mode == "none" or mode is None:
        return video_batch, flow_batch
    if mode == "flow_75":
        return video_batch, flow_batch * 0.75
    if mode == "flow_50":
        return video_batch, flow_batch * 0.50
    if mode == "flow_25":
        return video_batch, flow_batch * 0.25
    if mode == "flow_zero":
        return video_batch, torch.zeros_like(flow_batch)
    if mode == "static_video_zero_flow":
        static_video = video_batch[:, :1].expand_as(video_batch).contiguous()
        return static_video, torch.zeros_like(flow_batch)
    raise ValueError(
        f"Unknown visual evidence augmentation mode: {mode}. "
        f"Supported modes: {', '.join(VISUAL_EVIDENCE_AUG_MODES)}"
    )


class VisualEvidenceEstimator(nn.Module):
    """Deterministic visual evidence score for EC-VIAI-AV.

    The first version is deliberately parameter-free: it can be logged and
    sanity-checked before any EC gate or uncertainty module is trained.
    """

    def __init__(self):
        super(VisualEvidenceEstimator, self).__init__()

    def _flow_magnitude(self, flow_batch):
        if flow_batch.dim() == 5:
            if flow_batch.size(2) == 2:
                flow_x = flow_batch[:, :, 0]
                flow_y = flow_batch[:, :, 1]
            elif flow_batch.size(1) == 2:
                flow_x = flow_batch[:, 0]
                flow_y = flow_batch[:, 1]
            else:
                raise ValueError("Expected a 2-channel flow tensor.")
        elif flow_batch.dim() == 4 and flow_batch.size(1) == 2:
            flow_x = flow_batch[:, 0].unsqueeze(1)
            flow_y = flow_batch[:, 1].unsqueeze(1)
        else:
            raise ValueError("Expected flow_batch shape [B, T, 2, H, W] or [B, 2, T, H, W].")
        return torch.sqrt(torch.clamp(flow_x ** 2 + flow_y ** 2, min=1e-12))

    def forward(
        self,
        video_feature,
        flow_batch,
        mel_target_feature_flat,
        video_feature_flat,
    ):
        batch_size = video_feature.size(0)
        dtype = video_feature.dtype
        device = video_feature.device

        if flow_batch is None:
            mean_mag = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            temporal_signal = torch.zeros(batch_size, 1, device=device, dtype=dtype)
        else:
            magnitude = self._flow_magnitude(flow_batch.to(device=device, dtype=dtype))
            mean_mag = magnitude.flatten(1).mean(dim=1, keepdim=True)
            per_frame_mag = magnitude.mean(dim=(-2, -1))
            temporal_var = per_frame_mag.var(dim=1, unbiased=False, keepdim=True)
            if per_frame_mag.size(1) > 1:
                temporal_diff = torch.abs(per_frame_mag[:, 1:] - per_frame_mag[:, :-1]).mean(
                    dim=1,
                    keepdim=True,
                )
            else:
                temporal_diff = torch.zeros(batch_size, 1, device=device, dtype=dtype)
            temporal_signal = temporal_var + temporal_diff

        motion_score = 1.0 - torch.exp(-3.0 * mean_mag)
        temporal_score = 1.0 - torch.exp(-5.0 * temporal_signal)

        audio_embedding = F.normalize(mel_target_feature_flat.detach(), p=2, dim=1)
        video_embedding = F.normalize(video_feature_flat, p=2, dim=1)
        sync_distance = torch.norm(audio_embedding - video_embedding, p=2, dim=1, keepdim=True)
        sync_score = 1.0 - torch.clamp(sync_distance / 2.0, min=0.0, max=1.0)

        feature_score = torch.tanh(video_feature.abs().flatten(1).mean(dim=1, keepdim=True))
        logit = (
            -2.5
            + 4.0 * motion_score
            + 2.0 * temporal_score
            + 2.0 * sync_score
            + 0.5 * feature_score
        )
        return torch.sigmoid(logit)


class EvidenceFusionGate(nn.Module):
    """Calibrate video features according to audio context and visual evidence."""

    def __init__(self, feature_channels=256, hidden_channels=256):
        super(EvidenceFusionGate, self).__init__()
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        self.audio_prior = nn.Conv2d(
            self.feature_channels,
            self.feature_channels,
            kernel_size=1,
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.feature_channels * 2 + 1, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, 1),
        )
        self.initial()

    def initial(self):
        nn.init.eye_(self.audio_prior.weight.view(self.feature_channels, self.feature_channels))
        if self.audio_prior.bias is not None:
            nn.init.constant_(self.audio_prior.bias, 0.0)
        for module in self.gate_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _reshape_video_feature(self, video_feature, audio_bottleneck):
        if video_feature.dim() != 4:
            raise ValueError("video_feature must be a 4D tensor.")
        batch_size, _, height, width = audio_bottleneck.shape
        if video_feature.shape == audio_bottleneck.shape:
            return video_feature
        return video_feature.reshape(batch_size, -1, height, width)

    def forward(self, audio_bottleneck, video_feature, evidence):
        if audio_bottleneck.dim() != 4:
            raise ValueError("audio_bottleneck must be a 4D tensor.")
        if audio_bottleneck.size(1) != self.feature_channels:
            raise ValueError(
                f"Expected audio_bottleneck to have {self.feature_channels} channels, "
                f"got {audio_bottleneck.size(1)}."
            )

        video_feature = self._reshape_video_feature(video_feature, audio_bottleneck)
        if video_feature.shape != audio_bottleneck.shape:
            raise ValueError(
                "video_feature must reshape to audio_bottleneck shape; got "
                f"{tuple(video_feature.shape)} and {tuple(audio_bottleneck.shape)}."
            )

        evidence = evidence.to(device=audio_bottleneck.device, dtype=audio_bottleneck.dtype)
        evidence = evidence.reshape(audio_bottleneck.size(0), -1)
        if evidence.size(1) != 1:
            evidence = evidence.mean(dim=1, keepdim=True)

        audio_pool = F.adaptive_avg_pool2d(audio_bottleneck, output_size=1).flatten(1)
        video_pool = F.adaptive_avg_pool2d(video_feature, output_size=1).flatten(1)
        gate_input = torch.cat([audio_pool, video_pool, evidence], dim=1)
        gate = torch.sigmoid(self.gate_mlp(gate_input)).view(-1, 1, 1, 1)
        audio_prior = self.audio_prior(audio_bottleneck)
        calibrated_video = gate * video_feature + (1.0 - gate) * audio_prior
        return calibrated_video, gate, audio_prior


class CandidateScorer(nn.Module):
    """Score K inpainting candidates with test-time-safe proxy features."""

    def __init__(self, feature_channels=256, candidate_stat_dim=3, hidden_channels=256):
        super(CandidateScorer, self).__init__()
        self.feature_channels = int(feature_channels)
        self.candidate_stat_dim = int(candidate_stat_dim)
        self.hidden_channels = int(hidden_channels)
        self.scorer = nn.Sequential(
            nn.Linear(self.feature_channels * 2 + self.candidate_stat_dim + 1, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, 1),
        )
        self.initial()

    def initial(self):
        linear_layers = [module for module in self.scorer.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        last = linear_layers[-1]
        nn.init.constant_(last.weight, 0.0)
        if last.bias is not None:
            nn.init.constant_(last.bias, 0.0)

    def _pool_feature(self, feature):
        if feature.dim() != 4:
            raise ValueError("feature must be a 4D tensor.")
        return F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)

    def forward(self, candidate_stats, audio_bottleneck, video_feature, evidence):
        if candidate_stats.dim() != 3:
            raise ValueError("candidate_stats must have shape [B, K, S].")
        batch_size, num_candidates, stat_dim = candidate_stats.shape
        if stat_dim != self.candidate_stat_dim:
            raise ValueError(
                f"Expected candidate_stats dim {self.candidate_stat_dim}, got {stat_dim}."
            )

        audio_pool = self._pool_feature(audio_bottleneck)
        video_pool = self._pool_feature(video_feature)
        evidence = evidence.to(device=candidate_stats.device, dtype=candidate_stats.dtype)
        evidence = evidence.reshape(batch_size, -1)
        if evidence.size(1) != 1:
            evidence = evidence.mean(dim=1, keepdim=True)

        audio_pool = audio_pool.unsqueeze(1).expand(-1, num_candidates, -1)
        video_pool = video_pool.unsqueeze(1).expand(-1, num_candidates, -1)
        evidence = evidence.unsqueeze(1).expand(-1, num_candidates, -1)
        scorer_input = torch.cat(
            [
                candidate_stats,
                audio_pool.to(dtype=candidate_stats.dtype),
                video_pool.to(dtype=candidate_stats.dtype),
                evidence,
            ],
            dim=2,
        )
        logits = self.scorer(scorer_input.reshape(batch_size * num_candidates, -1))
        logits = logits.reshape(batch_size, num_candidates)
        pi = F.softmax(logits, dim=1)
        return logits, pi


class UncertaintyHead(nn.Module):
    """Predict sample-level uncertainty from pooled context and scorer statistics."""

    def __init__(self, feature_channels=256, stats_dim=7, hidden_channels=256):
        super(UncertaintyHead, self).__init__()
        self.feature_channels = int(feature_channels)
        self.stats_dim = int(stats_dim)
        self.hidden_channels = int(hidden_channels)
        self.head = nn.Sequential(
            nn.Linear(self.feature_channels * 2 + self.stats_dim, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(True),
            nn.Linear(self.hidden_channels, 1),
        )
        self.initial()

    def initial(self):
        linear_layers = [module for module in self.head.modules() if isinstance(module, nn.Linear)]
        for module in linear_layers[:-1]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        last = linear_layers[-1]
        nn.init.constant_(last.weight, 0.0)
        if last.bias is not None:
            nn.init.constant_(last.bias, 0.0)

    def _pool_feature(self, feature):
        if feature.dim() != 4:
            raise ValueError("feature must be a 4D tensor.")
        return F.adaptive_avg_pool2d(feature, output_size=1).flatten(1)

    def forward(self, audio_bottleneck, video_feature, uncertainty_stats):
        if uncertainty_stats.dim() != 2:
            raise ValueError("uncertainty_stats must have shape [B, S].")
        if uncertainty_stats.size(1) != self.stats_dim:
            raise ValueError(
                f"Expected uncertainty_stats dim {self.stats_dim}, got {uncertainty_stats.size(1)}."
            )
        audio_pool = self._pool_feature(audio_bottleneck).to(dtype=uncertainty_stats.dtype)
        video_pool = self._pool_feature(video_feature).to(dtype=uncertainty_stats.dtype)
        head_input = torch.cat([audio_pool, video_pool, uncertainty_stats], dim=1)
        return torch.sigmoid(self.head(head_input))


class BottleneckAdapter(nn.Module):
    """Residual adapter for deterministic and stochastic VIAI-AV bottlenecks."""

    def __init__(
        self,
        feature_channels=256,
        hidden_channels=256,
        init_scale=0.0,
        stochastic_init_scale=1e-3,
    ):
        super(BottleneckAdapter, self).__init__()
        self.feature_channels = int(feature_channels)
        self.hidden_channels = int(hidden_channels)
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale)))
        self.stochastic_residual_scale = nn.Parameter(
            torch.tensor(float(stochastic_init_scale))
        )
        self.adapter = nn.Sequential(
            nn.Conv2d(self.feature_channels * 2, self.hidden_channels, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, self.feature_channels, kernel_size=1),
        )
        self.mu_head = nn.Conv2d(
            self.feature_channels * 2,
            self.feature_channels,
            kernel_size=1,
        )
        self.logvar_head = nn.Conv2d(
            self.feature_channels * 2,
            self.feature_channels,
            kernel_size=1,
        )
        self.stochastic_adapter = nn.Sequential(
            nn.Conv2d(self.feature_channels, self.hidden_channels, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(self.hidden_channels, self.feature_channels, kernel_size=1),
        )
        self.initial()

    def initial(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _reshape_video_feature(self, video_feature, mel_bottleneck):
        if video_feature.dim() != 4:
            raise ValueError("video_feature must be a 4D tensor.")
        batch_size, _, height, width = mel_bottleneck.shape
        if video_feature.shape == mel_bottleneck.shape:
            return video_feature
        return video_feature.reshape(batch_size, -1, height, width)

    def _bottleneck_input(self, mel_bottleneck, video_feature):
        if mel_bottleneck.dim() != 4:
            raise ValueError("mel_bottleneck must be a 4D tensor.")
        if mel_bottleneck.size(1) != self.feature_channels:
            raise ValueError(
                f"Expected mel_bottleneck to have {self.feature_channels} channels, "
                f"got {mel_bottleneck.size(1)}."
            )

        video_bottleneck = self._reshape_video_feature(video_feature, mel_bottleneck)
        if video_bottleneck.shape != mel_bottleneck.shape:
            raise ValueError(
                "video_feature must reshape to the same bottleneck shape as "
                f"mel_bottleneck; got {tuple(video_bottleneck.shape)} and "
                f"{tuple(mel_bottleneck.shape)}."
            )

        return torch.cat([mel_bottleneck, video_bottleneck], dim=1)

    def forward(self, mel_bottleneck, video_feature):
        bottleneck_input = self._bottleneck_input(mel_bottleneck, video_feature)
        residual = self.adapter(bottleneck_input)
        return self.residual_scale.to(dtype=residual.dtype) * residual

    def latent_parameters(self, mel_bottleneck, video_feature):
        bottleneck_input = self._bottleneck_input(mel_bottleneck, video_feature)
        mu = self.mu_head(bottleneck_input)
        logvar = self.logvar_head(bottleneck_input).clamp(min=-10.0, max=2.0)
        return mu, logvar

    def sample_latent(
        self,
        mu,
        logvar,
        num_candidates,
        sigma_min=0.0,
        sigma_max=1.0,
        sigma_scale=None,
    ):
        num_candidates = int(num_candidates)
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1.")
        sigma_min = float(sigma_min)
        sigma_max = float(sigma_max)
        if sigma_min < 0.0:
            raise ValueError("sigma_min must be >= 0.")
        if sigma_max < sigma_min:
            raise ValueError("sigma_max must be >= sigma_min.")

        sigma = torch.exp(0.5 * logvar)
        if sigma_scale is not None:
            sigma_scale = sigma_scale.to(device=mu.device, dtype=mu.dtype)
            if sigma_scale.dim() == 1:
                sigma_scale = sigma_scale.view(-1, 1, 1, 1)
            elif sigma_scale.dim() == 2:
                sigma_scale = sigma_scale.view(sigma_scale.size(0), -1, 1, 1)
            if sigma_scale.size(0) != mu.size(0):
                raise ValueError(
                    "sigma_scale batch size must match mu batch size; got "
                    f"{sigma_scale.size(0)} and {mu.size(0)}."
                )
            if sigma_scale.size(1) != 1:
                sigma_scale = sigma_scale.mean(dim=1, keepdim=True)
            sigma = sigma * sigma_scale
        sigma = torch.clamp(sigma, min=sigma_min, max=sigma_max)
        eps = torch.randn(
            mu.size(0),
            num_candidates,
            mu.size(1),
            mu.size(2),
            mu.size(3),
            device=mu.device,
            dtype=mu.dtype,
        )
        eps[:, 0].zero_()
        z = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps
        return z, sigma

    def sample_adapter(self, z):
        if z.dim() != 5:
            raise ValueError("z must have shape [B, K, C, H, W].")
        batch_size, num_candidates, channels, height, width = z.shape
        if channels != self.feature_channels:
            raise ValueError(
                f"Expected z to have {self.feature_channels} channels, got {channels}."
            )
        residual = self.stochastic_adapter(
            z.reshape(batch_size * num_candidates, channels, height, width)
        )
        residual = residual.reshape(
            batch_size,
            num_candidates,
            self.feature_channels,
            height,
            width,
        )
        scale = self.stochastic_residual_scale.to(dtype=residual.dtype)
        return scale * residual

    def sample_residuals(
        self,
        mel_bottleneck,
        video_feature,
        num_candidates,
        sigma_min=0.0,
        sigma_max=1.0,
        sigma_scale=None,
    ):
        mu, logvar = self.latent_parameters(mel_bottleneck, video_feature)
        z, sigma = self.sample_latent(
            mu,
            logvar,
            num_candidates,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            sigma_scale=sigma_scale,
        )
        residual = self.sample_adapter(z)
        return residual, mu, logvar, sigma
