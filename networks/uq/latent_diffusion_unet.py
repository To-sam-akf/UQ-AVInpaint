"""Conditional U-Net for latent-space Mel inpainting (P4).

Operates on latent space [B, C=8, H=10, W=50] produced by the frozen Mel AE.

Input: 19 channels = z_t (8) + z_context (8) + mask_z (1) + boundary_map_z (2)
Output: 8 channels = epsilon prediction

Conditioning:
  - Time embedding: sinusoidal → MLP → injected via FiLM in each residual block.
  - Video tokens: frame positional embedding + gated multi-level cross-attention.

Architecture follows a standard 2D U-Net with residual blocks and skip connections.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------

def sinusoidal_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    Args:
        timesteps: [B] long tensor
        dim: embedding dimension (must be even)
        max_period: maximum period (default 10000)
    Returns:
        [B, dim]
    """
    half = dim // 2
    device = timesteps.device
    exponent = -math.log(max_period) * torch.arange(
        0, half, dtype=torch.float32, device=device
    ) / half
    freqs = torch.exp(exponent)
    args = timesteps.float()[:, None] * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding → MLP."""

    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.linear = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, t):
        emb = sinusoidal_embedding(t, self.linear[0].in_features)
        return self.linear(emb)


# ---------------------------------------------------------------------------
# Residual block with time conditioning (FiLM)
# ---------------------------------------------------------------------------

class ResBlock2D(nn.Module):
    """2D residual block with optional time FiLM conditioning."""

    def __init__(self, in_ch, out_ch, time_emb_dim=None, stride=1,
                 norm_type="batch"):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride

        Norm = nn.BatchNorm2d if norm_type == "batch" else nn.InstanceNorm2d
        use_bias = norm_type == "instance"

        self.norm1 = Norm(in_ch, affine=True) if in_ch > 0 else nn.Identity()
        self.conv1 = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, stride=stride, padding=1,
            bias=use_bias,
        )
        self.norm2 = Norm(out_ch, affine=True)
        self.conv2 = nn.Conv2d(
            out_ch, out_ch, kernel_size=3, stride=1, padding=1,
            bias=use_bias,
        )
        self.act = nn.SiLU()

        # 1×1 skip if channels or stride change
        self.skip = None
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Conv2d(
                in_ch, out_ch, kernel_size=1, stride=stride, bias=False,
            )

        # FiLM projection from time embedding
        self.time_proj = None
        if time_emb_dim is not None and time_emb_dim > 0:
            self.time_proj = nn.Linear(time_emb_dim, out_ch * 2)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, time_emb=None):
        """
        Args:
            x: [B, in_ch, H, W]
            time_emb: [B, time_emb_dim] or None
        Returns:
            [B, out_ch, H', W']
        """
        residual = x

        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        # FiLM
        if self.time_proj is not None and time_emb is not None:
            scale, shift = self.time_proj(time_emb).chunk(2, dim=1)
            h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.act(h)
        h = self.conv2(h)

        if self.skip is not None:
            residual = self.skip(residual)
        return h + residual


# ---------------------------------------------------------------------------
# Cross-attention for video token injection
# ---------------------------------------------------------------------------

class CrossAttention2D(nn.Module):
    """Cross-attention over spatial positions attending to video tokens.

    Reshapes spatial features [B, C, H, W] → [B, H*W, C],
    attends to video tokens [B, F, D], reshapes back.
    """

    def __init__(self, query_dim, kv_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        assert self.head_dim * num_heads == query_dim, \
            f"query_dim {query_dim} not divisible by num_heads {num_heads}"

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_v = nn.Linear(kv_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x_spatial, video_tokens):
        """
        Args:
            x_spatial: [B, C, H, W] query features
            video_tokens: [B, F, D] key/value features (F=50 frames)
        Returns:
            [B, C, H, W]
        """
        B, C, H, W = x_spatial.shape
        S = H * W  # number of spatial positions

        # Spatial queries: [B, S, C]
        q = x_spatial.reshape(B, C, S).transpose(1, 2)  # [B, S, C]
        q = self.to_q(q)

        # Video keys/values: [B, F, C]
        k = self.to_k(video_tokens)
        v = self.to_v(video_tokens)

        # Multi-head reshape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, nh, S, hd]
        out = out.transpose(1, 2).contiguous().view(B, S, C)
        out = self.to_out(out)

        return out.transpose(1, 2).view(B, C, H, W)


class GatedCrossAttention2D(nn.Module):
    """Residual gated video cross-attention for a 2D feature map."""

    def __init__(self, query_dim, kv_dim, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = CrossAttention2D(
            query_dim=query_dim,
            kv_dim=kv_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x_spatial, video_tokens):
        attn_out = self.attn(x_spatial, video_tokens)
        gate = torch.sigmoid(self.gate)
        return x_spatial + gate * attn_out, {
            "gate": gate.detach().mean(),
            "attn_norm": attn_out.detach().pow(2).mean().sqrt(),
        }


# ---------------------------------------------------------------------------
# Downsample / Upsample helpers
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, in_ch, stride=(2, 2)):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride,
                              padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsample via interpolation + conv to avoid spatial mismatch.

    Uses nearest-neighbour interpolation to the target size, followed
    by a regular 3×3 convolution.  This is more robust than
    ConvTranspose2d when the input spatial dims are small.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1,
                              padding=1)

    def forward(self, x, target_size):
        """Args:
            x: [B, C, H_in, W_in]
            target_size: (H_out, W_out)
        Returns:
            [B, out_ch, H_out, W_out]
        """
        x = F.interpolate(x, size=target_size, mode="nearest")
        return self.conv(x)


# ===================================================================
# Main U-Net
# ===================================================================

class LatentDiffusionUNet(nn.Module):
    """Conditional U-Net for latent-space Mel inpainting.

    Args:
        in_channels:  input channels (19 = 8 z_t + 8 z_ctx + 1 mask + 2 bdy)
        out_channels: output channels (8 = epsilon prediction)
        base_channels: base channel count
        channel_mult:  multiplier per encoder stage
        time_emb_dim:  time embedding dimension
        video_dim:     video token dimension (from VideoEvidenceEncoder)
        num_heads:     cross-attention heads
        norm_type:     "batch" or "instance"
    """

    max_video_frames = 50

    def __init__(
        self,
        in_channels=19,
        out_channels=8,
        base_channels=64,
        channel_mult=(1, 2, 4, 8),
        time_emb_dim=256,
        video_dim=256,
        num_heads=4,
        norm_type="batch",
        attn_resolutions=None,  # kept for checkpoint/config compatibility
    ):
        super().__init__()
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.video_dim = video_dim
        self.num_heads = num_heads

        self.frame_pos_embed = nn.Parameter(
            torch.zeros(1, self.max_video_frames, video_dim)
        )

        # Time embedding
        self.time_embed = TimeEmbedding(time_emb_dim)

        # Encoder stages — store (channels, spatial_size) per level
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        ch = base_channels
        prev_ch = in_channels

        encoder_channels = []
        encoder_spatial = []  # (H, W) after each encoder block
        cur_h, cur_w = 10, 50  # input spatial size
        for level, mult in enumerate(channel_mult):
            out_ch = base_channels * mult
            if level == 0:
                stride_hw = (1, 1)
            else:
                # Downsample freq at level 1, time+small-freq at level 2
                stride_hw = ((2, 1), (1, 2), (2, 2))[min(level - 1, 2)]
            self.encoder_blocks.append(
                ResBlock2D(prev_ch, out_ch, time_emb_dim=time_emb_dim,
                           stride=stride_hw, norm_type=norm_type)
            )
            self.encoder_attns.append(
                GatedCrossAttention2D(
                    query_dim=out_ch, kv_dim=video_dim, num_heads=num_heads,
                )
            )
            cur_h = (cur_h - 1) // stride_hw[0] + 1
            cur_w = (cur_w - 1) // stride_hw[1] + 1
            encoder_channels.append(out_ch)
            encoder_spatial.append((cur_h, cur_w))
            prev_ch = out_ch

        # Bottleneck
        self.bottleneck_block = ResBlock2D(
            prev_ch, prev_ch, time_emb_dim=time_emb_dim, stride=1,
            norm_type=norm_type,
        )
        bottleneck_spatial = (cur_h, cur_w)

        # Cross-attention at bottleneck
        self.bottleneck_attn = GatedCrossAttention2D(
            query_dim=prev_ch, kv_dim=video_dim, num_heads=num_heads,
        )

        # Decoder stages — use interpolation + conv upsample
        self.decoder_upsamples = nn.ModuleList()
        self.decoder_resblocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_target_sizes = []  # (H, W) per decoder level

        rev_mult = list(reversed(channel_mult))
        for level in range(len(rev_mult) - 1):
            skip_idx = -(level + 2)
            skip_ch = encoder_channels[skip_idx]
            target_h, target_w = encoder_spatial[skip_idx]
            out_ch = base_channels * rev_mult[level + 1]
            in_ch = prev_ch + skip_ch
            self.decoder_upsamples.append(Upsample(prev_ch, prev_ch))
            self.decoder_resblocks.append(
                ResBlock2D(
                    in_ch, out_ch, time_emb_dim=time_emb_dim,
                    stride=1, norm_type=norm_type,
                )
            )
            self.decoder_attns.append(
                GatedCrossAttention2D(
                    query_dim=out_ch, kv_dim=video_dim, num_heads=num_heads,
                )
            )
            self.decoder_target_sizes.append((target_h, target_w))
            prev_ch = out_ch

        # Final projection
        self.final_conv = nn.Conv2d(
            prev_ch + encoder_channels[0], out_channels,
            kernel_size=3, stride=1, padding=1,
        )

        self._init_weights()
        nn.init.normal_(self.frame_pos_embed, mean=0.0, std=0.02)
        self._reset_video_diagnostics(device=None)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _zero_scalar(self, device):
        return torch.tensor(0.0, device=device)

    def _reset_video_diagnostics(self, device):
        self.video_gate_mean = self._zero_scalar(device)
        self.video_attn_norm = self._zero_scalar(device)
        self.video_token_norm = self._zero_scalar(device)

    def apply_frame_positional_embedding(self, video_tokens):
        """Add frame positions while preserving all-zero token samples."""
        if video_tokens is None:
            return None
        if video_tokens.dim() != 3:
            raise ValueError(
                "video_tokens must have shape [B, F, D], got "
                f"{tuple(video_tokens.shape)}"
            )
        if video_tokens.size(-1) != self.video_dim:
            raise ValueError(
                f"video token dim {video_tokens.size(-1)} does not match "
                f"U-Net video_dim={self.video_dim}"
            )
        frame_count = video_tokens.size(1)
        if frame_count > self.max_video_frames:
            raise ValueError(
                f"video frame count {frame_count} exceeds supported "
                f"max_video_frames={self.max_video_frames}"
            )

        pos = self.frame_pos_embed[:, :frame_count].to(
            device=video_tokens.device, dtype=video_tokens.dtype,
        )
        nonzero = (
            video_tokens.detach().abs().sum(dim=(1, 2), keepdim=True) > 0
        ).to(dtype=video_tokens.dtype)
        return video_tokens + nonzero * pos

    def _record_video_diagnostics(self, video_tokens, attn_stats):
        if video_tokens is None:
            self._reset_video_diagnostics(device=None)
            return
        device = video_tokens.device
        if attn_stats:
            gate_values = torch.stack([stats["gate"] for stats in attn_stats])
            attn_values = torch.stack(
                [stats["attn_norm"] for stats in attn_stats]
            )
            self.video_gate_mean = gate_values.mean().detach()
            self.video_attn_norm = attn_values.mean().detach()
        else:
            self.video_gate_mean = self._zero_scalar(device)
            self.video_attn_norm = self._zero_scalar(device)
        self.video_token_norm = (
            video_tokens.detach().pow(2).mean().sqrt()
        )

    def forward(self, x, t, video_tokens=None):
        """
        Args:
            x: [B, 19, 10, 50] concatenated input
            t: [B] long tensor with timestep indices
            video_tokens: [B, 50, video_dim] or None
        Returns:
            epsilon_pred: [B, 8, 10, 50]
        """
        time_emb = self.time_embed(t)
        video_tokens = self.apply_frame_positional_embedding(video_tokens)
        attn_stats = []

        # Encoder
        skips = []
        h = x
        for block, attn in zip(self.encoder_blocks, self.encoder_attns):
            h = block(h, time_emb)
            if video_tokens is not None:
                h, stats = attn(h, video_tokens)
                attn_stats.append(stats)
            skips.append(h)

        # Bottleneck
        h = self.bottleneck_block(h, time_emb)
        if video_tokens is not None:
            h, stats = self.bottleneck_attn(h, video_tokens)
            attn_stats.append(stats)

        # Decoder with skip connections
        for level in range(len(self.decoder_upsamples)):
            target_size = self.decoder_target_sizes[level]
            h = self.decoder_upsamples[level](h, target_size)
            skip = skips[-(level + 2)]  # matching skip
            h = torch.cat([h, skip], dim=1)
            h = self.decoder_resblocks[level](h, time_emb)
            if video_tokens is not None:
                h, stats = self.decoder_attns[level](h, video_tokens)
                attn_stats.append(stats)

        # Final: concat first encoder skip and project
        h = torch.cat([h, skips[0]], dim=1)
        h = self.final_conv(h)
        self._record_video_diagnostics(video_tokens, attn_stats)

        return h
