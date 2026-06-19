"""Video Evidence Encoder — P3 minimal version.

Encodes video (RGB) and optical flow into per-frame tokens for
cross-attention conditioning of the latent diffusion U-Net.

P3 minimal: no evidence gate, no motion-strength, no sync_logits.
Just rgb_tokens + flow_tokens → video_tokens [B, 50, D].

Architecture:
  - Shared 2D CNN over 256×256 frames → per-frame feature vectors.
  - Separate stream for RGB (3-ch) and flow (2-ch).
  - Concatenate → projection → video_tokens.
"""

import torch
import torch.nn as nn


def _conv_relu(in_ch, out_ch, kernel=3, stride=2, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class PerFrameCNN(nn.Module):
    """Lightweight CNN that encodes a single frame → feature vector."""

    def __init__(self, in_channels=3, feature_dim=256, image_size=256):
        super().__init__()
        # 256 → 128 → 64 → 32 → 16 → 8
        self.conv1 = _conv_relu(in_channels, 32)   # 256 → 128
        self.conv2 = _conv_relu(32, 64)             # 128 → 64
        self.conv3 = _conv_relu(64, 128)            # 64 → 32
        self.conv4 = _conv_relu(128, 256)           # 32 → 16
        self.conv5 = _conv_relu(256, feature_dim)   # 16 → 8

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] single frame or [B*F, C, H, W] stacked frames.
        Returns:
            [B, feature_dim] or [B*F, feature_dim]
        """
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        h = self.pool(h)
        return h.view(h.size(0), self.feature_dim)


class VideoEvidenceEncoderP3(nn.Module):
    """P3-minimal video evidence encoder.

    Outputs:
        rgb_tokens:  [B, 50, D] per-frame RGB features
        flow_tokens: [B, 50, D] per-frame flow features
        video_tokens:[B, 50, D] fused (rgb + flow) tokens

    The P3 version does NOT produce evidence, motion_strength,
    sync_logits, or visual_logvar — those are added in P5.
    """

    def __init__(self, video_dim=256, image_size=256,
                 rgb_feature_dim=128, flow_feature_dim=128):
        super().__init__()
        self.video_dim = int(video_dim)

        self.rgb_cnn = PerFrameCNN(
            in_channels=3, feature_dim=rgb_feature_dim,
            image_size=image_size,
        )
        self.flow_cnn = PerFrameCNN(
            in_channels=2, feature_dim=flow_feature_dim,
            image_size=image_size,
        )

        self.rgb_projection = nn.Linear(rgb_feature_dim, video_dim)
        self.flow_projection = nn.Linear(flow_feature_dim, video_dim)

        fusion_dim = video_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, video_dim),
            nn.ReLU(inplace=True),
            nn.Linear(video_dim, video_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, video, flow):
        """
        Args:
            video: [B, F, 3, H, W]  F=50, H=W=256
            flow:  [B, F, 2, H, W]
        Returns:
            dict with keys:
                rgb_tokens:   [B, F, video_dim]
                flow_tokens:  [B, F, video_dim]
                video_tokens: [B, F, video_dim]
        """
        B, F = video.shape[:2]

        # Process all frames at once by stacking batch and frame dims
        video_flat = video.reshape(B * F, *video.shape[2:])
        flow_flat = flow.reshape(B * F, *flow.shape[2:])

        rgb_feat = self.rgb_cnn(video_flat)     # [B*F, rgb_feature_dim]
        flow_feat = self.flow_cnn(flow_flat)     # [B*F, flow_feature_dim]

        rgb_feat = rgb_feat.view(B, F, -1)       # [B, F, rgb_feature_dim]
        flow_feat = flow_feat.view(B, F, -1)     # [B, F, flow_feature_dim]

        rgb_tokens = self.rgb_projection(rgb_feat)       # [B, F, video_dim]
        flow_tokens = self.flow_projection(flow_feat)    # [B, F, video_dim]

        fused = torch.cat([rgb_tokens, flow_tokens], dim=-1)
        video_tokens = self.fusion(fused)                # [B, F, video_dim]

        return {
            "rgb_tokens": rgb_tokens,
            "flow_tokens": flow_tokens,
            "video_tokens": video_tokens,
        }


class VideoConditionDummy(nn.Module):
    """Placeholder when video conditioning is unavailable.

    Returns zero tokens of the correct shape so the U-Net can still
    run in audio-only mode.
    """

    def __init__(self, video_dim=256):
        super().__init__()
        self.video_dim = int(video_dim)

    def forward(self, video, flow):
        B, F = video.shape[:2]
        device = video.device
        z = torch.zeros(B, F, self.video_dim, device=device, dtype=video.dtype)
        return {
            "rgb_tokens": z,
            "flow_tokens": z,
            "video_tokens": z,
        }
