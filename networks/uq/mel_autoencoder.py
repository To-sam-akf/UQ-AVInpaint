"""Deterministic convolutional Mel autoencoder.

Encoder:  80×200 → 40×100 → 20×50 → 10×50
Decoder:  10×50 → 20×50 → 40×100 → 80×200  (symmetric, final sigmoid)

Latent shape: [B, 8, 10, 50] — 50 time frames aligned with 50 video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm_layer(norm_type: str, channels: int) -> nn.Module:
    if norm_type == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    return nn.BatchNorm2d(channels, affine=True)


def _conv_block(
    in_ch: int,
    out_ch: int,
    kernel_size: tuple = (3, 3),
    stride: tuple = (1, 1),
    padding: tuple = (1, 1),
    norm_type: str = "batch",
    activation: str = "relu",
) -> nn.Sequential:
    use_bias = norm_type == "instance"
    layers: list = [
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                  padding=padding, bias=use_bias),
        _norm_layer(norm_type, out_ch),
    ]
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _transpose_conv_block(
    in_ch: int,
    out_ch: int,
    kernel_size: tuple = (3, 3),
    stride: tuple = (1, 1),
    padding: tuple = (1, 1),
    output_padding: tuple = (0, 0),
    norm_type: str = "batch",
    activation: str = "relu",
) -> nn.Sequential:
    use_bias = norm_type == "instance"
    layers: list = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=use_bias),
        _norm_layer(norm_type, out_ch),
    ]
    if activation == "relu":
        layers.append(nn.ReLU(inplace=True))
    elif activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


class MelEncoder(nn.Module):
    """Encode Mel spectrogram [B, 1, 80, 200] → latent [B, lat_dim, 10, 50]."""

    def __init__(self, latent_dim: int = 8, base_channels: int = 32,
                 norm_type: str = "batch"):
        super().__init__()
        # 80×200 → 40×100
        self.enc1 = _conv_block(1, base_channels, stride=(2, 2),
                                norm_type=norm_type, activation="relu")
        # 40×100 → 20×50
        self.enc2 = _conv_block(base_channels, base_channels * 2, stride=(2, 2),
                                norm_type=norm_type, activation="relu")
        # 20×50 → 10×50  (downsample frequency only)
        self.enc3 = _conv_block(base_channels * 2, base_channels * 4, stride=(2, 1),
                                norm_type=norm_type, activation="relu")
        # 10×50 → 10×50  (project to latent)
        self.enc4 = _conv_block(base_channels * 4, latent_dim, stride=(1, 1),
                                norm_type=norm_type, activation="relu")

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, 80, 200] Mel spectrogram in [0, 1].
        Returns:
            z: [B, latent_dim, 10, 50] latent representation.
        """
        h = self.enc1(x)    # [B, 32, 40, 100]
        h = self.enc2(h)    # [B, 64, 20, 50]
        h = self.enc3(h)    # [B, 128, 10, 50]
        z = self.enc4(h)    # [B, latent_dim, 10, 50]
        return z


class MelDecoder(nn.Module):
    """Decode latent [B, lat_dim, 10, 50] → Mel [B, 1, 80, 200] with sigmoid."""

    def __init__(self, latent_dim: int = 8, base_channels: int = 32,
                 norm_type: str = "batch"):
        super().__init__()
        # 10×50 → 10×50 (expand from latent)
        self.dec1 = _conv_block(latent_dim, base_channels * 4, stride=(1, 1),
                                norm_type=norm_type, activation="relu")
        # 10×50 → 20×50  (upsample frequency only)
        self.dec2 = _transpose_conv_block(base_channels * 4, base_channels * 2,
                                          stride=(2, 1), padding=(1, 1),
                                          output_padding=(1, 0),
                                          norm_type=norm_type, activation="relu")
        # 20×50 → 40×100
        self.dec3 = _transpose_conv_block(base_channels * 2, base_channels,
                                          stride=(2, 2), padding=(1, 1),
                                          output_padding=(1, 1),
                                          norm_type=norm_type, activation="relu")
        # 40×100 → 80×200
        self.dec4 = _transpose_conv_block(base_channels, base_channels // 2,
                                          stride=(2, 2), padding=(1, 1),
                                          output_padding=(1, 1),
                                          norm_type=norm_type, activation="relu")
        # Final: project to 1 channel with sigmoid
        self.final = nn.Sequential(
            nn.Conv2d(base_channels // 2, 1, kernel_size=(3, 3),
                      stride=(1, 1), padding=(1, 1), bias=True),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [B, latent_dim, 10, 50] latent representation.
        Returns:
            mel: [B, 1, 80, 200] reconstructed Mel in [0, 1].
        """
        h = self.dec1(z)    # [B, 128, 10, 50]
        h = self.dec2(h)    # [B, 64, 20, 50]
        h = self.dec3(h)    # [B, 32, 40, 100]
        h = self.dec4(h)    # [B, 16, 80, 200]
        mel = self.final(h) # [B, 1, 80, 200]
        return mel


class MelAutoencoder(nn.Module):
    """Deterministic convolutional Mel autoencoder.

    Provides three interfaces:
        z = model.encode(mel)
        mel_recon = model.decode(z)
        mel_recon, z = model(mel)
    """

    def __init__(self, latent_dim: int = 8, base_channels: int = 32,
                 norm_type: str = "batch"):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = MelEncoder(latent_dim=latent_dim,
                                  base_channels=base_channels,
                                  norm_type=norm_type)
        self.decoder = MelDecoder(latent_dim=latent_dim,
                                  base_channels=base_channels,
                                  norm_type=norm_type)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode Mel spectrogram to latent.

        Args:
            mel: [B, 1, 80, 200] or [B, 80, 200].
        Returns:
            z: [B, latent_dim, 10, 50].
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)
        return self.encoder(mel)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to Mel spectrogram.

        Args:
            z: [B, latent_dim, 10, 50].
        Returns:
            mel: [B, 1, 80, 200] in [0, 1].
        """
        return self.decoder(z)

    def forward(self, mel: torch.Tensor):
        """Full encode-decode pass.

        Args:
            mel: [B, 1, 80, 200] or [B, 80, 200].
        Returns:
            mel_recon: [B, 1, 80, 200].
            z: [B, latent_dim, 10, 50].
        """
        z = self.encode(mel)
        mel_recon = self.decode(z)
        return mel_recon, z


def time_gradient(x: torch.Tensor) -> torch.Tensor:
    """Compute first-order difference along the time axis (dim=-1).

    Args:
        x: Tensor of shape [..., T].
    Returns:
        diff: Tensor of shape [..., T-1] with x[..., t+1] - x[..., t].
    """
    return x[..., 1:] - x[..., :-1]


def random_boundary_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    num_boundaries: int = 4,
    min_context: int = 5,
) -> torch.Tensor:
    """Penalise gradient discontinuity at random temporal boundaries.

    For each randomly selected boundary position t, we compute the L1
    difference between (a) the gradient around t in the reconstruction
    and (b) the gradient around t in the target.  This encourages the
    AE to preserve natural temporal transitions, which is critical for
    downstream inpainting.

    Args:
        recon:  [B, 1, n_freq, T] reconstructed Mel.
        target: [B, 1, n_freq, T] ground-truth Mel.
        num_boundaries: Number of random boundary positions per sample.
        min_context: Minimum distance from edges and between boundaries.
    Returns:
        Scalar loss averaged over batch, boundaries, and frequency bins.
    """
    B, _, n_freq, T = recon.shape
    if T < 2 * min_context + 2:
        return torch.tensor(0.0, device=recon.device, dtype=recon.dtype)

    grad_recon = time_gradient(recon)  # [B, 1, n_freq, T-1]
    grad_target = time_gradient(target)

    total_loss = torch.tensor(0.0, device=recon.device, dtype=recon.dtype)
    max_pos = T - min_context - 1
    if max_pos < min_context + 1:
        return total_loss

    for b in range(B):
        positions = torch.randint(
            min_context, max_pos, (num_boundaries,),
            device=recon.device,
        )
        for pos in positions:
            left = max(0, int(pos) - 2)
            right = min(T - 1, int(pos) + 2)
            if right - left < 2:
                continue
            total_loss = total_loss + torch.nn.functional.l1_loss(
                grad_recon[b, :, :, left:right],
                grad_target[b, :, :, left:right],
            )

    return total_loss / max(1, B * num_boundaries * n_freq)
