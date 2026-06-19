"""Mel Autoencoder training wrapper.

Follows the same pattern as VIAIAModel/VIAIAVModel but specialised for
deterministic convolutional Mel AE training without mask injection.

Training strategy:
  Phase 1 — L1 only
  Phase 2 — L1 + time-gradient + random-boundary (enabled after warmup steps)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.uq.mel_autoencoder import (
    MelAutoencoder,
    time_gradient,
    random_boundary_loss,
)


class MelAEModel:
    """Training wrapper for the deterministic Mel convolutional autoencoder."""

    def __init__(self, hparams, device=None):
        self.hparams = hparams
        self.device = device if device is not None else torch.device("cpu")

        latent_dim = int(getattr(hparams, "ae_latent_dim", 8))
        base_channels = int(getattr(hparams, "ae_base_channels", 32))
        norm_type = getattr(hparams, "ae_norm_type", "batch")

        self.net = MelAutoencoder(
            latent_dim=latent_dim,
            base_channels=base_channels,
            norm_type=norm_type,
        ).to(self.device)

        # Loss weights
        self.lambda_l1 = float(getattr(hparams, "ae_lambda_l1", 1.0))
        self.lambda_grad = float(getattr(hparams, "ae_lambda_grad", 0.1))
        self.lambda_boundary = float(getattr(hparams, "ae_lambda_boundary", 0.05))
        # Steps before enabling gradient + boundary losses
        self.warmup_steps = int(getattr(hparams, "ae_warmup_steps", 2000))

        self.criterion_l1 = nn.L1Loss()

        lr = float(getattr(hparams, "ae_lr", getattr(hparams, "lr", 1e-4)))
        beta1 = float(getattr(hparams, "beta1", 0.5))
        beta2 = float(getattr(hparams, "beta2", 0.999))
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, betas=(beta1, beta2),
        )
        self.current_lr = lr

        # Latent normalisation statistics (updated at end of training)
        self.latent_mean = None  # shape [latent_dim]
        self.latent_std = None   # shape [latent_dim]

        # Loss items
        self.loss_total_item = 0.0
        self.loss_l1_item = 0.0
        self.loss_grad_item = 0.0
        self.loss_boundary_item = 0.0

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        """Encode Mel to latent (deterministic)."""
        return self.net.encode(mel)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to Mel."""
        return self.net.decode(z)

    def set_input(self, mel: torch.Tensor):
        """Accept clean Mel spectrogram [B, 1, 80, 200]."""
        self.mel_target = mel.float().to(self.device)

    def _forward(self):
        self.mel_recon, self.z = self.net(self.mel_target)
        return self.mel_recon, self.z

    def _compute_loss(self, global_step: int):
        # L1 reconstruction
        self.loss_l1 = self.criterion_l1(self.mel_recon, self.mel_target)

        # Time-gradient L1
        grad_recon = time_gradient(self.mel_recon)
        grad_target = time_gradient(self.mel_target)
        self.loss_grad = F.l1_loss(grad_recon, grad_target)

        # Random boundary loss
        self.loss_boundary = random_boundary_loss(
            self.mel_recon, self.mel_target,
        )

        # Phase-aware weighting
        if global_step < self.warmup_steps:
            self.loss_total = self.lambda_l1 * self.loss_l1
        else:
            self.loss_total = (
                self.lambda_l1 * self.loss_l1
                + self.lambda_grad * self.loss_grad
                + self.lambda_boundary * self.loss_boundary
            )

    def optimize_parameters(self, global_step: int):
        self.net.train()
        self._forward()
        self._compute_loss(global_step)
        self.optimizer.zero_grad()
        self.loss_total.backward()
        self.optimizer.step()
        self.current_lr = self.optimizer.param_groups[0]["lr"]

    def test(self, global_step: int = 0):
        self.net.eval()
        with torch.no_grad():
            self._forward()
            self._compute_loss(global_step)

    def get_loss_items(self):
        self.loss_total_item = float(self.loss_total.detach().cpu().item())
        self.loss_l1_item = float(self.loss_l1.detach().cpu().item())
        self.loss_grad_item = float(self.loss_grad.detach().cpu().item())
        self.loss_boundary_item = float(self.loss_boundary.detach().cpu().item())

    def get_current_errors(self):
        return {
            "loss_total": self.loss_total_item,
            "loss_l1": self.loss_l1_item,
            "loss_grad": self.loss_grad_item,
            "loss_boundary": self.loss_boundary_item,
            "lr": self.current_lr,
        }

    def TF_writer(self, writer, step: int, prefix: str = "train"):
        if writer is None:
            return
        writer.add_scalar(f"{prefix}/loss_total", self.loss_total_item, step)
        writer.add_scalar(f"{prefix}/loss_l1", self.loss_l1_item, step)
        writer.add_scalar(f"{prefix}/loss_grad", self.loss_grad_item, step)
        writer.add_scalar(f"{prefix}/loss_boundary", self.loss_boundary_item, step)
        writer.add_scalar(f"{prefix}/lr", self.current_lr, step)
        writer.add_scalar(f"{prefix}/warmup", float(step < self.warmup_steps), step)

    # -----------------------------------------------------------------
    # Checkpoint save / load
    # -----------------------------------------------------------------
    def save_checkpoint(self, global_step: int, global_epoch: int,
                        checkpoint_dir: str, latent_stats: dict = None):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"MelAE_checkpoint_step{global_step:09d}.pth.tar",
        )
        checkpoint = {
            "encoder": self.net.encoder.state_dict(),
            "decoder": self.net.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict()
            if getattr(self.hparams, "save_optimizer_state", True)
            else None,
            "global_step": global_step,
            "global_epoch": global_epoch,
            "hparams": {
                "latent_dim": self.net.latent_dim,
                "norm_type": getattr(self.hparams, "ae_norm_type", "batch"),
            },
        }
        if latent_stats is not None:
            checkpoint["latent_mean"] = latent_stats.get("mean")
            checkpoint["latent_std"] = latent_stats.get("std")
        torch.save(checkpoint, checkpoint_path)
        print("Saved MelAE checkpoint:", checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str, reset_optimizer: bool = False):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.net.encoder.load_state_dict(checkpoint["encoder"])
        self.net.decoder.load_state_dict(checkpoint["decoder"])
        if not reset_optimizer and checkpoint.get("optimizer") is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.latent_mean = checkpoint.get("latent_mean")
        self.latent_std = checkpoint.get("latent_std")
        global_step = int(checkpoint.get("global_step", 0))
        global_epoch = int(checkpoint.get("global_epoch", 0))
        return global_step, global_epoch

    def compute_latent_stats(self, data_loader, max_batches: int = None):
        """Compute mean and std of latents over (a subset of) the dataset.

        Must be called in eval mode with torch.no_grad().
        """
        self.net.eval()
        sum_z = 0.0
        sum_z2 = 0.0
        count = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                mel = batch["mel_target"].float().to(self.device)
                z = self.net.encode(mel)  # [B, C, H, W]
                # Average over spatial dims, accumulate over batch
                z_flat = z.mean(dim=(2, 3))  # [B, C]
                sum_z += z_flat.sum(dim=0)
                sum_z2 += (z_flat ** 2).sum(dim=0)
                count += z.size(0)
                if max_batches is not None and batch_idx + 1 >= max_batches:
                    break

        mean = sum_z / count
        std = torch.sqrt(sum_z2 / count - mean ** 2).clamp(min=1e-6)
        self.latent_mean = mean.cpu()
        self.latent_std = std.cpu()
        return {"mean": self.latent_mean, "std": self.latent_std}
