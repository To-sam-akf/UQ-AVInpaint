"""Diffusion noise schedule and utilities for latent-space Mel inpainting.

P3 (K=1 AV Latent Diffusion) uses:
  - 1000 training timesteps with linear beta schedule.
  - Forward: masked noising so known latent is never corrupted.
  - Reverse: DDPM posterior or DDIM, with known-region clamp after every step.
  - mask_z is [B, 1, 10, 50] — downsampled from [B, 1, 80, 200] via avg pool.
"""

import torch
import torch.nn.functional as F


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    """Standard linear variance schedule from DDPM."""
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule from improved DDPM (Nichol & Dhariwal 2021)."""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, max=0.999)


def downsample_mask_2d(mask, target_size=(10, 50)):
    """Downsample binary mask [B, C, H_src, W_src] → [B, C, 10, 50].

    Uses adaptive average pooling followed by binarisation (> 0 → 1.0).
    This is conservative: any latent cell that overlaps a missing pixel
    in the original Mel space is treated as *missing*.
    """
    pooled = F.adaptive_avg_pool2d(mask.float(), target_size)
    return (pooled > 0.0).float()


def downsample_boundary_map(boundary_map, target_size=(10, 50)):
    """Downsample boundary map [B, 2, 80, 200] → [B, 2, 10, 50].

    Simple adaptive average pooling preserves the distance semantics.
    """
    return F.adaptive_avg_pool2d(boundary_map.float(), target_size)


def compose_known_region(z_generated, z_context, mask_z):
    """Enforce that known latent positions are exactly z_context.

    Returns:
        mask_z * z_generated + (1 - mask_z) * z_context
    """
    return mask_z.float() * z_generated + (1.0 - mask_z.float()) * z_context


class DiffusionSchedule:
    """Diffusion noise schedule with masked forward/reverse utilities.

    Precomputes alpha/beta/alpha_bar and provides q_sample,
    masked_q_sample, and DDIM step.
    """

    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02,
                 schedule="linear"):
        self.timesteps = int(timesteps)

        if schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            betas = linear_beta_schedule(timesteps, beta_start, beta_end)

        self.betas = betas.float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0),
                                          value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

        # Posterior variance for DDPM reverse
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )

    def to_device(self, device):
        for attr in (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance",
        ):
            val = getattr(self, attr)
            if val is not None:
                setattr(self, attr, val.to(device))

    def q_sample(self, z_0, t, noise=None):
        """Forward diffusion: z_t = sqrt(a_bar_t) * z_0 + sqrt(1-a_bar_t) * eps.

        Args:
            z_0: [B, C, H, W]
            t: [B] long tensor
            noise: [B, C, H, W] or None
        Returns:
            z_t: [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        sqrt_a = self.sqrt_alphas_cumprod[t].to(z_0.device)[
            :, None, None, None
        ]
        sqrt_1m_a = self.sqrt_one_minus_alphas_cumprod[t].to(z_0.device)[
            :, None, None, None
        ]
        return sqrt_a * z_0 + sqrt_1m_a * noise

    def masked_q_sample(self, z_0, z_context, mask_z, t, noise=None):
        """Masked forward diffusion.

        Only the *missing* region (mask_z == 1) receives noise;
        the known region stays at z_context.

        Args:
            z_0: [B, C, H, W] clean target latent
            z_context: [B, C, H, W] corrupted latent (AE(corrupted Mel))
            mask_z: [B, 1, H, W] binary, 1=missing
            t: [B] long tensor
            noise: [B, C, H, W] or None
        Returns:
            z_t: [B, C, H, W] masked noised latent
        """
        if noise is None:
            noise = torch.randn_like(z_0)
        z_t_noised = self.q_sample(z_0, t, noise)
        return compose_known_region(z_t_noised, z_context, mask_z)

    def compute_previous_z(self, z_t, epsilon_pred, t,
                           clamp_mask=None, z_context=None):
        """DDPM posterior: compute z_{t-1}.

        Args:
            z_t: [B, C, H, W]
            epsilon_pred: [B, C, H, W] model output
            t: [B] long tensor
            clamp_mask: [B, 1, H, W] optional
            z_context: [B, C, H, W] optional
        Returns:
            z_prev: [B, C, H, W]
        """
        device = z_t.device
        beta_t = self.betas[t].to(device)[:, None, None, None]
        alpha_t = self.alphas[t].to(device)[:, None, None, None]
        alpha_bar_t = self.alphas_cumprod[t].to(device)[:, None, None, None]

        # Predicted z_0
        z_0_pred = (
            z_t - torch.sqrt(1.0 - alpha_bar_t) * epsilon_pred
        ) / torch.sqrt(alpha_bar_t)
        z_0_pred = torch.clamp(z_0_pred, -4.0, 4.0)

        # Posterior mean coefficients
        sqrt_a_prev = torch.sqrt(
            self.alphas_cumprod_prev[t].to(device)
        )[:, None, None, None]
        coef1 = beta_t * sqrt_a_prev / (1.0 - alpha_bar_t)
        coef2 = (
            (1.0 - self.alphas_cumprod_prev[t].to(device))[:, None, None, None]
            * torch.sqrt(alpha_t)
            / (1.0 - alpha_bar_t)
        )
        z_prev_mean = coef1 * z_0_pred + coef2 * z_t

        # Add noise (zero at t=0 where posterior_variance=0)
        noise = torch.randn_like(z_t)
        post_var = self.posterior_variance[t].to(device)[:, None, None, None]
        z_prev = z_prev_mean + torch.sqrt(post_var) * noise

        if clamp_mask is not None and z_context is not None:
            z_prev = compose_known_region(z_prev, z_context, clamp_mask)

        return z_prev

    def ddim_step(self, z_t, epsilon_pred, t, t_next, eta=0.0,
                  clamp_mask=None, z_context=None):
        """Single DDIM step (Song et al. 2021).

        Args:
            z_t: [B, C, H, W]
            epsilon_pred: [B, C, H, W]
            t: [B] current timestep
            t_next: [B] next timestep
            eta: 0=deterministic DDIM, 1=stochastic DDPM
            clamp_mask: [B, 1, H, W] or None
            z_context: [B, C, H, W] or None
        Returns:
            z_next: [B, C, H, W]
        """
        device = z_t.device
        a_bar_t = self.alphas_cumprod[t].to(device)[:, None, None, None]
        a_bar_next = self.alphas_cumprod[t_next].to(device)[
            :, None, None, None
        ]

        # Predicted z_0
        z_0_pred = (
            z_t - torch.sqrt(1.0 - a_bar_t) * epsilon_pred
        ) / torch.sqrt(a_bar_t)
        z_0_pred = torch.clamp(z_0_pred, -4.0, 4.0)

        # DDIM update
        sigma = eta * torch.sqrt(
            (1.0 - a_bar_next) / (1.0 - a_bar_t)
            * (1.0 - a_bar_t / a_bar_next)
        )
        pred_dir = torch.sqrt(1.0 - a_bar_next - sigma ** 2) * epsilon_pred
        z_next = torch.sqrt(a_bar_next) * z_0_pred + pred_dir

        if eta > 0:
            z_next = z_next + sigma * torch.randn_like(z_t)

        if clamp_mask is not None and z_context is not None:
            z_next = compose_known_region(z_next, z_context, clamp_mask)

        return z_next

    def get_ddim_timesteps(self, inference_steps=50):
        """Return linearly-spaced timesteps for DDIM sampling.

        Args:
            inference_steps: number of DDIM steps (e.g. 50).
        Returns:
            timesteps: [inference_steps] descending from T-1 to 0.
        """
        step = self.timesteps // inference_steps
        order = torch.arange(
            self.timesteps - 1, -1, -step, dtype=torch.long
        )
        return order[:inference_steps]


def compute_diffusion_loss(epsilon_pred, epsilon, mask_z):
    """Masked diffusion loss — computed ONLY over missing latent region.

    Args:
        epsilon_pred: [B, C, H, W]
        epsilon: [B, C, H, W]
        mask_z: [B, 1, H, W]
    Returns:
        scalar loss
    """
    diff = (epsilon_pred - epsilon) ** 2
    masked_diff = diff * mask_z.float()
    return masked_diff.sum() / mask_z.float().sum().clamp(min=1.0)
