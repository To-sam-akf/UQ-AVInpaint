"""Unit tests for P3 K=1 AV Latent Diffusion.

Covers:
  - Diffusion schedule (beta, alpha, alpha_bar, q_sample)
  - mask_z downsampling contract
  - Masked noising (known region preserved)
  - Known-region clamp after DDPM/DDIM steps
  - U-Net shape contract
  - Video encoder (P3 minimal) shape contract
  - End-to-end forward/backward smoke test (1 batch)
  - Fixed seed reproducibility
  - Compose known Mel region error = 0
  - Known latent unchanged after single denoising step
"""

import importlib
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Data_loaders.uq_av_loader import create_uq_av_dataloader
from networks.uq.diffusion_schedule import (
    DiffusionSchedule,
    linear_beta_schedule,
    cosine_beta_schedule,
    downsample_mask_2d,
    downsample_boundary_map,
    compose_known_region,
    compute_diffusion_loss,
)
from networks.uq.latent_diffusion_unet import (
    LatentDiffusionUNet,
    sinusoidal_embedding,
    TimeEmbedding,
)
from networks.uq.video_evidence_encoder import (
    VideoEvidenceEncoderP3,
    VideoConditionDummy,
)
from networks.uq.mel_autoencoder import MelAutoencoder
from utils.viai_a_metrics import compute_inpainting_sample_metrics


# ===================================================================
# Fixtures
# ===================================================================
@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def mel_batch(batch_size):
    """Random Mel spectrograms [B, 1, 80, 200] in [0, 1]."""
    return torch.rand(batch_size, 1, 80, 200)


@pytest.fixture
def missing_mask(batch_size):
    """Binary mask: one contiguous gap per sample. 1 = missing."""
    mask = torch.zeros(batch_size, 1, 80, 200)
    for b in range(batch_size):
        start = 50 + b * 10
        end = start + 30
        mask[b, :, :, start:end] = 1.0
    return mask


@pytest.fixture
def boundary_map(batch_size):
    """Boundary distance map [B, 2, 80, 200]."""
    bmap = torch.zeros(batch_size, 2, 80, 200)
    for b in range(batch_size):
        start = 50 + b * 10
        end = start + 30
        positions = torch.arange(200, dtype=torch.float32)
        bmap[b, 0, :, :] = torch.abs(positions[None, :] - start) / 200.0
        bmap[b, 1, :, :] = torch.abs(positions[None, :] - (end - 1)) / 200.0
    return bmap


@pytest.fixture
def video_batch(batch_size):
    """Random video frames [B, 50, 3, 64, 64]."""
    return torch.randn(batch_size, 50, 3, 64, 64)


@pytest.fixture
def flow_batch(batch_size):
    """Random flow frames [B, 50, 2, 64, 64]."""
    return torch.randn(batch_size, 50, 2, 64, 64)


@pytest.fixture
def diffusion_schedule():
    return DiffusionSchedule(timesteps=1000)


# ===================================================================
# Diffusion schedule tests
# ===================================================================
class TestDiffusionSchedule:
    """Verify beta/alpha/alpha_bar schedules and q_sample."""

    def test_beta_shape_and_range(self):
        betas = linear_beta_schedule(1000)
        assert betas.shape == (1000,)
        assert torch.all(betas > 0) and torch.all(betas < 1)
        assert betas[0] < betas[-1], "betas should be increasing"

    def test_alpha_bar_decreasing(self, diffusion_schedule):
        ab = diffusion_schedule.alphas_cumprod
        assert torch.all(ab[1:] <= ab[:-1]), "alpha_bar must be non-increasing"
        assert ab[-1] < 0.01, f"alpha_bar at T should be near 0, got {ab[-1]}"

    def test_q_sample_shape(self, diffusion_schedule, batch_size, device):
        diffusion_schedule.to_device(device)
        z_0 = torch.randn(batch_size, 8, 10, 50, device=device)
        t = torch.randint(0, 1000, (batch_size,), device=device).long()
        z_t = diffusion_schedule.q_sample(z_0, t)
        assert z_t.shape == z_0.shape

    def test_q_sample_at_t0_has_minimal_noise(self, diffusion_schedule, device):
        """At t=0, beta=beta_start(~1e-4), so noise contribution is tiny."""
        diffusion_schedule.to_device(device)
        z_0 = torch.randn(2, 8, 10, 50, device=device)
        t = torch.zeros(2, device=device).long()
        z_t = diffusion_schedule.q_sample(z_0, t)
        # With beta=1e-4, the correlation between z_t and z_0 should be high
        corr = torch.nn.functional.cosine_similarity(
            z_t.flatten(1), z_0.flatten(1), dim=-1,
        )
        assert (corr > 0.99).all(), \
            f"At t=0, z_t should be nearly z_0, but cosine sim={corr}"

    def test_q_sample_at_tmax_is_noise(self, diffusion_schedule, device):
        """At t=T-1, z_t should be nearly pure noise."""
        diffusion_schedule.to_device(device)
        z_0 = torch.ones(2, 8, 10, 50, device=device)
        t = torch.full((2,), diffusion_schedule.timesteps - 1,
                       device=device).long()
        z_t = diffusion_schedule.q_sample(z_0, t)
        # Should be close to N(0, 1) distribution
        assert z_t.abs().mean() < 2.0  # sanity check

    def test_cosine_schedule(self):
        sched = DiffusionSchedule(timesteps=100, schedule="cosine")
        assert sched.alphas_cumprod[-1] < 0.01
        # Cosine schedule should have near-linear beta start
        assert sched.betas[0] < sched.betas[-1]

    def test_ddim_timesteps_include_zero_and_requested_transitions(self):
        sched = DiffusionSchedule(timesteps=1000)
        timesteps = sched.get_ddim_timesteps(inference_steps=50)
        assert timesteps[0].item() == 999
        assert timesteps[-1].item() == 0
        assert len(timesteps) - 1 == 50


# ===================================================================
# Mask downsampling tests
# ===================================================================
class TestMaskDownsampling:
    """Verify mask_z downsampling contract."""

    def test_downsample_shape(self, missing_mask):
        mask_z = downsample_mask_2d(missing_mask)
        assert mask_z.shape == (4, 1, 10, 50)

    def test_downsample_binary(self, missing_mask):
        mask_z = downsample_mask_2d(missing_mask)
        unique_vals = mask_z.unique().tolist()
        assert all(v in (0.0, 1.0) for v in unique_vals), \
            f"mask_z must be binary, got {unique_vals}"

    def test_known_pixels_map_to_zero(self, missing_mask):
        """Known region (mask=0) must map to 0 in latent space."""
        # Fill only the first 10 Mel freq bins and first 50 time frames
        # with 1s — this covers part of the 80×200 grid
        test_mask = torch.zeros(1, 1, 80, 200)
        test_mask[:, :, :4, :4] = 1.0  # small missing region
        mask_z = downsample_mask_2d(test_mask)
        # Some latent cells should be 0 (the fully-known cells)
        assert (mask_z == 0).sum() > 0, \
            "mask_z should contain zeros for fully-known latent cells"

    def test_full_missing_region(self):
        """If the entire Mel is missing, mask_z should be all 1."""
        full_mask = torch.ones(2, 1, 80, 200)
        mask_z = downsample_mask_2d(full_mask)
        assert mask_z.sum() == mask_z.numel()

    def test_boundary_map_downsample_shape(self, boundary_map):
        bmap_z = downsample_boundary_map(boundary_map)
        assert bmap_z.shape == (4, 2, 10, 50)

    def test_downsample_deterministic(self, missing_mask):
        z1 = downsample_mask_2d(missing_mask)
        z2 = downsample_mask_2d(missing_mask)
        assert torch.equal(z1, z2)


# ===================================================================
# Masked noising tests
# ===================================================================
class TestMaskedNoising:
    """Verify masked_q_sample and known-region clamp."""

    def test_known_region_preserved(self, diffusion_schedule, device):
        """Known latent values must be exactly z_context after masking."""
        diffusion_schedule.to_device(device)
        z_0 = torch.randn(2, 8, 10, 50, device=device)
        z_context = torch.randn(2, 8, 10, 50, device=device)
        mask_z = torch.zeros(2, 1, 10, 50, device=device)
        mask_z[:, :, :, 10:] = 1.0  # first 10 time steps are known

        t = torch.randint(0, 1000, (2,), device=device).long()
        z_t = diffusion_schedule.masked_q_sample(z_0, z_context, mask_z, t)

        # Known region must match z_context exactly
        known_diff = (
            (z_t - z_context) * (1.0 - mask_z)
        ).abs().max()
        assert known_diff.item() == 0.0, \
            f"Known region changed! max diff = {known_diff.item()}"

    def test_missing_region_is_noised(self, diffusion_schedule, device):
        """Missing region should differ from z_0 when t > 0."""
        diffusion_schedule.to_device(device)
        z_0 = torch.ones(2, 8, 10, 50, device=device)
        z_context = torch.zeros(2, 8, 10, 50, device=device)
        mask_z = torch.ones(2, 1, 10, 50, device=device)

        t = torch.full((2,), 500, device=device).long()
        z_t = diffusion_schedule.masked_q_sample(z_0, z_context, mask_z, t)
        # Missing region should differ from z_0
        assert not torch.allclose(z_t, z_0, atol=0.01)

    def test_compose_known_region(self, device):
        z_gen = torch.randn(2, 8, 10, 50, device=device)
        z_ctx = torch.ones(2, 8, 10, 50, device=device)
        mask = torch.zeros(2, 1, 10, 50, device=device)
        mask[:, :, :, :5] = 1.0  # time 0-4 is missing

        z_out = compose_known_region(z_gen, z_ctx, mask)
        # Known region (time 5-49) must equal z_ctx (= 1.0)
        known = z_out[:, :, :, 5:]
        assert torch.allclose(known, torch.ones_like(known))
        # Missing region (time 0-4) must equal z_gen
        missing = z_out[:, :, :, :5]
        assert torch.allclose(missing, z_gen[:, :, :, :5])

    def test_diffusion_loss_only_in_missing_region(self, device):
        epsilon_pred = torch.ones(2, 8, 10, 50, device=device)
        epsilon = torch.zeros(2, 8, 10, 50, device=device)
        mask_z = torch.zeros(2, 1, 10, 50, device=device)
        mask_z[:, :, :, :5] = 1.0  # only first 5 time steps are missing

        loss = compute_diffusion_loss(epsilon_pred, epsilon, mask_z)
        # Per-element squared error = 1.0, averaged over 8 channels × 10 freq × 5 time
        # mask_z broadcast gives loss over all 8 channels in those 5 time steps
        # = (8*10*5*2 * 1) / (2*1*10*5) = 8.0
        assert 7.5 < loss.item() < 8.5, \
            f"Expected ~8.0, got {loss.item()}"


# ===================================================================
# Known-region clamp after denoising step tests
# ===================================================================
class TestKnownRegionClamp:
    """Verify known latent unchanged after DDPM/DDIM steps."""

    def test_ddpm_step_clamp(self, diffusion_schedule, device):
        diffusion_schedule.to_device(device)
        z_0 = torch.randn(2, 8, 10, 50, device=device)
        z_context = torch.randn(2, 8, 10, 50, device=device)
        mask_z = torch.zeros(2, 1, 10, 50, device=device)
        mask_z[:, :, :, 20:] = 1.0  # first 20 known

        t_val = 500
        t = torch.full((2,), t_val, device=device).long()

        # Forward
        z_t = diffusion_schedule.masked_q_sample(z_0, z_context, mask_z, t)

        # Predict epsilon (random)
        epsilon_pred = torch.randn_like(z_t)

        # One DDPM step
        z_prev = diffusion_schedule.compute_previous_z(
            z_t, epsilon_pred, t,
            clamp_mask=mask_z, z_context=z_context,
        )

        # Known region must be exactly z_context
        known_error = (
            (z_prev - z_context) * (1.0 - mask_z)
        ).abs().max()
        assert known_error.item() == 0.0, \
            f"Known region not preserved after DDPM step: {known_error.item()}"

    def test_ddim_step_clamp(self, diffusion_schedule, device):
        diffusion_schedule.to_device(device)
        z_context = torch.randn(2, 8, 10, 50, device=device)
        mask_z = torch.zeros(2, 1, 10, 50, device=device)
        mask_z[:, :, :, :10] = 1.0  # first 10 are missing

        # Start from noise in missing region
        z_t = torch.randn(2, 8, 10, 50, device=device)
        z_t = compose_known_region(z_t, z_context, mask_z)

        t_val = 800
        t_next_val = 750
        t = torch.full((2,), t_val, device=device).long()
        t_next = torch.full((2,), t_next_val, device=device).long()

        epsilon_pred = torch.randn_like(z_t)
        z_next = diffusion_schedule.ddim_step(
            z_t, epsilon_pred, t, t_next, eta=0.0,
            clamp_mask=mask_z, z_context=z_context,
        )

        known_error = (
            (z_next - z_context) * (1.0 - mask_z)
        ).abs().max()
        assert known_error.item() == 0.0, \
            f"Known region not preserved after DDIM step: {known_error.item()}"

    def test_configurable_x0_clip_preserves_known_region(self,
                                                         diffusion_schedule,
                                                         device):
        diffusion_schedule.to_device(device)
        z_context = torch.randn(2, 8, 10, 50, device=device)
        mask_z = torch.zeros(2, 1, 10, 50, device=device)
        mask_z[:, :, :, :12] = 1.0
        z_t = compose_known_region(
            torch.randn_like(z_context) * 10.0,
            z_context,
            mask_z,
        )
        t = torch.full((2,), 800, device=device).long()
        t_next = torch.full((2,), 700, device=device).long()
        epsilon_pred = torch.randn_like(z_t) * 10.0

        for clip_value in (0.0, 1.0, 4.0):
            z_next = diffusion_schedule.ddim_step(
                z_t, epsilon_pred, t, t_next, eta=0.0,
                clamp_mask=mask_z, z_context=z_context,
                x0_clip_value=clip_value,
            )
            known_error = (
                (z_next - z_context) * (1.0 - mask_z)
            ).abs().max()
            assert known_error.item() == 0.0


# ===================================================================
# U-Net shape tests
# ===================================================================
class TestLatentDiffusionUNet:
    """Verify U-Net shape contract and forward pass."""

    def test_unet_input_output_shape(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.randint(0, 1000, (B,), device=device).long()
        video_tokens = torch.randn(B, 50, 128, device=device)

        out = unet(x, t, video_tokens=video_tokens)
        assert out.shape == (B, 8, 10, 50), \
            f"Expected (B, 8, 10, 50), got {out.shape}"

    def test_frame_positional_embedding_shape(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        tokens = torch.randn(2, 50, 128, device=device)

        positioned = unet.apply_frame_positional_embedding(tokens)

        assert unet.frame_pos_embed.shape == (1, 50, 128)
        assert positioned.shape == tokens.shape
        assert not torch.allclose(positioned, tokens)

    def test_zero_tokens_do_not_receive_positional_embedding(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        tokens = torch.zeros(2, 50, 128, device=device)

        positioned = unet.apply_frame_positional_embedding(tokens)

        assert torch.equal(positioned, tokens)

    def test_multi_level_attention_shape_and_count(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.randint(0, 1000, (B,), device=device).long()
        video_tokens = torch.randn(B, 50, 128, device=device)

        out = unet(x, t, video_tokens=video_tokens)

        expected_attn_layers = (
            len(unet.encoder_attns) + 1 + len(unet.decoder_attns)
        )
        assert expected_attn_layers == 8
        assert out.shape == (B, 8, 10, 50)

    def test_video_diagnostics_are_recorded(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.randint(0, 1000, (B,), device=device).long()
        video_tokens = torch.randn(B, 50, 128, device=device)

        _ = unet(x, t, video_tokens=video_tokens)

        assert torch.isfinite(unet.video_gate_mean)
        assert torch.isfinite(unet.video_attn_norm)
        assert torch.isfinite(unet.video_token_norm)
        assert unet.video_attn_norm.item() > 0.0
        assert unet.video_token_norm.item() > 0.0

    def test_unet_without_video(self, device):
        """U-Net should work without video tokens (audio-only mode)."""
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.zeros(B, device=device).long()

        out = unet(x, t, video_tokens=None)
        assert out.shape == (B, 8, 10, 50)
        assert unet.video_gate_mean.item() == 0.0
        assert unet.video_attn_norm.item() == 0.0
        assert unet.video_token_norm.item() == 0.0

    def test_unet_output_changes_for_wrong_and_zero_video(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        unet.eval()
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.full((B,), 42, device=device).long()
        original = torch.randn(B, 50, 128, device=device)
        wrong = original.roll(shifts=1, dims=1)
        zero = torch.zeros_like(original)

        with torch.no_grad():
            out_original = unet(x, t, video_tokens=original)
            out_wrong = unet(x, t, video_tokens=wrong)
            out_zero = unet(x, t, video_tokens=zero)

        wrong_diff = (out_original - out_wrong).abs().mean().item()
        zero_diff = (out_original - out_zero).abs().mean().item()
        assert wrong_diff > 1e-7
        assert zero_diff > 1e-7

    def test_unet_deterministic(self, device):
        """Same input should produce same output."""
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        unet.eval()
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.full((B,), 42, device=device).long()
        v = torch.randn(B, 50, 128, device=device)

        with torch.no_grad():
            out1 = unet(x, t, video_tokens=v)
            out2 = unet(x, t, video_tokens=v)
        assert torch.allclose(out1, out2, atol=1e-5)

    def test_unet_different_t_gives_different_output(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
            time_emb_dim=128, video_dim=128,
        ).to(device)
        unet.eval()
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t1 = torch.zeros(B, device=device).long()
        t2 = torch.full((B,), 500, device=device).long()

        with torch.no_grad():
            out1 = unet(x, t1, video_tokens=None)
            out2 = unet(x, t2, video_tokens=None)
        assert not torch.allclose(out1, out2, atol=1e-4), \
            "U-Net output should depend on timestep"

    def test_sinusoidal_embedding_shape(self):
        t = torch.tensor([0, 10, 100, 500])
        emb = sinusoidal_embedding(t, 256)
        assert emb.shape == (4, 256)

    def test_time_embedding_module(self):
        te = TimeEmbedding(dim=256, hidden_dim=512)
        t = torch.randint(0, 1000, (4,))
        emb = te(t)
        assert emb.shape == (4, 256)

    def test_unet_no_nan_inf(self, device):
        unet = LatentDiffusionUNet(
            in_channels=19, out_channels=8, base_channels=32,
        ).to(device)
        B = 2
        x = torch.randn(B, 19, 10, 50, device=device)
        t = torch.randint(0, 1000, (B,), device=device).long()
        out = unet(x, t, video_tokens=None)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ===================================================================
# Video encoder (P3 minimal) tests
# ===================================================================
class TestVideoEncoderP3:
    """Verify P3 video encoder shape contract and determinism."""

    def test_output_shapes(self, video_batch, flow_batch, device):
        encoder = VideoEvidenceEncoderP3(
            video_dim=128, image_size=64,
            rgb_feature_dim=64, flow_feature_dim=64,
        ).to(device)
        video = video_batch.to(device)
        flow = flow_batch.to(device)

        out = encoder(video, flow)
        B = video_batch.size(0)
        F = video_batch.size(1)

        assert out["rgb_tokens"].shape == (B, F, 128)
        assert out["flow_tokens"].shape == (B, F, 128)
        assert out["video_tokens"].shape == (B, F, 128)

    def test_different_videos_different_tokens(self, device):
        encoder = VideoEvidenceEncoderP3(
            video_dim=128, image_size=64,
        ).to(device)
        encoder.eval()
        v1 = torch.randn(2, 50, 3, 64, 64, device=device)
        v2 = torch.randn(2, 50, 3, 64, 64, device=device)
        f = torch.randn(2, 50, 2, 64, 64, device=device)

        with torch.no_grad():
            out1 = encoder(v1, f)["video_tokens"]
            out2 = encoder(v2, f)["video_tokens"]
        assert not torch.allclose(out1, out2, atol=1e-3)

    def test_deterministic(self, video_batch, flow_batch, device):
        encoder = VideoEvidenceEncoderP3(
            video_dim=128, image_size=64,
        ).to(device)
        encoder.eval()
        v = video_batch.to(device)
        f = flow_batch.to(device)

        with torch.no_grad():
            o1 = encoder(v, f)["video_tokens"]
            o2 = encoder(v, f)["video_tokens"]
        assert torch.allclose(o1, o2, atol=1e-5)

    def test_dummy_encoder(self, video_batch, flow_batch, device):
        encoder = VideoConditionDummy(video_dim=128).to(device)
        v = video_batch.to(device)
        f = flow_batch.to(device)
        out = encoder(v, f)
        B, F = video_batch.shape[:2]
        assert out["video_tokens"].shape == (B, F, 128)
        # Dummy should return zeros
        assert (out["video_tokens"] == 0).all()

    def test_no_nan_inf(self, video_batch, flow_batch, device):
        encoder = VideoEvidenceEncoderP3(
            video_dim=128, image_size=64,
        ).to(device)
        v = video_batch.to(device)
        f = flow_batch.to(device)
        out = encoder(v, f)
        for key in ("rgb_tokens", "flow_tokens", "video_tokens"):
            assert not torch.isnan(out[key]).any(), f"{key} has NaN"
            assert not torch.isinf(out[key]).any(), f"{key} has Inf"


# ===================================================================
# Model wrapper — use importlib to avoid Models/__init__.py chaining
# into wavenet/Config which conflicts with pytest args
# ===================================================================
_uq_av_spec = importlib.util.spec_from_file_location(
    "UQ_AV_Diffusion",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Models", "UQ_AV_Diffusion.py"),
)
_uq_av_mod = importlib.util.module_from_spec(_uq_av_spec)
_uq_av_spec.loader.exec_module(_uq_av_mod)
UQAVDiffusionModel = _uq_av_mod.UQAVDiffusionModel


# ===================================================================
# End-to-end forward/backward smoke test
# ===================================================================
class TestEndToEndSmoke:
    """Verify the full P3 pipeline can complete one training step."""

    def _make_hparams(self, tmpdir):
        """Create a minimal hparams object for testing."""
        from types import SimpleNamespace
        return SimpleNamespace(
            ae_latent_dim=4,
            ae_base_channels=16,
            ae_norm_type="batch",
            uq_video_dim=64,
            uq_unet_base_channels=16,
            uq_time_emb_dim=64,
            uq_lambda_boundary=0.1,
            uq_lambda_sync=0.0,
            uq_attn_heads=2,
            uq_no_video=False,
            uq_lr=1e-4,
            lr=1e-4,
            beta1=0.5,
            beta2=0.999,
            save_optimizer_state=True,
            image_size=64,
            diff_timesteps=100,
            diff_beta_start=1e-4,
            diff_beta_end=0.02,
            diff_schedule="linear",
            name="UQ-AV-test",
            uq_inference_steps=5,
            uq_grad_clip=1.0,
            checkpoint_dir=str(tmpdir),
        )

    def _make_batch(self, batch_size=2):
        batch = {
            "sample_id": [f"s{index}" for index in range(batch_size)],
            "mel_target": torch.rand(batch_size, 1, 80, 200),
            "mel_corrupted": torch.rand(batch_size, 1, 80, 200),
            "missing_mask": torch.zeros(batch_size, 1, 80, 200),
            "boundary_map": torch.zeros(batch_size, 2, 80, 200),
            "video": torch.randn(batch_size, 50, 3, 64, 64),
            "flow": torch.randn(batch_size, 50, 2, 64, 64),
            "audio_target": torch.randn(batch_size, 64000),
            "mask_spec": [None] * batch_size,
            "video_condition": ["original"] * batch_size,
            "video_degradation": [{} for _ in range(batch_size)],
        }
        batch["missing_mask"][:, :, :, 60:80] = 1.0
        return batch

    def test_one_training_step(self, device, tmp_path):
        """Run one optimizer step and verify losses are finite."""
        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)

        B = 2
        batch = {
            "sample_id": ["s1", "s2"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None, None],
            "video_condition": ["original", "original"],
            "video_degradation": [{}, {}],
        }
        # Inject a missing gap
        for b in range(B):
            start = 60 + b * 5
            batch["missing_mask"][b, :, :, start:start + 20] = 1.0

        model.set_input(batch)
        model.optimize_parameters(global_step=0)
        model.get_loss_items()
        errors = model.get_current_errors()

        assert model.loss_total_item is not None
        assert not torch.isnan(torch.tensor(model.loss_total_item))
        assert not torch.isinf(torch.tensor(model.loss_total_item))
        assert model.loss_diff_item > 0
        for key in ("video_gate_mean", "video_attn_norm", "video_token_norm"):
            assert key in errors
            assert math.isfinite(errors[key])

    def test_condition_loss_logging_is_finite(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)

        B = 2
        batch = {
            "sample_id": ["s1", "s2"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None, None],
            "video_condition": ["original", "wrong_video"],
            "conditioning_mode": ["drop_video", "wrong_video"],
            "video_degradation": [{}, {}],
        }
        batch["missing_mask"][:, :, :, 60:80] = 1.0

        model.set_input(batch)
        model.optimize_parameters(global_step=0)
        model.get_loss_items()

        assert model.condition_counts["drop_video"] == 1
        assert model.condition_counts["wrong_video"] == 1
        for value in model.condition_loss_items.values():
            assert math.isfinite(value)
            assert value >= 0.0

    def test_video_margin_loss_is_logged_and_in_total(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        hparams.uq_lambda_video_margin = 0.5
        hparams.uq_video_margin = 0.02
        hparams.uq_video_margin_negative = "batch_shuffle"
        model = UQAVDiffusionModel(hparams, device=device)

        B = 2
        video_original = torch.randn(B, 50, 3, 64, 64)
        flow_original = torch.randn(B, 50, 2, 64, 64)
        batch = {
            "sample_id": ["s1", "s2"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": video_original.roll(shifts=1, dims=0),
            "flow": flow_original.roll(shifts=1, dims=0),
            "video_original": video_original,
            "flow_original": flow_original,
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None, None],
            "video_condition": ["wrong_video", "wrong_video"],
            "conditioning_mode": ["wrong_video", "wrong_video"],
            "video_degradation": [{}, {}],
        }
        batch["missing_mask"][:, :, :, 60:80] = 1.0

        model.set_input(batch)
        model.optimize_parameters(global_step=0)
        model.get_loss_items()
        errors = model.get_current_errors()

        assert math.isfinite(model.loss_video_margin_item)
        assert model.loss_video_margin_item >= 0.0
        assert math.isfinite(model.video_margin_l_original_item)
        assert math.isfinite(model.video_margin_l_wrong_item)
        assert model.video_margin_negative_mode == "batch_shuffle"
        assert errors["loss_video_margin"] == model.loss_video_margin_item
        assert errors["video_margin_negative_mode"] == "batch_shuffle"
        expected = (
            model.loss_diff_item
            + hparams.uq_lambda_boundary * model.loss_boundary_item
            + hparams.uq_lambda_video_margin * model.loss_video_margin_item
        )
        assert math.isclose(model.loss_total_item, expected, rel_tol=1e-5)

    def test_distill_loss_is_logged_and_in_total(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)
        model.lambda_distill = 0.25
        batch = self._make_batch(batch_size=2)

        def teacher_completed():
            return torch.zeros_like(model.mel_target)

        model._teacher_completed_mel = teacher_completed
        model.set_input(batch)
        model.optimize_parameters(global_step=0)
        model.get_loss_items()
        errors = model.get_current_errors()

        assert math.isfinite(model.loss_distill_item)
        assert model.loss_distill_item >= 0.0
        assert errors["loss_distill"] == model.loss_distill_item
        expected = (
            model.loss_diff_item
            + hparams.uq_lambda_boundary * model.loss_boundary_item
            + model.lambda_distill * model.loss_distill_item
        )
        assert math.isclose(model.loss_total_item, expected, rel_tol=1e-5)

    @pytest.mark.parametrize("prediction_type", ["epsilon", "x0", "v"])
    def test_prediction_type_train_and_sample(self, prediction_type,
                                              device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        hparams.uq_prediction_type = prediction_type
        model = UQAVDiffusionModel(hparams, device=device)
        batch = self._make_batch(batch_size=1)

        model.set_input(batch)
        model.optimize_parameters(global_step=0)
        model.get_loss_items()
        assert math.isfinite(model.loss_total_item)

        result = model.sample(
            batch, num_candidates=1, inference_steps=2,
            ddim_eta=0.0, seed=123,
        )
        assert result["completed_mels"].shape == (1, 1, 1, 80, 200)

    def test_ema_updates_saves_loads_and_applies(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        hparams.uq_use_ema = True
        hparams.uq_ema_decay = 0.5
        hparams.uq_ema_start_step = 0
        model = UQAVDiffusionModel(hparams, device=device)
        batch = self._make_batch(batch_size=1)

        model.set_input(batch)
        model.optimize_parameters(global_step=1)
        assert model.ema_state is not None

        ckpt_path = model.save_checkpoint(
            global_step=3,
            global_epoch=1,
            checkpoint_dir=str(tmp_path),
        )
        assert Path(str(ckpt_path).replace(".pth.tar", "_ema.pth.tar")).is_file()

        eval_hparams = self._make_hparams(tmp_path)
        eval_hparams.uq_ema_eval = True
        eval_model = UQAVDiffusionModel(eval_hparams, device=device)
        step, epoch = eval_model.load_checkpoint(
            ckpt_path, reset_optimizer=True,
        )
        assert step == 3
        assert epoch == 1
        assert eval_model.ema_state is not None
        eval_model.apply_ema_weights()

    def test_require_latent_stats_raises_for_missing_stats(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        hparams.uq_require_latent_stats = True
        model = UQAVDiffusionModel(hparams, device=device)
        ae_path = Path(tmp_path) / "ae_no_stats.pth.tar"
        torch.save(
            {
                "encoder": model.ae.encoder.state_dict(),
                "decoder": model.ae.decoder.state_dict(),
            },
            ae_path,
        )

        with pytest.raises(RuntimeError, match="uq_require_latent_stats"):
            model.load_ae_checkpoint(str(ae_path))

    def test_drop_video_zeroes_tokens_without_no_video_flag(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)

        B = 2
        batch = {
            "sample_id": ["s1", "s2"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None, None],
            "video_condition": ["original", "original"],
            "conditioning_mode": ["drop_video", "audio_video"],
            "video_degradation": [{}, {}],
        }
        model.set_input(batch)
        video_out = model.video_encoder(model.video, model.flow)
        dropped = model._apply_video_conditioning_modes(video_out)

        assert model.use_video
        assert torch.count_nonzero(dropped["video_tokens"][0]).item() == 0
        assert torch.count_nonzero(dropped["video_tokens"][1]).item() > 0

    def test_training_step_backprops_into_video_and_boundary(self, device, tmp_path):
        """Video encoder and boundary auxiliary loss must both train."""
        hparams = self._make_hparams(tmp_path)
        hparams.uq_lambda_boundary = 1.0
        model = UQAVDiffusionModel(hparams, device=device)

        B = 1
        batch = {
            "sample_id": ["s1"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None],
            "video_condition": ["original"],
            "video_degradation": [{}],
        }
        batch["missing_mask"][:, :, :, 60:80] = 1.0

        model.set_input(batch)
        model.optimize_parameters(global_step=0)

        video_grad = sum(
            p.grad.abs().sum().item()
            for p in model.video_encoder.parameters()
            if p.grad is not None
        )
        assert video_grad > 0.0
        assert model.loss_boundary.requires_grad
        assert model.loss_total.requires_grad

    def test_uq_cli_schedule_params_take_precedence(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        hparams.diff_timesteps = 100
        hparams.diff_beta_start = 1e-4
        hparams.diff_beta_end = 0.02
        hparams.uq_diffusion_timesteps = 37
        hparams.uq_beta_start = 0.001
        hparams.uq_beta_end = 0.01
        hparams.uq_beta_schedule = "linear"

        model = UQAVDiffusionModel(hparams, device=device)
        assert model.diffusion.timesteps == 37
        assert torch.isclose(
            model.diffusion.betas[0].cpu(), torch.tensor(0.001),
            atol=1e-7,
        )
        assert torch.isclose(
            model.diffusion.betas[-1].cpu(), torch.tensor(0.01),
            atol=1e-7,
        )

    def test_test_forward(self, device, tmp_path):
        """Run test forward pass and verify Mel pred shape."""
        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)
        model.eval_mode = True

        B = 2
        batch = {
            "sample_id": ["s1", "s2"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None, None],
            "video_condition": ["original", "original"],
            "video_degradation": [{}, {}],
        }
        for b in range(B):
            batch["missing_mask"][b, :, :, 60:80] = 1.0

        model.set_input(batch)
        model.test(global_step=0)
        model.get_loss_items()

        assert model.loss_diff_item > 0
        assert model.mel_pred.shape == (B, 1, 80, 200)

    def test_sample_k1(self, device, tmp_path):
        """K=1 DDIM sampling produces correct shapes."""


        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)

        B = 2
        batch = {
            "sample_id": ["s1", "s2"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None, None],
            "video_condition": ["original", "original"],
            "video_degradation": [{}, {}],
        }
        for b in range(B):
            batch["missing_mask"][b, :, :, 60:80] = 1.0

        result = model.sample(
            batch, num_candidates=1, inference_steps=5,
            ddim_eta=0.0, seed=42,
        )
        latent_dim = hparams.ae_latent_dim
        assert result["candidate_mels"].shape == (B, 1, 1, 80, 200)
        assert result["completed_mels"].shape == (B, 1, 1, 80, 200)
        assert result["candidate_latents"].shape == (B, 1, latent_dim, 10, 50)
        # P3: no scorer or evidence yet
        assert result["candidate_scores"] is None
        assert result["uncertainty"] is None
        assert result["visual_evidence"] is None

        metrics = compute_inpainting_sample_metrics(
            result["completed_mels"][:, 0],
            batch["mel_target"],
            batch["missing_mask"],
            mel_corrupted=batch["mel_corrupted"],
            compute_ssim=False,
        )
        for key in (
            "psnr_missing_db",
            "mel_l1_missing",
            "ssim_full",
            "boundary_l1",
            "known_region_max_abs_error_max",
        ):
            assert key in metrics

    def test_compose_known_region_is_strict(self, device, tmp_path):
        """After sampling, known Mel bins must be exactly preserved."""


        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)

        B = 1
        # Use deterministic Mel input to avoid randomness masking bug
        mel_target = torch.rand(B, 1, 80, 200)
        mel_corrupted = mel_target.clone()
        # Create a clean missing mask (20 frames in the middle)
        missing_mask = torch.zeros(B, 1, 80, 200)
        missing_mask[:, :, :, 70:90] = 1.0
        # Zero out the missing region in corrupted
        mel_corrupted = mel_corrupted * (1.0 - missing_mask)

        batch = {
            "sample_id": ["s1"],
            "mel_target": mel_target,
            "mel_corrupted": mel_corrupted,
            "missing_mask": missing_mask,
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None],
            "video_condition": ["original"],
            "video_degradation": [{}],
        }

        result = model.sample(
            batch, num_candidates=1, inference_steps=5,
            ddim_eta=0.0, seed=42,
        )
        completed = result["completed_mels"][0, 0]  # [1, 80, 200]

        # Known region must equal corrupted input
        known_mask = 1.0 - missing_mask[0].to(device)
        known_error = (
            (completed - mel_corrupted.to(device)) * known_mask
        ).abs().max()
        assert known_error.item() < 1e-5, \
            f"Known region composition violated! max error = {known_error.item()}"

    def test_fixed_seed_reproducibility(self, device, tmp_path):
        """Same seed + same input → byte-identical output."""


        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)

        B = 1
        batch = {
            "sample_id": ["s1"],
            "mel_target": torch.rand(B, 1, 80, 200),
            "mel_corrupted": torch.rand(B, 1, 80, 200),
            "missing_mask": torch.zeros(B, 1, 80, 200),
            "boundary_map": torch.zeros(B, 2, 80, 200),
            "video": torch.randn(B, 50, 3, 64, 64),
            "flow": torch.randn(B, 50, 2, 64, 64),
            "audio_target": torch.randn(B, 64000),
            "mask_spec": [None],
            "video_condition": ["original"],
            "video_degradation": [{}],
        }
        batch["missing_mask"][:, :, :, 60:80] = 1.0

        r1 = model.sample(
            batch, num_candidates=1, inference_steps=5,
            ddim_eta=0.0, seed=42,
        )
        r2 = model.sample(
            batch, num_candidates=1, inference_steps=5,
            ddim_eta=0.0, seed=42,
        )
        assert torch.allclose(
            r1["candidate_mels"], r2["candidate_mels"], atol=1e-6, rtol=0.0,
        ), "Same seed should produce near-identical outputs (CUDA may introduce epsilon differences)"

    def test_real_video_checkpoint_loads_in_no_video_mode(self, device, tmp_path):
        hparams = self._make_hparams(tmp_path)
        model = UQAVDiffusionModel(hparams, device=device)
        ckpt_path = model.save_checkpoint(
            global_step=7,
            global_epoch=2,
            checkpoint_dir=str(tmp_path),
        )

        no_video_hparams = self._make_hparams(tmp_path)
        no_video_hparams.uq_no_video = True
        no_video_model = UQAVDiffusionModel(no_video_hparams, device=device)
        step, epoch = no_video_model.load_checkpoint(
            ckpt_path,
            reset_optimizer=True,
        )

        assert step == 7
        assert epoch == 2
        assert no_video_model._loaded_step == 7
        assert isinstance(no_video_model.video_encoder, VideoConditionDummy)

    def test_models_importable(self):
        """Verify all P3 modules can be imported via importlib."""
        assert UQAVDiffusionModel is not None
        # train_uq_av / test_uq_av trigger Config parsing at module level
        # and cannot be imported in pytest context.  Their CLI entry points
        # are verified separately via `python main.py train-uq-av -- --help`.
        assert hasattr(UQAVDiffusionModel, "optimize_parameters")
        assert hasattr(UQAVDiffusionModel, "sample")
        assert hasattr(UQAVDiffusionModel, "save_checkpoint")

    def test_validation_sampling_options_parse(self):
        from base_options import BaseOptions

        opt = BaseOptions().parse(args=[
            "--uq_val_inference_steps", "5",
            "--uq_early_stop_patience", "10",
            "--uq_early_stop_min_delta", "0.01",
            "--uq_early_stop_metric", "val_sample_mel_l1_missing",
            "--uq_disable_early_stop",
        ])

        assert opt.uq_val_inference_steps == 5
        assert opt.uq_early_stop_patience == 10
        assert opt.uq_early_stop_min_delta == 0.01
        assert opt.uq_early_stop_metric == "val_sample_mel_l1_missing"
        assert opt.uq_disable_early_stop

        shuffled_opt = BaseOptions().parse(args=[
            "--uq_video_degradation", "shuffled_video",
        ])
        assert shuffled_opt.uq_video_degradation == "shuffled_video"

        p2_opt = BaseOptions().parse(args=[
            "--uq_enable_modality_dropout",
            "--uq_p_audio_video", "0.4",
            "--uq_p_drop_video", "0.2",
            "--uq_p_partial_audio_video", "0.2",
            "--uq_p_wrong_video", "0.1",
            "--uq_p_shuffled_video", "0.1",
            "--uq_audio_context_drop_min_ratio", "0.15",
            "--uq_audio_context_drop_max_ratio", "0.35",
            "--uq_condition_override", "drop_audio",
        ])
        assert p2_opt.uq_enable_modality_dropout
        assert p2_opt.uq_p_audio_video == 0.4
        assert p2_opt.uq_p_drop_video == 0.2
        assert p2_opt.uq_p_partial_audio_video == 0.2
        assert p2_opt.uq_p_wrong_video == 0.1
        assert p2_opt.uq_p_shuffled_video == 0.1
        assert p2_opt.uq_audio_context_drop_min_ratio == 0.15
        assert p2_opt.uq_audio_context_drop_max_ratio == 0.35
        assert p2_opt.uq_condition_override == "drop_audio"

        p3_opt = BaseOptions().parse(args=[
            "--uq_lambda_video_margin", "0.1",
            "--uq_video_margin", "0.02",
            "--uq_video_margin_negative", "temporal_shuffle",
        ])
        assert p3_opt.uq_lambda_video_margin == 0.1
        assert p3_opt.uq_video_margin == 0.02
        assert p3_opt.uq_video_margin_negative == "temporal_shuffle"

        p5_p6_opt = BaseOptions().parse(args=[
            "--uq_teacher_type", "patchgan",
            "--uq_teacher_checkpoint", "/tmp/teacher.pth.tar",
            "--uq_teacher_ae_checkpoint", "/tmp/teacher_ae.pth.tar",
            "--uq_lambda_distill", "0.25",
            "--uq_teacher_inference_steps", "25",
            "--uq_teacher_ddim_eta", "0.1",
            "--uq_prediction_type", "v",
            "--uq_latent_clip_value", "3.5",
            "--uq_require_latent_stats",
            "--uq_use_ema",
            "--uq_ema_decay", "0.995",
            "--uq_ema_start_step", "10",
            "--uq_ema_eval",
        ])
        assert p5_p6_opt.uq_teacher_type == "patchgan"
        assert p5_p6_opt.uq_teacher_checkpoint == "/tmp/teacher.pth.tar"
        assert p5_p6_opt.uq_teacher_ae_checkpoint == "/tmp/teacher_ae.pth.tar"
        assert p5_p6_opt.uq_lambda_distill == 0.25
        assert p5_p6_opt.uq_teacher_inference_steps == 25
        assert p5_p6_opt.uq_teacher_ddim_eta == 0.1
        assert p5_p6_opt.uq_prediction_type == "v"
        assert p5_p6_opt.uq_latent_clip_value == 3.5
        assert p5_p6_opt.uq_require_latent_stats
        assert p5_p6_opt.uq_use_ema
        assert p5_p6_opt.uq_ema_decay == 0.995
        assert p5_p6_opt.uq_ema_start_step == 10
        assert p5_p6_opt.uq_ema_eval


# ===================================================================
# Diffusion schedule specific edge cases
# ===================================================================
class TestDiffusionEdgeCases:
    """Edge cases for diffusion schedule."""

    def test_single_timestep(self):
        sched = DiffusionSchedule(timesteps=1)
        assert sched.timesteps == 1
        assert sched.betas.shape == (1,)

    def test_deterministic_q_sample(self):
        """Same z_0, t, noise → same z_t."""
        sched = DiffusionSchedule(timesteps=100)
        z_0 = torch.randn(2, 4, 10, 50)
        t = torch.tensor([50, 75])
        noise = torch.randn(2, 4, 10, 50)
        z1 = sched.q_sample(z_0, t, noise=noise)
        z2 = sched.q_sample(z_0, t, noise=noise)
        assert torch.equal(z1, z2)

    def test_different_noise_different_sample(self):
        sched = DiffusionSchedule(timesteps=100)
        z_0 = torch.randn(2, 4, 10, 50)
        t = torch.tensor([50, 75])
        n1 = torch.randn(2, 4, 10, 50)
        n2 = torch.randn(2, 4, 10, 50)
        z1 = sched.q_sample(z_0, t, noise=n1)
        z2 = sched.q_sample(z_0, t, noise=n2)
        assert not torch.equal(z1, z2)


# ===================================================================
# No-video / wrong-video comparison readiness
# ===================================================================
class TestVideoConditions:
    """Verify model handles different video conditions gracefully."""

    def test_no_video_flag(self, device, tmp_path):
        """uq_no_video=True should use VideoConditionDummy."""
        from types import SimpleNamespace
        hparams = SimpleNamespace(
            ae_latent_dim=4, ae_base_channels=16, ae_norm_type="batch",
            uq_video_dim=64, uq_unet_base_channels=16, uq_time_emb_dim=64,
            uq_lambda_boundary=0.1, uq_lambda_sync=0.0, uq_attn_heads=2,
            uq_no_video=True, uq_lr=1e-4, lr=1e-4,
            beta1=0.5, beta2=0.999, save_optimizer_state=True,
            image_size=64, diff_timesteps=100, diff_beta_start=1e-4,
            diff_beta_end=0.02, diff_schedule="linear", name="test",
            uq_inference_steps=5, uq_grad_clip=1.0,
            checkpoint_dir=str(tmp_path),
        )
        model = UQAVDiffusionModel(hparams, device=device)
        assert isinstance(model.video_encoder, VideoConditionDummy)
        assert not model.use_video

    def _write_tiny_uq_dataset(self, root):
        sample_ids = [
            "processed/piano/video_a/clip_000001",
            "processed/piano/video_b/clip_000001",
        ]
        for sample_index, sample_id in enumerate(sample_ids):
            sample_dir = Path(root) / sample_id
            for name in ("image_crop", "flow_x_crop", "flow_y_crop"):
                (sample_dir / name).mkdir(parents=True, exist_ok=True)

            mel = np.full((200, 80), sample_index / 10.0, dtype=np.float32)
            audio = np.zeros((64000,), dtype=np.float32)
            np.save(sample_dir / "mel.npy", mel, allow_pickle=False)
            np.save(sample_dir / "raw_audio.npy", audio, allow_pickle=False)

            for frame_id in range(1, 51):
                image = np.full(
                    (8, 8, 3),
                    20 + sample_index * 100 + frame_id,
                    dtype=np.uint8,
                )
                flow_x = np.full((8, 8), 127 + sample_index, dtype=np.uint8)
                flow_y = np.full((8, 8), 127 - sample_index, dtype=np.uint8)
                cv2.imwrite(
                    str(sample_dir / "image_crop" / f"{frame_id}.jpg"),
                    image,
                )
                cv2.imwrite(
                    str(sample_dir / "flow_x_crop" / f"{frame_id}.jpg"),
                    flow_x,
                )
                cv2.imwrite(
                    str(sample_dir / "flow_y_crop" / f"{frame_id}.jpg"),
                    flow_y,
                )

        split_path = Path(root) / "test_av_split.txt"
        with split_path.open("w", encoding="utf-8") as handle:
            for sample_id in sample_ids:
                handle.write(
                    f"{sample_id}|{sample_id}/mel.npy|"
                    f"{sample_id}/raw_audio.npy|200\n"
                )

        manifest = {
            sample_id: [
                {
                    "mask_type": "random",
                    "start": 50,
                    "end": 70,
                    "gap_frames": 20,
                    "seed": 123,
                }
            ]
            for sample_id in sample_ids
        }
        return manifest

    def test_loader_video_degradation_conditions(self, tmp_path):
        manifest = self._write_tiny_uq_dataset(tmp_path)

        no_video_loader = create_uq_av_dataloader(
            data_root=tmp_path,
            split_name="test_av_split.txt",
            phase="test",
            mask_manifest=manifest,
            video_conditions=("no_video",),
            image_size=8,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        no_video_batch = next(iter(no_video_loader))
        assert no_video_batch["video_condition"] == ["no_video", "no_video"]
        assert no_video_batch["video"].abs().sum().item() == 0.0
        assert no_video_batch["flow"].abs().sum().item() == 0.0

        wrong_video_loader = create_uq_av_dataloader(
            data_root=tmp_path,
            split_name="test_av_split.txt",
            phase="test",
            mask_manifest=manifest,
            video_conditions=("wrong_video",),
            image_size=8,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        wrong_video_batch = next(iter(wrong_video_loader))
        assert wrong_video_batch["video_condition"] == [
            "wrong_video", "wrong_video",
        ]
        assert "wrong_video_sample_id" in wrong_video_batch[
            "video_degradation"
        ][0]
