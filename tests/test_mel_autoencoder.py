"""Unit tests for P2 Mel Latent Autoencoder.

Covers:
  - Shape contract (encode/decode/forward)
  - Output range [0, 1] and no NaN/Inf
  - Deterministic encode
  - Time gradient function
  - Random boundary loss
  - Model training wrapper smoke test
  - Checkpoint save/load round-trip
  - Latent statistics computation
"""

import importlib
import os
import sys
import tempfile

import pytest
import torch
import torch.nn as nn

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def mel_batch(batch_size):
    """Random Mel spectrograms in [0, 1]."""
    return torch.rand(batch_size, 1, 80, 200)


@pytest.fixture
def mel_batch_3d(batch_size):
    """Random Mel spectrograms in [0, 1] with shape [B, 80, 200]."""
    return torch.rand(batch_size, 80, 200)


# ---------------------------------------------------------------------------
# Imports — use importlib to bypass Models/__init__.py which pulls in
# the full wavenet/Config chain and conflicts with pytest flags like --tb.
# ---------------------------------------------------------------------------
from networks.uq.mel_autoencoder import (
    MelAutoencoder,
    MelEncoder,
    MelDecoder,
    time_gradient,
    random_boundary_loss,
)

# Load MelAEModel module directly, skipping Models/__init__.py
_mel_ae_spec = importlib.util.spec_from_file_location(
    "Mel_Autoencoder",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "Models", "Mel_Autoencoder.py"),
)
_mel_ae_mod = importlib.util.module_from_spec(_mel_ae_spec)
_mel_ae_spec.loader.exec_module(_mel_ae_mod)
MelAEModel = _mel_ae_mod.MelAEModel


# ===================================================================
# Network shape contract tests
# ===================================================================
class TestNetworkShapes:
    """Verify encode/decode/forward shapes strictly match the contract."""

    def test_encoder_shape_4d(self, mel_batch, batch_size):
        encoder = MelEncoder(latent_dim=8)
        z = encoder(mel_batch)
        assert z.shape == (batch_size, 8, 10, 50), \
            f"Expected (B, 8, 10, 50), got {z.shape}"

    def test_encoder_shape_custom_latent(self, mel_batch, batch_size):
        for ld in [4, 8, 16, 32]:
            encoder = MelEncoder(latent_dim=ld)
            z = encoder(mel_batch)
            assert z.shape == (batch_size, ld, 10, 50)

    def test_decoder_shape(self, batch_size):
        decoder = MelDecoder(latent_dim=8)
        z = torch.randn(batch_size, 8, 10, 50)
        mel = decoder(z)
        assert mel.shape == (batch_size, 1, 80, 200), \
            f"Expected (B, 1, 80, 200), got {mel.shape}"

    def test_decoder_shape_custom_latent(self, batch_size):
        for ld in [4, 8, 16, 32]:
            decoder = MelDecoder(latent_dim=ld)
            z = torch.randn(batch_size, ld, 10, 50)
            mel = decoder(z)
            assert mel.shape == (batch_size, 1, 80, 200)

    def test_autoencoder_forward_4d(self, mel_batch, batch_size):
        ae = MelAutoencoder(latent_dim=8)
        mel_recon, z = ae(mel_batch)
        assert mel_recon.shape == (batch_size, 1, 80, 200)
        assert z.shape == (batch_size, 8, 10, 50)

    def test_autoencoder_encode_decode(self, mel_batch, batch_size):
        ae = MelAutoencoder(latent_dim=8)
        z = ae.encode(mel_batch)
        assert z.shape == (batch_size, 8, 10, 50)
        mel_recon = ae.decode(z)
        assert mel_recon.shape == (batch_size, 1, 80, 200)

    def test_encode_accepts_3d(self, mel_batch_3d, batch_size):
        ae = MelAutoencoder(latent_dim=8)
        z = ae.encode(mel_batch_3d)
        assert z.shape == (batch_size, 8, 10, 50)

    def test_forward_accepts_3d(self, mel_batch_3d, batch_size):
        ae = MelAutoencoder(latent_dim=8)
        mel_recon, z = ae(mel_batch_3d)
        assert mel_recon.shape == (batch_size, 1, 80, 200)


# ===================================================================
# Output quality tests
# ===================================================================
class TestOutputQuality:
    """Verify output range, no NaN/Inf."""

    def test_output_in_01_range(self, mel_batch):
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        with torch.no_grad():
            mel_recon, _ = ae(mel_batch)
        assert mel_recon.min() >= 0.0, f"min={mel_recon.min()}"
        assert mel_recon.max() <= 1.0, f"max={mel_recon.max()}"

    def test_no_nan_inf(self, mel_batch):
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        with torch.no_grad():
            mel_recon, z = ae(mel_batch)
        assert not torch.isnan(mel_recon).any(), "NaN in reconstruction"
        assert not torch.isinf(mel_recon).any(), "Inf in reconstruction"
        assert not torch.isnan(z).any(), "NaN in latent"
        assert not torch.isinf(z).any(), "Inf in latent"

    def test_output_range_with_extreme_input(self, batch_size):
        """Extreme inputs should still produce valid outputs."""
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        # All zeros
        with torch.no_grad():
            r0, _ = ae(torch.zeros(batch_size, 1, 80, 200))
            assert not torch.isnan(r0).any()
            assert r0.min() >= 0.0 and r0.max() <= 1.0
        # All ones
        with torch.no_grad():
            r1, _ = ae(torch.ones(batch_size, 1, 80, 200))
            assert not torch.isnan(r1).any()
            assert r1.min() >= 0.0 and r1.max() <= 1.0
        # Random noise
        with torch.no_grad():
            rr, _ = ae(torch.rand(batch_size, 1, 80, 200))
            assert not torch.isnan(rr).any()
            assert rr.min() >= 0.0 and rr.max() <= 1.0


# ===================================================================
# Determinism tests
# ===================================================================
class TestDeterminism:
    """Verify encode is fully deterministic (no randomness)."""

    def test_same_input_same_latent(self, mel_batch):
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        with torch.no_grad():
            z1 = ae.encode(mel_batch)
            z2 = ae.encode(mel_batch)
        assert torch.allclose(z1, z2, atol=1e-6), \
            f"max diff: {(z1 - z2).abs().max()}"

    def test_different_inputs_different_latents(self, batch_size):
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        x1 = torch.rand(batch_size, 1, 80, 200)
        x2 = torch.rand(batch_size, 1, 80, 200)
        with torch.no_grad():
            z1 = ae.encode(x1)
            z2 = ae.encode(x2)
        assert not torch.allclose(z1, z2, atol=1e-4), \
            "Different inputs should produce different latents"

    def test_single_sample_determinism(self, batch_size):
        """Repeated encode of same single sample gives identical latent."""
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        x = torch.rand(1, 1, 80, 200)
        with torch.no_grad():
            z_ref = ae.encode(x)
            for _ in range(10):
                z = ae.encode(x)
                assert torch.allclose(z, z_ref, atol=1e-7)


# ===================================================================
# Time gradient tests
# ===================================================================
class TestTimeGradient:
    def test_shape(self):
        x = torch.randn(2, 1, 80, 200)
        diff = time_gradient(x)
        assert diff.shape == (2, 1, 80, 199)

    def test_constant_input_zero_gradient(self):
        x = torch.ones(2, 1, 80, 200)
        diff = time_gradient(x)
        assert torch.allclose(diff, torch.zeros_like(diff), atol=1e-7)

    def test_linear_input(self):
        # x[t] = t, gradient should be 1 everywhere
        x = torch.arange(200, dtype=torch.float32).view(1, 1, 1, 200).expand(2, 1, 80, 200)
        diff = time_gradient(x)
        assert diff.shape == (2, 1, 80, 199)
        assert torch.allclose(diff, torch.ones_like(diff), atol=1e-7)

    def test_step_change(self):
        x = torch.zeros(1, 1, 10, 200)
        x[:, :, :, 100:] = 5.0
        diff = time_gradient(x)
        # Only the step should be non-zero
        assert diff[:, :, :, 99].abs().max() > 4.0


# ===================================================================
# Random boundary loss tests
# ===================================================================
class TestRandomBoundaryLoss:
    def test_returns_scalar(self, mel_batch):
        loss = random_boundary_loss(mel_batch, mel_batch)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_identical_inputs_zero_loss(self, mel_batch):
        loss = random_boundary_loss(mel_batch, mel_batch)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_inputs_nonzero_loss(self, mel_batch):
        recon = torch.rand_like(mel_batch)
        target = torch.rand_like(mel_batch)
        loss = random_boundary_loss(recon, target)
        assert loss.item() > 0.0

    def test_short_sequence_does_not_crash(self):
        """Should handle sequences shorter than boundary contexts."""
        x = torch.rand(2, 1, 80, 20)  # Only 20 time steps
        loss = random_boundary_loss(x, x)
        assert loss.dim() == 0


# ===================================================================
# Model wrapper tests
# ===================================================================
class DummyHParams:
    """Minimal config for testing."""
    name = "test_ae"
    ae_latent_dim = 8
    ae_base_channels = 32
    ae_norm_type = "batch"
    ae_lambda_l1 = 1.0
    ae_lambda_grad = 0.1
    ae_lambda_boundary = 0.05
    ae_warmup_steps = 100
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.999
    save_optimizer_state = True


class TestMelAEModel:
    """Test the training wrapper."""

    def test_init(self):
        model = MelAEModel(DummyHParams())
        assert model.net is not None
        assert model.optimizer is not None
        assert model.latent_mean is None
        assert model.latent_std is None

    def test_set_input_and_forward(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        model.net.eval()
        with torch.no_grad():
            model._forward()
        assert model.mel_recon.shape == mel_batch.shape
        assert model.z.shape == (mel_batch.size(0), 8, 10, 50)

    def test_encode_decode_interface(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.net.eval()
        with torch.no_grad():
            z = model.encode(mel_batch)
            assert z.shape == (mel_batch.size(0), 8, 10, 50)
            mel_r = model.decode(z)
            assert mel_r.shape == mel_batch.shape

    def test_loss_computation(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        model.net.eval()
        with torch.no_grad():
            model._forward()
            model._compute_loss(global_step=200)  # past warmup
        assert model.loss_total_item == 0.0  # not yet extracted
        model.get_loss_items()
        assert model.loss_l1_item >= 0.0
        assert model.loss_grad_item >= 0.0
        assert model.loss_boundary_item >= 0.0

    def test_warmup_only_l1(self, mel_batch):
        """During warmup, total loss should equal lambda_l1 * L1."""
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        # Make grad and boundary losses non-zero
        model.net.eval()
        with torch.no_grad():
            model._forward()
            model._compute_loss(global_step=0)  # warmup
        model.get_loss_items()
        # In warmup: loss_total = lambda_l1 * loss_l1
        expected = model.lambda_l1 * model.loss_l1_item
        assert model.loss_total_item == pytest.approx(expected, abs=1e-6)

    def test_optimizer_step(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        params_before = [p.clone() for p in model.net.parameters()]
        model.optimize_parameters(global_step=200)
        params_after = list(model.net.parameters())
        changed = False
        for pb, pa in zip(params_before, params_after):
            if not torch.allclose(pb, pa, atol=1e-8):
                changed = True
                break
        assert changed, "Parameters did not change after optimizer step"

    def test_test_mode_no_grad(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        model.test(global_step=0)
        model.get_loss_items()
        assert model.loss_total_item >= 0.0

    def test_get_current_errors(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        model.test(global_step=0)
        model.get_loss_items()
        errors = model.get_current_errors()
        assert "loss_total" in errors
        assert "loss_l1" in errors
        assert "loss_grad" in errors
        assert "loss_boundary" in errors
        assert "lr" in errors


# ===================================================================
# Checkpoint save/load tests
# ===================================================================
class TestCheckpoint:
    def test_save_load_roundtrip(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.set_input(mel_batch)
        model.test(global_step=0)
        model.get_loss_items()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            ckpt_path = model.save_checkpoint(
                global_step=42, global_epoch=3,
                checkpoint_dir=tmpdir,
                latent_stats={"mean": torch.randn(8), "std": torch.rand(8).abs() + 0.1},
            )
            assert os.path.exists(ckpt_path)

            # Load into new model
            model2 = MelAEModel(DummyHParams(), device=torch.device("cpu"))
            gs, ge = model2.load_checkpoint(ckpt_path)
            assert gs == 42
            assert ge == 3
            assert model2.latent_mean is not None
            assert model2.latent_std is not None

            # Verify weights match
            for (n1, p1), (n2, p2) in zip(
                model.net.named_parameters(), model2.net.named_parameters()
            ):
                assert torch.allclose(p1, p2, atol=1e-6), \
                    f"Parameter {n1} mismatch after load"

    def test_load_preserves_latent_stats(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        with tempfile.TemporaryDirectory() as tmpdir:
            mean = torch.ones(8) * 0.5
            std = torch.ones(8) * 0.2
            ckpt_path = model.save_checkpoint(
                0, 0, tmpdir, latent_stats={"mean": mean, "std": std},
            )
            model2 = MelAEModel(DummyHParams(), device=torch.device("cpu"))
            model2.load_checkpoint(ckpt_path)
            assert torch.allclose(model2.latent_mean, mean, atol=1e-6)
            assert torch.allclose(model2.latent_std, std, atol=1e-6)


# ===================================================================
# Latent statistics tests
# ===================================================================
class TestLatentStats:
    def test_compute_stats_deterministic(self, mel_batch):
        """Repeated stat computation on same data gives same result."""
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.net.eval()

        class FixedLoader:
            def __init__(self, tensor):
                self.tensor = tensor
            def __iter__(self):
                yield {"mel_target": self.tensor}

        loader = FixedLoader(mel_batch)
        with torch.no_grad():
            s1 = model.compute_latent_stats(loader)
            s2 = model.compute_latent_stats(loader)
        assert torch.allclose(s1["mean"], s2["mean"], atol=1e-6)
        assert torch.allclose(s1["std"], s2["std"], atol=1e-6)

    def test_mean_std_shapes(self, mel_batch):
        model = MelAEModel(DummyHParams(), device=torch.device("cpu"))
        model.net.eval()

        class FixedLoader:
            def __init__(self, tensor):
                self.tensor = tensor
            def __iter__(self):
                yield {"mel_target": self.tensor}

        loader = FixedLoader(mel_batch)
        with torch.no_grad():
            stats = model.compute_latent_stats(loader)
        ld = DummyHParams.ae_latent_dim
        assert stats["mean"].shape == (ld,)
        assert stats["std"].shape == (ld,)
        assert (stats["std"] > 0).all(), "Std should be positive"


# ===================================================================
# Integration: encoder-decoder reconstruction sanity
# ===================================================================
class TestReconstructionSanity:
    """Basic reconstruction quality checks."""

    def test_reconstruction_not_nan(self, batch_size):
        """AE forward pass should produce valid outputs (no NaN/Inf)."""
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        mel = torch.rand(batch_size, 1, 80, 200)
        with torch.no_grad():
            recon, _ = ae(mel)
        assert not torch.isnan(recon).any(), "Reconstruction contains NaN"
        assert not torch.isinf(recon).any(), "Reconstruction contains Inf"

    def test_encode_decode_roundtrip_shape_preserved(self, mel_batch):
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        with torch.no_grad():
            z = ae.encode(mel_batch)
            mel_recon = ae.decode(z)
        assert mel_recon.shape == mel_batch.shape

    def test_batch_independence(self, batch_size):
        """Each sample's latent should only depend on its own input."""
        ae = MelAutoencoder(latent_dim=8)
        ae.eval()
        x1 = torch.rand(1, 1, 80, 200)
        x2 = torch.rand(1, 1, 80, 200)
        # Process in same batch
        with torch.no_grad():
            z_batch = ae.encode(torch.cat([x1, x2], dim=0))
        # Process individually
        with torch.no_grad():
            z1 = ae.encode(x1)
            z2 = ae.encode(x2)
        assert torch.allclose(z_batch[0:1], z1, atol=1e-6)
        assert torch.allclose(z_batch[1:2], z2, atol=1e-6)
