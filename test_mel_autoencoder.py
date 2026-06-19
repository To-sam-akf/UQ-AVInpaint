"""Test deterministic convolutional Mel autoencoder.

Evaluates AE reconstruction quality:
  - PSNR, SSIM, Mel L1, boundary gradient error
  - Output range check ([0, 1], NaN/Inf)
  - Determinism check (same input → same latent)
  - Optional Griffin-Lim audio export

Usage:
    python test_mel_autoencoder.py --name MelAE_test \
        --resume_path checkpoints/mel_ae/MelAE_checkpoint_step*.pth.tar \
        --data_root /path/to/data --batch_size 16
"""

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

import Options_inpainting
from Data_loaders.uq_av_loader import get_uq_av_data_loaders
from Models.Mel_Autoencoder import MelAEModel
from networks.uq.mel_autoencoder import time_gradient
from utils.viai_a_metrics import structural_similarity_2d

# ---------------------------------------------------------------------------
hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def _configure_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "MelAE_test"
    if not _arg_was_passed("--test_split_name"):
        hparams.test_split_name = "test_av_split.txt"
    if not _arg_was_passed("--results_dir"):
        hparams.results_dir = os.path.join(
            getattr(hparams, "checkpoint_dir", "./checkpoints/mel_ae"),
            "test_results",
        )


def _compute_ae_metrics(mel_recon, mel_target):
    """Compute per-sample AE reconstruction metrics.

    Args:
        mel_recon: [B, 1, F, T] or [1, F, T]
        mel_target: same shape
    Returns:
        dict with scalar metrics averaged over batch.
    """
    if mel_recon.dim() == 3:
        mel_recon = mel_recon.unsqueeze(0)
    if mel_target.dim() == 3:
        mel_target = mel_target.unsqueeze(0)

    B = mel_recon.size(0)
    recon = torch.clamp(mel_recon, 0.0, 1.0)
    target = torch.clamp(mel_target, 0.0, 1.0)

    # Mel L1
    mel_l1 = F.l1_loss(recon, target, reduction='none').mean(dim=(1, 2, 3))

    # PSNR
    mse = (recon - target) ** 2
    mse_full = mse.mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(torch.clamp(mse_full, min=1e-12))

    # SSIM (per-sample, CPU)
    ssim_vals = []
    recon_np = recon.squeeze(1).cpu().numpy()
    target_np = target.squeeze(1).cpu().numpy()
    for i in range(B):
        ssim_vals.append(structural_similarity_2d(recon_np[i], target_np[i]))
    ssim = float(np.mean(ssim_vals))

    # Boundary gradient error: L1 of time-gradient difference
    grad_recon = time_gradient(recon)
    grad_target = time_gradient(target)
    grad_l1 = F.l1_loss(grad_recon, grad_target, reduction='none').mean(dim=(1, 2, 3))

    return {
        "mel_l1": float(mel_l1.mean().cpu().item()),
        "psnr": float(psnr.mean().cpu().item()),
        "ssim": ssim,
        "gradient_l1": float(grad_l1.mean().cpu().item()),
    }


def _check_output_quality(mel_recon):
    """Verify output range [0, 1] and no NaN/Inf.

    Returns:
        dict with min, max, has_nan, has_inf.
    """
    return {
        "min": float(mel_recon.min().cpu().item()),
        "max": float(mel_recon.max().cpu().item()),
        "has_nan": bool(torch.isnan(mel_recon).any().cpu().item()),
        "has_inf": bool(torch.isinf(mel_recon).any().cpu().item()),
    }


def _griffin_lim_export(mel_spec, output_path, n_iter=32, sample_rate=16000,
                        n_fft=1280, hop_length=320):
    """Export a single Mel spectrogram to WAV via Griffin-Lim.

    Args:
        mel_spec: [1, F, T] numpy array in [0, 1].
        output_path: WAV file path.
    """
    try:
        import librosa
        import soundfile as sf
    except ImportError:
        print("[MelAE test] librosa/soundfile not available; skipping audio export.")
        return False

    mel = np.clip(mel_spec.squeeze(), 0.0, 1.0)  # [F, T]

    # Invert mel-to-linear conversion used by librosa
    # The project scales mels to [0, 1]; we map back to dB then to power
    min_db = getattr(hparams, "min_level_db", -100.0)
    ref_db = getattr(hparams, "ref_level_db", 20.0)

    # Approximate inverse: mel_db = mel * (-min_db) + min_db  (simplified)
    mel_db = mel * (-min_db) + min_db
    mel_power = np.power(10.0, mel_db / 10.0)

    # Build mel filterbank and invert
    mel_basis = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=mel_power.shape[0],
        fmin=getattr(hparams, "fmin", 125.0),
        fmax=getattr(hparams, "fmax", 7600.0),
    )

    # Pseudo-inverse of mel basis
    mel_inv = np.linalg.pinv(mel_basis)
    linear_spec = np.maximum(1e-10, mel_inv @ mel_power)  # [n_fft//2+1, T]

    # Griffin-Lim
    mel_linear_db = 10.0 * np.log10(linear_spec)
    audio = librosa.griffinlim(
        mel_linear_db, n_iter=n_iter, hop_length=hop_length,
        win_length=n_fft, window='hann',
    )

    # Normalise
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return True


def main():
    _configure_defaults()

    if not hparams.resume_path:
        raise SystemExit(
            "[MelAE test] --resume_path is required. "
            "Provide a trained MelAE checkpoint."
        )
    if not os.path.exists(hparams.resume_path):
        raise SystemExit(f"[MelAE test] checkpoint not found: {hparams.resume_path}")

    split_names = {"test": hparams.test_split_name}

    print(f"[MelAE test] data_root={hparams.data_root}")
    print(f"[MelAE test] checkpoint={hparams.resume_path}")

    data_loaders = get_uq_av_data_loaders(
        hparams.data_root, split_names,
        phases=("test",),
        batch_size=hparams.batch_size,
        num_workers=getattr(hparams, "num_workers", 4),
        pin_memory=getattr(hparams, "pin_memory", True),
    )
    test_loader = data_loaders["test"]

    model = MelAEModel(hparams, device=device)
    model.load_checkpoint(hparams.resume_path, reset_optimizer=True)
    model.net.eval()

    # -------------------------------------------------------------------
    # Collect metrics
    # -------------------------------------------------------------------
    all_metrics = defaultdict(list)
    all_quality = []
    determinism_ok = True
    num_nan = 0
    num_inf = 0

    os.makedirs(hparams.results_dir, exist_ok=True)
    wav_dir = os.path.join(hparams.results_dir, "wav")
    max_wav_exports = int(getattr(hparams, "vocoder_max_samples", None) or 3)

    progress = tqdm(test_loader, desc="[MelAE test]", unit="batch", dynamic_ncols=True)
    for batch_idx, batch in enumerate(progress):
        mel = batch["mel_target"].float().to(device)  # [B, 1, 80, 200]

        with torch.no_grad():
            mel_recon, z = model.net(mel)

        # Determinism check: second pass with same input
        with torch.no_grad():
            z2 = model.net.encode(mel)
        if not torch.allclose(z, z2, atol=1e-6):
            determinism_ok = False

        # Quality check
        quality = _check_output_quality(mel_recon)
        all_quality.append(quality)
        if quality["has_nan"]:
            num_nan += 1
        if quality["has_inf"]:
            num_inf += 1

        # Per-sample metrics
        metrics = _compute_ae_metrics(mel_recon, mel)
        for k, v in metrics.items():
            all_metrics[k].append(v)

        # Griffin-Lim export for first few batches
        use_vocoder = getattr(hparams, "use_vocoder", False)
        if use_vocoder and batch_idx < max_wav_exports:
            mel_target_np = mel[:1].cpu().numpy()
            mel_recon_np = mel_recon[:1].cpu().numpy()
            sid = batch.get("sample_id", [f"batch_{batch_idx:04d}"])[0]
            safe_id = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(sid))
            _griffin_lim_export(
                mel_recon_np,
                os.path.join(wav_dir, f"{safe_id}_recon.wav"),
                n_iter=getattr(hparams, "vocoder_n_iter", 32),
            )
            _griffin_lim_export(
                mel_target_np,
                os.path.join(wav_dir, f"{safe_id}_target.wav"),
                n_iter=getattr(hparams, "vocoder_n_iter", 32),
            )

    # -------------------------------------------------------------------
    # Aggregate and report
    # -------------------------------------------------------------------
    summary = {}
    for k, vals in all_metrics.items():
        summary[k] = float(np.mean(vals))

    quality_summary = {
        "min_overall": float(min(q["min"] for q in all_quality)),
        "max_overall": float(max(q["max"] for q in all_quality)),
        "any_nan": num_nan > 0,
        "any_inf": num_inf > 0,
        "nan_batches": num_nan,
        "inf_batches": num_inf,
    }

    print("\n" + "=" * 60)
    print("  Mel Autoencoder Test Results")
    print("=" * 60)
    print(f"  PSNR:           {summary.get('psnr', 0):.3f} dB")
    print(f"  SSIM:           {summary.get('ssim', 0):.6f}")
    print(f"  Mel L1:         {summary.get('mel_l1', 0):.6f}")
    print(f"  Gradient L1:    {summary.get('gradient_l1', 0):.6f}")
    print(f"  Output range:   [{quality_summary['min_overall']:.6f}, "
          f"{quality_summary['max_overall']:.6f}]")
    print(f"  NaN detected:   {quality_summary['any_nan']}")
    print(f"  Inf detected:   {quality_summary['any_inf']}")
    print(f"  Deterministic:  {'PASS' if determinism_ok else 'FAIL'}")
    print(f"  Latent mean:    {model.latent_mean}")
    print(f"  Latent std:     {model.latent_std}")
    print("=" * 60)

    # -------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------
    results = {
        "checkpoint": hparams.resume_path,
        "metrics": summary,
        "quality": quality_summary,
        "deterministic": determinism_ok,
        "latent_mean": (model.latent_mean.tolist()
                        if model.latent_mean is not None else None),
        "latent_std": (model.latent_std.tolist()
                       if model.latent_std is not None else None),
        "config": {
            "latent_dim": int(getattr(hparams, "ae_latent_dim", 8)),
            "base_channels": int(getattr(hparams, "ae_base_channels", 32)),
        },
    }
    results_path = os.path.join(hparams.results_dir, "ae_test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[MelAE test] Results saved to {results_path}")

    # Exit with error if quality checks fail
    if quality_summary["any_nan"] or quality_summary["any_inf"]:
        raise SystemExit("[MelAE test] FAIL: NaN or Inf detected in output.")
    if not determinism_ok:
        raise SystemExit("[MelAE test] FAIL: Encode is not deterministic.")


if __name__ == "__main__":
    main()
    sys.exit(0)
