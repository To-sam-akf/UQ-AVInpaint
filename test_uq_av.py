"""Test UQ-AV Latent Diffusion (P3) — K=1 evaluation.

Runs DDIM inference with K=1, computes metrics, exports Mel images,
and optionally generates wav via Griffin-Lim vocoder.

Usage (via main.py):
    python main.py test-uq-av -- \\
      --resume_path checkpoints/uq_av_k1/UQ-AV_checkpoint_step000010000.pth.tar \\
      --ae_checkpoint checkpoints/mel_ae/MelAE_checkpoint_step000005000.pth.tar \\
      --data_root /path/to/data \\
      --test_split_name test_av_split.txt \\
      --batch_size 8 \\
      --results_dir checkpoints/uq_av_k1_test_results \\
      --use_vocoder --vocoder_n_iter 32
"""

import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Options_inpainting
from Data_loaders.uq_av_loader import get_uq_av_data_loaders
from Models.UQ_AV_Diffusion import UQAVDiffusionModel
from utils.viai_a_metrics import (
    compute_viai_a_metrics,
    compose_inpainted_mel,
    save_mel_comparison_batch,
)

# ---------------------------------------------------------------------------
hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


def _arg_was_passed(name):
    return any(
        arg == name or arg.startswith(name + "=") for arg in sys.argv[1:]
    )


def _configure_defaults():
    if not _arg_was_passed("--name"):
        hparams.name = "UQ-AV-test"
    if not _arg_was_passed("--test_split_name"):
        hparams.test_split_name = "test_av_split.txt"
    if not _arg_was_passed("--results_dir"):
        hparams.results_dir = os.path.join(
            getattr(hparams, "checkpoints_dir", "./checkpoints"),
            "uq_av_k1_test_results",
        )


def _vocoder_export(mel_completed, mel_target, sample_ids,
                    output_dir, global_step, hparams):
    """Export wav files via Griffin-Lim."""
    from utils.vocoder import griffin_lim_mel_to_audio

    wav_dir = os.path.join(
        output_dir, "wav", f"step{global_step:09d}",
    )
    os.makedirs(wav_dir, exist_ok=True)
    max_samples = getattr(hparams, "vocoder_max_samples", None)
    n_iter = int(getattr(hparams, "vocoder_n_iter", 32))

    for i in range(mel_completed.size(0)):
        if max_samples is not None and i >= max_samples:
            break
        sample_id = sample_ids[i] if i < len(sample_ids) else f"sample_{i}"
        safe_id = "".join(
            c if c.isalnum() or c in "._-" else "_" for c in str(sample_id)
        )

        recon = griffin_lim_mel_to_audio(
            mel_completed[i, 0].cpu().numpy(),
            n_iter=n_iter,
            n_fft=int(getattr(hparams, "fft_size", 1280)),
            hop_length=int(getattr(hparams, "hop_size", 320)),
            win_length=int(getattr(hparams, "fft_size", 1280)),
        )
        target = griffin_lim_mel_to_audio(
            mel_target[i, 0].cpu().numpy(),
            n_iter=n_iter,
            n_fft=int(getattr(hparams, "fft_size", 1280)),
            hop_length=int(getattr(hparams, "hop_size", 320)),
            win_length=int(getattr(hparams, "fft_size", 1280)),
        )

        import soundfile as sf
        sf.write(
            os.path.join(wav_dir, f"{safe_id}_reconstructed.wav"),
            recon, int(getattr(hparams, "sample_rate", 16000)),
        )
        sf.write(
            os.path.join(wav_dir, f"{safe_id}_target.wav"),
            target, int(getattr(hparams, "sample_rate", 16000)),
        )


def main():
    _configure_defaults()

    ae_ckpt = getattr(hparams, "ae_checkpoint", None)
    if ae_ckpt is None:
        raise RuntimeError(
            "--ae_checkpoint is required for P3 testing."
        )

    os.makedirs(hparams.results_dir, exist_ok=True)

    print(f"[UQ-AV test] data_root={hparams.data_root}")
    print(f"[UQ-AV test] results_dir={hparams.results_dir}")

    data_loaders = get_uq_av_data_loaders(
        hparams.data_root,
        {"test": hparams.test_split_name},
        phases=("test",),
        batch_size=hparams.batch_size,
        num_workers=getattr(hparams, "num_workers", 4),
        pin_memory=getattr(hparams, "pin_memory", True),
        shuffle=False,
    )
    data_loader = data_loaders["test"]

    model = UQAVDiffusionModel(hparams, device=device)
    model.load_ae_checkpoint(ae_ckpt)

    if hparams.resume_path:
        step, epoch = model.load_checkpoint(
            hparams.resume_path,
            reset_optimizer=True,
        )
        print(
            f"[UQ-AV test] Loaded checkpoint step={step} epoch={epoch}"
        )
    else:
        raise RuntimeError("--resume_path is required for test-uq-av")

    inference_steps = int(
        getattr(hparams, "uq_inference_steps", 50)
    )

    # Metrics accumulators
    all_records = []
    total = {
        "psnr_full_sum": 0.0, "psnr_missing_sum": 0.0,
        "ssim_full_sum": 0.0, "mel_l1_full_sum": 0.0,
        "mel_l1_missing_sum": 0.0, "known_max_abs_err_sum": 0.0,
    }
    sample_count = 0
    ssim_sample_count = 0

    progress = tqdm(data_loader, desc="[UQ-AV test]", unit="batch",
                    dynamic_ncols=True)
    for batch_idx, batch in enumerate(progress):
        # K=1 DDIM inference
        result = model.sample(
            batch, num_candidates=1,
            inference_steps=inference_steps,
            ddim_eta=0.0,
        )

        mel_completed = result["completed_mels"][:, 0]  # [B, 1, 80, 200]
        mel_pred = result["candidate_mels"][:, 0]

        # Verify known-region preservation
        known_mask = 1.0 - model.missing_mask
        known_error = (
            torch.abs(mel_completed - model.mel_corrupted) * known_mask
        ).max()

        # Compute metrics
        metrics = compute_viai_a_metrics(
            mel_completed, model.mel_target, model.missing_mask,
            compute_ssim=True,
        )
        total["psnr_full_sum"] += metrics["psnr_full_sum"]
        total["psnr_missing_sum"] += metrics["psnr_missing_sum"]
        sample_count += metrics["num_samples"]
        if metrics.get("ssim_full_sum") is not None:
            total["ssim_full_sum"] += metrics["ssim_full_sum"]
            ssim_sample_count += metrics["num_samples"]

        # Per-sample records
        for i in range(mel_completed.size(0)):
            sid = (
                model.sample_ids[i]
                if i < len(model.sample_ids)
                else f"sample_{batch_idx}_{i}"
            )
            spec = (
                model.mask_specs[i]
                if i < len(model.mask_specs)
                else None
            )
            cond = (
                model.video_conditions[i]
                if i < len(model.video_conditions)
                else "unknown"
            )
            record = {
                "sample_id": str(sid),
                "mask_type": getattr(spec, "mask_type", "unknown"),
                "start": int(getattr(spec, "start", -1)),
                "end": int(getattr(spec, "end", -1)),
                "gap_frames": int(getattr(spec, "gap_frames", -1)),
                "video_condition": str(cond),
                "known_region_max_abs_error": float(
                    known_error.cpu().item()
                ),
            }
            all_records.append(record)

        # Mel images (first batch only at reasonable frequency)
        mel_img_dir = os.path.join(
            hparams.results_dir, "mel-image",
            f"step{getattr(model, '_loaded_step', 0):09d}",
        )
        if batch_idx % max(
            1, int(getattr(hparams, "test_image_batch_interval", 10))
        ) == 0:
            os.makedirs(mel_img_dir, exist_ok=True)
            save_mel_comparison_batch(
                mel_img_dir,
                batch_idx * hparams.batch_size,
                model.sample_ids,
                model.mel_corrupted,
                mel_completed,
                model.mel_target,
                missing_mask=model.missing_mask,
            )

    # ---- Summary ----
    n = max(1, sample_count)
    ns = max(1, ssim_sample_count)
    summary = {
        "checkpoint": hparams.resume_path,
        "ae_checkpoint": ae_ckpt,
        "num_samples": sample_count,
        "psnr_full_db": total["psnr_full_sum"] / n,
        "psnr_missing_db": total["psnr_missing_sum"] / n,
        "ssim_full": total["ssim_full_sum"] / ns,
        "inference_steps": inference_steps,
        "ddim_eta": 0.0,
        "num_candidates": 1,
    }

    # Write summary.json
    summary_path = os.path.join(hparams.results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[UQ-AV test] Summary saved to {summary_path}")

    # Write samples.jsonl
    samples_path = os.path.join(
        hparams.results_dir, "samples.jsonl",
    )
    with open(samples_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[UQ-AV test] Sample records saved to {samples_path}")

    # Write metrics.csv
    csv_path = os.path.join(hparams.results_dir, "metrics.csv")
    if all_records:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_records[0].keys())
            writer.writeheader()
            writer.writerows(all_records)
        print(f"[UQ-AV test] Metrics CSV saved to {csv_path}")

    # Optional vocoder export
    if getattr(hparams, "use_vocoder", False):
        print("[UQ-AV test] Exporting wav files via Griffin-Lim ...")
        # Re-run the last batch or a dedicated subset for wav export
        # For simplicity, re-iterate and export up to vocoder_max_samples
        exported = 0
        max_export = int(
            getattr(hparams, "vocoder_max_samples", 20)
        )
        for batch in data_loader:
            result = model.sample(
                batch, num_candidates=1,
                inference_steps=inference_steps,
                ddim_eta=0.0,
            )
            mel_completed = result["completed_mels"][:, 0]
            _vocoder_export(
                mel_completed, model.mel_target, model.sample_ids,
                hparams.results_dir, getattr(model, "_loaded_step", 0),
                hparams,
            )
            exported += mel_completed.size(0)
            if exported >= max_export:
                break

    # Print summary
    print("=" * 60)
    print("[UQ-AV test] Results summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    sys.exit(0)
