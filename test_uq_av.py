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
import torch.nn.functional as F
from tqdm import tqdm

import Options_inpainting
from Data_loaders.uq_av_loader import get_uq_av_data_loaders
from Models.UQ_AV_Diffusion import UQAVDiffusionModel
from networks.uq.mel_autoencoder import time_gradient
from utils.viai_a_metrics import (
    compute_viai_a_metrics,
    compose_inpainted_mel,
    save_mel_comparison_batch,
    structural_similarity_2d,
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


def _uq_dataset_kwargs():
    kwargs = {
        "video_conditions": (getattr(hparams, "uq_video_degradation", "original"),),
        "image_size": getattr(hparams, "image_size", 256),
        "seed": getattr(hparams, "eval_seed", 1234),
    }
    metadata_dir = getattr(hparams, "uq_metadata_dir", None)
    if metadata_dir:
        kwargs["metadata_dir"] = metadata_dir
    return kwargs


def _per_sample_metrics(mel_completed, mel_target, missing_mask, mel_corrupted):
    pred = torch.clamp(mel_completed.detach(), 0.0, 1.0)
    target = torch.clamp(mel_target.detach(), 0.0, 1.0)
    mask = missing_mask.detach().to(device=pred.device, dtype=pred.dtype)
    known_mask = 1.0 - mask

    abs_diff = torch.abs(pred - target)
    sq_diff = (pred - target) ** 2
    full_l1 = abs_diff.mean(dim=(1, 2, 3))
    full_mse = sq_diff.mean(dim=(1, 2, 3))

    missing_count = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    missing_l1 = (abs_diff * mask).sum(dim=(1, 2, 3)) / missing_count
    missing_mse = (sq_diff * mask).sum(dim=(1, 2, 3)) / missing_count
    psnr_full = -10.0 * torch.log10(full_mse.clamp(min=1e-12))
    psnr_missing = -10.0 * torch.log10(missing_mse.clamp(min=1e-12))

    known_error = (
        torch.abs(pred - mel_corrupted.detach()) * known_mask
    ).flatten(1).max(dim=1).values

    grad_pred = time_gradient(pred)
    grad_target = time_gradient(target)
    weight = F.max_pool2d(mask[:, :1], kernel_size=(1, 5),
                          stride=1, padding=(0, 2))
    weight = weight[:, :, :, :grad_pred.size(-1)].clamp(0, 1)
    boundary_count = weight.sum(dim=(1, 2, 3)).clamp(min=1.0)
    boundary_l1 = (
        torch.abs(grad_pred - grad_target) * weight
    ).sum(dim=(1, 2, 3)) / boundary_count

    pred_np = pred.squeeze(1).cpu().numpy()
    target_np = target.squeeze(1).cpu().numpy()
    ssim = [
        structural_similarity_2d(pred_np[index], target_np[index])
        for index in range(pred.size(0))
    ]

    return {
        "mel_l1_full": full_l1.cpu().tolist(),
        "mel_l1_missing": missing_l1.cpu().tolist(),
        "psnr_full_db": psnr_full.cpu().tolist(),
        "psnr_missing_db": psnr_missing.cpu().tolist(),
        "boundary_l1": boundary_l1.cpu().tolist(),
        "known_region_max_abs_error": known_error.cpu().tolist(),
        "ssim_full": ssim,
    }


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

    num_candidates = int(getattr(hparams, "uq_num_candidates", 1))
    if num_candidates != 1:
        raise ValueError("P3 test-uq-av only supports --uq_num_candidates 1.")

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
        **_uq_dataset_kwargs(),
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
    ddim_eta = float(getattr(hparams, "uq_ddim_eta", 0.0))

    # Metrics accumulators
    all_records = []
    total = {
        "psnr_full_sum": 0.0, "psnr_missing_sum": 0.0,
        "ssim_full_sum": 0.0, "mel_l1_full_sum": 0.0,
        "mel_l1_missing_sum": 0.0, "known_max_abs_err_sum": 0.0,
        "known_max_abs_err_max": 0.0, "boundary_l1_sum": 0.0,
    }
    sample_count = 0
    ssim_sample_count = 0

    progress = tqdm(data_loader, desc="[UQ-AV test]", unit="batch",
                    dynamic_ncols=True)
    for batch_idx, batch in enumerate(progress):
        # K=1 DDIM inference
        result = model.sample(
            batch, num_candidates=num_candidates,
            inference_steps=inference_steps,
            ddim_eta=ddim_eta,
        )

        mel_completed = result["completed_mels"][:, 0]  # [B, 1, 80, 200]
        mel_pred = result["candidate_mels"][:, 0]

        # Compute metrics
        metrics = compute_viai_a_metrics(
            mel_completed, model.mel_target, model.missing_mask,
            compute_ssim=True,
        )
        sample_metrics = _per_sample_metrics(
            mel_completed,
            model.mel_target,
            model.missing_mask,
            model.mel_corrupted,
        )
        total["psnr_full_sum"] += metrics["psnr_full_sum"]
        total["psnr_missing_sum"] += metrics["psnr_missing_sum"]
        total["mel_l1_full_sum"] += sum(sample_metrics["mel_l1_full"])
        total["mel_l1_missing_sum"] += sum(sample_metrics["mel_l1_missing"])
        total["known_max_abs_err_sum"] += sum(
            sample_metrics["known_region_max_abs_error"]
        )
        total["known_max_abs_err_max"] = max(
            total["known_max_abs_err_max"],
            max(sample_metrics["known_region_max_abs_error"]),
        )
        total["boundary_l1_sum"] += sum(sample_metrics["boundary_l1"])
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
                "mel_l1_full": float(sample_metrics["mel_l1_full"][i]),
                "mel_l1_missing": float(sample_metrics["mel_l1_missing"][i]),
                "psnr_full_db": float(sample_metrics["psnr_full_db"][i]),
                "psnr_missing_db": float(sample_metrics["psnr_missing_db"][i]),
                "ssim_full": float(sample_metrics["ssim_full"][i]),
                "boundary_l1": float(sample_metrics["boundary_l1"][i]),
                "known_region_max_abs_error": float(
                    sample_metrics["known_region_max_abs_error"][i]
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
        "mel_l1_full": total["mel_l1_full_sum"] / n,
        "mel_l1_missing": total["mel_l1_missing_sum"] / n,
        "ssim_full": total["ssim_full_sum"] / ns,
        "boundary_l1": total["boundary_l1_sum"] / n,
        "known_region_max_abs_error_mean": total["known_max_abs_err_sum"] / n,
        "known_region_max_abs_error_max": total["known_max_abs_err_max"],
        "video_degradation": getattr(hparams, "uq_video_degradation", "original"),
        "inference_steps": inference_steps,
        "ddim_eta": ddim_eta,
        "num_candidates": num_candidates,
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
                batch, num_candidates=num_candidates,
                inference_steps=inference_steps,
                ddim_eta=ddim_eta,
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
