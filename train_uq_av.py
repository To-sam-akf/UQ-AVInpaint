"""Train UQ-AV K=1 Latent Diffusion (P3).

Requires a pre-trained frozen Mel autoencoder (P2 checkpoint).

Usage (via main.py):
    python main.py train-uq-av -- \
      --name UQ-AV-K1 \\
      --data_root /path/to/data \\
      --train_split_name train_av_split.txt \\
      --val_split_name val_av_split.txt \\
      --ae_checkpoint checkpoints/mel_ae/MelAE_checkpoint_step000005000.pth.tar \\
      --batch_size 8 --nepochs 100 \\
      --uq_lambda_boundary 0.1

Key defaults:
  - 1000 diffusion timesteps, linear schedule.
  - DDIM inference with 50 steps.
  - K=1 only (multi-candidate in P4).
  - AE is frozen and used only for encode/decode.
"""

import os
import sys
import time
import math
from collections import defaultdict

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import Options_inpainting
from Data_loaders.uq_av_loader import (
    apply_audio_context_dropout_tensor,
    create_uq_av_dataloader,
)
from Models.UQ_AV_Diffusion import UQAVDiffusionModel
from utils.viai_a_metrics import (
    compute_viai_a_metrics,
    compute_inpainting_sample_metrics,
    write_mel_images,
    compose_inpainted_mel,
)


P2_VALIDATION_LABELS = (
    "original",
    "drop_video",
    "drop_audio_original",
    "drop_audio_wrong_video",
    "wrong_video",
    "shuffled_video",
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
        hparams.name = "UQ-AV-K1"
    if not _arg_was_passed("--train_split_name"):
        hparams.train_split_name = "train_av_split.txt"
    if not _arg_was_passed("--val_split_name"):
        hparams.val_split_name = "val_av_split.txt"
    if not _arg_was_passed("--checkpoint_dir"):
        hparams.checkpoint_dir = os.path.join(
            getattr(hparams, "checkpoints_dir", "./checkpoints"),
            "uq_av_k1",
        )
    if not _arg_was_passed("--log_event_path"):
        hparams.log_event_path = os.path.join(
            hparams.checkpoint_dir, "events",
        )


def _uq_condition_probabilities():
    return {
        "audio_video": getattr(hparams, "uq_p_audio_video", 0.4),
        "drop_video": getattr(hparams, "uq_p_drop_video", 0.2),
        "partial_audio_video": getattr(
            hparams, "uq_p_partial_audio_video", 0.2,
        ),
        "wrong_video": getattr(hparams, "uq_p_wrong_video", 0.1),
        "shuffled_video": getattr(hparams, "uq_p_shuffled_video", 0.1),
    }


def _uq_dataset_kwargs(phase):
    enable_dropout = bool(getattr(hparams, "uq_enable_modality_dropout", False))
    if enable_dropout and phase == "val":
        video_conditions = ("original", "wrong_video", "shuffled_video")
    else:
        video_conditions = (getattr(hparams, "uq_video_degradation", "original"),)
    kwargs = {
        "video_conditions": video_conditions,
        "image_size": getattr(hparams, "image_size", 256),
        "seed": getattr(hparams, "eval_seed", 1234),
        "enable_modality_dropout": enable_dropout and phase == "train",
        "condition_probabilities": _uq_condition_probabilities(),
        "audio_context_drop_min_ratio": getattr(
            hparams, "uq_audio_context_drop_min_ratio", 0.15,
        ),
        "audio_context_drop_max_ratio": getattr(
            hparams, "uq_audio_context_drop_max_ratio", 0.35,
        ),
        "condition_override": getattr(hparams, "uq_condition_override", "none"),
        "return_original_video": (
            phase == "train"
            and float(getattr(hparams, "uq_lambda_video_margin", 0.0)) > 0.0
            and not bool(getattr(hparams, "uq_no_video", False))
        ),
    }
    metadata_dir = getattr(hparams, "uq_metadata_dir", None)
    if metadata_dir:
        kwargs["metadata_dir"] = metadata_dir
    return kwargs


def _set_loader_epoch(data_loader, epoch):
    dataset = getattr(data_loader, "dataset", None)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def _fmt_metric(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _validation_sample_seed(batch_index):
    seed = getattr(hparams, "eval_seed", None)
    if seed is None:
        return None
    return int(seed) + int(batch_index)


def _validation_inference_steps():
    steps = getattr(hparams, "uq_val_inference_steps", None)
    if steps is None:
        steps = getattr(hparams, "uq_inference_steps", 50)
    return int(steps)


def _add_scalar(writer, tag, value, step):
    if writer is None or value is None:
        return
    writer.add_scalar(tag, value, step)


def _write_validation_metrics(writer, averages, step):
    _add_scalar(
        writer, "val_denoise/psnr_full",
        averages.get("val_denoise_psnr_full"), step,
    )
    _add_scalar(
        writer, "val_denoise/psnr_missing",
        averages.get("val_denoise_psnr_missing"), step,
    )
    _add_scalar(
        writer, "val_denoise/ssim_full",
        averages.get("val_denoise_ssim_full"), step,
    )
    _add_scalar(
        writer, "val_sample/psnr_full_db",
        averages.get("val_sample_psnr_full_db"), step,
    )
    _add_scalar(
        writer, "val_sample/psnr_missing_db",
        averages.get("val_sample_psnr_missing_db"), step,
    )
    _add_scalar(
        writer, "val_sample/mel_l1_full",
        averages.get("val_sample_mel_l1_full"), step,
    )
    _add_scalar(
        writer, "val_sample/mel_l1_missing",
        averages.get("val_sample_mel_l1_missing"), step,
    )
    _add_scalar(
        writer, "val_sample/ssim_full",
        averages.get("val_sample_ssim_full"), step,
    )
    _add_scalar(
        writer, "val_sample/boundary_l1",
        averages.get("val_sample_boundary_l1"), step,
    )
    _add_scalar(
        writer, "val_sample/known_region_max_abs_error_max",
        averages.get("val_sample_known_region_max_abs_error_max"), step,
    )
    for label in P2_VALIDATION_LABELS:
        prefix = f"val_sample_{label}"
        _add_scalar(
            writer, f"val_sample/{label}/psnr_missing_db",
            averages.get(f"{prefix}_psnr_missing_db"), step,
        )
        _add_scalar(
            writer, f"val_sample/{label}/mel_l1_missing",
            averages.get(f"{prefix}_mel_l1_missing"), step,
        )
        _add_scalar(
            writer, f"val_sample/{label}/ssim_full",
            averages.get(f"{prefix}_ssim_full"), step,
        )
        _add_scalar(
            writer, f"val_sample/{label}/boundary_l1",
            averages.get(f"{prefix}_boundary_l1"), step,
        )


def _empty_sample_totals():
    return {
        "psnr_full_sum": 0.0,
        "psnr_missing_sum": 0.0,
        "mel_l1_full_sum": 0.0,
        "mel_l1_missing_sum": 0.0,
        "ssim_full_sum": 0.0,
        "boundary_l1_sum": 0.0,
        "known_region_max_abs_error_sum": 0.0,
        "known_region_max_abs_error_max": 0.0,
        "sample_count": 0,
        "ssim_sample_count": 0,
    }


def _accumulate_sample_metrics(totals, metrics, indices=None):
    per_sample = metrics["per_sample"]
    if indices is None:
        indices = range(metrics["num_samples"])
    for index in indices:
        totals["psnr_full_sum"] += float(per_sample["psnr_full_db"][index])
        totals["psnr_missing_sum"] += float(
            per_sample["psnr_missing_db"][index]
        )
        totals["mel_l1_full_sum"] += float(per_sample["mel_l1_full"][index])
        totals["mel_l1_missing_sum"] += float(
            per_sample["mel_l1_missing"][index]
        )
        totals["boundary_l1_sum"] += float(per_sample["boundary_l1"][index])
        known = per_sample.get("known_region_max_abs_error")
        if known is not None:
            known_value = float(known[index])
            totals["known_region_max_abs_error_sum"] += known_value
            totals["known_region_max_abs_error_max"] = max(
                totals["known_region_max_abs_error_max"], known_value,
            )
        ssim = per_sample.get("ssim_full", [None])[index]
        if ssim is not None:
            totals["ssim_full_sum"] += float(ssim)
            totals["ssim_sample_count"] += 1
        totals["sample_count"] += 1


def _sample_average_dict(totals, prefix):
    sample_count = totals["sample_count"]
    ssim_count = totals["ssim_sample_count"]
    return {
        f"{prefix}_psnr_full_db": (
            totals["psnr_full_sum"] / max(1, sample_count)
            if sample_count > 0 else None
        ),
        f"{prefix}_psnr_missing_db": (
            totals["psnr_missing_sum"] / max(1, sample_count)
            if sample_count > 0 else None
        ),
        f"{prefix}_mel_l1_full": (
            totals["mel_l1_full_sum"] / max(1, sample_count)
            if sample_count > 0 else None
        ),
        f"{prefix}_mel_l1_missing": (
            totals["mel_l1_missing_sum"] / max(1, sample_count)
            if sample_count > 0 else None
        ),
        f"{prefix}_ssim_full": (
            totals["ssim_full_sum"] / max(1, ssim_count)
            if ssim_count > 0 else None
        ),
        f"{prefix}_boundary_l1": (
            totals["boundary_l1_sum"] / max(1, sample_count)
            if sample_count > 0 else None
        ),
        f"{prefix}_known_region_max_abs_error_mean": (
            totals["known_region_max_abs_error_sum"] / max(1, sample_count)
            if sample_count > 0 else None
        ),
        f"{prefix}_known_region_max_abs_error_max": (
            totals["known_region_max_abs_error_max"]
            if sample_count > 0 else None
        ),
    }


def _copy_batch(batch):
    copied = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            copied[key] = value.clone()
        elif isinstance(value, list):
            copied[key] = list(value)
        else:
            copied[key] = value
    return copied


def _batch_with_conditioning(batch, mode, batch_index):
    copied = _copy_batch(batch)
    copied["conditioning_mode"] = [mode] * len(copied["sample_id"])
    if mode == "drop_audio":
        copied["mel_corrupted"] = apply_audio_context_dropout_tensor(
            copied["mel_corrupted"],
            copied["missing_mask"],
            _validation_sample_seed(batch_index) or batch_index,
            min_ratio=getattr(hparams, "uq_audio_context_drop_min_ratio", 0.15),
            max_ratio=getattr(hparams, "uq_audio_context_drop_max_ratio", 0.35),
        )
    return copied


def _validation_sample_runs(batch, batch_index):
    if not bool(getattr(hparams, "uq_enable_modality_dropout", False)):
        return [("base", batch, ["original"] * len(batch["sample_id"]))]

    video_conditions = list(batch.get("video_condition", []))
    base_labels = []
    for condition in video_conditions:
        if condition == "original":
            base_labels.append("original")
        elif condition == "wrong_video":
            base_labels.append("wrong_video")
        elif condition == "shuffled_video":
            base_labels.append("shuffled_video")
        else:
            base_labels.append(None)

    drop_video_labels = [
        "drop_video" if condition == "original" else None
        for condition in video_conditions
    ]
    drop_audio_labels = []
    for condition in video_conditions:
        if condition == "original":
            drop_audio_labels.append("drop_audio_original")
        elif condition == "wrong_video":
            drop_audio_labels.append("drop_audio_wrong_video")
        else:
            drop_audio_labels.append(None)

    return [
        ("base", batch, base_labels),
        (
            "drop_video",
            _batch_with_conditioning(batch, "drop_video", batch_index),
            drop_video_labels,
        ),
        (
            "drop_audio",
            _batch_with_conditioning(batch, "drop_audio", batch_index),
            drop_audio_labels,
        ),
    ]


def _metric_direction(metric_name):
    lowered = metric_name.lower()
    if any(token in lowered for token in ("loss", "l1", "error", "mse")):
        return "min"
    return "max"


def _metric_value(averages, metric_name):
    if metric_name in averages:
        return averages[metric_name]
    underscored = metric_name.replace("/", "_")
    if underscored in averages:
        return averages[underscored]
    return None


def _is_metric_improved(metric_name, value, best_value, min_delta):
    if value is None:
        return False
    value = float(value)
    if not math.isfinite(value):
        return False
    if best_value is None:
        return True
    direction = _metric_direction(metric_name)
    if direction == "min":
        return value < float(best_value) - float(min_delta)
    return value > float(best_value) + float(min_delta)


def _maybe_update_best_checkpoint(
    model, averages, global_step, global_epoch,
    best_metric_name, best_metric_value, bad_validation_epochs,
):
    metric_value = _metric_value(averages, best_metric_name)
    min_delta = float(getattr(hparams, "uq_early_stop_min_delta", 0.0))
    improved = _is_metric_improved(
        best_metric_name, metric_value, best_metric_value, min_delta,
    )
    if improved:
        best_metric_value = float(metric_value)
        bad_validation_epochs = 0
        best_path = model.save_checkpoint(
            global_step,
            global_epoch,
            hparams.checkpoint_dir,
            filename="UQ-AV_best.pth.tar",
        )
        tqdm.write(
            f"[UQ-AV] New best checkpoint: {best_path} "
            f"{best_metric_name}={best_metric_value:.6f}"
        )
        return best_metric_value, bad_validation_epochs, False

    if metric_value is None:
        tqdm.write(
            f"[UQ-AV] Best metric '{best_metric_name}' was not found in "
            "validation averages; skipping early-stop update."
        )
        return best_metric_value, bad_validation_epochs, False
    if best_metric_value is None or not math.isfinite(float(metric_value)):
        tqdm.write(
            f"[UQ-AV] Best metric '{best_metric_name}' is not finite yet "
            f"(value={metric_value}); skipping early-stop update."
        )
        return best_metric_value, bad_validation_epochs, False

    bad_validation_epochs += 1
    patience = int(getattr(hparams, "uq_early_stop_patience", 10))
    disable_early_stop = bool(getattr(hparams, "uq_disable_early_stop", False))
    tqdm.write(
        f"[UQ-AV] No validation improvement for "
        f"{bad_validation_epochs}/{patience} eval epochs "
        f"({best_metric_name}={float(metric_value):.6f}, "
        f"best={float(best_metric_value):.6f})."
    )
    should_stop = (
        not disable_early_stop
        and patience > 0
        and bad_validation_epochs >= patience
    )
    if should_stop:
        tqdm.write(
            f"[UQ-AV] Early stopping triggered: {best_metric_name} did not "
            f"improve by > {min_delta} for {patience} validation epochs."
        )
    return best_metric_value, bad_validation_epochs, should_stop


def _run_phase(model, data_loader, phase, global_step, writer,
               global_epoch):
    train = phase == "train"
    totals = {
        "loss_total": 0.0, "loss_diff": 0.0,
        "loss_boundary": 0.0, "loss_sync": 0.0,
        "loss_video_margin": 0.0, "loss_distill": 0.0,
        "video_margin_l_original": 0.0,
        "video_margin_l_wrong": 0.0,
        "video_gate_mean": 0.0,
        "video_attn_norm": 0.0,
        "video_token_norm": 0.0,
    }
    denoise_totals = {
        "psnr_full_sum": 0.0,
        "psnr_missing_sum": 0.0,
        "ssim_full_sum": 0.0,
        "sample_count": 0,
        "ssim_sample_count": 0,
    }
    sample_totals = {
        "psnr_full_sum": 0.0,
        "psnr_missing_sum": 0.0,
        "mel_l1_full_sum": 0.0,
        "mel_l1_missing_sum": 0.0,
        "ssim_full_sum": 0.0,
        "boundary_l1_sum": 0.0,
        "known_region_max_abs_error_sum": 0.0,
        "known_region_max_abs_error_max": 0.0,
        "sample_count": 0,
        "ssim_sample_count": 0,
    }
    sample_totals_by_label = {
        label: _empty_sample_totals() for label in P2_VALIDATION_LABELS
    }
    condition_counts = defaultdict(int)
    condition_loss_sums = defaultdict(float)
    condition_total_samples = 0
    batch_count = 0
    stop_training = False

    progress = tqdm(
        data_loader,
        desc=f"[UQ-AV {phase}] epoch={global_epoch + 1}",
        unit="batch", dynamic_ncols=True,
    )
    for batch_index, batch in enumerate(progress):
        iter_start = time.time()

        model.set_input(batch)
        denoise_metrics = None
        sample_metrics = None

        if train:
            model.optimize_parameters(global_step)
            global_step += 1
        else:
            model.test(global_step=global_step)
            denoise_metrics = compute_viai_a_metrics(
                model.mel_pred,
                model.mel_target,
                model.missing_mask,
                compute_ssim=True,
            )
            denoise_totals["psnr_full_sum"] += (
                denoise_metrics["psnr_full_sum"]
            )
            denoise_totals["psnr_missing_sum"] += (
                denoise_metrics["psnr_missing_sum"]
            )
            denoise_totals["sample_count"] += denoise_metrics["num_samples"]
            if denoise_metrics.get("ssim_full_sum") is not None:
                denoise_totals["ssim_full_sum"] += (
                    denoise_metrics["ssim_full_sum"]
                )
                denoise_totals["ssim_sample_count"] += (
                    denoise_metrics["num_samples"]
                )

            for run_name, run_batch, labels in _validation_sample_runs(
                    batch, batch_index):
                result = model.sample(
                    run_batch,
                    num_candidates=1,
                    inference_steps=_validation_inference_steps(),
                    ddim_eta=float(getattr(hparams, "uq_ddim_eta", 0.0)),
                    seed=_validation_sample_seed(batch_index),
                )
                mel_completed = result["completed_mels"][:, 0]
                run_metrics = compute_inpainting_sample_metrics(
                    mel_completed,
                    model.mel_target,
                    model.missing_mask,
                    model.mel_corrupted,
                    compute_ssim=True,
                )
                if sample_metrics is None or run_name == "base":
                    sample_metrics = run_metrics
                for label in P2_VALIDATION_LABELS:
                    indices = [
                        index for index, value in enumerate(labels)
                        if value == label
                    ]
                    if indices:
                        _accumulate_sample_metrics(
                            sample_totals_by_label[label],
                            run_metrics,
                            indices,
                        )

            _accumulate_sample_metrics(
                sample_totals,
                sample_metrics,
                [
                    index for index, value in enumerate(
                        batch.get(
                            "video_condition",
                            ["original"] * len(batch["sample_id"]),
                        )
                    )
                    if value == "original"
                ] if bool(getattr(hparams, "uq_enable_modality_dropout", False))
                else None,
            )

        model.get_loss_items()

        totals["loss_total"] += model.loss_total_item
        totals["loss_diff"] += model.loss_diff_item
        totals["loss_boundary"] += model.loss_boundary_item
        totals["loss_sync"] += model.loss_sync_item
        totals["loss_video_margin"] += model.loss_video_margin_item
        totals["loss_distill"] += model.loss_distill_item
        totals["video_margin_l_original"] += (
            model.video_margin_l_original_item
        )
        totals["video_margin_l_wrong"] += model.video_margin_l_wrong_item
        totals["video_gate_mean"] += model.video_gate_mean_item
        totals["video_attn_norm"] += model.video_attn_norm_item
        totals["video_token_norm"] += model.video_token_norm_item
        for mode, count in getattr(model, "condition_counts", {}).items():
            condition_counts[mode] += int(count)
            condition_loss_sums[mode] += float(
                getattr(model, "condition_loss_sums", {}).get(mode, 0.0)
            )
            condition_total_samples += int(count)

        postfix = {
            "step": global_step,
            "diff": f"{model.loss_diff_item:.4f}",
            "bdy": f"{model.loss_boundary_item:.4f}",
            "dist": f"{model.loss_distill_item:.4f}",
            "vgate": f"{model.video_gate_mean_item:.3f}",
            "vattn": f"{model.video_attn_norm_item:.3f}",
            "vtok": f"{model.video_token_norm_item:.3f}",
        }
        if float(getattr(hparams, "uq_lambda_video_margin", 0.0)) > 0.0:
            postfix["vm"] = f"{model.loss_video_margin_item:.4f}"
            postfix["vm_mode"] = model.video_margin_negative_mode
        if sample_metrics is not None:
            postfix["sample_psnr_miss"] = _fmt_metric(
                sample_metrics.get("psnr_missing_db")
            )
            postfix["sample_l1_miss"] = _fmt_metric(
                sample_metrics.get("mel_l1_missing")
            )
        if denoise_metrics is not None:
            postfix["denoise_psnr_miss"] = _fmt_metric(
                denoise_metrics.get("psnr_missing")
            )
        progress.set_postfix(**postfix)

        if train:
            model.TF_writer(writer, global_step, prefix=phase)

            if (global_step > 0
                    and global_step % getattr(hparams, "tb_image_freq", 500) == 0
                    and model.mel_pred is not None):
                mel_completed = compose_inpainted_mel(
                    model.mel_corrupted, model.mel_pred,
                    model.missing_mask,
                )
                write_mel_images(
                    writer, phase, global_step,
                    model.mel_corrupted, mel_completed, model.mel_target,
                    max_items=getattr(hparams, "tb_image_count", 4),
                )

            if (global_step > 0
                    and global_step % getattr(hparams, "print_freq", 100) == 0):
                elapsed = (time.time() - iter_start) / max(
                    1, hparams.batch_size
                )
                tqdm.write(
                    f"[UQ-AV train] step={global_step} "
                    f"loss={model.loss_total_item:.6f} "
                    f"diff={model.loss_diff_item:.6f} "
                    f"bdy={model.loss_boundary_item:.6f} "
                    f"sync={model.loss_sync_item:.6f} "
                    f"loss_distill="
                    f"{model.loss_distill_item:.6f} "
                    f"loss_video_margin="
                    f"{model.loss_video_margin_item:.6f} "
                    f"video_margin_l_original="
                    f"{model.video_margin_l_original_item:.6f} "
                    f"video_margin_l_wrong="
                    f"{model.video_margin_l_wrong_item:.6f} "
                    f"video_margin_negative_mode="
                    f"{model.video_margin_negative_mode} "
                    f"video_gate_mean="
                    f"{model.video_gate_mean_item:.6f} "
                    f"video_attn_norm="
                    f"{model.video_attn_norm_item:.6f} "
                    f"video_token_norm="
                    f"{model.video_token_norm_item:.6f} "
                    f"lr={model.current_lr:.2e} "
                    f"time={elapsed:.4f}s/sample"
                )

            if (global_step > 0
                    and global_step % getattr(
                        hparams, "checkpoint_interval", 1000
                    ) == 0):
                model.save_checkpoint(
                    global_step, global_epoch, hparams.checkpoint_dir,
                )

        batch_count += 1

        if (train
                and getattr(hparams, "max_train_steps", None) is not None
                and global_step >= hparams.max_train_steps):
            tqdm.write(
                f"Reached max_train_steps={hparams.max_train_steps}"
            )
            stop_training = True
            break

    if batch_count == 0:
        return global_step, stop_training, None

    averages = {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_diff": totals["loss_diff"] / batch_count,
        "loss_boundary": totals["loss_boundary"] / batch_count,
        "loss_sync": totals["loss_sync"] / batch_count,
        "loss_video_margin": totals["loss_video_margin"] / batch_count,
        "loss_distill": totals["loss_distill"] / batch_count,
        "video_margin_l_original": (
            totals["video_margin_l_original"] / batch_count
        ),
        "video_margin_l_wrong": totals["video_margin_l_wrong"] / batch_count,
        "video_gate_mean": totals["video_gate_mean"] / batch_count,
        "video_attn_norm": totals["video_attn_norm"] / batch_count,
        "video_token_norm": totals["video_token_norm"] / batch_count,
    }
    for mode in sorted(condition_counts):
        count = condition_counts[mode]
        averages[f"cond_{mode}_count"] = count
        averages[f"cond_{mode}_ratio"] = (
            count / max(1, condition_total_samples)
        )
        averages[f"cond_{mode}_loss_diff"] = (
            condition_loss_sums[mode] / max(1, count)
        )
    if not train:
        denoise_count = denoise_totals["sample_count"]
        denoise_ssim_count = denoise_totals["ssim_sample_count"]
        averages.update({
            "val_denoise_psnr_full": (
                denoise_totals["psnr_full_sum"] / max(1, denoise_count)
                if denoise_count > 0 else None
            ),
            "val_denoise_psnr_missing": (
                denoise_totals["psnr_missing_sum"] / max(1, denoise_count)
                if denoise_count > 0 else None
            ),
            "val_denoise_ssim_full": (
                denoise_totals["ssim_full_sum"] / max(1, denoise_ssim_count)
                if denoise_ssim_count > 0 else None
            ),
        })
        averages.update(_sample_average_dict(sample_totals, "val_sample"))
        for label, totals_for_label in sample_totals_by_label.items():
            averages.update(
                _sample_average_dict(
                    totals_for_label, f"val_sample_{label}",
                )
            )
        _write_validation_metrics(writer, averages, global_step)
        p2_metric_parts = []
        if bool(getattr(hparams, "uq_enable_modality_dropout", False)):
            for label in P2_VALIDATION_LABELS:
                key = f"val_sample_{label}_psnr_missing_db"
                if averages.get(key) is not None:
                    p2_metric_parts.append(
                        f"{key}={_fmt_metric(averages[key])}"
                    )
        tqdm.write(
            f"[UQ-AV val] "
            f"val_sample_psnr_miss="
            f"{_fmt_metric(averages['val_sample_psnr_missing_db'])} "
            f"val_sample_l1_miss="
            f"{_fmt_metric(averages['val_sample_mel_l1_missing'])} "
            f"val_sample_ssim="
            f"{_fmt_metric(averages['val_sample_ssim_full'])} "
            f"val_sample_boundary_l1="
            f"{_fmt_metric(averages['val_sample_boundary_l1'])} "
            f"val_denoise_psnr_miss="
            f"{_fmt_metric(averages['val_denoise_psnr_missing'])} "
            f"loss={averages['loss_total']:.6f} "
            f"diff={averages['loss_diff']:.6f} "
            f"bdy={averages['loss_boundary']:.6f} "
            f"loss_distill="
            f"{averages['loss_distill']:.6f} "
            f"loss_video_margin="
            f"{averages['loss_video_margin']:.6f} "
            f"video_margin_l_original="
            f"{averages['video_margin_l_original']:.6f} "
            f"video_margin_l_wrong="
            f"{averages['video_margin_l_wrong']:.6f} "
            f"video_gate_mean="
            f"{averages['video_gate_mean']:.6f} "
            f"video_attn_norm="
            f"{averages['video_attn_norm']:.6f} "
            f"video_token_norm="
            f"{averages['video_token_norm']:.6f}"
        )
        if p2_metric_parts:
            tqdm.write("[UQ-AV val p2] " + " ".join(p2_metric_parts))
    else:
        condition_parts = []
        for mode in sorted(condition_counts):
            condition_parts.append(
                f"cond_{mode}_count={averages[f'cond_{mode}_count']} "
                f"cond_{mode}_ratio="
                f"{averages[f'cond_{mode}_ratio']:.4f} "
                f"cond_{mode}_loss_diff="
                f"{averages[f'cond_{mode}_loss_diff']:.6f}"
            )
        tqdm.write(
            f"[UQ-AV train] "
            f"loss={averages['loss_total']:.6f} "
            f"diff={averages['loss_diff']:.6f} "
            f"bdy={averages['loss_boundary']:.6f} "
            f"loss_distill="
            f"{averages['loss_distill']:.6f} "
            f"loss_video_margin="
            f"{averages['loss_video_margin']:.6f} "
            f"video_margin_l_original="
            f"{averages['video_margin_l_original']:.6f} "
            f"video_margin_l_wrong="
            f"{averages['video_margin_l_wrong']:.6f} "
            f"video_gate_mean="
            f"{averages['video_gate_mean']:.6f} "
            f"video_attn_norm="
            f"{averages['video_attn_norm']:.6f} "
            f"video_token_norm="
            f"{averages['video_token_norm']:.6f}"
        )
        if condition_parts:
            tqdm.write("[UQ-AV train p2] " + " ".join(condition_parts))
    return global_step, stop_training, averages


def train_loop(model, data_loaders, writer, start_step=0, start_epoch=0):
    global_step = start_step
    global_epoch = start_epoch
    best_metric_name = getattr(
        hparams, "uq_early_stop_metric", "val_sample_psnr_missing_db"
    )
    best_metric_value = None
    bad_validation_epochs = 0
    while global_epoch < hparams.nepochs:
        should_stop_early = False
        if "train" in data_loaders:
            _set_loader_epoch(data_loaders["train"], global_epoch)
            global_step, stop, _ = _run_phase(
                model, data_loaders["train"], "train",
                global_step, writer, global_epoch,
            )
            if stop:
                global_epoch += 1
                break

        if "val" in data_loaders and global_epoch % max(
            1, int(getattr(hparams, "test_eval_epoch_interval", 1))
        ) == 0:
            with torch.no_grad():
                _, _, val_averages = _run_phase(
                    model, data_loaders["val"], "val",
                    global_step, writer, global_epoch,
                )
            if val_averages is not None:
                (
                    best_metric_value,
                    bad_validation_epochs,
                    should_stop_early,
                ) = _maybe_update_best_checkpoint(
                    model,
                    val_averages,
                    global_step,
                    global_epoch,
                    best_metric_name,
                    best_metric_value,
                    bad_validation_epochs,
                )

        global_epoch += 1
        print(
            f"[UQ-AV] epoch={global_epoch} lr={model.current_lr}"
        )
        if should_stop_early:
            break

    # Final save
    model.save_checkpoint(
        global_step, global_epoch, hparams.checkpoint_dir,
    )


def main():
    _configure_defaults()

    if int(getattr(hparams, "uq_num_candidates", 1)) != 1:
        raise ValueError("P3 train-uq-av only supports --uq_num_candidates 1.")

    split_names = {
        "train": hparams.train_split_name,
        "val": hparams.val_split_name,
    }
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)

    ae_ckpt = getattr(hparams, "ae_checkpoint", None)
    if ae_ckpt is None:
        raise RuntimeError(
            "--ae_checkpoint is required for P3 training. "
            "Point to a trained Mel AE checkpoint from P2."
        )

    print(f"[UQ-AV] data_root={hparams.data_root}")
    print(f"[UQ-AV] checkpoint_dir={hparams.checkpoint_dir}")
    print(f"[UQ-AV] ae_checkpoint={ae_ckpt}")

    data_loaders = {
        phase: create_uq_av_dataloader(
            data_root=hparams.data_root,
            split_name=split_names[phase],
            phase=phase,
            batch_size=hparams.batch_size,
            num_workers=getattr(hparams, "num_workers", 4),
            pin_memory=getattr(hparams, "pin_memory", True),
            **_uq_dataset_kwargs(phase),
        )
        for phase in ("train", "val")
    }

    model = UQAVDiffusionModel(hparams, device=device)
    model.load_ae_checkpoint(ae_ckpt)

    global_step = 0
    global_epoch = 0
    if hparams.resume and hparams.resume_path:
        global_step, global_epoch = model.load_checkpoint(
            hparams.resume_path,
            reset_optimizer=getattr(hparams, "reset_optimizer", False),
        )
        print(
            f"[UQ-AV] resumed step={global_step} epoch={global_epoch}"
        )

    writer = SummaryWriter(log_dir=hparams.log_event_path)
    try:
        train_loop(
            model, data_loaders, writer,
            start_step=global_step, start_epoch=global_epoch,
        )
    except KeyboardInterrupt:
        print("[UQ-AV] Interrupted! Saving checkpoint ...")
        model.save_checkpoint(
            global_step, global_epoch, hparams.checkpoint_dir,
        )
    finally:
        writer.close()
    print("[UQ-AV] Finished.")


if __name__ == "__main__":
    main()
    sys.exit(0)
