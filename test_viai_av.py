import csv
import json
import os
import re
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel
from utils import util
from utils.viai_a_metrics import (
    compose_inpainted_mel,
    compute_calibration_bins,
    compute_multi_candidate_metrics,
    compute_risk_coverage_curve,
    compute_viai_a_metrics,
    save_mel_comparison_batch,
)
from utils.vocoder import save_vocoder_batch
from utils.semantic_evidence import infer_instrument_from_sample_dir as instrument_from_sample_dir
from utils.wrong_video_sampler import WrongVideoSampler, wrong_video_effective_mode


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


RESULT_FIELDS = [
    "checkpoint_path",
    "checkpoint_step",
    "global_step",
    "global_epoch",
    "test_split_name",
    "stage",
    "use_gan",
    "enable_sync_loss",
    "enable_probe_loss",
    "enable_ec_viai_av",
    "num_candidates",
    "test_num_candidates",
    "stochastic_adapter",
    "deterministic_adapter",
    "enable_evidence_gate",
    "freeze_gate_evidence_backbone",
    "enable_evidence_scaled_sigma",
    "enable_candidate_scorer",
    "train_candidate_heads_only",
    "evidence_source",
    "semantic_evidence_path",
    "semantic_evidence_weight",
    "semantic_missing_score",
    "enable_visual_evidence_aug",
    "visual_evidence_aug_prob",
    "visual_evidence_aug_modes",
    "sigma_min",
    "sigma_max",
    "evidence_sigma_scale_min",
    "evidence_sigma_scale_max",
    "calib_error_tau",
    "save_candidates",
    "video_perturbation",
    "wrong_video_effective_mode",
    "wrong_video_cross_instrument_available",
    "wrong_video_num_instruments",
    "video_blur_kernel",
    "video_frame_drop_stride",
    "video_temporal_shift_frames",
    "calibration_bins",
    "num_samples",
    "loss_total",
    "loss_av_gen",
    "loss_recon",
    "loss_g_gan",
    "loss_sync",
    "loss_probe_gen",
    "loss_probe_recon",
    "loss_probe_g_gan",
    "loss_d",
    "loss_anchor",
    "loss_min_k",
    "loss_mean_k",
    "loss_boundary",
    "loss_evidence_div",
    "loss_gate_evidence",
    "loss_candidate_scorer",
    "loss_uncertainty_calib",
    "loss_calib",
    "loss_multi_candidate",
    "weighted_loss_min_k",
    "weighted_loss_mean_k",
    "weighted_loss_boundary",
    "weighted_loss_evidence_div",
    "weighted_loss_gate_evidence",
    "weighted_loss_calib",
    "best_of_k_missing_l1",
    "mean_k_missing_l1",
    "top1_missing_l1",
    "candidate0_missing_l1",
    "random_expected_missing_l1",
    "oracle_gain",
    "uncertainty_mean",
    "uncertainty_error_corr",
    "uncertainty_error_spearman",
    "uncertainty_best_error_corr",
    "uncertainty_best_error_spearman",
    "uncertainty_corr_count",
    "candidate_pairwise_distance",
    "candidate_pairwise_mel_l1",
    "boundary_delta_error_top1",
    "boundary_delta_error_best",
    "boundary_delta_error_mean",
    "evidence_mean",
    "evidence_min",
    "evidence_max",
    "heuristic_evidence_mean",
    "semantic_evidence_mean",
    "evidence_diversity_gap",
    "gate_mean",
    "gate_target_mean",
    "gate_target_gap",
    "adapter_sigma_mean",
    "adapter_sigma_scale_mean",
    "adapter_effective_sigma_mean",
    "eta1",
    "eta2",
    "lambda_recon",
    "lambda_min_k",
    "lambda_mean_k",
    "lambda_boundary",
    "lambda_diversity",
    "lambda_calib",
    "lambda_gate_evidence",
    "evidence_diversity_d_min",
    "evidence_diversity_alpha",
    "evidence_gate_low",
    "evidence_gate_high",
    "mel_l1_full",
    "mel_l1_missing",
    "probe_l1_full",
    "probe_l1_missing",
    "psnr_full",
    "psnr_missing",
    "ssim",
    "retrieval_audio_to_video_r1",
    "retrieval_audio_to_video_r5",
    "retrieval_audio_to_video_r10",
    "retrieval_audio_to_video_r50",
    "retrieval_audio_to_video_medr",
    "retrieval_audio_to_video_meanr",
    "retrieval_video_to_audio_r1",
    "retrieval_video_to_audio_r5",
    "retrieval_video_to_audio_r10",
    "retrieval_video_to_audio_r50",
    "retrieval_video_to_audio_medr",
    "retrieval_video_to_audio_meanr",
    "use_vocoder",
    "vocoder_backend",
    "vocoder_checkpoint",
    "vocoder_splice_missing",
    "vocoder_crossfade_ms",
    "vocoder_n_iter",
    "vocoder_output_dir",
    "vocoder_num_samples",
    "candidate_image_dir",
    "candidate_vocoder_dir",
    "candidate_vocoder_num_samples",
    "per_sample_metrics_path",
    "risk_coverage_path",
    "calibration_bins_path",
]


def _arg_was_passed(name):
    return any(arg == name or arg.startswith(name + "=") for arg in sys.argv[1:])


def configure_viai_av_defaults():
    if not _arg_was_passed("--name"):
        if getattr(hparams, "enable_ec_viai_av", False):
            hparams.name = (
                "EC-VIAI-AV-PatchGAN"
                if getattr(hparams, "use_gan", False)
                else "EC-VIAI-AV"
            )
        else:
            hparams.name = (
                "VIAI-AV-PatchGAN"
                if getattr(hparams, "use_gan", False)
                else "VIAI-AV"
            )
    if not _arg_was_passed("--results_dir"):
        if getattr(hparams, "enable_ec_viai_av", False):
            hparams.results_dir = (
                "./checkpoints/ec_viai_av_patchgan_test_results"
                if getattr(hparams, "use_gan", False)
                else "./checkpoints/ec_viai_av_test_results"
            )
        else:
            hparams.results_dir = (
                "./checkpoints/viai_av_patchgan_test_results"
                if getattr(hparams, "use_gan", False)
                else "./checkpoints/viai_av_test_results"
            )


def print_viai_av_test_config():
    print(
        "[VIAI-AV test] run config: "
        f"name={hparams.name} "
        f"use_gan={getattr(hparams, 'use_gan', False)} "
        f"lambda_recon={getattr(hparams, 'lambda_recon', 1.0)} "
        f"lambda_gan={getattr(hparams, 'lambda_gan', 1.0)} "
        f"lambda_sync={getattr(hparams, 'lambda_sync', 1.0)} "
        f"lambda_probe={getattr(hparams, 'lambda_probe', 1.0)} "
        f"disable_sync_loss={getattr(hparams, 'disable_sync_loss', False)} "
        f"disable_probe_loss={getattr(hparams, 'disable_probe_loss', False)}"
    )
    print(
        "[VIAI-AV test] EC config: "
        f"enable_ec_viai_av={getattr(hparams, 'enable_ec_viai_av', False)} "
        f"num_candidates={getattr(hparams, 'num_candidates', 1)} "
        f"test_num_candidates={getattr(hparams, 'test_num_candidates', 1)} "
        f"stochastic_adapter={getattr(hparams, 'stochastic_adapter', False)} "
        f"deterministic_adapter={getattr(hparams, 'deterministic_adapter', False)} "
        f"enable_candidate_scorer={getattr(hparams, 'enable_candidate_scorer', False)} "
        f"train_candidate_heads_only={getattr(hparams, 'train_candidate_heads_only', False)} "
        f"save_candidates={getattr(hparams, 'save_candidates', False)} "
        f"video_perturbation={getattr(hparams, 'video_perturbation', 'none')} "
        f"video_blur_kernel={getattr(hparams, 'video_blur_kernel', 9)} "
        f"video_frame_drop_stride={getattr(hparams, 'video_frame_drop_stride', 2)} "
        f"video_temporal_shift_frames={getattr(hparams, 'video_temporal_shift_frames', 6)} "
        f"calibration_bins={getattr(hparams, 'calibration_bins', 10)}"
    )
    print(
        "[VIAI-AV test] EC losses/evidence: "
        f"lambda_min_k={getattr(hparams, 'lambda_min_k', 0.0)} "
        f"lambda_mean_k={getattr(hparams, 'lambda_mean_k', 0.0)} "
        f"lambda_boundary={getattr(hparams, 'lambda_boundary', 0.0)} "
        f"lambda_diversity={getattr(hparams, 'lambda_diversity', 0.0)} "
        f"lambda_calib={getattr(hparams, 'lambda_calib', 0.0)} "
        f"calib_error_tau={getattr(hparams, 'calib_error_tau', 0.1)} "
        f"lambda_gate_evidence={getattr(hparams, 'lambda_gate_evidence', 0.0)} "
        f"enable_evidence_gate={getattr(hparams, 'enable_evidence_gate', False)} "
        f"freeze_gate_evidence_backbone={getattr(hparams, 'freeze_gate_evidence_backbone', False)} "
        f"enable_evidence_scaled_sigma={getattr(hparams, 'enable_evidence_scaled_sigma', False)} "
        f"evidence_sigma_scale_min={getattr(hparams, 'evidence_sigma_scale_min', 0.5)} "
        f"evidence_sigma_scale_max={getattr(hparams, 'evidence_sigma_scale_max', 2.0)} "
        f"evidence_source={getattr(hparams, 'evidence_source', 'none')} "
        f"semantic_evidence_path={getattr(hparams, 'semantic_evidence_path', '')} "
        f"semantic_evidence_weight={getattr(hparams, 'semantic_evidence_weight', 0.35)} "
        f"semantic_missing_score={getattr(hparams, 'semantic_missing_score', 0.0)} "
        f"evidence_diversity_d_min={getattr(hparams, 'evidence_diversity_d_min', 0.02)} "
        f"evidence_diversity_alpha={getattr(hparams, 'evidence_diversity_alpha', 0.08)} "
        f"evidence_gate_low={getattr(hparams, 'evidence_gate_low', 0.24)} "
        f"evidence_gate_high={getattr(hparams, 'evidence_gate_high', 0.34)} "
        f"enable_visual_evidence_aug={getattr(hparams, 'enable_visual_evidence_aug', False)} "
        f"visual_evidence_aug_prob={getattr(hparams, 'visual_evidence_aug_prob', 0.5)} "
        f"visual_evidence_aug_modes={getattr(hparams, 'visual_evidence_aug_modes', '')} "
        f"sigma_min={getattr(hparams, 'sigma_min', 0.0)} "
        f"sigma_max={getattr(hparams, 'sigma_max', 1.0)}"
    )
    print(
        "[VIAI-AV test] paths: "
        f"checkpoint_dir={hparams.checkpoint_dir} "
        f"resume_path={hparams.resume_path} "
        f"results_dir={hparams.results_dir}"
    )


def checkpoint_step(path):
    match = re.search(r"checkpoint_step(\d+)", os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def resolve_checkpoint_path(resume_path, checkpoint_dir, name):
    search_dirs = []
    if resume_path is not None:
        candidate = os.path.abspath(resume_path)
        if os.path.exists(candidate):
            return candidate
        search_dirs.append(os.path.dirname(candidate))
    search_dirs.append(os.path.abspath(checkpoint_dir))

    candidates = []
    for directory in search_dirs:
        if not directory or not os.path.isdir(directory):
            continue
        for filename in os.listdir(directory):
            if filename.startswith(f"{name}_checkpoint_step") and filename.endswith(".pth.tar"):
                candidates.append(os.path.join(directory, filename))
    if not candidates:
        raise RuntimeError(
            "No VIAI-AV checkpoint found. Pass --resume_path or place a "
            f"{name}_checkpoint_step*.pth.tar file under {checkpoint_dir}."
        )
    return sorted(candidates, key=lambda path: (checkpoint_step(path), os.path.getmtime(path)))[-1]


def format_step(step):
    if step is None or step < 0:
        return "unknown"
    return f"{step:09d}"


def batch_metrics(model):
    mel_completed = compose_inpainted_mel(
        model.mel_input_4d,
        model.mel_pred,
        model.missing_mask,
    )
    metrics = compute_viai_a_metrics(
        mel_completed,
        model.mel_target_4d,
        model.missing_mask,
        compute_ssim=True,
    )
    return {
        "full_psnr": metrics["psnr_full_sum"],
        "missing_psnr": metrics["psnr_missing_sum"],
        "ssim": metrics["ssim_full_sum"],
        "num_samples": metrics["num_samples"],
    }


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < sorted_values.shape[0]:
        end = start + 1
        while end < sorted_values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _safe_corr(x_values, y_values, spearman=False):
    x = np.asarray(x_values, dtype=np.float64).reshape(-1)
    y = np.asarray(y_values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    count = int(x.shape[0])
    if count < 2:
        return 0.0, count
    if spearman:
        x = _rankdata(x)
        y = _rankdata(y)
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0, count
    return float(np.corrcoef(x, y)[0, 1]), count


def uncertainty_correlations(uncertainty_values, top1_errors, best_errors):
    if not uncertainty_values:
        return {
            "uncertainty_error_corr": 0.0,
            "uncertainty_error_spearman": 0.0,
            "uncertainty_best_error_corr": 0.0,
            "uncertainty_best_error_spearman": 0.0,
            "uncertainty_corr_count": 0,
        }
    uncertainty = np.concatenate(uncertainty_values, axis=0)
    top1 = np.concatenate(top1_errors, axis=0)
    best = np.concatenate(best_errors, axis=0)
    pearson_top1, count = _safe_corr(uncertainty, top1, spearman=False)
    spearman_top1, _ = _safe_corr(uncertainty, top1, spearman=True)
    pearson_best, _ = _safe_corr(uncertainty, best, spearman=False)
    spearman_best, _ = _safe_corr(uncertainty, best, spearman=True)
    return {
        "uncertainty_error_corr": pearson_top1,
        "uncertainty_error_spearman": spearman_top1,
        "uncertainty_best_error_corr": pearson_best,
        "uncertainty_best_error_spearman": spearman_best,
        "uncertainty_corr_count": count,
    }


def safe_token(value):
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return token if token else "none"


def perturbation_mode():
    return getattr(hparams, "video_perturbation", "none")


def perturbation_tag():
    return f"perturb-{safe_token(perturbation_mode())}"


def mel_image_output_dir(results_dir, checkpoint_step_value):
    base_dir = os.path.join(
        results_dir,
        "mel-image",
        f"step{format_step(checkpoint_step_value)}",
    )
    mode = perturbation_mode()
    if mode == "none":
        return base_dir
    return os.path.join(base_dir, perturbation_tag())


def candidate_image_output_dir(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "mel-candidates",
        f"step{format_step(checkpoint_step_value)}",
        perturbation_tag(),
    )


def candidate_vocoder_output_dir(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "wav-candidates",
        f"step{format_step(checkpoint_step_value)}",
        perturbation_tag(),
    )


def per_sample_metrics_path(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "sample-metrics",
        f"step{format_step(checkpoint_step_value)}_{perturbation_tag()}_samples.csv",
    )


def risk_coverage_path(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "risk-coverage",
        f"step{format_step(checkpoint_step_value)}_{perturbation_tag()}_risk_coverage.csv",
    )


def calibration_bins_path(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "calibration",
        f"step{format_step(checkpoint_step_value)}_{perturbation_tag()}_bins.csv",
    )


def blur_video_batch(video_batch, kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return video_batch
    batch_size, frames, channels, height, width = video_batch.shape
    flat = video_batch.reshape(batch_size * frames, channels, height, width)
    blurred = F.avg_pool2d(
        flat,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )
    return blurred.reshape(batch_size, frames, channels, height, width)


def frame_drop_batch(video_batch, flow_batch, stride):
    stride = max(1, int(stride))
    if stride <= 1:
        return video_batch, flow_batch
    frames = int(video_batch.size(1))
    frame_indices = torch.arange(frames, device=video_batch.device)
    source_indices = (frame_indices // stride) * stride
    source_indices = torch.clamp(source_indices, max=frames - 1)
    dropped_video = video_batch.index_select(1, source_indices)
    dropped_flow = flow_batch.clone()
    drop_mask = (frame_indices % stride) != 0
    if bool(drop_mask.any()):
        dropped_flow[:, drop_mask] = 0.0
    return dropped_video, dropped_flow


def temporal_shift_batch(video_batch, flow_batch, shift_frames):
    shift_frames = int(shift_frames)
    if shift_frames <= 0:
        return video_batch, flow_batch
    frames = int(video_batch.size(1))
    shift = min(shift_frames, frames)
    shifted_video = video_batch.clone()
    shifted_flow = torch.zeros_like(flow_batch)
    if shift < frames:
        shifted_video[:, shift:] = video_batch[:, : frames - shift]
        shifted_flow[:, shift:] = flow_batch[:, : frames - shift]
    shifted_video[:, :shift] = video_batch[:, :1].expand(-1, shift, -1, -1, -1)
    return shifted_video, shifted_flow


def apply_test_video_perturbation(model, wrong_video_sampler=None):
    mode = perturbation_mode()
    if mode == "none":
        return
    if mode == "blur":
        model.video_batch = blur_video_batch(
            model.video_batch,
            getattr(hparams, "video_blur_kernel", 9),
        )
        return
    if mode == "flow_zero":
        model.flow_batch = torch.zeros_like(model.flow_batch)
        return
    if mode == "frame_drop":
        model.video_batch, model.flow_batch = frame_drop_batch(
            model.video_batch,
            model.flow_batch,
            getattr(hparams, "video_frame_drop_stride", 2),
        )
        return
    if mode == "temporal_shift":
        model.video_batch, model.flow_batch = temporal_shift_batch(
            model.video_batch,
            model.flow_batch,
            getattr(hparams, "video_temporal_shift_frames", 6),
        )
        return
    if mode in {"wrong_video", "wrong_video_any", "wrong_video_cross_instrument"}:
        if wrong_video_sampler is None:
            raise RuntimeError("wrong_video perturbation requires a WrongVideoSampler.")
        model.video_batch, model.flow_batch = wrong_video_sampler.load_batch(
            model.path_batch,
            model.video_batch,
            model.flow_batch,
        )
        model.set_semantic_evidence_paths(
            wrong_video_sampler.last_wrong_dirs,
            target_instruments=wrong_video_sampler.last_source_instruments,
        )
        return
    if mode == "no_video":
        model.video_batch = torch.zeros_like(model.video_batch)
        model.flow_batch = torch.zeros_like(model.flow_batch)
        model.set_semantic_evidence_override(0.0)
        return
    raise RuntimeError(f"Unsupported video perturbation: {mode}")


def write_table(path, rows, fieldnames=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return path


def _to_flat_np(tensor):
    return tensor.detach().cpu().numpy().reshape(-1)


def _mean_per_sample(tensor):
    return tensor.detach().reshape(tensor.size(0), -1).mean(dim=1).cpu().numpy()


def collect_per_sample_rows(
    model,
    candidate_metrics,
    start_index,
    wrong_video_sampler=None,
):
    rows = []
    candidate_missing = candidate_metrics["candidate_missing_l1"].detach().cpu().numpy()
    candidate_pi = model.candidate_pi.detach().cpu().numpy()
    top1_indices = candidate_metrics["top1_indices"].detach().cpu().numpy()
    best_indices = np.argmin(candidate_missing, axis=1)
    uncertainty = _to_flat_np(model.uncertainty_score)
    evidence = _mean_per_sample(model.evidence_score)
    heuristic_evidence = _mean_per_sample(model.heuristic_evidence_score)
    semantic_evidence = _mean_per_sample(model.semantic_evidence_score)
    gate = _mean_per_sample(model.gate_value)
    sigma_scale = _mean_per_sample(model.adapter_sigma_scale)
    pairwise = _to_flat_np(candidate_metrics["candidate_pairwise_mel_l1_per_sample"])
    top1_error = _to_flat_np(candidate_metrics["top1_missing_l1_per_sample"])
    best_error = _to_flat_np(candidate_metrics["best_of_k_missing_l1_per_sample"])
    mean_error = _to_flat_np(candidate_metrics["mean_k_missing_l1_per_sample"])
    oracle_gain = _to_flat_np(candidate_metrics["oracle_gain_per_sample"])
    boundary_top1 = _to_flat_np(
        candidate_metrics.get(
            "top1_boundary_delta_error_per_sample",
            torch.zeros_like(candidate_metrics["top1_missing_l1_per_sample"]),
        )
    )
    boundary_best = _to_flat_np(
        candidate_metrics.get(
            "best_boundary_delta_error_per_sample",
            torch.zeros_like(candidate_metrics["top1_missing_l1_per_sample"]),
        )
    )
    boundary_mean = _to_flat_np(
        candidate_metrics.get(
            "mean_boundary_delta_error_per_sample",
            torch.zeros_like(candidate_metrics["top1_missing_l1_per_sample"]),
        )
    )

    for index, sample_path in enumerate(model.path_batch):
        semantic_target_instrument = ""
        if index < len(getattr(model, "semantic_evidence_target_instruments", [])):
            semantic_target_instrument = model.semantic_evidence_target_instruments[index] or ""
        row = {
            "sample_index": int(start_index + index),
            "sample_path": sample_path,
            "video_perturbation": perturbation_mode(),
            "top1_index": int(top1_indices[index]),
            "best_index": int(best_indices[index]),
            "uncertainty": float(uncertainty[index]),
            "evidence": float(evidence[index]),
            "heuristic_evidence": float(heuristic_evidence[index]),
            "semantic_evidence": float(semantic_evidence[index]),
            "semantic_target_instrument": semantic_target_instrument,
            "gate": float(gate[index]),
            "sigma_scale": float(sigma_scale[index]),
            "top1_missing_l1": float(top1_error[index]),
            "best_of_k_missing_l1": float(best_error[index]),
            "mean_k_missing_l1": float(mean_error[index]),
            "oracle_gain": float(oracle_gain[index]),
            "candidate_pairwise_mel_l1": float(pairwise[index]),
            "boundary_delta_error_top1": float(boundary_top1[index]),
            "boundary_delta_error_best": float(boundary_best[index]),
            "boundary_delta_error_mean": float(boundary_mean[index]),
        }
        if wrong_video_sampler is not None and index < len(wrong_video_sampler.last_wrong_dirs):
            row.update(
                {
                    "source_instrument": wrong_video_sampler.last_source_instruments[index],
                    "wrong_video_instrument": wrong_video_sampler.last_wrong_instruments[index],
                    "wrong_video_sample_path": wrong_video_sampler.last_wrong_dirs[index],
                    "wrong_video_is_cross_instrument": bool(
                        wrong_video_sampler.last_is_cross_instrument[index]
                    ),
                }
            )
        else:
            row.update(
                {
                    "source_instrument": "",
                    "wrong_video_instrument": "",
                    "wrong_video_sample_path": "",
                    "wrong_video_is_cross_instrument": "",
                }
            )
        for candidate_index in range(candidate_missing.shape[1]):
            row[f"candidate_{candidate_index:02d}_missing_l1"] = float(
                candidate_missing[index, candidate_index]
            )
            row[f"candidate_{candidate_index:02d}_pi"] = float(
                candidate_pi[index, candidate_index]
            )
        rows.append(row)
    return rows


def save_candidate_mel_batches(output_dir, start_index, model):
    if output_dir is None:
        return ""
    num_candidates = int(model.mel_candidates.size(1))
    for candidate_index in range(num_candidates):
        candidate_dir = os.path.join(output_dir, f"candidate_{candidate_index:02d}")
        save_mel_comparison_batch(
            candidate_dir,
            start_index,
            model.path_batch,
            model.mel_input_4d,
            model.mel_candidates[:, candidate_index],
            model.mel_target_4d,
            model.missing_mask,
        )
    return output_dir


def save_candidate_vocoder_batches(output_dir, start_index, model, max_items=None):
    if output_dir is None:
        return 0
    batch_size = int(model.mel_candidates.size(0))
    if max_items is not None:
        batch_size = min(batch_size, max(0, int(max_items)))
    if batch_size <= 0:
        return 0
    num_candidates = int(model.mel_candidates.size(1))
    for candidate_index in range(num_candidates):
        candidate_dir = os.path.join(output_dir, f"candidate_{candidate_index:02d}")
        save_vocoder_batch(
            candidate_dir,
            start_index,
            model.path_batch,
            model.mel_input_4d,
            model.mel_candidates[:, candidate_index],
            model.missing_mask,
            model.audio_target,
            hparams,
            backend=getattr(hparams, "vocoder_backend", "griffin_lim"),
            n_iter=getattr(hparams, "vocoder_n_iter", 32),
            checkpoint_path=getattr(hparams, "vocoder_checkpoint", None),
            splice_missing=getattr(hparams, "vocoder_splice_missing", True),
            crossfade_ms=getattr(hparams, "vocoder_crossfade_ms", 20.0),
            max_items=batch_size,
        )
    return batch_size


def evaluate(
    model,
    data_loader,
    global_step=0,
    image_dir=None,
    vocoder_dir=None,
    candidate_image_dir=None,
    candidate_vocoder_dir=None,
    per_sample_path=None,
    risk_path=None,
    calibration_path=None,
    wrong_video_sampler=None,
):
    totals = {
        "loss_total": 0.0,
        "loss_av_gen": 0.0,
        "loss_recon": 0.0,
        "loss_g_gan": 0.0,
        "loss_sync": 0.0,
        "loss_probe_gen": 0.0,
        "loss_probe_recon": 0.0,
        "loss_probe_g_gan": 0.0,
        "loss_d": 0.0,
        "loss_anchor": 0.0,
        "loss_min_k": 0.0,
        "loss_mean_k": 0.0,
        "loss_boundary": 0.0,
        "loss_evidence_div": 0.0,
        "loss_gate_evidence": 0.0,
        "loss_candidate_scorer": 0.0,
        "loss_uncertainty_calib": 0.0,
        "loss_calib": 0.0,
        "loss_multi_candidate": 0.0,
        "weighted_loss_min_k": 0.0,
        "weighted_loss_mean_k": 0.0,
        "weighted_loss_boundary": 0.0,
        "weighted_loss_evidence_div": 0.0,
        "weighted_loss_gate_evidence": 0.0,
        "weighted_loss_calib": 0.0,
        "heuristic_evidence_mean": 0.0,
        "semantic_evidence_mean": 0.0,
        "evidence_diversity_gap": 0.0,
        "gate_mean": 0.0,
        "gate_target_mean": 0.0,
        "gate_target_gap": 0.0,
        "adapter_sigma_mean": 0.0,
        "adapter_sigma_scale_mean": 0.0,
        "adapter_effective_sigma_mean": 0.0,
        "eta1": 0.0,
        "eta2": 0.0,
        "full_l1": 0.0,
        "missing_l1": 0.0,
        "probe_full_l1": 0.0,
        "probe_missing_l1": 0.0,
        "full_psnr": 0.0,
        "missing_psnr": 0.0,
        "ssim": 0.0,
    }
    sample_totals = {
        "best_of_k_missing_l1": 0.0,
        "mean_k_missing_l1": 0.0,
        "top1_missing_l1": 0.0,
        "candidate0_missing_l1": 0.0,
        "random_expected_missing_l1": 0.0,
        "oracle_gain": 0.0,
        "candidate_pairwise_mel_l1": 0.0,
        "boundary_delta_error_top1": 0.0,
        "boundary_delta_error_best": 0.0,
        "boundary_delta_error_mean": 0.0,
    }
    sample_count = 0
    batch_count = 0
    skipped_batches = 0
    vocoder_count = 0
    candidate_vocoder_count = 0
    vocoder_max_samples = getattr(hparams, "vocoder_max_samples", None)
    audio_embeddings = []
    video_embeddings = []
    uncertainty_values = []
    top1_error_values = []
    best_error_values = []
    oracle_gain_values = []
    evidence_values = []
    pairwise_values = []
    per_sample_rows = []

    progress = tqdm(
        data_loader,
        desc="[VIAI-AV test] evaluating",
        unit="batch",
        dynamic_ncols=True,
    )
    for data in progress:
        if data is None:
            skipped_batches += 1
            progress.set_postfix(skipped_batches=skipped_batches)
            continue
        model.get_blank_space_length(0)
        model.set_inputs(data)
        apply_test_video_perturbation(model, wrong_video_sampler=wrong_video_sampler)
        model.test(global_step=global_step)
        model.get_loss_items()
        metrics = batch_metrics(model)
        candidate_metrics = compute_multi_candidate_metrics(
            model.mel_candidates,
            model.mel_completed_candidates,
            model.mel_target_4d,
            model.missing_mask,
            top1_indices=model.candidate_top1_index,
            candidate_pi=model.candidate_pi,
            missing_span=model.missing_span,
        )
        batch_size = metrics["num_samples"]
        batch_start_index = sample_count

        totals["loss_total"] += model.loss_total_item
        totals["loss_av_gen"] += model.loss_av_gen_item
        totals["loss_recon"] += model.loss_recon_item
        totals["loss_g_gan"] += model.loss_G_GAN_item
        totals["loss_sync"] += model.loss_sync_item
        totals["loss_probe_gen"] += model.loss_probe_gen_item
        totals["loss_probe_recon"] += model.loss_probe_recon_item
        totals["loss_probe_g_gan"] += model.loss_probe_G_GAN_item
        totals["loss_d"] += model.loss_D_item
        totals["loss_anchor"] += model.loss_anchor_item
        totals["loss_min_k"] += model.loss_min_k_item
        totals["loss_mean_k"] += model.loss_mean_k_item
        totals["loss_boundary"] += model.loss_boundary_item
        totals["loss_evidence_div"] += model.loss_evidence_div_item
        totals["loss_gate_evidence"] += model.loss_gate_evidence_item
        totals["loss_candidate_scorer"] += model.loss_candidate_scorer_item
        totals["loss_uncertainty_calib"] += model.loss_uncertainty_calib_item
        totals["loss_calib"] += model.loss_calib_item
        totals["loss_multi_candidate"] += model.loss_multi_candidate_item
        totals["weighted_loss_min_k"] += model.weighted_loss_min_k_item
        totals["weighted_loss_mean_k"] += model.weighted_loss_mean_k_item
        totals["weighted_loss_boundary"] += model.weighted_loss_boundary_item
        totals["weighted_loss_evidence_div"] += model.weighted_loss_evidence_div_item
        totals["weighted_loss_gate_evidence"] += model.weighted_loss_gate_evidence_item
        totals["weighted_loss_calib"] += model.weighted_loss_calib_item
        totals["heuristic_evidence_mean"] += model.heuristic_evidence_mean_item
        totals["semantic_evidence_mean"] += model.semantic_evidence_mean_item
        totals["evidence_diversity_gap"] += model.evidence_diversity_gap_item
        totals["gate_mean"] += model.gate_mean_item
        totals["gate_target_mean"] += model.gate_target_mean_item
        totals["gate_target_gap"] += model.gate_target_gap_item
        totals["adapter_sigma_mean"] += model.adapter_sigma_mean_item
        totals["adapter_sigma_scale_mean"] += model.adapter_sigma_scale_mean_item
        totals["adapter_effective_sigma_mean"] += model.adapter_effective_sigma_mean_item
        totals["eta1"] += model.eta1_item
        totals["eta2"] += model.eta2_item
        totals["full_l1"] += model.loss_full_l1_item
        totals["missing_l1"] += model.loss_missing_l1_item
        totals["probe_full_l1"] += model.loss_probe_full_l1_item
        totals["probe_missing_l1"] += model.loss_probe_missing_l1_item
        totals["full_psnr"] += metrics["full_psnr"]
        totals["missing_psnr"] += metrics["missing_psnr"]
        totals["ssim"] += metrics["ssim"]

        batch_top1 = _to_flat_np(candidate_metrics["top1_missing_l1_per_sample"])
        batch_best = _to_flat_np(candidate_metrics["best_of_k_missing_l1_per_sample"])
        batch_mean = _to_flat_np(candidate_metrics["mean_k_missing_l1_per_sample"])
        batch_candidate0 = _to_flat_np(
            candidate_metrics["candidate0_missing_l1_per_sample"]
        )
        batch_random = _to_flat_np(
            candidate_metrics["random_expected_missing_l1_per_sample"]
        )
        batch_oracle = _to_flat_np(candidate_metrics["oracle_gain_per_sample"])
        batch_pairwise = _to_flat_np(
            candidate_metrics["candidate_pairwise_mel_l1_per_sample"]
        )
        batch_boundary_top1 = _to_flat_np(
            candidate_metrics["top1_boundary_delta_error_per_sample"]
        )
        batch_boundary_best = _to_flat_np(
            candidate_metrics["best_boundary_delta_error_per_sample"]
        )
        batch_boundary_mean = _to_flat_np(
            candidate_metrics["mean_boundary_delta_error_per_sample"]
        )
        sample_totals["top1_missing_l1"] += float(batch_top1.sum())
        sample_totals["best_of_k_missing_l1"] += float(batch_best.sum())
        sample_totals["mean_k_missing_l1"] += float(batch_mean.sum())
        sample_totals["candidate0_missing_l1"] += float(batch_candidate0.sum())
        sample_totals["random_expected_missing_l1"] += float(batch_random.sum())
        sample_totals["oracle_gain"] += float(batch_oracle.sum())
        sample_totals["candidate_pairwise_mel_l1"] += float(batch_pairwise.sum())
        sample_totals["boundary_delta_error_top1"] += float(batch_boundary_top1.sum())
        sample_totals["boundary_delta_error_best"] += float(batch_boundary_best.sum())
        sample_totals["boundary_delta_error_mean"] += float(batch_boundary_mean.sum())

        audio_embeddings.append(util.to_np(model.mel_net_norm))
        video_embeddings.append(util.to_np(model.video_net_norm))
        uncertainty_values.append(util.to_np(model.uncertainty_score.reshape(-1)))
        top1_error_values.append(batch_top1)
        best_error_values.append(batch_best)
        oracle_gain_values.append(batch_oracle)
        evidence_values.append(_mean_per_sample(model.evidence_score))
        pairwise_values.append(batch_pairwise)
        per_sample_rows.extend(
            collect_per_sample_rows(
                model,
                candidate_metrics,
                batch_start_index,
                wrong_video_sampler=wrong_video_sampler,
            )
        )
        sample_count += batch_size
        batch_count += 1
        if image_dir is not None:
            save_mel_comparison_batch(
                image_dir,
                sample_count - batch_size,
                model.path_batch,
                model.mel_input_4d,
                model.mel_pred,
                model.mel_target_4d,
                model.missing_mask,
            )
        if vocoder_dir is not None:
            remaining = None
            if vocoder_max_samples is not None:
                remaining = int(vocoder_max_samples) - vocoder_count
                if remaining <= 0:
                    remaining = 0
            if remaining is None or remaining > 0:
                written = save_vocoder_batch(
                    vocoder_dir,
                    sample_count - batch_size,
                    model.path_batch,
                    model.mel_input_4d,
                    model.mel_pred,
                    model.missing_mask,
                    model.audio_target,
                    hparams,
                    backend=getattr(hparams, "vocoder_backend", "griffin_lim"),
                    n_iter=getattr(hparams, "vocoder_n_iter", 32),
                    checkpoint_path=getattr(hparams, "vocoder_checkpoint", None),
                    splice_missing=getattr(hparams, "vocoder_splice_missing", True),
                    crossfade_ms=getattr(hparams, "vocoder_crossfade_ms", 20.0),
                    max_items=remaining,
                )
                vocoder_count += len(written)
        if candidate_image_dir is not None:
            save_candidate_mel_batches(
                candidate_image_dir,
                batch_start_index,
                model,
            )
        if candidate_vocoder_dir is not None:
            remaining = None
            if vocoder_max_samples is not None:
                remaining = int(vocoder_max_samples) - candidate_vocoder_count
                if remaining <= 0:
                    remaining = 0
            if remaining is None or remaining > 0:
                candidate_vocoder_count += save_candidate_vocoder_batches(
                    candidate_vocoder_dir,
                    batch_start_index,
                    model,
                    max_items=remaining,
                )
        progress.set_postfix(
            loss=f"{model.loss_total_item:.4f}",
            recon=f"{model.loss_recon_item:.4f}",
            min_k=f"{model.loss_min_k_item:.4f}",
            top1=f"{float(batch_top1.mean()):.4f}",
            mean_k=f"{model.loss_mean_k_item:.4f}",
            u=f"{model.uncertainty_mean_item:.3f}",
            gate=f"{model.gate_mean_item:.3f}",
            gate_t=f"{model.gate_target_mean_item:.3f}",
            pair=f"{float(batch_pairwise.mean()):.4f}",
            sync=f"{model.loss_sync_item:.4f}",
            probe=f"{model.loss_probe_gen_item:.4f}",
            g_gan=f"{model.loss_G_GAN_item:.4f}",
            d=f"{model.loss_D_item:.4f}",
            psnr=f"{metrics['full_psnr'] / batch_size:.2f}",
            ssim=f"{metrics['ssim'] / batch_size:.4f}",
        )

    if batch_count == 0:
        raise RuntimeError("VIAI-AV test dataloader is empty.")
    audio_embeddings = np.concatenate(audio_embeddings, axis=0)
    video_embeddings = np.concatenate(video_embeddings, axis=0)
    audio_to_video = util.L2retrieval(video_embeddings, audio_embeddings)
    video_to_audio = util.L2retrieval(audio_embeddings, video_embeddings)
    uncertainty_corrs = uncertainty_correlations(
        uncertainty_values,
        top1_error_values,
        best_error_values,
    )
    all_uncertainty = np.concatenate(uncertainty_values, axis=0)
    all_top1 = np.concatenate(top1_error_values, axis=0)
    all_best = np.concatenate(best_error_values, axis=0)
    all_oracle_gain = np.concatenate(oracle_gain_values, axis=0)
    all_evidence = np.concatenate(evidence_values, axis=0)
    all_pairwise = np.concatenate(pairwise_values, axis=0)
    risk_rows = compute_risk_coverage_curve(
        all_uncertainty,
        all_top1,
        num_points=20,
    )
    calibration_rows = compute_calibration_bins(
        all_uncertainty,
        all_top1,
        best_error=all_best,
        oracle_gain=all_oracle_gain,
        evidence=all_evidence,
        pairwise=all_pairwise,
        num_bins=getattr(hparams, "calibration_bins", 10),
    )
    if per_sample_path is not None:
        write_table(per_sample_path, per_sample_rows)
    if risk_path is not None:
        write_table(
            risk_path,
            risk_rows,
            fieldnames=[
                "coverage",
                "retained_count",
                "uncertainty_threshold",
                "mean_top1_error",
            ],
        )
    if calibration_path is not None:
        write_table(
            calibration_path,
            calibration_rows,
            fieldnames=[
                "bin_index",
                "bin_low",
                "bin_high",
                "count",
                "avg_uncertainty",
                "avg_top1_error",
                "avg_best_error",
                "avg_oracle_gain",
                "avg_evidence",
                "avg_pairwise",
            ],
        )

    def sample_average(name):
        return sample_totals[name] / sample_count

    return {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_av_gen": totals["loss_av_gen"] / batch_count,
        "loss_recon": totals["loss_recon"] / batch_count,
        "loss_g_gan": totals["loss_g_gan"] / batch_count,
        "loss_sync": totals["loss_sync"] / batch_count,
        "loss_probe_gen": totals["loss_probe_gen"] / batch_count,
        "loss_probe_recon": totals["loss_probe_recon"] / batch_count,
        "loss_probe_g_gan": totals["loss_probe_g_gan"] / batch_count,
        "loss_d": totals["loss_d"] / batch_count,
        "loss_anchor": totals["loss_anchor"] / batch_count,
        "loss_min_k": totals["loss_min_k"] / batch_count,
        "loss_mean_k": totals["loss_mean_k"] / batch_count,
        "loss_boundary": totals["loss_boundary"] / batch_count,
        "loss_evidence_div": totals["loss_evidence_div"] / batch_count,
        "loss_gate_evidence": totals["loss_gate_evidence"] / batch_count,
        "loss_candidate_scorer": totals["loss_candidate_scorer"] / batch_count,
        "loss_uncertainty_calib": totals["loss_uncertainty_calib"] / batch_count,
        "loss_calib": totals["loss_calib"] / batch_count,
        "loss_multi_candidate": totals["loss_multi_candidate"] / batch_count,
        "weighted_loss_min_k": totals["weighted_loss_min_k"] / batch_count,
        "weighted_loss_mean_k": totals["weighted_loss_mean_k"] / batch_count,
        "weighted_loss_boundary": totals["weighted_loss_boundary"] / batch_count,
        "weighted_loss_evidence_div": totals["weighted_loss_evidence_div"] / batch_count,
        "weighted_loss_gate_evidence": totals["weighted_loss_gate_evidence"] / batch_count,
        "weighted_loss_calib": totals["weighted_loss_calib"] / batch_count,
        "best_of_k_missing_l1": sample_average("best_of_k_missing_l1"),
        "mean_k_missing_l1": sample_average("mean_k_missing_l1"),
        "top1_missing_l1": sample_average("top1_missing_l1"),
        "candidate0_missing_l1": sample_average("candidate0_missing_l1"),
        "random_expected_missing_l1": sample_average("random_expected_missing_l1"),
        "oracle_gain": sample_average("oracle_gain"),
        "uncertainty_mean": float(all_uncertainty.mean()),
        "uncertainty_error_corr": uncertainty_corrs["uncertainty_error_corr"],
        "uncertainty_error_spearman": uncertainty_corrs["uncertainty_error_spearman"],
        "uncertainty_best_error_corr": uncertainty_corrs["uncertainty_best_error_corr"],
        "uncertainty_best_error_spearman": uncertainty_corrs[
            "uncertainty_best_error_spearman"
        ],
        "uncertainty_corr_count": uncertainty_corrs["uncertainty_corr_count"],
        "candidate_pairwise_distance": sample_average("candidate_pairwise_mel_l1"),
        "candidate_pairwise_mel_l1": sample_average("candidate_pairwise_mel_l1"),
        "boundary_delta_error_top1": sample_average("boundary_delta_error_top1"),
        "boundary_delta_error_best": sample_average("boundary_delta_error_best"),
        "boundary_delta_error_mean": sample_average("boundary_delta_error_mean"),
        "evidence_mean": float(all_evidence.mean()),
        "evidence_min": float(all_evidence.min()),
        "evidence_max": float(all_evidence.max()),
        "heuristic_evidence_mean": totals["heuristic_evidence_mean"] / batch_count,
        "semantic_evidence_mean": totals["semantic_evidence_mean"] / batch_count,
        "evidence_diversity_gap": totals["evidence_diversity_gap"] / batch_count,
        "gate_mean": totals["gate_mean"] / batch_count,
        "gate_target_mean": totals["gate_target_mean"] / batch_count,
        "gate_target_gap": totals["gate_target_gap"] / batch_count,
        "adapter_sigma_mean": totals["adapter_sigma_mean"] / batch_count,
        "adapter_sigma_scale_mean": totals["adapter_sigma_scale_mean"] / batch_count,
        "adapter_effective_sigma_mean": (
            totals["adapter_effective_sigma_mean"] / batch_count
        ),
        "eta1": totals["eta1"] / batch_count,
        "eta2": totals["eta2"] / batch_count,
        "mel_l1_full": totals["full_l1"] / batch_count,
        "mel_l1_missing": totals["missing_l1"] / batch_count,
        "probe_l1_full": totals["probe_full_l1"] / batch_count,
        "probe_l1_missing": totals["probe_missing_l1"] / batch_count,
        "psnr_full": totals["full_psnr"] / sample_count,
        "psnr_missing": totals["missing_psnr"] / sample_count,
        "ssim": totals["ssim"] / sample_count,
        "num_samples": sample_count,
        "retrieval_audio_to_video": audio_to_video,
        "retrieval_video_to_audio": video_to_audio,
        "skipped_batches": skipped_batches,
        "wrong_video_effective_mode": wrong_video_effective_mode(perturbation_mode()),
        "wrong_video_cross_instrument_available": bool(
            wrong_video_sampler.cross_instrument_available
            if wrong_video_sampler is not None
            else False
        ),
        "wrong_video_num_instruments": int(
            len(wrong_video_sampler.instruments)
            if wrong_video_sampler is not None
            else 0
        ),
        "vocoder_output_dir": "" if vocoder_dir is None else vocoder_dir,
        "vocoder_num_samples": vocoder_count,
        "candidate_image_dir": "" if candidate_image_dir is None else candidate_image_dir,
        "candidate_vocoder_dir": (
            "" if candidate_vocoder_dir is None else candidate_vocoder_dir
        ),
        "candidate_vocoder_num_samples": candidate_vocoder_count,
        "per_sample_metrics_path": "" if per_sample_path is None else per_sample_path,
        "risk_coverage_path": "" if risk_path is None else risk_path,
        "calibration_bins_path": "" if calibration_path is None else calibration_path,
    }


def build_result_record(checkpoint_path, checkpoint_step_value, global_step, global_epoch, results):
    audio_to_video = results["retrieval_audio_to_video"]
    video_to_audio = results["retrieval_video_to_audio"]
    return {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_step": int(checkpoint_step_value),
        "global_step": int(global_step),
        "global_epoch": int(global_epoch),
        "test_split_name": hparams.test_split_name,
        "stage": getattr(
            hparams,
            "loaded_stage",
            "EC-VIAI-AV-stage4-deterministic-adapter"
            if bool(getattr(hparams, "enable_ec_viai_av", False))
            and bool(getattr(hparams, "deterministic_adapter", False))
            else "VIAI-AV-stage4-sync-probe",
        ),
        "use_gan": bool(getattr(hparams, "use_gan", False)),
        "enable_sync_loss": not bool(getattr(hparams, "disable_sync_loss", False)),
        "enable_probe_loss": not bool(getattr(hparams, "disable_probe_loss", False)),
        "enable_ec_viai_av": bool(getattr(hparams, "enable_ec_viai_av", False)),
        "num_candidates": int(getattr(hparams, "num_candidates", 1)),
        "test_num_candidates": int(getattr(hparams, "test_num_candidates", 1)),
        "stochastic_adapter": bool(getattr(hparams, "stochastic_adapter", False)),
        "deterministic_adapter": bool(getattr(hparams, "deterministic_adapter", False)),
        "enable_evidence_gate": bool(getattr(hparams, "enable_evidence_gate", False)),
        "freeze_gate_evidence_backbone": bool(
            getattr(hparams, "freeze_gate_evidence_backbone", False)
        ),
        "enable_evidence_scaled_sigma": bool(
            getattr(hparams, "enable_evidence_scaled_sigma", False)
        ),
        "enable_candidate_scorer": bool(
            getattr(hparams, "enable_candidate_scorer", False)
        ),
        "train_candidate_heads_only": bool(
            getattr(hparams, "train_candidate_heads_only", False)
        ),
        "evidence_source": getattr(hparams, "evidence_source", "none"),
        "semantic_evidence_path": getattr(hparams, "semantic_evidence_path", ""),
        "semantic_evidence_weight": float(
            getattr(hparams, "semantic_evidence_weight", 0.35)
        ),
        "semantic_missing_score": float(
            getattr(hparams, "semantic_missing_score", 0.0)
        ),
        "enable_visual_evidence_aug": bool(
            getattr(hparams, "enable_visual_evidence_aug", False)
        ),
        "visual_evidence_aug_prob": float(
            getattr(hparams, "visual_evidence_aug_prob", 0.5)
        ),
        "visual_evidence_aug_modes": getattr(hparams, "visual_evidence_aug_modes", ""),
        "sigma_min": float(getattr(hparams, "sigma_min", 0.0)),
        "sigma_max": float(getattr(hparams, "sigma_max", 1.0)),
        "evidence_sigma_scale_min": float(
            getattr(hparams, "evidence_sigma_scale_min", 0.5)
        ),
        "evidence_sigma_scale_max": float(
            getattr(hparams, "evidence_sigma_scale_max", 2.0)
        ),
        "calib_error_tau": float(getattr(hparams, "calib_error_tau", 0.1)),
        "save_candidates": bool(getattr(hparams, "save_candidates", False)),
        "video_perturbation": getattr(hparams, "video_perturbation", "none"),
        "wrong_video_effective_mode": results.get("wrong_video_effective_mode", ""),
        "wrong_video_cross_instrument_available": bool(
            results.get("wrong_video_cross_instrument_available", False)
        ),
        "wrong_video_num_instruments": int(
            results.get("wrong_video_num_instruments", 0)
        ),
        "video_blur_kernel": int(getattr(hparams, "video_blur_kernel", 9)),
        "video_frame_drop_stride": int(
            getattr(hparams, "video_frame_drop_stride", 2)
        ),
        "video_temporal_shift_frames": int(
            getattr(hparams, "video_temporal_shift_frames", 6)
        ),
        "calibration_bins": int(getattr(hparams, "calibration_bins", 10)),
        "num_samples": int(results["num_samples"]),
        "loss_total": float(results["loss_total"]),
        "loss_av_gen": float(results["loss_av_gen"]),
        "loss_recon": float(results["loss_recon"]),
        "loss_g_gan": float(results["loss_g_gan"]),
        "loss_sync": float(results["loss_sync"]),
        "loss_probe_gen": float(results["loss_probe_gen"]),
        "loss_probe_recon": float(results["loss_probe_recon"]),
        "loss_probe_g_gan": float(results["loss_probe_g_gan"]),
        "loss_d": float(results["loss_d"]),
        "loss_anchor": float(results["loss_anchor"]),
        "loss_min_k": float(results["loss_min_k"]),
        "loss_mean_k": float(results["loss_mean_k"]),
        "loss_boundary": float(results["loss_boundary"]),
        "loss_evidence_div": float(results["loss_evidence_div"]),
        "loss_gate_evidence": float(results["loss_gate_evidence"]),
        "loss_candidate_scorer": float(results["loss_candidate_scorer"]),
        "loss_uncertainty_calib": float(results["loss_uncertainty_calib"]),
        "loss_calib": float(results["loss_calib"]),
        "loss_multi_candidate": float(results["loss_multi_candidate"]),
        "weighted_loss_min_k": float(results["weighted_loss_min_k"]),
        "weighted_loss_mean_k": float(results["weighted_loss_mean_k"]),
        "weighted_loss_boundary": float(results["weighted_loss_boundary"]),
        "weighted_loss_evidence_div": float(results["weighted_loss_evidence_div"]),
        "weighted_loss_gate_evidence": float(results["weighted_loss_gate_evidence"]),
        "weighted_loss_calib": float(results["weighted_loss_calib"]),
        "best_of_k_missing_l1": float(results["best_of_k_missing_l1"]),
        "mean_k_missing_l1": float(results["mean_k_missing_l1"]),
        "top1_missing_l1": float(results["top1_missing_l1"]),
        "candidate0_missing_l1": float(results["candidate0_missing_l1"]),
        "random_expected_missing_l1": float(results["random_expected_missing_l1"]),
        "oracle_gain": float(results["oracle_gain"]),
        "uncertainty_mean": float(results["uncertainty_mean"]),
        "uncertainty_error_corr": float(results["uncertainty_error_corr"]),
        "uncertainty_error_spearman": float(results["uncertainty_error_spearman"]),
        "uncertainty_best_error_corr": float(results["uncertainty_best_error_corr"]),
        "uncertainty_best_error_spearman": float(
            results["uncertainty_best_error_spearman"]
        ),
        "uncertainty_corr_count": int(results["uncertainty_corr_count"]),
        "candidate_pairwise_distance": float(results["candidate_pairwise_distance"]),
        "candidate_pairwise_mel_l1": float(results["candidate_pairwise_mel_l1"]),
        "boundary_delta_error_top1": float(results["boundary_delta_error_top1"]),
        "boundary_delta_error_best": float(results["boundary_delta_error_best"]),
        "boundary_delta_error_mean": float(results["boundary_delta_error_mean"]),
        "evidence_mean": float(results["evidence_mean"]),
        "evidence_min": float(results["evidence_min"]),
        "evidence_max": float(results["evidence_max"]),
        "heuristic_evidence_mean": float(results["heuristic_evidence_mean"]),
        "semantic_evidence_mean": float(results["semantic_evidence_mean"]),
        "evidence_diversity_gap": float(results["evidence_diversity_gap"]),
        "gate_mean": float(results["gate_mean"]),
        "gate_target_mean": float(results["gate_target_mean"]),
        "gate_target_gap": float(results["gate_target_gap"]),
        "adapter_sigma_mean": float(results["adapter_sigma_mean"]),
        "adapter_sigma_scale_mean": float(results["adapter_sigma_scale_mean"]),
        "adapter_effective_sigma_mean": float(results["adapter_effective_sigma_mean"]),
        "eta1": float(results["eta1"]),
        "eta2": float(results["eta2"]),
        "lambda_recon": float(getattr(hparams, "lambda_recon", 1.0)),
        "lambda_min_k": float(getattr(hparams, "lambda_min_k", 0.0)),
        "lambda_mean_k": float(getattr(hparams, "lambda_mean_k", 0.0)),
        "lambda_boundary": float(getattr(hparams, "lambda_boundary", 0.0)),
        "lambda_diversity": float(getattr(hparams, "lambda_diversity", 0.0)),
        "lambda_calib": float(getattr(hparams, "lambda_calib", 0.0)),
        "lambda_gate_evidence": float(getattr(hparams, "lambda_gate_evidence", 0.0)),
        "evidence_diversity_d_min": float(
            getattr(hparams, "evidence_diversity_d_min", 0.02)
        ),
        "evidence_diversity_alpha": float(
            getattr(hparams, "evidence_diversity_alpha", 0.08)
        ),
        "evidence_gate_low": float(getattr(hparams, "evidence_gate_low", 0.24)),
        "evidence_gate_high": float(getattr(hparams, "evidence_gate_high", 0.34)),
        "mel_l1_full": float(results["mel_l1_full"]),
        "mel_l1_missing": float(results["mel_l1_missing"]),
        "probe_l1_full": float(results["probe_l1_full"]),
        "probe_l1_missing": float(results["probe_l1_missing"]),
        "psnr_full": float(results["psnr_full"]),
        "psnr_missing": float(results["psnr_missing"]),
        "ssim": float(results["ssim"]),
        "retrieval_audio_to_video_r1": float(audio_to_video[0]),
        "retrieval_audio_to_video_r5": float(audio_to_video[1]),
        "retrieval_audio_to_video_r10": float(audio_to_video[2]),
        "retrieval_audio_to_video_r50": float(audio_to_video[3]),
        "retrieval_audio_to_video_medr": float(audio_to_video[4]),
        "retrieval_audio_to_video_meanr": float(audio_to_video[5]),
        "retrieval_video_to_audio_r1": float(video_to_audio[0]),
        "retrieval_video_to_audio_r5": float(video_to_audio[1]),
        "retrieval_video_to_audio_r10": float(video_to_audio[2]),
        "retrieval_video_to_audio_r50": float(video_to_audio[3]),
        "retrieval_video_to_audio_medr": float(video_to_audio[4]),
        "retrieval_video_to_audio_meanr": float(video_to_audio[5]),
        "use_vocoder": bool(getattr(hparams, "use_vocoder", False)),
        "vocoder_backend": getattr(hparams, "vocoder_backend", "griffin_lim"),
        "vocoder_checkpoint": getattr(hparams, "vocoder_checkpoint", "") or "",
        "vocoder_splice_missing": bool(getattr(hparams, "vocoder_splice_missing", True)),
        "vocoder_crossfade_ms": float(getattr(hparams, "vocoder_crossfade_ms", 20.0)),
        "vocoder_n_iter": int(getattr(hparams, "vocoder_n_iter", 32)),
        "vocoder_output_dir": results.get("vocoder_output_dir", ""),
        "vocoder_num_samples": int(results.get("vocoder_num_samples", 0)),
        "candidate_image_dir": results.get("candidate_image_dir", ""),
        "candidate_vocoder_dir": results.get("candidate_vocoder_dir", ""),
        "candidate_vocoder_num_samples": int(
            results.get("candidate_vocoder_num_samples", 0)
        ),
        "per_sample_metrics_path": results.get("per_sample_metrics_path", ""),
        "risk_coverage_path": results.get("risk_coverage_path", ""),
        "calibration_bins_path": results.get("calibration_bins_path", ""),
    }


def coerce_csv_record(row):
    record = {}
    bool_fields = {
        "wrong_video_cross_instrument_available",
    }
    int_fields = {
        "checkpoint_step",
        "global_step",
        "global_epoch",
        "num_candidates",
        "test_num_candidates",
        "num_samples",
        "uncertainty_corr_count",
        "vocoder_n_iter",
        "vocoder_num_samples",
        "candidate_vocoder_num_samples",
        "wrong_video_num_instruments",
        "video_blur_kernel",
        "video_frame_drop_stride",
        "video_temporal_shift_frames",
        "calibration_bins",
    }
    float_fields = {
        "loss_total",
        "loss_av_gen",
        "loss_recon",
        "loss_g_gan",
        "loss_sync",
        "loss_probe_gen",
        "loss_probe_recon",
        "loss_probe_g_gan",
        "loss_d",
        "loss_anchor",
        "loss_min_k",
        "loss_mean_k",
        "loss_boundary",
        "loss_evidence_div",
        "loss_gate_evidence",
        "loss_candidate_scorer",
        "loss_uncertainty_calib",
        "loss_calib",
        "loss_multi_candidate",
        "weighted_loss_min_k",
        "weighted_loss_mean_k",
        "weighted_loss_boundary",
        "weighted_loss_evidence_div",
        "weighted_loss_gate_evidence",
        "weighted_loss_calib",
        "best_of_k_missing_l1",
        "mean_k_missing_l1",
        "top1_missing_l1",
        "candidate0_missing_l1",
        "random_expected_missing_l1",
        "oracle_gain",
        "uncertainty_mean",
        "uncertainty_error_corr",
        "uncertainty_error_spearman",
        "uncertainty_best_error_corr",
        "uncertainty_best_error_spearman",
        "candidate_pairwise_distance",
        "candidate_pairwise_mel_l1",
        "boundary_delta_error_top1",
        "boundary_delta_error_best",
        "boundary_delta_error_mean",
        "evidence_mean",
        "evidence_min",
        "evidence_max",
        "heuristic_evidence_mean",
        "semantic_evidence_mean",
        "evidence_diversity_gap",
        "gate_mean",
        "gate_target_mean",
        "gate_target_gap",
        "adapter_sigma_mean",
        "adapter_sigma_scale_mean",
        "adapter_effective_sigma_mean",
        "eta1",
        "eta2",
        "lambda_recon",
        "lambda_min_k",
        "lambda_mean_k",
        "lambda_boundary",
        "lambda_diversity",
        "lambda_calib",
        "lambda_gate_evidence",
        "evidence_diversity_d_min",
        "evidence_diversity_alpha",
        "evidence_gate_low",
        "evidence_gate_high",
        "evidence_sigma_scale_min",
        "evidence_sigma_scale_max",
        "semantic_evidence_weight",
        "semantic_missing_score",
        "calib_error_tau",
        "visual_evidence_aug_prob",
        "sigma_min",
        "sigma_max",
        "mel_l1_full",
        "mel_l1_missing",
        "probe_l1_full",
        "probe_l1_missing",
        "psnr_full",
        "psnr_missing",
        "ssim",
        "retrieval_audio_to_video_r1",
        "retrieval_audio_to_video_r5",
        "retrieval_audio_to_video_r10",
        "retrieval_audio_to_video_r50",
        "retrieval_audio_to_video_medr",
        "retrieval_audio_to_video_meanr",
        "retrieval_video_to_audio_r1",
        "retrieval_video_to_audio_r5",
        "retrieval_video_to_audio_r10",
        "retrieval_video_to_audio_r50",
        "retrieval_video_to_audio_medr",
        "retrieval_video_to_audio_meanr",
    }
    for field in RESULT_FIELDS:
        value = row.get(field, "")
        if field in bool_fields and value != "":
            record[field] = str(value).lower() in {"1", "true", "yes"}
        elif field in int_fields and value != "":
            record[field] = int(value)
        elif field in float_fields and value != "":
            record[field] = float(value)
        else:
            record[field] = value
    return record


def write_result_files(record, results_dir, name):
    os.makedirs(results_dir, exist_ok=True)
    step_text = format_step(record["checkpoint_step"])
    mode = safe_token(record.get("video_perturbation", "none"))
    json_path = os.path.join(
        results_dir,
        f"{name}_step{step_text}_perturb-{mode}_test.json",
    )
    csv_path = os.path.join(results_dir, f"{name}_test_summary.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    records_by_key = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("checkpoint_step"):
                    continue
                old_record = coerce_csv_record(row)
                old_key = (
                    int(old_record["checkpoint_step"]),
                    str(old_record.get("video_perturbation", "none") or "none"),
                    int(old_record.get("test_num_candidates", 1) or 1),
                )
                records_by_key[old_key] = old_record
    record_key = (
        int(record["checkpoint_step"]),
        str(record.get("video_perturbation", "none") or "none"),
        int(record.get("test_num_candidates", 1) or 1),
    )
    records_by_key[record_key] = record
    sorted_records = [records_by_key[key] for key in sorted(records_by_key)]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for item in sorted_records:
            writer.writerow({field: item.get(field, "") for field in RESULT_FIELDS})
    return json_path, csv_path


def main():
    configure_viai_av_defaults()
    print_viai_av_test_config()
    data_loaders = av_loader.get_data_loaders(
        hparams.data_root,
        hparams.speaker_id,
        test_shuffle=False,
        phases=("test",),
    )
    if "test" not in data_loaders:
        raise RuntimeError(
            f"VIAI-AV test split is missing or empty: {os.path.join(hparams.data_root, hparams.test_split_name)}"
        )

    model = VIAIAVModel(hparams, device=device)
    checkpoint_path = resolve_checkpoint_path(hparams.resume_path, hparams.checkpoint_dir, hparams.name)
    global_step, global_epoch = model.load_checkpoint(checkpoint_path, reset_optimizer=True)
    print(f"[VIAI-AV test] loaded checkpoint: {checkpoint_path} (step={global_step}, epoch={global_epoch})")

    checkpoint_step_value = checkpoint_step(checkpoint_path)
    if checkpoint_step_value < 0:
        checkpoint_step_value = global_step
    image_dir = mel_image_output_dir(hparams.results_dir, checkpoint_step_value)
    candidate_image_dir = None
    if getattr(hparams, "save_candidates", False):
        candidate_image_dir = candidate_image_output_dir(
            hparams.results_dir,
            checkpoint_step_value,
        )
    vocoder_dir = None
    if getattr(hparams, "use_vocoder", False):
        vocoder_dir = hparams.vocoder_output_dir
        if not vocoder_dir:
            vocoder_dir = os.path.join(
                hparams.results_dir,
                "wav",
                f"step{format_step(checkpoint_step_value)}",
            )
    candidate_vocoder_dir = None
    if getattr(hparams, "save_candidates", False) and getattr(hparams, "use_vocoder", False):
        candidate_vocoder_dir = candidate_vocoder_output_dir(
            hparams.results_dir,
            checkpoint_step_value,
        )
    wrong_video_sampler = None
    if perturbation_mode() in {
        "wrong_video",
        "wrong_video_any",
        "wrong_video_cross_instrument",
    }:
        wrong_video_sampler = WrongVideoSampler(hparams)
    sample_metrics_path = per_sample_metrics_path(
        hparams.results_dir,
        checkpoint_step_value,
    )
    risk_path_value = risk_coverage_path(
        hparams.results_dir,
        checkpoint_step_value,
    )
    calibration_path_value = calibration_bins_path(
        hparams.results_dir,
        checkpoint_step_value,
    )
    results = evaluate(
        model,
        data_loaders["test"],
        global_step=checkpoint_step_value,
        image_dir=image_dir,
        vocoder_dir=vocoder_dir,
        candidate_image_dir=candidate_image_dir,
        candidate_vocoder_dir=candidate_vocoder_dir,
        per_sample_path=sample_metrics_path,
        risk_path=risk_path_value,
        calibration_path=calibration_path_value,
        wrong_video_sampler=wrong_video_sampler,
    )
    result_record = build_result_record(
        checkpoint_path,
        checkpoint_step_value,
        global_step,
        global_epoch,
        results,
    )
    json_path, csv_path = write_result_files(
        result_record,
        hparams.results_dir,
        hparams.name,
    )
    print(
        "[VIAI-AV test] "
        f"samples={results['num_samples']} "
        f"loss={results['loss_total']:.6f} "
        f"av_gen={results['loss_av_gen']:.6f} "
        f"recon={results['loss_recon']:.6f} "
        f"sync={results['loss_sync']:.6f} "
        f"probe={results['loss_probe_gen']:.6f} "
        f"g_gan={results['loss_g_gan']:.6f} "
        f"probe_g_gan={results['loss_probe_g_gan']:.6f} "
        f"d={results['loss_d']:.6f} "
        f"min_k={results['loss_min_k']:.6f} "
        f"mean_k={results['loss_mean_k']:.6f} "
        f"boundary={results['loss_boundary']:.6f} "
        f"evidence_div={results['loss_evidence_div']:.6f} "
        f"gate_ev={results['loss_gate_evidence']:.6f} "
        f"scorer={results['loss_candidate_scorer']:.6f} "
        f"unc_calib={results['loss_uncertainty_calib']:.6f} "
        f"calib={results['loss_calib']:.6f} "
        f"multi={results['loss_multi_candidate']:.6f} "
        f"best_of_k={results['best_of_k_missing_l1']:.6f} "
        f"top1={results['top1_missing_l1']:.6f} "
        f"candidate0={results['candidate0_missing_l1']:.6f} "
        f"random={results['random_expected_missing_l1']:.6f} "
        f"oracle_gain={results['oracle_gain']:.6f} "
        f"uncertainty={results['uncertainty_mean']:.6f} "
        f"u_top1_corr={results['uncertainty_error_corr']:.6f} "
        f"u_top1_spearman={results['uncertainty_error_spearman']:.6f} "
        f"pairwise={results['candidate_pairwise_mel_l1']:.6f} "
        f"boundary_top1={results['boundary_delta_error_top1']:.6f} "
        f"evidence={results['evidence_mean']:.6f} "
        f"div_gap={results['evidence_diversity_gap']:.6f} "
        f"gate_mean={results['gate_mean']:.6f} "
        f"gate_target={results['gate_target_mean']:.6f} "
        f"gate_gap={results['gate_target_gap']:.6f} "
        f"eta1={results['eta1']:.6f} "
        f"eta2={results['eta2']:.6f} "
        f"mel_l1_full={results['mel_l1_full']:.6f} "
        f"mel_l1_missing={results['mel_l1_missing']:.6f} "
        f"probe_l1_full={results['probe_l1_full']:.6f} "
        f"probe_l1_missing={results['probe_l1_missing']:.6f} "
        f"psnr_full={results['psnr_full']:.3f} "
        f"psnr_missing={results['psnr_missing']:.3f} "
        f"ssim={results['ssim']:.4f}"
    )
    print(
        "[VIAI-AV test] audio->video retrieval "
        f"R@1={results['retrieval_audio_to_video'][0]:.2f} "
        f"R@5={results['retrieval_audio_to_video'][1]:.2f} "
        f"R@10={results['retrieval_audio_to_video'][2]:.2f} "
        f"R@50={results['retrieval_audio_to_video'][3]:.2f} "
        f"MedR={results['retrieval_audio_to_video'][4]:.1f} "
        f"MeanR={results['retrieval_audio_to_video'][5]:.1f}"
    )
    print(
        "[VIAI-AV test] video->audio retrieval "
        f"R@1={results['retrieval_video_to_audio'][0]:.2f} "
        f"R@5={results['retrieval_video_to_audio'][1]:.2f} "
        f"R@10={results['retrieval_video_to_audio'][2]:.2f} "
        f"R@50={results['retrieval_video_to_audio'][3]:.2f} "
        f"MedR={results['retrieval_video_to_audio'][4]:.1f} "
        f"MeanR={results['retrieval_video_to_audio'][5]:.1f}"
    )
    print(f"[VIAI-AV test] wrote json: {json_path}")
    print(f"[VIAI-AV test] wrote summary csv: {csv_path}")
    print(f"[VIAI-AV test] wrote mel images: {image_dir}")
    print(f"[VIAI-AV test] wrote per-sample metrics: {results['per_sample_metrics_path']}")
    print(f"[VIAI-AV test] wrote risk-coverage: {results['risk_coverage_path']}")
    print(f"[VIAI-AV test] wrote calibration bins: {results['calibration_bins_path']}")
    if getattr(hparams, "save_candidates", False):
        print(f"[VIAI-AV test] wrote candidate mel images: {results['candidate_image_dir']}")
    if getattr(hparams, "use_vocoder", False):
        print(
            f"[VIAI-AV test] wrote vocoder wavs: {results['vocoder_output_dir']} "
            f"({results['vocoder_num_samples']} samples)"
        )
        if getattr(hparams, "save_candidates", False):
            print(
                "[VIAI-AV test] wrote candidate vocoder wavs: "
                f"{results['candidate_vocoder_dir']} "
                f"({results['candidate_vocoder_num_samples']} samples)"
            )


if __name__ == "__main__":
    main()
    sys.exit(0)
