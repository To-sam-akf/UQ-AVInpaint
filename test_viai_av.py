import csv
import json
import os
import re
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel
from utils import util
from utils.viai_a_metrics import (
    compose_inpainted_mel,
    compute_viai_a_metrics,
    save_mel_comparison_batch,
)
from utils.vocoder import save_vocoder_batch


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
    "evidence_source",
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
    "vocoder_n_iter",
    "vocoder_output_dir",
    "vocoder_num_samples",
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
        f"save_candidates={getattr(hparams, 'save_candidates', False)} "
        f"video_perturbation={getattr(hparams, 'video_perturbation', 'none')}"
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


def mel_image_output_dir(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "mel-image",
        f"step{format_step(checkpoint_step_value)}",
    )


def evaluate(model, data_loader, global_step=0, image_dir=None, vocoder_dir=None):
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
        "best_of_k_missing_l1": 0.0,
        "mean_k_missing_l1": 0.0,
        "top1_missing_l1": 0.0,
        "candidate0_missing_l1": 0.0,
        "random_expected_missing_l1": 0.0,
        "oracle_gain": 0.0,
        "uncertainty_mean": 0.0,
        "candidate_pairwise_distance": 0.0,
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
    sample_count = 0
    batch_count = 0
    skipped_batches = 0
    vocoder_count = 0
    vocoder_max_samples = getattr(hparams, "vocoder_max_samples", None)
    audio_embeddings = []
    video_embeddings = []
    uncertainty_values = []
    top1_error_values = []
    best_error_values = []

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
        model.test(global_step=global_step)
        model.get_loss_items()
        metrics = batch_metrics(model)
        batch_size = metrics["num_samples"]

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
        totals["best_of_k_missing_l1"] += model.best_of_k_missing_l1_item
        totals["mean_k_missing_l1"] += model.mean_k_missing_l1_item
        totals["top1_missing_l1"] += model.top1_missing_l1_item
        totals["candidate0_missing_l1"] += model.candidate0_missing_l1_item
        totals["random_expected_missing_l1"] += model.random_expected_missing_l1_item
        totals["oracle_gain"] += model.oracle_gain_item
        totals["uncertainty_mean"] += model.uncertainty_mean_item
        totals["candidate_pairwise_distance"] += model.candidate_pairwise_distance_item
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
        audio_embeddings.append(util.to_np(model.mel_net_norm))
        video_embeddings.append(util.to_np(model.video_net_norm))
        uncertainty_values.append(util.to_np(model.uncertainty_score.reshape(-1)))
        top1_error_values.append(util.to_np(model.top1_missing_l1_per_sample.reshape(-1)))
        best_error_values.append(util.to_np(model.best_of_k_missing_l1_per_sample.reshape(-1)))
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
                    max_items=remaining,
                )
                vocoder_count += len(written)
        progress.set_postfix(
            loss=f"{model.loss_total_item:.4f}",
            recon=f"{model.loss_recon_item:.4f}",
            min_k=f"{model.loss_min_k_item:.4f}",
            top1=f"{model.top1_missing_l1_item:.4f}",
            mean_k=f"{model.loss_mean_k_item:.4f}",
            u=f"{model.uncertainty_mean_item:.3f}",
            gate=f"{model.gate_mean_item:.3f}",
            gate_t=f"{model.gate_target_mean_item:.3f}",
            pair=f"{model.candidate_pairwise_distance_item:.4f}",
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
        "best_of_k_missing_l1": totals["best_of_k_missing_l1"] / batch_count,
        "mean_k_missing_l1": totals["mean_k_missing_l1"] / batch_count,
        "top1_missing_l1": totals["top1_missing_l1"] / batch_count,
        "candidate0_missing_l1": totals["candidate0_missing_l1"] / batch_count,
        "random_expected_missing_l1": (
            totals["random_expected_missing_l1"] / batch_count
        ),
        "oracle_gain": totals["oracle_gain"] / batch_count,
        "uncertainty_mean": totals["uncertainty_mean"] / batch_count,
        "uncertainty_error_corr": uncertainty_corrs["uncertainty_error_corr"],
        "uncertainty_error_spearman": uncertainty_corrs["uncertainty_error_spearman"],
        "uncertainty_best_error_corr": uncertainty_corrs["uncertainty_best_error_corr"],
        "uncertainty_best_error_spearman": uncertainty_corrs[
            "uncertainty_best_error_spearman"
        ],
        "uncertainty_corr_count": uncertainty_corrs["uncertainty_corr_count"],
        "candidate_pairwise_distance": totals["candidate_pairwise_distance"] / batch_count,
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
        "vocoder_output_dir": "" if vocoder_dir is None else vocoder_dir,
        "vocoder_num_samples": vocoder_count,
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
        "evidence_source": getattr(hparams, "evidence_source", "none"),
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
        "vocoder_n_iter": int(getattr(hparams, "vocoder_n_iter", 32)),
        "vocoder_output_dir": results.get("vocoder_output_dir", ""),
        "vocoder_num_samples": int(results.get("vocoder_num_samples", 0)),
    }


def coerce_csv_record(row):
    record = {}
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
        if field in int_fields and value != "":
            record[field] = int(value)
        elif field in float_fields and value != "":
            record[field] = float(value)
        else:
            record[field] = value
    return record


def write_result_files(record, results_dir, name):
    os.makedirs(results_dir, exist_ok=True)
    step_text = format_step(record["checkpoint_step"])
    json_path = os.path.join(results_dir, f"{name}_step{step_text}_test.json")
    csv_path = os.path.join(results_dir, f"{name}_test_summary.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    records_by_step = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row.get("checkpoint_step"):
                    continue
                old_record = coerce_csv_record(row)
                records_by_step[int(old_record["checkpoint_step"])] = old_record
    records_by_step[int(record["checkpoint_step"])] = record
    sorted_records = [records_by_step[step] for step in sorted(records_by_step)]
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
    vocoder_dir = None
    if getattr(hparams, "use_vocoder", False):
        vocoder_dir = hparams.vocoder_output_dir
        if not vocoder_dir:
            vocoder_dir = os.path.join(
                hparams.results_dir,
                "wav",
                f"step{format_step(checkpoint_step_value)}",
            )
    results = evaluate(
        model,
        data_loaders["test"],
        global_step=checkpoint_step_value,
        image_dir=image_dir,
        vocoder_dir=vocoder_dir,
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
        f"pairwise={results['candidate_pairwise_distance']:.6f} "
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
    if getattr(hparams, "use_vocoder", False):
        print(
            f"[VIAI-AV test] wrote vocoder wavs: {results['vocoder_output_dir']} "
            f"({results['vocoder_num_samples']} samples)"
        )


if __name__ == "__main__":
    main()
    sys.exit(0)
