import os
import re
import sys
import time
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from tqdm import tqdm

import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel
from utils.viai_a_metrics import compute_viai_a_metrics, write_mel_images


hparams = Options_inpainting.Inpainting_Config()
use_cuda = torch.cuda.is_available()
if use_cuda:
    cudnn.benchmark = False
device = torch.device("cuda" if use_cuda else "cpu")


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
    if not _arg_was_passed("--log_event_path"):
        if getattr(hparams, "enable_ec_viai_av", False):
            event_name = (
                "events_ec_viai_av_patchgan"
                if getattr(hparams, "use_gan", False)
                else "events_ec_viai_av"
            )
        else:
            event_name = (
                "events_viai_av_patchgan"
                if getattr(hparams, "use_gan", False)
                else "events_viai_av"
            )
        hparams.log_event_path = os.path.join(hparams.checkpoint_dir, event_name)


def print_viai_av_run_config():
    print(
        "[VIAI-AV] run config: "
        f"use_gan={getattr(hparams, 'use_gan', False)} "
        f"lambda_recon={getattr(hparams, 'lambda_recon', 1.0)} "
        f"lambda_gan={getattr(hparams, 'lambda_gan', 1.0)} "
        f"lambda_sync={getattr(hparams, 'lambda_sync', 1.0)} "
        f"lambda_probe={getattr(hparams, 'lambda_probe', 1.0)} "
        f"disable_sync_loss={getattr(hparams, 'disable_sync_loss', False)} "
        f"disable_probe_loss={getattr(hparams, 'disable_probe_loss', False)}"
    )
    print(
        "[VIAI-AV] EC config: "
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
        "[VIAI-AV] EC losses/evidence: "
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
        "[VIAI-AV] paths: "
        f"name={hparams.name} "
        f"checkpoint_dir={hparams.checkpoint_dir} "
        f"log_event_path={hparams.log_event_path} "
        f"init_from_viai_a={hparams.init_from_viai_a} "
        f"resume={hparams.resume} "
        f"resume_path={hparams.resume_path}"
    )


def checkpoint_step(path):
    match = re.search(r"checkpoint_step(\d+)", os.path.basename(str(path)))
    return int(match.group(1)) if match else -1


def resolve_latest_checkpoint(directory, prefix):
    if not directory or not os.path.isdir(directory):
        return None
    candidates = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith(".pth.tar"):
            path = os.path.join(directory, filename)
            candidates.append(path)
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (checkpoint_step(path), os.path.getmtime(path)))[-1]


def resolve_init_checkpoint():
    if hparams.init_from_viai_a is not None:
        candidate = os.path.abspath(hparams.init_from_viai_a)
        if not os.path.exists(candidate):
            raise RuntimeError(
                f"VIAI-A initialization checkpoint not found: {candidate}. "
                "Train VIAI-A first, or pass an existing checkpoint."
            )
        return candidate

    checkpoint_dir = os.path.abspath(hparams.checkpoint_dir)
    if getattr(hparams, "use_gan", False):
        candidate = resolve_latest_checkpoint(checkpoint_dir, "VIAI-A-PatchGAN_checkpoint_step")
        if candidate is not None:
            print(f"[VIAI-AV] using VIAI-A PatchGAN checkpoint for initialization: {candidate}")
            return candidate

        candidate = resolve_latest_checkpoint(checkpoint_dir, "VIAI-A_checkpoint_step")
        if candidate is not None:
            print(
                "[VIAI-AV] no VIAI-A-PatchGAN checkpoint found; "
                f"falling back to VIAI-A audio-only checkpoint: {candidate}"
            )
            return candidate
    else:
        candidate = resolve_latest_checkpoint(checkpoint_dir, "VIAI-A_checkpoint_step")
        if candidate is not None:
            print(f"[VIAI-AV] using VIAI-A audio-only checkpoint for baseline initialization: {candidate}")
            return candidate

    expected = (
        "VIAI-A-PatchGAN_checkpoint_step*.pth.tar or VIAI-A_checkpoint_step*.pth.tar"
        if getattr(hparams, "use_gan", False)
        else "VIAI-A_checkpoint_step*.pth.tar"
    )
    raise RuntimeError(
        "train-viai-av requires a VIAI-A initialization checkpoint. "
        "Pass --init_from_viai_a explicitly, or place "
        f"{expected} under --checkpoint_dir."
    )


def resolve_resume_checkpoint():
    if hparams.resume_path is not None:
        candidate = os.path.abspath(hparams.resume_path)
        if os.path.exists(candidate):
            return candidate
        raise RuntimeError(f"VIAI-AV resume checkpoint not found: {candidate}")
    candidate = resolve_latest_checkpoint(
        os.path.abspath(hparams.checkpoint_dir),
        f"{hparams.name}_checkpoint_step",
    )
    if candidate is None:
        raise RuntimeError(
            "No VIAI-AV checkpoint found to resume. Pass --resume_path or place "
            f"{hparams.name}_checkpoint_step*.pth.tar under --checkpoint_dir."
        )
    return candidate


def _positive_interval(name):
    return max(0, int(getattr(hparams, name, 0)))


def _should_compute_ssim(train, step):
    if not train:
        return True
    metric_freq = _positive_interval("metric_freq")
    return step <= 1 or (metric_freq > 0 and step % metric_freq == 0)


def _should_write_images(train, step, batch_index):
    image_freq = _positive_interval("tb_image_freq")
    if train:
        return step <= 1 or (image_freq > 0 and step % image_freq == 0)
    return batch_index == 0


def _write_monitoring(writer, prefix, step, model, metrics):
    if writer is None:
        return
    writer.add_scalar(f"{prefix}/psnr_full", metrics["psnr_full"], step)
    writer.add_scalar(f"{prefix}/psnr_missing", metrics["psnr_missing"], step)
    if metrics["ssim_full"] is not None:
        writer.add_scalar(f"{prefix}/ssim_full", metrics["ssim_full"], step)
    writer.add_scalar(f"{prefix}/blank_frames", model.blank_length, step)
    writer.add_scalar(f"{prefix}/lr", model.current_lr, step)


def run_phase(model, data_loader, phase, global_step, writer, global_epoch):
    train = phase == "train"
    totals = {
        "loss_total": 0.0,
        "loss_av_gen": 0.0,
        "loss_recon": 0.0,
        "loss_full_l1": 0.0,
        "loss_missing_l1": 0.0,
        "loss_g_gan": 0.0,
        "loss_sync": 0.0,
        "loss_probe_gen": 0.0,
        "loss_probe_recon": 0.0,
        "loss_probe_full_l1": 0.0,
        "loss_probe_missing_l1": 0.0,
        "loss_probe_g_gan": 0.0,
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
        "candidate_pi_entropy": 0.0,
        "candidate_pi_max": 0.0,
        "candidate_pairwise_distance": 0.0,
        "evidence_diversity_gap": 0.0,
        "gate_mean": 0.0,
        "gate_target_mean": 0.0,
        "gate_target_gap": 0.0,
        "visual_evidence_aug_applied": 0.0,
        "visual_evidence_aug_none": 0.0,
        "visual_evidence_aug_flow_75": 0.0,
        "visual_evidence_aug_flow_50": 0.0,
        "visual_evidence_aug_flow_25": 0.0,
        "visual_evidence_aug_flow_zero": 0.0,
        "visual_evidence_aug_static_video_zero_flow": 0.0,
        "loss_d": 0.0,
        "psnr_full": 0.0,
        "psnr_missing": 0.0,
        "ssim_full": 0.0,
    }
    sample_count = 0
    ssim_sample_count = 0
    batch_count = 0
    skipped_batches = 0
    stop_training = False

    progress = tqdm(
        data_loader,
        desc=f"[VIAI-AV {phase}] epoch={global_epoch + 1}",
        unit="batch",
        dynamic_ncols=True,
    )
    for data in progress:
        if data is None:
            skipped_batches += 1
            progress.set_postfix(skipped_batches=skipped_batches)
            continue
        iter_start_time = time.time()
        model.get_blank_space_length(global_step)
        model.set_inputs(data)
        if train:
            model.optimize_parameters(global_step)
            global_step += 1
        else:
            model.test(global_step=global_step)
        model.get_loss_items()
        metrics = compute_viai_a_metrics(
            model.mel_pred,
            model.mel_target_4d,
            model.missing_mask,
            compute_ssim=_should_compute_ssim(train, global_step),
        )

        totals["loss_total"] += model.loss_total_item
        totals["loss_av_gen"] += model.loss_av_gen_item
        totals["loss_recon"] += model.loss_recon_item
        totals["loss_full_l1"] += model.loss_full_l1_item
        totals["loss_missing_l1"] += model.loss_missing_l1_item
        totals["loss_g_gan"] += model.loss_G_GAN_item
        totals["loss_sync"] += model.loss_sync_item
        totals["loss_probe_gen"] += model.loss_probe_gen_item
        totals["loss_probe_recon"] += model.loss_probe_recon_item
        totals["loss_probe_full_l1"] += model.loss_probe_full_l1_item
        totals["loss_probe_missing_l1"] += model.loss_probe_missing_l1_item
        totals["loss_probe_g_gan"] += model.loss_probe_G_GAN_item
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
        totals["candidate_pi_entropy"] += model.candidate_pi_entropy_item
        totals["candidate_pi_max"] += model.candidate_pi_max_item
        totals["candidate_pairwise_distance"] += model.candidate_pairwise_distance_item
        totals["evidence_diversity_gap"] += model.evidence_diversity_gap_item
        totals["gate_mean"] += model.gate_mean_item
        totals["gate_target_mean"] += model.gate_target_mean_item
        totals["gate_target_gap"] += model.gate_target_gap_item
        totals["visual_evidence_aug_applied"] += model.visual_evidence_aug_applied_item
        for mode, value in model.visual_evidence_aug_mode_items.items():
            totals[f"visual_evidence_aug_{mode}"] += value
        totals["loss_d"] += model.loss_D_item
        totals["psnr_full"] += metrics["psnr_full_sum"]
        totals["psnr_missing"] += metrics["psnr_missing_sum"]
        sample_count += metrics["num_samples"]
        if metrics["ssim_full_sum"] is not None:
            totals["ssim_full"] += metrics["ssim_full_sum"]
            ssim_sample_count += metrics["num_samples"]

        postfix = dict(
            step=global_step,
            loss=f"{model.loss_total_item:.4f}",
            recon=f"{model.loss_recon_item:.4f}",
            min_k=f"{model.loss_min_k_item:.4f}",
            top1=f"{model.top1_missing_l1_item:.4f}",
            u=f"{model.uncertainty_mean_item:.3f}",
            mean_k=f"{model.loss_mean_k_item:.4f}",
            gate=f"{model.gate_mean_item:.3f}",
            gate_t=f"{model.gate_target_mean_item:.3f}",
            aug=model.visual_evidence_aug_mode,
            pair=f"{model.candidate_pairwise_distance_item:.4f}",
            sync=f"{model.loss_sync_item:.4f}",
            probe=f"{model.loss_probe_gen_item:.4f}",
            g_gan=f"{model.loss_G_GAN_item:.4f}",
            d=f"{model.loss_D_item:.4f}",
            eta2=f"{model.eta2_item:.4f}",
            psnr=f"{metrics['psnr_full']:.2f}",
            psnr_miss=f"{metrics['psnr_missing']:.2f}",
            blank=model.blank_length,
        )
        if metrics["ssim_full"] is not None:
            postfix["ssim"] = f"{metrics['ssim_full']:.4f}"
        progress.set_postfix(**postfix)

        if train and global_step % hparams.print_freq == 0:
            elapsed = (time.time() - iter_start_time) / max(1, hparams.batch_size)
            ssim_text = (
                f" ssim={metrics['ssim_full']:.6f}"
                if metrics["ssim_full"] is not None
                else ""
            )
            tqdm.write(
                f"[VIAI-AV train] step={global_step} "
                f"loss={model.loss_total_item:.6f} "
                f"av_gen={model.loss_av_gen_item:.6f} "
                f"recon={model.loss_recon_item:.6f} "
                f"full_l1={model.loss_full_l1_item:.6f} "
                f"missing_l1={model.loss_missing_l1_item:.6f} "
                f"sync={model.loss_sync_item:.6f} "
                f"probe={model.loss_probe_gen_item:.6f} "
                f"probe_full_l1={model.loss_probe_full_l1_item:.6f} "
                f"probe_missing_l1={model.loss_probe_missing_l1_item:.6f} "
                f"min_k={model.loss_min_k_item:.6f} "
                f"mean_k={model.loss_mean_k_item:.6f} "
                f"boundary={model.loss_boundary_item:.6f} "
                f"evidence_div={model.loss_evidence_div_item:.6f} "
                f"gate_ev={model.loss_gate_evidence_item:.6f} "
                f"scorer={model.loss_candidate_scorer_item:.6f} "
                f"unc_calib={model.loss_uncertainty_calib_item:.6f} "
                f"calib={model.loss_calib_item:.6f} "
                f"multi={model.loss_multi_candidate_item:.6f} "
                f"weighted_calib={model.weighted_loss_calib_item:.6f} "
                f"best_of_k={model.best_of_k_missing_l1_item:.6f} "
                f"top1={model.top1_missing_l1_item:.6f} "
                f"candidate0={model.candidate0_missing_l1_item:.6f} "
                f"random={model.random_expected_missing_l1_item:.6f} "
                f"oracle_gain={model.oracle_gain_item:.6f} "
                f"uncertainty={model.uncertainty_mean_item:.6f} "
                f"pi_entropy={model.candidate_pi_entropy_item:.6f} "
                f"pi_max={model.candidate_pi_max_item:.6f} "
                f"pairwise={model.candidate_pairwise_distance_item:.6f} "
                f"div_gap={model.evidence_diversity_gap_item:.6f} "
                f"gate_mean={model.gate_mean_item:.6f} "
                f"gate_target={model.gate_target_mean_item:.6f} "
                f"gate_gap={model.gate_target_gap_item:.6f} "
                f"aug={model.visual_evidence_aug_mode} "
                f"g_gan={model.loss_G_GAN_item:.6f} "
                f"weighted_gan={model.weighted_loss_gan_item:.6f} "
                f"weighted_recon={model.weighted_loss_recon_item:.6f} "
                f"probe_g_gan={model.loss_probe_G_GAN_item:.6f} "
                f"d={model.loss_D_item:.6f} "
                f"eta1={model.eta1_item:.6f} "
                f"eta2={model.eta2_item:.6f} "
                f"psnr={metrics['psnr_full']:.3f} "
                f"psnr_missing={metrics['psnr_missing']:.3f}"
                f"{ssim_text} "
                f"time_per_sample={elapsed:.4f}s"
            )
        if train and global_step > 0 and global_step % hparams.checkpoint_interval == 0:
            model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)
        model.TF_writer(writer, global_step, prefix=phase)
        _write_monitoring(writer, phase, global_step, model, metrics)
        if _should_write_images(train, global_step, batch_count):
            write_mel_images(
                writer,
                phase,
                global_step,
                model.mel_input_4d,
                model.mel_pred,
                model.mel_target_4d,
                max_items=getattr(hparams, "tb_image_count", 4),
            )

        batch_count += 1
        if train and hparams.max_train_steps is not None and global_step >= hparams.max_train_steps:
            tqdm.write(f"Reached VIAI-AV smoke-test max_train_steps={hparams.max_train_steps}")
            stop_training = True
            break

    if batch_count == 0:
        raise RuntimeError(
            f"No valid VIAI-AV {phase} batches remained in epoch {global_epoch + 1}. "
            "Check the bad-sample CSV log for skipped clips."
        )
    averages = {
        "loss_total": totals["loss_total"] / batch_count,
        "loss_av_gen": totals["loss_av_gen"] / batch_count,
        "loss_recon": totals["loss_recon"] / batch_count,
        "loss_full_l1": totals["loss_full_l1"] / batch_count,
        "loss_missing_l1": totals["loss_missing_l1"] / batch_count,
        "loss_g_gan": totals["loss_g_gan"] / batch_count,
        "loss_sync": totals["loss_sync"] / batch_count,
        "loss_probe_gen": totals["loss_probe_gen"] / batch_count,
        "loss_probe_recon": totals["loss_probe_recon"] / batch_count,
        "loss_probe_full_l1": totals["loss_probe_full_l1"] / batch_count,
        "loss_probe_missing_l1": totals["loss_probe_missing_l1"] / batch_count,
        "loss_probe_g_gan": totals["loss_probe_g_gan"] / batch_count,
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
        "candidate_pi_entropy": totals["candidate_pi_entropy"] / batch_count,
        "candidate_pi_max": totals["candidate_pi_max"] / batch_count,
        "candidate_pairwise_distance": totals["candidate_pairwise_distance"] / batch_count,
        "evidence_diversity_gap": totals["evidence_diversity_gap"] / batch_count,
        "gate_mean": totals["gate_mean"] / batch_count,
        "gate_target_mean": totals["gate_target_mean"] / batch_count,
        "gate_target_gap": totals["gate_target_gap"] / batch_count,
        "visual_evidence_aug_applied": totals["visual_evidence_aug_applied"] / batch_count,
        "visual_evidence_aug_none": totals["visual_evidence_aug_none"] / batch_count,
        "visual_evidence_aug_flow_75": totals["visual_evidence_aug_flow_75"] / batch_count,
        "visual_evidence_aug_flow_50": totals["visual_evidence_aug_flow_50"] / batch_count,
        "visual_evidence_aug_flow_25": totals["visual_evidence_aug_flow_25"] / batch_count,
        "visual_evidence_aug_flow_zero": totals["visual_evidence_aug_flow_zero"] / batch_count,
        "visual_evidence_aug_static_video_zero_flow": (
            totals["visual_evidence_aug_static_video_zero_flow"] / batch_count
        ),
        "loss_d": totals["loss_d"] / batch_count,
        "psnr_full": totals["psnr_full"] / max(1, sample_count),
        "psnr_missing": totals["psnr_missing"] / max(1, sample_count),
        "ssim_full": None
        if ssim_sample_count == 0
        else totals["ssim_full"] / ssim_sample_count,
    }
    ssim_summary = (
        f" ssim={averages['ssim_full']:.6f}"
        if averages["ssim_full"] is not None
        else ""
    )
    tqdm.write(
        f"[VIAI-AV {phase}] "
        f"loss={averages['loss_total']:.6f} "
        f"av_gen={averages['loss_av_gen']:.6f} "
        f"recon={averages['loss_recon']:.6f} "
        f"full_l1={averages['loss_full_l1']:.6f} "
        f"missing_l1={averages['loss_missing_l1']:.6f} "
        f"sync={averages['loss_sync']:.6f} "
        f"probe={averages['loss_probe_gen']:.6f} "
        f"probe_full_l1={averages['loss_probe_full_l1']:.6f} "
        f"probe_missing_l1={averages['loss_probe_missing_l1']:.6f} "
        f"min_k={averages['loss_min_k']:.6f} "
        f"mean_k={averages['loss_mean_k']:.6f} "
        f"boundary={averages['loss_boundary']:.6f} "
        f"evidence_div={averages['loss_evidence_div']:.6f} "
        f"gate_ev={averages['loss_gate_evidence']:.6f} "
        f"scorer={averages['loss_candidate_scorer']:.6f} "
        f"unc_calib={averages['loss_uncertainty_calib']:.6f} "
        f"calib={averages['loss_calib']:.6f} "
        f"multi={averages['loss_multi_candidate']:.6f} "
        f"weighted_calib={averages['weighted_loss_calib']:.6f} "
        f"best_of_k={averages['best_of_k_missing_l1']:.6f} "
        f"top1={averages['top1_missing_l1']:.6f} "
        f"candidate0={averages['candidate0_missing_l1']:.6f} "
        f"random={averages['random_expected_missing_l1']:.6f} "
        f"oracle_gain={averages['oracle_gain']:.6f} "
        f"uncertainty={averages['uncertainty_mean']:.6f} "
        f"pi_entropy={averages['candidate_pi_entropy']:.6f} "
        f"pi_max={averages['candidate_pi_max']:.6f} "
        f"pairwise={averages['candidate_pairwise_distance']:.6f} "
        f"div_gap={averages['evidence_diversity_gap']:.6f} "
        f"gate_mean={averages['gate_mean']:.6f} "
        f"gate_target={averages['gate_target_mean']:.6f} "
        f"gate_gap={averages['gate_target_gap']:.6f} "
        f"aug_rate={averages['visual_evidence_aug_applied']:.3f} "
        f"g_gan={averages['loss_g_gan']:.6f} "
        f"probe_g_gan={averages['loss_probe_g_gan']:.6f} "
        f"d={averages['loss_d']:.6f} "
        f"psnr={averages['psnr_full']:.3f} "
        f"psnr_missing={averages['psnr_missing']:.3f}"
        f"{ssim_summary}"
    )
    if skipped_batches:
        tqdm.write(f"[VIAI-AV {phase}] skipped_bad_batches={skipped_batches}")
    return global_step, stop_training, averages


def train_loop(model, data_loaders, writer, start_step=0, start_epoch=0):
    global_step = start_step
    global_epoch = start_epoch
    while global_epoch < hparams.nepochs:
        if "train" in data_loaders:
            global_step, stop_training, _ = run_phase(
                model,
                data_loaders["train"],
                "train",
                global_step,
                writer,
                global_epoch,
            )
            if stop_training:
                break
        if "val" in data_loaders:
            with torch.no_grad():
                run_phase(model, data_loaders["val"], "val", global_step, writer, global_epoch)
        global_epoch += 1
        print(f"[VIAI-AV] epoch={global_epoch} current_lr={model.current_lr}")

    model.save_checkpoint(global_step, global_epoch, hparams.checkpoint_dir)


def main():
    configure_viai_av_defaults()
    os.makedirs(hparams.checkpoint_dir, exist_ok=True)
    print_viai_av_run_config()
    data_loaders = av_loader.get_data_loaders(hparams.data_root, hparams.speaker_id, phases=("train", "val"))
    model = VIAIAVModel(hparams, device=device)

    global_step = 0
    global_epoch = 0
    if hparams.resume:
        checkpoint_path = resolve_resume_checkpoint()
        global_step, global_epoch = model.load_checkpoint(
            checkpoint_path,
            reset_optimizer=hparams.reset_optimizer,
        )
        print(f"[VIAI-AV] resumed checkpoint: {checkpoint_path} step={global_step} epoch={global_epoch}")
    else:
        init_checkpoint = resolve_init_checkpoint()
        source_step, source_epoch = model.load_viai_a_checkpoint(init_checkpoint)
        print(
            f"[VIAI-AV] initialized from VIAI-A checkpoint "
            f"source_step={source_step} source_epoch={source_epoch}"
        )

    log_event_path = hparams.log_event_path
    if log_event_path is None:
        log_event_path = "log/run-test-" + hparams.name + str(datetime.now()).replace(" ", "_")
    print("TensorBoard event log path:", log_event_path)
    writer = SummaryWriter(log_dir=log_event_path)

    try:
        train_loop(model, data_loaders, writer, start_step=global_step, start_epoch=global_epoch)
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        writer.close()
    print("Finished VIAI-AV training")


if __name__ == "__main__":
    main()
    sys.exit(0)
