import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel
from utils import util
from utils.baseline_evaluation import (
    aggregate_sample_records,
    build_run_metadata,
    compute_sample_records,
    seed_everything,
    select_prediction,
    utc_now,
    validate_record_coverage,
    write_run_metadata,
    write_standard_results,
)
from utils.baseline_protocol import (
    load_mask_manifest,
    resolve_mask_specs,
    sha256_file,
)
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
    "prediction_branch",
    "mask_manifest_sha256",
    "stage",
    "use_gan",
    "enable_sync_loss",
    "enable_probe_loss",
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
    "eta1",
    "eta2",
    "lambda_recon",
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
        if getattr(hparams, "eval_branch", "av") == "probe":
            hparams.name = "VIAI-AA-Prime"
        else:
            hparams.name = (
                "VIAI-AV-PatchGAN"
                if getattr(hparams, "use_gan", False)
                else "VIAI-AV"
            )
    if not _arg_was_passed("--results_dir"):
        hparams.results_dir = (
            "./checkpoints/viai_av_patchgan_test_results"
            if getattr(hparams, "use_gan", False)
            else "./checkpoints/viai_av_test_results"
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


def batch_metrics(model, branch="av"):
    prediction = select_prediction(model, branch)
    mel_completed = compose_inpainted_mel(
        model.mel_input_4d,
        prediction,
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


def mel_image_output_dir(results_dir, checkpoint_step_value):
    return os.path.join(
        results_dir,
        "mel-image",
        f"step{format_step(checkpoint_step_value)}",
    )


def evaluate(
    model,
    data_loader,
    global_step=0,
    image_dir=None,
    vocoder_dir=None,
    mask_manifest=None,
    prediction_branch="av",
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
    sample_records = []

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
        runtime_paths = data[-1]
        mask_specs = None
        if mask_manifest is not None:
            mask_specs = resolve_mask_specs(
                runtime_paths,
                mask_manifest,
                data_root=hparams.data_root,
            )
        else:
            model.get_blank_space_length(0)
        model.set_inputs(data, mask_specs=mask_specs)
        model.test(global_step=global_step)
        model.get_loss_items()
        prediction = select_prediction(model, prediction_branch)
        metrics = batch_metrics(model, branch=prediction_branch)
        batch_size = metrics["num_samples"]

        if prediction_branch == "probe":
            selected_gen = model.loss_probe_gen_item
            selected_recon = model.loss_probe_recon_item
            selected_g_gan = model.loss_probe_G_GAN_item
            selected_full_l1 = model.loss_probe_full_l1_item
            selected_missing_l1 = model.loss_probe_missing_l1_item
        else:
            selected_gen = model.loss_av_gen_item
            selected_recon = model.loss_recon_item
            selected_g_gan = model.loss_G_GAN_item
            selected_full_l1 = model.loss_full_l1_item
            selected_missing_l1 = model.loss_missing_l1_item
        totals["loss_total"] += model.loss_total_item * batch_size
        totals["loss_av_gen"] += selected_gen * batch_size
        totals["loss_recon"] += selected_recon * batch_size
        totals["loss_g_gan"] += selected_g_gan * batch_size
        totals["loss_sync"] += model.loss_sync_item * batch_size
        totals["loss_probe_gen"] += model.loss_probe_gen_item * batch_size
        totals["loss_probe_recon"] += model.loss_probe_recon_item * batch_size
        totals["loss_probe_g_gan"] += model.loss_probe_G_GAN_item * batch_size
        totals["loss_d"] += model.loss_D_item * batch_size
        totals["eta1"] += model.eta1_item * batch_size
        totals["eta2"] += model.eta2_item * batch_size
        totals["full_l1"] += selected_full_l1 * batch_size
        totals["missing_l1"] += selected_missing_l1 * batch_size
        totals["probe_full_l1"] += model.loss_probe_full_l1_item * batch_size
        totals["probe_missing_l1"] += model.loss_probe_missing_l1_item * batch_size
        totals["full_psnr"] += metrics["full_psnr"]
        totals["missing_psnr"] += metrics["missing_psnr"]
        totals["ssim"] += metrics["ssim"]
        audio_embeddings.append(util.to_np(model.mel_net_norm))
        video_embeddings.append(util.to_np(model.video_net_norm))
        sample_count += batch_size
        batch_count += 1
        if mask_specs is not None:
            sample_records.extend(
                compute_sample_records(
                    runtime_paths,
                    mask_specs,
                    "viai_av" if prediction_branch == "av" else "viai_aa_probe",
                    model.mel_input_4d,
                    prediction,
                    model.mel_target_4d,
                    model.missing_mask,
                    data_root=hparams.data_root,
                )
            )
        if image_dir is not None:
            save_mel_comparison_batch(
                image_dir,
                sample_count - batch_size,
                model.path_batch,
                model.mel_input_4d,
                prediction,
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
                    prediction,
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
    results = {
        "loss_total": totals["loss_total"] / sample_count,
        "loss_av_gen": totals["loss_av_gen"] / sample_count,
        "loss_recon": totals["loss_recon"] / sample_count,
        "loss_g_gan": totals["loss_g_gan"] / sample_count,
        "loss_sync": totals["loss_sync"] / sample_count,
        "loss_probe_gen": totals["loss_probe_gen"] / sample_count,
        "loss_probe_recon": totals["loss_probe_recon"] / sample_count,
        "loss_probe_g_gan": totals["loss_probe_g_gan"] / sample_count,
        "loss_d": totals["loss_d"] / sample_count,
        "eta1": totals["eta1"] / sample_count,
        "eta2": totals["eta2"] / sample_count,
        "mel_l1_full": totals["full_l1"] / sample_count,
        "mel_l1_missing": totals["missing_l1"] / sample_count,
        "probe_l1_full": totals["probe_full_l1"] / sample_count,
        "probe_l1_missing": totals["probe_missing_l1"] / sample_count,
        "psnr_full": totals["full_psnr"] / sample_count,
        "psnr_missing": totals["missing_psnr"] / sample_count,
        "ssim": totals["ssim"] / sample_count,
        "num_samples": sample_count,
        "retrieval_audio_to_video": audio_to_video,
        "retrieval_video_to_audio": video_to_audio,
        "skipped_batches": skipped_batches,
        "vocoder_output_dir": "" if vocoder_dir is None else vocoder_dir,
        "vocoder_num_samples": vocoder_count,
        "sample_records": sample_records,
    }
    if mask_manifest is not None:
        validate_record_coverage(sample_records, mask_manifest)
        results.update(aggregate_sample_records(sample_records))
    return results


def build_result_record(
    checkpoint_path,
    checkpoint_step_value,
    global_step,
    global_epoch,
    results,
    model,
):
    audio_to_video = results["retrieval_audio_to_video"]
    video_to_audio = results["retrieval_video_to_audio"]
    return {
        "checkpoint_path": os.path.abspath(checkpoint_path),
        "checkpoint_step": int(checkpoint_step_value),
        "global_step": int(global_step),
        "global_epoch": int(global_epoch),
        "test_split_name": hparams.test_split_name,
        "prediction_branch": (
            "viai_av" if hparams.eval_branch == "av" else "viai_aa_probe"
        ),
        "mask_manifest_sha256": (
            sha256_file(hparams.baseline_mask_manifest)
            if getattr(hparams, "baseline_mask_manifest", None)
            else ""
        ),
        "stage": getattr(model, "checkpoint_stage", "VIAI-AV-stage4-sync-probe"),
        "use_gan": bool(getattr(hparams, "use_gan", False)),
        "enable_sync_loss": not bool(getattr(hparams, "disable_sync_loss", False)),
        "enable_probe_loss": not bool(getattr(hparams, "disable_probe_loss", False)),
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
        "eta1": float(results["eta1"]),
        "eta2": float(results["eta2"]),
        "lambda_recon": float(getattr(hparams, "lambda_recon", 1.0)),
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
        "num_samples",
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
        "eta1",
        "eta2",
        "lambda_recon",
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
    started_at = utc_now()
    configure_viai_av_defaults()
    if getattr(hparams, "deterministic_eval", False) or getattr(
        hparams, "baseline_mask_manifest", None
    ):
        seed_everything(hparams.eval_seed)
    mask_manifest = None
    if getattr(hparams, "baseline_mask_manifest", None):
        mask_manifest = load_mask_manifest(hparams.baseline_mask_manifest)
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
    if hparams.eval_branch == "probe" and not getattr(
        model, "checkpoint_enable_probe_loss", False
    ):
        raise RuntimeError(
            "The selected VIAI-AV checkpoint was not saved with probe loss enabled; "
            "it cannot be frozen as the VIAI-AA' reference."
        )
    print(f"[VIAI-AV test] loaded checkpoint: {checkpoint_path} (step={global_step}, epoch={global_epoch})")

    checkpoint_step_value = checkpoint_step(checkpoint_path)
    if checkpoint_step_value < 0:
        checkpoint_step_value = global_step
    if mask_manifest is None:
        image_dir = mel_image_output_dir(hparams.results_dir, checkpoint_step_value)
    else:
        image_dir = os.path.join(hparams.results_dir, "mel-image")
    vocoder_dir = None
    if getattr(hparams, "use_vocoder", False):
        vocoder_dir = hparams.vocoder_output_dir
        if not vocoder_dir:
            if mask_manifest is None:
                vocoder_dir = os.path.join(
                    hparams.results_dir,
                    "wav",
                    f"step{format_step(checkpoint_step_value)}",
                )
            else:
                vocoder_dir = os.path.join(hparams.results_dir, "wav")
    results = evaluate(
        model,
        data_loaders["test"],
        global_step=checkpoint_step_value,
        image_dir=image_dir,
        vocoder_dir=vocoder_dir,
        mask_manifest=mask_manifest,
        prediction_branch=hparams.eval_branch,
    )
    result_record = build_result_record(
        checkpoint_path,
        checkpoint_step_value,
        global_step,
        global_epoch,
        results,
        model,
    )
    json_path, csv_path = write_result_files(
        result_record,
        hparams.results_dir,
        hparams.name,
    )
    if mask_manifest is not None:
        standard_summary = dict(result_record)
        standard_summary["known_region_max_abs_error"] = float(
            results["known_region_max_abs_error"]
        )
        summary_path, samples_path, metrics_path = write_standard_results(
            hparams.results_dir,
            results["sample_records"],
            standard_summary,
        )
        metadata = build_run_metadata(
            checkpoint_path,
            vars(hparams),
            sys.argv,
            hparams.eval_seed,
            started_at,
            Path(__file__).resolve().parent,
            mask_manifest_path=hparams.baseline_mask_manifest,
            protocol_path=getattr(hparams, "baseline_protocol_json", None),
            prediction_branch=result_record["prediction_branch"],
        )
        metadata_path = write_run_metadata(hparams.results_dir, metadata)
        print(f"[VIAI-AV test] wrote baseline summary: {summary_path}")
        print(f"[VIAI-AV test] wrote per-sample JSONL: {samples_path}")
        print(f"[VIAI-AV test] wrote per-sample CSV: {metrics_path}")
        print(f"[VIAI-AV test] wrote run metadata: {metadata_path}")
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
