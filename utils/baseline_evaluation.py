import csv
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from utils.baseline_protocol import canonical_sample_id, sha256_file
from utils.viai_a_metrics import compose_inpainted_mel, structural_similarity_2d


SAMPLE_FIELDS = [
    "sample_id",
    "prediction_branch",
    "mask_type",
    "start",
    "end",
    "gap_frames",
    "mel_l1_full",
    "mel_l1_missing",
    "psnr_full",
    "psnr_missing",
    "ssim",
    "known_region_max_abs_error",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def seed_everything(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def select_prediction(model, branch):
    if branch == "av":
        return model.mel_pred
    if branch == "probe":
        if not model.enable_probe_loss or model.mel_probe_pred is None:
            raise RuntimeError(
                "VIAI-AA' probe evaluation requires an enabled probe branch."
            )
        return model.mel_probe_pred
    raise ValueError(f"Unknown evaluation branch: {branch}")


def _as_bchw(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    if tensor.dim() == 4:
        return tensor
    raise ValueError("Expected Mel tensor with shape (B,C,T) or (B,1,C,T)")


def compute_sample_records(
    sample_paths,
    mask_specs,
    prediction_branch,
    mel_input,
    mel_prediction,
    mel_target,
    missing_mask,
    data_root=None,
    known_region_tolerance=1e-7,
):
    mel_input = _as_bchw(mel_input).detach()
    mel_prediction = _as_bchw(mel_prediction).detach()
    mel_target = _as_bchw(mel_target).detach()
    missing_mask = _as_bchw(missing_mask).detach().to(
        device=mel_prediction.device,
        dtype=mel_prediction.dtype,
    )
    completed = compose_inpainted_mel(mel_input, mel_prediction, missing_mask)
    known_mask = 1.0 - missing_mask
    records = []
    for index, spec in enumerate(mask_specs):
        pred = torch.clamp(completed[index], 0.0, 1.0)
        target = torch.clamp(mel_target[index], 0.0, 1.0)
        mask = missing_mask[index]
        known_error = torch.max(
            torch.abs(completed[index] - mel_input[index]) * known_mask[index]
        )
        known_error_value = float(known_error.cpu().item())
        if known_error_value > float(known_region_tolerance):
            raise RuntimeError(
                f"Known-region preservation failed for {sample_paths[index]}: "
                f"max_abs_error={known_error_value:.12g}"
            )
        diff = pred - target
        full_l1 = torch.mean(torch.abs(diff))
        missing_count = torch.clamp(mask.sum(), min=1.0)
        missing_l1 = (torch.abs(diff) * mask).sum() / missing_count
        full_mse = torch.mean(diff ** 2)
        missing_mse = ((diff ** 2) * mask).sum() / missing_count
        psnr_full = -10.0 * torch.log10(torch.clamp(full_mse, min=1e-12))
        psnr_missing = -10.0 * torch.log10(torch.clamp(missing_mse, min=1e-12))
        pred_np = pred.squeeze(0).cpu().numpy()
        target_np = target.squeeze(0).cpu().numpy()
        records.append(
            {
                "sample_id": canonical_sample_id(
                    sample_paths[index],
                    data_root=data_root,
                    strip_window_index=True,
                ),
                "prediction_branch": prediction_branch,
                "mask_type": spec["mask_type"],
                "start": int(spec["start"]),
                "end": int(spec["end"]),
                "gap_frames": int(spec["gap_frames"]),
                "mel_l1_full": float(full_l1.cpu().item()),
                "mel_l1_missing": float(missing_l1.cpu().item()),
                "psnr_full": float(psnr_full.cpu().item()),
                "psnr_missing": float(psnr_missing.cpu().item()),
                "ssim": float(structural_similarity_2d(pred_np, target_np)),
                "known_region_max_abs_error": known_error_value,
            }
        )
    return records


def aggregate_sample_records(records):
    if not records:
        raise ValueError("Cannot aggregate empty sample records")
    metric_names = [
        "mel_l1_full",
        "mel_l1_missing",
        "psnr_full",
        "psnr_missing",
        "ssim",
        "known_region_max_abs_error",
    ]
    summary = {"num_samples": len(records)}
    for name in metric_names:
        values = [float(record[name]) for record in records]
        summary[name] = (
            max(values)
            if name == "known_region_max_abs_error"
            else sum(values) / len(values)
        )
    return summary


def validate_record_coverage(records, manifest):
    actual = [record["sample_id"] for record in records]
    if len(actual) != len(set(actual)):
        raise ValueError("Evaluation produced duplicate sample_id records")
    expected = set(manifest)
    actual_set = set(actual)
    if actual_set != expected:
        missing = sorted(expected - actual_set)
        extra = sorted(actual_set - expected)
        raise ValueError(
            "Evaluation sample coverage does not match the mask manifest: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )


def write_standard_results(results_dir, records, summary):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    samples_path = results_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
    metrics_path = results_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in SAMPLE_FIELDS})
    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return summary_path, samples_path, metrics_path


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def git_metadata(repo_root):
    repo_root = str(repo_root)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status": []}
    return {"commit": commit, "dirty": bool(status), "status": status}


def build_run_metadata(
    checkpoint_path,
    config,
    command,
    seed,
    started_at,
    repo_root,
    mask_manifest_path=None,
    protocol_path=None,
    prediction_branch=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    recorded_command = list(command)
    command_override = os.environ.get("VIAI_BASELINE_COMMAND_JSON")
    if command_override:
        try:
            parsed_command = json.loads(command_override)
            if isinstance(parsed_command, list) and all(
                isinstance(item, str) for item in parsed_command
            ):
                recorded_command = parsed_command
        except json.JSONDecodeError:
            pass
    recorded_git = git_metadata(repo_root)
    git_override = os.environ.get("VIAI_BASELINE_GIT_JSON")
    if git_override:
        try:
            parsed_git = json.loads(git_override)
            if isinstance(parsed_git, dict):
                recorded_git = parsed_git
        except json.JSONDecodeError:
            pass
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_branch": prediction_branch,
        "git": recorded_git,
        "command": recorded_command,
        "config": _json_safe(config),
        "seed": int(seed),
        "started_at": started_at,
        "ended_at": utc_now(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
    }
    if mask_manifest_path:
        metadata["mask_manifest"] = {
            "path": str(Path(mask_manifest_path).resolve()),
            "sha256": sha256_file(mask_manifest_path),
        }
    if protocol_path:
        protocol_path = Path(protocol_path).resolve()
        metadata["protocol"] = {
            "path": str(protocol_path),
            "sha256": sha256_file(protocol_path),
        }
        with protocol_path.open("r", encoding="utf-8") as handle:
            protocol = json.load(handle)
        metadata["split_hashes"] = {
            model_name: {
                phase: split_record["sha256"]
                for phase, split_record in phases.items()
            }
            for model_name, phases in protocol.get("splits", {}).items()
        }
    return metadata


def write_run_metadata(results_dir, metadata):
    path = Path(results_dir) / "run_metadata.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return path
