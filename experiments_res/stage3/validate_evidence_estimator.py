#!/usr/bin/env python3
"""Validate Stage 3 visual evidence scores under controlled perturbations."""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run paired perturbation checks for VisualEvidenceEstimator. "
            "The main expected trend is original > weakened/zero/static flow."
        )
    )
    parser.add_argument("--data_root", type=str, default="/root/shared-nvme/data")
    parser.add_argument("--test_split_name", type=str, default="test_av_split.txt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/evidence_estimator_validation")
    parser.add_argument("--max_anchors", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--device", type=str, default=None, choices=[None, "cpu", "cuda"])
    return parser.parse_args()


def instrument_from_path(path):
    parts = str(path).split("/")
    if "processed" in parts:
        index = parts.index("processed")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def batch_path(batch):
    return batch[-1][0]


def batch_instrument(batch):
    return instrument_from_path(batch_path(batch))


def make_batch(video_batch, flow_batch, anchor_batch):
    _, _, c_batch, x_batch, y_batch, g_batch, input_lengths, path_batch = anchor_batch
    return (
        video_batch,
        flow_batch,
        c_batch,
        x_batch,
        y_batch,
        g_batch,
        input_lengths,
        path_batch,
    )


def static_first_frame(tensor):
    return tensor[:, :1].expand_as(tensor).contiguous()


def evidence_for(model, batch):
    model.get_blank_space_length(0)
    model.set_inputs(batch)
    model.test(global_step=0)
    return float(model.evidence_score.mean().detach().cpu())


def build_hparams(args):
    import Options_inpainting

    cli_args = [
        "--data_root",
        args.data_root,
        "--test_split_name",
        args.test_split_name,
        "--batch_size",
        "1",
        "--num_workers",
        str(args.num_workers),
    ]
    if args.use_gan:
        cli_args.append("--use_gan")
    return Options_inpainting.Inpainting_Config(force_reload=True, args=cli_args)


def get_test_loader(hparams):
    from Data_loaders import audio_loader as av_loader

    data_loaders = av_loader.get_data_loaders(
        hparams.data_root,
        hparams.speaker_id,
        test_shuffle=False,
        phases=("test",),
    )
    if "test" not in data_loaders:
        raise RuntimeError(f"Missing test split: {hparams.test_split_name}")
    return data_loaders["test"]


def collect_reference_batches(loader):
    references = {}
    for batch in loader:
        if batch is None:
            continue
        instrument = batch_instrument(batch)
        if instrument not in references:
            references[instrument] = batch
    if len(references) < 2:
        raise RuntimeError("Need at least two instruments to build cross-instrument wrong video.")
    return references


def choose_wrong_reference(references, anchor_instrument):
    for instrument, batch in references.items():
        if instrument != anchor_instrument:
            return instrument, batch
    raise RuntimeError(f"No wrong-instrument reference found for {anchor_instrument}.")


def paired_delta(rows, condition_a, condition_b):
    by_sample = defaultdict(dict)
    for row in rows:
        by_sample[int(row["sample_index"])][row["condition"]] = float(row["evidence"])

    deltas = []
    hits = 0
    for record in by_sample.values():
        if condition_a not in record or condition_b not in record:
            continue
        delta = record[condition_a] - record[condition_b]
        deltas.append(delta)
        hits += int(delta > 0.0)

    values = np.asarray(deltas, dtype=np.float64)
    total = int(values.size)
    return {
        "mean_delta": None if total == 0 else float(values.mean()),
        "std_delta": None if total == 0 else float(values.std()),
        "hit_rate": 0.0 if total == 0 else float(hits / total),
        "n": total,
    }


def summarize(rows):
    conditions = sorted({row["condition"] for row in rows})
    condition_means = {
        condition: float(
            np.mean([float(row["evidence"]) for row in rows if row["condition"] == condition])
        )
        for condition in conditions
    }
    paired_checks = {
        "original_gt_flow_75": paired_delta(rows, "original", "flow_75"),
        "flow_75_gt_flow_50": paired_delta(rows, "flow_75", "flow_50"),
        "flow_50_gt_flow_25": paired_delta(rows, "flow_50", "flow_25"),
        "flow_25_gt_flow_zero": paired_delta(rows, "flow_25", "flow_zero"),
        "original_gt_flow_zero": paired_delta(rows, "original", "flow_zero"),
        "original_gt_static_flow": paired_delta(rows, "original", "static_flow"),
        "original_gt_static_video_zero_flow": paired_delta(
            rows,
            "original",
            "static_video_zero_flow",
        ),
        "original_gt_cross_instrument_wrong_video": paired_delta(
            rows,
            "original",
            "cross_instrument_wrong_video",
        ),
        "original_gt_temporal_shift_aux": paired_delta(rows, "original", "temporal_shift_aux"),
    }
    return {
        "num_anchor_samples": len({int(row["sample_index"]) for row in rows}),
        "condition_means": condition_means,
        "paired_checks": paired_checks,
    }


def write_outputs(rows, summary, out_dir):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "evidence_validation.csv"
    summary_path = out_path / "evidence_validation_summary.json"

    fieldnames = [
        "sample_index",
        "sample_path",
        "instrument",
        "wrong_instrument",
        "condition",
        "evidence",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    return csv_path, summary_path


def print_summary(summary):
    print("condition means:")
    for condition, value in sorted(summary["condition_means"].items()):
        print(f"  {condition:36s} {value:.6f}")
    print("\npaired checks:")
    for name, stats in summary["paired_checks"].items():
        mean_delta = stats["mean_delta"]
        mean_text = "nan" if mean_delta is None else f"{mean_delta:.6f}"
        print(
            f"  {name:45s} "
            f"mean_delta={mean_text} "
            f"hit_rate={stats['hit_rate']:.3f} "
            f"n={stats['n']}"
        )


def main():
    args = parse_args()
    hparams = build_hparams(args)
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    references = collect_reference_batches(get_test_loader(hparams))

    from Models.VIAI_AV_inpainting import VIAIAVModel

    model = VIAIAVModel(hparams, device=device)
    model.load_checkpoint(args.checkpoint, reset_optimizer=True)

    rows = []
    processed = 0
    for anchor_batch in get_test_loader(hparams):
        if anchor_batch is None:
            continue
        anchor_instrument = batch_instrument(anchor_batch)
        wrong_instrument, wrong_batch = choose_wrong_reference(references, anchor_instrument)

        video_batch, flow_batch = anchor_batch[0], anchor_batch[1]
        wrong_video_batch, wrong_flow_batch = wrong_batch[0], wrong_batch[1]
        conditions = {
            "original": anchor_batch,
            "flow_75": make_batch(video_batch, flow_batch * 0.75, anchor_batch),
            "flow_50": make_batch(video_batch, flow_batch * 0.50, anchor_batch),
            "flow_25": make_batch(video_batch, flow_batch * 0.25, anchor_batch),
            "flow_zero": make_batch(video_batch, torch.zeros_like(flow_batch), anchor_batch),
            "static_flow": make_batch(video_batch, static_first_frame(flow_batch), anchor_batch),
            "static_video_zero_flow": make_batch(
                static_first_frame(video_batch),
                torch.zeros_like(flow_batch),
                anchor_batch,
            ),
            "temporal_shift_aux": make_batch(
                video_batch.roll(shifts=5, dims=1),
                flow_batch.roll(shifts=5, dims=1),
                anchor_batch,
            ),
            "cross_instrument_wrong_video": make_batch(
                wrong_video_batch,
                wrong_flow_batch,
                anchor_batch,
            ),
        }

        for condition, current_batch in conditions.items():
            rows.append(
                {
                    "sample_index": processed,
                    "sample_path": batch_path(anchor_batch),
                    "instrument": anchor_instrument,
                    "wrong_instrument": wrong_instrument,
                    "condition": condition,
                    "evidence": evidence_for(model, current_batch),
                }
            )

        processed += 1
        if processed >= args.max_anchors:
            break

    if not rows:
        raise RuntimeError("No valid test samples were processed.")

    summary = summarize(rows)
    csv_path, summary_path = write_outputs(rows, summary, args.out_dir)
    print(f"wrote: {csv_path}")
    print(f"wrote: {summary_path}\n")
    print_summary(summary)


if __name__ == "__main__":
    main()
