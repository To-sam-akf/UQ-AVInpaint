"""Summarize UQ-AV K=1 sampling-quality sweeps.

Each run points to either a ``summary.json`` file or a result directory
containing one. The tool writes a compact JSON payload and CSV table for
schedule, DDIM-step, prediction-type, and EMA comparisons.
"""

import argparse
import csv
import json
from pathlib import Path


REQUIRED_FIELDS = (
    "psnr_missing_db",
    "mel_l1_missing",
    "known_region_max_abs_error_max",
    "inference_steps",
)

OUTPUT_FIELDS = (
    "label",
    "summary_path",
    "checkpoint",
    "num_samples",
    "psnr_missing_db",
    "mel_l1_missing",
    "psnr_full_db",
    "mel_l1_full",
    "ssim_full",
    "boundary_l1",
    "known_region_max_abs_error_max",
    "uq_beta_schedule",
    "uq_prediction_type",
    "uq_latent_clip_value",
    "inference_steps",
    "ddim_eta",
    "latent_is_normalised",
    "uq_ema_eval",
    "elapsed_seconds",
    "seconds_per_sample",
    "samples_per_second",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize paired or independent UQ-AV sampling sweeps."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("LABEL", "SUMMARY_OR_DIR"),
        required=True,
        help="Run label plus path to summary.json or its result directory.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _summary_path(path_value):
    path = Path(path_value)
    if path.is_dir():
        path = path / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary.json: {path}")
    return path


def _read_summary(label, path_value):
    path = _summary_path(path_value)
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    missing = [field for field in REQUIRED_FIELDS if field not in summary]
    if missing:
        raise ValueError(
            f"Summary for run {label!r} is missing required fields: {missing}"
        )
    row = {"label": label, "summary_path": str(path)}
    for field in OUTPUT_FIELDS:
        if field in row:
            continue
        row[field] = summary.get(field, "")
    return row, summary


def summarize(args):
    rows = []
    summaries = {}
    for label, path_value in args.run:
        row, summary = _read_summary(label, path_value)
        rows.append(row)
        summaries[label] = summary

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "runs": rows,
        "summaries": summaries,
        "required_fields": list(REQUIRED_FIELDS),
    }

    json_path = output_dir / "sampling_sweep_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    csv_path = output_dir / "sampling_sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    return {
        "summary_json": json_path,
        "summary_csv": csv_path,
        "payload": payload,
    }


def main(argv=None):
    outputs = summarize(parse_args(argv))
    print("[summarize-uq-av-sampling-sweep] wrote:")
    print(f"  {outputs['summary_json']}")
    print(f"  {outputs['summary_csv']}")


if __name__ == "__main__":
    main()
