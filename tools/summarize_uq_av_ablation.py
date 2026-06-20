"""Summarize UQ-AV inference-time video ablations.

Reads five paired ``test-uq-av`` result directories and reports per-condition
metrics plus original-vs-ablation paired deltas.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


CONDITIONS = (
    "original",
    "no_video",
    "wrong_video",
    "zero_token",
    "shuffled_video",
)

METRICS = (
    "psnr_missing_db",
    "mel_l1_missing",
    "ssim_full",
    "boundary_l1",
)

PAIR_FIELDS = (
    "sample_id",
    "mask_type",
    "start",
    "end",
    "gap_frames",
)

HIGHER_IS_BETTER = {
    "psnr_missing_db": True,
    "mel_l1_missing": False,
    "ssim_full": True,
    "boundary_l1": False,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize paired UQ-AV video-condition ablations."
    )
    parser.add_argument("--original", required=True)
    parser.add_argument("--no-video", required=True)
    parser.add_argument("--wrong-video", required=True)
    parser.add_argument("--zero-token", required=True)
    parser.add_argument("--shuffled-video", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def _condition_dirs(args):
    return {
        "original": Path(args.original),
        "no_video": Path(args.no_video),
        "wrong_video": Path(args.wrong_video),
        "zero_token": Path(args.zero_token),
        "shuffled_video": Path(args.shuffled_video),
    }


def _read_summary(results_dir):
    path = results_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing summary.json: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_metrics(results_dir):
    path = results_dir / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing metrics.csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError(f"metrics.csv contains no rows: {path}")
    for field in PAIR_FIELDS + METRICS:
        if field not in records[0]:
            raise ValueError(f"Missing required column {field!r} in {path}")
    return records


def _pair_key(record):
    return tuple(str(record[field]) for field in PAIR_FIELDS)


def _metric_value(record, metric):
    try:
        return float(record[metric])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid value for metric {metric!r} in record "
            f"{_pair_key(record)}: {record.get(metric)!r}"
        ) from exc


def _stats(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("Cannot compute statistics for an empty value list")
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _validate_pairing(records_by_condition):
    original_keys = [_pair_key(record) for record in records_by_condition["original"]]
    if len(original_keys) != len(set(original_keys)):
        raise ValueError("original metrics.csv contains duplicate paired keys")

    for condition in CONDITIONS:
        records = records_by_condition[condition]
        keys = [_pair_key(record) for record in records]
        if len(keys) != len(set(keys)):
            raise ValueError(f"{condition} metrics.csv contains duplicate paired keys")
        if keys != original_keys:
            raise ValueError(
                "Ablation results are not fully paired in the same order: "
                f"condition={condition}"
            )


def _condition_stats(records_by_condition):
    output = {}
    for condition, records in records_by_condition.items():
        output[condition] = {}
        for metric in METRICS:
            output[condition][metric] = _stats(
                _metric_value(record, metric) for record in records
            )
    return output


def _paired_deltas(records_by_condition):
    original_records = records_by_condition["original"]
    rows = []
    for condition in CONDITIONS:
        if condition == "original":
            continue
        condition_records = records_by_condition[condition]
        for original, ablated in zip(original_records, condition_records):
            for metric in METRICS:
                original_value = _metric_value(original, metric)
                condition_value = _metric_value(ablated, metric)
                raw_delta = original_value - condition_value
                if HIGHER_IS_BETTER[metric]:
                    advantage = raw_delta
                else:
                    advantage = condition_value - original_value
                row = {
                    "comparison_condition": condition,
                    "metric": metric,
                    "original_value": original_value,
                    "condition_value": condition_value,
                    "raw_delta_original_minus_condition": raw_delta,
                    "original_advantage": advantage,
                }
                for field in PAIR_FIELDS:
                    row[field] = original[field]
                rows.append(row)
    return rows


def _comparison_stats(delta_rows):
    output = {}
    for condition in CONDITIONS:
        if condition == "original":
            continue
        output[condition] = {}
        for metric in METRICS:
            values = [
                row["original_advantage"]
                for row in delta_rows
                if row["comparison_condition"] == condition
                and row["metric"] == metric
            ]
            stats = _stats(values)
            wins = sum(1 for value in values if value > 0.0)
            stats["win_rate"] = wins / len(values)
            output[condition][metric] = stats
    return output


def _write_condition_summary_csv(path, condition_stats, comparison_stats):
    fieldnames = (
        "row_type",
        "condition",
        "metric",
        "count",
        "mean",
        "std",
        "median",
        "min",
        "max",
        "win_rate",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for condition in CONDITIONS:
            for metric in METRICS:
                row = {
                    "row_type": "condition",
                    "condition": condition,
                    "metric": metric,
                    "win_rate": "",
                }
                row.update(condition_stats[condition][metric])
                writer.writerow(row)
        for condition in CONDITIONS:
            if condition == "original":
                continue
            for metric in METRICS:
                row = {
                    "row_type": "original_advantage",
                    "condition": condition,
                    "metric": metric,
                }
                row.update(comparison_stats[condition][metric])
                writer.writerow(row)


def _write_delta_csv(path, delta_rows):
    fieldnames = (
        "sample_id",
        "mask_type",
        "start",
        "end",
        "gap_frames",
        "comparison_condition",
        "metric",
        "original_value",
        "condition_value",
        "raw_delta_original_minus_condition",
        "original_advantage",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(delta_rows)


def summarize(args):
    condition_dirs = _condition_dirs(args)
    summaries = {
        condition: _read_summary(results_dir)
        for condition, results_dir in condition_dirs.items()
    }
    records_by_condition = {
        condition: _read_metrics(results_dir)
        for condition, results_dir in condition_dirs.items()
    }

    _validate_pairing(records_by_condition)
    condition_stats = _condition_stats(records_by_condition)
    delta_rows = _paired_deltas(records_by_condition)
    comparison_stats = _comparison_stats(delta_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "conditions": {
            condition: {
                "results_dir": str(condition_dirs[condition]),
                "summary": summaries[condition],
                "metrics": condition_stats[condition],
            }
            for condition in CONDITIONS
        },
        "original_advantage": comparison_stats,
        "paired_fields": list(PAIR_FIELDS),
        "metrics": list(METRICS),
    }

    summary_json = output_dir / "ablation_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    summary_csv = output_dir / "ablation_summary.csv"
    _write_condition_summary_csv(
        summary_csv, condition_stats, comparison_stats,
    )

    deltas_csv = output_dir / "paired_deltas.csv"
    _write_delta_csv(deltas_csv, delta_rows)

    return {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "paired_deltas_csv": deltas_csv,
        "payload": payload,
    }


def main(argv=None):
    args = parse_args(argv)
    outputs = summarize(args)
    print("[summarize-uq-av-ablation] wrote:")
    print(f"  {outputs['summary_json']}")
    print(f"  {outputs['summary_csv']}")
    print(f"  {outputs['paired_deltas_csv']}")


if __name__ == "__main__":
    main()
