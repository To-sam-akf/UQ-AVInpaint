#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose target-conditioned semantic evidence for wrong-video tests."
    )
    parser.add_argument("--semantic-jsonl", required=True)
    parser.add_argument("--sample-metrics-csv", required=True)
    parser.add_argument("--limit-examples", type=int, default=12)
    return parser.parse_args()


def normalize_sample_dir(path):
    path = str(path).strip().replace("\\", "/")
    match = re.search(r"(processed(?:_viai_a)?/.*)$", path)
    if match:
        path = match.group(1)
    path = re.sub(r"/\d+$", "", path)
    return os.path.normpath(path).replace("\\", "/").strip("/")


def load_records(path):
    records = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records[normalize_sample_dir(record["sample_dir"])] = record
    return records


def score_for_target(record, target_instrument):
    probs = record.get("probs_by_instrument")
    if isinstance(probs, dict) and target_instrument in probs:
        return float(probs[target_instrument])
    return float(record.get("semantic_score", 0.0))


def rank_for_target(record, target_instrument):
    probs = record.get("probs_by_instrument")
    if not isinstance(probs, dict) or target_instrument not in probs:
        return None
    target_score = float(probs[target_instrument])
    return 1 + sum(float(score) > target_score for score in probs.values())


def mean(values):
    return sum(values) / len(values) if values else 0.0


def main():
    args = parse_args()
    records = load_records(args.semantic_jsonl)
    rows = []
    missing = 0
    with open(args.sample_metrics_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_path = normalize_sample_dir(row["sample_path"])
            wrong_path = normalize_sample_dir(row.get("wrong_video_sample_path", ""))
            source_record = records.get(source_path)
            wrong_record = records.get(wrong_path)
            if source_record is None or wrong_record is None:
                missing += 1
                continue
            rows.append((row, source_record, wrong_record))

    original_scores = [
        score_for_target(source_record, row["source_instrument"])
        for row, source_record, _wrong_record in rows
    ]
    wrong_self_scores = [
        score_for_target(wrong_record, row["wrong_video_instrument"])
        for row, _source_record, wrong_record in rows
    ]
    wrong_source_scores = [
        score_for_target(wrong_record, row["source_instrument"])
        for row, _source_record, wrong_record in rows
    ]
    source_ranks = [
        rank_for_target(wrong_record, row["source_instrument"])
        for row, _source_record, wrong_record in rows
    ]
    source_ranks = [rank for rank in source_ranks if rank is not None]
    top1_wrong = sum(
        wrong_record.get("top1_instrument") == row["wrong_video_instrument"]
        for row, _source_record, wrong_record in rows
    )
    top1_source = sum(
        wrong_record.get("top1_instrument") == row["source_instrument"]
        for row, _source_record, wrong_record in rows
    )

    print(f"matched={len(rows)} missing={missing}")
    print(f"original_source_score_mean={mean(original_scores):.6f}")
    print(f"wrong_self_score_mean={mean(wrong_self_scores):.6f}")
    print(f"wrong_source_score_mean={mean(wrong_source_scores):.6f}")
    print(f"wrong_top1_equals_wrong_instrument={top1_wrong}/{len(rows)}")
    print(f"wrong_top1_equals_source_instrument={top1_source}/{len(rows)}")
    print(f"wrong_source_target_rank_counts={dict(sorted(Counter(source_ranks).items()))}")

    examples = sorted(
        rows,
        key=lambda item: score_for_target(item[2], item[0]["source_instrument"]),
        reverse=True,
    )[: args.limit_examples]
    print("\nhighest_wrong_source_score_examples:")
    for row, _source_record, wrong_record in examples:
        print(
            {
                "source": row["source_instrument"],
                "wrong": row["wrong_video_instrument"],
                "wrong_self_score": round(
                    score_for_target(wrong_record, row["wrong_video_instrument"]), 6
                ),
                "wrong_source_score": round(
                    score_for_target(wrong_record, row["source_instrument"]), 6
                ),
                "source_rank": rank_for_target(
                    wrong_record, row["source_instrument"]
                ),
                "top1": wrong_record.get("top1_instrument"),
                "wrong_path": normalize_sample_dir(row["wrong_video_sample_path"]),
            }
        )


if __name__ == "__main__":
    main()
