import csv
import json

import pytest

from tools.summarize_uq_av_ablation import parse_args, summarize


CONDITION_OFFSETS = {
    "original": 0.0,
    "no_video": -1.0,
    "wrong_video": -2.0,
    "zero_token": -3.0,
    "shuffled_video": -4.0,
}


def _write_condition_dir(root, condition, records):
    results_dir = root / condition
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "video_ablation_condition": condition,
                "num_samples": len(records),
            },
            handle,
        )
    fieldnames = (
        "sample_id",
        "mask_type",
        "start",
        "end",
        "gap_frames",
        "psnr_missing_db",
        "mel_l1_missing",
        "ssim_full",
        "boundary_l1",
    )
    with (results_dir / "metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return results_dir


def _records(condition):
    offset = CONDITION_OFFSETS[condition]
    return [
        {
            "sample_id": "sample_a",
            "mask_type": "random",
            "start": "10",
            "end": "30",
            "gap_frames": "20",
            "psnr_missing_db": 20.0 + offset,
            "mel_l1_missing": 0.10 - offset * 0.01,
            "ssim_full": 0.90 + offset * 0.01,
            "boundary_l1": 0.20 - offset * 0.01,
        },
        {
            "sample_id": "sample_b",
            "mask_type": "long_gap",
            "start": "40",
            "end": "80",
            "gap_frames": "40",
            "psnr_missing_db": 18.0 + offset,
            "mel_l1_missing": 0.20 - offset * 0.01,
            "ssim_full": 0.80 + offset * 0.01,
            "boundary_l1": 0.30 - offset * 0.01,
        },
    ]


def _make_args(tmp_path):
    paths = {
        condition: _write_condition_dir(
            tmp_path, condition, _records(condition)
        )
        for condition in CONDITION_OFFSETS
    }
    return parse_args(
        [
            "--original",
            str(paths["original"]),
            "--no-video",
            str(paths["no_video"]),
            "--wrong-video",
            str(paths["wrong_video"]),
            "--zero-token",
            str(paths["zero_token"]),
            "--shuffled-video",
            str(paths["shuffled_video"]),
            "--output-dir",
            str(tmp_path / "summary"),
        ]
    )


def test_summarize_uq_av_ablation_outputs_stats_and_deltas(tmp_path):
    outputs = summarize(_make_args(tmp_path))

    assert outputs["summary_json"].is_file()
    assert outputs["summary_csv"].is_file()
    assert outputs["paired_deltas_csv"].is_file()

    payload = outputs["payload"]
    assert payload["conditions"]["original"]["metrics"][
        "psnr_missing_db"
    ]["mean"] == pytest.approx(19.0)
    assert payload["original_advantage"]["no_video"][
        "psnr_missing_db"
    ]["mean"] == pytest.approx(1.0)
    assert payload["original_advantage"]["no_video"][
        "mel_l1_missing"
    ]["mean"] == pytest.approx(0.01)
    assert payload["original_advantage"]["no_video"][
        "psnr_missing_db"
    ]["win_rate"] == pytest.approx(1.0)

    with outputs["paired_deltas_csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        delta_rows = list(csv.DictReader(handle))
    assert len(delta_rows) == 2 * 4 * 4
    first = delta_rows[0]
    assert first["sample_id"] == "sample_a"
    assert first["comparison_condition"] == "no_video"
    assert first["metric"] == "psnr_missing_db"
    assert float(first["original_advantage"]) == pytest.approx(1.0)


def test_summarize_uq_av_ablation_rejects_unpaired_rows(tmp_path):
    args = _make_args(tmp_path)
    rows = _records("wrong_video")
    rows[0]["start"] = "999"
    _write_condition_dir(tmp_path, "wrong_video", rows)

    with pytest.raises(ValueError, match="not fully paired"):
        summarize(args)
