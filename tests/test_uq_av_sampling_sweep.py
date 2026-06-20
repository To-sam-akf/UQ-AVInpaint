import csv
import json

import pytest

from tools.summarize_uq_av_sampling_sweep import parse_args, summarize


def _write_summary(path, **overrides):
    payload = {
        "checkpoint": "/ckpt/UQ-AV_best.pth.tar",
        "num_samples": 2,
        "psnr_missing_db": 18.5,
        "mel_l1_missing": 0.12,
        "psnr_full_db": 24.0,
        "mel_l1_full": 0.03,
        "ssim_full": 0.7,
        "boundary_l1": 0.08,
        "known_region_max_abs_error_max": 0.0,
        "uq_beta_schedule": "cosine",
        "uq_prediction_type": "v",
        "uq_latent_clip_value": 4.0,
        "inference_steps": 100,
        "ddim_eta": 0.0,
        "latent_is_normalised": True,
        "uq_ema_eval": False,
        "elapsed_seconds": 12.0,
        "seconds_per_sample": 6.0,
        "samples_per_second": 1.0 / 6.0,
    }
    payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_sampling_sweep_writes_json_and_csv(tmp_path):
    run_a = tmp_path / "linear50"
    run_b = tmp_path / "cosine100"
    _write_summary(run_a / "summary.json", uq_beta_schedule="linear",
                   inference_steps=50)
    _write_summary(run_b / "summary.json", uq_beta_schedule="cosine",
                   inference_steps=100, uq_ema_eval=True)

    args = parse_args([
        "--run", "linear-50", str(run_a),
        "--run", "cosine-100-ema", str(run_b / "summary.json"),
        "--output-dir", str(tmp_path / "summary"),
    ])
    outputs = summarize(args)

    payload = json.loads(outputs["summary_json"].read_text(encoding="utf-8"))
    assert [row["label"] for row in payload["runs"]] == [
        "linear-50",
        "cosine-100-ema",
    ]

    with outputs["summary_csv"].open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["uq_beta_schedule"] == "linear"
    assert rows[1]["uq_ema_eval"] == "True"
    assert rows[1]["inference_steps"] == "100"


def test_sampling_sweep_missing_summary_fails(tmp_path):
    args = parse_args([
        "--run", "missing", str(tmp_path / "missing_dir"),
        "--output-dir", str(tmp_path / "summary"),
    ])
    with pytest.raises(FileNotFoundError):
        summarize(args)


def test_sampling_sweep_required_fields(tmp_path):
    run_dir = tmp_path / "bad"
    payload = _write_summary(run_dir / "summary.json")
    payload.pop("known_region_max_abs_error_max")
    (run_dir / "summary.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )
    args = parse_args([
        "--run", "bad", str(run_dir),
        "--output-dir", str(tmp_path / "summary"),
    ])
    with pytest.raises(ValueError, match="missing required fields"):
        summarize(args)
