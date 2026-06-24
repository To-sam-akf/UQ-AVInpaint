import csv
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from test_viai_av import (
    WrongVideoSampler,
    blur_video_batch,
    coerce_csv_record,
    frame_drop_batch,
    instrument_from_sample_dir,
    temporal_shift_batch,
    write_result_files,
)
from utils.viai_a_metrics import (
    compute_calibration_bins,
    compute_multi_candidate_metrics,
    compute_risk_coverage_curve,
)


def _mask(batch_size=1, mel_bins=2, mel_steps=6, span=(2, 4)):
    mask = torch.zeros(batch_size, 1, mel_bins, mel_steps)
    mask[:, :, :, span[0] : span[1]] = 1.0
    return mask


def test_multi_candidate_metrics_k1_degenerates_to_single_candidate():
    target = torch.zeros(2, 1, 2, 6)
    mask = _mask(batch_size=2)
    candidates = target.unsqueeze(1).clone()
    candidates[:, :, :, :, 2:4] = 0.5

    metrics = compute_multi_candidate_metrics(
        candidates,
        candidates,
        target,
        mask,
        missing_span=(2, 4),
    )

    assert torch.allclose(
        metrics["top1_missing_l1_per_sample"],
        metrics["best_of_k_missing_l1_per_sample"],
    )
    assert torch.allclose(
        metrics["mean_k_missing_l1_per_sample"],
        metrics["candidate0_missing_l1_per_sample"],
    )
    assert torch.allclose(
        metrics["candidate_pairwise_mel_l1_per_sample"],
        torch.zeros(2),
    )
    assert torch.allclose(metrics["oracle_gain_per_sample"], torch.zeros(2))


def test_multi_candidate_metrics_boundary_delta_error_prefers_smooth_candidate():
    target = torch.arange(6, dtype=torch.float32).view(1, 1, 1, 6)
    mask = _mask(batch_size=1, mel_bins=1, mel_steps=6, span=(2, 4))
    candidates = target.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    candidates[:, 1, :, :, 2] += 10.0

    metrics = compute_multi_candidate_metrics(
        candidates,
        candidates,
        target,
        mask,
        top1_indices=torch.tensor([1]),
        missing_span=(2, 4),
    )

    assert metrics["top1_boundary_delta_error_per_sample"].item() > 0.0
    assert metrics["best_boundary_delta_error_per_sample"].item() == 0.0
    assert metrics["mean_boundary_delta_error_per_sample"].item() > 0.0


def test_risk_coverage_curve_sorts_by_low_uncertainty_first():
    rows = compute_risk_coverage_curve(
        uncertainty=np.array([0.9, 0.1, 0.2, 0.8]),
        top1_error=np.array([9.0, 1.0, 2.0, 8.0]),
        num_points=2,
    )

    assert rows[0]["coverage"] == 0.5
    assert rows[0]["retained_count"] == 2
    assert rows[0]["uncertainty_threshold"] == 0.2
    assert rows[0]["mean_top1_error"] == 1.5
    assert rows[1]["coverage"] == 1.0
    assert rows[1]["mean_top1_error"] == 5.0


def test_calibration_bins_reports_empty_and_filled_bins():
    rows = compute_calibration_bins(
        uncertainty=np.array([0.05, 0.15, 0.85]),
        top1_error=np.array([1.0, 2.0, 8.0]),
        best_error=np.array([0.5, 1.0, 4.0]),
        oracle_gain=np.array([0.5, 1.0, 4.0]),
        evidence=np.array([0.9, 0.8, 0.1]),
        pairwise=np.array([0.1, 0.2, 0.9]),
        num_bins=2,
    )

    assert len(rows) == 2
    assert rows[0]["count"] == 2
    assert rows[0]["avg_uncertainty"] == 0.1
    assert rows[0]["avg_top1_error"] == 1.5
    assert rows[1]["count"] == 1
    assert rows[1]["avg_evidence"] == 0.1


def test_video_perturbation_helpers_preserve_shape_and_change_expected_values():
    video = torch.zeros(1, 4, 1, 5, 5)
    video[:, :, :, 2, 2] = 1.0
    flow = torch.ones(1, 4, 2, 5, 5)

    blurred = blur_video_batch(video, kernel_size=3)
    assert blurred.shape == video.shape
    assert blurred[:, :, :, 2, 2].max().item() < 1.0

    dropped_video, dropped_flow = frame_drop_batch(video, flow, stride=2)
    assert dropped_video.shape == video.shape
    assert dropped_flow.shape == flow.shape
    assert torch.allclose(dropped_video[:, 1], video[:, 0])
    assert torch.allclose(dropped_flow[:, 1], torch.zeros_like(dropped_flow[:, 1]))
    assert torch.allclose(dropped_flow[:, 0], flow[:, 0])

    shifted_video, shifted_flow = temporal_shift_batch(video, flow, shift_frames=2)
    assert shifted_video.shape == video.shape
    assert shifted_flow.shape == flow.shape
    assert torch.allclose(shifted_video[:, :2], video[:, :1].expand(-1, 2, -1, -1, -1))
    assert torch.allclose(shifted_flow[:, :2], torch.zeros_like(shifted_flow[:, :2]))


def test_summary_csv_dedupes_by_step_perturbation_and_candidate_count(tmp_path):
    base_record = {
        "checkpoint_path": "ckpt",
        "checkpoint_step": 10,
        "global_step": 10,
        "global_epoch": 1,
        "test_split_name": "test.txt",
        "stage": "stage",
        "video_perturbation": "none",
        "test_num_candidates": 4,
        "num_samples": 1,
    }
    record_none = dict(base_record, top1_missing_l1=1.0)
    record_blur = dict(base_record, video_perturbation="blur", top1_missing_l1=2.0)
    record_k8 = dict(base_record, test_num_candidates=8, top1_missing_l1=3.0)
    record_none_update = dict(base_record, top1_missing_l1=4.0)

    write_result_files(record_none, tmp_path, "EC-VIAI-AV")
    write_result_files(record_blur, tmp_path, "EC-VIAI-AV")
    write_result_files(record_k8, tmp_path, "EC-VIAI-AV")
    _json_path, csv_path = write_result_files(record_none_update, tmp_path, "EC-VIAI-AV")

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    keyed = {
        (row["video_perturbation"], int(row["test_num_candidates"])): row
        for row in rows
    }
    assert keyed[("none", 4)]["top1_missing_l1"] == "4.0"
    assert keyed[("blur", 4)]["top1_missing_l1"] == "2.0"
    assert keyed[("none", 8)]["top1_missing_l1"] == "3.0"


def _write_split(tmp_path, rows):
    split_path = tmp_path / "test_av_split.txt"
    with split_path.open("w", encoding="utf-8") as handle:
        for sample_dir in rows:
            handle.write(f"{sample_dir}|{sample_dir}/mel.npy|{sample_dir}/raw_audio.npy|200\n")
    return split_path


def _hparams(tmp_path, mode):
    return SimpleNamespace(
        data_root=str(tmp_path),
        test_split_name="test_av_split.txt",
        video_perturbation=mode,
    )


def test_instrument_from_sample_dir_uses_processed_path_segment():
    assert (
        instrument_from_sample_dir("processed/accordion/video/shot_000000/clip_000001")
        == "accordion"
    )
    assert (
        instrument_from_sample_dir(
            "/root/data/processed/cello/video/shot_000000/clip_000001"
        )
        == "cello"
    )


def test_wrong_video_any_matches_legacy_wrong_video_alias(tmp_path):
    rows = [
        "processed/accordion/a/shot_000000/clip_000000",
        "processed/accordion/b/shot_000000/clip_000000",
        "processed/cello/c/shot_000000/clip_000000",
    ]
    _write_split(tmp_path, rows)
    legacy_sampler = WrongVideoSampler(_hparams(tmp_path, "wrong_video"))
    any_sampler = WrongVideoSampler(_hparams(tmp_path, "wrong_video_any"))
    sample_path = str(tmp_path / rows[0] / "12")

    assert legacy_sampler.mode == "wrong_video_any"
    assert legacy_sampler.wrong_dir_for(sample_path) == any_sampler.wrong_dir_for(sample_path)


def test_wrong_video_cross_instrument_selects_different_instrument(tmp_path):
    rows = [
        "processed/accordion/a/shot_000000/clip_000000",
        "processed/accordion/b/shot_000000/clip_000000",
        "processed/cello/c/shot_000000/clip_000000",
        "processed/flute/d/shot_000000/clip_000000",
    ]
    _write_split(tmp_path, rows)
    sampler = WrongVideoSampler(_hparams(tmp_path, "wrong_video_cross_instrument"))

    for row in rows:
        sample_path = str(tmp_path / row / "12")
        source_dir = os.path.abspath(str(tmp_path / row))
        wrong_dir = sampler.wrong_dir_for(sample_path)
        assert sampler.instrument_by_dir[source_dir] != sampler.instrument_by_dir[wrong_dir]


def test_wrong_video_cross_instrument_requires_multiple_instruments(tmp_path):
    rows = [
        "processed/accordion/a/shot_000000/clip_000000",
        "processed/accordion/b/shot_000000/clip_000000",
    ]
    _write_split(tmp_path, rows)

    with pytest.raises(RuntimeError, match="at least 2 instruments"):
        WrongVideoSampler(_hparams(tmp_path, "wrong_video_cross_instrument"))


def test_wrong_video_summary_fields_round_trip_through_csv_coercion():
    row = {
        "checkpoint_step": "10",
        "wrong_video_effective_mode": "wrong_video_cross_instrument",
        "wrong_video_cross_instrument_available": "True",
        "wrong_video_num_instruments": "3",
    }

    record = coerce_csv_record(row)

    assert record["wrong_video_effective_mode"] == "wrong_video_cross_instrument"
    assert record["wrong_video_cross_instrument_available"] is True
    assert record["wrong_video_num_instruments"] == 3
