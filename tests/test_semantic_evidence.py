import json
import os
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from test_viai_av import apply_test_video_perturbation, hparams
from utils.semantic_evidence import (
    SemanticEvidenceTable,
    combine_evidence_scores,
    infer_instrument_from_sample_dir,
)


def test_semantic_evidence_lookup_matches_trailing_start_index(tmp_path):
    data_root = tmp_path / "data"
    sample_dir = data_root / "processed" / "cello" / "video" / "clip_000000"
    evidence_dir = data_root / "semantic_evidence"
    evidence_dir.mkdir(parents=True)
    evidence_path = evidence_dir / "test_av_split.txt.jsonl"
    record = {
        "sample_dir": "processed/cello/video/clip_000000",
        "instrument": "cello",
        "semantic_score": 0.73,
        "target_prob": 0.73,
        "probs_by_instrument": {"cello": 0.73, "flute": 0.19},
        "top1_instrument": "cello",
        "top1_prob": 0.73,
        "target_rank": 1,
        "frame_consistency": 1.0,
        "frame_top1_instruments": ["cello"],
        "num_frames": 8,
    }
    with evidence_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    table = SemanticEvidenceTable(
        str(evidence_path),
        data_root=str(data_root),
        missing_score=0.11,
    )

    assert table.lookup_score(str(sample_dir)) == 0.73
    assert table.lookup_score(str(sample_dir), target_instrument="flute") == 0.19
    assert table.lookup_score(str(sample_dir), target_instrument="missing") == 0.11
    assert table.lookup_score(str(sample_dir / "37")) == 0.73
    assert table.lookup_score(str(data_root / "processed" / "flute" / "x")) == 0.11
    assert infer_instrument_from_sample_dir(str(sample_dir / "37")) == "cello"


def test_semantic_evidence_lookup_keeps_legacy_jsonl_compatible(tmp_path):
    data_root = tmp_path / "data"
    sample_dir = data_root / "processed" / "cello" / "video" / "clip_000000"
    evidence_path = tmp_path / "legacy.jsonl"
    record = {
        "sample_dir": "processed/cello/video/clip_000000",
        "instrument": "cello",
        "semantic_score": 0.62,
    }
    with evidence_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")

    table = SemanticEvidenceTable(
        str(evidence_path),
        data_root=str(data_root),
        missing_score=0.11,
    )

    assert table.lookup_score(str(sample_dir)) == 0.62
    assert table.lookup_score(str(sample_dir), target_instrument="flute") == 0.62
    assert table.lookup_scores(
        [str(sample_dir), str(sample_dir / "37")],
        target_instruments=["cello", "flute"],
    ) == [0.62, 0.62]


def test_combine_evidence_sources_preserves_default_heuristic_behavior():
    heuristic = torch.tensor([[0.2], [0.8]])
    semantic = torch.tensor([[1.0], [0.0]])

    assert torch.allclose(
        combine_evidence_scores(heuristic, semantic, source="none", weight=0.35),
        heuristic,
    )
    assert torch.allclose(
        combine_evidence_scores(heuristic, semantic, source="heuristic", weight=0.35),
        heuristic,
    )
    assert torch.allclose(
        combine_evidence_scores(heuristic, semantic, source="semantic", weight=0.35),
        semantic,
    )

    fused = combine_evidence_scores(heuristic, semantic, source="fused", weight=0.5)
    assert torch.allclose(fused, torch.tensor([[0.6], [0.4]]))
    assert bool(torch.all(fused >= 0.0) and torch.all(fused <= 1.0))


class _FakeModel:
    def __init__(self):
        self.video_batch = torch.ones(1, 2, 3, 4, 4)
        self.flow_batch = torch.ones(1, 2, 2, 4, 4)
        self.path_batch = ["/data/processed/cello/a/clip_000000/12"]
        self.semantic_evidence_paths = []
        self.semantic_evidence_target_instruments = []
        self.semantic_evidence_override = None

    def set_semantic_evidence_paths(self, paths, target_instruments=None):
        self.semantic_evidence_paths = list(paths)
        self.semantic_evidence_target_instruments = (
            list(target_instruments) if target_instruments is not None else []
        )
        self.semantic_evidence_override = None

    def set_semantic_evidence_override(self, value):
        self.semantic_evidence_override = value


class _FakeWrongVideoSampler:
    def __init__(self):
        self.last_wrong_dirs = ["/data/processed/flute/b/clip_000000"]
        self.last_source_instruments = ["cello"]

    def load_batch(self, paths, reference_video, reference_flow):
        return torch.zeros_like(reference_video), torch.zeros_like(reference_flow)


def test_wrong_and_no_video_perturbations_override_semantic_lookup():
    original_mode = hparams.video_perturbation
    try:
        model = _FakeModel()
        hparams.video_perturbation = "wrong_video_cross_instrument"
        apply_test_video_perturbation(model, wrong_video_sampler=_FakeWrongVideoSampler())
        assert model.semantic_evidence_paths == ["/data/processed/flute/b/clip_000000"]
        assert model.semantic_evidence_target_instruments == ["cello"]
        assert model.semantic_evidence_override is None

        model = _FakeModel()
        hparams.video_perturbation = "no_video"
        apply_test_video_perturbation(model)
        assert torch.allclose(model.video_batch, torch.zeros_like(model.video_batch))
        assert torch.allclose(model.flow_batch, torch.zeros_like(model.flow_batch))
        assert model.semantic_evidence_override == 0.0
    finally:
        hparams.video_perturbation = original_mode
