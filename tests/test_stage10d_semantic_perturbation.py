import os
import sys
from types import SimpleNamespace

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from networks.EC_VIAI_Modules import (  # noqa: E402
    apply_visual_evidence_augmentation,
    normalize_visual_evidence_aug_modes,
)
from utils.wrong_video_sampler import WrongVideoSampler  # noqa: E402


def test_visual_evidence_aug_modes_accept_stage10d_modes():
    modes = normalize_visual_evidence_aug_modes(
        "wrong_video_cross_instrument,no_video,flow_zero"
    )
    assert modes == ["wrong_video_cross_instrument", "no_video", "flow_zero"]


def test_no_video_visual_evidence_aug_zeros_video_and_flow():
    video = torch.ones(2, 3, 4, 5, 5)
    flow = torch.ones(2, 3, 2, 5, 5)

    aug_video, aug_flow = apply_visual_evidence_augmentation(video, flow, "no_video")

    assert torch.allclose(aug_video, torch.zeros_like(video))
    assert torch.allclose(aug_flow, torch.zeros_like(flow))


def test_train_wrong_video_sampler_picks_cross_instrument_sample(tmp_path):
    data_root = tmp_path / "data"
    split_path = data_root / "train_av_split.txt"
    split_path.parent.mkdir(parents=True)
    rows = [
        "processed/cello/video_a/shot_000000/clip_000000",
        "processed/flute/video_b/shot_000000/clip_000000",
        "processed/cello/video_c/shot_000000/clip_000000",
        "processed/flute/video_d/shot_000000/clip_000000",
    ]
    split_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    hparams = SimpleNamespace(
        data_root=str(data_root),
        train_split_name="train_av_split.txt",
        test_split_name="test_av_split.txt",
    )
    sampler = WrongVideoSampler(
        hparams,
        split_name="train_av_split.txt",
        mode="wrong_video_cross_instrument",
        train=True,
    )

    source_path = data_root / rows[0] / "17"
    wrong_dir = sampler.wrong_dir_for(str(source_path))

    assert "/processed/flute/" in wrong_dir
    assert wrong_dir != str(data_root / rows[0])
