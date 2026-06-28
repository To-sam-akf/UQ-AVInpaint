import os
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_vocoder import VocoderSplitDataset
from utils.hifigan import (
    HifiGanGenerator,
    load_matching_generator_checkpoint,
    normalized_mel_to_log_condition,
    splice_missing_audio,
)


def _hparams(tmp_path):
    return SimpleNamespace(
        data_root=str(tmp_path),
        hop_size=4,
        sample_rate=16,
        num_mels=4,
        cin_channels=4,
        hifigan_segment_mel_frames=5,
        min_level_db=-100.0,
        ref_level_db=20.0,
    )


def test_vocoder_split_dataset_reads_mel_and_audio_windows(tmp_path):
    sample = tmp_path / "processed" / "flute" / "clip"
    sample.mkdir(parents=True)
    mel = np.arange(10 * 4, dtype=np.float32).reshape(10, 4) / 40.0
    audio = np.linspace(-1.0, 1.0, 40, dtype=np.float32)
    np.save(sample / "mel.npy", mel)
    np.save(sample / "raw_audio.npy", audio)
    split = tmp_path / "train_av_split.txt"
    split.write_text(
        "processed/flute/clip|processed/flute/clip/mel.npy|processed/flute/clip/raw_audio.npy|10\n",
        encoding="utf-8",
    )

    dataset = VocoderSplitDataset(str(tmp_path), "train_av_split.txt", train=False, hparams=_hparams(tmp_path))
    item = dataset[0]

    assert tuple(item["mel"].shape) == (4, 5)
    assert tuple(item["audio"].shape) == (1, 20)
    assert item["path"] == "processed/flute/clip"


def test_normalized_mel_to_log_condition_is_finite_and_bct():
    hparams = SimpleNamespace(num_mels=4, cin_channels=4, min_level_db=-100.0, ref_level_db=20.0)
    mel = torch.linspace(0.0, 1.0, steps=20).view(1, 4, 5)

    condition = normalized_mel_to_log_condition(mel, hparams)

    assert tuple(condition.shape) == (1, 4, 5)
    assert torch.isfinite(condition).all()
    assert condition[:, :, -1].mean() > condition[:, :, 0].mean()


def test_hifigan_partial_checkpoint_load_skips_shape_mismatch(tmp_path):
    config = SimpleNamespace(
        num_mels=4,
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        upsample_initial_channel=16,
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[(1, 3, 5)],
    )
    model = HifiGanGenerator(config)
    state = model.state_dict()
    checkpoint_state = {
        "conv_pre.bias": state["conv_pre.bias"].clone(),
        "conv_pre.weight_g": torch.zeros(99),
        "not_in_model": torch.zeros(1),
    }
    checkpoint = tmp_path / "generator.pth.tar"
    torch.save({"generator": checkpoint_state}, checkpoint)

    report = load_matching_generator_checkpoint(str(checkpoint), model, device="cpu")

    assert report["loaded"] == 1
    assert report["skipped_shape"] == 1
    assert report["skipped_missing"] == 1


def test_splice_missing_audio_keeps_known_region_and_replaces_missing():
    hparams = SimpleNamespace(hop_size=4, sample_rate=100)
    original = np.arange(40, dtype=np.float32)
    generated = np.full(40, -1.0, dtype=np.float32)
    mask = np.zeros((1, 1, 4, 10), dtype=np.float32)
    mask[:, :, :, 3:6] = 1.0

    spliced = splice_missing_audio(original, generated, mask, hparams, crossfade_ms=0.0)

    assert spliced.shape == original.shape
    assert np.allclose(spliced[:12], original[:12])
    assert np.allclose(spliced[12:24], generated[12:24])
    assert np.allclose(spliced[24:], original[24:])
