import os
import zlib

import numpy as np
import torch

from Data_loaders import audio_loader as av_loader
from utils.semantic_evidence import infer_instrument_from_sample_dir


def sample_dir_from_path(path):
    path = os.path.abspath(str(path))
    if os.path.basename(path).isdigit():
        return os.path.dirname(path)
    return path


def read_split_sample_dirs(data_root, split_name):
    split_path = os.path.join(data_root, split_name)
    sample_dirs = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample_dir = line.split("|")[0]
            if not os.path.isabs(sample_dir):
                sample_dir = os.path.join(data_root, sample_dir)
            sample_dirs.append(os.path.abspath(sample_dir))
    return sample_dirs


def wrong_video_effective_mode(mode):
    if mode == "wrong_video":
        return "wrong_video_any"
    return mode


class WrongVideoSampler:
    def __init__(self, hparams, split_name=None, mode=None, train=False):
        self.hparams = hparams
        self.mode = wrong_video_effective_mode(
            mode if mode is not None else getattr(hparams, "video_perturbation", "none")
        )
        self.train = bool(train)
        if split_name is None:
            split_name = (
                getattr(hparams, "train_split_name", "train_av_split.txt")
                if self.train
                else getattr(hparams, "test_split_name", "test_av_split.txt")
            )
        self.split_name = split_name
        self.sample_dirs = read_split_sample_dirs(hparams.data_root, split_name)
        if len(self.sample_dirs) < 2:
            raise RuntimeError(
                "wrong_video/wrong_video_any/wrong_video_cross_instrument requires "
                f"at least 2 samples in split: {split_name}"
            )
        self.index_by_dir = {
            os.path.abspath(sample_dir): index
            for index, sample_dir in enumerate(self.sample_dirs)
        }
        self.instrument_by_dir = {
            os.path.abspath(sample_dir): infer_instrument_from_sample_dir(sample_dir)
            for sample_dir in self.sample_dirs
        }
        self.sample_dirs_by_instrument = {}
        for sample_dir in self.sample_dirs:
            instrument = self.instrument_by_dir[os.path.abspath(sample_dir)]
            self.sample_dirs_by_instrument.setdefault(instrument, []).append(sample_dir)
        self.instruments = sorted(self.sample_dirs_by_instrument)
        self.cross_instrument_available = len(self.instruments) >= 2
        if self.mode == "wrong_video_cross_instrument" and not self.cross_instrument_available:
            raise RuntimeError(
                "wrong_video_cross_instrument requires at least 2 instruments "
                f"in split: {split_name}"
            )
        self.last_wrong_dirs = []
        self.last_source_instruments = []
        self.last_wrong_instruments = []
        self.last_is_cross_instrument = []

    def _wrong_any_dir_for(self, sample_dir):
        index = self.index_by_dir[sample_dir]
        return self.sample_dirs[(index + 1) % len(self.sample_dirs)]

    def _wrong_cross_instrument_dir_for(self, sample_dir):
        source_instrument = self.instrument_by_dir[sample_dir]
        source_instrument_index = self.instruments.index(source_instrument)
        target_instrument = self.instruments[
            (source_instrument_index + 1) % len(self.instruments)
        ]
        target_dirs = self.sample_dirs_by_instrument[target_instrument]
        source_index = self.index_by_dir[sample_dir]
        return target_dirs[source_index % len(target_dirs)]

    def wrong_dir_for(self, sample_path):
        sample_dir = sample_dir_from_path(sample_path)
        if sample_dir not in self.index_by_dir:
            raise RuntimeError(
                f"Cannot map sample path to split {self.split_name} for wrong_video: "
                f"{sample_path}"
            )
        if self.mode == "wrong_video_cross_instrument":
            return self._wrong_cross_instrument_dir_for(sample_dir)
        if self.mode in {"wrong_video", "wrong_video_any"}:
            return self._wrong_any_dir_for(sample_dir)
        raise RuntimeError(f"Unsupported wrong video mode: {self.mode}")

    def load_batch(self, paths, reference_video, reference_flow):
        videos = []
        flows = []
        self.last_wrong_dirs = []
        self.last_source_instruments = []
        self.last_wrong_instruments = []
        self.last_is_cross_instrument = []
        for sample_path in paths:
            source_dir = sample_dir_from_path(sample_path)
            wrong_dir = self.wrong_dir_for(sample_path)
            source_instrument = self.instrument_by_dir[source_dir]
            wrong_instrument = self.instrument_by_dir[os.path.abspath(wrong_dir)]
            numpy_state = np.random.get_state()
            try:
                seed = zlib.crc32(wrong_dir.encode("utf-8")) & 0xFFFFFFFF
                np.random.seed(seed)
                video_block, flow_block, _start = av_loader.sample_data_new(
                    wrong_dir,
                    train=self.train,
                    hparams=self.hparams,
                )
            finally:
                np.random.set_state(numpy_state)
            self.last_wrong_dirs.append(wrong_dir)
            self.last_source_instruments.append(source_instrument)
            self.last_wrong_instruments.append(wrong_instrument)
            self.last_is_cross_instrument.append(source_instrument != wrong_instrument)
            videos.append(torch.from_numpy(video_block[0]))
            flows.append(torch.from_numpy(flow_block[0]))

        video = torch.stack(videos, dim=0).to(
            device=reference_video.device,
            dtype=reference_video.dtype,
        )
        flow = torch.stack(flows, dim=0).to(
            device=reference_flow.device,
            dtype=reference_flow.dtype,
        )
        if video.shape != reference_video.shape or flow.shape != reference_flow.shape:
            raise RuntimeError(
                "wrong_video sample shape mismatch: "
                f"video={tuple(video.shape)} expected={tuple(reference_video.shape)}; "
                f"flow={tuple(flow.shape)} expected={tuple(reference_flow.shape)}"
            )
        return video, flow
