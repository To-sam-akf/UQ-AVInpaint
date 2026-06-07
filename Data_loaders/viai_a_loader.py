import os

import numpy as np
import torch
from torch.utils import data as data_utils

import Options_inpainting


hparams = Options_inpainting.Inpainting_Config()


def split_name_for_phase(phase):
    if phase == "train":
        return hparams.train_split_name
    if phase == "val":
        return hparams.val_split_name
    if phase == "test":
        return hparams.test_split_name
    raise ValueError(f"Unknown VIAI-A data phase: {phase}")


def split_has_rows(data_root, split_name):
    split_path = os.path.join(data_root, split_name)
    if not os.path.exists(split_path):
        return False
    with open(split_path, "r", encoding="utf-8") as handle:
        return any(line.strip() for line in handle)


def read_split_rows(data_root, split_name):
    split_path = os.path.join(data_root, split_name)
    rows = []
    with open(split_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 4:
                raise ValueError(f"Expected 4 split columns in {split_path}: {line}")
            sample_dir, mel_path, audio_path, mel_frames = parts
            rows.append(
                {
                    "sample_dir": sample_dir,
                    "mel_path": os.path.join(data_root, mel_path),
                    "audio_path": os.path.join(data_root, audio_path),
                    "mel_frames": int(mel_frames),
                }
            )
    return rows


def pad_2d_time_first(array, target_len):
    if array.shape[0] >= target_len:
        return array[:target_len]
    pad_len = target_len - array.shape[0]
    return np.pad(array, [(0, pad_len), (0, 0)], mode="constant", constant_values=0.0)


def pad_1d(array, target_len):
    if array.shape[0] >= target_len:
        return array[:target_len]
    pad_len = target_len - array.shape[0]
    return np.pad(array, (0, pad_len), mode="constant", constant_values=0.0)


class VIAIASplitDataset(data_utils.Dataset):
    def __init__(self, data_root, split_name, train=True, hparams=hparams):
        self.data_root = data_root
        self.split_name = split_name
        self.train = train
        self.hparams = hparams
        self.rows = read_split_rows(data_root, split_name)
        # 4 秒裁剪发生在训练数据读取阶段
        self.mel_window = int(hparams.max_mel_lengths)
        self.audio_window = self.mel_window * int(hparams.hop_size)

    def __len__(self):
        return len(self.rows)
    # mel.shape = [mel_bins, mel_frames]
    # mel[:, start : start + mel_window]
    def _choose_mel_start(self, mel_frames):
        if mel_frames <= self.mel_window:
            return 0
        max_start = mel_frames - self.mel_window
        if self.train:
            return int(np.random.randint(0, max_start + 1))
        return max_start // 2

    def __getitem__(self, index):
        row = self.rows[index]
        mel = np.load(row["mel_path"]).astype(np.float32)
        raw_audio = np.load(row["audio_path"]).astype(np.float32)

        mel_start = self._choose_mel_start(mel.shape[0])
        mel_window = pad_2d_time_first(mel[mel_start:mel_start + self.mel_window], self.mel_window)
        audio_start = mel_start * int(self.hparams.hop_size)
        audio_window = pad_1d(raw_audio[audio_start:audio_start + self.audio_window], self.audio_window)

        return {
            "mel": torch.from_numpy(mel_window.T).float(),
            "audio": torch.from_numpy(audio_window).float(),
            "path": row["sample_dir"],
        }


def collate_fn(batch):
    mel = torch.stack([item["mel"] for item in batch], dim=0)
    audio = torch.stack([item["audio"] for item in batch], dim=0)
    paths = [item["path"] for item in batch]
    return {"mel": mel, "audio": audio, "path": paths}


def get_data_loaders(data_root, phases=("train", "val")):
    loaders = {}
    for phase in phases:
        split_name = split_name_for_phase(phase)
        train = phase == "train"
        if not split_has_rows(data_root, split_name):
            if train:
                raise RuntimeError(f"Training split is missing or empty: {os.path.join(data_root, split_name)}")
            print(f"[VIAI-A {phase}]: split missing or empty, skipping {split_name}")
            continue

        dataset = VIAIASplitDataset(data_root, split_name, train=train, hparams=hparams)
        loaders[phase] = data_utils.DataLoader(
            dataset,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            shuffle=train,
            collate_fn=collate_fn,
            pin_memory=hparams.pin_memory,
        )
        print(f"[VIAI-A {phase}]: length of the dataset is {len(dataset)}")
    return loaders
