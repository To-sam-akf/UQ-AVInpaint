import random
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Data_loaders.mask_sampler import (
    MASK_TYPES,
    MaskSampler,
    MaskSpec,
    build_boundary_map,
    corrupt_mel,
    load_uq_mask_manifest,
    spectral_flux,
    stable_seed,
)
from Data_loaders.visual_degradation import (
    VIDEO_CONDITIONS,
    apply_visual_degradation,
)
from utils.baseline_protocol import canonical_sample_id


def _resolve_data_path(data_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def _sample_identity(sample_id):
    parts = [
        part
        for part in PurePosixPath(canonical_sample_id(sample_id)).parts
        if part not in {"", "/", "."}
    ]
    if parts and parts[0] in {"processed", "processed_viai_a"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"Cannot infer instrument/source video from {sample_id}")
    return parts[0], "/".join(parts[:2])


def read_uq_split(data_root, split_name):
    split_path = _resolve_data_path(data_root, split_name)
    if not split_path.is_file():
        raise FileNotFoundError(f"UQ split file not found: {split_path}")
    rows = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) not in {4, 5}:
                raise ValueError(
                    f"Expected 4 or 5 columns in {split_path}:{line_number}, "
                    f"got {len(parts)}"
                )
            sample_id = canonical_sample_id(parts[0])
            instrument, source_video = _sample_identity(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_dir": _resolve_data_path(data_root, parts[0]),
                    "mel_path": _resolve_data_path(data_root, parts[1]),
                    "audio_path": _resolve_data_path(data_root, parts[2]),
                    "mel_frames": int(parts[-1]),
                    "instrument": instrument,
                    "source_video": source_video,
                }
            )
    if not rows:
        raise ValueError(f"UQ split file is empty: {split_path}")
    rows.sort(key=lambda row: row["sample_id"])
    return rows


class UQAVDataset(Dataset):
    def __init__(
        self,
        data_root,
        split_name,
        phase,
        metadata_dir=None,
        mask_manifest=None,
        seed=1234,
        mask_types=MASK_TYPES,
        video_conditions=("original",),
        mel_frames=200,
        mel_bins=80,
        audio_steps=64000,
        visual_frames=50,
        image_size=256,
        boundary_margin=3,
    ):
        if phase not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported UQ data phase: {phase}")
        self.data_root = Path(data_root)
        self.split_name = str(split_name)
        self.phase = phase
        self.seed = int(seed)
        self.epoch = 0
        self.mel_frames = int(mel_frames)
        self.mel_bins = int(mel_bins)
        self.audio_steps = int(audio_steps)
        self.visual_frames = int(visual_frames)
        self.image_size = int(image_size)
        self.boundary_margin = int(boundary_margin)
        self.rows = read_uq_split(self.data_root, split_name)
        self.row_by_sample_id = {row["sample_id"]: row for row in self.rows}
        if len(self.row_by_sample_id) != len(self.rows):
            raise ValueError(f"Duplicate sample_id in UQ split: {split_name}")

        self.mask_types = tuple(mask_types)
        if not self.mask_types or any(value not in MASK_TYPES for value in self.mask_types):
            raise ValueError(f"Invalid mask_types: {self.mask_types}")
        self.video_conditions = tuple(video_conditions)
        if not self.video_conditions or any(
            value not in VIDEO_CONDITIONS for value in self.video_conditions
        ):
            raise ValueError(f"Invalid video_conditions: {self.video_conditions}")

        self.mask_sampler = MaskSampler(
            mel_frames=self.mel_frames,
            boundary_margin=self.boundary_margin,
        )
        self.metadata_dir = (
            Path(metadata_dir)
            if metadata_dir is not None
            else self.data_root / "uq_metadata"
        )

        self.manifest = None
        self.eval_entries = None
        if self.phase != "train":
            if mask_manifest is None:
                mask_manifest = self.metadata_dir / f"{self.phase}_masks.jsonl"
            self.manifest = (
                load_uq_mask_manifest(
                    mask_manifest,
                    mel_frames=self.mel_frames,
                    boundary_margin=self.boundary_margin,
                )
                if not isinstance(mask_manifest, dict)
                else self._coerce_manifest(mask_manifest)
            )
            split_ids = set(self.row_by_sample_id)
            manifest_ids = set(self.manifest)
            if split_ids != manifest_ids:
                missing = sorted(split_ids - manifest_ids)
                extra = sorted(manifest_ids - split_ids)
                raise ValueError(
                    "UQ mask manifest coverage does not match split: "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            self.eval_entries = []
            for row_index, row in enumerate(self.rows):
                for spec in self.manifest[row["sample_id"]]:
                    for condition in self.video_conditions:
                        self.eval_entries.append((row_index, spec, condition))

    def _coerce_manifest(self, manifest):
        coerced = {}
        for sample_id, variants in manifest.items():
            canonical_id = canonical_sample_id(sample_id)
            coerced[canonical_id] = [
                variant
                if isinstance(variant, MaskSpec)
                else MaskSpec.from_dict(
                    variant,
                    mel_frames=self.mel_frames,
                    boundary_margin=self.boundary_margin,
                )
                for variant in variants
            ]
        return coerced

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        if self.phase == "train":
            return len(self.rows)
        return len(self.eval_entries)

    def _load_mel_audio(self, row):
        mel = np.load(row["mel_path"], allow_pickle=False)
        audio = np.load(row["audio_path"], allow_pickle=False)
        if mel.shape != (self.mel_frames, self.mel_bins):
            raise ValueError(
                f"Sample {row['sample_id']} has Mel shape {mel.shape}; expected "
                f"({self.mel_frames}, {self.mel_bins})"
            )
        if audio.shape != (self.audio_steps,):
            raise ValueError(
                f"Sample {row['sample_id']} has audio shape {audio.shape}; "
                f"expected ({self.audio_steps},)"
            )
        mel_target = torch.from_numpy(
            np.asarray(mel, dtype=np.float32).T.copy()
        ).unsqueeze(0)
        audio_target = torch.from_numpy(np.asarray(audio, dtype=np.float32).copy())
        return mel_target, audio_target

    def _load_visual(self, row):
        video = np.empty(
            (self.visual_frames, 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        flow = np.empty(
            (self.visual_frames, 2, self.image_size, self.image_size),
            dtype=np.float32,
        )
        for offset in range(self.visual_frames):
            frame_id = offset + 1
            image_path = row["sample_dir"] / "image_crop" / f"{frame_id}.jpg"
            flow_x_path = row["sample_dir"] / "flow_x_crop" / f"{frame_id}.jpg"
            flow_y_path = row["sample_dir"] / "flow_y_crop" / f"{frame_id}.jpg"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            flow_x = cv2.imread(str(flow_x_path), cv2.IMREAD_GRAYSCALE)
            flow_y = cv2.imread(str(flow_y_path), cv2.IMREAD_GRAYSCALE)
            if image is None or flow_x is None or flow_y is None:
                raise ValueError(
                    f"Unreadable visual frame for {row['sample_id']}: frame {frame_id}"
                )
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            flow_x = cv2.resize(
                flow_x,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            flow_y = cv2.resize(
                flow_y,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )
            video[offset] = np.transpose(
                (image.astype(np.float32) - 127.0) / 128.0,
                (2, 0, 1),
            )
            flow[offset, 0] = (flow_x.astype(np.float32) - 127.0) / 128.0
            flow[offset, 1] = (flow_y.astype(np.float32) - 127.0) / 128.0
        return video, flow

    def _load_onsets(self, row, mel=None):
        onset_path = (
            self.metadata_dir
            / "train_onsets"
            / Path(row["sample_id"]).with_suffix(".npy")
        )
        if onset_path.is_file():
            values = np.load(onset_path, allow_pickle=False)
        else:
            if mel is None:
                mel = np.load(row["mel_path"], allow_pickle=False)
            values = spectral_flux(mel)
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.shape != (self.mel_frames,):
            raise ValueError(
                f"Onset metadata for {row['sample_id']} has shape {values.shape}; "
                f"expected ({self.mel_frames},)"
            )
        return values

    def _training_spec_and_condition(self, row):
        selection_seed = stable_seed(
            "uq-train-selection",
            self.seed,
            self.epoch,
            row["sample_id"],
        )
        rng = random.Random(selection_seed)
        mask_type = rng.choice(self.mask_types)
        condition = rng.choice(self.video_conditions)
        mask_seed = stable_seed(
            "uq-train-mask",
            self.seed,
            self.epoch,
            row["sample_id"],
            mask_type,
        )
        onsets = self._load_onsets(row) if mask_type == "onset_centered" else None
        spec = self.mask_sampler.sample(mask_type, mask_seed, onset_strengths=onsets)
        return spec, condition

    def _wrong_video_row(self, row_index, visual_seed):
        row = self.rows[row_index]
        candidates = [
            index
            for index, candidate in enumerate(self.rows)
            if candidate["instrument"] == row["instrument"]
            and candidate["source_video"] != row["source_video"]
        ]
        if not candidates:
            candidates = [
                index
                for index, candidate in enumerate(self.rows)
                if candidate["source_video"] != row["source_video"]
            ]
        if not candidates:
            raise ValueError(
                "wrong_video requires at least two distinct source videos in the split"
            )
        candidates.sort(key=lambda index: self.rows[index]["sample_id"])
        rng = random.Random(int(visual_seed))
        return self.rows[rng.choice(candidates)]

    def __getitem__(self, index):
        if self.phase == "train":
            row_index = int(index)
            row = self.rows[row_index]
            spec, condition = self._training_spec_and_condition(row)
        else:
            row_index, spec, condition = self.eval_entries[int(index)]
            row = self.rows[row_index]

        mel_target, audio_target = self._load_mel_audio(row)
        mel_corrupted, missing_mask = corrupt_mel(
            mel_target,
            spec,
            boundary_margin=self.boundary_margin,
        )
        boundary_map = build_boundary_map(
            spec,
            mel_bins=self.mel_bins,
            mel_frames=self.mel_frames,
            dtype=mel_target.dtype,
        )

        video, flow = self._load_visual(row)
        visual_seed = stable_seed(
            "uq-visual",
            self.seed,
            self.epoch if self.phase == "train" else 0,
            row["sample_id"],
            spec.mask_type,
            spec.start,
            spec.end,
            condition,
        )
        degradation_kwargs = {}
        if condition == "wrong_video":
            replacement = self._wrong_video_row(row_index, visual_seed)
            wrong_video, wrong_flow = self._load_visual(replacement)
            degradation_kwargs = {
                "wrong_video": wrong_video,
                "wrong_flow": wrong_flow,
                "wrong_video_sample_id": replacement["sample_id"],
            }
        video, flow, degradation = apply_visual_degradation(
            video,
            flow,
            condition,
            visual_seed,
            **degradation_kwargs,
        )

        return {
            "sample_id": row["sample_id"],
            "mel_target": mel_target,
            "mel_corrupted": mel_corrupted,
            "missing_mask": missing_mask,
            "boundary_map": boundary_map,
            "video": torch.from_numpy(np.ascontiguousarray(video)),
            "flow": torch.from_numpy(np.ascontiguousarray(flow)),
            "audio_target": audio_target,
            "mask_spec": spec,
            "video_condition": condition,
            "video_degradation": degradation,
        }


def uq_av_collate_fn(batch):
    if not batch:
        raise ValueError("Cannot collate an empty UQ AV batch")
    tensor_fields = (
        "mel_target",
        "mel_corrupted",
        "missing_mask",
        "boundary_map",
        "video",
        "flow",
        "audio_target",
    )
    output = {
        field: torch.stack([sample[field] for sample in batch], dim=0)
        for field in tensor_fields
    }
    for field in (
        "sample_id",
        "mask_spec",
        "video_condition",
        "video_degradation",
    ):
        output[field] = [sample[field] for sample in batch]
    return output


def create_uq_av_dataloader(
    data_root,
    split_name,
    phase,
    batch_size=16,
    num_workers=4,
    shuffle=None,
    pin_memory=True,
    drop_last=False,
    **dataset_kwargs,
):
    dataset = UQAVDataset(
        data_root=data_root,
        split_name=split_name,
        phase=phase,
        **dataset_kwargs,
    )
    if shuffle is None:
        shuffle = phase == "train"
    generator = torch.Generator()
    generator.manual_seed(int(dataset.seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        drop_last=bool(drop_last),
        collate_fn=uq_av_collate_fn,
        generator=generator,
    )


def get_uq_av_data_loaders(
    data_root,
    split_names,
    phases=("train", "val"),
    **loader_kwargs,
):
    loaders = {}
    for phase in phases:
        if phase not in split_names:
            raise KeyError(f"Missing split name for phase {phase}")
        loaders[phase] = create_uq_av_dataloader(
            data_root=data_root,
            split_name=split_names[phase],
            phase=phase,
            **loader_kwargs,
        )
    return loaders
