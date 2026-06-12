from dataclasses import asdict, dataclass
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from Data_loaders.mel_loader import corrupt_mel_spectrogram_batch


MASK_TYPES = ("random", "onset_centered", "boundary_near", "long_gap")


def stable_seed(*parts):
    payload = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & ((1 << 63) - 1)


@dataclass(frozen=True)
class MaskSpec:
    mask_type: str
    start: int
    end: int
    gap_frames: int
    seed: int

    def validate(self, mel_frames=200, boundary_margin=3):
        if self.mask_type not in MASK_TYPES:
            raise ValueError(f"Unsupported mask type: {self.mask_type}")
        if self.start < boundary_margin:
            raise ValueError(
                f"Mask start {self.start} violates boundary margin {boundary_margin}"
            )
        if self.end > mel_frames - boundary_margin:
            raise ValueError(
                f"Mask end {self.end} violates boundary margin {boundary_margin}"
            )
        if self.end <= self.start or self.end - self.start != self.gap_frames:
            raise ValueError(
                "Mask span and gap_frames are inconsistent: "
                f"start={self.start} end={self.end} gap_frames={self.gap_frames}"
            )
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value, mel_frames=200, boundary_margin=3):
        required = {"mask_type", "start", "end", "seed"}
        missing = sorted(required - set(value))
        if missing:
            raise ValueError(f"Mask spec is missing fields: {', '.join(missing)}")
        start = int(value["start"])
        end = int(value["end"])
        spec = cls(
            mask_type=str(value["mask_type"]),
            start=start,
            end=end,
            gap_frames=int(value.get("gap_frames", end - start)),
            seed=int(value["seed"]),
        )
        return spec.validate(
            mel_frames=mel_frames,
            boundary_margin=boundary_margin,
        )


def spectral_flux(mel):
    mel = np.asarray(mel, dtype=np.float32)
    if mel.ndim != 2:
        raise ValueError(f"Expected a 2D Mel array, got shape {mel.shape}")
    if mel.shape[0] < 2:
        return np.zeros(mel.shape[0], dtype=np.float32)
    delta = np.diff(mel, axis=0)
    flux = np.maximum(delta, 0.0).mean(axis=1)
    return np.concatenate(
        [np.zeros(1, dtype=np.float32), flux.astype(np.float32)]
    )


def _high_flux_peaks(onset_strengths, minimum, maximum):
    values = np.asarray(onset_strengths, dtype=np.float32).reshape(-1)
    if values.size == 0 or maximum < minimum:
        return []
    minimum = max(0, int(minimum))
    maximum = min(values.size - 1, int(maximum))
    legal = values[minimum : maximum + 1]
    positive = legal[legal > 0]
    if not positive.size:
        return []
    threshold = float(np.percentile(positive, 75.0))
    peaks = []
    for index in range(minimum, maximum + 1):
        left = values[index - 1] if index > 0 else -np.inf
        right = values[index + 1] if index + 1 < values.size else -np.inf
        if values[index] >= threshold and values[index] >= left and values[index] >= right:
            peaks.append(index)
    return peaks


class MaskSampler:
    def __init__(
        self,
        mel_frames=200,
        min_gap_frames=20,
        max_gap_frames=50,
        boundary_margin=3,
        long_gap_frames=(60, 80, 100),
    ):
        self.mel_frames = int(mel_frames)
        self.min_gap_frames = int(min_gap_frames)
        self.max_gap_frames = int(max_gap_frames)
        self.boundary_margin = int(boundary_margin)
        self.long_gap_frames = tuple(int(value) for value in long_gap_frames)
        if self.min_gap_frames <= 0 or self.max_gap_frames < self.min_gap_frames:
            raise ValueError("Invalid random gap range")
        if not self.long_gap_frames or min(self.long_gap_frames) <= 0:
            raise ValueError("long_gap_frames must contain positive values")
        largest_gap = max(self.max_gap_frames, max(self.long_gap_frames))
        if largest_gap + 2 * self.boundary_margin > self.mel_frames:
            raise ValueError("Configured gaps do not fit the Mel window")

    def _random_width(self, rng):
        return rng.randint(self.min_gap_frames, self.max_gap_frames)

    def _random_start(self, rng, width):
        maximum = self.mel_frames - self.boundary_margin - width
        return rng.randint(self.boundary_margin, maximum)

    def sample(self, mask_type, seed, onset_strengths=None):
        if mask_type not in MASK_TYPES:
            raise ValueError(f"Unsupported mask type: {mask_type}")
        seed = int(seed)
        rng = random.Random(seed)

        if mask_type == "long_gap":
            width = rng.choice(self.long_gap_frames)
            start = self._random_start(rng, width)
        else:
            width = self._random_width(rng)
            if mask_type == "random":
                start = self._random_start(rng, width)
            elif mask_type == "boundary_near":
                if rng.choice(("left", "right")) == "left":
                    start = self.boundary_margin
                else:
                    start = self.mel_frames - self.boundary_margin - width
            else:
                if onset_strengths is None:
                    raise ValueError("onset_centered masks require onset strengths")
                minimum_center = self.boundary_margin + width // 2
                maximum_center = (
                    self.mel_frames
                    - self.boundary_margin
                    - (width - width // 2)
                )
                peaks = _high_flux_peaks(
                    onset_strengths,
                    minimum_center,
                    maximum_center,
                )
                values = np.asarray(onset_strengths, dtype=np.float32).reshape(-1)
                if peaks:
                    top_value = max(float(values[index]) for index in peaks)
                    top_peaks = [
                        index
                        for index in peaks
                        if np.isclose(float(values[index]), top_value)
                    ]
                    center = rng.choice(top_peaks)
                else:
                    legal = values[minimum_center : maximum_center + 1]
                    center = minimum_center + int(np.argmax(legal))
                start = center - width // 2

        spec = MaskSpec(
            mask_type=mask_type,
            start=int(start),
            end=int(start + width),
            gap_frames=int(width),
            seed=seed,
        )
        return spec.validate(
            mel_frames=self.mel_frames,
            boundary_margin=self.boundary_margin,
        )


def build_missing_mask(spec, mel_bins=80, mel_frames=200, dtype=torch.float32):
    spec = _coerce_spec(spec, mel_frames=mel_frames)
    mask = torch.zeros(1, int(mel_bins), int(mel_frames), dtype=dtype)
    mask[:, :, spec.start : spec.end] = 1.0
    return mask


def build_boundary_map(spec, mel_bins=80, mel_frames=200, dtype=torch.float32):
    spec = _coerce_spec(spec, mel_frames=mel_frames)
    positions = torch.arange(mel_frames, dtype=dtype)
    scale = float(max(1, mel_frames - 1))
    left = torch.abs(positions - float(spec.start)) / scale
    right = torch.abs(positions - float(spec.end - 1)) / scale
    return torch.stack((left, right), dim=0).unsqueeze(1).expand(
        2,
        int(mel_bins),
        int(mel_frames),
    ).clone()


def corrupt_mel(mel_target, spec, boundary_margin=3):
    if mel_target.dim() != 3 or mel_target.size(0) != 1:
        raise ValueError(
            "mel_target must have shape [1, mel_bins, mel_frames], "
            f"got {tuple(mel_target.shape)}"
        )
    spec = _coerce_spec(
        spec,
        mel_frames=mel_target.size(-1),
        boundary_margin=boundary_margin,
    )
    corrupted, mask, _ = corrupt_mel_spectrogram_batch(
        mel_target.unsqueeze(0),
        [spec.to_dict()],
        min_margin=boundary_margin,
    )
    return corrupted.squeeze(0), mask.squeeze(0)


def load_uq_mask_manifest(path, mel_frames=200, boundary_margin=3):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"UQ mask manifest not found: {path}")
    manifest = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            sample_id = str(record.get("sample_id", "")).strip()
            variants = record.get("variants")
            if not sample_id or not isinstance(variants, list) or not variants:
                raise ValueError(
                    f"Invalid UQ mask record in {path}:{line_number}"
                )
            if sample_id in manifest:
                raise ValueError(f"Duplicate sample_id in UQ mask manifest: {sample_id}")
            manifest[sample_id] = [
                MaskSpec.from_dict(
                    variant,
                    mel_frames=mel_frames,
                    boundary_margin=boundary_margin,
                )
                for variant in variants
            ]
    if not manifest:
        raise ValueError(f"UQ mask manifest is empty: {path}")
    return manifest


def _coerce_spec(spec, mel_frames=200, boundary_margin=3):
    if isinstance(spec, MaskSpec):
        return spec.validate(
            mel_frames=mel_frames,
            boundary_margin=boundary_margin,
        )
    return MaskSpec.from_dict(
        spec,
        mel_frames=mel_frames,
        boundary_margin=boundary_margin,
    )
