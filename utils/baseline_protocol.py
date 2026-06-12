import hashlib
import json
import random
import shutil
from pathlib import Path, PurePosixPath

import numpy as np


DEFAULT_SPLITS = {
    "viai_a": {
        "train": "train_viai_a_split.txt",
        "val": "val_viai_a_split.txt",
        "test": "test_viai_a_split.txt",
    },
    "viai_av": {
        "train": "train_av_split.txt",
        "val": "val_av_split.txt",
        "test": "test_av_split.txt",
    },
}


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sample_id(value, data_root=None, strip_window_index=False):
    raw = str(value).replace("\\", "/").rstrip("/")
    if data_root:
        root = str(Path(data_root).resolve()).replace("\\", "/").rstrip("/")
        if raw == root:
            raw = ""
        elif raw.startswith(root + "/"):
            raw = raw[len(root) + 1 :]
    parts = [part for part in PurePosixPath(raw).parts if part not in {"", "/", "."}]
    for marker in ("processed", "processed_viai_a"):
        if marker in parts:
            parts = parts[parts.index(marker) :]
            break
    if strip_window_index and parts and parts[-1].isdigit():
        parts = parts[:-1]
    return "/".join(parts)


def source_video_key(sample_id):
    parts = canonical_sample_id(sample_id).split("/")
    if parts and parts[0] in {"processed", "processed_viai_a"}:
        parts = parts[1:]
    if len(parts) < 2:
        raise ValueError(f"Cannot infer source video from sample id: {sample_id}")
    return "/".join(parts[:2])


def read_split_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) not in {4, 5}:
                raise ValueError(
                    f"Expected 4 or 5 columns in {path}:{line_number}, got {len(parts)}"
                )
            rows.append(
                {
                    "sample_id": canonical_sample_id(parts[0]),
                    "sample_dir": parts[0],
                    "mel_path": parts[1],
                    "audio_path": parts[2],
                    "mel_frames": int(parts[-1]),
                    "source_video": source_video_key(parts[0]),
                    "raw_line": line,
                }
            )
    if not rows:
        raise ValueError(f"Split file is empty: {path}")
    return rows


def audit_model_splits(rows_by_phase, model_name):
    assignments = {}
    summaries = {}
    for phase, rows in rows_by_phase.items():
        videos = {row["source_video"] for row in rows}
        summaries[phase] = {
            "num_samples": len(rows),
            "num_videos": len(videos),
        }
        for video in videos:
            previous = assignments.get(video)
            if previous is not None and previous != phase:
                raise ValueError(
                    f"{model_name} split leakage: source video {video} appears in "
                    f"both {previous} and {phase}"
                )
            assignments[video] = phase
    return assignments, summaries


def audit_cross_model_assignments(viai_a_assignments, viai_av_assignments):
    common = sorted(set(viai_a_assignments) & set(viai_av_assignments))
    conflicts = [
        {
            "source_video": video,
            "viai_a_phase": viai_a_assignments[video],
            "viai_av_phase": viai_av_assignments[video],
        }
        for video in common
        if viai_a_assignments[video] != viai_av_assignments[video]
    ]
    if conflicts:
        first = conflicts[0]
        raise ValueError(
            "VIAI-A/VIAI-AV split assignment conflict for "
            f"{first['source_video']}: VIAI-A={first['viai_a_phase']} "
            f"VIAI-AV={first['viai_av_phase']}"
        )
    return {
        "common_source_videos": len(common),
        "conflicts": conflicts,
    }


def _resolve_data_path(data_root, value):
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(data_root) / path


def validate_av_test_row(
    row,
    data_root,
    expected_mel_frames=200,
    required_visual_frames=50,
    hop_size=320,
):
    if row["mel_frames"] != expected_mel_frames:
        raise ValueError(
            f"AV test sample {row['sample_id']} declares {row['mel_frames']} Mel frames; "
            f"expected {expected_mel_frames}"
        )
    sample_dir = _resolve_data_path(data_root, row["sample_dir"])
    mel_path = _resolve_data_path(data_root, row["mel_path"])
    audio_path = _resolve_data_path(data_root, row["audio_path"])
    required_paths = [
        sample_dir,
        mel_path,
        audio_path,
        sample_dir / "image_crop",
        sample_dir / "flow_x_crop",
        sample_dir / "flow_y_crop",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise ValueError(
            f"Incomplete AV test sample {row['sample_id']}: missing {', '.join(missing)}"
        )
    mel = np.load(mel_path, mmap_mode="r")
    if mel.ndim != 2 or mel.shape[0] != expected_mel_frames:
        raise ValueError(
            f"AV test sample {row['sample_id']} has Mel shape {mel.shape}; "
            f"expected ({expected_mel_frames}, mel_bins)"
        )
    audio = np.load(audio_path, mmap_mode="r")
    expected_audio_steps = expected_mel_frames * int(hop_size)
    if audio.ndim != 1 or audio.shape[0] < expected_audio_steps:
        raise ValueError(
            f"AV test sample {row['sample_id']} has audio shape {audio.shape}; "
            f"expected at least ({expected_audio_steps},)"
        )
    for directory_name in ("image_crop", "flow_x_crop", "flow_y_crop"):
        directory = sample_dir / directory_name
        missing_frames = [
            index
            for index in range(1, required_visual_frames + 1)
            if not (directory / f"{index}.jpg").is_file()
        ]
        if missing_frames:
            raise ValueError(
                f"Incomplete AV test sample {row['sample_id']}: {directory_name} "
                f"is missing required frames {missing_frames[:5]}"
            )


def generate_mask_specs(
    rows,
    seed=1234,
    min_gap_frames=20,
    max_gap_frames=50,
    mel_frames=200,
    boundary_margin=3,
):
    if min_gap_frames <= 0 or max_gap_frames < min_gap_frames:
        raise ValueError("Invalid gap frame range")
    if 2 * boundary_margin + min_gap_frames > mel_frames:
        raise ValueError("Gap range and boundary margin do not fit the Mel window")
    specs = []
    seen_sample_ids = set()
    for row in sorted(rows, key=lambda item: item["sample_id"]):
        sample_id = canonical_sample_id(row["sample_id"])
        if not sample_id:
            raise ValueError("Cannot generate a mask for an empty sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id in AV test split: {sample_id}")
        seen_sample_ids.add(sample_id)
        digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
        sample_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = random.Random(sample_seed)
        width = rng.randint(min_gap_frames, max_gap_frames)
        max_start = mel_frames - width - boundary_margin
        if max_start < boundary_margin:
            raise ValueError(f"Gap does not fit sample {sample_id}")
        start = rng.randint(boundary_margin, max_start)
        specs.append(
            {
                "sample_id": sample_id,
                "mask_type": "random",
                "start": start,
                "end": start + width,
                "gap_frames": width,
                "seed": int(seed),
            }
        )
    return specs


def validate_mask_spec(spec, mel_frames=200, boundary_margin=3):
    required = {"sample_id", "mask_type", "start", "end", "gap_frames", "seed"}
    missing = sorted(required - set(spec))
    if missing:
        raise ValueError(f"Mask spec is missing fields: {', '.join(missing)}")
    start = int(spec["start"])
    end = int(spec["end"])
    gap_frames = int(spec["gap_frames"])
    if spec["mask_type"] != "random":
        raise ValueError(f"Unsupported P0 mask type: {spec['mask_type']}")
    if start < boundary_margin or end > mel_frames - boundary_margin:
        raise ValueError(
            f"Mask for {spec['sample_id']} violates boundary margin: {start}:{end}"
        )
    if end <= start or end - start != gap_frames:
        raise ValueError(
            f"Mask for {spec['sample_id']} has inconsistent span/gap: "
            f"start={start} end={end} gap_frames={gap_frames}"
        )
    return {
        "sample_id": canonical_sample_id(spec["sample_id"]),
        "mask_type": str(spec["mask_type"]),
        "start": start,
        "end": end,
        "gap_frames": gap_frames,
        "seed": int(spec["seed"]),
    }


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, ensure_ascii=True, sort_keys=True)
            handle.write("\n")
    return path


def load_mask_manifest(path, mel_frames=200, boundary_margin=3):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Mask manifest not found: {path}")
    specs = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            spec = validate_mask_spec(
                raw,
                mel_frames=mel_frames,
                boundary_margin=boundary_margin,
            )
            sample_id = spec["sample_id"]
            if sample_id in specs:
                raise ValueError(f"Duplicate sample_id in mask manifest: {sample_id}")
            specs[sample_id] = spec
    if not specs:
        raise ValueError(f"Mask manifest is empty: {path}")
    return specs


def resolve_mask_specs(sample_paths, manifest, data_root=None):
    resolved = []
    for sample_path in sample_paths:
        exact = canonical_sample_id(sample_path, data_root=data_root)
        candidates = [exact]
        stripped = canonical_sample_id(
            sample_path,
            data_root=data_root,
            strip_window_index=True,
        )
        if stripped != exact:
            candidates.append(stripped)
        spec = next((manifest[candidate] for candidate in candidates if candidate in manifest), None)
        if spec is None:
            raise KeyError(
                f"No baseline mask found for runtime sample {sample_path}; "
                f"tried {', '.join(candidates)}"
            )
        resolved.append(dict(spec))
    return resolved


def create_baseline_protocol(
    data_root,
    protocol_dir,
    split_names=None,
    seed=1234,
    min_gap_frames=20,
    max_gap_frames=50,
    mel_frames=200,
    boundary_margin=3,
    required_visual_frames=50,
    hop_size=320,
):
    data_root = Path(data_root).resolve()
    protocol_dir = Path(protocol_dir)
    split_names = split_names or DEFAULT_SPLITS
    rows = {}
    split_sources = {}
    split_records = {}
    assignments = {}
    summaries = {}
    split_output_dir = protocol_dir / "splits"
    frozen_names = set()

    for model_name, phases in split_names.items():
        rows[model_name] = {}
        split_sources[model_name] = {}
        for phase, filename in phases.items():
            source_path = data_root / filename
            phase_rows = read_split_rows(source_path)
            rows[model_name][phase] = phase_rows
            frozen_name = Path(filename).name
            if frozen_name in frozen_names:
                raise ValueError(
                    f"Split filenames must be unique when frozen: {frozen_name}"
                )
            frozen_names.add(frozen_name)
            split_sources[model_name][phase] = {
                "source_path": source_path,
                "frozen_name": frozen_name,
            }
        assignments[model_name], summaries[model_name] = audit_model_splits(
            rows[model_name],
            model_name,
        )

    cross_model = audit_cross_model_assignments(
        assignments["viai_a"],
        assignments["viai_av"],
    )
    av_test_rows = sorted(rows["viai_av"]["test"], key=lambda item: item["sample_id"])
    for row in av_test_rows:
        validate_av_test_row(
            row,
            data_root,
            expected_mel_frames=mel_frames,
            required_visual_frames=required_visual_frames,
            hop_size=hop_size,
        )
    mask_specs = generate_mask_specs(
        av_test_rows,
        seed=seed,
        min_gap_frames=min_gap_frames,
        max_gap_frames=max_gap_frames,
        mel_frames=mel_frames,
        boundary_margin=boundary_margin,
    )

    split_output_dir.mkdir(parents=True, exist_ok=True)
    for model_name, phases in split_sources.items():
        split_records[model_name] = {}
        for phase, source in phases.items():
            frozen_name = source["frozen_name"]
            source_path = source["source_path"]
            frozen_path = split_output_dir / frozen_name
            shutil.copy2(source_path, frozen_path)
            split_records[model_name][phase] = {
                "source_path": str(source_path),
                "frozen_path": str(frozen_path.resolve()),
                "sha256": sha256_file(source_path),
                "num_samples": len(rows[model_name][phase]),
            }

    mask_path = write_jsonl(protocol_dir / "test_masks.jsonl", mask_specs)
    protocol = {
        "version": 1,
        "data_root": str(data_root),
        "seed": int(seed),
        "mask_type": "random",
        "min_gap_frames": int(min_gap_frames),
        "max_gap_frames": int(max_gap_frames),
        "mel_frames": int(mel_frames),
        "boundary_margin": int(boundary_margin),
        "required_visual_frames": int(required_visual_frames),
        "hop_size": int(hop_size),
        "splits": split_records,
        "split_summary": summaries,
        "cross_model_audit": cross_model,
        "mask_manifest": {
            "path": str(mask_path.resolve()),
            "sha256": sha256_file(mask_path),
            "num_samples": len(mask_specs),
        },
    }
    protocol_path = protocol_dir / "protocol.json"
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return protocol_path, protocol
