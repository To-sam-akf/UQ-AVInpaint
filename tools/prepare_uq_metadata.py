import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from Data_loaders.mask_sampler import (
    MASK_TYPES,
    MaskSampler,
    spectral_flux,
    stable_seed,
)
from Data_loaders.uq_av_loader import read_uq_split
from utils.baseline_protocol import sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare deterministic UQ-AVInpaint onset and mask metadata."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-split-name", default="train_av_split.txt")
    parser.add_argument("--val-split-name", default="val_av_split.txt")
    parser.add_argument("--test-split-name", default="test_av_split.txt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--mel-frames", type=int, default=200)
    parser.add_argument("--mel-bins", type=int, default=80)
    parser.add_argument("--audio-steps", type=int, default=64000)
    parser.add_argument("--visual-frames", type=int, default=50)
    parser.add_argument("--boundary-margin", type=int, default=3)
    parser.add_argument("--min-gap-frames", type=int, default=20)
    parser.add_argument("--max-gap-frames", type=int, default=50)
    parser.add_argument(
        "--long-gap-frames",
        type=int,
        nargs="+",
        default=[60, 80, 100],
    )
    return parser.parse_args(argv)


def _count_jpg_files(path):
    path = Path(path)
    if not path.is_dir():
        return 0
    return sum(1 for item in path.glob("*.jpg") if item.is_file())


def validate_uq_row(row, mel_frames, mel_bins, audio_steps, visual_frames):
    if row["mel_frames"] != mel_frames:
        raise ValueError(
            f"Sample {row['sample_id']} declares {row['mel_frames']} Mel frames; "
            f"expected {mel_frames}"
        )
    mel = np.load(row["mel_path"], mmap_mode="r", allow_pickle=False)
    audio = np.load(row["audio_path"], mmap_mode="r", allow_pickle=False)
    if mel.shape != (mel_frames, mel_bins):
        raise ValueError(
            f"Sample {row['sample_id']} has Mel shape {mel.shape}; "
            f"expected ({mel_frames}, {mel_bins})"
        )
    if audio.shape != (audio_steps,):
        raise ValueError(
            f"Sample {row['sample_id']} has audio shape {audio.shape}; "
            f"expected ({audio_steps},)"
        )
    for directory_name in ("image_crop", "flow_x_crop", "flow_y_crop"):
        directory = row["sample_dir"] / directory_name
        count = _count_jpg_files(directory)
        if count != visual_frames:
            raise ValueError(
                f"Sample {row['sample_id']} has {count} frames in {directory_name}; "
                f"expected {visual_frames}"
            )
        missing = [
            index
            for index in range(1, visual_frames + 1)
            if not (directory / f"{index}.jpg").is_file()
        ]
        if missing:
            raise ValueError(
                f"Sample {row['sample_id']} is missing numbered frames in "
                f"{directory_name}: {missing[:5]}"
            )
    return np.asarray(mel, dtype=np.float32)


def _write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(
                record,
                handle,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write("\n")
    return path


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    return path


def _mask_distribution(records):
    type_counts = Counter()
    gap_counts = Counter()
    for record in records:
        for variant in record["variants"]:
            type_counts[variant["mask_type"]] += 1
            gap_counts[str(variant["gap_frames"])] += 1
    return {
        "mask_type_counts": dict(sorted(type_counts.items())),
        "gap_frame_counts": dict(
            sorted(gap_counts.items(), key=lambda item: int(item[0]))
        ),
    }


def prepare_uq_metadata(args):
    data_root = Path(args.data_root).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else data_root / "uq_metadata"
    )
    split_names = {
        "train": args.train_split_name,
        "val": args.val_split_name,
        "test": args.test_split_name,
    }
    rows_by_phase = {
        phase: read_uq_split(data_root, split_name)
        for phase, split_name in split_names.items()
    }
    sampler = MaskSampler(
        mel_frames=args.mel_frames,
        min_gap_frames=args.min_gap_frames,
        max_gap_frames=args.max_gap_frames,
        boundary_margin=args.boundary_margin,
        long_gap_frames=args.long_gap_frames,
    )

    mel_by_phase = {}
    for phase, rows in rows_by_phase.items():
        mel_by_phase[phase] = {}
        for row in rows:
            mel_by_phase[phase][row["sample_id"]] = validate_uq_row(
                row,
                mel_frames=args.mel_frames,
                mel_bins=args.mel_bins,
                audio_steps=args.audio_steps,
                visual_frames=args.visual_frames,
            )

    onset_root = output_dir / "train_onsets"
    for row in rows_by_phase["train"]:
        onset = spectral_flux(mel_by_phase["train"][row["sample_id"]])
        onset_path = onset_root / Path(row["sample_id"]).with_suffix(".npy")
        onset_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(onset_path, onset.astype(np.float32), allow_pickle=False)

    manifest_paths = {}
    manifest_records = {}
    for phase in ("val", "test"):
        records = []
        for row in rows_by_phase[phase]:
            onset = spectral_flux(mel_by_phase[phase][row["sample_id"]])
            variants = []
            for mask_type in MASK_TYPES:
                mask_seed = stable_seed(
                    "uq-eval-mask",
                    args.seed,
                    phase,
                    row["sample_id"],
                    mask_type,
                )
                spec = sampler.sample(
                    mask_type,
                    mask_seed,
                    onset_strengths=onset if mask_type == "onset_centered" else None,
                )
                variants.append(spec.to_dict())
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "variants": variants,
                }
            )
        records.sort(key=lambda record: record["sample_id"])
        path = _write_jsonl(output_dir / f"{phase}_masks.jsonl", records)
        manifest_paths[phase] = path
        manifest_records[phase] = records

    split_summary = {}
    for phase, split_name in split_names.items():
        split_path = Path(split_name)
        if not split_path.is_absolute():
            split_path = data_root / split_path
        split_summary[phase] = {
            "name": str(split_name),
            "sha256": sha256_file(split_path),
            "num_samples": len(rows_by_phase[phase]),
        }

    summary = {
        "version": 1,
        "seed": int(args.seed),
        "data_contract": {
            "mel_frames": int(args.mel_frames),
            "mel_bins": int(args.mel_bins),
            "audio_steps": int(args.audio_steps),
            "visual_frames": int(args.visual_frames),
            "boundary_margin": int(args.boundary_margin),
        },
        "mask_config": {
            "mask_types": list(MASK_TYPES),
            "min_gap_frames": int(args.min_gap_frames),
            "max_gap_frames": int(args.max_gap_frames),
            "long_gap_frames": [int(value) for value in args.long_gap_frames],
        },
        "splits": split_summary,
        "train_onsets": {
            "num_samples": len(rows_by_phase["train"]),
        },
        "manifests": {
            phase: {
                "filename": manifest_paths[phase].name,
                "sha256": sha256_file(manifest_paths[phase]),
                "num_samples": len(manifest_records[phase]),
                "num_variants": sum(
                    len(record["variants"]) for record in manifest_records[phase]
                ),
                **_mask_distribution(manifest_records[phase]),
            }
            for phase in ("val", "test")
        },
    }
    summary_path = _write_json(output_dir / "metadata_summary.json", summary)
    return {
        "output_dir": output_dir,
        "summary_path": summary_path,
        "summary": summary,
    }


def main(argv=None):
    args = parse_args(argv)
    result = prepare_uq_metadata(args)
    print(
        "[prepare-uq-metadata] wrote deterministic metadata: "
        f"{result['output_dir']}"
    )
    print(
        "[prepare-uq-metadata] samples: "
        f"train={result['summary']['splits']['train']['num_samples']} "
        f"val={result['summary']['splits']['val']['num_samples']} "
        f"test={result['summary']['splits']['test']['num_samples']}"
    )


if __name__ == "__main__":
    main()
