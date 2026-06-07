import argparse
from collections import defaultdict
import random
from pathlib import Path

from utils.av_sample_validation import BadAVSampleError, log_bad_sample, validate_av_sample


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create VIAI train/val/test split files from processed MUSICES samples."
    )
    parser.add_argument("--data-root", default="/root/shared-nvme/data")
    parser.add_argument("--processed-dir", default="processed")
    parser.add_argument("--train-split-name", default="train_new_split.txt")
    parser.add_argument("--val-split-name", default="val_new_split.txt")
    parser.add_argument("--test-split-name", default="test_new_split.txt")
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--hop-size", type=int, default=320)
    parser.add_argument("--max-time-steps", type=int, default=64000)
    parser.add_argument("--image-hope-size", type=int, default=1)
    parser.add_argument("--visual-frame-count", type=int, default=50)
    parser.add_argument("--visual-frame-interval-sec", type=float, default=0.08)
    parser.add_argument(
        "--bad-sample-log",
        "--bad_sample_log",
        dest="bad_sample_log",
        default=None,
    )
    parser.add_argument(
        "--strict-av-samples",
        "--strict_av_samples",
        dest="strict_av_samples",
        action="store_true",
    )
    parser.add_argument(
        "--audio-only",
        action="store_true",
        help="Only require raw_audio.npy and mel.npy, and default to VIAI-A split filenames.",
    )
    parser.add_argument(
        "--allow-empty-eval",
        action="store_true",
        help="Allow val/test split files to be empty. Useful for one-sample smoke tests.",
    )
    args = parser.parse_args()
    if args.audio_only:
        if args.train_split_name == "train_new_split.txt":
            args.train_split_name = "train_viai_a_split.txt"
        if args.val_split_name == "val_new_split.txt":
            args.val_split_name = "val_viai_a_split.txt"
        if args.test_split_name == "test_new_split.txt":
            args.test_split_name = "test_viai_a_split.txt"
    return args


def validate_ratio(name, value):
    if value < 0.0 or value >= 1.0:
        raise ValueError(f"{name} must be in [0, 1), got {value}")


def sample_ready(sample_dir, audio_only=False):
    required_files = [
        sample_dir / "raw_audio.npy",
        sample_dir / "mel.npy",
    ]
    if audio_only:
        return all(path.exists() for path in required_files)
    required_dirs = [
        sample_dir / "image_crop",
        sample_dir / "flow_x_crop",
        sample_dir / "flow_y_crop",
    ]
    return all(path.exists() for path in required_files) and all(path.exists() for path in required_dirs)


def source_video_key(sample_dir, data_root, processed_dir):
    relative = sample_dir.relative_to(Path(data_root) / processed_dir)
    parts = relative.parts
    if len(parts) < 2:
        raise ValueError(f"Cannot infer source video key from sample directory: {sample_dir}")
    return "/".join(parts[:2])


def discover_sample_dirs(processed_root, audio_only=False):
    sample_dirs = []
    for sample_dir in sorted(path for path in processed_root.rglob("*") if path.is_dir()):
        if sample_ready(sample_dir, audio_only=audio_only):
            sample_dirs.append(sample_dir)
    if audio_only:
        return sample_dirs
    filtered = []
    for sample_dir in sample_dirs:
        has_ready_shots = any(
            child.is_dir() and child.name.startswith("shot_") and sample_ready(child, audio_only=audio_only)
            for child in sample_dir.iterdir()
        )
        if has_ready_shots:
            continue
        filtered.append(sample_dir)
    return filtered


def discover_samples(data_root, processed_dir, max_samples=None, audio_only=False, args=None):
    import numpy as np
    from tqdm import tqdm

    processed_root = Path(data_root) / processed_dir
    if not processed_root.exists():
        raise RuntimeError(f"Processed data directory not found: {processed_root}")

    sample_dirs = discover_sample_dirs(processed_root, audio_only=audio_only)
    rows = []
    invalid_count = 0
    progress = tqdm(
        sample_dirs,
        desc="[split_musices] scanning",
        unit="sample",
    )
    for sample_dir in progress:
        progress.set_postfix_str(sample_dir.relative_to(data_root).as_posix())
        if not sample_ready(sample_dir, audio_only=audio_only):
            continue

        mel_path = sample_dir / "mel.npy"
        audio_path = sample_dir / "raw_audio.npy"
        if not audio_only:
            try:
                validate_av_sample(sample_dir, args)
            except BadAVSampleError as exc:
                if args.strict_av_samples:
                    raise
                invalid_count += 1
                log_path = log_bad_sample(
                    data_root,
                    args.bad_sample_log,
                    source="split-data",
                    phase="",
                    split_name="",
                    sample_path=sample_dir,
                    error=exc,
                )
                progress.write(
                    f"[split_musices] excluded invalid AV sample: "
                    f"{sample_dir.relative_to(data_root).as_posix()} "
                    f"reason={exc.reason} log={log_path}"
                )
                continue
        mel = np.load(mel_path, mmap_mode="r")
        rows.append(
            {
                "sample_dir": sample_dir.relative_to(data_root).as_posix(),
                "mel_path": mel_path.relative_to(data_root).as_posix(),
                "audio_path": audio_path.relative_to(data_root).as_posix(),
                "mel_frames": int(mel.shape[0]),
                "source_video": source_video_key(sample_dir, data_root, processed_dir),
            }
        )

    if max_samples is not None:
        rows = rows[:max_samples]
    discover_samples.invalid_count = invalid_count
    return rows


discover_samples.invalid_count = 0


def split_count(total, ratio, allow_empty):
    if ratio <= 0.0 or total <= 0:
        return 0
    count = int(round(total * ratio))
    if count == 0 and not allow_empty:
        count = 1
    return min(count, total)


def flatten_groups(groups, keys):
    rows = []
    for key in sorted(keys):
        rows.extend(sorted(groups[key], key=lambda item: item["sample_dir"]))
    return rows


def make_splits(rows, test_size, val_size, seed, allow_empty_eval):
    validate_ratio("test-size", test_size)
    validate_ratio("val-size", val_size)
    if test_size + val_size >= 1.0:
        raise ValueError("test-size + val-size must be less than 1")

    groups = defaultdict(list)
    for row in rows:
        groups[row["source_video"]].append(row)

    source_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(source_keys)

    total = len(source_keys)
    test_count = split_count(total, test_size, allow_empty_eval)
    val_count = min(split_count(total, val_size, allow_empty_eval), total - test_count)

    test_keys = source_keys[:test_count]
    val_keys = source_keys[test_count:test_count + val_count]
    train_keys = source_keys[test_count + val_count:]
    test_items = flatten_groups(groups, test_keys)
    val_items = flatten_groups(groups, val_keys)
    train_items = flatten_groups(groups, train_keys)
    if not train_items:
        raise RuntimeError(
            "Split would leave no training samples. Reduce --test-size/--val-size "
            "or use fewer eval samples for smoke tests."
        )
    split_keys = {
        "train": set(train_keys),
        "val": set(val_keys),
        "test": set(test_keys),
    }
    return train_items, val_items, test_items, split_keys


def write_split(path, rows):
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda item: item["sample_dir"]):
            handle.write(
                "|".join(
                    [
                        row["sample_dir"],
                        row["mel_path"],
                        row["audio_path"],
                        str(row["mel_frames"]),
                    ]
                )
                + "\n"
            )


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    rows = discover_samples(
        data_root,
        args.processed_dir,
        args.max_samples,
        audio_only=args.audio_only,
        args=args,
    )
    if not rows:
        if args.audio_only:
            raise RuntimeError("No audio-only processed samples found. Run `main.py prepare-viai-a` first.")
        raise RuntimeError("No processed samples found. Run the process stage first.")

    train_items, val_items, test_items, split_keys = make_splits(
        rows,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
        allow_empty_eval=args.allow_empty_eval,
    )

    outputs = {
        "train": (data_root / args.train_split_name, train_items),
        "val": (data_root / args.val_split_name, val_items),
        "test": (data_root / args.test_split_name, test_items),
    }
    for _, (path, items) in outputs.items():
        write_split(path, items)

    print(
        "[split_musices] wrote splits: "
        f"train={len(train_items)} samples/{len(split_keys['train'])} videos ({outputs['train'][0]}), "
        f"val={len(val_items)} samples/{len(split_keys['val'])} videos ({outputs['val'][0]}), "
        f"test={len(test_items)} samples/{len(split_keys['test'])} videos ({outputs['test'][0]})"
    )
    if not args.audio_only and discover_samples.invalid_count:
        log_path = args.bad_sample_log or (data_root / "viai_av_bad_samples.csv")
        print(
            f"[split_musices] excluded {discover_samples.invalid_count} invalid AV samples; "
            f"details: {log_path}"
        )


if __name__ == "__main__":
    main()
