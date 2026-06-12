import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from utils.baseline_evaluation import git_metadata, utc_now
from utils.baseline_protocol import (
    DEFAULT_SPLITS,
    create_baseline_protocol,
    sha256_file,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Freeze reproducible VIAI-A, VIAI-AV, and VIAI-AA' baselines."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--viai-a-checkpoint", required=True)
    parser.add_argument("--viai-av-checkpoint", required=True)
    parser.add_argument("--output-root", default="experiments/baselines")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-gap-frames", type=int, default=20)
    parser.add_argument("--max-gap-frames", type=int, default=50)
    parser.add_argument("--use-vocoder", action="store_true")
    parser.add_argument("--vocoder-max-samples", type=int, default=None)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--train-viai-a-split", default="train_viai_a_split.txt")
    parser.add_argument("--val-viai-a-split", default="val_viai_a_split.txt")
    parser.add_argument("--test-viai-a-split", default="test_viai_a_split.txt")
    parser.add_argument("--train-av-split", default="train_av_split.txt")
    parser.add_argument("--val-av-split", default="val_av_split.txt")
    parser.add_argument("--test-av-split", default="test_av_split.txt")
    return parser.parse_args(argv)


def ensure_empty_output_root(path):
    path = Path(path)
    if path.exists() and not path.is_dir():
        raise RuntimeError(f"Baseline output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(
            f"Baseline output directory is not empty: {path}. "
            "Choose a new directory; existing baseline results are never overwritten."
        )
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def ensure_clean_worktree(repo_root, allow_dirty=False):
    metadata = git_metadata(repo_root)
    if metadata["dirty"] and not allow_dirty:
        raise RuntimeError(
            "Git worktree is dirty. Commit/stash changes before freezing baselines, "
            "or rerun with --allow-dirty to record the dirty state explicitly."
        )
    return metadata


def inspect_checkpoint(path, require_probe=False):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu")
    use_gan = bool(checkpoint.get("use_gan", "netD" in checkpoint))
    probe_enabled = bool(checkpoint.get("enable_probe_loss", False))
    if require_probe and not probe_enabled:
        raise RuntimeError(
            f"VIAI-AV checkpoint does not declare enable_probe_loss=True: {path}"
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "use_gan": use_gan,
        "probe_enabled": probe_enabled,
        "stage": checkpoint.get("stage", ""),
        "global_step": int(checkpoint.get("global_step", 0)),
        "global_epoch": int(checkpoint.get("global_epoch", 0)),
    }


def build_baseline_commands(
    repo_root,
    data_root,
    output_root,
    protocol_path,
    mask_manifest_path,
    viai_a_checkpoint,
    viai_av_checkpoint,
    viai_a_use_gan=False,
    viai_av_use_gan=False,
    test_av_split="test_av_split.txt",
    batch_size=16,
    num_workers=4,
    seed=1234,
    use_vocoder=False,
    vocoder_max_samples=None,
):
    main_path = str(Path(repo_root) / "main.py")
    shared = [
        "--data_root",
        str(data_root),
        "--test_split_name",
        str(test_av_split),
        "--batch_size",
        str(batch_size),
        "--num_workers",
        str(num_workers),
        "--baseline-mask-manifest",
        str(mask_manifest_path),
        "--baseline-protocol-json",
        str(protocol_path),
        "--eval-seed",
        str(seed),
        "--deterministic-eval",
        "--strict_av_samples",
        "--display_id",
        "0",
    ]
    if use_vocoder:
        shared.append("--use_vocoder")
        if vocoder_max_samples is not None:
            shared.extend(["--vocoder_max_samples", str(vocoder_max_samples)])

    viai_a = [
        sys.executable,
        main_path,
        "test-viai-a",
        "--",
        *shared,
        "--resume_path",
        str(viai_a_checkpoint),
        "--results_dir",
        str(Path(output_root) / "viai_a"),
    ]
    if viai_a_use_gan:
        viai_a.append("--use_gan")

    def av_command(branch, directory):
        command = [
            sys.executable,
            main_path,
            "test-viai-av",
            "--",
            *shared,
            "--resume_path",
            str(viai_av_checkpoint),
            "--results_dir",
            str(Path(output_root) / directory),
            "--eval-branch",
            branch,
        ]
        if viai_av_use_gan:
            command.append("--use_gan")
        return command

    return {
        "viai_a": viai_a,
        "viai_av": av_command("av", "viai_av"),
        "viai_aa_probe": av_command("probe", "viai_aa_probe"),
    }


def run_commands(commands, repo_root, seed, git_info):
    base_environment = os.environ.copy()
    base_environment["PYTHONHASHSEED"] = str(seed)
    base_environment["VIAI_BASELINE_GIT_JSON"] = json.dumps(
        git_info,
        ensure_ascii=True,
        sort_keys=True,
    )
    for baseline_name, command in commands.items():
        print(f"[freeze-viai-baselines] running {baseline_name}: {' '.join(command)}")
        environment = base_environment.copy()
        environment["VIAI_BASELINE_COMMAND_JSON"] = json.dumps(
            command,
            ensure_ascii=True,
        )
        subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=True,
        )


def load_summary(output_root, baseline_name):
    path = Path(output_root) / baseline_name / "summary.json"
    if not path.exists():
        raise RuntimeError(f"Baseline did not produce summary.json: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None):
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    started_at = utc_now()
    git_info = ensure_clean_worktree(repo_root, allow_dirty=args.allow_dirty)
    output_root = ensure_empty_output_root(args.output_root)
    data_root = Path(args.data_root).resolve()

    viai_a_checkpoint = inspect_checkpoint(args.viai_a_checkpoint)
    viai_av_checkpoint = inspect_checkpoint(
        args.viai_av_checkpoint,
        require_probe=True,
    )
    split_names = {
        "viai_a": {
            "train": args.train_viai_a_split,
            "val": args.val_viai_a_split,
            "test": args.test_viai_a_split,
        },
        "viai_av": {
            "train": args.train_av_split,
            "val": args.val_av_split,
            "test": args.test_av_split,
        },
    }
    if set(split_names) != set(DEFAULT_SPLITS):
        raise RuntimeError("Unexpected baseline split groups")
    protocol_path, protocol = create_baseline_protocol(
        data_root,
        output_root / "protocol",
        split_names=split_names,
        seed=args.seed,
        min_gap_frames=args.min_gap_frames,
        max_gap_frames=args.max_gap_frames,
    )
    mask_manifest_path = protocol["mask_manifest"]["path"]
    commands = build_baseline_commands(
        repo_root=repo_root,
        data_root=data_root,
        output_root=output_root,
        protocol_path=protocol_path,
        mask_manifest_path=mask_manifest_path,
        viai_a_checkpoint=viai_a_checkpoint["path"],
        viai_av_checkpoint=viai_av_checkpoint["path"],
        viai_a_use_gan=viai_a_checkpoint["use_gan"],
        viai_av_use_gan=viai_av_checkpoint["use_gan"],
        test_av_split=protocol["splits"]["viai_av"]["test"]["frozen_path"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        use_vocoder=args.use_vocoder,
        vocoder_max_samples=args.vocoder_max_samples,
    )
    run_commands(commands, repo_root, args.seed, git_info)

    summaries = {
        name: load_summary(output_root, name)
        for name in ("viai_a", "viai_av", "viai_aa_probe")
    }
    suite = {
        "version": 1,
        "started_at": started_at,
        "ended_at": utc_now(),
        "data_root": str(data_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "git": git_info,
        "protocol": {
            "path": str(protocol_path.resolve()),
            "sha256": sha256_file(protocol_path),
        },
        "checkpoints": {
            "viai_a": viai_a_checkpoint,
            "viai_av": viai_av_checkpoint,
        },
        "commands": commands,
        "summaries": summaries,
    }
    suite_path = output_root / "suite.json"
    with suite_path.open("w", encoding="utf-8") as handle:
        json.dump(suite, handle, ensure_ascii=True, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[freeze-viai-baselines] complete: {suite_path}")


if __name__ == "__main__":
    main()
