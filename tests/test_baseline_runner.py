import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.freeze_viai_baselines import (
    build_baseline_commands,
    ensure_empty_output_root,
    inspect_checkpoint,
)
from utils.baseline_evaluation import build_run_metadata


class BaselineRunnerTests(unittest.TestCase):
    def test_runner_builds_three_commands_with_gan_and_probe_selection(self):
        commands = build_baseline_commands(
            repo_root="/repo",
            data_root="/data",
            output_root="/results",
            protocol_path="/results/protocol/protocol.json",
            mask_manifest_path="/results/protocol/test_masks.jsonl",
            viai_a_checkpoint="/checkpoints/a.pth.tar",
            viai_av_checkpoint="/checkpoints/av.pth.tar",
            viai_a_use_gan=True,
            viai_av_use_gan=True,
            batch_size=8,
            num_workers=2,
        )

        self.assertEqual(set(commands), {"viai_a", "viai_av", "viai_aa_probe"})
        self.assertIn("--use_gan", commands["viai_a"])
        self.assertIn("--use_gan", commands["viai_av"])
        self.assertIn("--strict_av_samples", commands["viai_av"])
        av_index = commands["viai_av"].index("--eval-branch")
        probe_index = commands["viai_aa_probe"].index("--eval-branch")
        self.assertEqual(commands["viai_av"][av_index + 1], "av")
        self.assertEqual(commands["viai_aa_probe"][probe_index + 1], "probe")
        self.assertIn("/results/viai_a", commands["viai_a"])
        self.assertIn("/results/viai_av", commands["viai_av"])
        self.assertIn("/results/viai_aa_probe", commands["viai_aa_probe"])

    def test_checkpoint_inspection_detects_gan_and_probe_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth.tar"
            torch.save(
                {
                    "netD": {},
                    "enable_probe_loss": True,
                    "global_step": 12,
                    "global_epoch": 3,
                },
                path,
            )
            info = inspect_checkpoint(path, require_probe=True)
            self.assertTrue(info["use_gan"])
            self.assertTrue(info["probe_enabled"])
            self.assertEqual(info["global_step"], 12)

            torch.save({"enable_probe_loss": False}, path)
            with self.assertRaises(RuntimeError):
                inspect_checkpoint(path, require_probe=True)

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "baseline"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                ensure_empty_output_root(output)

    def test_run_metadata_uses_runner_command_git_and_split_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pth.tar"
            checkpoint.write_bytes(b"checkpoint")
            protocol = root / "protocol.json"
            protocol.write_text(
                json.dumps(
                    {
                        "splits": {
                            "viai_a": {
                                "train": {"sha256": "a-train"},
                                "val": {"sha256": "a-val"},
                                "test": {"sha256": "a-test"},
                            },
                            "viai_av": {
                                "train": {"sha256": "av-train"},
                                "val": {"sha256": "av-val"},
                                "test": {"sha256": "av-test"},
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            environment = {
                "VIAI_BASELINE_COMMAND_JSON": json.dumps(
                    ["python", "main.py", "test-viai-a"]
                ),
                "VIAI_BASELINE_GIT_JSON": json.dumps(
                    {"commit": "abc123", "dirty": False, "status": []}
                ),
            }
            with patch.dict(os.environ, environment, clear=False):
                metadata = build_run_metadata(
                    checkpoint,
                    {"batch_size": 16},
                    ["test_viai_a"],
                    1234,
                    "start",
                    root,
                    protocol_path=protocol,
                )

            self.assertEqual(
                metadata["command"],
                ["python", "main.py", "test-viai-a"],
            )
            self.assertEqual(metadata["git"]["commit"], "abc123")
            self.assertEqual(
                metadata["split_hashes"]["viai_av"]["test"],
                "av-test",
            )


if __name__ == "__main__":
    unittest.main()
