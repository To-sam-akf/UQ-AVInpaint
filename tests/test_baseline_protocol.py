import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.baseline_protocol import (
    audit_cross_model_assignments,
    audit_model_splits,
    canonical_sample_id,
    create_baseline_protocol,
    generate_mask_specs,
    load_mask_manifest,
    resolve_mask_specs,
    validate_av_test_row,
    validate_mask_spec,
    write_jsonl,
)


def split_row(sample_id):
    return {
        "sample_id": sample_id,
        "source_video": "/".join(sample_id.split("/")[1:3]),
    }


class BaselineProtocolTests(unittest.TestCase):
    def _write_split(self, root, name, sample_dir, mel_frames=200):
        line = (
            f"{sample_dir}|{sample_dir}/mel.npy|"
            f"{sample_dir}/raw_audio.npy|{mel_frames}\n"
        )
        (root / name).write_text(line, encoding="utf-8")

    def _create_complete_av_sample(self, root, sample_dir):
        sample_path = root / sample_dir
        for directory_name in ("image_crop", "flow_x_crop", "flow_y_crop"):
            directory = sample_path / directory_name
            directory.mkdir(parents=True, exist_ok=True)
            for index in range(1, 51):
                (directory / f"{index}.jpg").touch()
        np.save(sample_path / "mel.npy", np.zeros((200, 80), dtype=np.float32))
        np.save(
            sample_path / "raw_audio.npy",
            np.zeros(64000, dtype=np.float32),
        )

    def test_manifest_generation_is_deterministic_and_per_sample(self):
        rows = [
            {"sample_id": f"processed/piano/video_{index}/clip_{index:03d}"}
            for index in range(20)
        ]
        first = generate_mask_specs(rows, seed=1234)
        second = generate_mask_specs(list(reversed(rows)), seed=1234)

        self.assertEqual(first, second)
        spans = {(item["start"], item["gap_frames"]) for item in first}
        self.assertGreater(len(spans), 1)

    def test_invalid_masks_and_missing_samples_are_rejected(self):
        with self.assertRaises(ValueError):
            validate_mask_spec(
                {
                    "sample_id": "processed/piano/video/clip",
                    "mask_type": "random",
                    "start": 2,
                    "end": 22,
                    "gap_frames": 20,
                    "seed": 1234,
                }
            )
        with self.assertRaises(ValueError):
            validate_mask_spec(
                {
                    "sample_id": "processed/piano/video/clip",
                    "mask_type": "random",
                    "start": 10,
                    "end": 30,
                    "gap_frames": 19,
                    "seed": 1234,
                }
            )

        manifest = {
            "processed/piano/video/clip": {
                "sample_id": "processed/piano/video/clip",
                "mask_type": "random",
                "start": 10,
                "end": 30,
                "gap_frames": 20,
                "seed": 1234,
            }
        }
        with self.assertRaises(KeyError):
            resolve_mask_specs(
                ["processed/piano/video/missing/0"],
                manifest,
            )

    def test_manifest_loader_rejects_duplicate_sample_ids(self):
        spec = {
            "sample_id": "processed/piano/video/clip",
            "mask_type": "random",
            "start": 10,
            "end": 30,
            "gap_frames": 20,
            "seed": 1234,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "masks.jsonl"
            write_jsonl(path, [spec, spec])
            with self.assertRaises(ValueError):
                load_mask_manifest(path)

    def test_split_leakage_and_cross_model_conflicts_are_rejected(self):
        rows = {
            "train": [split_row("processed/piano/video_a/clip_1")],
            "val": [split_row("processed/piano/video_a/clip_2")],
            "test": [split_row("processed/piano/video_b/clip_1")],
        }
        with self.assertRaises(ValueError):
            audit_model_splits(rows, "viai_av")

        with self.assertRaises(ValueError):
            audit_cross_model_assignments(
                {"piano/video_a": "train"},
                {"piano/video_a": "test"},
            )

    def test_absolute_av_runtime_path_restores_canonical_sample_id(self):
        data_root = "/dataset"
        runtime_path = (
            "/dataset/processed/piano/youtube_id/clip_000123/0"
        )
        sample_id = "processed/piano/youtube_id/clip_000123"
        manifest = {
            sample_id: {
                "sample_id": sample_id,
                "mask_type": "random",
                "start": 64,
                "end": 101,
                "gap_frames": 37,
                "seed": 1234,
            }
        }

        self.assertEqual(
            canonical_sample_id(
                runtime_path,
                data_root=data_root,
                strip_window_index=True,
            ),
            sample_id,
        )
        self.assertEqual(
            resolve_mask_specs([runtime_path], manifest, data_root=data_root),
            [manifest[sample_id]],
        )

    def test_protocol_generation_is_byte_reproducible_and_freezes_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data"
            root.mkdir()
            phases = {
                "train": "train_video",
                "val": "val_video",
                "test": "test_video",
            }
            for phase, video in phases.items():
                self._write_split(
                    root,
                    f"{phase}_viai_a_split.txt",
                    f"processed_viai_a/piano/{video}",
                )
                self._write_split(
                    root,
                    f"{phase}_av_split.txt",
                    f"processed/piano/{video}/clip_000001",
                )
            self._create_complete_av_sample(
                root,
                "processed/piano/test_video/clip_000001",
            )

            first_path, first = create_baseline_protocol(
                root,
                Path(directory) / "first",
            )
            second_path, second = create_baseline_protocol(
                root,
                Path(directory) / "second",
            )

            first_mask = Path(first["mask_manifest"]["path"]).read_bytes()
            second_mask = Path(second["mask_manifest"]["path"]).read_bytes()
            self.assertEqual(first_mask, second_mask)
            self.assertEqual(
                first["mask_manifest"]["sha256"],
                second["mask_manifest"]["sha256"],
            )
            self.assertTrue(
                (
                    first_path.parent
                    / "splits"
                    / "test_av_split.txt"
                ).is_file()
            )

    def test_incomplete_av_test_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = "processed/piano/video/clip_000001"
            self._create_complete_av_sample(root, sample_dir)
            row = {
                "sample_id": sample_dir,
                "sample_dir": sample_dir,
                "mel_path": f"{sample_dir}/mel.npy",
                "audio_path": f"{sample_dir}/raw_audio.npy",
                "mel_frames": 200,
            }
            (root / sample_dir / "image_crop" / "50.jpg").unlink()
            with self.assertRaises(ValueError):
                validate_av_test_row(row, root)


if __name__ == "__main__":
    unittest.main()
