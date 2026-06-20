import tempfile
import unittest
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from Data_loaders.mask_sampler import MASK_TYPES
from Data_loaders.uq_av_loader import (
    UQAVDataset,
    create_uq_av_dataloader,
    _normalise_condition_probabilities,
    _sample_conditioning_mode,
)
from Data_loaders.visual_degradation import VIDEO_CONDITIONS
from main import MODULE_MAP
from tools.prepare_uq_metadata import parse_args, prepare_uq_metadata


class UQAVDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.sample_ids = [
            "processed/piano/video_a/clip_000001",
            "processed/piano/video_b/clip_000001",
        ]
        for sample_index, sample_id in enumerate(self.sample_ids):
            self._create_sample(sample_id, sample_index)
        for phase in ("train", "val", "test"):
            self._write_split(f"{phase}_av_split.txt", self.sample_ids)
        self.metadata_one = self.root / "metadata_one"
        self.metadata_two = self.root / "metadata_two"

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _create_sample(self, sample_id, sample_index):
        sample_dir = self.root / sample_id
        for name in ("image_crop", "flow_x_crop", "flow_y_crop"):
            (sample_dir / name).mkdir(parents=True, exist_ok=True)

        time = np.arange(200, dtype=np.float32)[:, None]
        frequency = np.arange(80, dtype=np.float32)[None, :]
        mel = ((time + frequency + sample_index * 7) % 200) / 199.0
        audio = np.linspace(
            -0.5 + sample_index * 0.1,
            0.5 + sample_index * 0.1,
            64000,
            dtype=np.float32,
        )
        np.save(sample_dir / "mel.npy", mel.astype(np.float32), allow_pickle=False)
        np.save(sample_dir / "raw_audio.npy", audio, allow_pickle=False)

        for frame_id in range(1, 51):
            image_value = min(255, 20 + sample_index * 100 + frame_id)
            image = np.full((8, 8, 3), image_value, dtype=np.uint8)
            flow_x = np.full(
                (8, 8),
                min(255, 127 + sample_index * 20 + frame_id % 10),
                dtype=np.uint8,
            )
            flow_y = np.full(
                (8, 8),
                max(0, 127 - sample_index * 20 - frame_id % 10),
                dtype=np.uint8,
            )
            cv2.imwrite(str(sample_dir / "image_crop" / f"{frame_id}.jpg"), image)
            cv2.imwrite(str(sample_dir / "flow_x_crop" / f"{frame_id}.jpg"), flow_x)
            cv2.imwrite(str(sample_dir / "flow_y_crop" / f"{frame_id}.jpg"), flow_y)

    def _write_split(self, name, sample_ids):
        with (self.root / name).open("w", encoding="utf-8") as handle:
            for sample_id in sample_ids:
                handle.write(
                    f"{sample_id}|{sample_id}/mel.npy|"
                    f"{sample_id}/raw_audio.npy|200\n"
                )

    def _prepare_metadata(self, output_dir):
        args = parse_args(
            [
                "--data-root",
                str(self.root),
                "--output-dir",
                str(output_dir),
                "--seed",
                "77",
            ]
        )
        return prepare_uq_metadata(args)

    def _dataset(self, condition="original", phase="test"):
        return UQAVDataset(
            data_root=self.root,
            split_name=f"{phase}_av_split.txt",
            phase=phase,
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=(condition,),
            image_size=8,
        )

    def test_metadata_is_byte_reproducible_and_complete(self):
        first = self._prepare_metadata(self.metadata_one)
        second = self._prepare_metadata(self.metadata_two)

        for filename in (
            "val_masks.jsonl",
            "test_masks.jsonl",
            "metadata_summary.json",
        ):
            self.assertEqual(
                (self.metadata_one / filename).read_bytes(),
                (self.metadata_two / filename).read_bytes(),
            )
        for sample_id in self.sample_ids:
            relative = Path(sample_id).with_suffix(".npy")
            self.assertEqual(
                (self.metadata_one / "train_onsets" / relative).read_bytes(),
                (self.metadata_two / "train_onsets" / relative).read_bytes(),
            )

        summary = first["summary"]
        self.assertEqual(summary["version"], 1)
        self.assertEqual(summary["splits"]["train"]["num_samples"], 2)
        self.assertEqual(summary["manifests"]["test"]["num_variants"], 8)
        self.assertEqual(
            set(summary["manifests"]["test"]["mask_type_counts"]),
            set(MASK_TYPES),
        )
        self.assertEqual(MODULE_MAP["prepare-uq-metadata"], "tools.prepare_uq_metadata")
        self.assertEqual(first["summary"], second["summary"])

    def test_loader_contract_manifest_expansion_and_collate(self):
        self._prepare_metadata(self.metadata_one)
        dataset = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("original", "blur"),
            image_size=8,
        )
        self.assertEqual(len(dataset), 2 * 4 * 2)
        sample = dataset[0]
        self.assertEqual(sample["mel_target"].shape, (1, 80, 200))
        self.assertEqual(sample["mel_corrupted"].shape, (1, 80, 200))
        self.assertEqual(sample["missing_mask"].shape, (1, 80, 200))
        self.assertEqual(sample["boundary_map"].shape, (2, 80, 200))
        self.assertEqual(sample["video"].shape, (50, 3, 8, 8))
        self.assertEqual(sample["flow"].shape, (50, 2, 8, 8))
        self.assertEqual(sample["audio_target"].shape, (64000,))

        loader = create_uq_av_dataloader(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("original",),
            image_size=8,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
        )
        batch = next(iter(loader))
        self.assertEqual(batch["mel_target"].shape, (2, 1, 80, 200))
        self.assertEqual(batch["video"].shape, (2, 50, 3, 8, 8))
        self.assertEqual(len(batch["sample_id"]), 2)
        self.assertEqual(len(batch["mask_spec"]), 2)

    def test_training_masks_are_per_sample_and_epoch_reproducible(self):
        self._prepare_metadata(self.metadata_one)
        first = UQAVDataset(
            data_root=self.root,
            split_name="train_av_split.txt",
            phase="train",
            metadata_dir=self.metadata_one,
            seed=77,
            image_size=8,
        )
        second = UQAVDataset(
            data_root=self.root,
            split_name="train_av_split.txt",
            phase="train",
            metadata_dir=self.metadata_one,
            seed=77,
            image_size=8,
        )
        first_specs = [first[index]["mask_spec"] for index in range(len(first))]
        second_specs = [second[index]["mask_spec"] for index in range(len(second))]
        self.assertEqual(first_specs, second_specs)
        self.assertNotEqual(first_specs[0], first_specs[1])

        first.set_epoch(1)
        second.set_epoch(1)
        self.assertEqual(first[0]["mask_spec"], second[0]["mask_spec"])
        self.assertNotEqual(first_specs[0], first[0]["mask_spec"])

    def test_modality_dropout_condition_sampling_is_deterministic(self):
        probabilities = _normalise_condition_probabilities({
            "audio_video": 0.4,
            "drop_video": 0.2,
            "partial_audio_video": 0.2,
            "wrong_video": 0.1,
            "shuffled_video": 0.1,
        })
        rng_a = random.Random(123)
        rng_b = random.Random(123)
        modes_a = [
            _sample_conditioning_mode(rng_a, probabilities)
            for _ in range(10000)
        ]
        modes_b = [
            _sample_conditioning_mode(rng_b, probabilities)
            for _ in range(10000)
        ]
        self.assertEqual(modes_a, modes_b)
        counts = {mode: modes_a.count(mode) / len(modes_a) for mode in probabilities}
        self.assertAlmostEqual(counts["audio_video"], 0.4, delta=0.03)
        self.assertAlmostEqual(counts["drop_video"], 0.2, delta=0.03)
        self.assertAlmostEqual(counts["partial_audio_video"], 0.2, delta=0.03)
        self.assertAlmostEqual(counts["wrong_video"], 0.1, delta=0.03)
        self.assertAlmostEqual(counts["shuffled_video"], 0.1, delta=0.03)

    def test_partial_audio_changes_only_condition_context(self):
        self._prepare_metadata(self.metadata_one)
        base = UQAVDataset(
            data_root=self.root,
            split_name="train_av_split.txt",
            phase="train",
            metadata_dir=self.metadata_one,
            seed=77,
            image_size=8,
            enable_modality_dropout=False,
        )[0]
        partial = UQAVDataset(
            data_root=self.root,
            split_name="train_av_split.txt",
            phase="train",
            metadata_dir=self.metadata_one,
            seed=77,
            image_size=8,
            enable_modality_dropout=True,
            condition_probabilities={
                "audio_video": 0.0,
                "drop_video": 0.0,
                "partial_audio_video": 1.0,
                "wrong_video": 0.0,
                "shuffled_video": 0.0,
            },
        )[0]

        self.assertEqual(partial["conditioning_mode"], "partial_audio_video")
        self.assertTrue(torch.equal(base["mel_target"], partial["mel_target"]))
        self.assertTrue(torch.equal(base["missing_mask"], partial["missing_mask"]))
        self.assertTrue(torch.equal(base["boundary_map"], partial["boundary_map"]))
        self.assertTrue(torch.equal(base["video"], partial["video"]))
        self.assertFalse(
            torch.equal(base["mel_corrupted"], partial["mel_corrupted"])
        )
        missing = partial["missing_mask"].bool()
        self.assertTrue(torch.equal(
            base["mel_corrupted"][missing],
            partial["mel_corrupted"][missing],
        ))

    def test_eval_condition_override_drop_audio_is_reproducible(self):
        self._prepare_metadata(self.metadata_one)
        base = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("original",),
            image_size=8,
        )[0]
        first = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("original",),
            image_size=8,
            condition_override="drop_audio",
        )[0]
        second = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("original",),
            image_size=8,
            condition_override="drop_audio",
        )[0]

        self.assertEqual(first["conditioning_mode"], "drop_audio")
        self.assertTrue(torch.equal(first["mel_corrupted"], second["mel_corrupted"]))
        self.assertTrue(torch.equal(first["mel_target"], base["mel_target"]))
        self.assertTrue(torch.equal(first["missing_mask"], base["missing_mask"]))
        self.assertFalse(torch.equal(first["mel_corrupted"], base["mel_corrupted"]))

    def test_visual_degradations_are_reproducible_and_preserve_audio(self):
        self._prepare_metadata(self.metadata_one)
        original = self._dataset("original")[0]
        for condition in VIDEO_CONDITIONS:
            first = self._dataset(condition)[0]
            second = self._dataset(condition)[0]
            self.assertTrue(torch.equal(first["video"], second["video"]))
            self.assertTrue(torch.equal(first["flow"], second["flow"]))
            self.assertEqual(
                first["video_degradation"],
                second["video_degradation"],
            )
            self.assertTrue(torch.equal(first["mel_target"], original["mel_target"]))
            self.assertTrue(
                torch.equal(first["mel_corrupted"], original["mel_corrupted"])
            )
            self.assertTrue(
                torch.equal(first["audio_target"], original["audio_target"])
            )
            self.assertTrue(torch.equal(first["missing_mask"], original["missing_mask"]))
            self.assertTrue(torch.equal(first["boundary_map"], original["boundary_map"]))

        shifted = self._dataset("temporal_shift")[0]
        self.assertIn(
            "temporal_shift_frames",
            shifted["video_degradation"],
        )
        wrong = self._dataset("wrong_video")[0]
        self.assertEqual(
            wrong["video_degradation"]["wrong_video_sample_id"],
            self.sample_ids[1],
        )
        self.assertFalse(torch.equal(wrong["video"], original["video"]))
        no_video = self._dataset("no_video")[0]
        self.assertEqual(float(no_video["video"].abs().sum()), 0.0)
        self.assertEqual(float(no_video["flow"].abs().sum()), 0.0)
        shuffled = self._dataset("shuffled_video")[0]
        self.assertIn(
            "frame_permutation",
            shuffled["video_degradation"],
        )
        self.assertFalse(torch.equal(shuffled["video"], original["video"]))
        self.assertEqual(
            sorted(shuffled["video_degradation"]["frame_permutation"]),
            list(range(50)),
        )

    def test_return_original_video_preserves_positive_condition(self):
        self._prepare_metadata(self.metadata_one)
        original = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("original",),
            image_size=8,
            return_original_video=True,
        )[0]
        wrong = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("wrong_video",),
            image_size=8,
            return_original_video=True,
        )[0]
        shuffled = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("shuffled_video",),
            image_size=8,
            return_original_video=True,
        )[0]

        for sample in (wrong, shuffled):
            self.assertTrue(torch.equal(sample["mel_target"], original["mel_target"]))
            self.assertTrue(
                torch.equal(sample["mel_corrupted"], original["mel_corrupted"])
            )
            self.assertTrue(
                torch.equal(sample["missing_mask"], original["missing_mask"])
            )
            self.assertTrue(
                torch.equal(sample["boundary_map"], original["boundary_map"])
            )
            self.assertTrue(
                torch.equal(sample["audio_target"], original["audio_target"])
            )
            self.assertTrue(
                torch.equal(sample["video_original"], original["video"])
            )
            self.assertTrue(torch.equal(sample["flow_original"], original["flow"]))
            self.assertFalse(torch.equal(sample["video"], original["video"]))

        loader = create_uq_av_dataloader(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.metadata_one,
            seed=77,
            video_conditions=("wrong_video",),
            image_size=8,
            batch_size=2,
            num_workers=0,
            pin_memory=False,
            return_original_video=True,
        )
        batch = next(iter(loader))
        self.assertEqual(batch["video_original"].shape, (2, 50, 3, 8, 8))
        self.assertEqual(batch["flow_original"].shape, (2, 50, 2, 8, 8))


if __name__ == "__main__":
    unittest.main()
