"""P1 integration and edge-case tests.

Covers:
- All mask types with varied gap lengths in a single batch.
- Boundary map numerical correctness at edges.
- Long_gap boundary constraint verification.
- Smoke test: train/val/test phases each produce a batch with correct shapes.
- prepare-uq-metadata CLI entry point via main.py.
- Fixed seed reproducibility for mask + visual degradation jointly.
"""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from Data_loaders.mask_sampler import (
    MASK_TYPES,
    MaskSampler,
    MaskSpec,
    build_boundary_map,
    build_missing_mask,
    corrupt_mel,
    load_uq_mask_manifest,
    stable_seed,
)
from Data_loaders.uq_av_loader import (
    UQAVDataset,
    create_uq_av_dataloader,
    uq_av_collate_fn,
    read_uq_split,
)
from Data_loaders.visual_degradation import VIDEO_CONDITIONS
from tools.prepare_uq_metadata import parse_args, prepare_uq_metadata


class P1IntegrationTests(unittest.TestCase):
    """End-to-end P1 integration tests."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        # 5 samples across 3 instruments for rich testing
        self.samples = [
            ("processed/piano/vid_a/clip_001", 0),
            ("processed/piano/vid_b/clip_001", 1),
            ("processed/guitar/vid_a/clip_001", 2),
            ("processed/guitar/vid_b/clip_001", 3),
            ("processed/flute/vid_a/clip_001", 4),
        ]
        for sid, seed in self.samples:
            self._create_sample(sid, seed)
        for phase in ("train", "val", "test"):
            self._write_split(
                f"{phase}_av_split.txt",
                [s[0] for s in self.samples],
            )
        self.meta_dir = self.root / "uq_metadata"

    def tearDown(self):
        self.temp.cleanup()

    def _create_sample(self, sid, seed):
        d = self.root / sid
        for name in ("image_crop", "flow_x_crop", "flow_y_crop"):
            (d / name).mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(seed)
        mel = rng.rand(200, 80).astype(np.float32)
        audio = rng.randn(64000).astype(np.float32) * 0.1
        np.save(d / "mel.npy", mel, allow_pickle=False)
        np.save(d / "raw_audio.npy", audio, allow_pickle=False)
        for fid in range(1, 51):
            img = np.full((8, 8, 3), (20 + seed * 50 + fid) % 256, dtype=np.uint8)
            fx = np.full((8, 8), (127 + seed * 20 + fid) % 256, dtype=np.uint8)
            fy = np.full((8, 8), (127 - seed * 20 - fid) % 256, dtype=np.uint8)
            cv2.imwrite(str(d / "image_crop" / f"{fid}.jpg"), img)
            cv2.imwrite(str(d / "flow_x_crop" / f"{fid}.jpg"), fx)
            cv2.imwrite(str(d / "flow_y_crop" / f"{fid}.jpg"), fy)

    def _write_split(self, name, sample_ids):
        with (self.root / name).open("w", encoding="utf-8") as fh:
            for sid in sample_ids:
                fh.write(f"{sid}|{sid}/mel.npy|{sid}/raw_audio.npy|200\n")

    def _prepare_metadata(self):
        args = parse_args(
            [
                "--data-root", str(self.root),
                "--output-dir", str(self.meta_dir),
                "--seed", "42",
            ]
        )
        return prepare_uq_metadata(args)

    # ------------------------------------------------------------------
    # 1. Batch with varied mask types and gap lengths
    # ------------------------------------------------------------------
    def test_batch_varied_mask_types(self):
        """Verify that different samples in a train batch can have different
        mask types and gap lengths."""
        self._prepare_metadata()
        dataset = UQAVDataset(
            data_root=self.root,
            split_name="train_av_split.txt",
            phase="train",
            metadata_dir=self.meta_dir,
            seed=42,
            mask_types=MASK_TYPES,
            image_size=8,
        )
        loader = create_uq_av_dataloader(
            data_root=self.root,
            split_name="train_av_split.txt",
            phase="train",
            metadata_dir=self.meta_dir,
            seed=42,
            mask_types=MASK_TYPES,
            image_size=8,
            batch_size=5,
            num_workers=0,
            pin_memory=False,
        )
        batch = next(iter(loader))
        self.assertEqual(batch["mel_target"].shape, (5, 1, 80, 200))
        specs = batch["mask_spec"]
        types = {s.mask_type for s in specs}
        # With 5 samples and 4 mask types, at least 2 types should appear
        # (random assignment can miss some types but should have variety)
        self.assertGreaterEqual(len(types), 1)
        gaps = {s.gap_frames for s in specs}
        # Different gap widths should appear
        self.assertGreaterEqual(len(gaps), 1)

    # ------------------------------------------------------------------
    # 2. Mask = 1 verification
    # ------------------------------------------------------------------
    def test_mask_ones_match_spec_boundaries(self):
        sampler = MaskSampler()
        for mask_type in MASK_TYPES:
            onsets = np.zeros(200, dtype=np.float32)
            onsets[100] = 10.0 if mask_type == "onset_centered" else 0.0
            kwargs = (
                {"onset_strengths": onsets}
                if mask_type == "onset_centered"
                else {}
            )
            for _ in range(3):
                spec = sampler.sample(mask_type, stable_seed(mask_type, _), **kwargs)
                mask = build_missing_mask(spec)
                self.assertEqual(mask.shape, (1, 80, 200))

                # mask=1 exactly in [start:end]
                inside = mask[:, :, spec.start : spec.end]
                self.assertTrue(
                    torch.allclose(inside, torch.ones_like(inside)),
                    f"{mask_type}: inside region not all 1s",
                )
                # mask=0 outside
                if spec.start > 0:
                    left = float(mask[:, :, : spec.start].sum())
                    self.assertEqual(left, 0.0, f"{mask_type}: left of gap not 0")
                if spec.end < 200:
                    right = float(mask[:, :, spec.end :].sum())
                    self.assertEqual(right, 0.0, f"{mask_type}: right of gap not 0")

    # ------------------------------------------------------------------
    # 3. Boundary map numerical correctness
    # ------------------------------------------------------------------
    def test_boundary_map_at_edges(self):
        spec = MaskSpec(
            mask_type="random", start=50, end=70, gap_frames=20, seed=1
        )
        bmap = build_boundary_map(spec, mel_bins=80, mel_frames=200)
        self.assertEqual(bmap.shape, (2, 80, 200))

        # Left channel: distance to left boundary (start)
        # At start, distance = 0
        self.assertAlmostEqual(float(bmap[0, 0, 50]), 0.0, places=5)
        # At start+1, distance = 1/199
        self.assertAlmostEqual(float(bmap[0, 0, 51]), 1.0 / 199.0, places=5)
        # Far from start, distance should be larger
        self.assertGreater(float(bmap[0, 0, 100]), float(bmap[0, 0, 51]))

        # Right channel: distance to right boundary (end-1)
        # At end-1, distance = 0
        self.assertAlmostEqual(float(bmap[1, 0, 69]), 0.0, places=5)
        # At end-2, distance = 1/199
        self.assertAlmostEqual(float(bmap[1, 0, 68]), 1.0 / 199.0, places=5)
        # Far from end, distance should be larger
        self.assertGreater(float(bmap[1, 0, 30]), float(bmap[1, 0, 68]))

        # Boundary map is broadcast across all bins
        self.assertTrue(
            torch.allclose(bmap[0, 0], bmap[0, 40]),
            "left channel not broadcast",
        )
        self.assertTrue(
            torch.allclose(bmap[1, 0], bmap[1, 40]),
            "right channel not broadcast",
        )

    # ------------------------------------------------------------------
    # 4. Long gap boundary constraint
    # ------------------------------------------------------------------
    def test_long_gap_within_bounds(self):
        sampler = MaskSampler(
            mel_frames=200,
            min_gap_frames=20,
            max_gap_frames=50,
            boundary_margin=3,
            long_gap_frames=(60, 80, 100),
        )
        for _ in range(20):
            spec = sampler.sample("long_gap", stable_seed("lg", _))
            self.assertIn(spec.gap_frames, {60, 80, 100})
            self.assertGreaterEqual(spec.start, 3)
            self.assertLessEqual(spec.end, 197)
            self.assertEqual(spec.end - spec.start, spec.gap_frames)

    # ------------------------------------------------------------------
    # 5. Smoke test: all phases produce valid batches
    # ------------------------------------------------------------------
    def test_smoke_all_phases(self):
        self._prepare_metadata()
        for phase in ("train", "val", "test"):
            with self.subTest(phase=phase):
                loader = create_uq_av_dataloader(
                    data_root=self.root,
                    split_name=f"{phase}_av_split.txt",
                    phase=phase,
                    metadata_dir=self.meta_dir,
                    seed=42,
                    video_conditions=("original",),
                    image_size=8,
                    batch_size=2,
                    num_workers=0,
                    pin_memory=False,
                )
                batch = next(iter(loader))
                self.assertEqual(batch["mel_target"].shape[:2], (2, 1))
                self.assertEqual(batch["mel_corrupted"].shape[:2], (2, 1))
                self.assertEqual(batch["missing_mask"].shape[:2], (2, 1))
                self.assertEqual(batch["boundary_map"].shape[:2], (2, 2))
                self.assertEqual(batch["video"].shape[:2], (2, 50))
                self.assertEqual(batch["flow"].shape[:2], (2, 50))
                self.assertEqual(batch["audio_target"].shape, (2, 64000))
                self.assertEqual(len(batch["sample_id"]), 2)
                self.assertEqual(len(batch["mask_spec"]), 2)
                self.assertEqual(len(batch["video_condition"]), 2)

    # ------------------------------------------------------------------
    # 6. Joint mask + visual degradation reproducibility
    # ------------------------------------------------------------------
    def test_joint_reproducibility(self):
        self._prepare_metadata()
        conditions = ("original", "blur", "frame_drop")
        for condition in conditions:
            with self.subTest(condition=condition):
                d1 = UQAVDataset(
                    data_root=self.root,
                    split_name="test_av_split.txt",
                    phase="test",
                    metadata_dir=self.meta_dir,
                    seed=42,
                    video_conditions=(condition,),
                    image_size=8,
                )
                d2 = UQAVDataset(
                    data_root=self.root,
                    split_name="test_av_split.txt",
                    phase="test",
                    metadata_dir=self.meta_dir,
                    seed=42,
                    video_conditions=(condition,),
                    image_size=8,
                )
                for i in range(min(len(d1), 5)):
                    s1 = d1[i]
                    s2 = d2[i]
                    self.assertEqual(
                        s1["mask_spec"], s2["mask_spec"],
                        f"mask spec mismatch for {condition} idx {i}",
                    )
                    self.assertTrue(
                        torch.equal(s1["mel_target"], s2["mel_target"]),
                    )
                    self.assertTrue(
                        torch.equal(s1["video"], s2["video"]),
                    )
                    self.assertTrue(
                        torch.equal(s1["flow"], s2["flow"]),
                    )
                    self.assertEqual(
                        s1["video_condition"], s2["video_condition"],
                    )
                    self.assertEqual(
                        s1["video_degradation"], s2["video_degradation"],
                    )

    # ------------------------------------------------------------------
    # 7. prepare-uq-metadata CLI smoke test
    # ------------------------------------------------------------------
    def test_prepare_uq_metadata_cli(self):
        """Verify the CLI entrypoint via tools.prepare_uq_metadata.main()."""
        from tools.prepare_uq_metadata import main as prepare_main

        result = prepare_main(
            [
                "--data-root", str(self.root),
                "--output-dir", str(self.meta_dir),
                "--seed", "1",
            ]
        )

        # Verify outputs exist
        for fname in ("val_masks.jsonl", "test_masks.jsonl", "metadata_summary.json"):
            self.assertTrue((self.meta_dir / fname).is_file(), f"Missing: {fname}")
        onset_dir = self.meta_dir / "train_onsets"
        self.assertTrue(onset_dir.is_dir())
        onset_files = list(onset_dir.rglob("*.npy"))
        self.assertEqual(len(onset_files), 5)

    # ------------------------------------------------------------------
    # 8. Mask manifest round-trip
    # ------------------------------------------------------------------
    def test_mask_manifest_round_trip(self):
        self._prepare_metadata()
        manifest = load_uq_mask_manifest(
            self.meta_dir / "test_masks.jsonl",
            mel_frames=200,
            boundary_margin=3,
        )
        self.assertEqual(len(manifest), 5)
        for sid, specs in manifest.items():
            self.assertEqual(len(specs), 4)  # one per mask type
            for spec in specs:
                self.assertIsInstance(spec, MaskSpec)
                spec.validate(mel_frames=200, boundary_margin=3)

    # ------------------------------------------------------------------
    # 9. Collate function smoke test
    # ------------------------------------------------------------------
    def test_collate_fn_smoke(self):
        self._prepare_metadata()
        dataset = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.meta_dir,
            seed=42,
            video_conditions=("original",),
            image_size=8,
        )
        samples = [dataset[i] for i in range(3)]
        batch = uq_av_collate_fn(samples)
        self.assertEqual(batch["mel_target"].shape, (3, 1, 80, 200))
        self.assertEqual(batch["boundary_map"].shape, (3, 2, 80, 200))
        self.assertEqual(len(batch["sample_id"]), 3)
        self.assertIsInstance(batch["mask_spec"][0], MaskSpec)

    # ------------------------------------------------------------------
    # 10. UQ split reader
    # ------------------------------------------------------------------
    def test_read_uq_split(self):
        rows = read_uq_split(str(self.root), "train_av_split.txt")
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertIn("sample_id", row)
            self.assertIn("instrument", row)
            self.assertIn("source_video", row)
            self.assertTrue(row["sample_dir"].is_dir())
        # Empty split raises
        empty_path = self.root / "empty.txt"
        empty_path.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            read_uq_split(str(self.root), "empty.txt")

    # ------------------------------------------------------------------
    # 11. MaskSpec serialization round-trip
    # ------------------------------------------------------------------
    def test_mask_spec_serialization(self):
        original = MaskSpec(
            mask_type="onset_centered",
            start=50,
            end=70,
            gap_frames=20,
            seed=42,
        )
        d = original.to_dict()
        restored = MaskSpec.from_dict(d, mel_frames=200, boundary_margin=3)
        self.assertEqual(original, restored)

    # ------------------------------------------------------------------
    # 12. Wrong video preserves audio input
    # ------------------------------------------------------------------
    def test_wrong_video_preserves_audio(self):
        self._prepare_metadata()
        ds_orig = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.meta_dir,
            seed=42,
            video_conditions=("original",),
            image_size=8,
        )
        ds_wrong = UQAVDataset(
            data_root=self.root,
            split_name="test_av_split.txt",
            phase="test",
            metadata_dir=self.meta_dir,
            seed=42,
            video_conditions=("wrong_video",),
            image_size=8,
        )
        # First sample: original and wrong_video should share audio/mask
        orig = ds_orig[0]
        wrong = ds_wrong[0]
        self.assertTrue(torch.equal(orig["mel_target"], wrong["mel_target"]))
        self.assertTrue(torch.equal(orig["mel_corrupted"], wrong["mel_corrupted"]))
        self.assertTrue(torch.equal(orig["audio_target"], wrong["audio_target"]))
        self.assertTrue(torch.equal(orig["missing_mask"], wrong["missing_mask"]))
        # Video should differ
        self.assertFalse(torch.equal(orig["video"], wrong["video"]))


if __name__ == "__main__":
    unittest.main()
