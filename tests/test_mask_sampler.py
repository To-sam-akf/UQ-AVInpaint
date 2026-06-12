import unittest

import numpy as np
import torch

from Data_loaders.mask_sampler import (
    MASK_TYPES,
    MaskSampler,
    build_boundary_map,
    build_missing_mask,
    corrupt_mel,
    stable_seed,
)
from utils.viai_a_metrics import compose_inpainted_mel


class MaskSamplerTests(unittest.TestCase):
    def setUp(self):
        self.sampler = MaskSampler()
        self.onsets = np.zeros(200, dtype=np.float32)
        self.onsets[100] = 10.0

    def test_all_mask_types_are_deterministic_and_valid(self):
        for mask_type in MASK_TYPES:
            seed = stable_seed("mask-test", mask_type)
            kwargs = {
                "onset_strengths": self.onsets
                if mask_type == "onset_centered"
                else None
            }
            first = self.sampler.sample(mask_type, seed, **kwargs)
            second = self.sampler.sample(mask_type, seed, **kwargs)
            self.assertEqual(first, second)
            self.assertEqual(first.end - first.start, first.gap_frames)
            self.assertGreaterEqual(first.start, 3)
            self.assertLessEqual(first.end, 197)

        onset_spec = self.sampler.sample(
            "onset_centered",
            stable_seed("onset"),
            onset_strengths=self.onsets,
        )
        self.assertEqual(onset_spec.start + onset_spec.gap_frames // 2, 100)
        empty_onset_spec = self.sampler.sample(
            "onset_centered",
            stable_seed("empty-onset"),
            onset_strengths=np.zeros(200, dtype=np.float32),
        )
        self.assertEqual(empty_onset_spec.start, 3)

        boundary_spec = self.sampler.sample(
            "boundary_near",
            stable_seed("boundary"),
        )
        self.assertTrue(boundary_spec.start == 3 or boundary_spec.end == 197)

        long_spec = self.sampler.sample("long_gap", stable_seed("long"))
        self.assertIn(long_spec.gap_frames, {60, 80, 100})

    def test_per_sample_seeds_produce_independent_spans(self):
        specs = [
            self.sampler.sample("random", stable_seed("sample", index))
            for index in range(12)
        ]
        spans = {(spec.start, spec.end) for spec in specs}
        self.assertGreater(len(spans), 1)

    def test_mask_boundary_map_and_corruption_follow_contract(self):
        spec = self.sampler.sample("random", stable_seed("contract"))
        mask = build_missing_mask(spec)
        boundary = build_boundary_map(spec)
        self.assertEqual(mask.shape, (1, 80, 200))
        self.assertEqual(boundary.shape, (2, 80, 200))
        self.assertTrue(torch.equal(mask[:, :, spec.start : spec.end], torch.ones(
            1, 80, spec.gap_frames
        )))
        self.assertEqual(float(mask[:, :, : spec.start].sum()), 0.0)
        self.assertEqual(float(mask[:, :, spec.end :].sum()), 0.0)
        self.assertEqual(float(boundary[0, 0, spec.start]), 0.0)
        self.assertEqual(float(boundary[1, 0, spec.end - 1]), 0.0)
        self.assertAlmostEqual(
            float(boundary[0, 0, spec.start + 1]),
            1.0 / 199.0,
        )

        mel = torch.arange(80 * 200, dtype=torch.float32).reshape(1, 80, 200)
        corrupted, actual_mask = corrupt_mel(mel, spec)
        self.assertTrue(torch.equal(mask, actual_mask))
        prediction = torch.full_like(corrupted, -5.0)
        completed = compose_inpainted_mel(
            corrupted.unsqueeze(0),
            prediction.unsqueeze(0),
            actual_mask.unsqueeze(0),
        )
        known = 1.0 - actual_mask.unsqueeze(0)
        known_error = torch.max(
            torch.abs(completed - corrupted.unsqueeze(0)) * known
        )
        self.assertEqual(float(known_error), 0.0)


if __name__ == "__main__":
    unittest.main()
