import unittest
from types import SimpleNamespace

import torch

from Data_loaders.mel_loader import (
    corrupt_mel_spectrogram,
    corrupt_mel_spectrogram_batch,
)
from utils.baseline_evaluation import (
    aggregate_sample_records,
    compute_sample_records,
    select_prediction,
)


class BaselineEvaluationTests(unittest.TestCase):
    def test_batch_corruption_uses_each_samples_span_and_interpolation(self):
        mel = torch.arange(2 * 2 * 12, dtype=torch.float32).reshape(2, 2, 12)
        specs = [
            {"start": 3, "end": 6, "gap_frames": 3},
            {"start": 5, "end": 9, "gap_frames": 4},
        ]

        corrupted, mask, spans = corrupt_mel_spectrogram_batch(
            mel,
            specs,
            min_margin=3,
        )

        self.assertEqual(spans, [(3, 6), (5, 9)])
        self.assertTrue(torch.equal(mask[0, :, :, 3:6], torch.ones(1, 2, 3)))
        self.assertEqual(float(mask[0, :, :, :3].sum()), 0.0)
        expected_first = torch.tensor([2.0, 4.0, 6.0])
        self.assertTrue(torch.allclose(corrupted[0, 0, 3:6], expected_first))
        self.assertTrue(torch.equal(corrupted[0, :, :3], mel[0, :, :3]))

    def test_existing_scalar_corruption_interface_remains_compatible(self):
        mel = torch.zeros(2, 4, 20)
        corrupted, mask, span = corrupt_mel_spectrogram(
            mel,
            blank_frames=5,
            start=7,
        )
        self.assertEqual(corrupted.shape, mel.shape)
        self.assertEqual(mask.shape, (2, 1, 4, 20))
        self.assertEqual(span, (7, 12))

    def test_compose_metrics_preserve_known_region_strictly(self):
        mel_target = torch.zeros(1, 1, 8, 12)
        mel_input = torch.full((1, 1, 8, 12), 0.25)
        mel_prediction = torch.full((1, 1, 8, 12), 0.75)
        mask = torch.zeros(1, 1, 8, 12)
        mask[:, :, :, 4:8] = 1.0
        spec = {
            "sample_id": "processed/piano/video/clip",
            "mask_type": "random",
            "start": 4,
            "end": 8,
            "gap_frames": 4,
            "seed": 1234,
        }

        records = compute_sample_records(
            [spec["sample_id"]],
            [spec],
            "viai_a",
            mel_input,
            mel_prediction,
            mel_target,
            mask,
        )

        self.assertEqual(records[0]["known_region_max_abs_error"], 0.0)
        self.assertAlmostEqual(records[0]["mel_l1_missing"], 0.75)

    def test_prediction_branch_selection(self):
        av = torch.tensor([1.0])
        probe = torch.tensor([2.0])
        model = SimpleNamespace(
            mel_pred=av,
            mel_probe_pred=probe,
            enable_probe_loss=True,
        )
        self.assertIs(select_prediction(model, "av"), av)
        self.assertIs(select_prediction(model, "probe"), probe)
        model.enable_probe_loss = False
        with self.assertRaises(RuntimeError):
            select_prediction(model, "probe")

    def test_per_sample_aggregation_is_independent_of_batch_partition(self):
        records = []
        for value in (1.0, 3.0, 8.0):
            records.append(
                {
                    "mel_l1_full": value,
                    "mel_l1_missing": value,
                    "psnr_full": value,
                    "psnr_missing": value,
                    "ssim": value,
                    "known_region_max_abs_error": value / 100.0,
                }
            )
        summary = aggregate_sample_records(records)
        self.assertAlmostEqual(summary["mel_l1_full"], 4.0)
        self.assertAlmostEqual(summary["known_region_max_abs_error"], 0.08)


if __name__ == "__main__":
    unittest.main()
