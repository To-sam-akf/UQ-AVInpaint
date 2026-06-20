"""Standalone tests for the visual_degradation module."""

import unittest

import numpy as np

from Data_loaders.visual_degradation import (
    VIDEO_CONDITIONS,
    apply_visual_degradation,
    _validate_inputs,
    _gaussian_blur,
    _shift_with_edge_padding,
)


class VisualDegradationTests(unittest.TestCase):
    def setUp(self):
        self.frames = 50
        self.height = 16
        self.width = 16
        rng = np.random.RandomState(42)
        self.video = rng.randn(self.frames, 3, self.height, self.width).astype(np.float32)
        self.flow = rng.randn(self.frames, 2, self.height, self.width).astype(np.float32)

    def test_input_validation_rejects_mismatched_shapes(self):
        # Different number of frames
        bad_flow = np.zeros((self.frames + 1, 2, self.height, self.width), dtype=np.float32)
        with self.assertRaises(ValueError):
            _validate_inputs(self.video, bad_flow)
        # Different spatial dimensions
        bad_flow2 = np.zeros((self.frames, 2, self.height + 1, self.width), dtype=np.float32)
        with self.assertRaises(ValueError):
            _validate_inputs(self.video, bad_flow2)

    def test_original_preserves_input(self):
        video_out, flow_out, params = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "original", seed=42
        )
        self.assertTrue(np.allclose(video_out, self.video))
        self.assertTrue(np.allclose(flow_out, self.flow))
        self.assertEqual(params["condition"], "original")
        self.assertEqual(params["seed"], 42)

    def test_original_is_deterministic(self):
        first = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "original", seed=77
        )
        second = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "original", seed=77
        )
        self.assertTrue(np.allclose(first[0], second[0]))
        self.assertTrue(np.allclose(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_no_video_zeros_everything(self):
        video_out, flow_out, params = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "no_video", seed=42
        )
        self.assertEqual(float(np.abs(video_out).sum()), 0.0)
        self.assertEqual(float(np.abs(flow_out).sum()), 0.0)
        self.assertEqual(params["condition"], "no_video")

    def test_shuffled_video_uses_same_frame_permutation(self):
        video_out, flow_out, params = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "shuffled_video", seed=42
        )
        permutation = params["frame_permutation"]
        self.assertEqual(sorted(permutation), list(range(self.frames)))
        self.assertNotEqual(permutation, list(range(self.frames)))
        self.assertTrue(np.allclose(video_out, self.video[permutation]))
        self.assertTrue(np.allclose(flow_out, self.flow[permutation]))
        self.assertEqual(params["condition"], "shuffled_video")

    def test_shuffled_video_is_deterministic(self):
        first = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "shuffled_video", seed=123
        )
        second = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "shuffled_video", seed=123
        )
        self.assertTrue(np.allclose(first[0], second[0]))
        self.assertTrue(np.allclose(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_blur_records_sigma(self):
        for iteration in range(10):
            video_out, flow_out, params = apply_visual_degradation(
                self.video.copy(), self.flow.copy(), "blur", seed=42 + iteration
            )
            self.assertIn("blur_sigma", params)
            sigma = float(params["blur_sigma"])
            self.assertGreaterEqual(sigma, 0.8)
            self.assertLessEqual(sigma, 3.0)

    def test_blur_is_deterministic(self):
        first = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "blur", seed=123
        )
        second = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "blur", seed=123
        )
        self.assertTrue(np.allclose(first[0], second[0]))
        self.assertTrue(np.allclose(first[1], second[1]))
        self.assertEqual(first[2], second[2])

    def test_occlusion_records_box(self):
        rng = np.random.RandomState(99)
        video_in = rng.randn(10, 3, 64, 64).astype(np.float32)
        flow_in = rng.randn(10, 2, 64, 64).astype(np.float32)
        video_out, flow_out, params = apply_visual_degradation(
            video_in.copy(), flow_in.copy(), "occlusion", seed=42
        )
        self.assertIn("occlusion_box", params)
        x0, y0, x1, y1 = params["occlusion_box"]
        self.assertGreater(x1, x0)
        self.assertGreater(y1, y0)
        # Verify the occluded region is zeroed
        occluded_video = video_out[:, :, y0:y1, x0:x1]
        self.assertEqual(float(np.abs(occluded_video).sum()), 0.0)
        occluded_flow = flow_out[:, :, y0:y1, x0:x1]
        self.assertEqual(float(np.abs(occluded_flow).sum()), 0.0)

    def test_frame_drop_records_ratio(self):
        video_out, flow_out, params = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "frame_drop", seed=42
        )
        self.assertIn("frame_keep_ratio", params)
        ratio = float(params["frame_keep_ratio"])
        self.assertGreaterEqual(ratio, 0.50)
        self.assertLessEqual(ratio, 0.90)
        self.assertIn("dropped_frames", params)
        # Verify dropped frames are zeroed
        for idx in params["dropped_frames"]:
            self.assertEqual(float(np.abs(video_out[idx]).sum()), 0.0)
            self.assertEqual(float(np.abs(flow_out[idx]).sum()), 0.0)

    def test_temporal_shift_records_offset(self):
        video_out, flow_out, params = apply_visual_degradation(
            self.video.copy(), self.flow.copy(), "temporal_shift", seed=42
        )
        self.assertIn("temporal_shift_frames", params)
        shift = int(params["temporal_shift_frames"])
        self.assertNotEqual(shift, 0)
        self.assertLessEqual(abs(shift), min(10, self.frames - 1))

    def test_temporal_shift_edge_padding(self):
        """Edge padding should copy first/last frame when shifting."""
        small = np.arange(5, dtype=np.float32).reshape(5, 1, 1, 1)
        shifted_right, _, params = apply_visual_degradation(
            small.copy(), small.copy(), "temporal_shift", seed=999
        )
        # With seed 999, shift should be deterministic
        shift = int(params["temporal_shift_frames"])
        self.assertNotEqual(shift, 0)

    def test_wrong_video_uses_replacement(self):
        replacement_video = np.ones_like(self.video) * 5.0
        replacement_flow = np.ones_like(self.flow) * 3.0
        video_out, flow_out, params = apply_visual_degradation(
            self.video.copy(),
            self.flow.copy(),
            "wrong_video",
            seed=42,
            wrong_video=replacement_video,
            wrong_flow=replacement_flow,
            wrong_video_sample_id="piano/other_video/clip_0001",
        )
        self.assertTrue(np.allclose(video_out, replacement_video))
        self.assertTrue(np.allclose(flow_out, replacement_flow))
        self.assertEqual(
            params["wrong_video_sample_id"],
            "piano/other_video/clip_0001",
        )

    def test_wrong_video_requires_replacement(self):
        with self.assertRaises(ValueError):
            apply_visual_degradation(
                self.video.copy(), self.flow.copy(), "wrong_video", seed=42
            )

    def test_invalid_condition_raises(self):
        with self.assertRaises(ValueError):
            apply_visual_degradation(
                self.video.copy(), self.flow.copy(), "nonexistent", seed=42
            )

    def test_all_video_conditions_registered(self):
        self.assertEqual(
            set(VIDEO_CONDITIONS),
            {
                "original",
                "blur",
                "occlusion",
                "frame_drop",
                "temporal_shift",
                "wrong_video",
                "no_video",
                "shuffled_video",
            },
        )

    def test_gaussian_blur_standalone(self):
        # Use non-constant data so blur has a visible effect
        rng = np.random.RandomState(42)
        small = rng.randn(3, 2, 8, 8).astype(np.float32)
        blurred = _gaussian_blur(small, sigma=3.0)
        self.assertEqual(blurred.shape, small.shape)
        # Blur should change the values
        max_diff = float(np.abs(blurred - small).max())
        self.assertGreater(max_diff, 0.0, "Gaussian blur had no effect on non-constant image")

    def test_shift_with_edge_padding_standalone(self):
        small = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1)
        shifted = _shift_with_edge_padding(small, 1)
        # First frame is padded (value 0), then frames 0,1 follow
        self.assertEqual(float(shifted[0]), 0.0)
        self.assertEqual(float(shifted[1]), 0.0)
        self.assertEqual(float(shifted[2]), 1.0)

        shifted_neg = _shift_with_edge_padding(small, -1)
        self.assertEqual(float(shifted_neg[0]), 1.0)
        self.assertEqual(float(shifted_neg[1]), 2.0)
        self.assertEqual(float(shifted_neg[2]), 2.0)


if __name__ == "__main__":
    unittest.main()
