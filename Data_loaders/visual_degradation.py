import random

import cv2
import numpy as np


VIDEO_CONDITIONS = (
    "original",
    "blur",
    "occlusion",
    "frame_drop",
    "temporal_shift",
    "wrong_video",
    "no_video",
    "shuffled_video",
)


def _validate_inputs(video, flow):
    video = np.asarray(video, dtype=np.float32)
    flow = np.asarray(flow, dtype=np.float32)
    if video.ndim != 4 or flow.ndim != 4:
        raise ValueError("video and flow must have shape [frames, channels, height, width]")
    if video.shape[0] != flow.shape[0] or video.shape[2:] != flow.shape[2:]:
        raise ValueError(
            f"RGB/flow shapes are not aligned: video={video.shape} flow={flow.shape}"
        )
    return video.copy(), flow.copy()


def _gaussian_blur(array, sigma):
    output = np.empty_like(array)
    for frame_index in range(array.shape[0]):
        hwc = np.transpose(array[frame_index], (1, 2, 0))
        blurred = cv2.GaussianBlur(
            hwc,
            (0, 0),
            sigmaX=float(sigma),
            sigmaY=float(sigma),
            borderType=cv2.BORDER_REFLECT_101,
        )
        if blurred.ndim == 2:
            blurred = blurred[:, :, None]
        output[frame_index] = np.transpose(blurred, (2, 0, 1))
    return output


def _shift_with_edge_padding(array, shift):
    if shift > 0:
        padding = np.repeat(array[:1], shift, axis=0)
        return np.concatenate((padding, array[:-shift]), axis=0)
    amount = abs(shift)
    padding = np.repeat(array[-1:], amount, axis=0)
    return np.concatenate((array[amount:], padding), axis=0)


def _frame_permutation(frame_count, rng):
    permutation = list(range(int(frame_count)))
    rng.shuffle(permutation)
    if frame_count > 1 and permutation == list(range(int(frame_count))):
        permutation = permutation[1:] + permutation[:1]
    return permutation


def apply_visual_degradation(
    video,
    flow,
    condition,
    seed,
    wrong_video=None,
    wrong_flow=None,
    wrong_video_sample_id=None,
):
    if condition not in VIDEO_CONDITIONS:
        raise ValueError(f"Unsupported video condition: {condition}")
    video, flow = _validate_inputs(video, flow)
    rng = random.Random(int(seed))
    parameters = {
        "condition": condition,
        "seed": int(seed),
    }

    if condition == "original":
        return video, flow, parameters

    if condition == "wrong_video":
        if wrong_video is None or wrong_flow is None or not wrong_video_sample_id:
            raise ValueError("wrong_video requires replacement RGB, flow, and sample id")
        video, flow = _validate_inputs(wrong_video, wrong_flow)
        parameters["wrong_video_sample_id"] = str(wrong_video_sample_id)
        return video, flow, parameters

    if condition == "no_video":
        video.fill(0.0)
        flow.fill(0.0)
        return video, flow, parameters

    if condition == "shuffled_video":
        permutation = _frame_permutation(video.shape[0], rng)
        video = video[permutation].copy()
        flow = flow[permutation].copy()
        parameters["frame_permutation"] = permutation
        return video, flow, parameters

    if condition == "blur":
        sigma = rng.uniform(0.8, 3.0)
        video = _gaussian_blur(video, sigma)
        flow = _gaussian_blur(flow, sigma)
        parameters["blur_sigma"] = float(sigma)
        return video, flow, parameters

    if condition == "occlusion":
        height, width = video.shape[-2:]
        box_height = max(1, int(round(height * rng.uniform(0.20, 0.45))))
        box_width = max(1, int(round(width * rng.uniform(0.20, 0.45))))
        y0 = rng.randint(0, max(0, height - box_height))
        x0 = rng.randint(0, max(0, width - box_width))
        y1 = y0 + box_height
        x1 = x0 + box_width
        video[:, :, y0:y1, x0:x1] = 0.0
        flow[:, :, y0:y1, x0:x1] = 0.0
        parameters["occlusion_box"] = [x0, y0, x1, y1]
        return video, flow, parameters

    if condition == "frame_drop":
        requested_keep_ratio = rng.uniform(0.50, 0.90)
        frame_count = video.shape[0]
        keep_count = max(1, min(frame_count, int(round(frame_count * requested_keep_ratio))))
        kept = set(rng.sample(range(frame_count), keep_count))
        dropped = [index for index in range(frame_count) if index not in kept]
        if dropped:
            video[dropped] = 0.0
            flow[dropped] = 0.0
        parameters["frame_keep_ratio"] = float(keep_count / frame_count)
        parameters["dropped_frames"] = dropped
        return video, flow, parameters

    maximum_shift = min(10, max(1, video.shape[0] - 1))
    shift = rng.choice(
        [value for value in range(-maximum_shift, maximum_shift + 1) if value != 0]
    )
    video = _shift_with_edge_padding(video, shift)
    flow = _shift_with_edge_padding(flow, shift)
    parameters["temporal_shift_frames"] = int(shift)
    return video, flow, parameters
