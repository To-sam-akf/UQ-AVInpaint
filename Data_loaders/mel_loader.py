import random
import torch


def build_missing_mask(batch_size, mel_bins, mel_steps, start, width, device):
    mask = torch.zeros(batch_size, 1, mel_bins, mel_steps, device=device)
    end = min(start + width, mel_steps)
    # 生成mask 缺失区域，mask=1 ，mask=0 表示已知区域
    mask[:, :, :, start:end] = 1.0
    return mask

#  用左右边界插值填入缺失区域
def interpolate_missing_region(mel_4d, start, width):
    end = min(start + width, mel_4d.size(-1))
    if end <= start:
        return mel_4d

    left_idx = max(start - 1, 0)
    right_idx = min(end, mel_4d.size(-1) - 1)

    left = mel_4d[:, :, :, left_idx:left_idx + 1]
    right = mel_4d[:, :, :, right_idx:right_idx + 1]

    if right_idx == left_idx:
        mel_4d[:, :, :, start:end] = left
        return mel_4d

    span = end - start
    alpha = torch.linspace(
        0.0, 1.0, steps=span, device=mel_4d.device, dtype=mel_4d.dtype
    ).view(1, 1, 1, span)
    mel_4d[:, :, :, start:end] = left * (1.0 - alpha) + right * alpha
    return mel_4d


def corrupt_mel_spectrogram(
    mel_batch,
    blank_frames,
    start=None,
    min_margin=3,
):
    """
    Corrupt a contiguous temporal region in mel-spectrograms and initialize it
    with boundary interpolation (following VIAI paper setup).

    Args:
        mel_batch: Tensor with shape (B, C, T) or (B, 1, C, T)
        blank_frames: Missing span length on the time axis
        start: Optional fixed start index. If None, sample randomly.
        min_margin: Keep a small clean margin near boundaries.
    Returns:
        corrupted, missing_mask, (start, end)
    """
    if mel_batch.dim() == 3:
        mel_4d = mel_batch.unsqueeze(1).clone()
    elif mel_batch.dim() == 4:
        mel_4d = mel_batch.clone()
    else:
        raise ValueError("mel_batch must be a 3D or 4D tensor")

    _, _, mel_bins, mel_steps = mel_4d.shape
    blank_frames = max(1, min(blank_frames, mel_steps))
    if start is None:
        max_start = max(min_margin, mel_steps - blank_frames - min_margin)
        start = random.randint(min_margin, max_start)
    else:
        start = max(min_margin, min(start, mel_steps - blank_frames))

    end = start + blank_frames
    mask = build_missing_mask(
        mel_4d.size(0), mel_bins, mel_steps, start, blank_frames, mel_4d.device
    )
    mel_4d = interpolate_missing_region(mel_4d, start, blank_frames)

    if mel_batch.dim() == 3:
        return mel_4d.squeeze(1), mask, (start, end)
    return mel_4d, mask, (start, end)


def corrupt_mel_spectrogram_batch(mel_batch, mask_specs, min_margin=3):
    """Corrupt each sample with its own fixed missing span."""
    if mel_batch.dim() == 3:
        mel_4d = mel_batch.unsqueeze(1).clone()
        squeeze_channel = True
    elif mel_batch.dim() == 4:
        mel_4d = mel_batch.clone()
        squeeze_channel = False
    else:
        raise ValueError("mel_batch must be a 3D or 4D tensor")
    if len(mask_specs) != mel_4d.size(0):
        raise ValueError(
            f"Expected {mel_4d.size(0)} mask specs, got {len(mask_specs)}"
        )

    batch_size, _, mel_bins, mel_steps = mel_4d.shape
    mask = torch.zeros(
        batch_size,
        1,
        mel_bins,
        mel_steps,
        device=mel_4d.device,
        dtype=mel_4d.dtype,
    )
    spans = []
    for index, spec in enumerate(mask_specs):
        start = int(spec["start"])
        end = int(spec["end"])
        gap_frames = int(spec.get("gap_frames", end - start))
        if end - start != gap_frames:
            raise ValueError(
                f"Mask spec {index} has inconsistent span/gap: "
                f"start={start} end={end} gap_frames={gap_frames}"
            )
        if start < min_margin or end > mel_steps - min_margin or end <= start:
            raise ValueError(
                f"Mask spec {index} is outside the valid Mel range: {start}:{end}"
            )
        mask[index, :, :, start:end] = 1.0
        mel_4d[index : index + 1] = interpolate_missing_region(
            mel_4d[index : index + 1],
            start,
            gap_frames,
        )
        spans.append((start, end))

    if squeeze_channel:
        return mel_4d.squeeze(1), mask, spans
    return mel_4d, mask, spans
