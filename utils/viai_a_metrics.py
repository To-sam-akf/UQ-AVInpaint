import numpy as np
import torch
import torch.nn.functional as F

from networks.uq.mel_autoencoder import time_gradient


_IMAGE_WRITE_WARNING_SHOWN = False


def _as_bchw(tensor):
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    if tensor.dim() == 4:
        return tensor
    raise ValueError("Expected a 3D (B, C, T) or 4D (B, 1, C, T) tensor.")

# 只替换 missing 区域”的拼接图
def compose_inpainted_mel(mel_input, mel_pred, missing_mask):
    """Return the final inpainted Mel: keep known bins, replace only the gap."""
    mel_input = _as_bchw(mel_input)
    mel_pred = _as_bchw(mel_pred)
    mask = _as_bchw(missing_mask).to(device=mel_pred.device, dtype=mel_pred.dtype)
    return mel_input * (1.0 - mask) + mel_pred * mask


def structural_similarity_2d(pred, target):
    try:
        from skimage.metrics import structural_similarity

        return float(structural_similarity(target, pred, data_range=1.0))
    except ModuleNotFoundError:
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        pred_mean = float(pred.mean())
        target_mean = float(target.mean())
        pred_var = float(pred.var())
        target_var = float(target.var())
        covariance = float(((pred - pred_mean) * (target - target_mean)).mean())
        numerator = (2 * pred_mean * target_mean + c1) * (2 * covariance + c2)
        denominator = (pred_mean ** 2 + target_mean ** 2 + c1) * (
            pred_var + target_var + c2
        )
        return numerator / max(denominator, 1e-12)


def compute_viai_a_metrics(mel_pred, mel_target, missing_mask, compute_ssim=True):
    pred = torch.clamp(_as_bchw(mel_pred).detach(), 0.0, 1.0)
    target = torch.clamp(_as_bchw(mel_target).detach(), 0.0, 1.0)
    mask = _as_bchw(missing_mask.detach()).to(device=pred.device, dtype=pred.dtype)
    batch_size = pred.size(0)

    full_mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    masked_sse = torch.sum(((pred - target) ** 2) * mask, dim=(1, 2, 3))
    masked_count = torch.clamp(torch.sum(mask, dim=(1, 2, 3)), min=1.0)
    missing_mse = masked_sse / masked_count

    full_psnr = -10.0 * torch.log10(torch.clamp(full_mse, min=1e-12))
    missing_psnr = -10.0 * torch.log10(torch.clamp(missing_mse, min=1e-12))
    full_psnr_sum = float(full_psnr.sum().cpu().item())
    missing_psnr_sum = float(missing_psnr.sum().cpu().item())

    ssim_full_sum = None
    if compute_ssim:
        pred_np = pred.squeeze(1).cpu().numpy()
        target_np = target.squeeze(1).cpu().numpy()
        ssim_values = [
            structural_similarity_2d(pred_np[index], target_np[index])
            for index in range(batch_size)
        ]
        ssim_full_sum = float(np.sum(ssim_values))

    metrics = {
        "psnr_full_sum": full_psnr_sum,
        "psnr_missing_sum": missing_psnr_sum,
        "ssim_full_sum": ssim_full_sum,
        "num_samples": batch_size,
        "psnr_full": full_psnr_sum / batch_size,
        "psnr_missing": missing_psnr_sum / batch_size,
        "ssim_full": None if ssim_full_sum is None else ssim_full_sum / batch_size,
    }
    return metrics


def compute_inpainting_sample_metrics(
    mel_completed,
    mel_target,
    missing_mask,
    mel_corrupted=None,
    compute_ssim=True,
):
    """Compute sampling/inpainting metrics for completed Mel predictions.

    This is intentionally shared by validation sampling and test-uq-av so the
    training-time validation curve has the same scale as formal evaluation.
    """
    pred = torch.clamp(_as_bchw(mel_completed).detach(), 0.0, 1.0)
    target = torch.clamp(_as_bchw(mel_target).detach(), 0.0, 1.0).to(
        device=pred.device, dtype=pred.dtype
    )
    mask = _as_bchw(missing_mask.detach()).to(
        device=pred.device, dtype=pred.dtype
    )
    batch_size = pred.size(0)

    abs_diff = torch.abs(pred - target)
    sq_diff = (pred - target) ** 2
    full_l1 = abs_diff.mean(dim=(1, 2, 3))
    full_mse = sq_diff.mean(dim=(1, 2, 3))

    missing_count = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)
    missing_l1 = (abs_diff * mask).sum(dim=(1, 2, 3)) / missing_count
    missing_mse = (sq_diff * mask).sum(dim=(1, 2, 3)) / missing_count
    psnr_full = -10.0 * torch.log10(full_mse.clamp(min=1e-12))
    psnr_missing = -10.0 * torch.log10(missing_mse.clamp(min=1e-12))

    grad_pred = time_gradient(pred)
    grad_target = time_gradient(target)
    weight = F.max_pool2d(
        mask[:, :1], kernel_size=(1, 5), stride=1, padding=(0, 2)
    )
    weight = weight[:, :, :, :grad_pred.size(-1)].clamp(0, 1)
    boundary_count = weight.sum(dim=(1, 2, 3)).clamp(min=1.0)
    boundary_l1 = (
        torch.abs(grad_pred - grad_target) * weight
    ).sum(dim=(1, 2, 3)) / boundary_count

    known_error = None
    if mel_corrupted is not None:
        corrupted = torch.clamp(
            _as_bchw(mel_corrupted).detach(), 0.0, 1.0
        ).to(device=pred.device, dtype=pred.dtype)
        known_mask = 1.0 - mask
        known_error = (
            torch.abs(pred - corrupted) * known_mask
        ).flatten(1).max(dim=1).values

    ssim_values = None
    ssim_full_sum = None
    if compute_ssim:
        pred_np = pred.squeeze(1).cpu().numpy()
        target_np = target.squeeze(1).cpu().numpy()
        ssim_values = [
            structural_similarity_2d(pred_np[index], target_np[index])
            for index in range(batch_size)
        ]
        ssim_full_sum = float(np.sum(ssim_values))

    per_sample = {
        "mel_l1_full": full_l1.cpu().tolist(),
        "mel_l1_missing": missing_l1.cpu().tolist(),
        "psnr_full_db": psnr_full.cpu().tolist(),
        "psnr_missing_db": psnr_missing.cpu().tolist(),
        "boundary_l1": boundary_l1.cpu().tolist(),
        "ssim_full": (
            [None] * batch_size if ssim_values is None else ssim_values
        ),
    }
    if known_error is not None:
        per_sample["known_region_max_abs_error"] = known_error.cpu().tolist()

    metrics = {
        "num_samples": batch_size,
        "psnr_full_sum": float(psnr_full.sum().cpu().item()),
        "psnr_missing_sum": float(psnr_missing.sum().cpu().item()),
        "mel_l1_full_sum": float(full_l1.sum().cpu().item()),
        "mel_l1_missing_sum": float(missing_l1.sum().cpu().item()),
        "ssim_full_sum": ssim_full_sum,
        "boundary_l1_sum": float(boundary_l1.sum().cpu().item()),
        "psnr_full_db": float(psnr_full.mean().cpu().item()),
        "psnr_missing_db": float(psnr_missing.mean().cpu().item()),
        "mel_l1_full": float(full_l1.mean().cpu().item()),
        "mel_l1_missing": float(missing_l1.mean().cpu().item()),
        "ssim_full": None if ssim_full_sum is None else ssim_full_sum / batch_size,
        "boundary_l1": float(boundary_l1.mean().cpu().item()),
        "per_sample": per_sample,
    }
    if known_error is not None:
        metrics["known_region_max_abs_error_sum"] = float(
            known_error.sum().cpu().item()
        )
        metrics["known_region_max_abs_error_mean"] = float(
            known_error.mean().cpu().item()
        )
        metrics["known_region_max_abs_error_max"] = float(
            known_error.max().cpu().item()
        )
    else:
        metrics["known_region_max_abs_error_sum"] = None
        metrics["known_region_max_abs_error_mean"] = None
        metrics["known_region_max_abs_error_max"] = None
    return metrics


def mel_image_batches(mel_input, mel_pred, mel_target, max_items=4):
    mel_input = torch.clamp(_as_bchw(mel_input).detach().cpu(), 0.0, 1.0)
    mel_pred = torch.clamp(_as_bchw(mel_pred).detach().cpu(), 0.0, 1.0)
    mel_target = torch.clamp(_as_bchw(mel_target).detach().cpu(), 0.0, 1.0)
    abs_error = torch.clamp(torch.abs(mel_pred - mel_target), 0.0, 1.0)

    count = min(max(1, int(max_items)), mel_target.size(0))
    return {
        "input_masked": mel_input[:count],
        "prediction": mel_pred[:count],
        "target": mel_target[:count],
        "abs_error": abs_error[:count],
    }


def _mel_to_uint8_image(mel_2d):
    array = torch.clamp(mel_2d.detach().cpu(), 0.0, 1.0).numpy()
    anchors = np.array(
        [
            [0.0015, 0.0005, 0.0139],
            [0.2515, 0.0380, 0.4034],
            [0.5783, 0.1480, 0.4044],
            [0.9023, 0.3645, 0.2711],
            [0.9871, 0.9914, 0.7495],
        ],
        dtype=np.float32,
    )
    scaled = array * (len(anchors) - 1)
    lower = np.floor(scaled).astype(np.int64)
    upper = np.clip(lower + 1, 0, len(anchors) - 1)
    lower = np.clip(lower, 0, len(anchors) - 1)
    weight = (scaled - lower)[..., None]
    rgb = anchors[lower] * (1.0 - weight) + anchors[upper] * weight
    return (rgb * 255.0).round().astype(np.uint8)


def save_mel_comparison_png(
    path,
    mel_masked,
    mel_interpolated,
    mel_raw_pred,
    mel_completed,
    mel_target,
):
    from PIL import Image, ImageDraw

    panels = [
        ("masked", _mel_to_uint8_image(mel_masked)),
        ("interpolated", _mel_to_uint8_image(mel_interpolated)),
        ("raw_prediction", _mel_to_uint8_image(mel_raw_pred)),
        ("completed", _mel_to_uint8_image(mel_completed)),
        ("groundtruth", _mel_to_uint8_image(mel_target)),
    ]
    label_height = 18
    gap = 4
    panel_width = panels[0][1].shape[1]
    panel_height = panels[0][1].shape[0]
    canvas_width = panel_width * len(panels) + gap * (len(panels) - 1)
    canvas_height = panel_height + label_height
    canvas = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x_offset = 0
    for label, array in panels:
        panel = Image.fromarray(array, mode="RGB")
        canvas.paste(panel, (x_offset, label_height))
        draw.text((x_offset + 2, 2), label, fill=(0, 0, 0))
        x_offset += panel_width + gap

    canvas.save(path)


def save_mel_comparison_batch(
    output_dir,
    start_index,
    paths,
    mel_input,
    mel_pred,
    mel_target,
    missing_mask=None,
):
    import os
    import re

    os.makedirs(output_dir, exist_ok=True)
    mel_input = torch.clamp(_as_bchw(mel_input).detach().cpu(), 0.0, 1.0)
    mel_pred = torch.clamp(_as_bchw(mel_pred).detach().cpu(), 0.0, 1.0)
    mel_target = torch.clamp(_as_bchw(mel_target).detach().cpu(), 0.0, 1.0)
    if missing_mask is None:
        mel_masked = mel_input
        mel_completed = mel_pred
    else:
        mask = _as_bchw(missing_mask).detach().cpu().to(dtype=mel_target.dtype)
        mel_masked = torch.clamp(mel_target * (1.0 - mask), 0.0, 1.0)
        mel_completed = torch.clamp(
            compose_inpainted_mel(mel_input, mel_pred, mask),
            0.0,
            1.0,
        )

    written_paths = []
    for index in range(mel_target.size(0)):
        sample_path = paths[index] if index < len(paths) else f"sample_{start_index + index}"
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_path).strip("_")
        if not safe_name:
            safe_name = "sample"
        filename = f"{start_index + index:06d}_{safe_name}.png"
        output_path = os.path.join(output_dir, filename)
        save_mel_comparison_png(
            output_path,
            mel_masked[index, 0],
            mel_input[index, 0],
            mel_pred[index, 0],
            mel_completed[index, 0],
            mel_target[index, 0],
        )
        written_paths.append(output_path)
    return written_paths


def write_mel_images(writer, prefix, step, mel_input, mel_pred, mel_target, max_items=4):
    global _IMAGE_WRITE_WARNING_SHOWN
    if writer is None:
        return
    for name, images in mel_image_batches(
        mel_input,
        mel_pred,
        mel_target,
        max_items=max_items,
    ).items():
        try:
            writer.add_images(f"{prefix}/mel_{name}", images, step)
        except ModuleNotFoundError as exc:
            if exc.name != "PIL":
                raise
            if not _IMAGE_WRITE_WARNING_SHOWN:
                print(
                    "[VIAI-A] TensorBoard Mel image logging requires Pillow. "
                    "Install `pillow` to enable image panels."
                )
                _IMAGE_WRITE_WARNING_SHOWN = True
            return
