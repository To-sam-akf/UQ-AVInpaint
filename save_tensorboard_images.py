#!/usr/bin/env python3
"""Export TensorBoard scalar curves as PNG images.

Edit the CONFIG block below for the common case, or override values from the
command line. Example:

    uv run python save_tensorboard_images.py
    uv run python save_tensorboard_images.py --events-dir checkpoints/run/events --list-tags
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# =============================================================================
# CONFIG: usually you only need to edit this block.
# =============================================================================

# TensorBoard event directory or a single events.out.tfevents.* file.
EVENTS_DIR = Path("/home/sanmu/VIAIpro/checkpoints/formal_ec_viai_av/stage6_multi_candidate/events")

# Output directory for PNG curve images.
OUTPUT_DIR = Path("/home/sanmu/VIAIpro/tensorboard_curve_images/formal_ec_viai_av/stage6_multi_candidate")
    
# Scalar tags to export. Shell-style wildcards are supported.
# Examples:
#   ["train/loss_*", "train/gate/*", "val/*"]
#   ["*"] exports every scalar tag.
TAG_PATTERNS = ["*"]

# Tags to skip after TAG_PATTERNS are applied.
EXCLUDE_PATTERNS: list[str] = []

# Save one PNG for every selected tag.
SAVE_INDIVIDUAL_CURVES = True

# Save grouped PNGs, for example train/loss_*, train/gate/*, train/evidence/*.
SAVE_GROUPED_CURVES = True

# Save one large PNG containing every selected scalar.
SAVE_ALL_CURVES = False

# Optional CSV export for the selected scalar values.
SAVE_CSV = True

# 0 disables smoothing. Values such as 0.6 or 0.8 make noisy curves easier to read.
SMOOTHING = 0.0

# Optional step/value ranges. Use None to auto-scale.
X_MIN = None
X_MAX = None
Y_MIN = None
Y_MAX = None

# PNG canvas size.
IMAGE_WIDTH = 1400
IMAGE_HEIGHT = 850


# =============================================================================
# Implementation
# =============================================================================

PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#005f73",
    "#ae2012",
    "#6a994e",
    "#5a189a",
    "#ca6702",
]


@dataclass(frozen=True)
class Curve:
    tag: str
    steps: np.ndarray
    values: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TensorBoard scalar curves to PNG files."
    )
    parser.add_argument("--events-dir", type=Path, default=EVENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--tags",
        nargs="+",
        default=TAG_PATTERNS,
        help='Scalar tag patterns, for example: --tags "train/loss_*" "val/*"',
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=EXCLUDE_PATTERNS,
        help='Scalar tag patterns to skip, for example: --exclude "*/lr"',
    )
    parser.add_argument("--smooth", type=float, default=SMOOTHING)
    parser.add_argument("--x-min", type=float, default=X_MIN)
    parser.add_argument("--x-max", type=float, default=X_MAX)
    parser.add_argument("--y-min", type=float, default=Y_MIN)
    parser.add_argument("--y-max", type=float, default=Y_MAX)
    parser.add_argument("--width", type=int, default=IMAGE_WIDTH)
    parser.add_argument("--height", type=int, default=IMAGE_HEIGHT)
    parser.add_argument("--list-tags", action="store_true")
    parser.add_argument("--no-individual", action="store_true")
    parser.add_argument("--no-grouped", action="store_true")
    parser.add_argument("--all-curves", action="store_true")
    parser.add_argument("--no-csv", action="store_true")
    return parser.parse_args()


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def load_scalars(events_dir: Path) -> dict[str, Curve]:
    accumulator = EventAccumulator(str(events_dir), size_guidance={"scalars": 0})
    accumulator.Reload()

    curves: dict[str, Curve] = {}
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        steps = np.array([event.step for event in events], dtype=np.float64)
        values = np.array([event.value for event in events], dtype=np.float64)
        finite = np.isfinite(steps) & np.isfinite(values)
        if finite.any():
            curves[tag] = Curve(tag=tag, steps=steps[finite], values=values[finite])
    return curves


def match_any(text: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(text, pattern) for pattern in patterns)


def select_curves(
    curves: dict[str, Curve], include_patterns: list[str], exclude_patterns: list[str]
) -> dict[str, Curve]:
    selected = {
        tag: curve
        for tag, curve in curves.items()
        if match_any(tag, include_patterns) and not match_any(tag, exclude_patterns)
    }
    return dict(sorted(selected.items()))


def smooth_values(values: np.ndarray, smoothing: float) -> np.ndarray:
    smoothing = max(0.0, min(0.99, float(smoothing)))
    if smoothing == 0.0 or len(values) < 2:
        return values

    smoothed = np.empty_like(values, dtype=np.float64)
    last = values[0]
    smoothed[0] = last
    for index in range(1, len(values)):
        last = last * smoothing + values[index] * (1.0 - smoothing)
        smoothed[index] = last
    return smoothed


def infer_group_name(tag: str) -> str:
    parts = tag.split("/")
    if len(parts) >= 3:
        return "/".join(parts[:2])
    if len(parts) == 2:
        prefix, metric = parts
        for known in (
            "weighted_loss",
            "loss_probe",
            "loss",
            "psnr",
            "ssim",
            "gate",
            "evidence",
            "candidate",
            "adapter",
            "visual_evidence_aug",
        ):
            if metric == known or metric.startswith(f"{known}_"):
                return f"{prefix}/{known}"
        return f"{prefix}/{metric.split('_')[0]}"
    return "scalars"


def grouped_curves(curves: dict[str, Curve]) -> dict[str, list[Curve]]:
    groups: dict[str, list[Curve]] = {}
    for curve in curves.values():
        groups.setdefault(infer_group_name(curve.tag), []).append(curve)
    return dict(sorted(groups.items()))


def safe_filename(text: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip("/"))
    return filename.strip("_") or "curve"


def compact_number(value: float) -> str:
    if value == 0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 10000 or abs_value < 0.001:
        return f"{value:.2e}"
    if abs_value >= 100:
        return f"{value:.0f}"
    if abs_value >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def nice_ticks(min_value: float, max_value: float, count: int = 6) -> list[float]:
    if not math.isfinite(min_value) or not math.isfinite(max_value):
        return []
    if min_value == max_value:
        return [min_value]
    raw_step = (max_value - min_value) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice_step = magnitude
    elif normalized <= 2:
        nice_step = 2 * magnitude
    elif normalized <= 5:
        nice_step = 5 * magnitude
    else:
        nice_step = 10 * magnitude

    start = math.ceil(min_value / nice_step) * nice_step
    ticks = []
    value = start
    while value <= max_value + nice_step * 0.5:
        ticks.append(value)
        value += nice_step
    return ticks


def fit_ranges(
    curves: list[Curve],
    smoothing: float,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> tuple[float, float, float, float]:
    all_steps = np.concatenate([curve.steps for curve in curves])
    all_values = np.concatenate(
        [smooth_values(curve.values, smoothing) for curve in curves]
    )

    x0 = float(np.min(all_steps) if x_min is None else x_min)
    x1 = float(np.max(all_steps) if x_max is None else x_max)
    y0 = float(np.min(all_values) if y_min is None else y_min)
    y1 = float(np.max(all_values) if y_max is None else y_max)

    if x0 == x1:
        x0 -= 1.0
        x1 += 1.0
    if y0 == y1:
        padding = max(1.0, abs(y0) * 0.1)
        y0 -= padding
        y1 += padding
    elif y_min is None or y_max is None:
        padding = (y1 - y0) * 0.08
        if y_min is None:
            y0 -= padding
        if y_max is None:
            y1 += padding

    return x0, x1, y0, y1


def draw_curve_image(
    curves: list[Curve],
    output_path: Path,
    title: str,
    smoothing: float,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
    width: int,
    height: int,
) -> None:
    if not curves:
        return

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(25, bold=True)
    font = load_font(16)
    small_font = load_font(13)

    has_legend = len(curves) > 1
    left = 100
    top = 72
    right = 340 if has_legend else 48
    bottom = 82
    plot_left = left
    plot_top = top
    plot_right = width - right
    plot_bottom = height - bottom
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)

    x0, x1, y0, y1 = fit_ranges(curves, smoothing, x_min, x_max, y_min, y_max)

    def map_x(step: float) -> int:
        return int(plot_left + (step - x0) / (x1 - x0) * plot_width)

    def map_y(value: float) -> int:
        return int(plot_bottom - (value - y0) / (y1 - y0) * plot_height)

    draw.text((plot_left, 24), title[:90], fill="#111111", font=title_font)
    subtitle = f"{len(curves)} curve(s), smoothing={smoothing:g}, step=[{compact_number(x0)}, {compact_number(x1)}]"
    draw.text((plot_left, 50), subtitle, fill="#555555", font=small_font)

    draw.rectangle([plot_left, plot_top, plot_right, plot_bottom], outline="#222222")

    for tick in nice_ticks(x0, x1):
        x = map_x(tick)
        draw.line([(x, plot_top), (x, plot_bottom)], fill="#eeeeee")
        label = compact_number(tick)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 12), label, fill="#444444", font=small_font)

    for tick in nice_ticks(y0, y1):
        y = map_y(tick)
        draw.line([(plot_left, y), (plot_right, y)], fill="#eeeeee")
        label = compact_number(tick)
        bbox = draw.textbbox((0, 0), label, font=small_font)
        draw.text((plot_left - (bbox[2] - bbox[0]) - 12, y - 8), label, fill="#444444", font=small_font)

    draw.text((plot_left + plot_width / 2 - 18, height - 42), "step", fill="#333333", font=font)
    draw.text((18, plot_top + plot_height / 2 - 8), "value", fill="#333333", font=font)

    for index, curve in enumerate(curves):
        color = PALETTE[index % len(PALETTE)]
        values = smooth_values(curve.values, smoothing)
        points = [
            (map_x(float(step)), map_y(float(value)))
            for step, value in zip(curve.steps, values)
            if x0 <= step <= x1 and y0 <= value <= y1
        ]
        if len(points) == 1:
            x, y = points[0]
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)
        elif len(points) > 1:
            draw.line(points, fill=color, width=3, joint="curve")

    if has_legend:
        legend_x = plot_right + 24
        legend_y = plot_top
        draw.text((legend_x, legend_y), "Legend", fill="#111111", font=font)
        legend_y += 30
        for index, curve in enumerate(curves[:28]):
            color = PALETTE[index % len(PALETTE)]
            y = legend_y + index * 24
            draw.line([(legend_x, y + 8), (legend_x + 28, y + 8)], fill=color, width=4)
            label = curve.tag
            if len(label) > 36:
                label = "..." + label[-33:]
            draw.text((legend_x + 38, y), label, fill="#222222", font=small_font)
        if len(curves) > 28:
            draw.text(
                (legend_x, legend_y + 28 * 24),
                f"... {len(curves) - 28} more",
                fill="#666666",
                font=small_font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def save_curve_csv(curve: Curve, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("step,value\n")
        for step, value in zip(curve.steps, curve.values):
            handle.write(f"{int(step)},{value:.10g}\n")


def main() -> None:
    args = parse_args()
    if not args.events_dir.exists():
        raise FileNotFoundError(f"TensorBoard path does not exist: {args.events_dir}")

    all_scalars = load_scalars(args.events_dir)
    if args.list_tags:
        print(f"Scalar tags in {args.events_dir}:")
        for tag in sorted(all_scalars):
            print(f"  {tag}")
        return

    selected = select_curves(all_scalars, args.tags, args.exclude)
    if not selected:
        print("No scalar tags matched. Use --list-tags to inspect available tags.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_INDIVIDUAL_CURVES and not args.no_individual:
        for curve in selected.values():
            path = args.output_dir / "individual" / f"{safe_filename(curve.tag)}.png"
            draw_curve_image(
                [curve],
                path,
                curve.tag,
                args.smooth,
                args.x_min,
                args.x_max,
                args.y_min,
                args.y_max,
                args.width,
                args.height,
            )

    if SAVE_GROUPED_CURVES and not args.no_grouped:
        for group, curves in grouped_curves(selected).items():
            path = args.output_dir / "grouped" / f"{safe_filename(group)}.png"
            draw_curve_image(
                curves,
                path,
                group,
                args.smooth,
                args.x_min,
                args.x_max,
                args.y_min,
                args.y_max,
                args.width,
                args.height,
            )

    if SAVE_ALL_CURVES or args.all_curves:
        draw_curve_image(
            list(selected.values()),
            args.output_dir / "all_scalars.png",
            "All selected TensorBoard scalars",
            args.smooth,
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
            args.width,
            args.height,
        )

    if SAVE_CSV and not args.no_csv:
        for curve in selected.values():
            save_curve_csv(
                curve,
                args.output_dir / "csv" / f"{safe_filename(curve.tag)}.csv",
            )

    print(f"Exported {len(selected)} scalar tag(s) to: {args.output_dir}")


if __name__ == "__main__":
    main()
