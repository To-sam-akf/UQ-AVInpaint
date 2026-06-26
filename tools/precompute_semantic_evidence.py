#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.semantic_evidence import (  # noqa: E402
    infer_instrument_from_sample_dir,
    relative_sample_dir,
)


DEFAULT_SPLITS = [
    "train_av_split.txt",
    "val_av_split.txt",
    "test_av_split.txt",
]

PROMPT_TEMPLATES = [
    "a video of a person playing {instrument}",
    "a musician playing {instrument}",
    "a close-up video of a {instrument} performance",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute frozen CLIP semantic evidence for VIAI-AV samples."
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split-name", action="append", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default="ViT-B-32")
    parser.add_argument("--pretrained", default="openai")
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--instrument-list", default="")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_split_sample_dirs(data_root, split_name):
    split_path = Path(data_root) / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")
    sample_dirs = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample_dir = line.split("|")[0]
            if not os.path.isabs(sample_dir):
                sample_dir = os.path.join(data_root, sample_dir)
            sample_dirs.append(os.path.abspath(os.path.normpath(sample_dir)))
    return sample_dirs


def frame_sort_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def sample_frame_paths(sample_dir, num_frames):
    image_dir = Path(sample_dir) / "image_crop"
    paths = sorted(image_dir.glob("*.jpg"), key=frame_sort_key)
    if not paths:
        return []
    if len(paths) <= num_frames:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=num_frames)
    return [paths[int(round(index))] for index in indices]


def load_open_clip(model_name, pretrained, device):
    try:
        import open_clip
    except ImportError as exc:
        raise RuntimeError(
            "Semantic evidence precompute requires the optional open_clip dependency. "
            "Install it with `python -m pip install open_clip_torch` or "
            "`python -m pip install .[semantic]`."
        ) from exc
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tokenizer


def build_text_features(model, tokenizer, instruments, device):
    texts = []
    offsets = []
    for instrument in instruments:
        start = len(texts)
        texts.extend(template.format(instrument=instrument.replace("_", " ")) for template in PROMPT_TEMPLATES)
        offsets.append((start, len(texts)))
    with torch.no_grad():
        tokens = tokenizer(texts).to(device)
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        pooled = []
        for start, end in offsets:
            feature = text_features[start:end].mean(dim=0)
            feature = feature / feature.norm(dim=-1, keepdim=True)
            pooled.append(feature)
        return torch.stack(pooled, dim=0)


def encode_images(model, preprocess, frame_paths, device, batch_size):
    features = []
    with torch.no_grad():
        for start in range(0, len(frame_paths), batch_size):
            batch_paths = frame_paths[start : start + batch_size]
            images = []
            for path in batch_paths:
                with Image.open(path) as image:
                    images.append(preprocess(image.convert("RGB")))
            image_batch = torch.stack(images, dim=0).to(device)
            image_features = model.encode_image(image_batch)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features.append(image_features)
    return torch.cat(features, dim=0)


def score_sample(
    model,
    preprocess,
    text_features,
    instruments,
    sample_dir,
    data_root,
    num_frames,
    batch_size,
    device,
):
    instrument = infer_instrument_from_sample_dir(sample_dir)
    target_index = instruments.index(instrument)
    frame_paths = sample_frame_paths(sample_dir, num_frames)
    if not frame_paths:
        raise RuntimeError(f"No image frames found for semantic evidence: {sample_dir}")

    image_features = encode_images(model, preprocess, frame_paths, device, batch_size)
    logit_scale = getattr(model, "logit_scale", None)
    scale = logit_scale.exp() if logit_scale is not None else 100.0
    probs = (scale * image_features @ text_features.T).softmax(dim=-1)
    avg_probs = probs.mean(dim=0)
    top1_index = int(torch.argmax(avg_probs).detach().cpu().item())
    target_prob = float(avg_probs[target_index].detach().cpu().item())
    top1_prob = float(avg_probs[top1_index].detach().cpu().item())
    target_rank = int((avg_probs > avg_probs[target_index]).sum().detach().cpu().item()) + 1
    frame_top1 = torch.argmax(probs, dim=1)
    frame_top1_indices = frame_top1.detach().cpu().tolist()
    frame_consistency = float((frame_top1 == target_index).float().mean().detach().cpu().item())
    avg_probs_cpu = avg_probs.detach().cpu().tolist()
    probs_by_instrument = {
        instrument_name: float(avg_probs_cpu[index])
        for index, instrument_name in enumerate(instruments)
    }
    sample_dir_out = relative_sample_dir(sample_dir, data_root=data_root) or sample_dir
    return {
        "sample_dir": sample_dir_out,
        "instrument": instrument,
        "semantic_score": target_prob,
        "target_prob": target_prob,
        "probs_by_instrument": probs_by_instrument,
        "top1_instrument": instruments[top1_index],
        "top1_prob": top1_prob,
        "target_rank": target_rank,
        "frame_consistency": frame_consistency,
        "frame_top1_instruments": [
            instruments[index] for index in frame_top1_indices
        ],
        "num_frames": len(frame_paths),
    }


def main():
    args = parse_args()
    data_root = os.path.abspath(args.data_root)
    split_names = args.split_name or DEFAULT_SPLITS
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir or os.path.join(
        data_root,
        "semantic_evidence",
        "clip_vit_b32",
    )
    os.makedirs(output_dir, exist_ok=True)

    split_dirs = {}
    for split_name in split_names:
        dirs = read_split_sample_dirs(data_root, split_name)
        split_dirs[split_name] = dirs[: args.limit] if args.limit is not None else dirs

    if args.instrument_list:
        instruments = [item.strip() for item in args.instrument_list.split(",") if item.strip()]
    else:
        instruments = sorted(
            {
                infer_instrument_from_sample_dir(sample_dir)
                for dirs in split_dirs.values()
                for sample_dir in dirs
            }
        )
    if not instruments:
        raise RuntimeError("No instruments found for semantic evidence precompute.")

    model, preprocess, tokenizer = load_open_clip(args.model_name, args.pretrained, device)
    text_features = build_text_features(model, tokenizer, instruments, device)
    print(
        "[semantic-evidence] "
        f"splits={','.join(split_names)} instruments={len(instruments)} "
        f"model={args.model_name}/{args.pretrained} device={device}"
    )

    for split_name, sample_dirs in split_dirs.items():
        output_path = os.path.join(output_dir, f"{Path(split_name).stem}.jsonl")
        with open(output_path, "w", encoding="utf-8") as handle:
            for sample_dir in tqdm(sample_dirs, desc=f"[semantic-evidence] {split_name}"):
                record = score_sample(
                    model,
                    preprocess,
                    text_features,
                    instruments,
                    sample_dir,
                    data_root,
                    args.num_frames,
                    args.batch_size,
                    device,
                )
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"[semantic-evidence] wrote {output_path}")


if __name__ == "__main__":
    main()
