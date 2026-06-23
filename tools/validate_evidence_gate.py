import argparse
import os
import random
import sys

import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import Options_inpainting
from networks.EC_VIAI_Modules import apply_visual_evidence_augmentation


CONTROLLED_CONDITIONS = [
    "original",
    "flow_75",
    "flow_50",
    "flow_25",
    "flow_zero",
    "static_video_zero_flow",
]
HARDER_CONDITIONS = ["wrong_video", "temporal_shift"]
METRIC_NAMES = [
    "evidence_mean",
    "gate_mean",
    "gate_target",
    "gate_gap",
    "candidate_pairwise_distance",
    "evidence_diversity_gap",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare evidence gate diagnostics under controlled video perturbations."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--test_split_name", default="test_av_split.txt")
    parser.add_argument("--num_candidates", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--blank_frames", type=int, default=40)
    parser.add_argument("--temporal_shift", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_anchors", type=int, default=1)
    parser.add_argument("--use_gan", action="store_true")
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_gate_evidence", type=float, default=0.0)
    parser.add_argument("--lambda_diversity", type=float, default=0.0)
    parser.add_argument("--evidence_gate_low", type=float, default=0.24)
    parser.add_argument("--evidence_gate_high", type=float, default=0.34)
    parser.add_argument("--freeze_gate_evidence_backbone", action="store_true")
    return parser.parse_args()


def instrument_from_batch(batch):
    path = str(batch[-1][0])
    parts = path.split("/")
    if "processed" in parts:
        index = parts.index("processed")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def collect_anchor_batches(loader, max_anchors):
    anchors = []
    wrong_pool = []
    seen_paths = set()
    max_anchors = max(1, int(max_anchors))
    for batch in loader:
        if batch is None:
            continue
        path = str(batch[-1][0])
        if path not in seen_paths:
            wrong_pool.append(batch)
            seen_paths.add(path)
        if len(anchors) < max_anchors:
            anchors.append(batch)
        if len(anchors) >= max_anchors:
            anchor_instruments = {instrument_from_batch(item) for item in anchors}
            pool_instruments = {instrument_from_batch(item) for item in wrong_pool}
            if len(pool_instruments - anchor_instruments) > 0 or len(pool_instruments) > 1:
                break
    if not anchors:
        raise RuntimeError("No valid test batches found.")
    return anchors, wrong_pool


def find_wrong_batch(anchor_batch, wrong_pool):
    anchor_instrument = instrument_from_batch(anchor_batch)
    anchor_path = str(anchor_batch[-1][0])
    for candidate in wrong_pool:
        if instrument_from_batch(candidate) != anchor_instrument:
            return candidate
    for candidate in wrong_pool:
        if str(candidate[-1][0]) != anchor_path:
            return candidate
    return None


def replace_video(anchor_batch, source_batch):
    return (
        source_batch[0],
        source_batch[1],
        anchor_batch[2],
        anchor_batch[3],
        anchor_batch[4],
        anchor_batch[5],
        anchor_batch[6],
        anchor_batch[7],
    )


def make_batch(video_batch, flow_batch, anchor_batch):
    return (
        video_batch,
        flow_batch,
        anchor_batch[2],
        anchor_batch[3],
        anchor_batch[4],
        anchor_batch[5],
        anchor_batch[6],
        anchor_batch[7],
    )


def controlled_aug_batch(batch, mode):
    video_batch, flow_batch = apply_visual_evidence_augmentation(batch[0], batch[1], mode)
    return make_batch(video_batch, flow_batch, batch)


def temporal_shift_batch(batch, shift):
    return (
        batch[0].roll(shifts=shift, dims=1),
        batch[1].roll(shifts=shift, dims=1),
        batch[2],
        batch[3],
        batch[4],
        batch[5],
        batch[6],
        batch[7],
    )


def evaluate_condition(model, batch, label, seed, blank_frames):
    random.seed(seed)
    torch.manual_seed(seed)
    model.blank_length = int(blank_frames)
    model.set_inputs(batch)
    model.test(global_step=0)
    model.get_loss_items()
    return {
        "label": label,
        "evidence_mean": model.evidence_mean_item,
        "gate_mean": model.gate_mean_item,
        "gate_target": model.gate_target_mean_item,
        "gate_gap": model.gate_target_gap_item,
        "candidate_pairwise_distance": model.candidate_pairwise_distance_item,
        "evidence_diversity_gap": model.evidence_diversity_gap_item,
        "path": model.path_batch[0],
    }


def print_single_anchor(records_by_condition):
    for label in CONTROLLED_CONDITIONS + HARDER_CONDITIONS:
        record = records_by_condition.get(label)
        if record is None:
            continue
        print(
            f"{label}: "
            f"evidence_mean={record['evidence_mean']:.6f} "
            f"gate_mean={record['gate_mean']:.6f} "
            f"gate_target={record['gate_target']:.6f} "
            f"gate_gap={record['gate_gap']:.6f} "
            f"candidate_pairwise_distance={record['candidate_pairwise_distance']:.6f} "
            f"evidence_diversity_gap={record['evidence_diversity_gap']:.6f} "
            f"path={record['path']}"
        )


def mean(values):
    return sum(values) / max(1, len(values))


def summarize(anchor_results):
    condition_records = {}
    for records_by_condition in anchor_results:
        for label, record in records_by_condition.items():
            condition_records.setdefault(label, []).append(record)

    print("\ncondition means:")
    for label in CONTROLLED_CONDITIONS + HARDER_CONDITIONS:
        records = condition_records.get(label, [])
        if not records:
            continue
        metric_text = " ".join(
            f"{metric}={mean([record[metric] for record in records]):.6f}"
            for metric in METRIC_NAMES
        )
        print(f"  {label:<30} n={len(records):03d} {metric_text}")

    def paired_check(name, lhs_label, rhs_label, metric, direction="gt"):
        deltas = []
        for records_by_condition in anchor_results:
            lhs = records_by_condition.get(lhs_label)
            rhs = records_by_condition.get(rhs_label)
            if lhs is None or rhs is None:
                continue
            delta = lhs[metric] - rhs[metric]
            deltas.append(delta)
        if not deltas:
            return
        if direction == "gt":
            hits = [delta > 0.0 for delta in deltas]
            mean_delta = mean(deltas)
        else:
            hits = [delta < 0.0 for delta in deltas]
            mean_delta = -mean(deltas)
        print(
            f"  {name:<48} "
            f"mean_delta={mean_delta:.6f} "
            f"hit_rate={mean([1.0 if hit else 0.0 for hit in hits]):.3f} "
            f"n={len(deltas)}"
        )

    print("\npaired checks:")
    paired_check("evidence_original_gt_flow_75", "original", "flow_75", "evidence_mean")
    paired_check("evidence_flow_75_gt_flow_50", "flow_75", "flow_50", "evidence_mean")
    paired_check("evidence_flow_50_gt_flow_25", "flow_50", "flow_25", "evidence_mean")
    paired_check("evidence_flow_25_gt_flow_zero", "flow_25", "flow_zero", "evidence_mean")
    paired_check(
        "evidence_flow_25_gt_static_video_zero_flow",
        "flow_25",
        "static_video_zero_flow",
        "evidence_mean",
    )
    paired_check("gate_original_gt_flow_zero", "original", "flow_zero", "gate_mean")
    paired_check(
        "gate_original_gt_static_video_zero_flow",
        "original",
        "static_video_zero_flow",
        "gate_mean",
    )
    paired_check(
        "pairwise_flow_zero_gt_original",
        "flow_zero",
        "original",
        "candidate_pairwise_distance",
    )
    paired_check(
        "pairwise_static_video_zero_flow_gt_original",
        "static_video_zero_flow",
        "original",
        "candidate_pairwise_distance",
    )
    paired_check(
        "evidence_original_gt_wrong_video_observation",
        "original",
        "wrong_video",
        "evidence_mean",
    )
    paired_check(
        "evidence_original_gt_temporal_shift_observation",
        "original",
        "temporal_shift",
        "evidence_mean",
    )


def main():
    args = parse_args()
    hparam_args = [
        "--enable_ec_viai_av",
        "--stochastic_adapter",
        "--enable_evidence_gate",
        "--num_candidates",
        str(args.num_candidates),
        "--test_num_candidates",
        str(args.num_candidates),
        "--data_root",
        args.data_root,
        "--test_split_name",
        args.test_split_name,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--resume_path",
        args.checkpoint,
        "--lambda_gan",
        str(args.lambda_gan),
        "--lambda_gate_evidence",
        str(args.lambda_gate_evidence),
        "--lambda_diversity",
        str(args.lambda_diversity),
        "--evidence_gate_low",
        str(args.evidence_gate_low),
        "--evidence_gate_high",
        str(args.evidence_gate_high),
    ]
    if args.use_gan:
        hparam_args.append("--use_gan")
    if args.freeze_gate_evidence_backbone:
        hparam_args.append("--freeze_gate_evidence_backbone")
    hparams = Options_inpainting.Inpainting_Config(force_reload=True, args=hparam_args)

    from Data_loaders import audio_loader as av_loader
    from Models.VIAI_AV_inpainting import VIAIAVModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = av_loader.get_data_loaders(
        hparams.data_root,
        hparams.speaker_id,
        test_shuffle=False,
        phases=("test",),
    )
    if "test" not in data_loaders:
        raise RuntimeError(f"Missing test split: {hparams.test_split_name}")

    anchor_batches, wrong_pool = collect_anchor_batches(
        data_loaders["test"],
        args.max_anchors,
    )
    model = VIAIAVModel(hparams, device=device)
    model.load_checkpoint(args.checkpoint, reset_optimizer=True)

    print(f"num_anchors: {len(anchor_batches)}")
    print("first_anchor_instrument:", instrument_from_batch(anchor_batches[0]))
    first_wrong = find_wrong_batch(anchor_batches[0], wrong_pool)
    if first_wrong is not None:
        print("first_wrong_video_source_instrument:", instrument_from_batch(first_wrong))

    anchor_results = []
    for anchor_index, anchor_batch in enumerate(anchor_batches):
        seed = args.seed + anchor_index
        records = {
            "original": evaluate_condition(
                model,
                anchor_batch,
                "original",
                seed,
                args.blank_frames,
            )
        }
        for mode in CONTROLLED_CONDITIONS[1:]:
            records[mode] = evaluate_condition(
                model,
                controlled_aug_batch(anchor_batch, mode),
                mode,
                seed,
                args.blank_frames,
            )
        wrong_batch = find_wrong_batch(anchor_batch, wrong_pool)
        if wrong_batch is not None:
            records["wrong_video"] = evaluate_condition(
                model,
                replace_video(anchor_batch, wrong_batch),
                "wrong_video",
                seed,
                args.blank_frames,
            )
        records["temporal_shift"] = evaluate_condition(
            model,
            temporal_shift_batch(anchor_batch, args.temporal_shift),
            "temporal_shift",
            seed,
            args.blank_frames,
        )
        anchor_results.append(records)
        if len(anchor_batches) == 1:
            print_single_anchor(records)

    summarize(anchor_results)


if __name__ == "__main__":
    main()
