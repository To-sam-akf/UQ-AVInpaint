# Stage 3 Evidence Estimator Validation

This folder contains a paired perturbation experiment for `VisualEvidenceEstimator`.
It tests whether evidence scores decrease as visual evidence is weakened on the same
anchor audio/video sample.

## Run

Set cloud paths:

```bash
export DATA_ROOT=/root/shared-nvme/data
export VIAI_AV_CKPT=checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar
export EVIDENCE_OUT=/tmp/evidence_estimator_validation
```

Run the validation:

```bash
python experiments/stage3/validate_evidence_estimator.py \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --checkpoint "$VIAI_AV_CKPT" \
  --use_gan \
  --max_anchors 100 \
  --num_workers 0 \
  --out_dir "$EVIDENCE_OUT"
```

Inspect the summary:

```bash
python - <<'PY'
import json
import os

summary_path = os.path.join(
    os.environ.get("EVIDENCE_OUT", "/tmp/evidence_estimator_validation"),
    "evidence_validation_summary.json",
)
with open(summary_path, "r", encoding="utf-8") as handle:
    summary = json.load(handle)

print("condition means:")
for condition, value in sorted(summary["condition_means"].items()):
    print(f"{condition:36s} {value:.6f}")

print("\npaired checks:")
for name, stats in summary["paired_checks"].items():
    print(
        f"{name:45s} "
        f"mean_delta={stats['mean_delta']:.6f} "
        f"hit_rate={stats['hit_rate']:.3f} "
        f"n={stats['n']}"
    )
PY
```

## Expected Evidence Trend

The strongest Stage 3 checks are:

```text
original > flow_75 > flow_50 > flow_25 > flow_zero
original > static_flow
original > static_video_zero_flow
```

`cross_instrument_wrong_video` is useful but not a hard failure case. If it does
not drop clearly, the current estimator is still valid as a lightweight visual
evidence-strength signal; it just means this stage is not yet a strong audio-video
matching classifier.

`temporal_shift_aux` is observational only because tensor rolling preserves most
motion statistics.
