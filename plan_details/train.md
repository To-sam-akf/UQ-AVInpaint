对，下面给你两套：**正式训练** 和 **周期性 Stage9 测试**。路径按云端 `/root/EC-ViAv-vgpu` 写，你改 checkpoint 路径即可。

**0. 基础环境**

```bash
cd /root/EC-ViAv-vgpu

export DATA_ROOT=/root/shared-nvme/data
export BASELINE_CKPT=/root/EC-ViAv-vgpu/checkpoints/viai_av_patchgan_reference/VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export EXP_ROOT=checkpoints/formal_ec_viai_av
```

---

**1. 正式训练 Stage6：multi-candidate**

如果你之前的 1-batch smoke checkpoint 只是过拟合测试，不建议从它继续。建议从 VIAI-AV / VIAI-AV-PatchGAN reference checkpoint 开始。

```bash
export STAGE6_DIR=$EXP_ROOT/stage6_multi_candidate

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --resume \
  --resume_path "$BASELINE_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE6_DIR" \
  --log_event_path "$STAGE6_DIR/events" \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.0001 \
  --max_train_steps 50000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

---

**2. 正式训练 Stage7/full gate：evidence gate + evidence-scaled sigma**

把 `STAGE6_CKPT` 改成 Stage6 的最新或最佳 checkpoint。

```bash
export STAGE6_CKPT=$STAGE6_DIR/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export STAGE7_DIR=$EXP_ROOT/stage7_evidence_gate_sigma

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --resume \
  --resume_path "$STAGE6_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE7_DIR" \
  --log_event_path "$STAGE7_DIR/events" \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.00005 \
  --max_train_steps 80000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

---

**3. 正式训练 Stage8/full：candidate scorer + uncertainty**

把 `STAGE7_CKPT` 改成 Stage7 的 checkpoint。

```bash
export STAGE7_CKPT=$STAGE7_DIR/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export STAGE8_DIR=$EXP_ROOT/stage8_full_scorer_uncertainty

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.1 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE7_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE8_DIR" \
  --log_event_path "$STAGE8_DIR/events" \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

如果显存不够，把 `--batch_size 4` 改成 `1` 或 `2`。

---

**4. 周期性 Stage9 测试脚本**

每个 checkpoint 跑 4 个核心扰动：

```bash
export TEST_ROOT=checkpoints/stage9_periodic_eval
export CKPT=$STAGE8_DIR/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  echo "===== Stage9 eval: $MODE ====="

  python main.py test-viai-av -- \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --enable_evidence_gate \
    --freeze_gate_evidence_backbone \
    --enable_evidence_scaled_sigma \
    --enable_candidate_scorer \
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.1 \
    --lambda_calib 0.1 \
    --calib_error_tau 0.1 \
    --resume_path "$CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --results_dir "$TEST_ROOT/$(basename "$CKPT" .pth.tar)/$MODE" \
    --video_perturbation "$MODE" \
    --calibration_bins 10 \
    --display_id 0
done
```

---

**5. 多 checkpoint 自动评估**

比如评估 Stage8 目录下所有 5000 interval checkpoint：

```bash
export TEST_ROOT=checkpoints/stage9_periodic_eval

for CKPT in $(ls $STAGE8_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step*.pth.tar | sort)
do
  STEP=$(basename "$CKPT" .pth.tar)
  echo "========== Evaluating $STEP =========="

  for MODE in none flow_zero no_video wrong_video_cross_instrument
  do
    python main.py test-viai-av -- \
      --use_gan \
      --lambda_gan 0.001 \
      --enable_ec_viai_av \
      --stochastic_adapter \
      --enable_evidence_gate \
      --freeze_gate_evidence_backbone \
      --enable_evidence_scaled_sigma \
      --enable_candidate_scorer \
      --evidence_sigma_scale_min 0.5 \
      --evidence_sigma_scale_max 2.0 \
      --sigma_max 2.0 \
      --num_candidates 4 \
      --test_num_candidates 4 \
      --lambda_min_k 1.0 \
      --lambda_mean_k 0.1 \
      --lambda_boundary 0.05 \
      --lambda_diversity 0.05 \
      --lambda_gate_evidence 0.1 \
      --lambda_calib 0.1 \
      --calib_error_tau 0.1 \
      --resume_path "$CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$TEST_ROOT/$STEP/$MODE" \
      --video_perturbation "$MODE" \
      --calibration_bins 10 \
      --display_id 0
  done
done
```

---

**6. 汇总每个 checkpoint 的趋势**

```bash
find "$TEST_ROOT" -name "*_test.json" -print | sort | while read f
do
  python - <<PY
import json
p="$f"
d=json.load(open(p))
print(
    p,
    d["video_perturbation"],
    "top1=", round(d.get("top1_missing_l1", 0), 6),
    "bestK=", round(d.get("best_of_k_missing_l1", 0), 6),
    "oracle=", round(d.get("oracle_gain", 0), 6),
    "u=", round(d.get("uncertainty_mean", 0), 6),
    "evidence=", round(d.get("evidence_mean", 0), 6),
    "pairwise=", round(d.get("candidate_pairwise_mel_l1", 0), 6),
)
PY
done
```

你主要看这几个趋势：

```text
best_of_k_missing_l1 <= top1_missing_l1
wrong_video_cross_instrument / no_video / flow_zero 的 uncertainty 高于 none
flow_zero / no_video 的 evidence 低于 none
低 evidence 条件下 candidate_pairwise_mel_l1 更高
top1_missing_l1 <= candidate0_missing_l1 或至少接近
```

正式论文结果应该用 Stage8/full 的完整 train split checkpoint 跑 Stage9，而不是 1-batch overfit checkpoint。