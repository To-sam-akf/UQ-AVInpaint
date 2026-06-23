下面这套命令用于做正式对照：`Stage6`、`Stage7c`、`Stage7d` 三组都跑 normal test 和 controlled validation。

先设置路径：

```bash
export DATA_ROOT=/root/shared-nvme/data

export STAGE6_CKPT=checkpoints/stage6_multi_candidate_3k/EC-VIAI-AV-PatchGAN_checkpoint_step000024000.pth.tar
export STAGE7C_CKPT=checkpoints/stage7c_frozen_evidence_gate_1k/EC-VIAI-AV-PatchGAN_checkpoint_step000025000.pth.tar
export STAGE7D_CKPT=checkpoints/stage7d_evidence_scaled_sigma_3k/EC-VIAI-AV-PatchGAN_checkpoint_step000027000.pth.tar

export OUT_ROOT=checkpoints/stage7d_ablation_results
mkdir -p "$OUT_ROOT"
```

**1. Normal Test**

Stage6：

```bash
python main.py test-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --resume_path "$STAGE6_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir "$OUT_ROOT/stage6_normal"
```

Stage7c：

```bash
python main.py test-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_gate_evidence 0.1 \
  --lambda_diversity 0.2 \
  --resume_path "$STAGE7C_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir "$OUT_ROOT/stage7c_normal"
```

Stage7d：

```bash
python main.py test-viai-av -- \
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
  --lambda_gate_evidence 0.1 \
  --lambda_diversity 0.05 \
  --resume_path "$STAGE7D_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir "$OUT_ROOT/stage7d_normal"
```

**2. Controlled Validation**

Stage7c：

```bash
python tools/validate_evidence_gate.py \
  --checkpoint "$STAGE7C_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --use_gan \
  --lambda_gan 0.001 \
  --lambda_gate_evidence 0.1 \
  --lambda_diversity 0.2 \
  --freeze_gate_evidence_backbone \
  --num_candidates 4 \
  --batch_size 1 \
  --num_workers 0 \
  --max_anchors 100
```

Stage7d：

```bash
python tools/validate_evidence_gate.py \
  --checkpoint "$STAGE7D_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --use_gan \
  --lambda_gan 0.001 \
  --lambda_gate_evidence 0.1 \
  --lambda_diversity 0.05 \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --batch_size 1 \
  --num_workers 0 \
  --max_anchors 100
```

重点记录这几项：`mel_l1_missing`、`psnr_missing`、`gate_original_gt_flow_zero`、`pairwise_flow_zero_gt_original`、`sigma_scale_original_lt_flow_25`。