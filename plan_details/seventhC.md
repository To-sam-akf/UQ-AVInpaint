# Stage 7c：Frozen Evidence Backbone 修正计划

## Summary
Stage7b 已证明 gate 不再饱和，但 stronger 版本暴露了更深问题：`evidence_score` 虽然已在 `torch.no_grad()` 下计算，但它依赖可训练的 `Mel_Encoder` / `VideoEncoder` 特征，训练 1000 steps 后 evidence 数值空间漂移，导致 `original < flow_zero/static` 这类反常结果。Stage7c 的目标是固定 evidence 的特征基准，让 gate 学习稳定的 motion evidence 排序，而不是继续调大 loss。

## Key Changes
- 新增开关：
  - `--freeze_gate_evidence_backbone`
  - 默认 `False`，保持旧 Stage7/Stage7b 兼容。
- 启用后：
  - 冻结 `Mel_Encoder` 和 `VideoEncoder` 参数，不加入 `optimizer_G`。
  - `optimize_parameters()` 中保持二者 `eval()`，避免训练态状态漂移。
  - 继续训练 `Mel_Decoder`、`BottleneckAdapter`、`EvidenceFusionGate`。
  - `EvidenceEstimator` 继续用当前 full formula，但输入来自冻结后的 encoder 特征，因此 `evidence_score` 稳定。
- 保留现有：
  - `torch.no_grad()` evidence 计算。
  - `gate_target = clamp((evidence - low) / (high - low), 0, 1)`。
  - `L_gate_evidence = SmoothL1(gate, gate_target.detach())`。
  - `L_evidence_div`。
- 推荐 Stage7c 权重回到温和配置：
  - `--lambda_gate_evidence 0.1`
  - `--lambda_diversity 0.2`
  - 不再使用失败的 `0.2 / 0.4` stronger 配置。

## Validation Improvements
- 扩展 `tools/validate_evidence_gate.py`：
  - 增加 `--lambda_gan`、`--lambda_gate_evidence`、`--lambda_diversity`、`--evidence_gate_low`、`--evidence_gate_high`、`--freeze_gate_evidence_backbone`。
  - 增加 `--max_anchors 100`，不再只看单个 accordion 样本。
  - 输出每个 condition 的均值：`evidence_mean`、`gate_mean`、`gate_target`、`candidate_pairwise_distance`、`evidence_diversity_gap`。
  - 输出 paired hit-rate：
    - `original > flow_75 > flow_50 > flow_25 > flow_zero/static` for evidence。
    - `original gate_mean > flow_zero/static gate_mean`。
    - `flow_zero/static pairwise > original pairwise`。
  - wrong video / temporal shift 继续只作为 harder stress test，不作为 Stage7c 成功硬指标。

## Experiment Command
从 stage6 checkpoint 重新跑，不使用 `stage7b_controlled_gate_1k` 或 `stage7b_controlled_gate_stronger_1k`：

```bash
export EXP_DIR=checkpoints/stage7c_frozen_evidence_gate_1k

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.5 \
  --visual_evidence_aug_modes flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.2 \
  --lambda_gate_evidence 0.1 \
  --evidence_gate_low 0.24 \
  --evidence_gate_high 0.34 \
  --resume \
  --resume_path "$STAGE6_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$EXP_DIR" \
  --log_event_path "$EXP_DIR/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 25000 \
  --print_freq 20 \
  --save_latest_freq 500 \
  --save_epoch_freq 1000 \
  --display_id 0
```

验证命令：

```bash
python tools/validate_evidence_gate.py \
  --checkpoint "$EXP_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000025000.pth.tar" \
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

## Success Criteria
- TensorBoard：
  - `gate/mean` 不饱和到 `0.95+`。
  - `gate/target_gap` smoothed 接近 0，至少不长期大幅为正。
  - `visual_evidence_aug/applied` smoothed 接近 `0.5`。
  - `evidence/mean` 不出现 stronger 版本那种明显反向漂移。
- 验证脚本：
  - controlled degradation 下 evidence 排序稳定成立。
  - `original gate_mean` 明显高于 `flow_zero/static_video_zero_flow`。
  - `flow_zero/static_video_zero_flow candidate_pairwise_distance` 高于 original。
  - `evidence_diversity_gap` 不再长期约 `-0.06`，目标先压到 `>-0.03`。
- 质量指标：
  - `mel_l1_missing` / `psnr_missing` 相比 stage6 不明显崩坏。
  - 1000 steps 成立后再跑 3000 steps；只有正式对照时再跑 5000 steps。

## Test Plan
- 单元测试：
  - `--freeze_gate_evidence_backbone` 时 `Mel_Encoder` / `VideoEncoder` 参数不在 `optimizer_G`。
  - 一次 `optimize_parameters()` 后 frozen encoder 参数不变。
  - frozen 模式下训练时 encoder 保持 `eval()`。
  - `lambda_gate_evidence=0` 且不开 freeze 时旧 Stage7b 行为保持兼容。
- 静态检查：
  ```bash
  uv run python -m py_compile Models/VIAI_AV_inpainting.py base_options.py tools/validate_evidence_gate.py
  uv run pytest -q tests/test_multi_candidate_losses.py tests/test_evidence_fusion_gate.py
  ```

## Assumptions
- Stage7c 选择“冻结现有 encoder”而不是复制一套 frozen reference encoder，优先降低实现复杂度和显存开销。
- Stage7c 仍讲 “motion-evidence-aware fusion”，不声称解决 wrong video / temporal shift。
- 失败的 stronger checkpoint 不再继续训练，所有 Stage7c 实验从 stage6 checkpoint 重新开始。
