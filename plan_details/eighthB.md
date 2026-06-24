# Stage8 Heads-Only Fine-Tuning Plan

## Summary

从 Stage8 step `27750` checkpoint 开始做一个新小实验：冻结生成链路，只训练 `CandidateScorer` 和 `UncertaintyHead`。目标是进一步降低 scorer CE、增强 uncertainty-error 相关性，同时保持候选生成质量不被破坏。

当前代码没有 heads-only 训练模式，单靠命令行无法实现。因此先加一个显式训练开关，再跑 500-1000 个 global steps 的 sweep。

## Key Changes

- 新增命令行参数：
  - `--train_candidate_heads_only`，默认 `False`
  - 仅允许在 `--enable_candidate_scorer` 下使用
  - 要求 `lambda_calib > 0`
  - 建议与 `--resume_path step27750 checkpoint` 一起使用

- heads-only 模式行为：
  - 冻结以下模块参数：`Mel_Encoder`、`VideoEncoder`、`Mel_Decoder`、`BottleneckAdapter`、`EvidenceFusionGate`、`EvidenceEstimator`、`netD`
  - 仅 `CandidateScorer` 和 `UncertaintyHead` 保持 `requires_grad=True`
  - `optimizer_G` 只包含 `CandidateScorer + UncertaintyHead` 参数
  - 不更新 `optimizer_D`
  - forward 仍完整执行，继续记录 reconstruction、top1、candidate0、best-of-K、uncertainty 等指标
  - backward 只使用 `weighted_loss_calib`，避免 baseline reconstruction/GAN loss 影响冻结链路
  - resume 时跳过旧 `optimizer_G/optimizer_D` state，自动等价于 `--reset_optimizer`

- checkpoint 记录：
  - 保存新增字段 `train_candidate_heads_only`
  - `_stage_name()` 可保持 `EC-VIAI-AV-stage8-candidate-scorer-calib`，避免把它当成新模型结构；实验名通过 checkpoint dir 区分

## Experiment Command

起点 checkpoint：

```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE8_MAIN_CKPT=checkpoints/stage8_candidate_scorer_lowlr_sweep/EC-VIAI-AV-PatchGAN_checkpoint_step000027750.pth.tar
export EXP_DIR=checkpoints/stage8_candidate_heads_only_from27750
```

训练到 step `28250`，即从 `27750` 额外训练 500 steps：

```bash
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --train_candidate_heads_only \
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
  --lr 0.00002 \
  --resume \
  --resume_path "$STAGE8_MAIN_CKPT" \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$EXP_DIR" \
  --log_event_path "$EXP_DIR/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 28250 \
  --checkpoint_interval 250 \
  --print_freq 10 \
  --display_id 0
```

注意：本项目的 `--max_train_steps` 是 global step 上限，不是额外步数。因此从 `27750` 继续 500 steps 要写 `28250`；继续 1000 steps 要写 `28750`。

验证 checkpoint：

```bash
export STAGE8_HEADS_CKPT=checkpoints/stage8_candidate_heads_only_from27750/EC-VIAI-AV-PatchGAN_checkpoint_step000028000.pth.tar
export RESULT_DIR=checkpoints/stage8_candidate_heads_only_test_step28000

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
  --resume_path "$STAGE8_HEADS_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir "$RESULT_DIR" \
  --display_id 0
```

优先测试这些 checkpoint：

```text
step28000
step28250
```

如果 `step28250` 仍未退化，再继续同一设置到：

```text
step28500
step28750
```

## Test Plan

- 静态检查：
  - `python -m py_compile base_options.py Models/VIAI_AV_inpainting.py train_viai_av.py test_viai_av.py`
  - `pytest -q tests/test_multi_candidate_losses.py tests/test_evidence_fusion_gate.py`

- 新增/补充单元测试：
  - `--train_candidate_heads_only` 未启用 scorer 时直接报错
  - heads-only 模式下 optimizer 只包含 `CandidateScorer` 和 `UncertaintyHead` 参数
  - heads-only 模式下 generator / adapter / gate / discriminator 参数 `requires_grad=False`
  - resume heads-only checkpoint 时跳过旧 optimizer state，不报 optimizer param group mismatch
  - 单步 backward 后，只有 scorer/head 参数发生变化

- 实验接受标准：
  - `top1_missing_l1` 不差于 step27750 太多，建议阈值 `<= 0.0670`
  - `top1_missing_l1 <= candidate0_missing_l1`
  - `loss_candidate_scorer < 1.197279`
  - `uncertainty_error_spearman >= 0.460` 或至少不明显下降
  - `best_of_k_missing_l1` 接近 step27750，确认候选生成质量未被破坏

## Assumptions

- step27750 是当前 Stage8 主结果，作为新实验起点。
- 本实验不再从 step28100 或 step30100 继续训练。
- 训练目标不是提升 generator，而是只优化 candidate selection 和 uncertainty calibration。
- 如果 heads-only 后 top1 没提升，但 uncertainty 更稳定且 reconstruction 不退化，也可以作为 calibration-only ablation 记录。
