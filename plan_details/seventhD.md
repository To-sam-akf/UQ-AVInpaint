# Stage 7d：Evidence-Conditioned Uncertainty Scaling

## Summary
Stage7c 已经证明 frozen evidence backbone 能稳定产生 `original > flow_zero/static` 的 evidence 排序，并且 gate 会随 evidence 降低。但候选多样性仍然失败：`flow_zero/static` 的 `candidate_pairwise_distance` 低于 `original`。Stage7d 不再只依赖 `L_evidence_div` 间接拉开候选，而是把 evidence 直接接入 stochastic adapter 的采样尺度：视觉证据弱时显式增大候选扰动，视觉证据强时压低候选扰动。

## Key Changes
- 新增开关和参数：
  - `--enable_evidence_scaled_sigma`
  - `--evidence_sigma_scale_min 0.5`
  - `--evidence_sigma_scale_max 2.0`
- 计算方式：
  - 复用 Stage7c 的 frozen evidence 和 `gate_target`。
  - `uncertainty = 1 - gate_target.detach()`。
  - `sigma_scale = min_scale + (max_scale - min_scale) * uncertainty`。
  - 传入 `BottleneckAdapter.sample_residuals(...)`，只影响 stochastic candidates 的采样噪声；由于 `eps[:, 0] = 0`，第 0 个 anchor candidate 不被随机噪声破坏。
- `BottleneckAdapter.sample_latent()` 扩展可选参数 `sigma_scale=None`：
  - 默认 `None` 时旧行为完全不变。
  - 启用后：`effective_sigma = clamp(base_sigma * sigma_scale, sigma_min, sigma_max)`。
  - Stage7d 实验命令显式设置 `--sigma_max 2.0`，否则低 evidence 的 sigma 会被旧上限 `1.0` 截断。
- 保留 Stage7c：
  - `--freeze_gate_evidence_backbone`
  - `--lambda_gate_evidence 0.1`
  - evidence gate 逻辑不变。
- 调整 diversity loss 用法：
  - 首轮 Stage7d smoke 将 `--lambda_diversity` 从 `0.2` 降到 `0.05`，避免旧 `abs(pairwise-target)` 过强地压制低 evidence 多样性。
  - `L_evidence_div` 继续记录和参与训练，但 Stage7d 的主要多样性机制改为 evidence-scaled sigma。

## Logging / Validation
- TensorBoard 新增：
  - `adapter/sigma_scale_mean`
  - `adapter/sigma_scale_min`
  - `adapter/sigma_scale_max`
  - `adapter/effective_sigma_mean`
- `tools/validate_evidence_gate.py` 扩展：
  - 增加 `--enable_evidence_scaled_sigma`
  - 增加 `--evidence_sigma_scale_min`
  - 增加 `--evidence_sigma_scale_max`
  - 输出每个 condition 的 `adapter_sigma_mean`、`sigma_scale_mean`。
- Stage7d 成功标准：
  - evidence 排序保持 Stage7c 水平。
  - gate 排序保持：`original gate_mean > flow_zero/static gate_mean`，hit-rate 接近 `1.0`。
  - 新增核心指标成立：`flow_zero/static candidate_pairwise_distance > original`，hit-rate 目标先达到 `>0.7`。
  - `sigma_scale_mean` 满足：`original < flow_25 < flow_zero/static`。
  - `psnr_missing` / `mel_l1_missing` 相比 Stage7c 不明显崩坏。

## Experiment Commands
先从 Stage7c 1k checkpoint 继续一个快速修复实验，验证 sigma scaling 是否能翻转 pairwise 排序：

```bash
export STAGE7C_CKPT=checkpoints/stage7c_frozen_evidence_gate_1k/EC-VIAI-AV-PatchGAN_checkpoint_step000025000.pth.tar
export EXP_DIR=checkpoints/stage7d_evidence_scaled_sigma_1k

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
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.5 \
  --visual_evidence_aug_modes flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --evidence_gate_low 0.24 \
  --evidence_gate_high 0.34 \
  --resume \
  --resume_path "$STAGE7C_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$EXP_DIR" \
  --log_event_path "$EXP_DIR/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 26000 \
  --print_freq 20 \
  --save_latest_freq 500 \
  --save_epoch_freq 1000 \
  --display_id 0
```

验证：

```bash
python tools/validate_evidence_gate.py \
  --checkpoint "$EXP_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000026000.pth.tar" \
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

若 1k 修复实验成功，再从 Stage6 checkpoint 做正式 3000-step 对照：

```bash
export EXP_DIR=checkpoints/stage7d_evidence_scaled_sigma_3k
# 同上参数，但 --resume_path "$STAGE6_CKPT"，--max_train_steps 27000
```

## Test Plan
- 单元测试：
  - `sample_latent(..., sigma_scale=None)` 与旧行为 shape/range 兼容。
  - `sigma_scale` 为 `[B,1,1,1]` 时能广播到 `[B,C,H,W]`。
  - `gate_target` 高时 `sigma_scale` 低，`gate_target` 低时 `sigma_scale` 高。
  - `eps[:,0]=0` 下第 0 个 latent candidate 不受 sigma scale 改变。
  - 不启用 `--enable_evidence_scaled_sigma` 时 Stage7c 路径不变。
- 静态检查：
  ```bash
  uv run python -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py base_options.py tools/validate_evidence_gate.py
  uv run pytest -q tests/test_multi_candidate_losses.py tests/test_evidence_fusion_gate.py
  ```

## Assumptions
- Stage7d 继续沿用 Stage7c 的 frozen evidence backbone，因为它已验证能稳定 evidence/gate 排序。
- Stage7d 不再把旧 `L_evidence_div` 当作主要多样性驱动，而是使用 evidence-scaled sigma 直接控制候选不确定性。
- wrong video / temporal shift 仍作为 harder stress test，仅观察，不作为 Stage7d 成功硬指标。
