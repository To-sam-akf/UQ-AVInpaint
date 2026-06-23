# 第 7 步：Evidence-Aware Fusion Gate

## Summary
在现有 EC-VIAI-AV stage6 基础上新增 evidence-aware gate。默认不启用，baseline/stage5/stage6 行为保持不变；启用 `--enable_ec_viai_av --enable_evidence_gate` 后，模型用视觉证据分数调节视频特征，并用 `L_evidence_div` 约束低 evidence 样本产生更高候选多样性。

## Public API / Interfaces
- 在 `networks/EC_VIAI_Modules.py` 新增：
  ```python
  class EvidenceFusionGate(nn.Module):
      def forward(self, audio_bottleneck, video_feature, evidence):
          # returns video_feature_calibrated, gate, learned_audio_prior
  ```
- 新增参数：
  - `--evidence_diversity_d_min 0.02`
  - `--evidence_diversity_alpha 0.08`
- 复用已有 `--lambda_diversity` 作为 `L_evidence_div` 权重。
- checkpoint 可选新增 key：`EvidenceFusionGate`；老 checkpoint 缺少该 key 时保持新模块初始化。

## Implementation Changes
- `EvidenceFusionGate`：
  - 对 `audio_bottleneck` 和 `video_feature` 做 adaptive avg pool，拼接 `[audio_pool, video_pool, evidence]`。
  - 用 `MLP(513 -> 256 -> 1)` 计算 `g = sigmoid(...)`，输出 shape `[B,1,1,1]`。
  - `learned_audio_prior` 采用用户选定的 audio-conditioned prior：`audio_bottleneck` 经 trainable `1x1 Conv` 投影到 video feature 同形状，Conv 初始化为 identity。
  - 输出 `video_feature_calibrated = g * video_feature + (1 - g) * learned_audio_prior`。
- `Models/VIAI_AV_inpainting.py`：
  - 新增 `self.use_evidence_gate = enable_ec_viai_av and enable_evidence_gate`。
  - gate 参数只在启用时加入 `optimizer_G`。
  - `_forward_inpainter()` 中保留 raw `video_feature` 用于 evidence/sync 统计；启用 gate 后，将 calibrated video feature 传给 `BottleneckAdapter` 和 `MelDecoderImage`。
  - probe branch 继续使用 `mel_target_features[-1]`，不走 gate。
  - 新增 `L_evidence_div = mean(abs(pairwise_distance_per_sample - (d_min + alpha * (1 - evidence))))`。
  - pairwise distance 只在 missing region 计算；`K < 2` 时为 0，若 `lambda_diversity > 0` 且候选数小于 2，直接报清晰错误。
  - `loss_multi_candidate` 扩展为原 min-K/mean-K/boundary 加 `lambda_diversity * L_evidence_div`。
- 记录项：
  - TensorBoard：`gate/mean`、`gate/min`、`gate/max`、`candidate/pairwise_distance`、`evidence_diversity_gap`、`loss_evidence_div`、`weighted_loss_evidence_div`。
  - `train_viai_av.py` / `test_viai_av.py` totals、progress、summary 增加 gate/diversity 标量。
  - `test_viai_av.py` JSON/CSV 增加 `loss_evidence_div`、`weighted_loss_evidence_div`、`candidate_pairwise_distance`、`evidence_diversity_gap`、`gate_mean`、`evidence_diversity_d_min`、`evidence_diversity_alpha`。
- 新增轻量验证脚本 `tools/validate_evidence_gate.py`：
  - 加载 checkpoint 和 test split。
  - 对同一音频构造 original、flow_zero、wrong_video、temporal_shift。
  - 打印每种条件下的 `evidence_mean`、`gate_mean`、`candidate_pairwise_distance`。

## Test Plan
- 本地静态/单元测试：
  ```bash
  uv run python -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py train_viai_av.py test_viai_av.py base_options.py
  uv run pytest -q tests/test_multi_candidate_losses.py tests/test_evidence_fusion_gate.py
  ```
- 新增单元测试覆盖：
  - gate 输出 shape 与 video feature 一致，`g` 在 `[0,1]`。
  - audio-conditioned prior identity 初始化可用，gate 参与反向传播。
  - `L_evidence_div` 目标满足低 evidence target 大于高 evidence target。
  - `candidate_pairwise_distance` 只统计 missing region。
  - `lambda_diversity=0` 时总损失不变。
- 云端 smoke：
  ```bash
  uv run python main.py train-viai-av -- \
    --enable_ec_viai_av --stochastic_adapter --enable_evidence_gate \
    --num_candidates 4 --test_num_candidates 4 \
    --lambda_min_k 1.0 --lambda_mean_k 0.1 --lambda_boundary 0.05 \
    --lambda_diversity 0.1 \
    --resume --resume_path "$STAGE6_CKPT" --reset_optimizer \
    --data_root "$DATA_ROOT" --train_split_name train_av_split.txt --val_split_name val_av_split.txt \
    --batch_size 1 --num_workers 0 --max_train_steps 1 --print_freq 1 --display_id 0
  ```
- 验证脚本：
  ```bash
  uv run python tools/validate_evidence_gate.py \
    --checkpoint "$STAGE7_CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --num_candidates 4
  ```
  期望 original 的 `gate_mean` 高于 flow_zero/wrong_video/temporal_shift；低 evidence 样本的 `candidate_pairwise_distance` 高于高 evidence 样本。

## Assumptions
- 已采用用户确认的默认：audio-conditioned prior，`d_min=0.02`、`alpha=0.08`。
- `lambda_diversity` 专用于第 7 步的 `L_evidence_div`，不新增重复 loss 权重。
- `--video_perturbation` 的完整测试协议仍留到第 9 步；第 7 步只增加专用验证脚本。
- 当前本地已通过语法检查和现有多候选 loss 测试；完整训练/鲁棒性验证在云端运行。

## 训练验证
第 7 步云端验证重点不是“能不能跑”，而是看论文故事是否成立：

- original video: `gate_mean` 较高，pairwise distance 较低
- flow_zero / wrong_video / temporal_shift: `gate_mean` 降低
- low evidence 条件下 `candidate_pairwise_distance` 更高
- original 上质量不要明显掉

下面给你一套直接跑的命令。

**1. 设置变量**
```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE6_CKPT=/path/to/ec_viai_av_stage6_checkpoint.pth.tar
export OUT_DIR=checkpoints/ec_viai_av_stage7_gate_k4
```

如果你的 checkpoint 是 PatchGAN 版本，下面所有 `train-viai-av` / `test-viai-av` / `validate_evidence_gate.py` 命令都保留 `--use_gan`；如果不是 PatchGAN，就删掉 `--use_gan`。

**2. Stage7 训练**
建议从第 6 步稳定 checkpoint 继续训，不从零开始：

```bash
uv run python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.1 \
  --evidence_diversity_d_min 0.02 \
  --evidence_diversity_alpha 0.08 \
  --resume \
  --resume_path "$STAGE6_CKPT" \
  --reset_optimizer \
  --use_gan \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$OUT_DIR" \
  --log_event_path "$OUT_DIR/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 5000 \
  --checkpoint_interval 500 \
  --print_freq 20 \
  --display_id 0
```

看 TensorBoard：

```bash
uv run tensorboard --logdir "$OUT_DIR/events" --port 6006
```

重点看这些曲线：

```text
train/gate/mean
val/gate/mean
train/candidate/pairwise_distance
val/candidate/pairwise_distance
train/evidence_diversity_gap
val/evidence_diversity_gap
train/loss_evidence_div
val/loss_evidence_div
train/loss_min_k
train/loss_mean_k
train/loss_boundary
```

**3. 跑 checkpoint 测试**
把 checkpoint 路径换成实际 step：

```bash
export STAGE7_CKPT=$OUT_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000005000.pth.tar
```

```bash
uv run python main.py test-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.1 \
  --evidence_diversity_d_min 0.02 \
  --evidence_diversity_alpha 0.08 \
  --use_gan \
  --resume_path "$STAGE7_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir checkpoints/ec_viai_av_stage7_gate_k4_test \
  --display_id 0
```

测试 JSON/CSV 里重点看：

```text
gate_mean
candidate_pairwise_distance
evidence_diversity_gap
loss_evidence_div
best_of_k_missing_l1
mean_k_missing_l1
mel_l1_missing
psnr_missing
ssim
```

**4. 核心故事验证：视频扰动对比**
这是第 7 步最重要的验证：

```bash
uv run python tools/validate_evidence_gate.py \
  --checkpoint "$STAGE7_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --num_candidates 4 \
  --batch_size 1 \
  --num_workers 0 \
  --blank_frames 40 \
  --temporal_shift 5 \
  --use_gan
```

你希望看到类似趋势：

```text
original:      evidence_mean 高, gate_mean 高, candidate_pairwise_distance 低
flow_zero:     evidence_mean 低, gate_mean 低, candidate_pairwise_distance 高
wrong_video:   gate_mean 低于 original
temporal_shift gate_mean 低于 original
```

不是要求每个样本都严格满足，但整体趋势要成立。建议对多个 checkpoint 跑：

```bash
for ckpt in "$OUT_DIR"/EC-VIAI-AV-PatchGAN_checkpoint_step*.pth.tar; do
  echo "===== $ckpt ====="
  uv run python tools/validate_evidence_gate.py \
    --checkpoint "$ckpt" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --num_candidates 4 \
    --batch_size 1 \
    --num_workers 0 \
    --blank_frames 40 \
    --temporal_shift 5 \
    --use_gan
done
```

**5. 消融对比：不开 gate**
用同一个 stage6 checkpoint 训练一个 no-gate 对照，步数保持一致：

```bash
export OUT_DIR_NOGATE=checkpoints/ec_viai_av_stage7_nogate_k4
```

```bash
uv run python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.1 \
  --evidence_diversity_d_min 0.02 \
  --evidence_diversity_alpha 0.08 \
  --resume \
  --resume_path "$STAGE6_CKPT" \
  --reset_optimizer \
  --use_gan \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$OUT_DIR_NOGATE" \
  --log_event_path "$OUT_DIR_NOGATE/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 5000 \
  --checkpoint_interval 500 \
  --print_freq 20 \
  --display_id 0
```

这组主要比较普通 test 指标和 `candidate_pairwise_distance`。真正完整的 perturbation 鲁棒性批量测试会在第 9 步扩展。第 7 步先用 `validate_evidence_gate.py` 证明 gate 行为方向是对的。
