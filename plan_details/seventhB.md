# Stage 7b：Controlled Evidence Gate 修正计划

## Summary
当前 Stage7 证明了 naive gate 会塌缩到 `gate≈1`。下一版不再期待原始室内演奏数据天然提供大量低 evidence 样本，而是在训练中加入受控视觉退化，并用显式 gate-evidence loss 约束 gate。目标是验证：在同一样本的 controlled degradation 下，视觉运动证据越弱，gate 越低，候选 pairwise distance 越高。

## Key Changes
- 新增训练增强开关：
  - `--enable_visual_evidence_aug`
  - `--visual_evidence_aug_prob 0.5`
  - `--visual_evidence_aug_modes flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow`
- 增强只作用于训练阶段的 video/flow 输入，不改变 `mel_target`、mask、split 或测试默认路径。
- 新增 gate 监督：
  - `--lambda_gate_evidence 0.05`
  - `--evidence_gate_low 0.24`
  - `--evidence_gate_high 0.34`
  - `gate_target = clamp((evidence - low) / (high - low), 0, 1)`
  - `L_gate_evidence = SmoothL1(gate, gate_target.detach())`
- 保留现有 `L_evidence_div`，但建议默认实验把 `--lambda_diversity` 从 `0.1` 提到 `0.2`。
- 训练日志和测试 JSON/CSV 增加：
  - `loss_gate_evidence`
  - `weighted_loss_gate_evidence`
  - `gate_target_mean`
  - `gate_target_gap = gate_mean - gate_target_mean`
  - `visual_evidence_aug_mode` 的 batch 统计。
- PatchGAN 继续实验时，命令必须显式传 `--lambda_gan 0.001`，避免和 baseline/reference loss 权重不一致。

## Validation Protocol
- 从 stage6 checkpoint 重新训练，不继续使用 gate 已饱和的 stage7 checkpoint。
- 先跑 1000 steps 小实验：
  ```bash
  --enable_ec_viai_av --stochastic_adapter --enable_evidence_gate \
  --enable_visual_evidence_aug --visual_evidence_aug_prob 0.5 \
  --num_candidates 4 --test_num_candidates 4 \
  --lambda_min_k 1.0 --lambda_mean_k 0.1 --lambda_boundary 0.05 \
  --lambda_diversity 0.2 --lambda_gate_evidence 0.05 \
  --evidence_gate_low 0.24 --evidence_gate_high 0.34 \
  --lambda_gan 0.001
  ```
- 成功标准：
  - `gate_mean` 不再快速饱和到 `0.95+`。
  - `gate_target_gap` 不长期大幅为正。
  - `flow_zero/static_video_zero_flow` 的 `gate_mean` 明显低于 original。
  - 低 evidence 条件下 `candidate_pairwise_distance` 高于 original。
  - `mel_l1_missing` / `psnr_missing` 相比 stage6 不明显崩坏。
- 若 1000 steps 趋势成立，再跑 3000 steps；只有需要正式对照时再跑 5000 steps。

## Test Plan
- 单元测试：
  - visual evidence augmentation 不改变 audio/mel/mask shape。
  - `flow_zero`、`static_video_zero_flow` 生成正确。
  - `gate_target` 在 `[0,1]`，且 original evidence target 高于 flow_zero target。
  - `lambda_gate_evidence=0` 时总损失与旧 Stage7 等价。
- 脚本验证：
  - 扩展 `tools/validate_evidence_gate.py`，输出 original、flow_75、flow_50、flow_25、flow_zero、static_video_zero_flow 的 `evidence_mean / gate_mean / pairwise_distance`。
  - 暂时把 wrong video 和 temporal shift 标为 harder stress test，只观察，不作为 Stage7b 成功硬指标。
- 回归检查：
  - 不开 `--enable_visual_evidence_aug` 和 `--lambda_gate_evidence` 时，现有 Stage7 路径保持兼容。

## Assumptions
- 原始室内演奏视频之间 evidence 差异天然有限，因此主验证改为同一样本下的 controlled visual degradation。
- Stage3 evidence estimator 已验证能稳定响应 flow/静态/无运动退化；wrong video 和 temporal shift 不作为第一版 gate 的强约束目标。
- Stage7b 的论文表述改为“motion-evidence-aware fusion”，不声称第一版能可靠解决所有语义错配和细粒度时移。


## 实验
下面这套命令建议从 **stage6 checkpoint** 重新开始，不要接着已经 gate 饱和的 stage7 checkpoint 训练。

```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE6_CKPT=/path/to/stage6_checkpoint.pth.tar
export EXP_DIR=checkpoints/stage7b_controlled_gate_1k
```

先跑 1000 steps 小实验：

```bash
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.5 \
  --visual_evidence_aug_modes flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.2 \
  --lambda_gate_evidence 0.05 \
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

如果你的 stage6 checkpoint 是 step 24000，那么 `--max_train_steps 25000` 就是继续训练 1000 steps。训练时重点看这些 TensorBoard 指标：

```text
train/gate/mean
train/gate/target_mean
train/gate/target_gap
train/evidence/mean
train/candidate/pairwise_distance
train/evidence_diversity_gap
train/loss_gate_evidence
train/weighted_loss_gate_evidence
train/visual_evidence_aug/applied
```

训练完跑 gate 验证脚本：

```bash
python tools/validate_evidence_gate.py \
  --checkpoint "$EXP_DIR/latest.pth.tar" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --use_gan \
  --num_candidates 4 \
  --batch_size 1 \
  --num_workers 0
```

期望趋势：

```text
original gate_mean > flow_zero gate_mean
original gate_mean > static_video_zero_flow gate_mean
flow_zero/static_video_zero_flow 的 candidate_pairwise_distance 高于 original
gate_target_gap 不长期大幅为正
gate_mean 不再快速冲到 0.95+
```

如果 1000 steps 趋势成立，再跑 3000 steps，把目录和步数改掉即可：

```bash
export EXP_DIR=checkpoints/stage7b_controlled_gate_3k

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.5 \
  --visual_evidence_aug_modes flow_75,flow_50,flow_25,flow_zero,static_video_zero_flow \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.2 \
  --lambda_gate_evidence 0.05 \
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
  --max_train_steps 27000 \
  --print_freq 20 \
  --save_latest_freq 500 \
  --save_epoch_freq 1000 \
  --display_id 0
```

这里假设 stage6 是 step 24000，所以 `27000` 对应继续训练 3000 steps。