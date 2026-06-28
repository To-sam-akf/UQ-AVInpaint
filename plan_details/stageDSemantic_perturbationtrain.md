# Stage10-D Semantic Perturbation Training 计划

## Summary

Stage10-95000 失败原因是训练只见过正常视频，虽然 corrected semantic evidence 能把 `wrong_video_cross_instrument` 压到低分，但 gate 没学会在训练中响应这种低 target。Stage10-D 目标是在训练阶段加入 semantic-aware 视频扰动，让模型实际见到 `wrong_video_cross_instrument / no_video / flow_zero`，并学习低 evidence 时降低 gate、提高 uncertainty，同时尽量不伤正常重建。

## Key Changes

- 扩展训练时 visual evidence augmentation：
  - 在现有 `--enable_visual_evidence_aug` / `--visual_evidence_aug_modes` 基础上新增支持模式：`no_video`、`wrong_video_cross_instrument`。
  - 默认 Stage10-D 使用：
    ```text
    --enable_visual_evidence_aug
    --visual_evidence_aug_prob 0.35
    --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero
    ```
  - `flow_zero` 保持现有逻辑。
  - `no_video` 将 video/flow 置零，并设置 semantic override 为 `0.0`。
  - `wrong_video_cross_instrument` 从 train split 中按 source instrument 选择另一个 instrument 的视频，替换当前 batch 的 video/flow，并设置：
    ```text
    semantic path = wrong video path
    semantic target instrument = source instrument
    ```

- 新增训练用 wrong-video sampler：
  - 将测试脚本里的 `WrongVideoSampler` 思路抽成可复用工具，供 train/test 共用，避免重复逻辑。
  - 训练 sampler 使用 `train_split_name`，测试仍使用 `test_split_name`。
  - wrong-video 选择保持确定性：对每个 source sample 用 split index 映射到下一个 instrument 的 sample，避免随机噪声导致实验不可复现。
  - 若 train split 少于两个 instrument，启动时报错。

- 训练策略：
  - 从 Stage8-90000 重新开始，不使用失败的 Stage10-95000 checkpoint。
  - Stage10-D 先训 5k step：`90000 -> 95000`。
  - 使用 corrected fused semantic evidence：
    ```text
    --evidence_source fused
    --semantic_evidence_weight 0.35
    ```
  - 提高 gate supervision：
    ```text
    --lambda_gate_evidence 0.2
    ```
  - 其余 Stage8 参数保持不变：candidate scorer、evidence-scaled sigma、K=4、minK/meanK/boundary/diversity/calib 均沿用。

- 日志与 CSV：
  - 训练日志继续记录 `visual_evidence_aug/*`。
  - 新增 augmentation mode 统计：`wrong_video_cross_instrument`、`no_video`。
  - per-sample 测试 CSV 已有 `semantic_target_instrument`，继续保留用于诊断。

## Experiment Protocol

- Stage10-D 训练命令核心参数：
  ```bash
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --evidence_source fused \
  --semantic_evidence_path "$TRAIN_SEM" \
  --semantic_evidence_weight 0.35 \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.35 \
  --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero \
  --lambda_gate_evidence 0.2 \
  --lr 0.00002 \
  --max_train_steps 95000
  ```

- Stage10-D-95000 测试：
  - 使用 corrected `TEST_SEM`。
  - 跑四组：
    ```text
    none
    flow_zero
    no_video
    wrong_video_cross_instrument
    ```
  - 重点比较 Stage8-90000、Stage10-B read-only、失败 Stage10-95000、Stage10-D-95000。

- 如果 Stage10-D-95000 满足条件，再继续到 100000；否则不继续训练，回到参数调整。

## Test Plan

- Unit tests：
  - `visual_evidence_aug_modes` 接受 `wrong_video_cross_instrument`、`no_video`、`flow_zero`。
  - train-time wrong-video sampler 能从 train split 为每个 source sample 找到 cross-instrument wrong sample。
  - wrong-video augmentation 后：
    ```text
    video/flow 被替换
    semantic_evidence_paths == wrong video paths
    semantic_evidence_target_instruments == source instruments
    ```
  - no-video augmentation 后：
    ```text
    video/flow 全零
    semantic_evidence_override == 0.0
    ```
  - 原有 flow/static augmentation 行为不变。

- Smoke tests：
  - 用 tiny split 或 1 batch 跑 Stage10-D train smoke，确认不 crash。
  - 检查 TensorBoard/日志里 augmentation mode 占比：
    ```text
    applied ≈ 0.35
    wrong/no_video/flow_zero 约各占三分之一
    ```
  - 检查训练中 `semantic_evidence/mean` 在 augmented batch 时低于 normal batch。

- Acceptance criteria：
  - `wrong_video_cross_instrument gate_mean` 明显低于 Stage8-90000 的 `0.728`，目标优先 `< 0.55`。
  - `no_video gate_mean` 低于失败 Stage10-95000 的 `0.530`，目标优先 `< 0.45`。
  - `none top1_missing_l1` 不明显差于 Stage8-90000 的 `0.0597`，容忍上限约 `0.0610`。
  - `wrong_video semantic_evidence_mean` 保持约 `0.052`，说明 corrected target lookup 没回退。
  - `wrong_video uncertainty_mean` 高于 `none`，且 uncertainty Spearman 保持正相关。

## Assumptions

- Stage10-D 从 Stage8-90000 开始，废弃失败的 Stage10-95000 作为训练起点。
- 继续使用 CLIP corrected semantic evidence，不引入 ImageBind/AudioCLIP。
- 融合公式暂时保持 `(1-w)*heuristic + w*semantic`，先通过训练扰动解决 gate 不响应问题。
- augmentation 概率默认 `0.35`，避免过度扰动导致正常重建质量下降。
- 如果 Stage10-D-95000 仍然 gate 不降，再考虑第二轮调参：`lambda_gate_evidence=0.3`、`visual_evidence_aug_prob=0.5` 或 semantic-only gate auxiliary loss。


## 训练测试
`tests/test_stage10d_semantic_perturbation.py` 是 **pytest 单元测试**，不是可执行的命令行脚本。它包含 3 个测试：

| 测试函数 | 验证内容 |
|:---|---|
| `test_visual_evidence_aug_modes_accept_stage10d_modes` | augmentation modes 能正确解析 `wrong_video_cross_instrument,no_video,flow_zero` |
| `test_no_video_visual_evidence_aug_zeros_video_and_flow` | no_video 扰动后 video/flow 全零 |
| `test_train_wrong_video_sampler_picks_cross_instrument_sample` | train-time wrong-video sampler 能为 cello sample 找到 flute 的 wrong video |

运行方式：

```bash
cd ~/EC-ViAv-vgpu
python -m pytest tests/test_stage10d_semantic_perturbation.py -v
```

---

**实际的 Stage10-D 训练命令**（来自 `stageDSemantic_perturbationtrain.md` 第53-65行）：

```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
export TRAIN_SEM=$DATA_ROOT/semantic_evidence/clip_vit_b32/train_av_split.jsonl
export STAGE10D_DIR=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_source fused \
  --semantic_evidence_path "$TRAIN_SEM" \
  --semantic_evidence_weight 0.35 \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.35 \
  --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.2 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10D_DIR" \
  --log_event_path "$STAGE10D_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 95000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

启动日志必须看到：

```text
enable_visual_evidence_aug=True
visual_evidence_aug_prob=0.35
visual_evidence_aug_modes=wrong_video_cross_instrument,no_video,flow_zero
lambda_gate_evidence=0.2
```

**训练完成后测试命令**（四种扰动）：

```bash
export STAGE10D_CKPT=/root/EC-ViAv-vgpu/checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar

export TEST_SEM=$DATA_ROOT/semantic_evidence/clip_vit_b32/test_av_split.jsonl
export TEST_ROOT=checkpoints/stage10d_eval_95000

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  python test_viai_av.py \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --enable_evidence_gate \
    --freeze_gate_evidence_backbone \
    --enable_evidence_scaled_sigma \
    --enable_candidate_scorer \
    --evidence_source fused \
    --semantic_evidence_path "$TEST_SEM" \
    --semantic_evidence_weight 0.35 \
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.2 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume_path "$STAGE10D_CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --results_dir "$TEST_ROOT/$MODE" \
    --video_perturbation "$MODE" \
    --calibration_bins 10 \
    --display_id 0
done
```