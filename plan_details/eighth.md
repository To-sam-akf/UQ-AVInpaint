# 第 8 步：Candidate Scorer 与 Uncertainty Calibration

## Summary
- 在现有 Stage7d 多候选链路上新增显式 `CandidateScorer` 和 `UncertaintyHead`，默认关闭，只有 `--enable_ec_viai_av --stochastic_adapter --enable_candidate_scorer` 时启用。
- `self.mel_pred` 采用用户确认的 hard argmax：`argmax(pi_k)` 选出的 raw candidate；`self.mel_completed_pred` 同步保存 mask-compose 后结果。
- scorer 推理输入只使用可在测试时得到的 proxy/feature；真实 target error 只用于训练监督和测试指标，不做信息泄漏。

## Public API / Interfaces
- `base_options.py` 新增：
  - `--enable_candidate_scorer`，默认 `False`
  - `--calib_error_tau 0.1`，用于把 error 映射到 `[0,1]` difficulty target
- `networks/EC_VIAI_Modules.py` 新增：
  - `CandidateScorer.forward(candidate_stats, audio_bottleneck, video_feature, evidence) -> logits [B,K], pi [B,K]`
  - `UncertaintyHead.forward(audio_bottleneck, video_feature, uncertainty_stats) -> u [B,1]`
- `VIAIAVModel` 新增公开张量/标量：
  - `candidate_logits [B,K]`
  - `candidate_pi [B,K]`
  - `candidate_top1_index [B]`
  - `uncertainty_score [B,1]`
  - `top1_missing_l1`, `candidate0_missing_l1`, `random_expected_missing_l1`, `oracle_gain`
  - `loss_candidate_scorer`, `loss_uncertainty_calib`, `loss_calib`, `weighted_loss_calib`

## Implementation Changes
- `CandidateScorer` 每个候选输入固定为：
  - `missing_input_l1_proxy`: missing 区域内 `|candidate - mel_input_interpolation|` 均值
  - `boundary_jump`: completed mel 在左右缺失边界的无监督跳变均值
  - `sync_score`: 将 completed candidate 重新过 `Mel_Encoder` 得到 bottleneck，与 `video_feature` 的 normalized L2 sync score；计算时临时 `eval()` 并 `no_grad()`，随后恢复 encoder 训练状态
  - `audio_context_feature`: pooled `mel_features[-1]`
  - `evidence_score`: `[B,1]` broadcast 到 K
- `CandidateScorer` 用 MLP 输出 logits，`softmax(dim=1)` 得到 `pi_k`；最后一层初始化为 0，使启用初期 argmax 默认落到 candidate 0，保持 Stage6/7d 稳定性。
- `UncertaintyHead` 输入 pooled audio/video feature 加全局统计：`entropy(pi)`、`max(pi)`、top1 proxy、pairwise distance、evidence、gate、sigma_scale，输出 sigmoid `u [B,1]`。
- `_forward_inpainter()` 在生成 `mel_candidates` / `mel_completed_candidates` 后调用 `_score_and_select_candidates()`：
  - 未启用 scorer：`pi[:,0]=1`、`top1_index=0`、`u=0`，行为等价当前代码。
  - 启用 scorer：用 `argmax(pi)` gather raw/top1 completed candidate，更新 `self.mel_pred` 和 `self.mel_completed_pred`。
- `_multi_candidate_losses()` 保存 per-sample `candidate_missing_l1 [B,K]`，并计算 top1/candidate0/random-expected/best-of-K/oracle-gain 指标。
- 新增 `_calibration_losses()`：
  - `best_idx = argmin(candidate_missing_l1.detach())`
  - `loss_candidate_scorer = cross_entropy(candidate_logits, best_idx)`
  - `best_error = min_k candidate_missing_l1.detach()`
  - `difficulty = 1 - exp(-best_error / calib_error_tau)`
  - `loss_uncertainty_calib = SmoothL1(uncertainty_score, difficulty)`
  - `weighted_loss_calib = lambda_calib * (loss_candidate_scorer + loss_uncertainty_calib)`
- `loss_total` 增加 `weighted_loss_calib`；若 `lambda_calib > 0` 但未启用 `--enable_candidate_scorer`，直接报错。
- optimizer/checkpoint/stage：
  - scorer/head 参数加入 `optimizer_G`
  - checkpoint 可选保存/加载 `CandidateScorer`、`UncertaintyHead`
  - `_stage_name()` 在启用 scorer 时返回 `EC-VIAI-AV-stage8-candidate-scorer-calib`
- `train_viai_av.py` / TensorBoard 增加 scorer/calibration loss、top1/oracle/uncertainty 标量。
- `test_viai_av.py` JSON/CSV 增加：
  - `top1_missing_l1`
  - `candidate0_missing_l1`
  - `random_expected_missing_l1`
  - `best_of_k_missing_l1`
  - `oracle_gain = top1_missing_l1 - best_of_k_missing_l1`
  - `uncertainty_error_corr` / `uncertainty_error_spearman` for `u` vs top1 error
  - `uncertainty_best_error_corr` / `uncertainty_best_error_spearman` for `u` vs min-K difficulty
- Pearson/Spearman 用 NumPy 实现，不新增 SciPy 依赖；样本数不足或方差为 0 时返回 `0.0` 并记录 correlation count。

## Test Plan
- 静态检查：
  - `python -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py base_options.py train_viai_av.py test_viai_av.py`
  - `pytest -q tests/test_multi_candidate_losses.py tests/test_evidence_fusion_gate.py`
- 新增单元测试：
  - `CandidateScorer` 输出 shape 正确，`pi.sum(dim=1)==1`
  - zero-init logits 下 hard argmax 选择 candidate 0
  - `UncertaintyHead` 输出 `[B,1]` 且在 `[0,1]`
  - synthetic candidates 中 scorer CE target 等于真实 best candidate
  - `oracle_gain >= 0`，`best_of_k_missing_l1 <= top1/random/candidate0` 在构造样本上成立
  - `lambda_calib=0` 时新增 loss 不改变总损失
- 云端 smoke：
  - 从 Stage7d checkpoint 跑 `--enable_candidate_scorer --lambda_calib 0.1 --num_candidates 4`
  - 检查 `pi_k` 行和为 1、`u.shape == [B,1]`、loss finite、checkpoint 可保存/加载
- 云端 validation：
  - 比较 `top1_missing_l1`、`candidate0_missing_l1`、`random_expected_missing_l1`
  - 确认 `oracle_gain` 为正或接近 0
  - 计算 `uncertainty_error_corr` 与 Spearman，目标为正相关

## Assumptions
- 第 8 步默认不改 Stage6/7c/7d 旧命令行为；必须显式传 `--enable_candidate_scorer` 才启用新 top-1 选择。
- hard argmax 是默认输出策略；soft-weighted candidate 可作为内部诊断张量保留，但不写入 `self.mel_pred`。
- `lambda_calib` 同时控制 scorer CE 和 uncertainty calibration，避免再引入一个重复 loss 权重。


## 训练和验证
去云端做 **Stage8 smoke + short validation**。建议别直接长训，先确认 scorer/calibration 链路真的稳定。

先从 Stage7d checkpoint 继续训一个短实验：

```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE7D_CKPT=/path/to/stage7d_checkpoint.pth.tar
export EXP_DIR=checkpoints/stage8_candidate_scorer_smoke

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
  --resume_path "$STAGE7D_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$EXP_DIR" \
  --log_event_path "$EXP_DIR/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 100 \
  --checkpoint_interval 100 \
  --print_freq 10 \
  --display_id 0
```

重点看这些 TensorBoard/日志指标：

```text
loss_candidate_scorer
loss_uncertainty_calib
loss_calib
weighted_loss_calib
candidate/top1_missing_l1
candidate/candidate0_missing_l1
candidate/random_expected_missing_l1
candidate/best_of_k_missing_l1
candidate/oracle_gain
candidate/pi_entropy
candidate/pi_max
uncertainty/mean
```

如果 100 steps 没 NaN、`pi` 没明显异常，再跑测试：

```bash
export STAGE8_CKPT=$EXP_DIR/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar

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
  --lambda_calib 0.1 \
  --resume_path "$STAGE8_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir checkpoints/stage8_candidate_scorer_test \
  --display_id 0
```

成功标准先定得朴素一点：

- `top1_missing_l1` 优于或接近 `candidate0_missing_l1`
- `top1_missing_l1` 优于 `random_expected_missing_l1`
- `oracle_gain = top1 - best_of_k` 为正但逐步变小
- `uncertainty_error_corr` / Spearman 为正
- normal 质量不要比 Stage7d 明显崩坏

如果这些成立，再把 `--max_train_steps` 提到继续训练 1000 steps，然后做正式 Stage7d vs Stage8 对照。


下面给你两组：**继续训练 Stage8** 和 **验证 Stage8 checkpoint**。你只需要把 `DATA_ROOT` 和 `STAGE8_CKPT` 改成你云端真实路径。

**1. 继续训练 Stage8**

建议先继续到 `1000` 或 `3000` steps。这里示例用 `3000`：

```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE8_CKPT=/root/EC-ViAv-vgpu/checkpoints/stage8_candidate_scorer_smoke/EC-VIAI-AV-PatchGAN_checkpoint_step000027100.pth.tar
export EXP_DIR=checkpoints/stage8_candidate_scorer_3k

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
  --resume_path "$STAGE8_CKPT" \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$EXP_DIR" \
  --log_event_path "$EXP_DIR/events" \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 3000 \
  --checkpoint_interval 500 \
  --print_freq 10 \
  --display_id 0
```

注意：这里**不要加 `--reset_optimizer`**，因为你是从 Stage8 smoke 继续训。  
如果你是从 Stage7d checkpoint 第一次启动 Stage8，则才加 `--reset_optimizer`。

**2. 验证训练后的 Stage8 checkpoint**

比如训练后 checkpoint 是：

```bash
export STAGE8_CKPT=checkpoints/stage8_candidate_scorer_3k/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export DATA_ROOT=/root/shared-nvme/data
export RESULT_DIR=checkpoints/stage8_candidate_scorer_3k_test
```

验证命令：

```bash
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
  --resume_path "$STAGE8_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir "$RESULT_DIR" \
  --display_id 0
```

验证后重点看：

```text
top1_missing_l1
candidate0_missing_l1
best_of_k_missing_l1
oracle_gain
loss_candidate_scorer
uncertainty_error_corr
uncertainty_error_spearman
uncertainty_best_error_corr
uncertainty_best_error_spearman
```

判断标准：`top1_missing_l1` 最好要小于或接近 `candidate0_missing_l1`，uncertainty correlation 保持正数并变大。当前 short validation 已经说明方向可行，下一轮主要看 scorer 能不能真正超过 candidate0。