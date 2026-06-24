# 拆分 Wrong Video 扰动模式

## Summary
- 将当前 `wrong_video` 拆成两种更可解释的测试条件：
  - `wrong_video_any`：保留当前行为，同 split 内确定性错配任意另一个样本。
  - `wrong_video_cross_instrument`：优先选择不同乐器的视频，增强错误视频的语义区分度。
- 为了兼容旧命令，保留 `wrong_video` 作为 `wrong_video_any` 的别名。
- instrument 从 split 第 0 列样本路径解析，当前格式如 `processed/accordion/...`，即第二段路径为乐器名。

## Key Changes
- `base_options.py` 扩展 `--video_perturbation` choices：
  `none, blur, flow_zero, frame_drop, temporal_shift, wrong_video, wrong_video_any, wrong_video_cross_instrument, no_video`。
- `test_viai_av.py` 重构 `WrongVideoSampler`：
  - 建立 `sample_dir -> index`、`sample_dir -> instrument`、`instrument -> sample_dirs` 映射。
  - `wrong_video` 和 `wrong_video_any` 使用当前 `+1` deterministic mismatch。
  - `wrong_video_cross_instrument` 对每个样本按 sorted instrument 顺序选择下一个不同 instrument 的样本，再在该 instrument 内用当前样本 index 做 deterministic offset。
  - 若 test split 只有一个 instrument，`wrong_video_cross_instrument` 直接报错，避免悄悄退化成 same-instrument wrong video。
- 输出记录增加诊断字段：
  - `wrong_video_effective_mode`
  - `wrong_video_cross_instrument_available`
  - `wrong_video_num_instruments`
- per-sample CSV 增加：
  - `source_instrument`
  - `wrong_video_instrument`
  - `wrong_video_sample_path`
  - `wrong_video_is_cross_instrument`

## Test Plan
- 单元测试：
  - 路径 `processed/accordion/...` 能解析出 `accordion`。
  - `wrong_video_any` 与旧 `wrong_video` 选择结果一致。
  - 多 instrument split 下，`wrong_video_cross_instrument` 总是选择不同 instrument。
  - 单 instrument split 下，`wrong_video_cross_instrument` 抛出清晰错误。
  - JSON/CSV 新字段可写出并被 `coerce_csv_record()` 正确读取。
- 静态检查：
  `uv run python -m py_compile base_options.py test_viai_av.py`
- 回归测试：
  `uv run pytest -q tests/test_stage9_eval_protocol.py`
- 云端验证：
  分别跑 `wrong_video_any` 和 `wrong_video_cross_instrument`，比较 evidence/retrieval/uncertainty；论文中将 cross-instrument 作为主 wrong-video 鲁棒性实验，same/any 作为 harder diagnostic 或 appendix。

## Assumptions
- split 第 0 列路径稳定形如 `<processed_dir>/<instrument>/<youtube_id>/...`。
- `wrong_video_cross_instrument` 不做乐器语义识别，只利用数据目录标签进行 deterministic 错配。
- 保留旧 `wrong_video` 别名，避免已有脚本失效；新论文实验命令优先使用 `wrong_video_cross_instrument`。


## 测试验证
下面这套直接给你云端跑第 9 步测试协议。你只需要改 `DATA_ROOT`、`STAGE8_CKPT`、`RESULT_BASE`。

**1. 上传代码到云端**

本地执行：

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='data/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pth.tar' \
  --exclude='*.Zone.Identifier' \
  --exclude='.agents/' \
  --exclude='.codex' \
  --exclude='.python-version' \
  --exclude='NUL' \
  --exclude='uv.lock' \
  --exclude='VIAI.pdf' \
  -e "ssh -p 2233" \
  ~/VIAIpro/ \
  root@ackcs-00gjh78a@ssh.bj8.bz1.paratera.com:/root/EC-ViAv-vgpu/
```

**2. 云端先检查 test split 是否有多乐器**

```bash
cd /root/EC-ViAv-vgpu
export DATA_ROOT=/root/shared-nvme/data

cut -d'|' -f1 "$DATA_ROOT/test_av_split.txt" | awk -F'/' '{print $2}' | sort | uniq -c
```

如果只看到一种乐器，`wrong_video_cross_instrument` 会正常报错；这时要换更完整的 test split。

**3. 基础变量**

```bash
cd /root/EC-ViAv-vgpu

export DATA_ROOT=/root/shared-nvme/data
export STAGE8_CKPT=/root/EC-ViAv-vgpu/checkpoints/stage8_candidate_scorer_lowlr_sweep/EC-VIAI-AV-PatchGAN_checkpoint_step000027750.pth.tar
export RESULT_BASE=checkpoints/stage9_eval_protocol
```

**4. 单个扰动测试命令模板**

先跑 `none`：

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
  --results_dir "$RESULT_BASE/none" \
  --video_perturbation none \
  --display_id 0
```

**5. 批量跑 8 种模式**

```bash
for MODE in none blur flow_zero frame_drop temporal_shift wrong_video_any wrong_video_cross_instrument no_video
do
  echo "===== Running $MODE ====="
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
    --results_dir "$RESULT_BASE/$MODE" \
    --video_perturbation "$MODE" \
    --video_blur_kernel 9 \
    --video_frame_drop_stride 2 \
    --video_temporal_shift_frames 6 \
    --calibration_bins 10 \
    --display_id 0
done
```

**6. 保存候选 Mel 图**

建议先只对少量结果或最终 checkpoint 开启，因为会写很多图：

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
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_calib 0.1 \
  --resume_path "$STAGE8_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir "$RESULT_BASE/candidate_images_cross_instrument" \
  --video_perturbation wrong_video_cross_instrument \
  --save_candidates \
  --display_id 0
```

输出位置大致是：

```text
$RESULT_BASE/.../mel-image/stepXXXXXXXXX/...
$RESULT_BASE/.../mel-candidates/stepXXXXXXXXX/perturb-wrong_video_cross_instrument/candidate_00/
$RESULT_BASE/.../sample-metrics/
$RESULT_BASE/.../risk-coverage/
$RESULT_BASE/.../calibration/
```

**7. 保存候选 wav，可选**

这个会慢，建议先限制样本数：

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
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_calib 0.1 \
  --resume_path "$STAGE8_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir "$RESULT_BASE/wav_cross_instrument" \
  --video_perturbation wrong_video_cross_instrument \
  --save_candidates \
  --use_vocoder \
  --vocoder_n_iter 32 \
  --vocoder_max_samples 5 \
  --display_id 0
```

**8. 快速汇总关键指标**

```bash
find "$RESULT_BASE" -name "*_test.json" -print | sort | while read f
do
  python - <<PY
import json
p="$f"
d=json.load(open(p))
print(
    d["video_perturbation"],
    "top1=", round(d.get("top1_missing_l1", 0), 6),
    "bestK=", round(d.get("best_of_k_missing_l1", 0), 6),
    "u=", round(d.get("uncertainty_mean", 0), 6),
    "evidence=", round(d.get("evidence_mean", 0), 6),
    "pairwise=", round(d.get("candidate_pairwise_mel_l1", 0), 6),
    "wrong_mode=", d.get("wrong_video_effective_mode", "")
)
PY
done
```

重点看：

```text
none vs blur/flow_zero/no_video:
  evidence 是否下降，uncertainty/pairwise 是否上升

wrong_video_any vs wrong_video_cross_instrument:
  cross_instrument 是否比 any 更明显影响 evidence/retrieval/uncertainty

best_of_k_missing_l1 <= top1_missing_l1:
  K 候选 oracle 上限是否成立
```