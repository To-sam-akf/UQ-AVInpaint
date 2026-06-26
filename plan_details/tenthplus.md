# Stage10 Semantic Evidence 升级计划

## Summary
- 当前 Stage6 正式训练保持不变，不接入语义模块，不修改已在跑的 multi-candidate 实验。
- 新增一个 **Stage10: Semantic Evidence Sidecar**：用冻结 CLIP 类视觉-文本模型离线预计算“视频是否像目标乐器/演奏场景”的语义分数，再作为可选 evidence 来源接入 gate、sigma、diversity、scorer。
- 目标不是替换现有 flow/sync heuristic evidence，而是增加语义旁路，让模型在 `wrong_video_cross_instrument`、无乐器画面、语义不匹配时更会降权视频。

## Key Changes
- 新增离线脚本 `tools/precompute_semantic_evidence.py`：
  - 读取 `train_av_split.txt` / `val_av_split.txt` / `test_av_split.txt`。
  - 从每个 sample 的 `image_crop/` 均匀抽 8 帧。
  - 从路径推断目标乐器：`processed/<instrument>/...`。
  - 用 frozen CLIP 计算 instrument prompt 相似度。
  - 输出 JSONL：`$DATA_ROOT/semantic_evidence/clip_vit_b32/<split>.jsonl`。
- JSONL 每行固定字段：
  - `sample_dir`
  - `instrument`
  - `semantic_score`，范围 `[0,1]`
  - `target_prob`
  - `top1_instrument`
  - `top1_prob`
  - `target_rank`
  - `frame_consistency`
  - `num_frames`
- 新增可选参数：
  - `--evidence_source {none,heuristic,semantic,fused}`
  - `--semantic_evidence_path PATH`
  - `--semantic_evidence_weight 0.35`
  - `--semantic_missing_score 0.0`
- 兼容规则：
  - 默认 `evidence_source=none` 保持当前行为，等价于现有 heuristic evidence。
  - `semantic` 只用离线语义分数。
  - `fused` 使用：`e = clamp((1-w) * heuristic_e + w * semantic_e, 0, 1)`，默认 `w=0.35`。
- 模型接入：
  - 在 `VIAIAVModel.set_inputs()` 保存 `path_batch` 后，通过 sample path 查 semantic evidence。
  - 新增日志：`semantic_evidence/mean`、`heuristic_evidence/mean`、`evidence/fused_mean`。
  - `self.evidence_score` 继续作为 gate、sigma、diversity、candidate scorer 的唯一入口，避免改动下游模块。
- 测试扰动兼容：
  - `wrong_video*` 时 semantic lookup 必须使用 wrong video 的 sample_dir，而不是原始 audio sample_dir。
  - `no_video` 时 semantic score 设为 `0.0`。
  - `flow_zero`、`blur`、`frame_drop`、`temporal_shift` 默认沿用原 sample semantic score，因为语义 identity 没变。

## Training And Experiment Plan
- Stage6：
  - 继续当前 `stochastic_adapter + K=4 + minK/meanK/boundary` 正式训练。
  - 不加 semantic 参数，不改 checkpoint。
- Stage10-A 离线验证：
  - 对 train/val/test split 预计算 semantic JSONL。
  - 在 Stage9 per-sample metrics 上关联分析：`semantic_score` vs `top1_missing_l1`、`best_of_k_missing_l1`、`uncertainty`、`wrong_video_cross_instrument`。
- Stage10-B 只读接入验证：
  - 用已有 Stage8/full checkpoint 跑测试，不训练。
  - 对比 `heuristic`、`semantic`、`fused` 三种 evidence source 在 `none / no_video / wrong_video_cross_instrument` 下的 evidence 分布。
- Stage10-C 正式训练：
  - 从最佳 Stage8 checkpoint 开始。
  - 命令核心新增：
    ```bash
    --evidence_source fused \
    --semantic_evidence_path "$DATA_ROOT/semantic_evidence/clip_vit_b32/train_av_split.jsonl" \
    --semantic_evidence_weight 0.35
    ```
  - 保持 Stage8 其他参数不变：`--enable_evidence_gate --enable_evidence_scaled_sigma --enable_candidate_scorer`。
- 消融实验：
  - `EC-VIAI-AV full`
  - `+ semantic only`
  - `+ heuristic only`
  - `+ fused semantic evidence`
  - `fused w=0.2 / 0.35 / 0.5`

## Test Plan
- Unit tests：
  - semantic JSONL lookup 能匹配普通 sample path 和带 trailing start index 的 path。
  - `evidence_source=none` 时输出与当前 heuristic evidence 一致。
  - `fused` 公式输出在 `[0,1]`。
  - 缺失 semantic 记录时使用 `semantic_missing_score=0.0`，不 crash。
- Script smoke test：
  - 用 2-5 个 sample 运行 semantic precompute，确认 JSONL 字段齐全。
  - 对同一 batch 跑 `heuristic/semantic/fused`，确认 shape 都是 `[B,1]`。
- Perturbation tests：
  - `wrong_video_cross_instrument` 使用 wrong video 的 semantic score。
  - `no_video` semantic score 为 0。
  - 原始 video 的平均 semantic score 应高于 cross-instrument wrong video，若不高则记录为 CLIP 语义信号失败，不进入 Stage10-C 正式训练。
- Acceptance criteria：
  - Stage6 结果不受影响。
  - Stage10-B 中 `fused` evidence 在 `none` 下高于 `wrong_video_cross_instrument/no_video`。
  - Stage10-C 中 original video 的 `top1_missing_l1` 不明显差于 Stage8，wrong/no video 下 `uncertainty_mean` 更高，`gate_mean` 更低。

## Assumptions
- 默认选择：Stage6 不动，语义升级放到 Stage10。
- 默认语义源：frozen CLIP 类模型，作为可选依赖和离线预计算工具，不进入主训练反向传播。
- 当前环境依赖里没有 CLIP/open_clip，因此实现时应放到 optional dependency，不影响普通 VIAI-AV/EC-VIAI-AV 训练。
- MUSICES sample path 可以从 `processed/<instrument>/...` 推断 instrument label。


## 训练测试
下面按“先不影响 Stage6，单独做 Stage10 语义旁路”的顺序来跑。

**1. 安装可选 CLIP 依赖**
```bash
cd /home/sanmu/VIAIpro

uv sync --extra semantic
```

云端如果不用 `uv`，可以：

```bash
python -m pip install open_clip_torch
```

**2. 设置路径**
```bash
export DATA_ROOT=/root/shared-nvme/data
export REPO=/root/EC-ViAv-vgpu
cd "$REPO"
```

**3. 先做小规模 semantic evidence smoke test**
```bash
uv run python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name test_av_split.txt \
  --limit 5 \
  --num-frames 8 \
  --batch-size 8
```

检查输出：

```bash
head -n 2 "$DATA_ROOT/semantic_evidence/clip_vit_b32/test_av_split.jsonl"
```

每行应该有：

```text
sample_dir, instrument, semantic_score, target_prob, top1_instrument, top1_prob, target_rank, frame_consistency, num_frames
```

**4. 正式预计算 train/val/test**
```bash
uv run python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --split-name val_av_split.txt \
  --split-name test_av_split.txt \
  --num-frames 8 \
  --batch-size 32
```

输出路径：

```bash
$DATA_ROOT/semantic_evidence/clip_vit_b32/train_av_split.jsonl
$DATA_ROOT/semantic_evidence/clip_vit_b32/val_av_split.jsonl
$DATA_ROOT/semantic_evidence/clip_vit_b32/test_av_split.jsonl
```

**5. Stage10-B：先只测试，不训练**
用已有 Stage8/full checkpoint 跑三种 evidence source 对比：

```bash
export STAGE8_CKPT=/path/to/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export TEST_ROOT=checkpoints/stage10_semantic_eval
```

```bash
for SOURCE in heuristic semantic fused
do
  for MODE in none no_video wrong_video_cross_instrument
  do
    uv run python main.py test-viai-av -- \
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
      --evidence_source "$SOURCE" \
      --semantic_evidence_path "$DATA_ROOT/semantic_evidence/clip_vit_b32/test_av_split.jsonl" \
      --semantic_evidence_weight 0.35 \
      --resume_path "$STAGE8_CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$TEST_ROOT/$SOURCE/$MODE" \
      --video_perturbation "$MODE" \
      --display_id 0
  done
done
```

快速看结果：

```bash
find "$TEST_ROOT" -name "*_test.json" -print | sort | while read f
do
  uv run python - <<PY
import json
p="$f"
d=json.load(open(p))
print(
    p,
    "mode=", d.get("video_perturbation"),
    "source=", d.get("evidence_source"),
    "e=", round(d.get("evidence_mean", 0), 4),
    "heur=", round(d.get("heuristic_evidence_mean", 0), 4),
    "sem=", round(d.get("semantic_evidence_mean", 0), 4),
    "gate=", round(d.get("gate_mean", 0), 4),
    "u=", round(d.get("uncertainty_mean", 0), 4),
    "top1=", round(d.get("top1_missing_l1", 0), 6),
)
PY
done
```

**6. Stage10-C：正式训练 fused semantic evidence**
从最佳 Stage8 checkpoint 开始：

```bash
export STAGE10_DIR=checkpoints/formal_ec_viai_av/stage10_semantic_fused

uv run python main.py train-viai-av -- \
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
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.1 \
  --calib_error_tau 0.1 \
  --evidence_source fused \
  --semantic_evidence_path "$DATA_ROOT/semantic_evidence/clip_vit_b32/train_av_split.jsonl" \
  --semantic_evidence_weight 0.35 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10_DIR" \
  --log_event_path "$STAGE10_DIR/events" \
  --batch_size 4 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 50000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

**7. 注意**
Stage6 正在跑的话，不要改它的命令；Stage10 是从 Stage8/full checkpoint 之后单独开新实验。测试时用 `test_av_split.jsonl`，训练时用 `train_av_split.jsonl`。