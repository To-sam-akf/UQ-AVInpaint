# Stage10 Semantic Evidence 修正计划：从“视频自分类分数”改为“视频对 Source Instrument 的条件分数”

## Summary

当前 Stage10-B 暴露的问题不是 CLIP 识别失败，而是 semantic evidence 的取分目标错了：wrong video 使用了 wrong video 自己 instrument 的分数，导致 `accordion video -> accordion prompt` 得高分。修改目标是让 semantic evidence 表示：

```text
P(source_instrument | current_video_frames)
```

也就是：当前视频是否像原始 audio/sample 的目标乐器。正常视频仍应高分，`wrong_video_cross_instrument` 应明显低分，`no_video` 仍为 0。

## Key Changes

- 扩展 semantic JSONL schema：
  - 保留现有字段：`sample_dir`、`instrument`、`semantic_score`、`target_prob`、`top1_instrument`、`top1_prob`、`target_rank`、`frame_consistency`、`num_frames`。
  - 新增 `probs_by_instrument`：保存该视频帧对所有 instrument prompt 的平均概率。
  - 新增 `frame_top1_instruments`：保存 8 帧各自 top1 instrument，用于诊断 target-specific frame consistency。
  - 旧字段 `semantic_score` 仍表示视频自身 `instrument` 的概率，保证旧 JSONL 可读。

- 修改 semantic lookup 逻辑：
  - `SemanticEvidenceTable.lookup_score(path, target_instrument=None)`。
  - 如果 `target_instrument` 存在且 JSONL 有 `probs_by_instrument`，返回 `probs_by_instrument[target_instrument]`。
  - 如果没有 `target_instrument`，保持旧行为，返回 `semantic_score`。
  - 如果目标 instrument 缺失或记录缺失，返回 `semantic_missing_score`，不 crash。
  - 新增批量接口 `lookup_scores(paths, target_instruments=None)`。

- 修改模型 semantic evidence 接入：
  - `VIAIAVModel.set_semantic_evidence_paths(paths, target_instruments=None)`。
  - 正常训练/测试时：
    - `paths = original sample paths`
    - `target_instruments = infer_instrument_from_sample_dir(original sample paths)`
  - `wrong_video_cross_instrument` 测试时：
    - `paths = wrong_video_sampler.last_wrong_dirs`
    - `target_instruments = wrong_video_sampler.last_source_instruments`
  - `no_video` 仍使用 `set_semantic_evidence_override(0.0)`。
  - `flow_zero / blur / frame_drop / temporal_shift` 沿用原 sample path 和 source instrument。

- 增加诊断工具或测试输出：
  - 在 per-sample CSV 中保留 `semantic_evidence`，并可额外加入 `semantic_target_instrument`。
  - 增加一个小型诊断脚本/命令逻辑，用 Stage10-B CSV 关联 JSONL，输出：
    - original semantic mean
    - wrong video semantic mean using source target
    - wrong video top1 是否等于 wrong instrument
    - source target rank 分布
  - Stage10-C 之前必须先重跑 Stage10-B，确认 corrected semantic 能压低 wrong video。

## Test Plan

- Unit tests：
  - 新 JSONL 中 `probs_by_instrument={"cello":0.8,"flute":0.1}` 时：
    - `lookup_score(cello_path)` 返回旧 `semantic_score`。
    - `lookup_score(cello_path, target_instrument="flute")` 返回 `0.1`。
  - 旧 JSONL 没有 `probs_by_instrument` 时保持兼容，不报错。
  - trailing start index path 仍能匹配。
  - missing target instrument 返回 `semantic_missing_score`。
  - `wrong_video_cross_instrument` 扰动后：
    - semantic lookup path 是 wrong video path。
    - semantic target instrument 是 source instrument。
  - `no_video` semantic score 仍为 0。

- Smoke tests：
  - 用 `--limit 5` 重新预计算 JSONL，确认新增字段存在。
  - 对一个 `source=xylophone, wrong=accordion` 样本，检查：
    - `probs_by_instrument["accordion"]` 高。
    - `probs_by_instrument["xylophone"]` 低。
  - 重跑 Stage10-B `semantic/fused/heuristic`，重点看：
    - `semantic none evidence` 明显高于 `semantic wrong_video_cross_instrument evidence`。
    - `fused wrong_video evidence` 低于 `fused none evidence`。
    - `wrong_video gate_target` 不再接近 0.85。

- Acceptance criteria：
  - `none` 下 semantic evidence 保持高。
  - `no_video` semantic evidence 为 0。
  - `wrong_video_cross_instrument` semantic evidence 显著低于 `none`。
  - corrected fused 不会把 wrong video 训练成高可信视频。
  - 只有满足以上条件后，再进入 Stage10-C 正式训练。

## Assumptions

- 默认字段名使用 `probs_by_instrument`，与当前诊断结论保持一致。
- 旧 semantic JSONL 继续可读，但 Stage10-C 必须使用重新预计算的新 JSONL。
- 当前仍使用 CLIP video-to-text label evidence，不引入 ImageBind/AudioCLIP；audio-video embedding 相似度留到 Stage10+。
- 融合公式暂时保持现有 `fused = (1-w)*heuristic + w*semantic`，先修正 semantic target；如重跑 Stage10-B 后 wrong video 仍偏高，再考虑 `min()` 融合或降低 `semantic_evidence_weight`。
