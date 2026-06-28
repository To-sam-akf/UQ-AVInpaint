# Stage10-D Semantic Perturbation Training Results

## Summary

Stage10-D 的目标是修复 Stage10-C 的失败点：corrected semantic evidence 已经能把
`wrong_video_cross_instrument` 的语义分数压低，但模型在正常视频上训练时没有见过低
semantic evidence 的坏视频，因此 gate 没学会关闭。

本阶段从 Stage8-90000 重新开始，加入训练时 semantic-aware perturbation：

```text
wrong_video_cross_instrument,no_video,flow_zero
```

最终结果表明：Stage10-D 成功让 gate 对 `no_video` 和
`wrong_video_cross_instrument` 明显降权，并提升 uncertainty 排序能力；代价是正常
`none` 重建质量有轻微下降。

## Training Setup

起点 checkpoint：

```text
checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
```

Stage10-D 训练目录：

```text
checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected
```

关键训练参数：

```bash
--evidence_source fused
--semantic_evidence_weight 0.35
--enable_visual_evidence_aug
--visual_evidence_aug_prob 0.35
--visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero
--lambda_gate_evidence 0.2
--lambda_calib 0.05
--lr 0.00002
```

Semantic evidence 使用 corrected lookup：

```text
normal video:
  semantic_score = P(source_instrument | original_video)

wrong_video_cross_instrument:
  semantic_score = P(source_instrument | wrong_video)
```

## Corrected Semantic Evidence Diagnosis

`tools/diagnose_semantic_evidence.py` 在 Stage10-D 测试集上的诊断结果：

```text
matched = 376
missing = 0
original_source_score_mean = 0.746694
wrong_self_score_mean = 0.689679
wrong_source_score_mean = 0.051944
wrong_top1_equals_wrong_instrument = 314 / 376
wrong_top1_equals_source_instrument = 13 / 376
```

解释：

- CLIP 能正确识别 wrong video 自己的乐器，`wrong_self_score_mean=0.689679`。
- corrected lookup 后，wrong video 对 source instrument 的分数很低，`wrong_source_score_mean=0.051944`。
- 只有 `13/376` 个 wrong video 的 top1 被误判为 source instrument，主要集中在相似管乐器，如 `saxophone/trumpet/tuba/flute`。

这说明 semantic evidence 本身已经修正成功。

## Stage10-D Results

### Step 95000

| perturbation | top1 L1 | best-of-k L1 | evidence | semantic | gate mean | gate target | uncertainty | Spearman | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.060798 | 0.059416 | 0.471147 | 0.746694 | 0.766591 | 0.900631 | 0.406399 | 0.520428 | 22.594556 |
| flow_zero | 0.060397 | 0.058480 | 0.410464 | 0.746694 | 0.625260 | 0.851467 | 0.415201 | 0.524521 | 22.609168 |
| no_video | 0.060381 | 0.058098 | 0.135282 | 0.000000 | 0.152721 | 0.000000 | 0.425762 | 0.614799 | 22.561280 |
| wrong_video_cross_instrument | 0.062194 | 0.059386 | 0.213473 | 0.051944 | 0.566348 | 0.108308 | 0.421923 | 0.460743 | 22.355970 |

95000 结论：

- `no_video gate_mean=0.152721`，已经明显学会关闭。
- `wrong_video gate_mean=0.566348`，相比 Stage8-90000 的约 `0.728` 明显下降，但还略高于目标 `<0.55`。
- `none top1=0.060798`，正常重建质量轻微下降但仍可接受。

### Step 100000

| perturbation | top1 L1 | best-of-k L1 | evidence | semantic | gate mean | gate target | uncertainty | Spearman | PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 0.061787 | 0.059799 | 0.471147 | 0.746694 | 0.685346 | 0.900631 | 0.417021 | 0.523370 | 22.391672 |
| flow_zero | 0.059554 | 0.057383 | 0.410464 | 0.746694 | 0.589253 | 0.851467 | 0.425839 | 0.571383 | 22.744617 |
| no_video | 0.061554 | 0.058588 | 0.135282 | 0.000000 | 0.089303 | 0.000000 | 0.429910 | 0.576389 | 22.440425 |
| wrong_video_cross_instrument | 0.060251 | 0.057581 | 0.213473 | 0.051944 | 0.413393 | 0.108308 | 0.430720 | 0.548191 | 22.627191 |

100000 结论：

- `wrong_video gate_mean=0.413393`，明显优于 95000，也显著优于 Stage8-90000。
- `no_video gate_mean=0.089303`，低 evidence 下 gate 关闭非常明确。
- `wrong_video top1=0.060251`，比 95000 的 `0.062194` 明显改善。
- `uncertainty_error_spearman` 在所有 perturbation 下均超过 `0.52`，uncertainty 排序能力强。
- 代价是 `none top1=0.061787`，相比 Stage8-90000 的 `0.059706` 和 Stage10-D-95000 的 `0.060798` 有正常输入退化。

## Key Comparisons

### Gate Response

| checkpoint | none gate | no_video gate | wrong_video gate |
|---|---:|---:|---:|
| Stage8-90000 | 0.822830 | 0.382400 | 0.728142 |
| Failed Stage10-95000 | 0.926190 | 0.529982 | 0.865662 |
| Stage10-D-95000 | 0.766591 | 0.152721 | 0.566348 |
| Stage10-D-100000 | 0.685346 | 0.089303 | 0.413393 |

Stage10-D 成功解决了失败 Stage10-C 的核心问题：模型终于学会在低 semantic evidence 下关闭 gate。

### Normal Reconstruction Trade-off

| checkpoint | none top1 L1 |
|---|---:|
| Stage8-90000 | 0.059706 |
| Stage10-D-95000 | 0.060798 |
| Stage10-D-100000 | 0.061787 |

Stage10-D-100000 的鲁棒性最好，但正常输入重建质量下降更明显。Stage10-D-95000 是质量和鲁棒性的折中点。

## Final Checkpoint Recommendation

建议保留两个 checkpoint：

```text
Stage10-D-95000:
  quality/robustness trade-off checkpoint

Stage10-D-100000:
  semantic robustness checkpoint
```

如果论文主表只能选择一个 Stage10 checkpoint，推荐使用：

```text
checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
```

理由：

- Stage10 的核心贡献是 semantic evidence robustness，不是单纯刷 `none` top1。
- 100000 将 `wrong_video gate_mean` 从 Stage8 的 `0.728142` 降到 `0.413393`。
- 100000 将 `no_video gate_mean` 降到 `0.089303`。
- 100000 的 uncertainty ranking 在所有模式下均保持较强正相关。

推荐实验表述：

```text
Stage10-D improves semantic perturbation robustness: wrong-video gate decreases
from 0.728 in Stage8 to 0.413, no-video gate decreases to 0.089, and uncertainty
Spearman remains above 0.52 across all perturbations, with a modest normal
reconstruction trade-off.
```

## Notes

- Stage10-D 证明 corrected semantic evidence 需要配合 perturbation training 才能驱动 gate 学会关闭。
- 单独使用 corrected semantic evidence 做 read-only 测试只能改变 target/evidence，不能保证 gate 跟随。
- 后续如果希望降低 `none` 退化，可以尝试：
  - 降低 `visual_evidence_aug_prob` 到 `0.25`。
  - 使用 95000 作为主 checkpoint，100000 作为 robustness ablation。
  - 对 normal batch 增加轻量 gate target regularization，避免 `none gate` 被过度压低。
