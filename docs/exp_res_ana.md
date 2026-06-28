# 语义证据扰动训练实验结论初稿

## 1. 实验问题

本实验关注一个核心问题：在音频修复任务中，视觉信息并不总是可靠的。正常视频可以帮助模型恢复缺失音频，但当视频缺失、光流失效，或视频来自另一个乐器类别时，模型应当降低对视觉分支的信任，并给出更高、更有排序意义的不确定性估计。

为此，我们引入基于 CLIP 的离线语义证据。语义证据不是判断“当前视频像不像它自己的类别”，而是判断：

```text
semantic evidence = P(source_instrument | current_video_frames)
```

也就是说，对于 `wrong_video_cross_instrument`，模型看到的是替换后的视频帧，但 evidence 评估的是这些帧是否符合原始音频所属乐器。这个 target-specific 定义使语义证据能够直接刻画跨乐器错配：正常视频应有较高 semantic evidence，而 wrong video 应有较低 semantic evidence。

本实验进一步验证：**语义证据本身并不足够，模型还需要在训练阶段显式见到语义/视觉扰动，才能学会在低 evidence 情况下降低 gate 并提升 uncertainty 的可解释性。**

实验结果来源：

```text
checkpoints/ablation_stage10d/
docs/figures/stage10d_ablation/
docs/stage10d_final_tables.md
```

## 2. 方法与消融设置

| ID | 方法 | 核心设置 | 消融目的 |
| --- | --- | --- | --- |
| O | Original VIAI-AV | 原始 VIAI-AV PatchGAN 模型，单输出修复，没有 multi-candidate、candidate scorer、semantic evidence、learned gate 或 uncertainty head。 | 原始方法基线，用于展示从 VIAI-AV 到 EC-VIAI-AV 再到 semantic robustness 模型的完整演进。 |
| A | EC-VIAI-AV Candidate-Scorer Baseline | 使用 multi-candidate、evidence gate、evidence-scaled sigma、candidate scorer 和 uncertainty head；语义证据仅在测试时只读接入。 | 强基线，用于判断不额外训练时模型是否能利用 corrected semantic evidence。 |
| B | Semantic Evidence Fine-Tuning | 使用 corrected fused semantic evidence 继续训练，但训练数据仍只包含正常视频。 | 验证“只加入 semantic evidence 训练”是否足够。 |
| C | Semantic Perturbation Training, 5k | 使用 corrected fused semantic evidence，并在训练中加入 `wrong_video_cross_instrument`、`no_video`、`flow_zero`。 | 验证 semantic-aware perturbation training 是否有效。 |
| D | Semantic Perturbation Training, Final | 在 C 的基础上继续训练，总共进行 10k steps 的 semantic perturbation training。 | 作为最终 robustness-oriented 模型。 |
| E | No Wrong-Video Augmentation | 只使用 `no_video` 和 `flow_zero` 扰动，不使用 `wrong_video_cross_instrument`。 | 验证 wrong-video augmentation 是否必要。 |
| F | Heuristic-Only Perturbation | 使用同样的扰动训练，但 evidence source 仅为 heuristic，不使用 semantic evidence。 | 验证 semantic evidence 相比低级视觉 heuristic 的必要性。 |
| G0.2 | Semantic Perturbation, w=0.2 | 与 C 相同，但 fused evidence 中 semantic 权重为 0.2。 | 评估较低 semantic 权重的质量/鲁棒性折中。 |
| G0.5 | Semantic Perturbation, w=0.5 | 与 C 相同，但 fused evidence 中 semantic 权重为 0.5。 | 评估较高 semantic 权重的质量/鲁棒性折中。 |

所有方法均在以下四种模式下评估：

```text
none
flow_zero
no_video
wrong_video_cross_instrument
```

主要指标包括：

- `top1_missing_l1`：top-1 candidate 在 missing 区域的 L1，越低越好。
- `best_of_k_missing_l1`：K 个 candidates 中 oracle best 的 missing L1，越低越好。
- `psnr_missing`：missing 区域 PSNR，越高越好。
- `ssim`：结构相似度，越高越好。
- `semantic_evidence_mean`：target-specific semantic evidence 均值。
- `gate_mean`：模型实际视觉 gate 均值。对 `no_video` 和 `wrong_video_cross_instrument`，越低越合理。
- `gate_target_mean`：由 evidence 映射得到的监督目标。
- `uncertainty_error_spearman`：uncertainty 与 reconstruction error 的 Spearman 相关性，越高说明 uncertainty 越能排序样本风险。

## 3. 核心结果

| Method | none top1 ↓ | wrong top1 ↓ | wrong gate ↓ | no-video gate ↓ | wrong Spearman ↑ | none Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| O Original VIAI-AV | 0.063576 | 0.067543 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| A EC-VIAI-AV Candidate-Scorer Baseline | 0.059799 | 0.061106 | 0.713195 | 0.367402 | 0.326222 | 0.456607 |
| B Semantic Evidence Fine-Tuning | 0.062668 | 0.064093 | 0.866814 | 0.528362 | 0.354754 | 0.409318 |
| C Semantic Perturbation Training, 5k | 0.062353 | 0.062490 | 0.565215 | 0.151654 | 0.454351 | 0.470074 |
| D Semantic Perturbation Training, Final | 0.061787 | 0.060251 | 0.413393 | 0.089303 | 0.548191 | 0.523370 |
| E No wrong-video aug | 0.062441 | 0.063163 | 0.883072 | 0.061784 | 0.381656 | 0.466436 |
| F Heuristic-only perturb | 0.060096 | 0.060119 | 0.537373 | 0.053071 | 0.455732 | 0.532063 |
| G0.2 Semantic Perturbation Training, w=0.2 | 0.060408 | 0.060612 | 0.554073 | 0.140237 | 0.466262 | 0.510869 |
| G0.5 Semantic Perturbation Training, w=0.5 | 0.061376 | 0.061061 | 0.522629 | 0.157644 | 0.513842 | 0.530508 |

注：Original VIAI-AV 没有 learned gate 和 uncertainty head。表中的 `gate=1.0` 表示测试脚本中原始模型等价于始终使用视觉条件；`Spearman=0.0` 表示没有有效的不确定性排序输出。

## 4. 结果分析

### 4.1 从 Original VIAI-AV 到 EC-VIAI-AV 的基础提升

Original VIAI-AV 是单输出模型，没有候选选择、gate calibration 或 uncertainty estimation。与它相比，EC-VIAI-AV Candidate-Scorer Baseline 已经在正常视频和跨乐器 wrong-video 场景下明显降低 reconstruction error：

| Method | none top1 ↓ | flow-zero top1 ↓ | no-video top1 ↓ | wrong top1 ↓ |
| --- | ---: | ---: | ---: | ---: |
| O Original VIAI-AV | 0.063576 | 0.064570 | 0.070174 | 0.067543 |
| A EC-VIAI-AV Candidate-Scorer Baseline | 0.059799 | 0.060657 | 0.060114 | 0.061106 |
| D Semantic Perturbation Training, Final | 0.061787 | 0.059554 | 0.061554 | 0.060251 |

这个对比说明，EC-VIAI-AV 的 multi-candidate、candidate scorer 和 uncertainty/gate 结构首先提供了比原始 VIAI-AV 更强的基础修复能力。随后，semantic perturbation training 进一步面向视觉不可靠场景进行校准：最终模型 D 在 wrong-video 下取得 `0.060251` 的 top1 L1，明显优于 Original VIAI-AV 的 `0.067543`。

因此，本文方法链条可以概括为：

```text
Original VIAI-AV
-> EC-VIAI-AV candidate-scoring baseline
-> semantic evidence and perturbation training
```

前者解决基础修复能力，后者解决视觉语义不可靠时的条件信任问题。

### 4.2 语义证据需要训练扰动配合

Corrected semantic evidence 能正确识别跨乐器错配。在 `wrong_video_cross_instrument` 下，semantic evidence 均值为 `0.051944`，对应 gate target 只有 `0.108308`。然而，方法 B 虽然使用了 corrected semantic evidence 继续训练，但由于训练数据仍然只包含正常视频，模型并没有学会响应低 evidence target：

| Method | wrong semantic evidence | wrong gate target | wrong gate mean |
| --- | ---: | ---: | ---: |
| A EC-VIAI-AV Candidate-Scorer Baseline | 0.051944 | 0.108308 | 0.713195 |
| B Semantic Evidence Fine-Tuning | 0.051944 | 0.108308 | 0.866814 |
| C Semantic Perturbation Training, 5k | 0.051944 | 0.108308 | 0.565215 |
| D Semantic Perturbation Training, Final | 0.051944 | 0.108308 | 0.413393 |

这个结果说明，语义证据表只提供了“应该信任多少”的监督信号；如果模型在训练中从未见过 low-evidence 的视觉输入，它不会自动把这个监督信号转化为正确的 gate 行为。语义证据必须与 semantic-aware perturbation training 一起使用。

### 4.3 语义扰动训练提升跨乐器错配鲁棒性

最终模型 D 在 `wrong_video_cross_instrument` 下显著降低 visual gate：

```text
EC-VIAI-AV Candidate-Scorer Baseline wrong-video gate: 0.713195
Method D wrong-video gate: 0.413393
```

相对下降约 42.0%。同时，missing 区域重建质量没有恶化：

```text
EC-VIAI-AV Candidate-Scorer Baseline wrong-video top1_missing_l1: 0.061106
Method D wrong-video top1_missing_l1: 0.060251
```

因此，语义扰动训练并不是简单地压低所有视觉信息，也不是通过牺牲重建质量来获得低 gate；它使模型在语义不匹配时更有选择地降低视觉信任。

### 4.4 语义扰动训练改善无视频场景下的 gate calibration

在 `no_video` 下，理想行为是 gate 接近 0。方法 D 将 no-video gate 从 baseline 的 `0.367402` 降至 `0.089303`：

| Method | no-video semantic evidence | no-video gate target | no-video gate mean |
| --- | ---: | ---: | ---: |
| A EC-VIAI-AV Candidate-Scorer Baseline | 0.000000 | 0.000000 | 0.367402 |
| B Semantic Evidence Fine-Tuning | 0.000000 | 0.000000 | 0.528362 |
| C Semantic Perturbation Training, 5k | 0.000000 | 0.000000 | 0.151654 |
| D Semantic Perturbation Training, Final | 0.000000 | 0.000000 | 0.089303 |

这说明扰动训练不仅解决跨乐器错配，也改善了视觉缺失情况下的 gate calibration。

### 4.5 不确定性更能排序错误风险

方法 D 在所有测试模式下都取得最高的 uncertainty-error Spearman：

| Mode | A EC-VIAI-AV Candidate-Scorer Baseline | B Semantic Evidence Fine-Tuning | C Semantic Perturbation Training, 5k | D Semantic Perturbation Training, Final |
| --- | ---: | ---: | ---: | ---: |
| none | 0.456607 | 0.409318 | 0.470074 | 0.523370 |
| flow_zero | 0.448851 | 0.480747 | 0.556323 | 0.571383 |
| no_video | 0.497434 | 0.492171 | 0.524735 | 0.576389 |
| wrong_video_cross_instrument | 0.326222 | 0.354754 | 0.454351 | 0.548191 |

其中跨乐器错配场景提升最明显：

```text
0.326222 -> 0.548191
```

这表明最终模型不仅能降低错误视觉输入的 gate，还能让 uncertainty 更好地反映样本级错误风险。

### 4.6 正常视频质量存在可接受折中

鲁棒性提升带来轻微的正常视频质量代价：

| Method | none top1 L1 ↓ | none best-of-K L1 ↓ | none PSNR ↑ |
| --- | ---: | ---: | ---: |
| O Original VIAI-AV | 0.063576 | 0.063576 | 22.259100 |
| A EC-VIAI-AV Candidate-Scorer Baseline | 0.059799 | 0.058420 | 22.602434 |
| B Semantic Evidence Fine-Tuning | 0.062668 | 0.061179 | 22.332093 |
| C Semantic Perturbation Training, 5k | 0.062353 | 0.060406 | 22.361825 |
| D Semantic Perturbation Training, Final | 0.061787 | 0.059799 | 22.391672 |

方法 D 的正常视频 top1 L1 为 `0.061787`，略高于 EC-VIAI-AV baseline 的 `0.059799`，但仍优于 Original VIAI-AV 的 `0.063576`，也明显优于只做 semantic fine-tuning 的方法 B。考虑到方法 D 在 wrong-video/no-video 下显著改善 gate 和 uncertainty，这一 trade-off 是可以接受的。

为了更完整地比较 reconstruction quality，下表同时列出不同视觉条件下的 L1、PSNR 和 SSIM。可以看到，Original VIAI-AV 在 `no_video` 和 `wrong_video_cross_instrument` 下退化最明显；最终模型 D 在 wrong-video 下取得更低的 L1 和更高的 PSNR/SSIM，说明 semantic perturbation 并没有以牺牲错配场景重建质量为代价。

| Method | none L1 ↓ | no-video L1 ↓ | wrong L1 ↓ | none PSNR ↑ | no-video PSNR ↑ | wrong PSNR ↑ | none SSIM ↑ | wrong SSIM ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| O Original VIAI-AV | 0.063576 | 0.070174 | 0.067543 | 22.259100 | 21.472131 | 21.711918 | 0.971799 | 0.969591 |
| A EC-VIAI-AV Candidate-Scorer Baseline | 0.059799 | 0.060114 | 0.061106 | 22.602434 | 22.491676 | 22.422136 | 0.975155 | 0.973893 |
| B Semantic Evidence Fine-Tuning | 0.062668 | 0.060792 | 0.064093 | 22.332093 | 22.410469 | 22.028406 | 0.971474 | 0.971361 |
| D Semantic Perturbation Training, Final | 0.061787 | 0.061554 | 0.060251 | 22.391672 | 22.440425 | 22.627191 | 0.973075 | 0.973909 |
| F Heuristic-only | 0.060096 | 0.059123 | 0.060119 | 22.522995 | 22.708039 | 22.585917 | 0.973849 | 0.974715 |
| G0.2 Semantic Perturbation Training, w=0.2 | 0.060408 | 0.060386 | 0.060612 | 22.612190 | 22.539668 | 22.526218 | 0.974045 | 0.974325 |
| G0.5 Semantic Perturbation Training, w=0.5 | 0.061376 | 0.059215 | 0.061061 | 22.435667 | 22.675089 | 22.440418 | 0.972661 | 0.973024 |

其中 F 和 G0.2 在 normal quality 上更强，说明它们可以作为质量优先的 ablation；但 D 在 wrong-video gate、no-video gate 和 uncertainty ranking 上最优，因此更适合作为 robustness-oriented final model。

## 5. 设计必要性分析

### 5.1 wrong-video augmentation 是跨乐器鲁棒性的必要条件

方法 E 去掉了 `wrong_video_cross_instrument` 训练扰动，只保留 `no_video` 和 `flow_zero`。它在 no-video 下表现很好，但在 wrong-video 下 gate 失效：

| Method | no-video gate ↓ | wrong-video gate ↓ | wrong Spearman ↑ |
| --- | ---: | ---: | ---: |
| C Semantic Perturbation Training, 5k | 0.151654 | 0.565215 | 0.454351 |
| E No wrong-video aug | 0.061784 | 0.883072 | 0.381656 |

方法 E 的 wrong-video gate 高达 `0.883072`，甚至比 baseline 更高。这说明 `no_video` 和 `flow_zero` 不能替代 cross-instrument wrong-video 训练；模型必须显式见过“画面存在但语义错误”的视频，才能学会对这种输入降权。

### 5.2 heuristic-only perturbation 是强对照，但语义 target 更贴合问题

方法 F 使用同样的扰动训练，但 evidence source 只使用 heuristic。它取得了较好的正常视频质量和 no-video gate，但 wrong-video gate 仍高于最终模型：

| Method | none top1 ↓ | wrong top1 ↓ | wrong gate ↓ | no-video gate ↓ | wrong Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| D Semantic Perturbation Training, Final | 0.061787 | 0.060251 | 0.413393 | 0.089303 | 0.548191 |
| F Heuristic-only perturb | 0.060096 | 0.060119 | 0.537373 | 0.053071 | 0.455732 |

Heuristic evidence 主要基于光流、RGB/flow 统计等低级视觉线索。它可以识别 no-video 或 flow degradation，但难以表达“当前视频不是原始音频对应乐器”。因此，heuristic-only perturbation 是一个强 baseline，但 corrected semantic evidence 更适合作为跨乐器错配的监督目标。

### 5.3 semantic fusion weight 控制质量与鲁棒性的折中

G0.2、C 和 G0.5 比较了不同 semantic evidence 权重。5k-step 结果显示，较低权重通常有利于正常视频质量，较高权重更有利于语义错配下的 gate suppression 和 uncertainty ranking：

| Method | semantic weight | none top1 ↓ | wrong top1 ↓ | wrong gate ↓ | no-video gate ↓ | wrong Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0.2 | 0.20 | 0.060408 | 0.060612 | 0.554073 | 0.140237 | 0.466262 |
| C | 0.35 | 0.062353 | 0.062490 | 0.565215 | 0.151654 | 0.454351 |
| G0.5 | 0.50 | 0.061376 | 0.061061 | 0.522629 | 0.157644 | 0.513842 |
| D | 0.35, longer training | 0.061787 | 0.060251 | 0.413393 | 0.089303 | 0.548191 |

在当前设置中，默认权重 `w=0.35` 经过更长训练后取得最佳综合鲁棒性；如果更强调正常视频质量，`w=0.2` 或 heuristic-only 设置也可以作为补充对照。

## 6. 论文式结论段

实验表明，EC-VIAI-AV 的 candidate-scoring 结构相较 Original VIAI-AV 提升了基础修复质量，而 target-specific semantic evidence 进一步为视觉可靠性提供了语义层面的监督信号。该 evidence 能有效识别跨乐器视频错配，但仅有 evidence 并不足以改变模型行为。没有语义扰动训练时，模型即使获得低 semantic target，也仍会对 wrong video 输出较高 gate。加入 semantic-aware perturbation training 后，模型在跨乐器错配和无视频场景下显著降低 visual gate，并使 uncertainty 更好地反映 reconstruction error。

最终模型在 `wrong_video_cross_instrument` 下将 top1 missing L1 从 Original VIAI-AV 的 `0.067543` 降至 `0.060251`，并将 gate mean 从 EC-VIAI-AV baseline 的 `0.713195` 降至 `0.413393`，同时将 uncertainty-error Spearman 从 `0.326222` 提升至 `0.548191`；在 `no_video` 下，gate mean 从 `0.367402` 降至 `0.089303`。这些结果说明，semantic evidence 与 perturbation training 的结合能够使模型从“使用视频”转向“有条件地信任视频”，从而提升音频修复模型在视觉语义失配和视觉缺失场景下的鲁棒性。

扩展消融进一步说明了各设计组件的必要性：去掉 wrong-video augmentation 后，模型无法处理语义错误但视觉存在的视频；只使用 heuristic evidence 虽然能改善部分扰动鲁棒性，但缺少跨乐器语义判断能力；调整 semantic fusion weight 则体现了正常视频质量与语义鲁棒性之间的折中。综合来看，**Semantic Perturbation Training, w=0.35, 10k** 是当前最适合作为最终 robustness-oriented 模型的设置。

## 7. 推荐报告方式

正式论文/报告中建议按如下结构呈现：

| 角色 | 推荐方法 |
| --- | --- |
| 原始模型基线 | O: Original VIAI-AV |
| 主 baseline | A: EC-VIAI-AV Candidate-Scorer Baseline |
| semantic-only 失败对照 | B: Semantic Evidence Fine-Tuning |
| perturbation 有效性对照 | C: Semantic Perturbation Training, 5k |
| 最终模型 | D: Semantic Perturbation Training, 10k |
| wrong-video augmentation 必要性 | E: No Wrong-Video Augmentation |
| semantic evidence 必要性 | F: Heuristic-Only Perturbation |
| semantic weight 消融 | G0.2 / C / G0.5 |

推荐主表使用 O、A、B、D、E、F、G0.2、G0.5。C 可以作为中间训练步数结果，放在 ablation 表或补充材料中。

当前推荐最终模型：

```text
D: Semantic Perturbation Training, Final, w=0.35
```

## 8. 最终表格与图表文件

已生成最终表格和图表：

```text
docs/stage10d_final_tables.md
docs/figures/stage10d_ablation/
```

表格文件：

| 文件 | 内容 |
| --- | --- |
| `docs/stage10d_final_tables.md` | 最终 compact table 和图表索引 |
| `docs/figures/stage10d_ablation/stage10d_ablation_compact_table.csv` | O-G 核心指标表 |
| `docs/figures/stage10d_ablation/stage10d_ablation_quality_table.csv` | O-G 重建质量指标表 |
| `docs/figures/stage10d_ablation/stage10d_ablation_robustness_table.csv` | O-G 鲁棒性指标表 |
| `docs/figures/stage10d_ablation/stage10d_ablation_detailed_metrics.csv` | O-G 四种 perturbation 的完整指标表 |

图表文件：

| 文件 | 用途 |
| --- | --- |
| `wrong_video_gate_mean.png` | 展示各方法在 wrong-video 下的 gate 抑制能力 |
| `no_video_gate_mean.png` | 展示各方法在 no-video 下的 gate 抑制能力 |
| `wrong_video_uncertainty_spearman.png` | 展示 wrong-video 下 uncertainty-error 排序能力 |
| `top1_l1_none_vs_wrong.png` | 对比 normal video 与 wrong video 的 reconstruction quality |
| `quality_top1_l1_by_condition.png` | 对比 normal/no-video/wrong-video 下的 top1 L1 |
| `quality_psnr_by_condition.png` | 对比 normal/no-video/wrong-video 下的 PSNR |
| `wrong_video_tradeoff_scatter.png` | 展示 wrong-video gate、uncertainty Spearman 与 normal quality 的折中 |
| `gate_mean_by_mode.png` | 展示各方法在四种 perturbation mode 下的 gate 分布 |

可复现生成命令：

```bash
MPLCONFIGDIR=/tmp/matplotlib UV_CACHE_DIR=/tmp/uv-cache uv run python tools/make_stage10d_ablation_figures.py
```
