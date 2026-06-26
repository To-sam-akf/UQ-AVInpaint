## stage 6
### 50000 steps 实验与结果
```
step    top1      bestK     oracle    pairwise   meanK     loss     psnr_m
20000   0.071956  0.071795  0.000161  0.000644  0.071944  0.6003   21.429
25000   0.080241  0.079461  0.000780  0.036125  0.106197  0.5655   21.212
30000   0.072566  0.071605  0.000961  0.019332  0.081373  0.5138   21.474
35000   0.066118  0.064988  0.001130  0.027449  0.078676  0.4969   21.973
40000   0.075961  0.072832  0.003129  0.025618  0.084288  0.5097   21.309
45000   0.068953  0.067256  0.001696  0.023121  0.077901  0.5037   21.753
50000   0.066078  0.063609  0.002469  0.019889  0.070951  0.4864   21.969
```
Stage6 已经达到目标：
1. 多候选没有塌缩；
2. best-of-K 明显优于 top1；
3. mean-K 没有严重崩坏；
4. reconstruction 质量保持较好。

推荐 Stage6 final checkpoint:
EC-VIAI-AV-PatchGAN_checkpoint_step000050000.pth.tar

| perturbation | top1 / missing L1 | best_of_k | PSNR missing | evidence | sync | retrieval |
|---|---:|---:|---:|---:|---:|---|
| `none` | **0.0670** | 0.0644 | 21.93 | 0.323 | 0.336 | 正常 |
| `flow_zero` | **0.0640** | 0.0616 | 22.18 | 0.229 | 0.408 | 明显下降 |
| `no_video` | **0.0743** | 0.0738 | 20.96 | 0.208 | 0.573 | 崩得最明显 |
| `wrong_video_cross_instrument` | **0.0674** | 0.0654 | 21.80 | 0.300 | 0.446 | retrieval 几乎崩 |

1. Stage6 主任务质量已经够用，可以结束
none 下：
top1_missing_l1 = 0.067018
best_of_k = 0.064360
oracle_gain = 0.002658
pairwise = 0.020582
这比前面 30000/40000 更稳。50000 不是完美，但已经是你目前 Stage6 里最适合作为下一阶段起点的 checkpoint。
2. multi-candidate 确实学到了一点多样性
oracle_gain = 0.002658，说明 4 个 candidate 里确实偶尔有比 top1 更好的结果。
但是这个 gain 还不大，说明 Stage6 只是把候选机制跑通了，还没有真正学会“根据不确定性选更可靠的候选”。这正好是 Stage8 candidate scorer 要解决的。
3. 最大问题：wrong video 没被有效惩罚
这是最关键的发现。
wrong_video_cross_instrument 的 retrieval 已经几乎崩了：
audio→video R@1: 4.52 -> 0.00
video→audio R@1: 3.19 -> 0.27
说明视频和音频确实语义/匹配关系错了。
但生成质量几乎没怎么变：
none top1 = 0.067018
wrong_video top1 = 0.067422
也就是说：模型在 Stage6 里还没有强依赖正确视频语义。它可能更多靠 audio/local context/reconstruction prior，而不是理解“这个视频是不是对应这个乐器”。
同时 heuristic evidence 也暴露了短板：
none evidence = 0.323
wrong_video evidence = 0.300
只降了一点点。
这说明当前手工 evidence 对 no_video、flow_zero 有反应，但对 cross-instrument wrong video 不敏感。这和你前面判断完全一致：它没有语义级理解能力。
另外，gate_mean=1.0、uncertainty=0.0 现在不用紧张，因为这次是 Stage6 测试：
enable_evidence_gate=False
enable_evidence_scaled_sigma=False
enable_candidate_scorer=False
所以 gate/uncertainty 还没有真正启用。

### 总结

第6步可以认为验证通过。

本阶段达成了三个核心目标：

- `L_minK` 生效：K 个候选中至少有一个候选能比 candidate 0 更贴近真实缺失片段。
- `L_meanK` 起到基本约束：平均候选质量没有完全崩坏，但仍明显弱于 oracle best candidate。
- `L_boundary` 生效：训练曲线中 boundary loss 后半段下降，mel 图中的 completed 边界没有明显灾难性断裂。

本阶段也暴露了两个后续问题：

- 候选池仍需要 scorer。当前最终输出仍然是 candidate 0，而不是 oracle best candidate；`best_of_k_missing_l1` 的收益还不能自动转化为 top-1 输出收益。
- 候选平均质量仍需改善。`mean_k_missing_l1` 高于 candidate 0 和 best-of-K，说明部分随机候选仍偏弱，后续需要 candidate scorer、evidence gate 或更合理的 diversity/calibration 约束。

因此，第6步的结论是：multi-candidate 训练方向成立，best-of-K 已经产生可观收益，completed mel 的测试质量可接受。可以进入第7步 Evidence-Aware Fusion Gate，但在论文叙事中应明确：第6步证明的是候选池 oracle 上限提升，第8步 candidate scorer 才负责把 oracle 候选收益转化为模型可用的 top-1 输出。

另外，当前 checkpoint 记录中的 stage 仍显示为：

```text
EC-VIAI-AV-stage5-stochastic-adapter
```

但该 checkpoint 实际已经加入第6步 multi-candidate loss。后续建议更新 `_stage_name()` 或 checkpoint metadata，避免实验记录中 stage5/stage6 混淆。

---

## stage 7
80000 的结果：**比 70000 好，但整体仍不如 60000 适合作为 Stage7 最终 checkpoint。建议 Stage7 最终选 60000。**

**60000 / 70000 / 80000 对比**

| 模式 | 60000 top1 | 70000 top1 | 80000 top1 | 最好 |
|---|---:|---:|---:|---|
| `none` | **0.060592** | 0.061993 | 0.060991 | 60000 |
| `flow_zero` | 0.060438 | 0.060343 | **0.060107** | 80000 微好 |
| `no_video` | **0.059681** | 0.059900 | 0.060786 | 60000 |
| `wrong_video_cross_instrument` | **0.061584** | 0.063773 | 0.063621 | 60000 |

80000 在 `flow_zero` 上稍好，但 `none / no_video / wrong_video` 都不如 60000。尤其 `wrong_video`，80000 还是偏差：

```text
60000 wrong_video top1 = 0.061584
80000 wrong_video top1 = 0.063621
```

**Gate 行为**

| 模式 | 60000 gate | 70000 gate | 80000 gate | 判断 |
|---|---:|---:|---:|---|
| `none` | 0.873 | 0.860 | 0.844 | 逐步下降，正常输入也被压低了一点 |
| `flow_zero` | 0.788 | 0.741 | **0.616** | 80000 最好 |
| `no_video` | 0.565 | 0.651 | **0.476** | 80000 最好 |
| `wrong_video` | 0.827 | 0.810 | **0.766** | 80000 略好，但仍偏高 |

80000 的 gate 学得更“狠”：对 `flow_zero/no_video/wrong_video` 降得更多，这是好现象。  
但它也把正常输入 `none` 的 gate 从 `0.873` 压到 `0.844`，同时主任务 top1 没有超过 60000。所以 80000 更偏向“降权视频”，60000 更偏向“质量稳定”。

**我的选择**

如果你的 Stage7 目标是论文主结果里的最佳重建质量，选：

```bash
checkpoints/formal_ec_viai_av/stage7_evidence_gate_sigma/EC-VIAI-AV-PatchGAN_checkpoint_step000060000.pth.tar
```

如果你的 Stage7 目标是单独展示 evidence gate 对扰动的响应，可以把 80000 作为辅助分析 checkpoint，因为它的 gate 分离更明显：

```text
80000:
none gate = 0.844
flow_zero gate = 0.616
no_video gate = 0.476
wrong_video gate = 0.766
```

但正式往 Stage8/Stage10 继续，我建议用 **60000**。它的综合质量最好，而且没有过度压制正常视频。

下一步：**进入 Stage8 candidate scorer / uncertainty**，从 Stage7-60000 开始。Stage7 不建议继续训了。

---

## stage 8
我把 `65000 warmup / 80000 / 90000` 三组一起看了。结论很明确：**Stage8-90000 是目前最适合作为 Stage8 主 checkpoint 的结果；65000 可以作为 warmup ablation，80000 不建议选。**

**总体 Top1 对比**

| checkpoint | none | flow_zero | no_video | wrong_video | 平均 |
|---|---:|---:|---:|---:|---:|
| `65000 warmup` | 0.061239 | 0.061169 | **0.058796** | 0.062079 | 0.060821 |
| `80000` | 0.060420 | 0.060657 | 0.060275 | 0.062644 | 0.061000 |
| `90000` | **0.059706** | **0.058998** | 0.061461 | **0.059949** | **0.060029** |

90000 在 `none / flow_zero / wrong_video` 三项最好，整体平均也最好。唯一问题是 `no_video` 不如 65000。

**Uncertainty 表现**

| checkpoint | none spearman | flow_zero spearman | no_video spearman | wrong_video spearman |
|---|---:|---:|---:|---:|
| `65000` | 0.436 | 0.478 | 0.478 | 0.333 |
| `80000` | **0.438** | 0.408 | 0.452 | 0.336 |
| `90000` | 0.394 | **0.499** | **0.505** | **0.397** |

90000 的 uncertainty 在扰动条件下最好，尤其：

```text
flow_zero spearman = 0.499
no_video spearman = 0.505
wrong_video spearman = 0.397
```

这说明 Stage8-90000 的 uncertainty 已经有实际排序能力：高不确定性的样本更可能高误差。

**Gate 表现**

| checkpoint | none gate | flow_zero gate | no_video gate | wrong_video gate |
|---|---:|---:|---:|---:|
| `65000` | 0.873 | 0.788 | 0.566 | 0.827 |
| `80000` | 0.845 | 0.647 | 0.461 | 0.777 |
| `90000` | **0.823** | **0.493** | **0.382** | **0.728** |

这个非常漂亮。90000 明显学会了对坏视频降权：

```text
none gate = 0.823
flow_zero gate = 0.493
no_video gate = 0.382
wrong_video gate = 0.728
```

虽然 `wrong_video` 仍然偏高，但比 65000/80000 都低。这也再次说明：heuristic evidence 能压低低层损坏，但语义错配还是需要 Stage10 semantic evidence。

**Candidate scorer 情况**

90000 的 `loss_candidate_scorer` 变高：

```text
65000 none scorer loss = 1.046
80000 none scorer loss = 1.161
90000 none scorer loss = 1.175
```

并且 candidate diversity 变小：

```text
65000 pairwise ≈ 0.028-0.034
90000 pairwise ≈ 0.013-0.016
```

这说明后期全量微调让候选收敛了一些，多样性下降。但最终 top1 反而整体更好，说明它更偏向“稳定重建”，不是“最大多样性”。

**最终建议**

选这个作为 Stage8 主结果：

```bash
checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
```

实验记录可以这样写：

```text
Stage8-90000 achieves the best overall reconstruction and strongest perturbation-aware gate separation. 
Uncertainty ranking is effective under perturbations, while no_video reconstruction slightly degrades compared with the 65000 warmup checkpoint.
```

下一步可以进入 **Stage10 semantic evidence**。  
原因是 Stage8 已经把 gate/scorer/uncertainty 路径跑通了，但 `wrong_video_cross_instrument gate = 0.728` 仍然偏高，说明现在缺的正是语义 evidence。

## 附录：EC-VIAI-AV 核心指标含义

本文档解释 EC-VIAI-AV（Evidence-Calibrated Multi-Hypothesis VIAI-AV）训练/测试过程中涉及的各项核心指标含义，供实验记录和论文写作参考。

### 1. `top1` — Top-1 Missing L1 Error

**简称**：Top-1 error  
**全称**：Top-1 Missing-region L1 Distance  
**含义**：scorer（候选评分器）从 K 个候选中选出的"最佳"候选（置信度最高的那个候选）与 ground truth 缺失音频之间的 L1 距离。

- **物理意义**：模型实际输出的修复质量。数值越小越好。
- **对标 baseline**：此值应与 VIAI-AV baseline 的 `mel_l1_missing` 接近或更优。如果 `top1` 明显高于 baseline，说明 scorer 选择策略有问题或候选质量不足。
- **论文价值**：证明模型在多候选框架下，通过 scorer 能选出高质量候选，而不是靠 oracle 刷分。

### 2. `bestK` — Best-of-K Missing L1 Error

**简称**：Best-of-K error  
**全称**：Best-of-K Missing-region L1 Distance  
**含义**：从 K 个候选中选出距离 ground truth **最近**的那个候选（无论 scorer 选的是谁）的 L1 误差。

- **物理意义**：候选池的"上限潜力"。即如果模型有上帝视角，能从 K 个候选中挑出最好的，能达到多好的质量。
- **始终 ≤ `top1`**：因为 best-of-K 是 oracle 选择，scorer 不可能比 oracle 更强。如果 `bestK` 明显低于 `top1`，说明 scorer 还有提升空间。
- **论文价值**：证明多候选策略带来了"候选池"的价值，为不确定性校准提供支撑。

### 3. `oracle` — Oracle Gain

**简称**：Oracle gain  
**全称**：Oracle Gain = `top1` - `bestK`  
**含义**：scorer 选出的 top-1 结果与 oracle best-of-K 结果之间的差距。

- **物理意义**：scorer 选出来的结果与"上帝视角最佳候选"之间的差距。
- **数值较小**（理想情况在 0.001~0.005 量级）：说明 scorer 基本选得不错。
- **论文价值**：oracle gain 越小，说明 scorer 越可靠；oracle gain 越大，说明候选池中有明显更好的候选但 scorer 没选中，需要改进 scorer 或校准策略。
- **预期趋势**：在低视觉证据条件下（wrong_video、flow_zero），oracle gain 应增大，因为候选多样性上升，scorer 选择难度增加。

### 4. `pairwise` — Candidate Pairwise Mel L1 Distance

**简称**：Pairwise diversity  
**全称**：Candidate Pairwise Mel L1 Distance  
**含义**：K 个候选 Mel 之间两两计算 L1 距离的平均值。

- **物理意义**：候选池的**多样性**。数值越大，说明 K 个候选之间差异越大；数值越小，候选越趋同（可能塌缩成相同输出）。
- **论文价值**：
  - 在清晰视频下，`pairwise` 应小（候选集中，因证据强、不确定性低）。
  - 在模糊/遮挡/错位视频下，`pairwise` 应大（候选多样，因证据弱、不确定性高）。
  - `pairwise` 应与 `evidence_score` 负相关：证据越弱，多样性越高。

### 5. `meanK` — Mean-K Quality（Mean-K Missing L1）

**简称**：Mean-K error  
**全称**：Mean-K Missing-region L1 Distance  
**含义**：对所有 K 个候选的 missing region L1 取平均。

- **物理意义**：候选池的**平均质量**，防止只优化了 best-of-K 而让其他候选变成噪声。
- **理想的数值关系**：`meanK` 应介于 `top1` 和 `bestK` 之间，且不应远高于 `top1`（否则说明大多数候选质量差）。
- **论文价值**：证明模型生成的所有候选都有合理质量，而不是只靠一个候选"救命"。

### 6. `loss` — Total Training Loss

**含义**：训练总损失值。

- **组成**（以 Stage6 为例）：
  ```text
  loss = L_anchor + lambda_min_k * L_minK + lambda_mean_k * L_meanK
       + lambda_boundary * L_boundary + (baseline losses: recon, sync, probe, GAN...)
  ```
- **趋势**：随训练 step 持续下降说明模型正常收敛。

### 7. `psnr_m` — Missing-region PSNR

**含义**：缺失区域的 Peak Signal-to-Noise Ratio（峰值信噪比），衡量修复 Mel 缺失部分与 ground truth 的相似度，单位 dB。

- **物理意义**：数值越高，修复质量越好。这是传统图像/信号恢复的经典指标，对大幅误差更敏感。
- **论文价值**：作为 L1 误差的补充——PSNR 更关注人眼/人耳敏感的大幅度误差。
- **常见范围**：对于 Mel-spectrogram 修复，~20-22 dB 是合理范围（完全噪声时 PSNR 约 0-6 dB，完全完美时约 30+ dB）。

### 8. `loss_min_k` — Best-of-K Loss

**含义**：`L_minK = min_k L1(mel_candidate_k, mel_gt, missing_mask)`。在 K 个候选中选出与 ground truth 最近的候选计算 missing L1。

- **物理意义**：最优候选的重建质量。迫使模型至少产生一个好候选。
- **预期**：在训练过程中应逐步下降，且低于 `loss_recon`（candidate 0 的 L1）。

### 9. `loss_mean_k` — Mean-K Quality Loss

**含义**：`L_meanK = mean_k L1(mel_candidate_k, mel_gt, missing_mask)`。K 个候选的 average missing L1。

- **物理意义**：候选池的整体质量。防止只有一个候选好而其他候选退化为噪声。
- **预期**：应保持在合理范围，不应远高于 `loss_min_k`。

### 10. `loss_boundary` — Boundary Continuity Loss

**含义**：缺失片段左右边界处的 Mel 一阶差分连续性误差。

- **物理意义**：修复 Mel 与已知 Mel 在边界处是否平滑过渡，直接对应听感中的 click / discontinuity 问题。
- **预期**：在训练中下降，测试时越小越好。

### 11. Metrics 关系总结

```text
                           oracle gain（越小越好）
                        ┌──────────────────────────┐
                        │                          │
                    top1 ─── should be ≤ ─── bestK
                     │                            │
                     │                            │
        meanK ─── should be ≈ ─── top1    已包含在 bestK 中
          ↑                                    ↑
          │                                    │
    平均候选质量                            oracle 上限
```

**正常训练期望的趋势**：

| 关系 | 期望 |
|------|------|
| `bestK` ≤ `top1` | 始终成立，差距 oracle gain 应小 |
| `top1` ≤ `meanK` | 通常成立，top-1 应优于平均 |
| `top1` 接近 baseline `mel_l1_missing` | 目标：不弱于 baseline |
| `pairwise` > 0 | 候选必须有差异 |
| 高 evidence 时 `pairwise` 小 | 视觉证据强 → 候选集中 |
| 低 evidence 时 `pairwise` 大 | 视觉证据弱 → 候选多样 |
| `psnr_m` 与 `bestK` 负相关 | 数值上应自然对应 |

### 12. 论文级指标定位速查表

| 指标 | 论文术语 | 回答的问题 |
|------|----------|-----------|
| `top1_missing_l1` | Top-1 Quality | 模型实际输出好不好？ |
| `best_of_k_missing_l1` | Oracle Upper Bound | 候选池有没有更好的解？ |
| `oracle_gain` | Scorer Gap | scorer 是否可靠？ |
| `candidate_pairwise_mel_l1` | Diversity | 候选之间是否足够多样？ |
| `mean_k_missing_l1` | Mean-K Quality | 所有候选整体质量如何？ |
| `uncertainty_mean` | Predictive Uncertainty | 模型对当前样本是否确定？ |
| `evidence_mean` | Visual Evidence Score | 视频证据强还是弱？ |
| `psnr_missing` | Reconstruction PSNR | 缺失区域大尺度质量如何？ |
| `loss_boundary` | Boundary Continuity | 边界是否有点击感/断裂？ |