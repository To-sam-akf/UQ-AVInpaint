核心叙事不要写成“我重新做了一个更强的音频修复模型”，而要写成：

**EC-VIAI-AV 是在 VIAI-AV 已验证主干上的可退化扩展：当视频证据可靠时，模型更确定地利用视觉信息；当视频证据弱、错位、缺失或不可靠时，模型生成多个合理候选，并输出不确定性。**
这个定位和你当前工作稿里的主线是一致的：模型不是替换 VIAI-AV backbone，而是在 bottleneck 与输出选择层加入 evidence calibration、multi-hypothesis sampling、candidate scorer 和 uncertainty head。

---

## 一、这篇中文初稿应该包括哪些内容

### 1. 研究背景

这一部分要回答：**为什么音视频联合音频修复需要不确定性建模？**

可以这样展开：

音频修复任务中，模型需要根据已知音频上下文和同步视频信息，补全缺失的 Mel-spectrogram 片段。VIAI-AV 这类方法证明了 RGB 图像和 optical flow 对音频修复有帮助，尤其是当视频中存在明显发声动作时。

但问题在于，现有方法通常是**确定性单输出**：

```text
missing_mel = f(audio_context, video, mask)
```

这意味着模型默认“缺失音频只有一个正确答案”。实际场景中，这个假设并不总成立。比如：

* 视频画面静止；
* 光流很弱；
* 乐器被遮挡；
* 视频和音频不同步；
* 视频来自错误乐器；
* 视频缺失或不可用；
* 同一个上下文下可能存在多个合理的音频延续。

因此，你的研究可以从“**确定性 AV inpainting 无法表达多解性和视觉证据可靠性**”这个角度切入。

---

### 2. 研究问题

建议明确提出两个核心问题：

**问题 1：视觉证据是否总是可靠？**
不是。视频可能提供强证据，也可能是弱证据、错证据，甚至无证据。

**问题 2：当视觉证据弱时，模型是否应该仍然输出一个确定答案？**
不应该。更合理的做法是输出多个候选，并告诉用户模型对当前样本有多不确定。

因此，本文目标可以写成：

> 本文希望将 VIAI-AV 从确定性单输出模型扩展为 evidence-calibrated multi-hypothesis 模型，使其能够根据视觉证据强弱自适应地调整视频依赖程度、候选多样性与预测不确定性。

---

### 3. 方法贡献

建议贡献写三点，不要太多。

**贡献 1：提出 EC-VIAI-AV 框架。**
在不替换 VIAI-AV 主干的前提下，将原本的单输出音视频修复模型扩展为多候选生成模型。当 `K=1` 或关闭随机分支时，模型可以退化为原始 VIAI-AV 风格的确定性路径。

**贡献 2：提出 evidence-calibrated candidate generation。**
通过 Visual Evidence Estimator、Evidence-Aware Fusion Gate 和 evidence-conditioned sigma scaling，让模型在视频证据强时生成更集中的候选，在视频证据弱时降低视频依赖并提高候选多样性。

**贡献 3：提出面向多候选 AV inpainting 的评价协议。**
不仅评估 top-1 Mel L1、PSNR、SSIM，还评估 best-of-K、candidate diversity、boundary continuity、uncertainty-error correlation 等指标，从而更全面地分析多候选修复质量与不确定性校准能力。

---

### 4. 方法部分

方法部分是这篇初稿最重要的部分，建议先写清楚以下模块：

#### 4.1 问题定义

定义输入和输出：

* 输入：masked Mel-spectrogram、RGB frames、optical flow、mask；
* 输出：K 个候选 Mel completion；
* 最终通过 Candidate Scorer 选择 top-1；
* 同时输出 uncertainty score。

可以写成：

```text
{y_1, ..., y_K} = F_\theta(x_m, v, m)
```

其中 `x_m` 是被遮挡后的 Mel，`v` 是视频输入，`m` 是缺失区域 mask。

最终 completed Mel 为：

```text
\hat{x}_k = x_m \odot (1 - m) + y_k \odot m
```

强调：**已知区域始终保留原输入，模型只负责补全缺失区域。**

---

#### 4.2 EC-VIAI-AV Backbone

说明你不是重新设计整个生成器，而是复用原始 VIAI-AV：

* Mel Encoder；
* RGB / Optical Flow Video Encoder；
* Audio-Visual Fusion Decoder；
* Sync loss；
* Probe branch；
* 可选 PatchGAN。

这部分的重点是：**你的创新集中在 bottleneck 和候选选择层，而不是换 backbone。**

这样写有两个好处：

第一，和 baseline 对比更公平；
第二，论文故事更稳，因为你是在一个已验证有效的架构上做 uncertainty-aware extension。

---

#### 4.3 Visual Evidence Estimation

这一节说明模型如何判断视频证据强弱。

可以写：

VisualEvidenceEstimator 输出一个视觉证据分数：

```text
e \in [0, 1]
```

其中：

* `e` 高：说明视频中存在可靠的发声相关运动；
* `e` 低：说明视频证据弱，模型不应盲目依赖视觉分支。

当前 heuristic evidence 主要由以下信号组成：

* optical flow magnitude；
* temporal variation；
* audio-video embedding sync score；
* video feature activation strength。

要注意：这里需要诚实说明，当前 evidence estimator 更擅长衡量**运动证据强弱**，还不是完整的语义错配检测器。你的工作稿里也明确指出，它对 flow degradation 反应稳定，但对 wrong-video semantic mismatch 和 temporal shift 的区分能力有限。

---

#### 4.4 Evidence-Aware Fusion Gate

这一节说明模型如何根据 evidence 控制视频依赖。

核心公式：

```text
g = sigmoid(MLP([p_A, p_V, e]))
```

```text
V_c = g * V + (1 - g) * P
```

其中：

* `A` 是 audio bottleneck；
* `V` 是 video feature；
* `P` 是由 audio bottleneck 生成的 audio prior；
* `g` 是 evidence-aware gate；
* `V_c` 是校准后的视频特征。

解释逻辑：

当视觉证据强时，`g` 较大，模型更多保留视频信息；
当视觉证据弱时，`g` 较小，模型更多依赖 audio prior，避免错误视频误导解码器。

---

#### 4.5 Stochastic Bottleneck Adapter

这一节是多候选生成的核心。

你可以这样写：

EC-VIAI-AV 不直接改变 Mel decoder，而是在 bottleneck 处引入 stochastic adapter。模型根据音频 bottleneck 和校准后的视频特征预测 latent distribution：

```text
\mu, \log\sigma^2 = h([A, V_c])
```

然后采样 K 个 latent residual：

```text
z_k = \mu + s_\sigma \sigma \epsilon_k
```

其中 `s_sigma` 由 evidence 或 gate 控制：

* 视觉证据强：`s_sigma` 小，候选更集中；
* 视觉证据弱：`s_sigma` 大，候选更多样。

要重点强调 deterministic anchor：

```text
\epsilon_1 = 0
```

也就是说，第一个候选是稳定的 deterministic candidate，用于保持 baseline reconstruction 路径，其余候选负责表达多解性。

---

#### 4.6 Candidate Scorer

这一节说明：**有 K 个候选之后，测试时怎么选 top-1？**

注意：测试时不能用 ground truth missing Mel，所以 scorer 不能看真实误差。

它只能使用测试时可获得的 proxy statistics，例如：

* 候选与已知 Mel 区域的一致性；
* 缺失边界处的 boundary jump；
* audio-video sync proxy；
* pooled audio context；
* pooled calibrated video context；
* evidence score。

输出：

```text
\pi_k = softmax(Scorer(stats_k, p_A, p_{V_c}, e))
```

最终选择：

```text
k^* = argmax_k \pi_k
```

这里一定要区分：

* **top-1**：模型真实测试输出；
* **best-of-K**：oracle 指标，只用于评估候选池上限，不能作为实际输出。

---

#### 4.7 Uncertainty Head

Uncertainty Head 输出样本级不确定性：

```text
u \in [0, 1]
```

输入可以包括：

* candidate probability entropy；
* top candidate confidence；
* candidate pairwise distance；
* evidence score；
* gate value；
* sigma scale；
* top-1 proxy error。

训练目标是让 `u` 与真实 error 正相关。

实验中你已经有比较有价值的结果：EC-VIAI-AV Candidate-Scorer Baseline 的 uncertainty-error Pearson/Spearman 已经呈现有效正相关，说明 uncertainty head 能反映样本难度排序。

---

### 5. 损失函数

损失函数建议不要写得太散，可以统一成：

```text
L = L_anchor
  + λ_minK L_minK
  + λ_meanK L_meanK
  + λ_boundary L_boundary
  + λ_div L_evidence_div
  + λ_calib L_calib
  + λ_sync L_sync
  + λ_probe L_probe
  + λ_gan L_gan
```

每个损失的作用：

| 损失项              | 作用                                             |
| ---------------- | ---------------------------------------------- |
| `L_anchor`       | 保持 deterministic candidate 接近 baseline，防止训练不稳定 |
| `L_minK`         | 鼓励候选池中至少一个候选接近真实缺失片段                           |
| `L_meanK`        | 防止只有一个候选好、其余候选崩坏                               |
| `L_boundary`     | 保证缺失边界平滑，减少断裂                                  |
| `L_evidence_div` | 让候选多样性随视觉证据弱化而增加                               |
| `L_calib`        | 让 uncertainty score 与真实 error 正相关              |
| `L_sync`         | 保持音视频同步约束                                      |
| `L_probe`        | 沿用 VIAI-AV probe branch                        |
| `L_gan`          | 可选 PatchGAN 约束                                 |

初稿里不需要把所有数学推导写得很复杂，但要把每个 loss 的动机讲清楚。

---

### 6. 实验部分

实验部分建议分成四类。

#### 6.1 Baseline 对比

至少包括：

* VIAI-AV-PatchGAN reference；
* Multi-Candidate EC-VIAI-AV；
* Evidence-Calibrated EC-VIAI-AV；
* EC-VIAI-AV Candidate-Scorer Baseline。

你当前最适合作为基础主结果的是 EC-VIAI-AV Candidate-Scorer Baseline，因为它同时包含 top-1 reconstruction、candidate scorer 和 uncertainty calibration。工作稿中记录它的 top-1 missing L1 基本匹配 VIAI-AV-PatchGAN reference，同时 PSNR Missing 和 SSIM 有提升。

---

#### 6.2 多候选有效性

这一部分回答：

> 生成多个候选是否真的有意义？

可以报告：

* candidate0 missing L1；
* top1 missing L1；
* best-of-K missing L1；
* random expected missing L1；
* oracle gain；
* candidate pairwise distance。

你要证明：

* best-of-K 优于 candidate0；
* top-1 优于随机选候选；
* candidate scorer 能把部分 oracle improvement 转化为实际输出；
* 候选之间确实存在非零多样性。

---

#### 6.3 Evidence calibration 实验

这一部分是你论文最有特色的实验。

可以做 controlled degradation：

* original；
* flow_75；
* flow_50；
* flow_25；
* flow_zero；
* static_video_zero_flow；
* no_video。

报告：

* evidence；
* gate；
* sigma scale；
* pairwise distance。

你已经有比较清晰的现象：随着 optical flow 被逐渐削弱，evidence 下降，gate 下降，sigma scale 上升，candidate pairwise distance 上升。Evidence-Calibrated EC-VIAI-AV 中，`pairwise_flow_zero > original` 的 hit rate 相比早期 gate-only 版本明显提升，这个结论很适合作为方法有效性的核心证据。

---

#### 6.4 不确定性校准实验

这一部分回答：

> 模型输出的不确定性是否真的和错误相关？

建议报告：

* Pearson correlation；
* Spearman correlation；
* uncertainty-error scatter；
* risk-coverage curve；
* calibration bins。

初稿中可以先写 Pearson 和 Spearman，图可以后续补。

---

### 7. Discussion 和 Limitations

这一节一定要写，而且要诚实。

建议写以下几点：

1. 当前 heuristic evidence 更擅长检测 motion evidence，而不是完整语义错配。
2. wrong video 如果仍然包含明显演奏动作，heuristic evidence 不一定显著下降。
3. temporal shift 在动作较弱的视频中不一定容易检测。
4. best-of-K 仍优于 scorer top-1，说明 candidate scorer 仍有提升空间。
5. semantic evidence sidecar 可以作为后续增强，但需要谨慎处理 wrong-video 条件下的 source instrument 匹配问题。

这部分写好，反而会让论文显得更可信。

---

## 二、推荐中文论文初稿大纲

下面是一个比较适合你当前阶段的中文初稿结构。

# 题目

**EC-VIAI-AV：证据校准的多候选音视频联合音频修复方法**

或者更学术一点：

**面向音视频联合音频修复的证据校准多假设生成方法**

---

# 摘要

摘要建议最后写。先占位即可。

需要包括：

1. 任务背景：音视频联合音频修复；
2. 现有问题：确定性单输出无法表达多解性和视觉证据不可靠；
3. 方法：Evidence-Calibrated Multi-Hypothesis VIAI-AV；
4. 模块：visual evidence、fusion gate、stochastic adapter、candidate scorer、uncertainty head；
5. 结果：匹配 baseline top-1，提升 best-of-K，获得有效 uncertainty-error correlation。

---

# 1 引言

## 1.1 研究背景

介绍音频修复、Mel-spectrogram inpainting、视觉信息对音频修复的帮助。

## 1.2 现有方法局限

重点写确定性单输出的问题：

* 视频证据并不总可靠；
* 缺失音频可能存在多个合理答案；
* 现有模型无法表达不确定性；
* 错误视频可能误导模型。

## 1.3 本文方法

介绍 EC-VIAI-AV 的整体想法：

* 复用 VIAI-AV backbone；
* 引入 evidence estimation；
* 根据 evidence 调整视频融合；
* 在 bottleneck 处生成多个候选；
* 通过 scorer 选择 top-1；
* 输出 uncertainty score。

## 1.4 主要贡献

写三点贡献。

---

# 2 相关工作

这一节初稿可以先写得简洁，后续再补文献。

## 2.1 Audio Inpainting

介绍音频缺失补全、spectrogram completion、audio-only inpainting。

## 2.2 Audio-Visual Learning

介绍音视频同步、视觉辅助音频生成、visual sound source modeling。

## 2.3 Multi-Hypothesis Prediction

介绍多候选预测在不确定任务中的意义。

## 2.4 Uncertainty Estimation and Calibration

介绍不确定性估计、选择性预测、error correlation。

---

# 3 方法

这是全文核心。

## 3.1 问题定义

定义输入、mask、输出候选、top-1 和 uncertainty。

## 3.2 EC-VIAI-AV Backbone

说明模型继承 Mel encoder、video encoder、fusion decoder、sync loss、probe branch 和 PatchGAN。

## 3.3 Visual Evidence Estimation

说明 heuristic evidence 的构成和意义。

## 3.4 Evidence-Aware Fusion Gate

说明 gate 如何控制视频特征和 audio prior 的融合。

## 3.5 Stochastic Bottleneck Adapter

说明如何生成 K 个候选，为什么保留 deterministic anchor。

## 3.6 Evidence-Conditioned Sigma Scaling

说明低 evidence 对应更高采样尺度，高 evidence 对应更集中候选。

## 3.7 Candidate Scorer

说明测试时如何在无 ground truth 条件下选择 top-1。

## 3.8 Uncertainty Head

说明 uncertainty score 的输入、输出和训练目标。

## 3.9 Training Objective

统一写总损失函数和每项损失的作用。

## 3.10 Inference Procedure

用步骤列出推理流程：

1. 输入 masked Mel 和视频；
2. 计算 audio/video feature；
3. 估计 evidence；
4. 得到 calibrated video feature；
5. 采样 K 个候选；
6. scorer 选择 top-1；
7. uncertainty head 输出 `u`。

---

# 4 实验设置

## 4.1 数据集与输入设置

写 MUSICES、4 秒音频、80×200 Mel、50 帧 RGB、50 帧 optical flow、20–50 frame mask。

## 4.2 Baselines

包括：

* VIAI-AV-PatchGAN reference；
* Multi-Candidate EC-VIAI-AV；
* Evidence-Calibrated EC-VIAI-AV；
* EC-VIAI-AV Candidate-Scorer Baseline。

## 4.3 Evaluation Metrics

分成几组：

**重建质量：**

* Missing L1；
* Full L1；
* PSNR Missing；
* PSNR Full；
* SSIM。

**多候选质量：**

* Top-1 L1；
* Candidate0 L1；
* Best-of-K L1；
* Random Expected L1；
* Oracle Gain；
* Pairwise Distance。

**证据与不确定性：**

* Evidence mean；
* Gate mean；
* Sigma scale；
* Uncertainty-error Pearson；
* Uncertainty-error Spearman。

## 4.4 Video Perturbation Protocol

写：

* original；
* flow degradation；
* flow zero；
* static video；
* no video；
* wrong video；
* temporal shift。

---

# 5 实验结果与分析

## 5.1 主结果对比

放主表，对比 Original VIAI-AV、Multi-Candidate EC-VIAI-AV、Evidence-Calibrated EC-VIAI-AV 和 EC-VIAI-AV Candidate-Scorer Baseline。

重点结论：

* EC-VIAI-AV Candidate-Scorer Baseline 的 top-1 missing L1 接近 VIAI-AV-PatchGAN reference；
* PSNR Missing 和 SSIM 有提升；
* best-of-K 优于 top-1，说明候选池仍有 oracle 上限；
* uncertainty Spearman 达到有效正相关。

## 5.2 多候选生成是否有效

分析：

* best-of-K 是否优于 candidate0；
* top-1 是否优于 random candidate；
* pairwise distance 是否非零；
* K 个候选是否提供了合理多样性。

## 5.3 Evidence Gate 是否有效

分析 controlled degradation：

* flow 越弱，evidence 越低；
* evidence 越低，gate 越低；
* evidence 越低，sigma scale 越高；
* evidence 越低，candidate diversity 越高。

## 5.4 Candidate Scorer 是否有效

分析：

* top-1 vs candidate0；
* top-1 vs random expected；
* top-1 vs best-of-K；
* scorer 还有多少 oracle headroom。

## 5.5 Uncertainty Calibration

分析：

* uncertainty-error Pearson；
* uncertainty-error Spearman；
* 高不确定性样本是否更容易出错；
* uncertainty 是否可用于风险排序。

## 5.6 Ablation Study

建议包括：

* no evidence gate；
* fixed sigma；
* no candidate scorer；
* no uncertainty calibration；
* no boundary loss；
* no diversity target；
* audio-only；
* full model。

---

# 6 讨论

## 6.1 为什么多候选是必要的

从真实场景多解性解释。

## 6.2 视觉证据可靠性的重要性

强调不是所有视频信息都应该被信任。

## 6.3 Top-1 与 Best-of-K 的差距

说明候选池有潜力，但 scorer 仍需改进。

## 6.4 Heuristic Evidence 的边界

说明它更偏 motion evidence，不是强语义理解模块。

---

# 7 局限性与未来工作

建议写：

1. 当前 evidence estimator 对 wrong-video semantic mismatch 不够敏感；
2. temporal shift 检测仍不稳定；
3. candidate scorer 仍未完全达到 oracle best-of-K；
4. 目前主要在 Mel 级别评估，后续需要 vocoder 听感实验；
5. semantic evidence 和更强的 video-language prior 可以作为后续方向；
6. diffusion-based latent sampler 可以作为增强版，而不是当前第一版核心。

---

# 8 结论

结论部分回答三句话：

1. 本文提出了 EC-VIAI-AV，将 VIAI-AV 从确定性单输出扩展为证据校准的多候选生成框架。
2. 通过 evidence-aware fusion 和 evidence-conditioned sampling，模型能够在视觉证据弱时提高候选多样性，在视觉证据强时保持更确定输出。
3. 实验表明，该方法在保持 top-1 修复质量的同时，提供了 best-of-K 上限和有效的不确定性估计。

---

## 三、建议你实际写作的顺序

不要从摘要开始。建议按这个顺序写：

1. **方法部分**：先把模型结构写清楚；
2. **实验设置**：固定数据、baseline、metric；
3. **实验结果**：先整理主表和关键结论；
4. **Discussion / Limitation**：把边界说清楚；
5. **Introduction**：最后回头压缩故事；
6. **Abstract**：最后写摘要。

现在最适合先写的是：

> 第 3 节 方法
> 第 4 节 实验设置
> 第 5 节 实验结果与分析

引言和摘要可以等方法、实验写稳之后再润色。
