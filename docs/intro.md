# 引言初稿

音频修复旨在根据上下文信息恢复音频信号中的缺失片段。在音乐表演、视频编辑、历史音视频修复和多媒体内容生成等场景中，音频缺失往往并不是孤立发生的：缺失音频通常伴随可见的演奏动作、乐器外观、手部运动或舞台画面。因此，相比仅依赖音频上下文的修复方法，音视频联合音频修复可以利用视频中与发声过程相关的视觉线索，为缺失频谱的恢复提供额外条件。

已有 VIAI-AV 类方法已经证明，RGB 图像和 optical flow 对 Mel-spectrogram inpainting 具有积极作用。尤其当视频中存在清晰的发声动作时，视觉分支能够帮助模型补充仅凭音频上下文难以确定的信息。例如，演奏动作的节奏、乐器的类别、手指或身体运动都可能与音频事件存在对应关系。然而，现有方法通常将视频视为始终可靠的条件输入，并使用确定性单输出形式完成修复：

```text
\hat{x} = f_\theta(x_m, v, m)
```

其中 `x_m` 表示 masked Mel-spectrogram，`v` 表示视频输入，`m` 表示缺失区域 mask。该建模方式隐含两个假设：第一，缺失音频存在一个确定的最优补全；第二，输入视频始终与缺失音频相关且可靠。在实际音视频数据中，这两个假设并不总成立。

一方面，音频修复本身具有多解性。给定相同的音频上下文，缺失片段可能存在多个听感合理的延续方式。确定性模型只能输出单一预测，容易在多种可能解之间产生平均化结果，也无法表达候选之间的不确定性。另一方面，视频条件并不总是可靠。视频可能因为运动微弱、画面静止、乐器遮挡、光流失效、视频缺失或时序偏移而无法提供有效发声线索；更严重的是，视频也可能来自另一个乐器或另一个样本。此时如果模型仍然强制依赖视觉分支，错误视觉信息反而会误导音频修复。

因此，音视频联合音频修复不应只关注“如何更充分地利用视频”，还应回答一个更基础的问题：**模型应当在何时信任视频，以及在视频证据不足时如何表达不确定性**。本文围绕这一问题提出 EC-VIAI-AV，即 Evidence-Calibrated Multi-Hypothesis VIAI-AV。该方法不是重新设计一个全新的音频修复主干，而是在已有 VIAI-AV backbone 上进行可退化扩展：当视觉证据可靠时，模型充分利用视频信息；当视觉证据弱、缺失或语义不匹配时，模型降低视觉条件的影响，生成多个合理候选，并输出样本级不确定性。

具体而言，本文首先引入视觉证据估计机制，为每个样本计算 evidence score。该分数由两类信息组成：一类是基于 optical flow、temporal variation 和音视频同步 proxy 的低级 heuristic evidence，用于衡量视频是否存在可用运动和同步线索；另一类是基于冻结 CLIP 模型离线预计算的 semantic evidence，用于判断当前视频帧是否符合原始音频对应的乐器类别。与简单判断“视频像不像自己的类别”不同，本文采用 target-specific semantic evidence：

```text
e_{sem} = P(source_instrument | current_video_frames)
```

该定义使模型能够识别 `wrong_video_cross_instrument` 这类低级视觉信号仍然存在、但语义类别与音频不匹配的情况。

其次，本文设计 Evidence-Aware Fusion Gate，根据视觉证据强度自适应调整视频特征的贡献。当 evidence 较高时，模型更多保留视频特征；当 evidence 较低时，模型更多依赖由音频 bottleneck 生成的 audio prior，从而降低错误视频对解码器的影响。为了表达缺失片段的多解性，本文进一步在 bottleneck 处引入 stochastic adapter，生成 `K` 个候选 Mel completion，并使用 evidence-conditioned sigma scaling 控制候选分布：高 evidence 样本生成更集中的候选，低 evidence 样本允许更大的候选多样性。

最后，本文引入 Candidate Scorer 与 Uncertainty Head。Candidate Scorer 在测试时不访问 ground truth，仅根据候选统计特征、音频上下文、视频上下文和 evidence score 选择 top-1 candidate；Uncertainty Head 输出样本级不确定性，用于估计当前修复风险。这样，模型不仅输出一个最终修复结果，还能够给出候选池上限、候选选择质量和不确定性排序能力等更完整的分析。

仅有 semantic evidence 并不足以解决视觉错配问题。实验发现，虽然 corrected semantic evidence 能够将跨乐器 wrong video 的 evidence 显著压低，但如果训练阶段从未显式出现过低 evidence 的视频扰动，gate 仍然不会自动学会响应这种低证据目标。为此，本文进一步提出 semantic perturbation training，在训练中引入 `wrong_video_cross_instrument`、`no_video` 和 `flow_zero` 等扰动，使模型实际见到视觉缺失、低运动证据和语义错配输入。该训练策略使 gate、candidate diversity 和 uncertainty 在低 evidence 情况下形成一致响应。

本文的主要贡献可以概括为以下三点：

1. 提出 EC-VIAI-AV 框架，在不替换 VIAI-AV 主干的前提下，将确定性单输出音视频修复扩展为 evidence-calibrated multi-hypothesis prediction，使模型能够根据视觉证据强弱自适应调整视频依赖程度。

2. 引入 target-specific semantic evidence 与 Evidence-Aware Fusion Gate，将低级运动证据和语义类别证据结合起来，使模型能够区分正常视频、无视频、光流退化和跨乐器 wrong video 等不同视觉可靠性状态。

3. 提出 semantic perturbation training 与多维度评估协议，验证模型不仅在正常视频下保持可接受的重建质量，而且在无视频和跨乐器错配场景下获得更合理的 gate calibration 与 uncertainty-error correlation。

总体而言，本文的目标不是简单追求单一重建指标上的提升，而是将音视频联合音频修复从“默认信任视频的确定性预测”推进到“根据证据校准视觉信任、表达多候选和不确定性”的鲁棒修复范式。
