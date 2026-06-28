# 相关工作初稿

## 1. 音频修复与频谱补全

音频修复旨在恢复音频信号中缺失、损坏或被遮挡的部分。传统方法通常依赖信号处理、稀疏表示或频谱插值，根据缺失片段前后的局部上下文进行补全。随着深度学习的发展，基于神经网络的音频 inpainting 方法逐渐成为主流。这类方法通常将音频表示为 waveform、STFT 或 Mel-spectrogram，并使用卷积网络、循环网络、Transformer 或生成模型预测缺失区域。

在频谱补全任务中，Mel-spectrogram 是常用表示形式。模型通常接收 masked Mel-spectrogram 和 mask，输出缺失区域的预测结果，再与已知区域拼接得到完整频谱。此类方法能够利用音频上下文中的节奏、谐波和局部结构信息，但在较长缺失片段或上下文歧义较强时，仅凭音频本身往往无法唯一确定缺失内容。尤其在音乐场景中，同一段上下文可能对应多个合理的演奏延续。因此，单纯的音频修复方法容易面临多解性和不确定性表达不足的问题。

本文与纯音频修复方法的区别在于，我们关注音视频联合条件下的音频补全。视频提供了额外的发声线索，但这种线索并不总是可靠。因此，本文不仅利用视觉条件提升修复能力，还显式建模视觉证据强弱和预测不确定性。

## 2. 音视频联合学习与音视频修复

音视频联合学习利用声音与视觉之间的自然同步关系，已经被广泛应用于音视频检索、声音源定位、音视频分离、唇形驱动语音生成、音乐表演分析和多模态生成等任务。对于音乐和演奏视频而言，视觉中的乐器外观、手部运动、身体动作和节奏变化往往与音频事件存在相关性，因此视频可以为音频建模提供重要补充。

在音视频音频修复任务中，VIAI-AV 类方法将 masked audio 与对应视频共同输入模型，通过 RGB encoder、optical flow encoder 和 audio encoder 提取多模态特征，再由音视频融合解码器恢复缺失 Mel-spectrogram。此类方法表明，在视频包含有效发声动作时，视觉分支能够提高音频补全质量。同步约束、probe branch 和对抗训练等辅助目标也可进一步改善音视频表示与生成质量。

然而，现有音视频修复方法通常默认视频与音频匹配且可靠。模型在训练和测试中倾向于始终使用视觉条件，而没有显式回答“当前视频是否值得信任”。在真实场景中，这一假设较为脆弱：视频可能静止、模糊、缺帧、无光流、与音频不同步，甚至来自另一个乐器类别。如果模型缺少视觉可靠性判断机制，错误视频可能比无视频更危险，因为它提供了看似有效但语义错误的条件信息。

本文延续 VIAI-AV 的音视频修复主干，但将研究重点从“如何融合音视频特征”推进到“如何校准视觉条件的可信度”。通过 evidence estimation、fusion gate 和 semantic perturbation training，模型能够在视觉证据可靠时使用视频，在视觉证据弱或语义错配时降低视频依赖。

## 3. 多候选预测与不确定性建模

许多生成和补全任务天然具有多解性。给定相同的输入上下文，可能存在多个合理输出。例如，在轨迹预测、图像补全、语音生成和音频延续中，单一 ground truth 只是多个可行答案中的一个。确定性单输出模型往往会在不同可能解之间平均，导致结果过于保守或缺乏细节。

多候选预测通过生成多个候选来缓解这一问题。常见训练方式包括 min-of-K loss、best-of-K supervision、随机 latent sampling 和 mixture density modeling。Min-of-K loss 鼓励候选池中至少一个候选接近 ground truth，能够提升 oracle upper bound；mean-of-K 或候选质量约束则用于避免只有一个候选有效、其他候选崩坏。对于实际系统而言，仅有候选池还不够，模型还需要在测试时选择最终输出，因此 candidate scoring 或 ranking module 是多候选方法的重要组成部分。

不确定性建模与多候选预测密切相关。当输入证据不足或候选之间分歧较大时，模型应输出更高的不确定性。已有研究通常从 Bayesian approximation、ensemble、预测分布、calibration loss 或 risk-coverage 分析等角度估计不确定性。对于音视频修复而言，不确定性具有特殊意义：它不仅反映音频上下文的歧义，也反映视觉条件是否可靠。

本文将多候选预测和不确定性估计引入音视频音频修复。模型生成多个 candidate，并通过 Candidate Scorer 在测试时选择 top-1；同时使用 Uncertainty Head 输出样本级风险分数。与普通多候选生成不同，本文的候选多样性受 evidence score 调节：视觉证据强时候选更集中，视觉证据弱时候选更分散。这使多候选机制与多模态证据可靠性直接关联。

## 4. 视觉证据估计与条件融合

多模态模型通常需要决定不同模态在预测中的贡献。早期音视频融合方法常采用特征拼接、注意力、门控机制或跨模态 Transformer，将音频与视频表示联合建模。此类融合方式能够学习模态间交互，但如果没有显式可靠性约束，模型可能在某一模态退化时仍然过度依赖该模态。

针对模态质量变化的问题，已有工作提出了 modality dropout、missing modality training、confidence-aware fusion、gated fusion 等策略。这些方法的共同思想是：不同输入模态在不同样本上可靠性不同，融合权重应当随样本自适应变化。对于音视频修复而言，视频可靠性尤其重要，因为视频可能在低级质量上退化，也可能在语义层面与音频不匹配。

本文提出的 Evidence-Aware Fusion Gate 属于条件门控融合方法，但其关键区别在于 gate 由显式 evidence score 监督和调节。Heuristic evidence 主要刻画 optical flow、temporal variation 和视频特征强度等低级视觉证据；semantic evidence 则刻画当前视频是否符合源音频的乐器类别。二者融合后，模型能够同时处理 `flow_zero`、`no_video` 和 `wrong_video_cross_instrument` 等不同退化类型。

此外，本文使用 audio prior 替代低可信视频特征。当视觉证据较低时，模型不是简单丢弃视频分支，而是用音频 bottleneck 生成的 prior 对视频特征进行替换或校准。这种设计使模型在缺少可靠视觉信息时仍保留稳定的音频上下文路径。

## 5. 视觉-语言语义模型与语义证据

CLIP 等视觉-语言模型通过大规模图文对比学习获得了较强的开放词汇图像识别能力。冻结的视觉-语言模型常被用作下游任务的语义打分器、检索器或伪标签生成器。相比重新训练一个视频语义分类器，使用冻结 CLIP 具有实现简单、可离线预计算、不会影响主模型反向传播等优点。

在本文中，CLIP 并不直接参与音频修复生成，也不计算音频与视频 embedding 的相似度。它被作为 semantic evidence sidecar，用于回答一个更具体的问题：当前视频帧是否像源音频对应的乐器。为此，本文为每个视频样本预计算其对所有乐器 prompt 的概率分布，并在训练或测试时根据 source instrument 读取 target-specific score：

```text
e_{sem} = P(source_instrument | current_video_frames)
```

这一设计与普通视频自分类不同。如果 wrong video 来自 accordion，而源音频来自 xylophone，普通自分类会认为该视频是高置信 accordion，从而给出高分；target-specific semantic evidence 则查询 `P(xylophone | accordion video)`，因此能够正确反映跨乐器错配。

需要指出的是，CLIP semantic evidence 仍然存在局限。它主要利用视觉帧和文本标签之间的语义相似度，并不建模真实音频 embedding 与视频 embedding 之间的跨模态对齐；对于 saxophone、trumpet、tuba 等视觉外观或演奏场景相近的乐器，也可能出现混淆。ImageBind、AudioCLIP 等原生支持 audio-image 对齐的模型可能进一步增强音视频语义一致性判断。本文暂时将这些方向作为后续工作，而将重点放在证明 target-specific semantic evidence 与 perturbation training 对视觉错配鲁棒性的作用。

## 6. 本文与已有工作的区别

综上，现有音频修复方法主要关注如何根据音频上下文补全缺失片段；音视频修复方法进一步利用视频条件提升重建质量；多候选和不确定性方法关注生成多个合理输出并估计预测风险；视觉-语言模型提供了额外的语义识别能力。本文将这些方向结合到一个统一的音视频修复框架中，但核心关注点有所不同：

1. 本文不把视频视为始终可靠的条件，而是显式估计视觉证据，并用 gate 控制视觉分支对生成过程的影响。

2. 本文不只输出单一 Mel completion，而是生成多个候选，并通过 Candidate Scorer 和 Uncertainty Head 同时评估候选选择与样本风险。

3. 本文将 semantic evidence 定义为 source-instrument 条件分数，而非视频自身分类置信度，从而能够检测跨乐器 wrong video。

4. 本文进一步证明，仅有 semantic evidence 不足以改变模型行为；模型必须通过 semantic perturbation training 在训练阶段见到低 evidence 场景，才能在测试时形成合理的 gate suppression 和 uncertainty calibration。

因此，本文的贡献并不是简单增加一个语义分类器，而是构建一个 evidence-calibrated 音视频修复机制，使模型在视觉可靠性变化时能够自适应地调整信任、候选多样性和不确定性输出。
