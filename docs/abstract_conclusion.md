# Abstract and Conclusion 初稿

## 摘要

音视频联合音频修复利用视频中的演奏动作和视觉上下文补全缺失音频，相比仅依赖音频上下文的方法具有更强的条件信息。然而，现有 VIAI-AV 类方法通常将视频视为始终可靠的输入，并采用确定性单输出预测，这使模型难以处理视频缺失、光流失效、视觉语义错配以及同一上下文下存在多个合理补全的情况。针对这一问题，本文提出 EC-VIAI-AV，即 Evidence-Calibrated Multi-Hypothesis VIAI-AV，在保留原始 VIAI-AV backbone 的基础上，引入视觉证据估计、evidence-aware fusion gate、多候选生成、candidate scorer 和 uncertainty head。该框架根据视觉证据强弱自适应调整视频特征的贡献：当视觉证据可靠时，模型更充分地利用视频；当视觉证据弱、缺失或语义不匹配时，模型降低视觉依赖，生成多个候选，并输出样本级不确定性。

进一步地，本文引入基于冻结 CLIP 的 target-specific semantic evidence，用于判断当前视频是否符合源音频对应的乐器类别，而不是简单判断视频自身类别置信度。该语义证据能够有效识别跨乐器 wrong video，但实验表明，仅接入 semantic evidence 并不足以改变模型行为。为此，本文提出 semantic perturbation training，在训练阶段显式加入 `wrong_video_cross_instrument`、`no_video` 和 `flow_zero` 等视觉扰动，使模型学习在低 evidence 情况下降低 gate 并提高不确定性排序能力。

在 MUSICES 数据集上的实验表明，EC-VIAI-AV 相比 Original VIAI-AV 提升了基础修复质量，而 semantic evidence 与 perturbation training 的结合进一步改善了视觉不可靠场景下的鲁棒性。最终模型在 `wrong_video_cross_instrument` 下将 top-1 missing L1 从 Original VIAI-AV 的 `0.067543` 降至 `0.060251`，并将 visual gate 从 EC-VIAI-AV Candidate-Scorer Baseline 的 `0.713195` 降至 `0.413393`；同时，uncertainty-error Spearman 从 `0.326222` 提升至 `0.548191`。在 `no_video` 下，gate mean 从 `0.367402` 降至 `0.089303`。这些结果说明，本文方法能够使音视频修复模型从“默认使用视频”转向“根据证据有条件地信任视频”，从而在视觉缺失和语义错配场景下获得更可靠的修复行为。

**关键词**：音视频音频修复；多候选生成；视觉证据校准；语义证据；不确定性估计

## Abstract

Audio-visual audio inpainting restores missing audio segments by leveraging both audio context and visual cues from the corresponding video. Existing VIAI-AV style methods typically treat the video stream as a reliable conditioning signal and produce a deterministic single output. This assumption becomes fragile when the video is missing, motion cues are weak, optical flow fails, or the video is semantically mismatched with the audio. To address this issue, we propose EC-VIAI-AV, an Evidence-Calibrated Multi-Hypothesis extension of VIAI-AV. Built on top of the original VIAI-AV backbone, EC-VIAI-AV introduces visual evidence estimation, an evidence-aware fusion gate, multi-hypothesis generation, a candidate scorer, and an uncertainty head. The model adaptively controls its reliance on visual features: it exploits video cues when evidence is reliable, and reduces visual dependence while producing multiple plausible candidates when evidence is weak or semantically inconsistent.

We further introduce target-specific semantic evidence computed offline with a frozen CLIP sidecar. Instead of measuring whether a video resembles its own class, the semantic evidence estimates whether the current video frames match the source instrument associated with the audio. This formulation effectively detects cross-instrument wrong videos. However, our experiments show that semantic evidence alone is insufficient: without seeing low-evidence cases during training, the model does not learn to suppress the visual gate. We therefore propose semantic perturbation training, which explicitly exposes the model to wrong-video, no-video, and flow-zero conditions.

Experiments on MUSICES show that EC-VIAI-AV improves the base inpainting capability over Original VIAI-AV, while semantic perturbation training further improves robustness under unreliable visual conditions. Under cross-instrument wrong-video perturbation, the final model reduces top-1 missing L1 from `0.067543` to `0.060251`, lowers the visual gate from `0.713195` to `0.413393`, and improves uncertainty-error Spearman correlation from `0.326222` to `0.548191`. These results demonstrate that evidence calibration enables audio-visual inpainting models to move from blindly using video to conditionally trusting video based on its reliability.

## 结论

本文研究了音视频联合音频修复中的一个关键但容易被忽视的问题：视频条件并不总是可靠。传统 VIAI-AV 类模型通常默认视频与音频匹配，并输出单一确定性结果。当视频正常且包含清晰演奏动作时，这一假设可以带来有效视觉补充；但当视频缺失、光流失效或来自另一个乐器类别时，盲目信任视频会导致错误视觉信息进入修复过程。本文的目标不是简单替换 VIAI-AV 主干，而是在已有 backbone 上加入证据校准、多候选生成和不确定性估计，使模型能够根据视觉证据强弱自适应地调整视频依赖。

为此，本文提出 EC-VIAI-AV。该方法首先估计视觉证据分数，并通过 Evidence-Aware Fusion Gate 在视频特征与 audio prior 之间进行动态融合；随后通过 stochastic adapter 生成多个候选补全，并使用 candidate scorer 在测试时选择 top-1 输出；同时，uncertainty head 输出样本级风险估计。与确定性单输出模型相比，EC-VIAI-AV 能够更好地表达缺失音频的多解性，并在视觉证据不足时给出更合理的不确定性响应。

本文进一步发现，低级 heuristic evidence 虽然能识别 `flow_zero` 和 `no_video` 等视觉退化，但难以判断跨乐器语义错配。为解决这一问题，本文引入 target-specific semantic evidence，将语义分数定义为：

```text
P(source_instrument | current_video_frames)
```

该定义避免了普通视频自分类在 wrong video 上给出高置信分数的问题，能够直接衡量当前视频是否符合源音频所属乐器。实验诊断表明，corrected semantic evidence 可以将 wrong video 的 source-instrument score 显著压低。但进一步实验也表明，仅有 semantic evidence 并不足以使模型自动降低 gate；模型必须在训练阶段显式见到低 evidence 的视觉扰动。Semantic Perturbation Training 因此成为关键设计，它通过引入 `wrong_video_cross_instrument`、`no_video` 和 `flow_zero`，使 gate、candidate diversity 和 uncertainty 与 evidence target 形成一致响应。

实验结果验证了上述设计的必要性。相比 Original VIAI-AV，EC-VIAI-AV Candidate-Scorer Baseline 已经提升了基础修复质量；相比只进行 Semantic Evidence Fine-Tuning，Semantic Perturbation Training 显著改善了 wrong-video 和 no-video 下的 gate calibration。最终模型在跨乐器 wrong-video 场景下明显降低 visual gate，并提升 uncertainty-error Spearman correlation，同时保持正常视频下可接受的重建质量。消融实验进一步表明，去掉 wrong-video augmentation 后模型无法处理“画面存在但语义错误”的视频；只使用 heuristic evidence 虽然能改善部分视觉退化场景，但缺少跨乐器语义判断能力。

总体而言，本文证明了音视频修复模型不应只追求更强的视频利用能力，更应具备判断视频可靠性的能力。EC-VIAI-AV 将音视频音频修复从“默认使用视频的确定性预测”推进到“根据视觉证据有条件地信任视频的多候选预测”。这一思路对于真实音视频修复系统具有实际意义，因为真实输入中常常存在视频缺失、弱运动、遮挡、同步误差和语义错配。未来工作可以进一步引入 ImageBind、AudioCLIP 等原生 audio-video 对齐模型，使用更强的视觉语义模型减少相近乐器混淆，并在更多数据集和真实扰动场景下验证该框架的泛化能力。
