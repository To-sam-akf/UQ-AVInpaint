# Discussion and Limitations 初稿

## 1. Discussion

本文的核心发现是：在音视频联合音频修复中，视觉信息不应被默认视为可靠条件。Original VIAI-AV 和一般音视频修复模型通常假设输入视频与音频匹配，并倾向于始终利用视觉分支。然而实验表明，当视频缺失、光流失效或视频来自另一个乐器类别时，错误视觉条件可能对修复结果产生负面影响。因此，模型不仅需要学习如何使用视频，还需要学习何时降低对视频的信任。

EC-VIAI-AV 的设计正是围绕这一点展开。Visual evidence 为每个样本提供视觉可靠性估计，Evidence-Aware Fusion Gate 将这种可靠性转化为视频特征的动态权重，multi-candidate generation 则在证据不足时提供多个可能补全。最终模型不再是单一的确定性修复器，而是一个 evidence-calibrated prediction system：当视觉证据强时，模型更确定地利用视频；当视觉证据弱或语义不匹配时，模型降低 gate、增加候选不确定性，并通过 uncertainty head 输出更高的样本风险。

实验结果也说明，semantic evidence 的作用不是简单提升重建指标，而是改善模型对视觉条件的判断方式。Target-specific semantic evidence 能够将 `wrong_video_cross_instrument` 的 evidence 显著压低，说明跨乐器语义错配可以被离线视觉-语言模型有效检测。但仅仅把 semantic evidence 接入模型还不够：如果训练阶段只包含正常视频，模型虽然获得了低 evidence target，却不会自动学会对 wrong video 降权。Semantic Perturbation Training 的必要性正在于此。通过在训练中显式加入 `wrong_video_cross_instrument`、`no_video` 和 `flow_zero`，模型实际见到低 evidence 输入，并学会把 semantic target 转化为 gate suppression 和 uncertainty calibration。

因此，本文的实验结论可以概括为两层。第一，EC-VIAI-AV 的多候选、candidate scorer 和 uncertainty head 相比 Original VIAI-AV 提供了更完整的修复与评估能力。第二，semantic evidence 与 perturbation training 的结合进一步解决了视觉语义不可靠的问题，使模型从“使用视频”推进到“有条件地信任视频”。这种变化对于真实音视频修复系统尤其重要，因为真实场景中的视频条件往往并不完美。

## 2. Limitations

尽管本文方法在当前实验设置下取得了较清晰的鲁棒性提升，但仍存在以下限制。

### 2.1 CLIP semantic evidence 仍是类别条件语义，而非真实音视频对齐

本文使用冻结 CLIP 作为 semantic evidence sidecar。该设计的优点是实现简单、可离线预计算、不会增加主模型训练开销，也不会破坏 VIAI-AV backbone 的可比性。但需要明确的是，当前 semantic evidence 本质上是视觉帧与乐器文本标签之间的相似度：

```text
e_{sem} = P(source_instrument | current_video_frames)
```

它并没有直接计算音频 embedding 与视频 embedding 的跨模态相似度。因此，它回答的是“当前视频是否像源音频对应的乐器类别”，而不是“当前视频和当前音频是否在细粒度事件层面对齐”。这意味着 semantic evidence 能较好处理跨乐器 wrong video，但对同一乐器内部的错配、细粒度演奏动作不一致、节奏不同步等问题仍然有限。

换言之，本文的 semantic evidence 是类别条件的视觉语义证据，而不是完整的 audio-video semantic alignment。这个限制并不削弱本文关于 cross-instrument robustness 的结论，但说明当前方法还没有完全解决所有形式的音视频不匹配。

### 2.2 相近乐器之间仍可能出现语义混淆

CLIP 的乐器识别能力并非完美。实验诊断中可以看到，saxophone、trumpet、tuba 等外观、演奏姿态或场景相近的乐器仍可能发生混淆。对于这类乐器，wrong video 的 source-instrument probability 有时仍会偏高，说明视觉语义模型难以稳定地区分所有细粒度类别。

这一现象有两个影响。第一，semantic evidence 对某些跨乐器错配的压低效果可能弱于 accordion、xylophone、guitar 等外观差异更明显的类别。第二，如果 semantic evidence 本身给出错误高分，fusion gate 也可能保留过多错误视频信息。因此，当前方法的鲁棒性上限部分受限于冻结视觉-语言模型的类别辨别能力。

本文通过 fusion weight、heuristic evidence 和 perturbation training 缓解这一问题：最终 evidence 并不完全依赖 CLIP，而是融合了低级视觉证据；模型也通过训练见过 wrong-video 场景，从而学习到一定的稳健响应。但对于更细粒度的乐器类别或更复杂的真实场景，仍需要更强的语义模型或更直接的跨模态对齐信号。

### 2.3 当前验证主要局限于 MUSICES

本文实验目前主要在 MUSICES 数据集上完成。该数据集适合研究音乐演奏场景中的音视频修复，因为它包含乐器类别、演奏视频和对应音频，便于构造 `wrong_video_cross_instrument`、`no_video` 和 `flow_zero` 等扰动协议。然而，单一数据集也限制了结论的外推范围。

不同数据集可能具有不同的视频质量、拍摄角度、乐器分布、音频混响、遮挡情况和同步误差。本文的 target-specific semantic evidence 依赖从路径或标签中推断 source instrument；在标签体系不同、类别更细或没有明确乐器标签的数据集上，需要重新设计 prompt、类别映射或 semantic evidence table。因此，当前结果主要证明方法在 MUSICES 风格音乐演奏数据上的有效性，而不能直接等价为跨数据集泛化结论。

### 2.4 语义扰动协议仍是受控扰动

本文通过 `wrong_video_cross_instrument`、`no_video` 和 `flow_zero` 构造视觉不可靠场景。这些扰动有利于可控评估，使模型行为和指标变化更容易解释。但真实世界中的视频错误可能更加连续和复杂，例如局部遮挡、弱同步偏移、同一乐器不同演奏片段错配、多乐器共现、镜头切换或画面中存在非发声主体。

因此，本文的扰动协议应被理解为对关键失败模式的受控近似，而不是对所有真实退化情况的完整覆盖。未来仍需要在更自然的 corrupted video 或 real-world mismatch 场景中验证模型。

## 3. Future Work

基于上述限制，未来工作可以从以下几个方向扩展。

第一，可以引入真正支持 audio-image 或 audio-video 对齐的多模态模型，例如 ImageBind、AudioCLIP 或其他 audio-visual contrastive models。与当前 CLIP-based semantic evidence 相比，这类模型可以直接计算音频片段和视频帧之间的 embedding similarity，从而判断当前视频是否与当前音频在语义和事件层面一致。这有望补充当前方法对同类乐器内部错配和细粒度同步错误的识别能力。

第二，可以使用更强的视觉语义模型。Larger CLIP、domain-adapted CLIP 或针对乐器演奏场景微调的视觉分类器，可能减少 saxophone/trumpet/tuba 等相近类别的混淆。更好的 prompt engineering、多模板 prompt ensemble、帧级一致性建模和视频级 temporal aggregation 也可能提升 semantic evidence 的稳定性。

第三，可以扩展跨数据集验证。未来应在更多音乐演奏数据集、不同乐器类别、更复杂拍摄条件以及非音乐音视频修复任务上测试 EC-VIAI-AV，以验证 evidence calibration 和 semantic perturbation training 是否具有通用性。跨数据集实验还可以评估 semantic evidence table 的迁移能力，以及不同标签体系下 source-condition semantic score 的可行性。

第四，可以进一步改进 evidence fusion 机制。当前使用线性融合：

```text
e = \operatorname{clamp}((1-w)e_h + w e_s, 0, 1)
```

这种形式简单、可解释，但可能无法处理 heuristic evidence 与 semantic evidence 冲突的情况。例如 wrong video 可能具有强运动证据但语义错误，no_video 则同时缺少低级和语义证据。未来可以探索 learned evidence fusion、min-based conservative fusion、uncertainty-aware fusion 或按扰动类型自适应的 gate target。

第五，可以改进 candidate scorer 与 uncertainty calibration。当前 top-1 选择依赖 proxy statistics，仍可能无法完全利用候选池中的 best-of-K 上限。未来可以设计更强的 candidate ranking module，或者将 uncertainty 与 candidate distribution、evidence inconsistency 和 audio-video alignment score 更紧密地结合。

总体而言，本文当前版本已经证明：在音视频音频修复中，引入 evidence calibration 和 semantic perturbation training 能显著改善视觉缺失和跨乐器错配下的模型行为。未来更强的跨模态语义模型和跨数据集验证，将进一步推动该框架从受控实验走向更真实、更开放的音视频修复场景。
