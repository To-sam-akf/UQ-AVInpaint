## 第3步：实现Evidentce Estimator

### 实验现象
当前 evidence 更像“视觉运动证据强弱”，不是完整的“音视频匹配可信度”。

本阶段主要验证 `VisualEvidenceEstimator` 是否能够稳定量化视频证据强弱，而不是验证它是否已经具备完整的音视频错配判别能力。实验采用 paired perturbation protocol：对同一条 anchor 音频和原始视频，构造不同证据强度的视频条件，包括 `original`、逐级削弱 optical flow 的 `flow_75 / flow_50 / flow_25 / flow_zero`、静态 flow、静态视频加 zero flow、跨乐器 wrong video，以及辅助观察用的 temporal shift。

实验结果显示，evidence 对 optical flow 强度非常敏感。当 flow 从原始值逐步衰减到 75%、50%、25% 和 0 时，`evidence_score` 的均值基本单调下降，而且在 paired sample 级别也大多成立。这说明当前 evidence estimator 确实捕捉到了“视频运动证据强弱”这一核心信号。

同时，`cross_instrument_wrong_video` 的 evidence 均值低于 original，但下降不如 flow_zero 稳定。这符合当前模块定位：第3步的 estimator 是一个轻量、可解释的视觉证据强度分数，而不是一个强音视频匹配分类器。跨乐器 wrong video 仍可能包含明显演奏动作和有效 optical flow，因此不一定在每个样本上都被打成低 evidence。同时当前 evidence estimator 没有真正做“语义级乐器识别”或“细粒度动作-声音同步识别”，它只是用：
flow 强弱
flow 时间变化
audio/video embedding L2 距离
video feature 活跃度

`temporal_shift_aux` 基本没有明显下降。原因是当前 数据是室内演奏，很多视频动作确实很弱。

### 实验结果

使用 `experiments/stage3/validate_evidence_estimator.py` 在 test split 上验证 100 个 paired samples，得到以下 condition means：

```text
cross_instrument_wrong_video         0.278844
flow_25                              0.247023
flow_50                              0.277057
flow_75                              0.305204
flow_zero                            0.236960
original                             0.330855
static_flow                          0.236960
static_video_zero_flow               0.236349
temporal_shift_aux                   0.327832
```

关键 paired checks 如下：

```text
original_gt_flow_75                           mean_delta=0.025651 hit_rate=1.000 n=100
flow_75_gt_flow_50                            mean_delta=0.028147 hit_rate=1.000 n=100
flow_50_gt_flow_25                            mean_delta=0.030034 hit_rate=0.960 n=100
flow_25_gt_flow_zero                          mean_delta=0.010062 hit_rate=0.590 n=100
original_gt_flow_zero                         mean_delta=0.093895 hit_rate=0.990 n=100
original_gt_static_flow                       mean_delta=0.093895 hit_rate=0.990 n=100
original_gt_static_video_zero_flow            mean_delta=0.094507 hit_rate=0.970 n=100
original_gt_cross_instrument_wrong_video      mean_delta=0.052011 hit_rate=0.690 n=100
original_gt_temporal_shift_aux                mean_delta=0.003023 hit_rate=0.590 n=100
```

其中最重要的是 `original > flow_zero`，在 100 个 paired samples 中有 99% 成立，平均 evidence drop 为 `0.093895`。这说明当视频 motion evidence 被完全移除时，estimator 能稳定给出更低的视觉证据分数。

逐级 flow 衰减也基本符合预期：`original > flow_75 > flow_50 > flow_25 > flow_zero` 在均值上单调下降，前三个相邻比较的 hit rate 分别为 `1.000`、`1.000` 和 `0.960`。`flow_25 > flow_zero` 的 hit rate 只有 `0.590`，说明在 flow 已经很弱时，evidence 接近低位平台，剩余分数主要来自 sync score 和 feature score，因此 25% flow 与 zero flow 不一定在每个样本上都能严格区分。

`static_flow` 和 `flow_zero` 几乎相同，说明重复静态 flow 会消除 temporal evidence；`static_video_zero_flow` 进一步略低，说明静态视频加 zero flow 被 estimator 视为弱视觉证据。`cross_instrument_wrong_video` 平均低于 original，`mean_delta=0.052011`，但 hit rate 只有 `0.690`，说明它能提供一定错配敏感性，但还不能作为强错配判别器使用。

### 总结

第3步可以认为验证通过。当前 `VisualEvidenceEstimator` 能稳定量化视频证据强弱，特别是对 optical flow magnitude、temporal variance 和静态/无运动视频条件具有清晰响应。它不会改变 baseline 输出，也不会替代后续 gate、candidate scorer 或 uncertainty head，而是为这些模块提供一个轻量、可解释、可记录的 evidence input。

本阶段的结论应表述为：evidence estimator 能稳定衡量“视频中是否存在可用视觉运动证据”，但不能单独保证“视频一定与音频严格匹配”。跨乐器 wrong video 和 temporal shift 的结果说明，音视频匹配可信度仍需要后续 Evidence-Aware Fusion Gate、Candidate Scorer 和 calibration loss 共同建模。

因此，第3步为论文主线提供了基础支撑：模型可以先获得一个可解释的视觉证据分数，在视觉证据弱时减少对视频的依赖，并在后续多候选和不确定性模块中允许更高的不确定性。

## 第4步：添加确定性瓶颈适配器

### 实验现象

### 实验结果

### 总结


## 第5步： 实现 stochastic bottleneck adapter，让模型生成 K 个缺失音频候选。

### 实验现象

本阶段主要验证 stochastic bottleneck adapter 的工程可行性，而不是追求最终指标提升。模型在 VIAI-AV baseline 的 bottleneck residual 上加入随机变量，`MelDecoderImage` 权重继续共享。K=4 short train 可以正常前向、反向和写出 TensorBoard，说明多候选链路已经打通。

初始版本中，`mel_pred = mel_candidates[:, 0]`，但 candidate 0 也是随机采样得到的。由于当前第5步还没有 `min-K / mean-K / boundary` 多候选损失，原有 VIAI-AV loss 只监督 candidate 0，因此 top-1 输出会被随机噪声直接扰动。实验中可以看到 `candidate/pairwise_l1` 非零，说明候选之间确实产生差异，但 `loss_recon` 和 `loss_missing_l1` 出现不稳定上升，mel 图中的 `raw_prediction` 也出现明显劣化和黑块。

为稳定兼容路径，后续将 first candidate 改为 deterministic anchor：

```python
eps[:, 0].zero_()
```

修改后，candidate 0 变为 `z_0 = mu`，继续作为 `self.mel_pred` 供现有训练和测试代码使用；candidate 1 到 K-1 仍使用 `z_k = mu + sigma * eps_k` 生成随机候选。重新训练后，`raw_prediction` 的黑块明显减轻，`completed` 图没有破坏已知区域，同时 `candidate/pairwise_l1` 仍保持非零。

### 实验结果

第5步验证了以下结果：

- K-sampling 输出形状正确：`self.mel_candidates` 为 `[B, K, 1, 80, 200]`。
- K=1 时保持与原 `mel_pred` 兼容；K=4 时可以一次前向生成 4 个候选；测试阶段也可以通过 `test_num_candidates` 使用不同 K。
- mask compose 生效：`self.mel_completed_candidates = mel_input * (1 - mask) + mel_candidates * mask`，已知区域保持 `mel_input`，只替换 missing 区域。
- `candidate/pairwise_l1` 非零，说明 stochastic branch 产生了候选差异。
- `adapter/sigma_mean`、`adapter/logvar_mean`、`adapter/stochastic_scale` 在训练中发生变化，说明 stochastic adapter 参数收到训练信号。
- naive stochastic top-1 会导致重建不稳定；deterministic first candidate 能明显改善 top-1 路径稳定性。

需要注意的是，当前 mel 图默认只展示 candidate 0。PNG 中的 `raw_prediction` 和 `completed` 都来自 `self.mel_pred = self.mel_candidates[:, 0]`，不是 K 个候选全部展示。`raw_prediction` 是 decoder 对整张 Mel 的原始输出，最终推理和指标更应该关注 mask compose 后的 `completed`。

### 总结

第5步可以认为已经完成工程验证：VIAI-AV 可以在共享原有 `MelDecoderImage` 权重的前提下，从确定性单输出扩展为 K 个候选输出，并保持现有训练、测试和 checkpoint 路径兼容。

本阶段最重要的设计结论是：多候选模型需要稳定的 top-1 anchor。若 candidate 0 也随机采样，原有单输出 loss 会直接监督一个随机目标，导致重建 loss 和可视化结果不稳定。因此当前实现固定 candidate 0 为 deterministic anchor，仅让其余候选承担随机采样。

第5步只能证明候选生成机制可用，不能保证随机候选都是合理补全。因为当前仍只有 candidate 0 被原有重建 loss 直接监督，candidate 1 到 K-1 尚未受到 best-of-K、mean-K 或 boundary loss 约束。后续第6步应加入 `L_minK`、`L_meanK` 和 `L_boundary`，让多候选不仅有差异，而且朝合理缺失区域补全方向训练。


## 第6步：新增多候选训练损失

### 实验现象

本阶段在第5步 stochastic bottleneck adapter 的基础上加入多候选训练损失：

```text
loss_multi_candidate =
  lambda_min_k * loss_min_k
  + lambda_mean_k * loss_mean_k
  + lambda_boundary * loss_boundary
```

本次训练使用 K=4，权重为：

```text
lambda_min_k = 1.0
lambda_mean_k = 0.1
lambda_boundary = 0.05
```

训练曲线显示，多候选损失开始产生有效约束。`train/loss_min_k` 和 `train/candidate/best_of_k_missing_l1` 在后半段整体下降，说明 K 个候选中最接近 ground truth 的候选质量在变好。`train/loss_mean_k` 和 `train/candidate/mean_k_missing_l1` 中途有上升，但后半段回落，没有出现平均候选质量完全崩坏的情况。

`train/loss_boundary` 后半段下降明显，说明边界一阶差分约束对缺失片段左右边界连续性有帮助。`candidate/pairwise_l1` 从接近 0 持续升高，说明候选没有塌缩成同一个输出，而是形成了较明显的候选差异。

需要注意的是，训练阶段 `train/psnr_full` 和 `train/ssim_full` 出现明显下降，但这主要来自监控口径问题。训练监控中使用 raw decoder 输出 `mel_pred` 与完整 `mel_target` 直接比较，而最终 inpainting 输出实际是：

```python
mel_completed = mel_input * (1 - mask) + mel_pred * mask
```

也就是说，最终 completed mel 会保留 known region，只替换 missing region。因此 raw full PSNR/SSIM 会被 known region 上的 decoder 原始输出误差拖低，不能直接代表最终修复质量。测试阶段使用 completed mel 后，`psnr_full` 和 `ssim` 恢复到合理水平，也验证了这一点。

### 实验结果

使用 checkpoint：

```text
EC-VIAI-AV-PatchGAN_checkpoint_step000024000.pth.tar
```

在 test split 上测试 376 个样本，关键结果如下：

```text
loss_total              2.010418
loss_av_gen             1.425303
loss_recon              0.119143
loss_g_gan              1.306160
loss_sync               0.335745
loss_probe_gen          1.285016

loss_anchor             0.119143
loss_min_k              0.099246
loss_mean_k             0.150553
loss_boundary           0.131333
loss_multi_candidate    0.120868

best_of_k_missing_l1    0.099246
mean_k_missing_l1       0.150553
mel_l1_missing          0.107352
mel_l1_full             0.117918

psnr_full               26.468713
psnr_missing            18.688163
ssim                    0.931001
```

其中最重要的是：

```text
best_of_k_missing_l1 = 0.099246
mel_l1_missing       = 0.107352
mean_k_missing_l1    = 0.150553
```

`best_of_k_missing_l1` 低于 candidate 0 的 `mel_l1_missing`，说明 K=4 的候选池中确实存在比默认输出更接近真实缺失片段的候选。相对 candidate 0，best-of-K missing L1 约有 7.5% 改善：

```text
(0.107352 - 0.099246) / 0.107352 ≈ 7.5%
```

同时，`mean_k_missing_l1` 明显高于 best-of-K，说明候选之间存在差异，其中一部分候选质量仍然较弱。这符合第6步的阶段预期：当前还没有 candidate scorer，模型只能证明“候选池里有更好的 oracle candidate”，还不能自己选择最优候选。

`loss_multi_candidate` 的数值与权重组合一致：

```text
0.099246 + 0.1 * 0.150553 + 0.05 * 0.131333 ≈ 0.120868
```

这说明第6步新增损失的记录和加权逻辑正常。

Mel 图像整体可接受。个别样本的 `raw_prediction` 中仍可见竖条或平滑伪影，但 `completed` 经过 mask compose 后保留了已知区域，因此最终图像大部分没有明显破坏。与 ground truth 相比，缺失区域的高频纹理仍偏平滑，说明当前候选生成还不是最终形态，但边界没有出现严重断裂，整体听感风险可控。

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
