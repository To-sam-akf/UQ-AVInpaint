# Stage6 / Stage7c / Stage7d Ablation

## 目的

本实验用于验证 Stage7 系列的核心论文故事：

- Stage6：multi-candidate baseline，不显式根据视觉证据调节视频依赖或候选不确定性。
- Stage7c：加入 frozen evidence backbone 和 Evidence-Aware Fusion Gate，验证低视觉证据时 gate 会降低。
- Stage7d：在 Stage7c 基础上加入 evidence-conditioned sigma scaling，验证低视觉证据时候选不确定性和多样性会上升。

最终希望证明：

> 视觉运动证据强时，模型更信任视频，候选更集中；视觉运动证据弱、flow 缺失、静态或无视频时，模型降低视频依赖，并提高候选多样性。

## 实验设置

三组模型均在同一 test split 上测试：

- `Stage6`: `stage6_normal`
- `Stage7c`: `stage7c_normal`
- `Stage7d`: `stage7d_normal`

Controlled validation 使用同一音频样本构造以下视觉条件：

- `original`
- `flow_75`
- `flow_50`
- `flow_25`
- `flow_zero`
- `static_video_zero_flow`
- `wrong_video`
- `temporal_shift`

其中 `wrong_video` 和 `temporal_shift` 只作为 harder stress test 观察项，不作为 Stage7d 的硬成功标准。当前数据主要是室内演奏，人物动作幅度较小，细粒度时移和语义错配本身较难通过 motion evidence 稳定识别。

## Normal Test 结果

| Metric | Stage6 | Stage7c | Stage7d |
|---|---:|---:|---:|
| `mel_l1_missing` | 0.089159 | 0.118861 | **0.069959** |
| `mel_l1_full` | 0.065882 | 0.083251 | **0.048202** |
| `psnr_missing` | 20.139059 | 18.186492 | **21.634339** |
| `psnr_full` | 27.849662 | 25.840262 | **29.355774** |
| `ssim` | 0.947783 | 0.919678 | **0.967027** |
| `loss_total` | 0.539025 | 0.587779 | **0.526890** |
| `loss_recon` | 0.095748 | 0.127186 | **0.074780** |
| `loss_sync` | 0.435859 | 0.435859 | 0.435859 |
| `candidate_pairwise_distance` | 0.026912 | 0.118600 | 0.040506 |
| `gate_mean` | 1.000000 | 0.632117 | 0.686265 |
| `adapter_sigma_scale_mean` | 1.000000 | 1.000000 | **1.267553** |
| `adapter_sigma_mean` | 0.954334 | 0.954606 | **1.647209** |

### Normal Test 分析

Stage7d 在正常测试集上没有牺牲重建质量，反而优于 Stage6：

- `psnr_missing` 从 Stage6 的 `20.139059` 提升到 Stage7d 的 `21.634339`。
- `mel_l1_missing` 从 Stage6 的 `0.089159` 降到 Stage7d 的 `0.069959`。
- `ssim` 从 Stage6 的 `0.947783` 提升到 Stage7d 的 `0.967027`。

Stage7c 的正常测试质量明显变差，说明单纯加入 gate 和 evidence diversity loss 不够稳定。Stage7c 可以证明 gate 会响应 evidence，但无法可靠完成“低 evidence -> 高候选不确定性”的闭环。

Stage7d 的 normal `candidate_pairwise_distance` 为 `0.040506`，高于 Stage6 的 `0.026912`，但显著低于 Stage7c 的 `0.118600`。这符合预期：正常视频证据较强时，候选不应该无脑扩散；Stage7d 只在 evidence 弱时显式提高 sigma。

## Controlled Validation 结果

### Stage7c

| Condition | Evidence | Gate | Sigma Scale | Pairwise Distance |
|---|---:|---:|---:|---:|
| `original` | 0.329930 | 0.616220 | 1.000000 | 0.144205 |
| `flow_zero` | 0.231002 | 0.280651 | 1.000000 | 0.122805 |
| `static_video_zero_flow` | 0.229477 | 0.273173 | 1.000000 | 0.122589 |

Stage7c paired checks：

| Check | Mean Delta | Hit Rate |
|---|---:|---:|
| `gate_original_gt_flow_zero` | 0.335569 | **1.000** |
| `gate_original_gt_static_video_zero_flow` | 0.343047 | **1.000** |
| `pairwise_flow_zero_gt_original` | -0.021400 | 0.220 |
| `pairwise_static_video_zero_flow_gt_original` | -0.021616 | 0.200 |

Stage7c 结论：

- evidence 排序成立。
- gate 排序成立，说明模型确实会在低 evidence 条件下降低视频分支权重。
- 但 pairwise 排序失败：`flow_zero/static` 的候选距离低于 `original`。
- 因此 Stage7c 只完成了 evidence-aware fusion，没有完成 evidence-conditioned uncertainty。

### Stage7d

| Condition | Evidence | Gate | Sigma Scale | Pairwise Distance |
|---|---:|---:|---:|---:|
| `original` | 0.329930 | 0.731482 | 1.054056 | 0.047343 |
| `flow_75` | 0.309021 | 0.686949 | 1.175739 | 0.055154 |
| `flow_50` | 0.284536 | 0.600128 | 1.376436 | 0.072065 |
| `flow_25` | 0.254397 | 0.426159 | 1.687393 | 0.100539 |
| `flow_zero` | 0.231002 | 0.302670 | 1.862996 | 0.113556 |
| `static_video_zero_flow` | 0.229477 | 0.293259 | 1.861304 | 0.114550 |

Stage7d paired checks：

| Check | Mean Delta | Hit Rate |
|---|---:|---:|
| `evidence_original_gt_flow_75` | 0.020909 | **1.000** |
| `evidence_flow_75_gt_flow_50` | 0.024485 | **1.000** |
| `evidence_flow_50_gt_flow_25` | 0.030139 | **1.000** |
| `evidence_flow_25_gt_flow_zero` | 0.023395 | 0.860 |
| `evidence_flow_25_gt_static_video_zero_flow` | 0.024919 | 0.830 |
| `gate_original_gt_flow_zero` | 0.428812 | **1.000** |
| `gate_original_gt_static_video_zero_flow` | 0.438223 | **1.000** |
| `pairwise_flow_zero_gt_original` | 0.066212 | **0.980** |
| `pairwise_static_video_zero_flow_gt_original` | 0.067207 | **0.970** |
| `sigma_scale_original_lt_flow_25` | 0.633337 | 0.930 |
| `sigma_scale_flow_25_lt_flow_zero` | 0.175603 | 0.540 |
| `sigma_scale_flow_25_lt_static_video_zero_flow` | 0.173911 | 0.530 |

Stage7d 结论：

- controlled degradation 下 evidence 按视觉运动强弱稳定下降。
- gate 随 evidence 下降：`original gate_mean = 0.731482`，`flow_zero gate_mean = 0.302670`，`static gate_mean = 0.293259`。
- sigma scale 随 evidence 下降而升高：`original sigma_scale = 1.054056`，`flow_zero sigma_scale = 1.862996`。
- candidate pairwise distance 随低 evidence 明显升高：`original = 0.047343`，`flow_zero = 0.113556`，`static = 0.114550`。
- 最关键的 pairwise hit-rate 从 Stage7c 的 `0.220 / 0.200` 提升到 Stage7d 的 `0.980 / 0.970`。

## 核心对比

| Capability | Stage6 | Stage7c | Stage7d |
|---|---|---|---|
| Evidence estimator stable under controlled degradation | Yes | Yes | Yes |
| Gate decreases under weak visual evidence | No | Yes | Yes |
| Sigma increases under weak visual evidence | No | No | Yes |
| Low-evidence candidate diversity increases | No | No | Yes |
| Normal reconstruction quality preserved | Yes | No | Yes |

Stage7c 证明：

> Frozen evidence backbone + EvidenceFusionGate 可以让模型在视觉证据弱时降低视频依赖。

Stage7d 进一步证明：

> Evidence-conditioned sigma scaling 可以把低视觉证据转化为更高候选不确定性，从而使低 evidence 条件下候选多样性显著上升。

## 论文表述建议

推荐将 Stage7c 和 Stage7d 分开叙述：

1. Stage7c validates evidence-aware fusion.
   - 低视觉证据时 gate 下降。
   - 说明模型不再盲目信任视频分支。

2. Stage7d validates evidence-conditioned uncertainty.
   - 低视觉证据时 sigma scale 上升。
   - 候选 pairwise distance 在 `flow_zero/static` 条件下显著高于 `original`。
   - 说明模型在视觉证据不足时会表达更高不确定性，而不是输出单一过度自信结果。

建议论文中的一句总结：

> Stage7c validates evidence-aware fusion, while Stage7d further converts weak visual evidence into calibrated candidate uncertainty.

## 注意事项

- `wrong_video` 和 `temporal_shift` 当前只作为 harder stress test 观察项。由于数据主要是室内演奏，人物动作微弱，且 temporal roll 保留了大部分 motion statistics，因此不能要求第一版 evidence estimator 完全解决语义错配和细粒度时移。
- `gate_gap` 在 Stage7d 中仍然偏正，说明 gate 不是严格校准概率。论文中应将 gate 描述为 monotonic evidence-aware modulation，而不是 calibrated probability。
- Stage7d 的核心成功指标不是 normal pairwise distance 最大，而是 controlled degradation 下 `flow_zero/static` 的 pairwise distance 显著高于 `original`。

## 下一步

当前不建议继续调 Stage7d 机制。下一步可以做：

1. 固定 Stage7d 作为当前主模型版本。
2. 将 Stage6 / Stage7c / Stage7d ablation 表格整理进论文实验章节。
3. 后续若继续扩展，可在新的 stage 中专门处理 `wrong_video` 和 `temporal_shift`，例如加入更强的 AV semantic alignment 或 temporal consistency estimator。
