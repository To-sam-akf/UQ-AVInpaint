# Stage8 Candidate Scorer 与 Uncertainty Calibration 结果

本文档记录当前 Stage8 主结果。Stage8 在 Stage7d 多候选生成链路上加入
`CandidateScorer` 和 `UncertaintyHead`，使用 scorer hard argmax 选择 top-1 candidate，
并训练 uncertainty score 与真实错误正相关。

## 主结果选择

当前将 low-lr sweep 的 step `27750` 作为 Stage8 主结果。

| 字段 | 值 |
| --- | --- |
| 实验名称 | `stage8_candidate_scorer_lowlr_sweep` |
| checkpoint | `checkpoints/stage8_candidate_scorer_lowlr_sweep/EC-VIAI-AV-PatchGAN_checkpoint_step000027750.pth.tar` |
| 测试 JSON | `checkpoints/stage8_candidate_scorer_lowlr_sweep_test_step27750/EC-VIAI-AV-PatchGAN_step000027750_test.json` |
| checkpoint step | `27750` |
| stage | `EC-VIAI-AV-stage8-candidate-scorer-calib` |
| test split | `test_av_split.txt` |
| test samples | `376` |
| `num_candidates` / `test_num_candidates` | `4` / `4` |
| scorer 输出策略 | hard argmax top-1 |
| `lambda_calib` | `0.1` |

## Stage8 主指标

| 指标 | 值 |
| --- | ---: |
| `top1_missing_l1` / `mel_l1_missing` | `0.066448638` |
| `candidate0_missing_l1` | `0.066803026` |
| `best_of_k_missing_l1` | `0.064646071` |
| `random_expected_missing_l1` | `0.095471428` |
| `oracle_gain = top1 - best_of_k` | `0.001802567` |
| `mel_l1_full` | `0.049125503` |
| `psnr_missing` | `22.162446494` |
| `psnr_full` | `30.005445011` |
| `ssim` | `0.968745863` |
| `loss_candidate_scorer` | `1.197279264` |
| `loss_uncertainty_calib` | `0.005648712` |
| `uncertainty_error_corr` | `0.413092152` |
| `uncertainty_error_spearman` | `0.460230739` |
| `uncertainty_best_error_corr` | `0.412686772` |
| `uncertainty_best_error_spearman` | `0.462385149` |
| `candidate_pairwise_distance` | `0.057882457` |
| `gate_mean` | `0.634836607` |
| `adapter_sigma_mean` | `1.644178948` |
| `adapter_sigma_scale_mean` | `1.267553285` |

## Baseline 对照

Stage7d normal 来自 `docs/Stage6_Stage7c_Stage7d Ablation.md`。`VIAI-AV-PatchGAN reference`
来自 `docs/baseline_reference.md`，作为原始 PatchGAN baseline 参照。

| 指标 | VIAI-AV-PatchGAN reference | Stage7d normal | Stage8 step27750 |
| --- | ---: | ---: | ---: |
| `mel_l1_missing` / top-1 missing L1 | `0.066439455` | `0.069959` | `0.066448638` |
| `mel_l1_full` | `0.040078492` | `0.048202` | `0.049125503` |
| `psnr_missing` | `21.879183181` | `21.634339` | `22.162446494` |
| `psnr_full` | `29.092075267` | `29.355774` | `30.005445011` |
| `ssim` | `0.966241622` | `0.967027` | `0.968745863` |
| `candidate_pairwise_distance` | N/A | `0.040506` | `0.057882457` |
| `gate_mean` | N/A | `0.686265` | `0.634836607` |
| `adapter_sigma_scale_mean` | N/A | `1.267553` | `1.267553285` |
| `adapter_sigma_mean` | N/A | `1.647209` | `1.644178948` |

相对 Stage7d normal，Stage8 step27750 的主要变化：

| 指标 | 变化 |
| --- | ---: |
| `mel_l1_missing` | `0.069959 -> 0.066448638`，降低 `0.003510362`，约 `5.02%` |
| `psnr_missing` | `21.634339 -> 22.162446494`，提升 `0.528107494` dB |
| `ssim` | `0.967027 -> 0.968745863`，提升 `0.001718863` |
| `candidate_pairwise_distance` | `0.040506 -> 0.057882457`，候选差异更大 |

## Candidate Scorer 结论

Stage8 的 top-1 输出来自 scorer hard argmax，而不是 oracle best-of-K。测试集上：

| 对比 | 结果 |
| --- | ---: |
| top-1 vs candidate0 | `0.066448638 < 0.066803026` |
| top-1 vs random expected | `0.066448638 < 0.095471428` |
| best-of-K vs top-1 | `0.064646071 < 0.066448638` |

结论：

- Stage8 top-1 优于 Stage7d normal 的 missing L1。
- scorer 选择出的 top-1 优于默认 `candidate0`。
- top-1 明显优于随机候选期望。
- best-of-K 仍优于 top-1，说明候选池中仍有 oracle headroom，后续还有改进 scorer 的空间。

## Uncertainty Calibration 结论

Stage8 的 uncertainty score 与真实错误呈明显正相关：

| 相关性 | 值 |
| --- | ---: |
| `u` vs top-1 error Pearson | `0.413092152` |
| `u` vs top-1 error Spearman | `0.460230739` |
| `u` vs best-of-K error Pearson | `0.412686772` |
| `u` vs best-of-K error Spearman | `0.462385149` |

结论：

- uncertainty-error Pearson 和 Spearman 均为正，且强于 smoke / short validation。
- `u` 不只是数值存在，而是已经能反映样本难度排序。
- 当前可将 uncertainty head 描述为有效的 error-aware uncertainty signal。

## 当前判断

step `27750` 是当前 Stage8 最合适的主结果：它同时满足重建质量、top-1 选择和
uncertainty calibration 三个目标。后续如果继续优化，不建议直接从更晚的 `28100` 或
`30100` checkpoint 继续训练，因为更长训练已经观察到候选质量回落。更合理的方向是固定
generator / adapter，仅继续训练 `CandidateScorer` 和 `UncertaintyHead`，以提高 scorer
选择能力而不破坏候选生成质量。
