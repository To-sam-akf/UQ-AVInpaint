# Semantic Perturbation Ablation Tables and Figures

## Main Quality Table

| id | method | none_top1_l1 | flow_top1_l1 | no_video_top1_l1 | wrong_top1_l1 | none_best_k_l1 | wrong_best_k_l1 | none_psnr | no_video_psnr | wrong_psnr | none_ssim | wrong_ssim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O | Original VIAI-AV | 0.063576 | 0.064570 | 0.070174 | 0.067543 | 0.063576 | 0.067543 | 22.259100 | 21.472131 | 21.711918 | 0.971799 | 0.969591 |
| A | EC-VIAI-AV Candidate-Scorer | 0.059799 | 0.060657 | 0.060114 | 0.061106 | 0.058420 | 0.059664 | 22.602434 | 22.491676 | 22.422136 | 0.975155 | 0.973893 |
| B | Semantic Evidence Fine-Tuning | 0.062668 | 0.060174 | 0.060792 | 0.064093 | 0.061179 | 0.061917 | 22.332093 | 22.410469 | 22.028406 | 0.971474 | 0.971361 |
| C | Semantic Perturbation 5k, w=0.35 | 0.062353 | 0.058096 | 0.060880 | 0.062490 | 0.060406 | 0.059467 | 22.361825 | 22.515589 | 22.314199 | 0.972749 | 0.973235 |
| D | Semantic Perturbation Final, w=0.35 | 0.061787 | 0.059554 | 0.061554 | 0.060251 | 0.059799 | 0.057581 | 22.391672 | 22.440425 | 22.627191 | 0.973075 | 0.973909 |
| E | No Wrong-Video Augmentation | 0.062441 | 0.060505 | 0.059828 | 0.063163 | 0.060501 | 0.060669 | 22.266902 | 22.589515 | 22.160035 | 0.972576 | 0.970387 |
| F | Heuristic-Only Perturbation | 0.060096 | 0.059095 | 0.059123 | 0.060119 | 0.058009 | 0.057471 | 22.522995 | 22.708039 | 22.585917 | 0.973849 | 0.974715 |
| G0.2 | Semantic Perturbation 5k, w=0.2 | 0.060408 | 0.059124 | 0.060386 | 0.060612 | 0.058712 | 0.058455 | 22.612190 | 22.539668 | 22.526218 | 0.974045 | 0.974325 |
| G0.5 | Semantic Perturbation 5k, w=0.5 | 0.061376 | 0.058062 | 0.059215 | 0.061061 | 0.059584 | 0.058504 | 22.435667 | 22.675089 | 22.440418 | 0.972661 | 0.973024 |

## Main Robustness Table

| id | method | wrong_gate | no_video_gate | wrong_unc_spearman | none_unc_spearman |
| --- | --- | --- | --- | --- | --- |
| O | Original VIAI-AV | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| A | EC-VIAI-AV Candidate-Scorer | 0.713195 | 0.367402 | 0.326222 | 0.456607 |
| B | Semantic Evidence Fine-Tuning | 0.866814 | 0.528362 | 0.354754 | 0.409318 |
| C | Semantic Perturbation 5k, w=0.35 | 0.565215 | 0.151654 | 0.454351 | 0.470074 |
| D | Semantic Perturbation Final, w=0.35 | 0.413393 | 0.089303 | 0.548191 | 0.523370 |
| E | No Wrong-Video Augmentation | 0.883072 | 0.061784 | 0.381656 | 0.466436 |
| F | Heuristic-Only Perturbation | 0.537373 | 0.053071 | 0.455732 | 0.532063 |
| G0.2 | Semantic Perturbation 5k, w=0.2 | 0.554073 | 0.140237 | 0.466262 | 0.510869 |
| G0.5 | Semantic Perturbation 5k, w=0.5 | 0.522629 | 0.157644 | 0.513842 | 0.530508 |

## Full Compact Table

| id | method | none_top1_l1 | flow_top1_l1 | no_video_top1_l1 | wrong_top1_l1 | none_best_k_l1 | wrong_best_k_l1 | none_psnr | no_video_psnr | wrong_psnr | none_ssim | wrong_ssim | wrong_gate | no_video_gate | wrong_unc_spearman | none_unc_spearman |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| O | Original VIAI-AV | 0.063576 | 0.064570 | 0.070174 | 0.067543 | 0.063576 | 0.067543 | 22.259100 | 21.472131 | 21.711918 | 0.971799 | 0.969591 | 1.000000 | 1.000000 | 0.000000 | 0.000000 |
| A | EC-VIAI-AV Candidate-Scorer | 0.059799 | 0.060657 | 0.060114 | 0.061106 | 0.058420 | 0.059664 | 22.602434 | 22.491676 | 22.422136 | 0.975155 | 0.973893 | 0.713195 | 0.367402 | 0.326222 | 0.456607 |
| B | Semantic Evidence Fine-Tuning | 0.062668 | 0.060174 | 0.060792 | 0.064093 | 0.061179 | 0.061917 | 22.332093 | 22.410469 | 22.028406 | 0.971474 | 0.971361 | 0.866814 | 0.528362 | 0.354754 | 0.409318 |
| C | Semantic Perturbation 5k, w=0.35 | 0.062353 | 0.058096 | 0.060880 | 0.062490 | 0.060406 | 0.059467 | 22.361825 | 22.515589 | 22.314199 | 0.972749 | 0.973235 | 0.565215 | 0.151654 | 0.454351 | 0.470074 |
| D | Semantic Perturbation Final, w=0.35 | 0.061787 | 0.059554 | 0.061554 | 0.060251 | 0.059799 | 0.057581 | 22.391672 | 22.440425 | 22.627191 | 0.973075 | 0.973909 | 0.413393 | 0.089303 | 0.548191 | 0.523370 |
| E | No Wrong-Video Augmentation | 0.062441 | 0.060505 | 0.059828 | 0.063163 | 0.060501 | 0.060669 | 22.266902 | 22.589515 | 22.160035 | 0.972576 | 0.970387 | 0.883072 | 0.061784 | 0.381656 | 0.466436 |
| F | Heuristic-Only Perturbation | 0.060096 | 0.059095 | 0.059123 | 0.060119 | 0.058009 | 0.057471 | 22.522995 | 22.708039 | 22.585917 | 0.973849 | 0.974715 | 0.537373 | 0.053071 | 0.455732 | 0.532063 |
| G0.2 | Semantic Perturbation 5k, w=0.2 | 0.060408 | 0.059124 | 0.060386 | 0.060612 | 0.058712 | 0.058455 | 22.612190 | 22.539668 | 22.526218 | 0.974045 | 0.974325 | 0.554073 | 0.140237 | 0.466262 | 0.510869 |
| G0.5 | Semantic Perturbation 5k, w=0.5 | 0.061376 | 0.058062 | 0.059215 | 0.061061 | 0.059584 | 0.058504 | 22.435667 | 22.675089 | 22.440418 | 0.972661 | 0.973024 | 0.522629 | 0.157644 | 0.513842 | 0.530508 |

## Figure Files

- `docs/figures/stage10d_ablation/wrong_video_gate_mean.png`
- `docs/figures/stage10d_ablation/no_video_gate_mean.png`
- `docs/figures/stage10d_ablation/wrong_video_uncertainty_spearman.png`
- `docs/figures/stage10d_ablation/top1_l1_none_vs_wrong.png`
- `docs/figures/stage10d_ablation/quality_top1_l1_by_condition.png`
- `docs/figures/stage10d_ablation/quality_psnr_by_condition.png`
- `docs/figures/stage10d_ablation/wrong_video_tradeoff_scatter.png`
- `docs/figures/stage10d_ablation/gate_mean_by_mode.png`

## Recommended Main Result

Use `D Semantic Perturbation Final, w=0.35` as the robustness-oriented final model.
It has the lowest wrong-video gate, strong no-video suppression, and the best
wrong-video uncertainty-error Spearman among the tested methods.
