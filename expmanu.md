我建议你现在不要继续盲训。Stage10-D 已经把核心问题打通了：semantic evidence corrected 后，`wrong_video_cross_instrument` 和 `no_video` 的 gate 确实降下来了。下一步应该进入“定稿 + 消融 + 论文结果整理”。

我的建议顺序是：

**1. 确定主 checkpoint**

保留两个 checkpoint：

- **Stage10-D-95000**：质量/鲁棒性折中更好  
  - `none top1_missing_l1` 更低，正常视频损伤较小。
  - 适合作为主表中“balanced”结果。

- **Stage10-D-100000**：semantic robustness 最强  
  - `wrong_video gate_mean = 0.413`
  - `no_video gate_mean = 0.089`
  - uncertainty 相关性也更强。
  - 适合作为强调鲁棒性的 Stage10-D 最终模型。

如果只能选一个作为 Stage10-D 主结果，我偏向 **100000**，因为 Stage10-D 的目标就是让模型真正学会低 semantic evidence 下的降权。但论文/报告里要说明它牺牲了一点正常重建质量。

**2. 补齐消融实验**

接下来最重要的是做 ablation，证明 Stage10-D 的提升来自 semantic perturbation training，而不是偶然。

建议至少跑这几组：

```text
A. Stage8-90000
B. Stage10-C-95000 corrected semantic, no perturbation
C. Stage10-D-95000 fused + perturbation
D. Stage10-D-100000 fused + perturbation
```

如果时间够，再补：

```text
E. Stage10-D without wrong_video aug，只用 no_video,flow_zero
F. Stage10-D without semantic，只用 heuristic + perturbation
G. Stage10-D semantic weight 0.2 / 0.5
```

最关键的指标表：

```text
none:
  top1_missing_l1
  best_of_k_missing_l1
  psnr_missing

no_video:
  gate_mean
  uncertainty_mean
  top1_missing_l1

wrong_video_cross_instrument:
  semantic_evidence_mean
  gate_mean
  gate_target_mean
  uncertainty_mean
  uncertainty_error_spearman
  top1_missing_l1
```

**3. 做 per-sample 诊断图**

现在平均值已经很好看了，但下一步最好做图：

- `semantic_evidence` vs `gate_mean`
- `semantic_evidence` vs `uncertainty`
- `top1_missing_l1` vs `uncertainty`
- wrong video 中 source target rank 的分布
- Stage8 vs Stage10-D 的 gate 分布直方图

这些图能非常直接地说明：Stage10-D 学会了“看见语义不匹配时降低视觉信任”。

**4. 整理实验结论**

你现在可以把 Stage10-D 写成一个完整结论：

> Stage10-D 在 corrected semantic evidence 基础上加入 semantic-aware perturbation training，使模型在 no-video 和 cross-instrument wrong-video 情况下显著降低 gate，并提升 uncertainty-error 相关性。相比 Stage8，Stage10-D-100000 将 wrong-video gate 从约 0.728 降到 0.413，将 no-video gate 降到 0.089，同时保持 missing-region L1 在可接受范围。

这就是一个很清楚的贡献点。

**5. 下一阶段建议**

如果你还要继续做，我建议叫：

```text
Stage11: Semantic Evidence Fusion Refinement
```

目标不是大训，而是小规模调融合策略：

- 当前 fused 是线性：
  ```text
  e = (1-w) * heuristic + w * semantic
  ```
- 可以测试：
  ```text
  e = min(heuristic, semantic)
  e = heuristic * semantic
  e = 0.5 * heuristic + 0.5 * semantic
  ```
- 重点看 wrong-video gate 是否进一步下降，同时 `none top1_missing_l1` 不恶化。

但在进入 Stage11 前，我建议先把 **Stage10-D 的 ablation 和图表做完整**。现在已经不是“模型有没有用”的阶段了，而是“把证据链做漂亮”的阶段。这个时候稳一点，比再多跑一轮更值。


## details
下面给你一套完整操作。建议先跑 **A-D 必做组**，确认表格；E/F/G 是可选补充消融。

**0. 设置路径**

```bash
cd /root/EC-ViAv-vgpu

export DATA_ROOT=/root/shared-nvme/data
export TEST_SEM=$DATA_ROOT/semantic_evidence/clip_vit_b32/test_av_split.jsonl
export TRAIN_SEM=$DATA_ROOT/semantic_evidence/clip_vit_b32/train_av_split.jsonl

export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar

# 这里改成你失败的 Stage10-C 95000 checkpoint 实际路径
export STAGE10C_CKPT=checkpoints/formal_ec_viai_av/stage10_semantic_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar

export STAGE10D_95000_CKPT=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar
export STAGE10D_100000_CKPT=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
```

如果不确定 Stage10-C 路径，先查：

```bash
find checkpoints/formal_ec_viai_av -name '*000095000.pth.tar'
```

---

**1. A-D 必做组测试**

为了公平，A-D 都用 corrected fused semantic evidence 测同一套 perturbation。

```bash
run_eval () {
  CKPT="$1"
  OUT="$2"

  for MODE in none flow_zero no_video wrong_video_cross_instrument
  do
    python test_viai_av.py \
      --use_gan \
      --lambda_gan 0.001 \
      --enable_ec_viai_av \
      --stochastic_adapter \
      --enable_evidence_gate \
      --freeze_gate_evidence_backbone \
      --enable_evidence_scaled_sigma \
      --enable_candidate_scorer \
      --evidence_source fused \
      --semantic_evidence_path "$TEST_SEM" \
      --semantic_evidence_weight 0.35 \
      --semantic_missing_score 0.0 \
      --evidence_sigma_scale_min 0.5 \
      --evidence_sigma_scale_max 2.0 \
      --sigma_max 2.0 \
      --num_candidates 4 \
      --test_num_candidates 4 \
      --lambda_min_k 1.0 \
      --lambda_mean_k 0.1 \
      --lambda_boundary 0.05 \
      --lambda_diversity 0.05 \
      --lambda_gate_evidence 0.2 \
      --lambda_calib 0.05 \
      --calib_error_tau 0.1 \
      --resume_path "$CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$OUT/$MODE" \
      --video_perturbation "$MODE" \
      --calibration_bins 10 \
      --display_id 0
  done
}
```

然后依次跑：

```bash
run_eval "$STAGE8_CKPT" checkpoints/ablation_stage10d/A_stage8_90000_fused_readonly
run_eval "$STAGE10C_CKPT" checkpoints/ablation_stage10d/B_stage10c_95000_no_perturb
run_eval "$STAGE10D_95000_CKPT" checkpoints/ablation_stage10d/C_stage10d_95000_perturb
run_eval "$STAGE10D_100000_CKPT" checkpoints/ablation_stage10d/D_stage10d_100000_perturb
```

如果你已经跑过 C/D，可以不重跑，直接把已有结果路径用于汇总即可。

---

**2. 汇总 A-D 表格**

```bash
python - <<'PY'
import json, glob, os

experiments = {
    "A_stage8_90000": "checkpoints/ablation_stage10d/A_stage8_90000_fused_readonly",
    "B_stage10c_95000": "checkpoints/ablation_stage10d/B_stage10c_95000_no_perturb",
    "C_stage10d_95000": "checkpoints/ablation_stage10d/C_stage10d_95000_perturb",
    "D_stage10d_100000": "checkpoints/ablation_stage10d/D_stage10d_100000_perturb",
}

modes = ["none", "no_video", "wrong_video_cross_instrument"]
fields = [
    "top1_missing_l1",
    "best_of_k_missing_l1",
    "psnr_missing",
    "evidence_mean",
    "heuristic_evidence_mean",
    "semantic_evidence_mean",
    "gate_mean",
    "gate_target_mean",
    "gate_target_gap",
    "uncertainty_mean",
    "uncertainty_error_spearman",
]

print("exp,mode," + ",".join(fields))
for exp, root in experiments.items():
    for mode in modes:
        files = glob.glob(f"{root}/{mode}/*_test.json")
        if not files:
            print(f"{exp},{mode},MISSING")
            continue
        d = json.load(open(files[0]))
        vals = []
        for f in fields:
            v = d.get(f, "")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        print(f"{exp},{mode}," + ",".join(vals))
PY
```

保存成 CSV：

```bash
python - <<'PY' > checkpoints/ablation_stage10d/stage10d_ablation_summary.csv
import json, glob

experiments = {
    "A_stage8_90000": "checkpoints/ablation_stage10d/A_stage8_90000_fused_readonly",
    "B_stage10c_95000": "checkpoints/ablation_stage10d/B_stage10c_95000_no_perturb",
    "C_stage10d_95000": "checkpoints/ablation_stage10d/C_stage10d_95000_perturb",
    "D_stage10d_100000": "checkpoints/ablation_stage10d/D_stage10d_100000_perturb",
}
modes = ["none", "flow_zero", "no_video", "wrong_video_cross_instrument"]
fields = ["top1_missing_l1","best_of_k_missing_l1","psnr_missing","semantic_evidence_mean","gate_mean","gate_target_mean","uncertainty_mean","uncertainty_error_spearman"]

print("exp,mode," + ",".join(fields))
for exp, root in experiments.items():
    for mode in modes:
        files = glob.glob(f"{root}/{mode}/*_test.json")
        if not files:
            continue
        d = json.load(open(files[0]))
        vals = [f"{d.get(f, ''):.6f}" if isinstance(d.get(f, ''), float) else str(d.get(f, '')) for f in fields]
        print(f"{exp},{mode}," + ",".join(vals))
PY
```

---

**3. 可选 E：without wrong_video aug**

只训练 `no_video,flow_zero`，验证 wrong-video 降权是否真的来自 wrong-video perturbation。

```bash
export STAGE10D_E_DIR=checkpoints/formal_ec_viai_av/stage10d_ablate_no_wrong_aug

python train_viai_av.py \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_source fused \
  --semantic_evidence_path "$TRAIN_SEM" \
  --semantic_evidence_weight 0.35 \
  --semantic_missing_score 0.0 \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.35 \
  --visual_evidence_aug_modes no_video,flow_zero \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.2 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10D_E_DIR" \
  --log_event_path "$STAGE10D_E_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 95000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

测试：

```bash
export STAGE10D_E_CKPT=$STAGE10D_E_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar
run_eval "$STAGE10D_E_CKPT" checkpoints/ablation_stage10d/E_no_wrong_aug_95000
```

---

**4. 可选 F：without semantic，只用 heuristic + perturbation**

这组证明 semantic evidence 不是摆设。

```bash
export STAGE10D_F_DIR=checkpoints/formal_ec_viai_av/stage10d_ablate_heuristic_perturb

python train_viai_av.py \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_source heuristic \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.35 \
  --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.2 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10D_F_DIR" \
  --log_event_path "$STAGE10D_F_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 95000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

测试时也用 heuristic：

```bash
export STAGE10D_F_CKPT=$STAGE10D_F_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar
export TEST_ROOT=checkpoints/ablation_stage10d/F_heuristic_perturb_95000

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  python test_viai_av.py \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --enable_evidence_gate \
    --freeze_gate_evidence_backbone \
    --enable_evidence_scaled_sigma \
    --enable_candidate_scorer \
    --evidence_source heuristic \
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.2 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume_path "$STAGE10D_F_CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --results_dir "$TEST_ROOT/$MODE" \
    --video_perturbation "$MODE" \
    --calibration_bins 10 \
    --display_id 0
done
```

---

**5. 可选 G：semantic weight 0.2 / 0.5**

建议先只训到 95000。

```bash
for W in 0.2 0.5
do
  DIR=checkpoints/formal_ec_viai_av/stage10d_ablate_sem_w${W}

  python train_viai_av.py \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --enable_evidence_gate \
    --freeze_gate_evidence_backbone \
    --enable_evidence_scaled_sigma \
    --enable_candidate_scorer \
    --evidence_source fused \
    --semantic_evidence_path "$TRAIN_SEM" \
    --semantic_evidence_weight "$W" \
    --semantic_missing_score 0.0 \
    --enable_visual_evidence_aug \
    --visual_evidence_aug_prob 0.35 \
    --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero \
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.2 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume \
    --resume_path "$STAGE8_CKPT" \
    --reset_optimizer \
    --data_root "$DATA_ROOT" \
    --train_split_name train_av_split.txt \
    --val_split_name val_av_split.txt \
    --checkpoint_dir "$DIR" \
    --log_event_path "$DIR/events" \
    --batch_size 2 \
    --num_workers 4 \
    --lr 0.00002 \
    --max_train_steps 95000 \
    --checkpoint_interval 5000 \
    --print_freq 50 \
    --display_id 0
done
```

测试：

```bash
for W in 0.2 0.5
do
  CKPT=checkpoints/formal_ec_viai_av/stage10d_ablate_sem_w${W}/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar
  OUT=checkpoints/ablation_stage10d/G_sem_w${W}_95000
  run_eval "$CKPT" "$OUT"
done
```

---

**6. 推荐优先级**

我建议你实际执行顺序是：

```text
1. 先汇总已有 C/D。
2. 补跑 A：Stage8-90000 fused read-only。
3. 补跑 B：Stage10-C-95000 failed no perturbation。
4. 如果论文时间够，再跑 E 和 F。
5. G 放最后，只有当前面结果已经清楚时再跑。
```

最关键的判断：

```text
如果 D 比 A/B 在 wrong_video gate_mean 明显低，同时 none top1_missing_l1 没大崩，
Stage10-D 的主结论就成立。

如果 E 的 wrong_video gate 降不下来，
就证明 wrong_video augmentation 是必要的。

如果 F 的 wrong_video gate 高于 D，
就证明 semantic evidence 是必要的。
```