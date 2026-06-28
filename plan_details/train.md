对，下面给你两套：**正式训练** 和 **周期性 Stage9 测试**。路径按云端 `/root/EC-ViAv-vgpu` 写，你改 checkpoint 路径即可。

**0. 基础环境**

```bash
cd /root/EC-ViAv-vgpu

export DATA_ROOT=/root/shared-nvme/data
export BASELINE_CKPT=/root/EC-ViAv-vgpu/checkpoints/viai_av_patchgan_reference/VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export EXP_ROOT=checkpoints/formal_ec_viai_av
```

---

## **1. 正式训练 Stage6：multi-candidate**

如果你之前的 1-batch smoke checkpoint 只是过拟合测试，不建议从它继续。建议从 VIAI-AV / VIAI-AV-PatchGAN reference checkpoint 开始。

```bash
export STAGE6_DIR=$EXP_ROOT/stage6_multi_candidate

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --resume \
  --resume_path "$BASELINE_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE6_DIR" \
  --log_event_path "$STAGE6_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00005 \
  --max_train_steps 50000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

```
export DATA_ROOT=/root/shared-nvme/data
export STAGE6_DIR=checkpoints/formal_ec_viai_av/stage6_multi_candidate
export STAGE6_CKPT=$STAGE6_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000030000.pth.tar
```

```
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --resume \
  --resume_path "$STAGE6_CKPT" \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE6_DIR" \
  --log_event_path "$STAGE6_DIR/events" \
  --batch_size 2 \ 
  --num_workers 4 \
  --lr 0.0001 \
  --max_train_steps 50000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```
50000 跑完整扰动测试
```
export STAGE6_CKPT=checkpoints/formal_ec_viai_av/stage6_multi_candidate/EC-VIAI-AV-PatchGAN_checkpoint_step000050000.pth.tar
export TEST_ROOT=checkpoints/stage9_stage6_selected_50000

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  python test_viai_av.py \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --resume_path "$STAGE6_CKPT" \
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

## **2. 正式训练 Stage7/full gate：evidence gate + evidence-scaled sigma**

 **Stage7 正式训练操作**。当前建议是：**从 Stage6 50000 checkpoint 接着训 Stage7，开启 evidence gate + evidence-scaled sigma，但暂时不启用 semantic / candidate scorer。**

**1. 设置环境变量**

在服务器项目目录执行：

```bash
cd ~/EC-ViAv-vgpu

export DATA_ROOT=下面是一套 **Stage10 semantic evidence** 的完整操作。建议顺序是：

1. 安装 optional CLIP 依赖  
2. 离线预计算 semantic JSONL  
3. 先做 Stage10-B 只读验证  
4. 再做 Stage10-C fused 正式训练  
5. 最后跑四种 perturbation 测试

**0. 环境变量**

```bash
cd ~/EC-ViAv-vgpu

export DATA_ROOT=/root/EC-ViAv-vgpu/data
export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar

export SEM_DIR=$DATA_ROOT/semantic_evidence/clip_vit_b32
export TRAIN_SEM=$SEM_DIR/train_av_split.jsonl
export VAL_SEM=$SEM_DIR/val_av_split.jsonl
export TEST_SEM=$SEM_DIR/test_av_split.jsonl
```

确认 Stage8 checkpoint：

```bash
ls -lh "$STAGE8_CKPT"
```

**1. 安装 semantic optional dependency**

如果你用 `uv`：

```bash
uv sync --extra semantic
```

如果服务器不用 `uv`，直接 pip：

```bash
python -m pip install open_clip_torch
```

确认：

```bash
python - <<'PY'
import open_clip
print("open_clip ok")
PY
```

**2. 离线预计算 CLIP semantic evidence**

先用 5 个 sample smoke test：

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --output-dir "$SEM_DIR/smoke" \
  --num-frames 8 \
  --batch-size 16 \
  --limit 5
```

检查输出：

```bash
head -n 2 "$SEM_DIR/smoke/train_av_split.jsonl"
```

正式预计算 train / val / test：

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --split-name val_av_split.txt \
  --split-name test_av_split.txt \
  --output-dir "$SEM_DIR" \
  --model-name ViT-B-32 \
  --pretrained openai \
  --num-frames 8 \
  --batch-size 32 \
  --device auto
```

确认文件：

```bash
wc -l "$TRAIN_SEM" "$VAL_SEM" "$TEST_SEM"
head -n 1 "$TEST_SEM"
```

**3. Stage10-B：只读验证 fused / semantic / heuristic**

先不训练，用 Stage8-90000 checkpoint 比较 evidence 分布。

```bash
export TEST_ROOT=checkpoints/stage10_readonly_stage8_90000

for SRC in heuristic semantic fused
do
  for MODE in none no_video wrong_video_cross_instrument
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
      --evidence_source "$SRC" \
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
      --lambda_gate_evidence 0.1 \
      --lambda_calib 0.05 \
      --calib_error_tau 0.1 \
      --resume_path "$STAGE8_CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$TEST_ROOT/$SRC/$MODE" \
      --video_perturbation "$MODE" \
      --calibration_bins 10 \
      --display_id 0
  done
done
```

这里重点看 JSON 里的：

```text
evidence_mean
heuristic_evidence_mean
semantic_evidence_mean
gate_mean
uncertainty_mean
top1_missing_l1
```

期望：

```text
fused none evidence > wrong_video evidence > no_video evidence
semantic wrong_video_cross_instrument 明显低于 none
```

如果 `semantic` 下 wrong_video 没明显降低，说明 CLIP 语义信号失败，不建议进入 Stage10-C。

**4. Stage10-C：fused semantic evidence 正式训练**

从 Stage8-90000 开始，训练到 100000。推荐先小心一点，学习率 `2e-5`。

```bash
export STAGE10_DIR=checkpoints/formal_ec_viai_av/stage10_semantic_fused

python main.py train-viai-av -- \
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
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10_DIR" \
  --log_event_path "$STAGE10_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

启动日志必须看到：

```text
enable_candidate_scorer=True
evidence_source=fused
semantic_evidence_path=...train_av_split.jsonl
semantic_evidence_weight=0.35
[VIAI-AV] loaded semantic evidence
```

**5. TensorBoard**

```bash
tensorboard \
  --logdir checkpoints/formal_ec_viai_av/stage10_semantic_fused/events \
  --host 0.0.0.0 \
  --port 6010
```

重点看：

```text
semantic_evidence/mean
heuristic_evidence/mean
evidence/fused_mean
gate/mean
uncertainty/mean
loss/min_k
loss_candidate_scorer
loss_uncertainty_calib
```

**6. Stage10 测试**

比如测试 100000：

```bash
export STAGE10_CKPT=checkpoints/formal_ec_viai_av/stage10_semantic_fused/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
export TEST_ROOT=checkpoints/stage10_eval_100000_fused

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
    --lambda_gate_evidence 0.1 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume_path "$STAGE10_CKPT" \
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

Stage10 成功标准：

```text
none top1 不明显差于 Stage8-90000 的 0.059706
wrong_video_cross_instrument gate_mean 明显低于 Stage8 的 0.728
wrong_video_cross_instrument uncertainty_mean 高于 none
fused evidence: none > wrong_video_cross_instrument > no_video
no_video gate 继续保持低
```

最关键的一句话：**Stage10 的主要目标不是再刷 reconstruction，而是让 wrong_video_cross_instrument 被 semantic evidence 识别出来并降权。**
export STAGE6_CKPT=checkpoints/formal_ec_viai_av/stage6_multi_candidate/EC-VIAI-AV-PatchGAN_checkpoint_step000050000.pth.tar
export STAGE7_DIR=checkpoints/formal_ec_viai_av/stage7_evidence_gate_sigma
```

先确认 checkpoint 存在：

```bash
ls -lh "$STAGE6_CKPT"
```

**2. 启动 Stage7 训练**

```bash
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --resume \
  --resume_path "$STAGE6_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE7_DIR" \
  --log_event_path "$STAGE7_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00005 \
  --max_train_steps 80000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

注意看启动日志里必须出现：

```text
resume_path=...step000050000.pth.tar
enable_evidence_gate=True
enable_evidence_scaled_sigma=True
lambda_gate_evidence=0.1
lambda_diversity=0.05
reset_optimizer=True
```

**3. 训练时看 TensorBoard**

另开一个终端：

```bash
cd ~/EC-ViAv-vgpu

tensorboard \
  --logdir checkpoints/formal_ec_viai_av/stage7_evidence_gate_sigma/events \
  --host 0.0.0.0 \
  --port 6007
```

重点看这些曲线：

```text
loss/recon
loss/min_k
loss/mean_k
loss/boundary
evidence/mean
gate/mean
gate/target_mean
gate/gap
sigma/mean
```

Stage7 的理想现象是：

```text
none: gate_mean 较高
flow_zero/no_video: gate_target 降低，训练后 gate_mean 也应下降
recon/min_k 不明显劣化
```

**4. 中途 checkpoint 测试命令**

比如训到 60000 / 65000 / 70000 后，可以这样测：

```bash
export STAGE7_CKPT=checkpoints/formal_ec_viai_av/stage7_evidence_gate_sigma/EC-VIAI-AV-PatchGAN_checkpoint_step000060000.pth.tar
export TEST_ROOT=checkpoints/stage9_stage7_eval_60000

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
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.1 \
    --resume_path "$STAGE7_CKPT" \
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

**5. Stage7 是否成功的判断标准**

跟 Stage6-50000 对比，重点不是只看 `none`，而是看扰动下有没有更合理：

```text
none:
top1_missing_l1 不要明显差于 0.067

flow_zero / no_video:
gate_mean 应该下降
uncertainty 或 sigma 应该升高
evidence/gate_target 应该明显低于 none

wrong_video_cross_instrument:
如果仍然不明显下降，这是正常的，说明需要 Stage10 semantic evidence
```

推荐先训到 **60000** 测一次；如果正常，再继续到 **70000 / 80000**。Stage7 不建议盲目训练太久，核心是看 gate 和 sigma 是否开始响应 evidence。

---

## **3. 正式训练 Stage8/full：candidate scorer + uncertainty**

下面给你 **Stage8 candidate scorer / uncertainty** 的推荐操作。  
建议采用两段式：**先只训练 scorer/uncertainty head 5k step，再全量微调到 90000**。这样比直接全量训更稳。

**0. 准备环境变量**

```bash
cd ~/EC-ViAv-vgpu

export DATA_ROOT=下面是一套 **Stage10 semantic evidence** 的完整操作。建议顺序是：

1. 安装 optional CLIP 依赖  
2. 离线预计算 semantic JSONL  
3. 先做 Stage10-B 只读验证  
4. 再做 Stage10-C fused 正式训练  
5. 最后跑四种 perturbation 测试

**0. 环境变量**

```bash
cd ~/EC-ViAv-vgpu

export DATA_ROOT=/root/EC-ViAv-vgpu/data
export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar

export SEM_DIR=$DATA_ROOT/semantic_evidence/clip_vit_b32
export TRAIN_SEM=$SEM_DIR/train_av_split.jsonl
export VAL_SEM=$SEM_DIR/val_av_split.jsonl
export TEST_SEM=$SEM_DIR/test_av_split.jsonl
```

确认 Stage8 checkpoint：

```bash
ls -lh "$STAGE8_CKPT"
```

**1. 安装 semantic optional dependency**

如果你用 `uv`：

```bash
uv sync --extra semantic
```

如果服务器不用 `uv`，直接 pip：

```bash
python -m pip install open_clip_torch
```

确认：

```bash
python - <<'PY'
import open_clip
print("open_clip ok")
PY
```

**2. 离线预计算 CLIP semantic evidence**

先用 5 个 sample smoke test：

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --output-dir "$SEM_DIR/smoke" \
  --num-frames 8 \
  --batch-size 16 \
  --limit 5
```

检查输出：

```bash
head -n 2 "$SEM_DIR/smoke/train_av_split.jsonl"
```

正式预计算 train / val / test：

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --split-name val_av_split.txt \
  --split-name test_av_split.txt \
  --output-dir "$SEM_DIR" \
  --model-name ViT-B-32 \
  --pretrained openai \
  --num-frames 8 \
  --batch-size 32 \
  --device auto
```

确认文件：

```bash
wc -l "$TRAIN_SEM" "$VAL_SEM" "$TEST_SEM"
head -n 1 "$TEST_SEM"
```

**3. Stage10-B：只读验证 fused / semantic / heuristic**

先不训练，用 Stage8-90000 checkpoint 比较 evidence 分布。

```bash
export TEST_ROOT=checkpoints/stage10_readonly_stage8_90000

for SRC in heuristic semantic fused
do
  for MODE in none no_video wrong_video_cross_instrument
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
      --evidence_source "$SRC" \
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
      --lambda_gate_evidence 0.1 \
      --lambda_calib 0.05 \
      --calib_error_tau 0.1 \
      --resume_path "$STAGE8_CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$TEST_ROOT/$SRC/$MODE" \
      --video_perturbation "$MODE" \
      --calibration_bins 10 \
      --display_id 0
  done
done
```

这里重点看 JSON 里的：

```text
evidence_mean
heuristic_evidence_mean
semantic_evidence_mean
gate_mean
uncertainty_mean
top1_missing_l1
```

期望：

```text
fused none evidence > wrong_video evidence > no_video evidence
semantic wrong_video_cross_instrument 明显低于 none
```

如果 `semantic` 下 wrong_video 没明显降低，说明 CLIP 语义信号失败，不建议进入 Stage10-C。

**4. Stage10-C：fused semantic evidence 正式训练**

从 Stage8-90000 开始，训练到 100000。推荐先小心一点，学习率 `2e-5`。

```bash
export STAGE10_DIR=checkpoints/formal_ec_viai_av/stage10_semantic_fused

python main.py train-viai-av -- \
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
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10_DIR" \
  --log_event_path "$STAGE10_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

启动日志必须看到：

```text
enable_candidate_scorer=True
evidence_source=fused
semantic_evidence_path=...train_av_split.jsonl
semantic_evidence_weight=0.35
[VIAI-AV] loaded semantic evidence
```

**5. TensorBoard**

```bash
tensorboard \
  --logdir checkpoints/formal_ec_viai_av/stage10_semantic_fused/events \
  --host 0.0.0.0 \
  --port 6010
```

重点看：

```text
semantic_evidence/mean
heuristic_evidence/mean
evidence/fused_mean
gate/mean
uncertainty/mean
loss/min_k
loss_candidate_scorer
loss_uncertainty_calib
```

**6. Stage10 测试**

比如测试 100000：

```bash
export STAGE10_CKPT=checkpoints/formal_ec_viai_av/stage10_semantic_fused/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
export TEST_ROOT=checkpoints/stage10_eval_100000_fused

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
    --lambda_gate_evidence 0.1 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume_path "$STAGE10_CKPT" \
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

Stage10 成功标准：

```text
none top1 不明显差于 Stage8-90000 的 0.059706
wrong_video_cross_instrument gate_mean 明显低于 Stage8 的 0.728
wrong_video_cross_instrument uncertainty_mean 高于 none
fused evidence: none > wrong_video_cross_instrument > no_video
no_video gate 继续保持低
```

最关键的一句话：**Stage10 的主要目标不是再刷 reconstruction，而是让 wrong_video_cross_instrument 被 semantic evidence 识别出来并降权。**
export STAGE7_CKPT=checkpoints/formal_ec_viai_av/stage7_evidence_gate_sigma/EC-VIAI-AV-PatchGAN_checkpoint_step000060000.pth.tar

export STAGE8_WARMUP_DIR=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_warmup
export STAGE8_DIR=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty
```

确认路径：

```bash
ls -lh "$STAGE7_CKPT"
ls "$DATA_ROOT/train_av_split.txt" "$DATA_ROOT/val_av_split.txt"
```

**1. Stage8-A：只训练 candidate scorer / uncertainty head**

从 Stage7-60000 开始，先训到 65000：

```bash
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --train_candidate_heads_only \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.1 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE7_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE8_WARMUP_DIR" \
  --log_event_path "$STAGE8_WARMUP_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.0001 \
  --max_train_steps 65000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

启动日志里要看到：

```text
enable_candidate_scorer=True
train_candidate_heads_only=True
lambda_calib=0.1
loaded CandidateScorer
loaded UncertaintyHead
```

**2. Stage8-B：全量微调**

用 warmup 的 65000 checkpoint 继续训到 90000：

```bash
export STAGE8_WARMUP_CKPT=$STAGE8_WARMUP_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000065000.pth.tar

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_WARMUP_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE8_DIR" \
  --log_event_path "$STAGE8_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00005 \
  --max_train_steps 90000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

**3. TensorBoard**

```bash
tensorboard \
  --logdir checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/events \
  --host 0.0.0.0 \
  --port 6008
```

重点看：

```text
loss_candidate_scorer
loss_uncertainty_calib
loss_calib
weighted_loss_calib
uncertainty/mean
gate/mean
loss/min_k
loss/mean_k
```

**4. Stage8 测试命令**

先测 65000 warmup，再测 80000/85000/90000 full。比如测 90000：

```bash
export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
export TEST_ROOT=checkpoints/stage9_stage8_eval_90000

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
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.1 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume_path "$STAGE8_CKPT" \
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

Stage8 成功标准：

```text
top1 <= Stage7-60000 或接近
oracle_gain 不下降
uncertainty 在 no_video / wrong_video 下高于 none
u_top1_corr / u_top1_spearman 变成正相关
risk-coverage 曲线有用
```

如果 Stage8-90000 主质量退化，就回看 75000/80000/85000，优先选 `none top1` 不差、同时 uncertainty 对扰动更高的 checkpoint。

---

**4. 周期性 Stage9 测试脚本**

每个 checkpoint 跑 4 个核心扰动：

```bash
export TEST_ROOT=checkpoints/stage9_periodic_eval
export CKPT=$STAGE8_DIR/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  echo "===== Stage9 eval: $MODE ====="

  python main.py test-viai-av -- \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --enable_evidence_gate \
    --freeze_gate_evidence_backbone \
    --enable_evidence_scaled_sigma \
    --enable_candidate_scorer \
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.1 \
    --lambda_calib 0.1 \
    --calib_error_tau 0.1 \
    --resume_path "$CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --results_dir "$TEST_ROOT/$(basename "$CKPT" .pth.tar)/$MODE" \
    --video_perturbation "$MODE" \
    --calibration_bins 10 \
    --display_id 0
done
```

---

**5. 多 checkpoint 自动评估**

比如评估 Stage8 目录下所有 5000 interval checkpoint：

```bash
export TEST_ROOT=checkpoints/stage9_periodic_eval

for CKPT in $(ls $STAGE8_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step*.pth.tar | sort)
do
  STEP=$(basename "$CKPT" .pth.tar)
  echo "========== Evaluating $STEP =========="

  for MODE in none flow_zero no_video wrong_video_cross_instrument
  do
    python main.py test-viai-av -- \
      --use_gan \
      --lambda_gan 0.001 \
      --enable_ec_viai_av \
      --stochastic_adapter \
      --enable_evidence_gate \
      --freeze_gate_evidence_backbone \
      --enable_evidence_scaled_sigma \
      --enable_candidate_scorer \
      --evidence_sigma_scale_min 0.5 \
      --evidence_sigma_scale_max 2.0 \
      --sigma_max 2.0 \
      --num_candidates 4 \
      --test_num_candidates 4 \
      --lambda_min_k 1.0 \
      --lambda_mean_k 0.1 \
      --lambda_boundary 0.05 \
      --lambda_diversity 0.05 \
      --lambda_gate_evidence 0.1 \
      --lambda_calib 0.1 \
      --calib_error_tau 0.1 \
      --resume_path "$CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$TEST_ROOT/$STEP/$MODE" \
      --video_perturbation "$MODE" \
      --calibration_bins 10 \
      --display_id 0
  done
done
```

---

**6. 汇总每个 checkpoint 的趋势**

```bash
find "$TEST_ROOT" -name "*_test.json" -print | sort | while read f
do
  python - <<PY
import json
p="$f"
d=json.load(open(p))
print(
    p,
    d["video_perturbation"],
    "top1=", round(d.get("top1_missing_l1", 0), 6),
    "bestK=", round(d.get("best_of_k_missing_l1", 0), 6),
    "oracle=", round(d.get("oracle_gain", 0), 6),
    "u=", round(d.get("uncertainty_mean", 0), 6),
    "evidence=", round(d.get("evidence_mean", 0), 6),
    "pairwise=", round(d.get("candidate_pairwise_mel_l1", 0), 6),
)
PY
done
```

你主要看这几个趋势：

```text
best_of_k_missing_l1 <= top1_missing_l1
wrong_video_cross_instrument / no_video / flow_zero 的 uncertainty 高于 none
flow_zero / no_video 的 evidence 低于 none
低 evidence 条件下 candidate_pairwise_mel_l1 更高
top1_missing_l1 <= candidate0_missing_l1 或至少接近
```

正式论文结果应该用 Stage8/full 的完整 train split checkpoint 跑 Stage9，而不是 1-batch overfit checkpoint。

## **4.Stage10 semantic evidence**
下面按 **重新生成 corrected semantic JSONL → 重跑 Stage10-B → 诊断 → 决定是否进入 Stage10-C** 给你命令。


**hugging face镜像**
```
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_CACHE=/root/.cache/huggingface
```

**1. 同步代码到服务器**

在本地：

```bash
cd /home/sanmu/VIAIpro
```

用你 README 里的 rsync，同步到云端。注意这次要同步这些新改动：

```text
tools/precompute_semantic_evidence.py
tools/diagnose_semantic_evidence.py
utils/semantic_evidence.py
Models/VIAI_AV_inpainting.py
test_viai_av.py
tests/test_semantic_evidence.py
```

**2. 服务器设置环境变量**

在服务器：

```bash
cd ~/EC-ViAv-vgpu

export DATA_ROOT=/root/shared-nvme/data
export SEM_DIR=$DATA_ROOT/semantic_evidence/clip_vit_b32
export TRAIN_SEM=$SEM_DIR/train_av_split.jsonl
export VAL_SEM=$SEM_DIR/val_av_split.jsonl
export TEST_SEM=$SEM_DIR/test_av_split.jsonl

export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
```

确认：

```bash
ls -lh "$STAGE8_CKPT"
ls "$DATA_ROOT/train_av_split.txt" "$DATA_ROOT/val_av_split.txt" "$DATA_ROOT/test_av_split.txt"
```

**3. 安装 / 确认 open_clip**

```bash
python - <<'PY'
import open_clip
print("open_clip ok")
PY
```

如果失败：

```bash
python -m pip install open_clip_torch
```

**4. 先 smoke test 新 JSONL 字段**

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name test_av_split.txt \
  --output-dir "$SEM_DIR/smoke_corrected" \
  --model-name ViT-B-32 \
  --pretrained openai \
  --num-frames 8 \
  --batch-size 16 \
  --device auto \
  --limit 5
```

检查必须有 `probs_by_instrument` 和 `frame_top1_instruments`：

```bash
head -n 1 "$SEM_DIR/smoke_corrected/test_av_split.jsonl"
```

**5. 正式重新预计算 train / val / test**

建议覆盖旧 JSONL：

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --split-name val_av_split.txt \
  --split-name test_av_split.txt \
  --output-dir "$SEM_DIR" \
  --model-name ViT-B-32 \
  --pretrained openai \
  --num-frames 8 \
  --batch-size 32 \
  --device auto
```

确认：

```bash
wc -l "$TRAIN_SEM" "$VAL_SEM" "$TEST_SEM"

python - <<'PY'
import json, os
p = os.environ["TEST_SEM"]
r = json.loads(open(p).readline())
print(r.keys())
print("has probs_by_instrument:", "probs_by_instrument" in r)
print("has frame_top1_instruments:", "frame_top1_instruments" in r)
print("instrument:", r["instrument"])
print("top1:", r["top1_instrument"])
print("prob keys sample:", list(r["probs_by_instrument"].keys())[:10])
PY
```

**6. 重跑 Stage10-B 只读验证**

建议换一个新目录，避免和旧结果混淆：

```bash
export TEST_ROOT=checkpoints/stage10_readonly_stage8_90000_corrected

for SRC in heuristic semantic fused
do
  for MODE in none no_video wrong_video_cross_instrument
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
      --evidence_source "$SRC" \
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
      --lambda_gate_evidence 0.1 \
      --lambda_calib 0.05 \
      --calib_error_tau 0.1 \
      --resume_path "$STAGE8_CKPT" \
      --data_root "$DATA_ROOT" \
      --test_split_name test_av_split.txt \
      --batch_size 1 \
      --num_workers 0 \
      --results_dir "$TEST_ROOT/$SRC/$MODE" \
      --video_perturbation "$MODE" \
      --calibration_bins 10 \
      --display_id 0
  done
done
```

**7. 跑 corrected semantic 诊断**

```bash
python tools/diagnose_semantic_evidence.py \
  --semantic-jsonl "$TEST_SEM" \
  --sample-metrics-csv "$TEST_ROOT/fused/wrong_video_cross_instrument/sample-metrics/step000090000_perturb-wrong_video_cross_instrument_samples.csv"
```

理想输出应该类似：

```text
original_source_score_mean 高
wrong_self_score_mean 高
wrong_source_score_mean 低
wrong_top1_equals_wrong_instrument 很高
wrong_top1_equals_source_instrument 很低
```

重点不是 `wrong_self_score_mean`，它高是正常的。关键是：

```text
wrong_source_score_mean
```

它必须明显低于：

```text
original_source_score_mean
```

**8. 汇总 Stage10-B JSON 指标**

```bash
python - <<'PY'
import json, glob, os

root = os.environ["TEST_ROOT"]
sources = ["heuristic", "semantic", "fused"]
modes = ["none", "no_video", "wrong_video_cross_instrument"]
fields = [
    "top1_missing_l1",
    "evidence_mean",
    "heuristic_evidence_mean",
    "semantic_evidence_mean",
    "gate_mean",
    "gate_target_mean",
    "uncertainty_mean",
    "psnr_missing",
]

for src in sources:
    print("\n###", src)
    print("mode," + ",".join(fields))
    for mode in modes:
        p = glob.glob(f"{root}/{src}/{mode}/*_test.json")[0]
        d = json.load(open(p))
        vals = []
        for f in fields:
            v = d.get(f, "")
            vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
        print(mode + "," + ",".join(vals))
PY
```

**9. 是否进入 Stage10-C 的判断**

可以进入 Stage10-C 的条件：

```text
semantic none evidence 明显高于 wrong_video_cross_instrument
fused none evidence 明显高于 wrong_video_cross_instrument
wrong_video gate_target 不再接近 0.85
no_video semantic_evidence_mean = 0
none top1 没有明显变差
```

如果满足，再训练 Stage10-C：

```bash
export STAGE10_DIR=checkpoints/formal_ec_viai_av/stage10_semantic_fused_corrected

python main.py train-viai-av -- \
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
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10_DIR" \
  --log_event_path "$STAGE10_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

先跑到 `95000` 可以中途测一次；如果稳定再到 `100000`。
下面是一套 **Stage10 semantic evidence** 的完整操作。建议顺序是：

1. 安装 optional CLIP 依赖  
2. 离线预计算 semantic JSONL  
3. 先做 Stage10-B 只读验证  
4. 再做 Stage10-C fused 正式训练  
5. 最后跑四种 perturbation 测试

## stage10 D semantic_perturbation_training

`tests/test_stage10d_semantic_perturbation.py` 是 **pytest 单元测试**，不是可执行的命令行脚本。它包含 3 个测试：

| 测试函数 | 验证内容 |
|:---|---|
| `test_visual_evidence_aug_modes_accept_stage10d_modes` | augmentation modes 能正确解析 `wrong_video_cross_instrument,no_video,flow_zero` |
| `test_no_video_visual_evidence_aug_zeros_video_and_flow` | no_video 扰动后 video/flow 全零 |
| `test_train_wrong_video_sampler_picks_cross_instrument_sample` | train-time wrong-video sampler 能为 cello sample 找到 flute 的 wrong video |

运行方式：

```bash
cd ~/EC-ViAv-vgpu
python -m pytest tests/test_stage10d_semantic_perturbation.py -v
```

---
### 训练到95000

**实际的 Stage10-D 训练命令**（来自 `stageDSemantic_perturbationtrain.md` 第53-65行）：

```bash
export DATA_ROOT=/root/shared-nvme/data
export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
export TRAIN_SEM=$DATA_ROOT/semantic_evidence/clip_vit_b32/train_av_split.jsonl
export STAGE10D_DIR=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected

python main.py train-viai-av -- \
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
  --checkpoint_dir "$STAGE10D_DIR" \
  --log_event_path "$STAGE10D_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 95000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

启动日志必须看到：

```text
enable_visual_evidence_aug=True
visual_evidence_aug_prob=0.35
visual_evidence_aug_modes=wrong_video_cross_instrument,no_video,flow_zero
lambda_gate_evidence=0.2
```

**训练完成后测试命令**（四种扰动）：

```bash
export STAGE10D_CKPT=/root/EC-ViAv-vgpu/checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar

export TEST_SEM=$DATA_ROOT/semantic_evidence/clip_vit_b32/test_av_split.jsonl
export TEST_ROOT=checkpoints/stage10d_eval_95000

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
    --resume_path "$STAGE10D_CKPT" \
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
### 训练到10000step
下面给你 **Stage10-D 从 95000 继续到 100000** 的训练和测试命令。注意：从 **Stage10-D-95000** 继续，不要从失败的 Stage10-95000 继续。

**1. 设置环境变量**

```bash
cd ~/EC-ViAv-vgpu

export DATA_ROOT=/root/shared-nvme/data
export SEM_DIR=$DATA_ROOT/semantic_evidence/clip_vit_b32
export TRAIN_SEM=$SEM_DIR/train_av_split.jsonl
export TEST_SEM=$SEM_DIR/test_av_split.jsonl

export STAGE10D_DIR=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected
export STAGE10D_95000_CKPT=$STAGE10D_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar
```

确认：

```bash
ls -lh "$STAGE10D_95000_CKPT"
ls -lh "$TRAIN_SEM" "$TEST_SEM"
```

**2. 从 95000 继续训练到 100000**

```bash
python main.py train-viai-av -- \
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
  --resume_path "$STAGE10D_95000_CKPT" \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$STAGE10D_DIR" \
  --log_event_path "$STAGE10D_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

启动日志里确认：

```text
resume_path=...step000095000.pth.tar
enable_visual_evidence_aug=True
visual_evidence_aug_prob=0.35
visual_evidence_aug_modes=wrong_video_cross_instrument,no_video,flow_zero
lambda_gate_evidence=0.2
evidence_source=fused
semantic_evidence_path=...train_av_split.jsonl
```

**3. 测试 100000**

```bash
export STAGE10D_100000_CKPT=$STAGE10D_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
export TEST_ROOT=checkpoints/stage10d_eval_100000_perturbation
```

确认：

```bash
ls -lh "$STAGE10D_100000_CKPT"
```

跑四组：

```bash
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
    --resume_path "$STAGE10D_100000_CKPT" \
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

**4. 汇总测试结果**

```bash
python - <<'PY'
import json, glob, os

root = os.environ["TEST_ROOT"]
modes = ["none", "flow_zero", "no_video", "wrong_video_cross_instrument"]
fields = [
    "top1_missing_l1",
    "best_of_k_missing_l1",
    "evidence_mean",
    "heuristic_evidence_mean",
    "semantic_evidence_mean",
    "gate_mean",
    "gate_target_mean",
    "gate_target_gap",
    "uncertainty_mean",
    "uncertainty_error_spearman",
    "psnr_missing",
]

print("mode," + ",".join(fields))
for mode in modes:
    p = glob.glob(f"{root}/{mode}/*_test.json")[0]
    d = json.load(open(p))
    vals = []
    for f in fields:
        v = d.get(f, "")
        vals.append(f"{v:.6f}" if isinstance(v, float) else str(v))
    print(mode + "," + ",".join(vals))
PY
```

**5. 再跑 wrong-video semantic 诊断**

```bash
python tools/diagnose_semantic_evidence.py \
  --semantic-jsonl "$TEST_SEM" \
  --sample-metrics-csv "$TEST_ROOT/wrong_video_cross_instrument/sample-metrics/step000100000_perturb-wrong_video_cross_instrument_samples.csv"
```

**6. 选择 95000 还是 100000**

选 `100000` 的条件：

```text
none top1_missing_l1 <= 0.0610
wrong_video gate_mean < 0.55
no_video gate_mean <= 0.153 或接近
wrong_video uncertainty_mean > none uncertainty_mean
wrong_video semantic_evidence_mean ≈ 0.052
```

否则回退用 `95000`：

```text
none top1 = 0.060798
wrong_video gate = 0.566
no_video gate = 0.153
```