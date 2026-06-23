# 第 6 步：Multi-Candidate Training Loss

## Summary
在现有 stochastic adapter 输出 `self.mel_candidates: [B,K,1,80,200]` 的基础上，为 `VIAIAVModel` 增加多候选损失。默认权重仍为 `0.0`，所以 baseline 和 stage5 行为不变；传入 `--lambda_min_k / --lambda_mean_k / --lambda_boundary` 后才影响 `loss_total`。

## Key Changes
- 在 `Models/VIAI_AV_inpainting.py` 新增 `_multi_candidate_losses()`：
  - `L_anchor` 采用 candidate 0，因为现有代码已设置 `eps[:,0]=0` 且 `self.mel_pred = self.mel_candidates[:,0]`；baseline recon/GAN/sync 路径继续锚定 candidate 0。
  - `L_minK = mean_b min_k missing_l1(candidate_k, target)`。
  - `L_meanK = mean_{b,k} missing_l1(candidate_k, target)`。
  - `L_boundary` 对 `self.mel_completed_candidates` 在 `missing_span=(start,end)` 左右边界计算一阶时间差分误差，并跳过 `start==0` 或 `end>=T` 的无效边界，保证不越界。
- 扩展总损失：
  ```python
  loss_multi_candidate = (
      lambda_min_k * loss_min_k
      + lambda_mean_k * loss_mean_k
      + lambda_boundary * loss_boundary
  )
  loss_total = baseline_loss_total + loss_multi_candidate
  ```
- 新增并记录标量：
  - `loss_anchor`
  - `loss_min_k`
  - `loss_mean_k`
  - `loss_boundary`
  - `loss_multi_candidate`
  - `weighted_loss_min_k`
  - `weighted_loss_mean_k`
  - `weighted_loss_boundary`
  - `best_of_k_missing_l1`
  - `mean_k_missing_l1`
- 在 `get_loss_items()` 和 `TF_writer()` 写入上述新增项；TensorBoard 路径使用 `{prefix}/loss_min_k`、`{prefix}/loss_boundary`、`{prefix}/candidate/best_of_k_missing_l1` 等。
- 在 `train_viai_av.py` 和 `test_viai_av.py` 的 batch totals、summary print、JSON/CSV 字段中加入新增 loss，方便云端直接比较 `K=1` 和 `K=4`。

## Test Plan
- 语法检查：
  ```bash
  python -m py_compile Models/VIAI_AV_inpainting.py train_viai_av.py test_viai_av.py
  ```
- 增加一个轻量 CPU synthetic 检查脚本或顶层测试：
  - 构造 `B=2,K=4,T=10` 的 dummy candidates、target、mask。
  - 令 candidate 0 完全正确、其他候选在 missing region 加固定偏差，确认 `L_minK < L_meanK` 且 `K=1` 时二者等于 candidate 0 missing L1。
  - 分别构造 `missing_span=(0,3)`、`(3,7)`、`(7,10)`，确认 boundary loss finite 且不越界。
- 云端 smoke：
  ```bash
  uv run python main.py train-viai-av -- \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --resume \
    --resume_path "$VIAI_AV_CKPT" \
    --reset_optimizer \
    --data_root "$DATA_ROOT" \
    --train_split_name train_av_split.txt \
    --val_split_name val_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --max_train_steps 1 \
    --print_freq 1 \
    --display_id 0
  ```
- 验收标准：
  - 所有新增 loss 和 `loss_total` 都是 finite。
  - `best_of_k_missing_l1 <= mean_k_missing_l1`。
  - `K=1` 时 `best_of_k_missing_l1 == mean_k_missing_l1`。
  - 默认 lambdas 为 `0.0` 时，baseline 总损失不变。

## Assumptions
- `L_anchor` 固定使用 candidate 0，不使用候选均值；这是为了保持 candidate 0 与现有 `mel_pred`、GAN、测试图像和 checkpoint 兼容。
- `L_boundary` 对所有候选取平均，不只优化 best candidate，防止非最佳候选在边界处退化成噪声。
- 第 6 步不实现 scorer、uncertainty、evidence gate、candidate 保存或视频扰动；这些保留给第 7 到第 9 步。

## 验证测试
下面这组是第 6 步中等训练验证命令。假设你用已有 VIAI-AV checkpoint 继续训练：

```bash
export DATA_ROOT=/root/shared-nvme/data
export VIAI_AV_CKPT=/path/to/VIAI-AV_checkpoint_stepXXXXXXXXX.pth.tar
```

如果 checkpoint 是 PatchGAN 版本，命令里加 `--use_gan`；如果不是 PatchGAN，就不要加。

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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
  --resume_path "/root/EC-ViAv-vgpu/checkpoints/ec_viai_av_stage6_k4/EC-VIAI-AV-PatchGAN_checkpoint_step000023500.pth.tar" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir checkpoints/ec_viai_av_stage6_k4 \
  --log_event_path checkpoints/ec_viai_av_stage6_k4/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 24000 \
  --checkpoint_interval 500 \
  --print_freq 1 \
  --display_id 0
```

TensorBoard：

```bash
uv run tensorboard --logdir checkpoints/ec_viai_av_stage6_k4/events --port 6006
```

重点看：`loss_min_k`、`loss_mean_k`、`loss_boundary`、`loss_multi_candidate`、`candidate/best_of_k_missing_l1`、`candidate/mean_k_missing_l1`、`candidate/pairwise_l1`。

