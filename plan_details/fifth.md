# 第 5 步：Stochastic Bottleneck Adapter

## Summary
在现有 VIAI-AV stage4 deterministic adapter 基础上启用 `--stochastic_adapter`，把单输出扩展为 K 个候选。默认 baseline 和 deterministic adapter 路径保持不变；多候选只在 `--enable_ec_viai_av --stochastic_adapter` 时启用。

## Key Changes
- 在 `networks/EC_VIAI_Modules.py` 扩展 `BottleneckAdapter`：
  - 保留现有 `forward(mel_bottleneck, video_feature)` deterministic residual 行为。
  - 新增 `mu_head`、`logvar_head`，输入为 `cat([mel_bottleneck, video_feature], dim=1)`，输出 `[B, 256, 1, 25]`。
  - 新增 `sample_adapter(z)`，把 `z_k = mu + sigma * eps_k` 映射成 residual。
  - `logvar` clamp 到 `[-10, 2]`，`sigma` 再 clamp 到 `[sigma_min, sigma_max]`。
  - 新增 `stochastic_residual_scale`，初始化为 `1e-3`，保证初始接近 baseline 但 K>1 时有非零差异。

- 在 `Models/VIAI_AV_inpainting.py`：
  - 新增 `self.use_stochastic_adapter = enable_ec_viai_av and stochastic_adapter`。
  - 移除当前 `--stochastic_adapter` 的 `NotImplementedError`。
  - `self.use_bottleneck_adapter` 改为 deterministic 或 stochastic 任一启用。
  - 若 `num_candidates > 1` 但未启用 stochastic adapter，直接报错，避免静默忽略 K。
  - 训练时使用 `num_candidates`，eval/test 时使用 `test_num_candidates`。
  - stochastic 路径把所有 `mel_features` 和 `video_feature` 沿 batch 维展开为 `B*K`，调用同一个 `MelDecoderImage`，再 reshape 为 `[B, K, 1, 80, 200]`。
  - 保存：
    - `self.mel_candidates`: raw decoder candidates。
    - `self.mel_completed_candidates`: mask-compose 后 candidates。
    - `self.mel_pred = self.mel_candidates[:, 0]`，兼容现有训练、GAN、测试代码。
    - `self.mel_completed_pred = self.mel_completed_candidates[:, 0]`，供推理/可视化使用。
  - 非 stochastic 路径也设置 `self.mel_candidates = self.mel_pred.unsqueeze(1)`，保持统一接口。

- 训练和指标兼容：
  - 第 5 步不实现 `min-K / mean-K / boundary` 多候选损失；现有 loss 仍只使用 candidate 0。
  - `test_viai_av.py` 当前 top-1 指标继续用 mask compose，因此已知区域保持 `mel_input`，只替换 missing 区域。
  - 可在 `get_loss_items()` / TensorBoard 记录轻量诊断项：`adapter/sigma_mean`、`adapter/logvar_mean`、`candidate/pairwise_l1`。
  - checkpoint 保存 `stochastic_adapter`、`num_candidates`、`test_num_candidates`；老 checkpoint 加载时允许 stochastic 新 head missing，保持初始化。

## Test Plan
- 语法检查：
  - `python -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py train_viai_av.py test_viai_av.py`

- Adapter dummy 测试：
  - 构造 `mel_bottleneck/video_feature = torch.randn(B, 256, 1, 25)`。
  - `K=1` residual shape 为 `[B, 1, 256, 1, 25]`。
  - `K=4/8` residual shape 正确，`sigma` 在 `[sigma_min, sigma_max]`。
  - 固定 `torch.manual_seed()` 两次采样结果一致。
  - 不重置 seed 时，K>1 candidates 的 pairwise L1 大于 0。

- 模型 smoke：
  - `--enable_ec_viai_av --stochastic_adapter --num_candidates 1` 前向后 `mel_pred.shape == [B, 1, 80, 200]`，`mel_candidates.shape == [B, 1, 1, 80, 200]`。
  - `--num_candidates 4 --test_num_candidates 8` 时，训练前向输出 K=4，`model.test()` 输出 K=8。
  - 确认 `MelDecoderImage` 参数只实例化一份，decoder 权重共享。

- 云端验证：
  - 用 `batch_size=1 --max_train_steps 1 --num_candidates 4` 跑训练 smoke，确认 loss finite、显存可接受。
  - 用保存的 checkpoint 跑 `test-viai-av --test_num_candidates 4`，确认 JSON/CSV 仍写出 top-1 指标，mel 图已知区域不被污染。

## Assumptions
- 未收到进一步选择时，采用推荐接口：`self.mel_candidates` 存 raw candidates，另存 `self.mel_completed_candidates` 给推理/指标使用。
- 第 5 步只打通 stochastic K-sampling，不做 best-of-K loss、candidate scorer、uncertainty head 或 candidate 文件保存；这些留到第 6/8/9 步。
- 当前工作树已有 stage2-4 相关改动和未跟踪文件，实现时需在这些现有改动上增量修改，不回退它们。




## 建议
下面这组命令适合第 5 步验证：先 K=1 保兼容，再 K=4 验证 stochastic candidates。假设你用已有 VIAI-AV checkpoint 初始化。

先设变量：

```bash
export DATA_ROOT=/root/shared-nvme/data
export VIAI_AV_CKPT=/path/to/VIAI-AV_checkpoint_stepXXXXXXXXX.pth.tar
```

如果你的 checkpoint 是 `VIAI-AV-PatchGAN`，下面命令都加 `--use_gan`；如果不是 PatchGAN，就不要加。

**1. K=1 smoke：验证兼容性**

```bash
uv run python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 1 \
  --test_num_candidates 1 \
  --resume \
  --resume_path "$VIAI_AV_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir /tmp/ec_viai_av_stage5_k1_smoke \
  --log_event_path /tmp/ec_viai_av_stage5_k1_smoke/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 5 \
  --checkpoint_interval 5 \
  --print_freq 1 \
  --display_id 0
```

**2. K=4 short train：验证多候选采样**

```bash
uv run python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --resume \
  --resume_path "$VIAI_AV_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir checkpoints/ec_viai_av_stage5_k4 \
  --log_event_path checkpoints/ec_viai_av_stage5_k4/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 500 \
  --checkpoint_interval 100 \
  --print_freq 20 \
  --display_id 0
```

**3. K=8 显存 smoke，可选**

```bash
uv run python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 8 \
  --test_num_candidates 8 \
  --resume \
  --resume_path "$VIAI_AV_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir /tmp/ec_viai_av_stage5_k8_smoke \
  --log_event_path /tmp/ec_viai_av_stage5_k8_smoke/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 10 \
  --checkpoint_interval 10 \
  --print_freq 1 \
  --display_id 0
```

**4. 测试 K=4 checkpoint**

把路径换成实际生成的 checkpoint：

```bash
export EC_K4_CKPT=checkpoints/ec_viai_av_stage5_k4/EC-VIAI-AV_checkpoint_step000000500.pth.tar
```

```bash
uv run python main.py test-viai-av -- \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --resume_path "$EC_K4_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --results_dir checkpoints/ec_viai_av_stage5_k4_test \
  --display_id 0
```

看 TensorBoard：

```bash
uv run tensorboard --logdir checkpoints/ec_viai_av_stage5_k4/events --port 6006
```

重点看这些曲线：`loss_recon`、`loss_missing_l1`、`adapter/sigma_mean`、`adapter/logvar_mean`、`adapter/stochastic_scale`、`candidate/pairwise_l1`。K=4 时 `candidate/pairwise_l1` 应该非零，loss 不应爆炸。

