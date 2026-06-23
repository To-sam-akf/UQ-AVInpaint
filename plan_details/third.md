# 第 3 步：Visual Evidence Estimator

## Summary
在当前 VIAI-AV baseline 上新增一个只读证据估计分支，计算 `self.evidence_score: [B, 1]` 并写入 TensorBoard。该分支默认不参与 loss、不改 `mel_pred`、不改 optimizer，因此 baseline 输出保持不变；后续 evidence gate 再复用这个分数。

## Public Interface
- 新增 `networks/EC_VIAI_Modules.py`
- 新增类：
  ```python
  class VisualEvidenceEstimator(nn.Module):
      def forward(
          self,
          video_feature,
          flow_batch,
          mel_target_feature_flat,
          video_feature_flat,
      ) -> torch.Tensor  # [B, 1], sigmoid 到 [0, 1]
  ```
- 第一版采用确定性可解释公式，而不是随机初始化 MLP，保证未训练前也能做“normal > flow zero / wrong video”的前向验证。

## Implementation Changes
- 在 `VisualEvidenceEstimator` 中计算三类证据：
  - flow magnitude：`sqrt(flow_x^2 + flow_y^2)` 的 batch 级均值，转成 `motion_score = 1 - exp(-3 * mean_mag)`。
  - flow temporal variance：对每帧 mean magnitude 算 `var(unbiased=False)`，并加入相邻帧 magnitude diff，转成 `temporal_score = 1 - exp(-5 * temporal_signal)`。
  - sync distance：对 `mel_target_feature_flat.detach()` 和 `video_feature_flat` 做 L2 normalize，计算逐样本 L2 distance，并转为 `sync_score = 1 - clamp(distance / 2, 0, 1)`。
- 合成 evidence：
  ```python
  logit = -2.5 + 4.0 * motion_score + 2.0 * temporal_score + 2.0 * sync_score + 0.5 * feature_score
  evidence = torch.sigmoid(logit)
  ```
  其中 `feature_score = tanh(mean(abs(video_feature)))`，作为弱的视频特征活跃度补充。
- 在 `Models/VIAI_AV_inpainting.py`：
  - import `EC_VIAI_Modules`
  - `__init__` 中创建 `self.EvidenceEstimator`
  - `_forward_inpainter` 在已有 `mel_target_feature_flat` / `video_feature_flat` 后计算 `self.evidence_score`
  - 初始化并在 `get_loss_items()` 中保存 `evidence_mean_item`、`evidence_min_item`、`evidence_max_item`
  - `TF_writer()` 写入 `{prefix}/evidence/mean`、`{prefix}/evidence/min`、`{prefix}/evidence/max`
- 不把 EvidenceEstimator 参数加入 `optimizer_G`；当前模块无可训练参数，不需要改 checkpoint 保存/加载。

## Test Plan
- 语法检查：
  ```bash
  python -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py
  ```
- 轻量 shape/range 测试：
  - 构造 dummy `video_feature [B,256,1,25]`
  - 构造 dummy `flow_batch [B,50,2,256,256]`
  - 确认输出 shape 为 `[B,1]`，且所有值在 `[0,1]`
- 云端小 batch 前向验证：
  - original video
  - `flow_batch.zero_()`
  - wrong video 或 temporal shift
  - 比较 `model.evidence_score.mean()`，期望 original 高于 flow zero / wrong / shift
- baseline 不变性：
  - 在不开 `--enable_evidence_gate` 时，同一 batch 的 `mel_pred`、loss、metric 路径不因 evidence 参与 decoder 或 loss 而改变。

## Concrete Commands

运行前先按云端实际路径设置变量：

```bash
export DATA_ROOT=/root/shared-nvme/data
export VIAI_A_CKPT=checkpoints/VIAI-A_checkpoint_step000006800.pth.tar
export VIAI_AV_CKPT=checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar
```

语法检查：

```bash
python -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py
```

独立 dummy shape/range 测试，确认 `VisualEvidenceEstimator` 输出为 `[B, 1]` 且在 `[0, 1]`：

```bash
python - <<'PY'
import torch
from networks.EC_VIAI_Modules import VisualEvidenceEstimator

estimator = VisualEvidenceEstimator()
B = 2
video_feature = torch.randn(B, 256, 1, 25)
flow_batch = torch.randn(B, 50, 2, 256, 256) * 0.2
mel_target_feature_flat = torch.randn(B, 256 * 25)
video_feature_flat = video_feature.flatten(1)
evidence = estimator(video_feature, flow_batch, mel_target_feature_flat, video_feature_flat)
print("shape:", tuple(evidence.shape))
print("min:", float(evidence.min()))
print("max:", float(evidence.max()))
assert tuple(evidence.shape) == (B, 1)
assert bool(torch.all(evidence >= 0.0) and torch.all(evidence <= 1.0))
PY
```

zero-flow sanity check，确认正常 flow 的 evidence 高于 `flow_batch.zero_()`：

```bash
python - <<'PY'
import torch
from networks.EC_VIAI_Modules import VisualEvidenceEstimator

torch.manual_seed(7)
estimator = VisualEvidenceEstimator()
B = 2
video_feature = torch.randn(B, 256, 1, 25)
flow_batch = torch.randn(B, 50, 2, 32, 32).abs() * 0.2
mel_target_feature_flat = video_feature.flatten(1) + 0.05 * torch.randn(B, 256 * 25)
video_feature_flat = video_feature.flatten(1)
normal = estimator(video_feature, flow_batch, mel_target_feature_flat, video_feature_flat)
zero_flow = estimator(video_feature, torch.zeros_like(flow_batch), mel_target_feature_flat, video_feature_flat)
print("normal_mean:", float(normal.mean()))
print("zero_flow_mean:", float(zero_flow.mean()))
assert float(normal.mean()) > float(zero_flow.mean())
PY
```

模型实例化检查，确认 `EvidenceEstimator` 没有被加入 `optimizer_G`：

```bash
python - <<'PY'
import torch
import Options_inpainting
from Models.VIAI_AV_inpainting import VIAIAVModel

hparams = Options_inpainting.Inpainting_Config(force_reload=True, args=[])
model = VIAIAVModel(hparams, device=torch.device("cpu"))
evidence_params = sum(p.numel() for p in model.EvidenceEstimator.parameters())
optimizer_params = sum(p.numel() for group in model.optimizer_G.param_groups for p in group["params"])
backbone_params = (
    sum(p.numel() for p in model.Mel_Encoder.parameters())
    + sum(p.numel() for p in model.VideoEncoder.parameters())
    + sum(p.numel() for p in model.Mel_Decoder.parameters())
)
print("evidence_params:", evidence_params)
print("optimizer_matches_backbone:", optimizer_params == backbone_params)
assert evidence_params == 0
assert optimizer_params == backbone_params
PY
```

云端小 batch 前向验证。该脚本会加载训练好的 VIAI-AV-PatchGAN checkpoint，并用同一条音频样本分别比较 original、zero-flow、跨乐器 wrong video 和 temporal shift：

```bash
python - <<'PY'
import os
import torch
import Options_inpainting
from Data_loaders import audio_loader as av_loader
from Models.VIAI_AV_inpainting import VIAIAVModel


def instrument_from_batch(batch):
    path = batch[-1][0]
    parts = str(path).split("/")
    if "processed" in parts:
        index = parts.index("processed")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def evidence_for(model, batch, label):
    model.get_blank_space_length(0)
    model.set_inputs(batch)
    model.test(global_step=0)
    value = float(model.evidence_score.mean().detach().cpu())
    path = batch[-1][0]
    print(f"{label}: evidence={value:.6f} instrument={instrument_from_batch(batch)} path={path}")
    return value


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparams = Options_inpainting.Inpainting_Config(force_reload=True, args=[
    "--data_root", "/root/shared-nvme/data",
    "--test_split_name", "test_av_split.txt",
    "--batch_size", "1",
    "--num_workers", "0",
    "--use_gan",
])
data_loaders = av_loader.get_data_loaders(hparams.data_root, hparams.speaker_id, test_shuffle=False, phases=("test",))
loader = data_loaders["test"]

anchor_batch = None
wrong_batch = None
anchor_instrument = None
for batch in loader:
    if batch is None:
        continue
    current_instrument = instrument_from_batch(batch)
    if anchor_batch is None:
        anchor_batch = batch
        anchor_instrument = current_instrument
        continue
    if current_instrument != anchor_instrument:
        wrong_batch = batch
        break

if anchor_batch is None or wrong_batch is None:
    raise RuntimeError("Need at least two valid test samples from different instruments.")

ckpt_path = os.environ.get("VIAI_AV_CKPT", "checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar")
model = VIAIAVModel(hparams, device=device)
model.load_checkpoint(ckpt_path, reset_optimizer=True)

video_batch, flow_batch, c_batch, x_batch, y_batch, g_batch, input_lengths, path_batch = anchor_batch
wrong_video_batch, wrong_flow_batch = wrong_batch[0], wrong_batch[1]

zero_flow_batch = (
    video_batch,
    torch.zeros_like(flow_batch),
    c_batch,
    x_batch,
    y_batch,
    g_batch,
    input_lengths,
    path_batch,
)
wrong_video_batch = (
    wrong_video_batch,
    wrong_flow_batch,
    c_batch,
    x_batch,
    y_batch,
    g_batch,
    input_lengths,
    path_batch,
)
temporal_shift_batch = (
    video_batch.roll(shifts=5, dims=1),
    flow_batch.roll(shifts=5, dims=1),
    c_batch,
    x_batch,
    y_batch,
    g_batch,
    input_lengths,
    path_batch,
)

print("anchor instrument:", instrument_from_batch(anchor_batch))
print("wrong-video source instrument:", instrument_from_batch(wrong_batch))
original = evidence_for(model, anchor_batch, "original")
zero_flow = evidence_for(model, zero_flow_batch, "zero_flow")
wrong_video = evidence_for(model, wrong_video_batch, "wrong_cross_instrument_video")
temporal_shift = evidence_for(model, temporal_shift_batch, "temporal_shift_aux")

print("checks:")
print("zero_flow < original:", zero_flow < original)
print("wrong_cross_instrument_video < original:", wrong_video < original)
print("temporal_shift_aux is observational only:", temporal_shift)
PY
```

注意：`zero_flow < original` 是第 3 步最稳定的 sanity check；跨乐器 wrong video 加载 checkpoint 后期望下降，但不作为硬断言。`temporal_shift_aux` 只是观察值，因为 tensor `roll` 会保留大部分 motion statistics，不能等价于真正的数据层时间错位。

baseline 不变性 smoke test。不开 `--enable_evidence_gate`，仅验证第 3 步接入后训练入口仍能正常前向：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a "$VIAI_A_CKPT" \
  --checkpoint_dir /tmp/viai_av_evidence_smoke \
  --log_event_path /tmp/viai_av_evidence_smoke/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 1 \
  --print_freq 1 \
  --display_id 0
```

若使用 PatchGAN baseline checkpoint 做测试 smoke，可确认结果路径照常写出，Evidence 只记录不改输出：

```bash
python main.py test-viai-av -- \
  --use_gan \
  --name VIAI-AV-PatchGAN \
  --resume_path "$VIAI_AV_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir /tmp/viai_av_evidence_test_smoke
```

## Assumptions
- 第 3 步只实现 evidence score 与 TensorBoard 记录，不实现 gate、adapter、多候选或新增 loss。
- TensorBoard tag 按现有代码风格带 phase 前缀，即 `train/evidence/mean`、`val/evidence/mean`。
- `flow_batch` 采用当前 dataloader 输出布局 `[B, T, 2, H, W]`；实现会用 `unbiased=False` 避免小 batch / 短序列 var 产生 NaN。


## 实验
已把第 3 步 evidence 有效性验证脚本写入 [validate_evidence_estimator.py](/home/sanmu/VIAIpro/experiments/stage3/validate_evidence_estimator.py:1)，并补了 [README.md](/home/sanmu/VIAIpro/experiments/stage3/README.md:1)。脚本会输出 `evidence_validation.csv` 和 `evidence_validation_summary.json`，用于证明 evidence 随 `flow_75 -> flow_50 -> flow_25 -> flow_zero` 稳定下降。

云端运行命令：

```bash
export DATA_ROOT=/root/shared-nvme/data
export VIAI_AV_CKPT=checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar
export EVIDENCE_OUT=/tmp/evidence_estimator_validation

python experiments/stage3/validate_evidence_estimator.py \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --checkpoint "$VIAI_AV_CKPT" \
  --use_gan \
  --max_anchors 100 \
  --num_workers 0 \
  --out_dir "$EVIDENCE_OUT"
```

查看结果：

```bash
cat "$EVIDENCE_OUT/evidence_validation_summary.json"
```

本地已验证脚本语法通过；`python3` 环境缺 `numpy`，所以我用 `.venv/bin/python experiments/stage3/validate_evidence_estimator.py --help` 确认了参数入口正常。