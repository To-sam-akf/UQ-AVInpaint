# 第 4 步：Deterministic Bottleneck Adapter

## Summary
在现有 VIAI-AV 主干上加入一个可退化的 bottleneck residual adapter。默认 VIAI-AV 路径不变；只有同时传 `--enable_ec_viai_av --deterministic_adapter --num_candidates 1` 时，adapter 插入 `MelEncoder` 的瓶颈输出与 `MelDecoderImage` 之间。初始 residual 为 0，使开启 adapter 的初始输出与 baseline 等价。

## Public Interface
- 在 `networks/EC_VIAI_Modules.py` 新增：
  ```python
  class BottleneckAdapter(nn.Module):
      def __init__(self, feature_channels=256, hidden_channels=256, init_scale=0.0)
      def forward(self, mel_bottleneck, video_feature) -> torch.Tensor
  ```
- `forward()` 输入 `mel_features[-1]` 和 `video_feature`，二者 reshape 后都必须是 `[B, 256, 1, 25]`；输出 residual 同 shape。
- checkpoint 新增可选 keys：`EvidenceEstimator`、`BottleneckAdapter`。老 checkpoint 缺少这些 key 时允许加载，新模块保持当前随机/零 residual 初始化。

## Implementation Changes
- `BottleneckAdapter` 使用轻量 Conv2d residual 分支：
  - `cat([mel_bottleneck, video_feature], dim=1)` 得到 512 通道输入。
  - `1x1 conv -> ReLU -> 3x3 conv -> ReLU -> 1x1 conv` 输出 256 通道 residual。
  - 不使用 BatchNorm/Dropout，避免 adapter 开关影响 baseline 统计。
  - `residual_scale` 为 learnable scalar，初始化为 `0.0`；卷积正常初始化、bias 为 0，保证初始 residual 精确为 0，同时 scale 第一轮可获得梯度。
- 在 `Models/VIAI_AV_inpainting.py`：
  - 初始化 `self.BottleneckAdapter`。
  - 定义 `self.use_bottleneck_adapter = enable_ec_viai_av and deterministic_adapter`。
  - `optimizer_G` 在 `self.use_bottleneck_adapter` 为真时加入 adapter 参数；baseline 默认 optimizer 不变。
  - `_forward_inpainter()` 中复制 `mel_features` list，只把主分支的 `features[-1]` 替换为 `features[-1] + adapter_residual` 后送入 `Mel_Decoder`。
  - probe branch 暂不走 adapter，保持 VIAI-AA' 辅助路径与 baseline 一致。
  - 可记录 `adapter/scale`、`adapter/residual_l1` 到 TensorBoard，便于确认从 0 residual 稳定学习。
- checkpoint 逻辑：
  - `save_checkpoint()` 在 EC adapter 启用时保存 `BottleneckAdapter.state_dict()`，并始终保存 `EvidenceEstimator.state_dict()`。
  - `load_checkpoint()` 若发现新 key 则加载；缺失则打印提示并保留新模块初始化。
  - `optimizer_G.load_state_dict()` 用兼容 wrapper；若旧 checkpoint 的 optimizer 参数组与当前 EC optimizer 不匹配，则跳过 optimizer_G 并提示，模型权重继续加载。
  - `--stochastic_adapter` 在第 4 步仍不实现；若与 `--enable_ec_viai_av` 同时使用，应明确报 `NotImplementedError`，留到第 5 步。

## Test Plan
- 语法检查：
  ```bash
  python3 -m py_compile networks/EC_VIAI_Modules.py Models/VIAI_AV_inpainting.py train_viai_av.py test_viai_av.py
  ```
- adapter 单元 smoke：
  - 构造 `mel_bottleneck/video_feature = torch.randn(B, 256, 1, 25)`。
  - 确认 residual shape 为 `[B, 256, 1, 25]`。
  - 初始 `residual.abs().max() == 0`。
  - backward 后 `residual_scale.grad` 非空且 finite。
- optimizer 检查：
  - 不带 EC 参数实例化模型，确认 `optimizer_G` 只含 `Mel_Encoder + VideoEncoder + Mel_Decoder`。
  - 带 `--enable_ec_viai_av --deterministic_adapter --num_candidates 1` 实例化模型，确认 `optimizer_G` 额外包含 `BottleneckAdapter` 参数。
- baseline 等价验证：
  - 同一 VIAI-AV checkpoint 加载两份模型：baseline 和 EC deterministic。
  - 固定 `random.seed()`、固定 `blank_length`，对同一 batch 前向。
  - 比较 `mel_pred` max abs diff、Mel L1、PSNR、SSIM；初始 diff 应为 0 或数值级极小。
- 云端训练/保存/加载 smoke：
  ```bash
  python main.py train-viai-av -- \
    --enable_ec_viai_av \
    --deterministic_adapter \
    --num_candidates 1 \
    --resume \
    --resume_path /path/to/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar \
    --reset_optimizer \
    --use_gan \
    --data_root "$DATA_ROOT" \
    --train_split_name train_av_split.txt \
    --val_split_name val_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --max_train_steps 2 \
    --checkpoint_interval 1 \
    --display_id 0
  ```
  然后用生成的 EC checkpoint 运行 `test-viai-av`，确认可加载、loss finite、JSON/CSV 正常写出。

## Assumptions
- 第 4 步只实现 deterministic adapter，不实现 K-sampling、多候选 loss、evidence gate 或 stochastic adapter。
- adapter 只影响主 `mel_pred` 路径；sync embedding、evidence score 和 probe branch 保持当前 baseline 逻辑。
- 老 VIAI-AV checkpoint 可作为 EC 初始化来源；若 optimizer 结构不兼容，跳过 optimizer 是预期行为，建议云端训练 smoke 显式传 `--reset_optimizer`。


## 实验
你要验证 Stage 4，核心不是“指标马上变好”，而是证明这件事：

`--enable_ec_viai_av --deterministic_adapter --num_candidates 1` 打开后，工程结构能稳定训练，并且初始行为几乎等价于 VIAI-AV baseline。

建议按 4 层验证。

**1. 本地结构验证**
先确认 adapter 本身是“可退化”的：

```bash
uv run python - <<'PY'
import torch
from networks.EC_VIAI_Modules import BottleneckAdapter

adapter = BottleneckAdapter()
mel = torch.randn(2, 256, 1, 13, requires_grad=True)
video = torch.randn(2, 256, 1, 13)

residual = adapter(mel, video)
print("residual shape:", tuple(residual.shape))
print("residual max abs:", float(residual.detach().abs().max()))

loss = (mel + residual).sum()
loss.backward()
print("scale grad:", adapter.residual_scale.grad)
PY
```

期望：
- shape 是 `(2, 256, 1, 13)`
- `residual max abs` 初始为 `0.0`
- `scale grad` 非空且 finite

**2. optimizer / checkpoint 验证**
确认 baseline 不受影响、EC 模式才训练 adapter：

```bash
uv run python - <<'PY'
import torch
import Options_inpainting
from Models.VIAI_AV_inpainting import VIAIAVModel

def opt_params(model):
    return sum(p.numel() for g in model.optimizer_G.param_groups for p in g["params"])

def backbone_params(model):
    return (
        sum(p.numel() for p in model.Mel_Encoder.parameters()) +
        sum(p.numel() for p in model.VideoEncoder.parameters()) +
        sum(p.numel() for p in model.Mel_Decoder.parameters())
    )

base_hp = Options_inpainting.Inpainting_Config(force_reload=True, args=[])
base = VIAIAVModel(base_hp, device=torch.device("cpu"))
print("baseline has adapter:", base.BottleneckAdapter is not None)
print("baseline optimizer ok:", opt_params(base) == backbone_params(base))

ec_hp = Options_inpainting.Inpainting_Config(force_reload=True, args=[
    "--enable_ec_viai_av", "--deterministic_adapter", "--num_candidates", "1"
])
ec = VIAIAVModel(ec_hp, device=torch.device("cpu"))
adapter_params = sum(p.numel() for p in ec.BottleneckAdapter.parameters())
print("adapter params:", adapter_params)
print("ec optimizer ok:", opt_params(ec) == backbone_params(ec) + adapter_params)
PY
```

期望：
- baseline `BottleneckAdapter is None`
- EC 模式 adapter 参数进入 `optimizer_G`

**3. 最关键：同 batch 初始等价验证**
云端有 checkpoint 和数据后，加载同一个 VIAI-AV checkpoint，构建两份模型：

- baseline：不带 EC 参数
- EC：`--enable_ec_viai_av --deterministic_adapter --num_candidates 1`

固定同一个 batch、同一个 mask，比较输出：

```python
max_abs_diff = (base_model.mel_pred - ec_model.mel_pred).abs().max()
```

期望：
- `max_abs_diff` 应该是 `0` 或非常接近 0
- `mel_l1_missing / PSNR / SSIM` 基本一致

这是 Stage 4 最重要的验证。因为 adapter 初始 residual 为 0，开启它不应该破坏 baseline。

**4. 云端训练 smoke**
用 baseline checkpoint resume，但重置 optimizer：

```bash
uv run python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --deterministic_adapter \
  --num_candidates 1 \
  --resume \
  --resume_path /path/to/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar \
  --reset_optimizer \
  --use_gan \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 2 \
  --checkpoint_interval 1 \
  --display_id 0
```

期望：
- loss 不 NaN、不爆炸
- 日志里能看到 `use_bottleneck_adapter=True`
- 旧 checkpoint 缺 `BottleneckAdapter` 时提示保持初始化
- 新 checkpoint 能保存

然后测试刚保存的 EC checkpoint：

```bash
uv run python main.py test-viai-av -- \
  --enable_ec_viai_av \
  --deterministic_adapter \
  --num_candidates 1 \
  --use_gan \
  --resume_path /path/to/EC-VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir /tmp/ec_stage4_test
```

期望 JSON/CSV 里：
- `enable_ec_viai_av=true`
- `deterministic_adapter=true`
- `num_candidates=1`
- `stage=EC-VIAI-AV-stage4-deterministic-adapter`

判定标准可以很简单：初始等价验证通过 + 2 step 训练稳定 + checkpoint 可保存/加载，就说明 Stage 4 有效。后面指标提升不是 Stage 4 的目标，那是 Stage 5 之后 stochastic / multi-candidate 的事情。