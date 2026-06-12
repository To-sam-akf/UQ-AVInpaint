## P0 操作：冻结并记录 VIAI 基线

### 1. 进入项目并启用环境

```bash
cd /home/sanmu/VIAIpro
source .venv/bin/activate
```

确认统一入口可用：

```bash
python main.py freeze-viai-baselines -- --help
```

### 2. 配置数据、checkpoint 和输出路径

`DATA_ROOT` 下必须包含 VIAI-A、VIAI-AV 的六份 split 文件，以及 split
中引用的 Mel、音频、RGB 和 optical flow 数据。

```bash
export DATA_ROOT="/path/to/data"
export VIAI_A_CKPT="/path/to/checkpoints/VIAI-A_checkpoint_stepXXXXXXXXX.pth.tar"
export VIAI_AV_CKPT="/path/to/checkpoints/VIAI-AV_checkpoint_stepXXXXXXXXX.pth.tar"
export BASELINE_ROOT="/path/to/experiments/baselines"
```

默认读取以下文件：

```text
train_viai_a_split.txt
val_viai_a_split.txt
test_viai_a_split.txt
train_av_split.txt
val_av_split.txt
test_av_split.txt
```

### 3. 检查 Git 状态

默认只允许在干净的 Git worktree 上冻结基线：

```bash
git status --short
```

建议先提交当前实现，再运行正式基线。输出目录必须不存在或为空，程序不会覆盖已有结果。

### 4. 执行 P0

```bash
python main.py freeze-viai-baselines -- \
  --data-root "$DATA_ROOT" \
  --viai-a-checkpoint "$VIAI_A_CKPT" \
  --viai-av-checkpoint "$VIAI_AV_CKPT" \
  --output-root "$BASELINE_ROOT" \
  --batch-size 16 \
  --num-workers 4 \
  --seed 1234 \
  --min-gap-frames 20 \
  --max-gap-frames 50
```

该命令会依次执行：

```text
test-viai-a
test-viai-av --eval-branch av
test-viai-av --eval-branch probe
```

三个分支统一使用冻结后的 AV test split 和同一份固定 mask manifest。
runner 会自动检查 checkpoint 是否包含 PatchGAN，并为子命令添加 `--use_gan`。
VIAI-AV checkpoint 必须记录 `enable_probe_loss=True`，否则 probe 基线会被拒绝。

### 5. 可选参数

评测未提交代码时，必须显式允许脏工作区：

```bash
python main.py freeze-viai-baselines -- \
  --data-root "$DATA_ROOT" \
  --viai-a-checkpoint "$VIAI_A_CKPT" \
  --viai-av-checkpoint "$VIAI_AV_CKPT" \
  --output-root "$BASELINE_ROOT" \
  --batch-size 16 \
  --num-workers 4 \
  --allow-dirty
```

导出少量 wav：

```bash
python main.py freeze-viai-baselines -- \
  --data-root "$DATA_ROOT" \
  --viai-a-checkpoint "$VIAI_A_CKPT" \
  --viai-av-checkpoint "$VIAI_AV_CKPT" \
  --output-root "$BASELINE_ROOT" \
  --use-vocoder \
  --vocoder-max-samples 20
```

使用自定义 split 文件名时：

```bash
python main.py freeze-viai-baselines -- \
  --data-root "$DATA_ROOT" \
  --viai-a-checkpoint "$VIAI_A_CKPT" \
  --viai-av-checkpoint "$VIAI_AV_CKPT" \
  --output-root "$BASELINE_ROOT" \
  --train-viai-a-split train_viai_a_split.txt \
  --val-viai-a-split val_viai_a_split.txt \
  --test-viai-a-split test_viai_a_split.txt \
  --train-av-split train_av_split.txt \
  --val-av-split val_av_split.txt \
  --test-av-split test_av_split.txt
```

### 6. 检查输出

```bash
find "$BASELINE_ROOT" -maxdepth 3 -type f | sort
```

核心输出结构：

```text
experiments/baselines/
  protocol/
    splits/
    test_masks.jsonl
    protocol.json
  viai_a/
    summary.json
    samples.jsonl
    metrics.csv
    run_metadata.json
    mel-image/
    wav/
  viai_av/
    ...
  viai_aa_probe/
    ...
  suite.json
```

快速检查三个分支的汇总指标：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["BASELINE_ROOT"])
for name in ("viai_a", "viai_av", "viai_aa_probe"):
    with (root / name / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    print(
        name,
        "samples=", summary["num_samples"],
        "L1=", summary["mel_l1_full"],
        "PSNR=", summary["psnr_full"],
        "SSIM=", summary["ssim"],
    )
PY
```

### 7. 常见失败

- `Git worktree is dirty`：提交修改，或明确添加 `--allow-dirty`。
- `Baseline output directory is not empty`：更换新的输出目录。
- `split leakage`：同一 source video 同时出现在 train、val 或 test 中。
- `split assignment conflict`：VIAI-A 与 VIAI-AV 对同一 source video 的集合归属不一致。
- `Incomplete AV test sample`：test clip 缺少 200 帧 Mel、音频或必需的 RGB/flow 帧。
- `enable_probe_loss=True` 相关错误：VIAI-AV checkpoint 未包含可用的 VIAI-AA' probe。

## P1 操作：数据契约、mask 和视觉退化

### 1. P1 整体流程

P1 不训练模型，主要完成以下数据准备和检查：

```text
检查 AV split 和样本完整性
        ↓
为训练集计算 spectral-flux/onset 元数据
        ↓
为 val/test 生成固定的四类 mask manifest
        ↓
通过 UQAVDataset 读取 Mel、音频、RGB 和 optical flow
        ↓
生成 mel_corrupted、missing_mask 和 boundary_map
        ↓
按需应用可复现的视觉退化
        ↓
检查 batch 数据契约
```

训练阶段的 mask 根据 `seed + epoch + sample_id` 独立生成。验证和测试阶段
不再随机生成 mask，而是读取固定 manifest。原 VIAI-A/VIAI-AV loader 不受影响。

### 2. 进入项目并配置数据目录

```bash
cd /home/sanmu/VIAIpro
source .venv/bin/activate

export DATA_ROOT="/path/to/data"
export UQ_METADATA_ROOT="$DATA_ROOT/uq_metadata"
```

确认 P1 命令已经注册：

```bash
python main.py prepare-uq-metadata -- --help
```

如果使用 `uv` 管理环境，也可以将本节命令中的 `python` 替换为：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python
```

### 3. 输入数据要求

默认读取：

```text
$DATA_ROOT/train_av_split.txt
$DATA_ROOT/val_av_split.txt
$DATA_ROOT/test_av_split.txt
```

每一行格式为：

```text
sample_dir|mel_path|audio_path|mel_frames
```

例如：

```text
processed/piano/video_id/shot_000000/clip_000001|processed/piano/video_id/shot_000000/clip_000001/mel.npy|processed/piano/video_id/shot_000000/clip_000001/raw_audio.npy|200
```

每个样本必须满足：

```text
mel.npy:                  [200, 80]
raw_audio.npy:            [64000]
image_crop/1.jpg...50.jpg
flow_x_crop/1.jpg...50.jpg
flow_y_crop/1.jpg...50.jpg
```

快速检查 split 数量：

```bash
wc -l \
  "$DATA_ROOT/train_av_split.txt" \
  "$DATA_ROOT/val_av_split.txt" \
  "$DATA_ROOT/test_av_split.txt"

head -n 3 "$DATA_ROOT/train_av_split.txt"
```

### 4. 生成 P1 metadata

使用默认参数：

```bash
python main.py prepare-uq-metadata -- \
  --data-root "$DATA_ROOT"
```

默认配置为：

```text
seed:                    1234
Mel:                     200 x 80
audio:                   64000 samples
visual frames:           50
boundary margin:         3 Mel frames
random gap:              20-50 Mel frames
long gap:                60, 80, 100 Mel frames
```

显式写出全部关键参数的正式命令：

```bash
python main.py prepare-uq-metadata -- \
  --data-root "$DATA_ROOT" \
  --train-split-name train_av_split.txt \
  --val-split-name val_av_split.txt \
  --test-split-name test_av_split.txt \
  --output-dir "$UQ_METADATA_ROOT" \
  --seed 1234 \
  --mel-frames 200 \
  --mel-bins 80 \
  --audio-steps 64000 \
  --visual-frames 50 \
  --boundary-margin 3 \
  --min-gap-frames 20 \
  --max-gap-frames 50 \
  --long-gap-frames 60 80 100
```

该命令会先严格检查全部 split 样本。任意样本 shape 不符、帧目录不完整，
或者帧编号不是连续的 `1.jpg` 到 `50.jpg`，命令都会立即失败。

### 5. Metadata 输出

默认输出到：

```text
$DATA_ROOT/uq_metadata/
  train_onsets/
    processed/<instrument>/<video>/<shot>/<clip>.npy
  val_masks.jsonl
  test_masks.jsonl
  metadata_summary.json
```

检查输出：

```bash
find "$UQ_METADATA_ROOT" -type f | sort | head -n 30

head -n 2 "$UQ_METADATA_ROOT/val_masks.jsonl"
head -n 2 "$UQ_METADATA_ROOT/test_masks.jsonl"

python -m json.tool \
  "$UQ_METADATA_ROOT/metadata_summary.json" | sed -n '1,220p'
```

每个 val/test 样本固定包含四个 variant：

```text
random
onset_centered
boundary_near
long_gap
```

manifest 中的 `start` 为缺失区左边界，`end` 为右侧开区间：

```text
缺失区域 = [start, end)
gap_frames = end - start
mask[..., start:end] = 1
```

### 6. 检查 metadata 可复现性

使用同一数据、split、seed 和参数运行两次，JSON 输出应完全一致：

```bash
export UQ_METADATA_A="/tmp/uq_metadata_a"
export UQ_METADATA_B="/tmp/uq_metadata_b"

python main.py prepare-uq-metadata -- \
  --data-root "$DATA_ROOT" \
  --output-dir "$UQ_METADATA_A" \
  --seed 1234

python main.py prepare-uq-metadata -- \
  --data-root "$DATA_ROOT" \
  --output-dir "$UQ_METADATA_B" \
  --seed 1234

sha256sum \
  "$UQ_METADATA_A/val_masks.jsonl" \
  "$UQ_METADATA_B/val_masks.jsonl"

sha256sum \
  "$UQ_METADATA_A/test_masks.jsonl" \
  "$UQ_METADATA_B/test_masks.jsonl"

cmp \
  "$UQ_METADATA_A/metadata_summary.json" \
  "$UQ_METADATA_B/metadata_summary.json"
```

两组对应文件的 SHA256 应一致，`cmp` 不应输出差异。

### 7. UQ loader 输出契约

`UQAVDataset` 经 DataLoader collate 后返回：

```text
sample_id:          list[str]
mel_target:         Tensor[B, 1, 80, 200]
mel_corrupted:      Tensor[B, 1, 80, 200]
missing_mask:       Tensor[B, 1, 80, 200]
boundary_map:       Tensor[B, 2, 80, 200]
video:              Tensor[B, 50, 3, 256, 256]
flow:               Tensor[B, 50, 2, 256, 256]
audio_target:       Tensor[B, 64000]
mask_spec:          list[MaskSpec]
video_condition:    list[str]
video_degradation:  list[dict]
```

其中：

```text
missing_mask = 1：需要生成的区域
missing_mask = 0：必须保留的已知区域
boundary_map[:, 0]：到左边界 start 的归一化距离
boundary_map[:, 1]：到右边界 end - 1 的归一化距离
```

### 8. 训练 loader smoke test

训练 loader 默认启用四类 mask，但视觉条件默认为 `original`。每个样本在同一个
epoch 内结果稳定，不同样本独立采样。训练循环每进入一个新 epoch，需要调用
`loader.dataset.set_epoch(epoch)`。

```bash
python - <<'PY'
import os

from Data_loaders.uq_av_loader import create_uq_av_dataloader

data_root = os.environ["DATA_ROOT"]
metadata_root = os.environ["UQ_METADATA_ROOT"]

loader = create_uq_av_dataloader(
    data_root=data_root,
    split_name="train_av_split.txt",
    phase="train",
    metadata_dir=metadata_root,
    batch_size=2,
    num_workers=0,
    pin_memory=False,
    seed=1234,
)

loader.dataset.set_epoch(0)
batch = next(iter(loader))

for name in (
    "mel_target",
    "mel_corrupted",
    "missing_mask",
    "boundary_map",
    "video",
    "flow",
    "audio_target",
):
    print(name, tuple(batch[name].shape), batch[name].dtype)

print("sample_id:", batch["sample_id"])
print("mask_spec:", batch["mask_spec"])
print("video_condition:", batch["video_condition"])
print("video_degradation:", batch["video_degradation"])
PY
```

云端正式训练时可改为：

```text
batch_size=16
num_workers=4
pin_memory=True
```

### 9. 训练时启用视觉退化

支持的视觉条件：

```text
original
blur
occlusion
frame_drop
temporal_shift
wrong_video
no_video
```

训练时显式传入全部条件：

```bash
python - <<'PY'
import os

from Data_loaders.uq_av_loader import create_uq_av_dataloader

loader = create_uq_av_dataloader(
    data_root=os.environ["DATA_ROOT"],
    split_name="train_av_split.txt",
    phase="train",
    metadata_dir=os.environ["UQ_METADATA_ROOT"],
    batch_size=2,
    num_workers=0,
    pin_memory=False,
    seed=1234,
    video_conditions=(
        "original",
        "blur",
        "occlusion",
        "frame_drop",
        "temporal_shift",
        "wrong_video",
        "no_video",
    ),
)

for epoch in range(2):
    loader.dataset.set_epoch(epoch)
    batch = next(iter(loader))
    print("epoch", epoch)
    print("mask:", batch["mask_spec"])
    print("condition:", batch["video_condition"])
    print("parameters:", batch["video_degradation"])
PY
```

当前训练阶段会在传入的 `video_conditions` 中均匀选择。若不传该参数，则只使用
`original`，不会静默改变训练数据分布。

`wrong_video` 优先选择同乐器、不同 source video 的样本；如果没有，再回退到任意
不同 source video。split 中只有一个 source video 时不能使用该条件。

### 10. 固定 val/test loader

验证和测试会展开：

```text
split 中的每个样本
x manifest 中的 4 个固定 mask
x video_conditions 中显式请求的视觉条件
```

只测试原视频：

```bash
python - <<'PY'
import os

from Data_loaders.uq_av_loader import create_uq_av_dataloader

loader = create_uq_av_dataloader(
    data_root=os.environ["DATA_ROOT"],
    split_name="test_av_split.txt",
    phase="test",
    metadata_dir=os.environ["UQ_METADATA_ROOT"],
    batch_size=4,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
    seed=1234,
    video_conditions=("original",),
)

print("dataset entries:", len(loader.dataset))
batch = next(iter(loader))
print("sample_id:", batch["sample_id"])
print("mask_type:", [spec.mask_type for spec in batch["mask_spec"]])
print("condition:", batch["video_condition"])
PY
```

测试全部视觉条件：

```bash
python - <<'PY'
import os

from Data_loaders.uq_av_loader import create_uq_av_dataloader

conditions = (
    "original",
    "blur",
    "occlusion",
    "frame_drop",
    "temporal_shift",
    "wrong_video",
    "no_video",
)

loader = create_uq_av_dataloader(
    data_root=os.environ["DATA_ROOT"],
    split_name="test_av_split.txt",
    phase="test",
    metadata_dir=os.environ["UQ_METADATA_ROOT"],
    batch_size=8,
    num_workers=4,
    shuffle=False,
    seed=1234,
    video_conditions=conditions,
)

print("test samples:", len(loader.dataset.rows))
print("mask variants per sample:", 4)
print("video conditions:", len(conditions))
print("total evaluation entries:", len(loader.dataset))
PY
```

如果 test split 有 `N` 个样本，以上配置会生成 `N × 4 × 7` 个评测条目。

### 11. 运行 P1 单元测试

只运行 P1 测试：

```bash
python -m unittest \
  tests.test_mask_sampler \
  tests.test_uq_av_data \
  -v
```

运行包含 P0/P1 的全部测试：

```bash
python -m unittest discover -s tests -v
```

当前预期结果为：

```text
Ran 23 tests
OK
```

### 12. 常见失败

- `UQ split file not found`：检查 `--data-root` 和 split 文件名。
- `has Mel shape ... expected (200, 80)`：重新执行 AV 数据预处理，确保每个 clip 为固定 200 帧 Mel。
- `has audio shape ... expected (64000,)`：音频没有对齐为 4 秒、16 kHz。
- `has ... frames in image_crop/flow_*_crop`：样本不是固定 50 帧，或目录中有额外 JPG。
- `missing numbered frames`：帧数虽然是 50，但文件名不是连续的 `1.jpg` 到 `50.jpg`。
- `manifest coverage does not match split`：生成 metadata 后修改了 val/test split，需要重新运行 `prepare-uq-metadata`。
- `wrong_video requires at least two distinct source videos`：当前 split 无法构造错配视频，移除该条件或增加不同源视频。
- 多次结果不同：确认 metadata 和 loader 使用相同 `seed`，训练循环正确调用了 `set_epoch(epoch)`。
