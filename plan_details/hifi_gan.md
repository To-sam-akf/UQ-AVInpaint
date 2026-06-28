# HiFi-GAN Fine-Tune Vocoder Plan

## Summary
- 目标：用 **HiFi-GAN** 作为第一版神经 vocoder，基于官方预训练 `UNIVERSAL_V1` 初始化，再用 MUSICES 的 `raw_audio.npy + mel.npy` fine-tune，快速替换 Griffin-Lim 做 demo wav。
- 输出策略：默认 **只替换 missing 音频段**，已知区域保留原始 `raw_audio`，边界做 crossfade，避免整段重合成劣化已知音质。
- 依据：HiFi-GAN 官方仓库提供预训练模型，并说明 `UNIVERSAL_V1` 可作为迁移学习基座；仓库采用 MIT license。参考：<https://github.com/jik876/hifi-gan>

## Key Changes
- 新增 vocoder 训练入口：
  - `main.py` 增加 `train-vocoder`，调用新的 HiFi-GAN fine-tune 脚本。
  - 训练数据直接读取现有 split 中的 `mel.npy` 和 `raw_audio.npy`，复用项目参数：`sample_rate=16000`、`num_mels=80`、`fft_size=1280`、`hop_size=320`、`fmin=125`、`fmax=7600`。
- 新增 HiFi-GAN backend：
  - `--vocoder_backend` 扩展为 `griffin_lim|hifigan`。
  - 新增 `--vocoder_checkpoint`、`--vocoder_splice_missing`、`--vocoder_crossfade_ms`。
  - Griffin-Lim 路径保持兼容；HiFi-GAN 路径加载 fine-tuned generator。
- HiFi-GAN 配置：
  - 使用 V1 风格 generator/discriminators。
  - 为 `hop_size=320` 设置 `upsample_rates=[8,8,5]`、`upsample_kernel_sizes=[16,16,10]`。
  - 从官方 `UNIVERSAL_V1` checkpoint **部分加载 shape 匹配权重**，跳过不匹配的 upsample 层并打印加载报告。
- Mel 条件处理：
  - 不直接喂 `[0,1]` normalized Mel。
  - 统一把项目 Mel 反归一化为 amplitude，再转 `log amplitude` 条件，以更接近 HiFi-GAN 预训练分布。
- Demo 导出：
  - 生成 completed Mel 的 full waveform。
  - 从 mask 推出 missing mel span，换算为 audio samples。
  - wav 输出默认：`raw_audio` 已知区域 + HiFi-GAN 生成 missing 区域 + 20ms crossfade。
  - candidate wav 使用同一逻辑，每个 candidate 单独导出。

## Training Flow
- 准备官方 HiFi-GAN `UNIVERSAL_V1` checkpoint，路径作为参数传入，不把大权重提交进仓库。
- 训练命令形态：
  ```bash
  python main.py train-vocoder -- \
    --vocoder_model hifigan \
    --hifigan_pretrained_generator /path/to/UNIVERSAL_V1/generator \
    --data_root "$DATA_ROOT" \
    --train_split_name train_av_split.txt \
    --val_split_name val_av_split.txt \
    --checkpoint_dir checkpoints/hifigan_musices_ft \
    --batch_size 16 \
    --max_steps 50000
  ```
- 快速 demo 默认先跑 `20k-50k` steps；每隔固定 step 保存 generator，并在 val split 导出少量 `target/generated/spliced` wav 供人工试听。

## Test Plan
- Unit tests：
  - split loader 能正确读 `mel.npy/raw_audio.npy`，输出 `[80,T]` mel 和 `[T*hop]` audio。
  - Mel 转 HiFi-GAN condition 后 shape 正确、数值 finite。
  - partial checkpoint load 能跳过 shape mismatch，并报告 loaded/skipped keys。
  - missing splice 后 wav 长度等于原始音频；missing 外区域保持原始音频，crossfade 区除外。
- Smoke tests：
  - `main.py train-vocoder -- --max_steps 2 --batch_size 1` 可完成前向、反向、保存 checkpoint。
  - `test_viai_av.py ... --use_vocoder --vocoder_backend hifigan --vocoder_checkpoint <ckpt>` 能导出 top-1 和 candidate wav。
- Acceptance：
  - HiFi-GAN 导出的 demo wav 主观听感明显少于 Griffin-Lim 的相位噪声。
  - wav 无 NaN、无明显 clipping，长度与 4 秒输入一致。
  - 原有 Griffin-Lim 命令不受影响。

## Assumptions
- 第一版目标是“快速 demo”，不是最高音质或完整论文级 vocoder 消融。
- 默认使用 HiFi-GAN，不做 BigVGAN/WaveNet 第一版实现。
- 官方预训练 checkpoint 由用户手动下载到本地路径；代码只支持加载，不自动联网下载。
- 已知区域保留原始 `raw_audio` 是允许的，因为音频修复任务本来只需要补 missing 区域；实现不得使用 ground-truth missing 段。


## train and manu
下面都不用 `uv`，云端直接用 `python`；如果云端只有 `python3`，把命令里的 `python` 替换成 `python3`。

**1. 环境**
```bash
python -m pip install --upgrade pip
python -m pip install imageio-ffmpeg librosa nnmnkwii numpy opencv-contrib-python-headless pillow scikit-image tensorboard tensorboardX tqdm "yt-dlp[default]"
```

**2. 设置路径**
```bash
export DATA_ROOT=/root/shared-nvme/data
export HIFIGAN_DIR=checkpoints/hifigan_musices_ft
mkdir -p "$HIFIGAN_DIR"
```

把官方 HiFi-GAN `UNIVERSAL_V1` generator checkpoint 放到云端，例如：

```bash
export HIFIGAN_PRETRAIN=/root/shared-nvme/pretrained_hifigan/UNIVERSAL_V1/generator
```

如果暂时没有预训练权重，也可以不传 `--hifigan_pretrained_generator`，但效果会差很多。

**3. 先跑 2 step smoke test**
```bash
python main.py train-vocoder -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$HIFIGAN_DIR/smoke" \
  --batch_size 1 \
  --num_workers 0 \
  --max_steps 2 \
  --checkpoint_interval 1 \
  --hifigan_segment_mel_frames 8 \
  --hifigan_eval_samples 1 \
  --display_id 0
```

确认生成：
```bash
ls "$HIFIGAN_DIR/smoke"/hifigan_generator_step*.pth.tar
find "$HIFIGAN_DIR/smoke/audio" -type f | head
```

**4. 正式 fine-tune**
```bash
python main.py train-vocoder -- \
  --hifigan_pretrained_generator "$HIFIGAN_PRETRAIN" \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$HIFIGAN_DIR/full" \
  --batch_size 16 \
  --num_workers 4 \
  --max_steps 50000 \
  --checkpoint_interval 5000 \
  --print_freq 100 \
  --hifigan_segment_mel_frames 32 \
  --hifigan_eval_samples 4 \
  --display_id 0
```

取最新 vocoder checkpoint：
```bash
export VOCODER_CKPT=$(ls -t "$HIFIGAN_DIR/full"/hifigan_generator_step*.pth.tar | head -n 1)
echo "$VOCODER_CKPT"
```

**5. 用 HiFi-GAN 导出 demo wav**

在你现有 `test_viai_av.py` 命令里，把 Griffin-Lim 相关部分改成：

```bash
  --use_vocoder \
  --vocoder_backend hifigan \
  --vocoder_checkpoint "$VOCODER_CKPT" \
  --vocoder_crossfade_ms 20 \
```

完整地说，就是保留你原来的 EC-VIAI-AV 参数，只把：

```bash
  --use_vocoder \
  --vocoder_max_samples 5 \
  --vocoder_n_iter 128 \
```

改为：

```bash
  --use_vocoder \
  --vocoder_backend hifigan \
  --vocoder_checkpoint "$VOCODER_CKPT" \
  --vocoder_max_samples 5 \
  --vocoder_crossfade_ms 20 \
```

默认会只替换 missing 段，已知区域保留原始 `raw_audio`。如果想整段都由 HiFi-GAN 重合成，加：

```bash
--no_vocoder_splice_missing
```

**6. 查看输出**
```bash
find "$DEMO_ROOT/none/wav" -type f | head
find "$DEMO_ROOT/none/wav-candidates" -type f | head -n 40
```

建议先用 `smoke` checkpoint 跑通导出链路，再换 `full` checkpoint 做正式试听。