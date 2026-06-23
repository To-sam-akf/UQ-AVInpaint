下面是一份可以直接保存为 `VIAI_code_vs_paper_report.md` 的 Markdown 报告。

---

````markdown
# Vision-Infused-Audio-Inpainter-VIAI 项目代码与论文内容对照报告

## 1. 报告目的

本文对 GitHub 项目 [Vision-Infused-Audio-Inpainter-VIAI](https://github.com/Hangz-nju-cuhk/Vision-Infused-Audio-Inpainter-VIAI) 与论文 **Vision-Infused Deep Audio Inpainting, ICCV 2019** 进行对照分析，目标是明确：

1. 论文中完整方法包含哪些模块；
2. GitHub 项目中已经公开了哪些内容；
3. GitHub 项目中缺失或不完整的部分有哪些；
4. 如果要复现论文，需要补充哪些工程模块。

该项目 README 中已经明确说明：当前代码仍在整理中，**not complete thus not runable**，也就是“不完整、不能直接运行”，但公开了部分网络 architecture。论文则说明该工作提出了视觉信息注入的音频修复任务，即根据视频内容补全缺失音频片段。  
资料来源：GitHub README 与 ICCV 2019 论文。  

---

## 2. 论文方法概述

论文研究的是 **Vision-Infused Audio Inpainting**，即视觉信息辅助的音频缺失补全任务。

### 2.1 任务定义

给定：

- 一段存在缺失片段的音频；
- 与该音频对应的完整视频；

目标是生成缺失的音频片段，使其同时满足：

1. 音频本身听起来连续、自然；
2. 生成的声音与视频画面中的动作、节奏、语义保持一致。

论文指出，音频修复比图像修复更困难，因为音频具有高采样率和长程依赖特性。传统音频修复方法通常依赖稀疏表示或寻找相似结构，但当缺失片段较长、上下文中没有相似结构时，效果会明显下降。

### 2.2 论文核心贡献

论文主要贡献包括：

1. 将音频修复从 raw audio 空间转到 **Mel-spectrogram** 空间；
2. 借鉴图像 inpainting 的 encoder-decoder + GAN 思路进行谱图补全；
3. 引入视频帧与光流信息，建立音频-视频 joint feature space；
4. 使用 contrastive sync loss 让视频特征与音频特征对齐；
5. 构建 MUSICES 数据集，即 MUSIC-Extra-Solo；
6. 使用 WaveNet vocoder 将补全后的 Mel-spectrogram 转回 raw audio。

---

## 3. GitHub 项目总体情况

GitHub 仓库名称为：

```text
Hangz-nju-cuhk/Vision-Infused-Audio-Inpainter-VIAI
````

仓库 README 中说明：

```text
Training and Testing

We are still sorting out the code. For now it is not complete thus not runable, but the architecture is revealed. Please wait for more details.
```

也就是说，该项目公开版本并不是完整复现工程，而更像是一个 **网络结构参考代码**。

仓库中可以看到的主要目录和文件包括：

```text
Data_loaders/
misc/
networks/
utils/
visdom_utils/
wavenet_vocoder/
base_options.py
loss_functions.py
train_whole_sync.py
```

其中：

* `networks/` 中包含部分网络结构；
* `wavenet_vocoder/` 中包含 vocoder 相关代码；
* `Data_loaders/` 中包含部分数据读取和预处理代码；
* `train_whole_sync.py` 是训练入口的外壳；
* 但是关键的 `Models/` 目录和 `Options_inpainting.py` 配置文件缺失。

---

## 4. 与论文一致或基本对应的部分

### 4.1 任务定义一致

论文和仓库都围绕 **Vision-Infused Audio Inpainting** 展开。

论文中的任务是：根据视频补全缺失音频片段。
仓库 README 中也说明该项目是 VIAI，即 Vision-Infused Audio Inpainter。

因此，在任务层面，二者是一致的。

---

### 4.2 Mel-spectrogram 音频修复思路一致

论文不是直接在 raw waveform 上进行补全，而是先将音频转换成 Mel-spectrogram，再将谱图看作二维图像进行 inpainting。

论文中的 VIAI-A 结构大致为：

```text
corrupted audio
    ↓
Mel-spectrogram
    ↓
Audio Encoder Ea
    ↓
Audio Decoder Ga
    ↓
Reconstructed Mel-spectrogram
```

仓库中 `networks/Inpainting_Networks.py` 和 `networks/New_Inpainting_Networks.py` 提供了类似结构，例如：

* `MelEncoder`
* `MelDecoder`
* `MelDecoderImage`

这说明仓库中保留了论文中音频谱图 encoder-decoder 的核心结构。

---

### 4.3 音频 Encoder 结构基本对应

论文中描述：

* 音频 encoder `Ea` 由 5 层 stride-2 convolution layers 组成；
* 目标是把输入 Mel-spectrogram 压缩到 bottleneck feature；
* bottleneck feature 用于后续谱图重建或与视频特征融合。

仓库中的 `MelEncoder` 也采用多层卷积、BatchNorm、LeakyReLU、Pooling 等结构，用来压缩 Mel-spectrogram 特征。

因此，音频 encoder 部分和论文基本一致。

---

### 4.4 音频 Decoder 结构基本对应

论文中 decoder `Ga` 负责从 bottleneck feature 恢复 Mel-spectrogram，结构包括：

* 多层卷积；
* 多次 bilinear upsampling；
* skip connection；
* 最终输出补全后的谱图。

仓库中的 `MelDecoder` 也体现了这种上采样恢复结构。

因此，音频 decoder 的基本架构是有的。

---

### 4.5 视频 Encoder 结构基本对应

论文 VIAI-AV 部分使用：

```text
image encoder EI
flow encoder EF
feature fusion module Efuse
```

其中 image encoder 和 flow encoder 基于 ResNet-18。

仓库中的 `networks/Image_Embedding.py` 中包含 ResNet 相关结构，并提供了图像/光流特征提取的基础实现。

因此，视频分支的基础网络结构在仓库中是存在的。

---

### 4.6 音频-视频融合 Decoder 有对应实现

论文中的 VIAI-AV 核心公式可以理解为：

```text
reconstructed spectrogram = Gav(audio feature, visual feature)
```

也就是将：

```text
Ea(s_i)
```

和视频特征：

```text
f_v
```

拼接后送入 audio-visual decoder。

仓库中的 `MelDecoderImage` 体现了这一点：它会把 audio bottleneck feature 与 video feature 进行拼接，然后再进行解码。

因此，论文中的音频-视觉融合 decoder 在仓库中有对应雏形。

---

### 4.7 WaveNet vocoder 目录存在

论文最后需要将补全后的 Mel-spectrogram 转换回 waveform。为此，论文使用了 modified WaveNet decoder。

仓库中存在：

```text
wavenet_vocoder/
```

说明 vocoder 相关代码至少部分公开。

不过，仓库没有给出完整的 WaveNet 训练、推理和与 VIAI 主模型对接的流程。

---

## 5. 缺失或不完整的部分

### 5.1 核心模型封装缺失

这是最严重的问题。

仓库中的 `train_whole_sync.py` 里导入了：

```python
import Models.Whole_Sync_inpainting_modify as Audio_model
```

但仓库公开文件树中没有：

```text
Models/
```

目录，也没有：

```text
Whole_Sync_inpainting_modify.py
```

这意味着训练脚本依赖的核心模型类缺失。

在训练脚本中，真正的模型训练逻辑大概率封装在：

```python
Audio_model.AudioModel(...)
```

里面，例如：

```python
model.optimize_parameters()
model.test()
model.eval_model_test()
model.get_loss_items()
```

但由于 `AudioModel` 所在文件缺失，训练脚本无法运行。

---

### 5.2 配置文件 `Options_inpainting.py` 缺失

仓库代码中还引用了：

```python
import Options_inpainting
```

并使用：

```python
Options_inpainting.Inpainting_Config()
```

但是仓库中没有 `Options_inpainting.py`。

仓库中虽然有 `base_options.py`，但它只是较基础的参数配置，无法覆盖论文复现所需的全部超参数。

论文中需要配置的内容包括：

```text
Mel-spectrogram 参数
采样率
STFT window length
hop size
mel bins
缺失片段长度
batch size
learning rate
GAN loss 权重
reconstruction loss 权重
sync loss margin
probe loss 权重
η1(t) 和 η2(t) 衰减策略
WaveNet 参数
```

这些内容没有完整公开。

---

### 5.3 数据下载流程缺失

论文使用 MUSICES 数据集。该数据集基于 MUSIC 数据集扩展而来，包含 9 类 solo 乐器视频。

论文中数据构建包括：

1. 从 YouTube 视频中收集数据；
2. 筛选 solo 演奏视频；
3. 检测 video shots；
4. 去掉黑屏、无声或非演奏片段；
5. 裁掉每个视频前 6 秒；
6. 按视频级别划分 train/val/test。

仓库只提供了 MUSICES 数据集入口或 JSON 形式的视频 ID 列表，但没有完整的：

```text
YouTube 批量下载脚本
失效视频过滤脚本
shot detection 脚本
数据清洗脚本
train/val/test split 生成脚本
```

因此，从原始 MUSICES JSON 到可训练数据的完整 pipeline 是缺失的。

---

### 5.4 Shot detection 缺失

论文中提到会进行 video shot detection，用于确定视频有效片段的开始和结束位置，并去除不合适的片段。

但是仓库中没有看到完整的 shot detection 实现。

这会影响复现，因为如果不做 shot 清洗，下载到的视频可能包含：

```text
片头
黑屏
字幕
转场
无声片段
非演奏片段
```

这些都会影响模型训练。

---

### 5.5 视频预处理流程不完整

论文中视频预处理包括：

```text
视频帧率调整到 12.5 fps
提取 RGB frames
提取 TV-L1 optical flow
限制 flow 最大值到 20 pixels
根据全视频 flow 平均值确定 motion salient region
裁剪图像和光流
padding 成 square
像素归一化到 [-1, 1]
```

仓库中虽然存在图像预处理和 image embedding 相关代码，但没有完整的一键流程来完成：

```text
视频抽帧
光流提取
光流裁剪
motion region 检测
图像裁剪
padding
归一化
保存为 dataloader 所需格式
```

因此，视频预处理部分是不完整的。

---

### 5.6 音频预处理流程不完整

论文中音频预处理参数包括：

```text
sampling rate = 16 kHz
Mel bins = 80
STFT length = 1280 samples
hop = 320 samples
video fps = 12.5
1 个视频帧对应 4 个 spectrogram bins
```

这些参数对于音频-视频时间对齐非常关键。

仓库虽然有 `Data_loaders/audio_loader.py` 等相关代码，但没有完整提供从原始 mp4 到：

```text
source.wav
raw_audio.npy
mel.npy
masked mel
ground truth mel
```

的稳定处理流程。

---

### 5.7 Contrastive sync loss 实现不可确认

论文中 VIAI-AV 的关键在于学习 audio-video joint feature space。

其思想是：

```text
匹配的视频特征和音频特征拉近
不匹配的视频特征和音频特征推远
```

这对应 contrastive sync loss。

但仓库中由于缺少核心 `AudioModel` 封装，无法确认：

```text
positive pair 如何构造
negative pair 如何采样
margin γ 如何设置
sync loss 如何加入总损失
sync loss 训练阶段如何安排
```

因此，sync loss 相关训练逻辑是不完整或不可确认的。

---

### 5.8 Probe loss / VIAI-AA' 缺失或不可确认

论文中有一个重要设计：VIAI-AA'。

其思想是：用 clean audio bottleneck feature 作为理想条件，帮助 audio-visual decoder 学会使用条件信息。

也就是：

```text
Gav(Ea(s_i), f_a^t)
```

其中 `f_a^t` 是 clean target audio 的 bottleneck feature。

这个 probe loss 可以帮助模型确认 decoder 确实利用了条件特征，而不是忽略视频信息。

但是仓库中没有完整训练主体，因此无法确认 probe loss 是否实现；从公开文件看，也没有清晰的可运行入口。

---

### 5.9 η1(t) 和 η2(t) 衰减训练策略缺失或不可确认

论文中有两个重要的动态权重：

```text
η1(t)：reconstruction loss 中已知区域与缺失区域的权重调整
η2(t)：probe loss 的动态权重调整
```

这些设计对应论文中的训练技巧。

但是仓库中没有完整配置和训练逻辑，因此无法确认这些动态权重是否完整实现。

---

### 5.10 PatchGAN 训练流程不完整

论文中使用 PatchGAN discriminator 来增强谱图局部纹理的真实感。

仓库中有 discriminator 相关网络和 loss 文件，因此结构部分可能存在。

但是完整 GAN 训练需要：

```text
生成器 forward
判别器 forward
real/fake loss
generator adversarial loss
reconstruction loss
loss 权重组合
optimizer_G
optimizer_D
反向传播顺序
checkpoint 保存
```

这些训练主体很可能在缺失的 `AudioModel` 中，因此公开仓库中并不能直接运行 GAN 训练。

---

### 5.11 WaveNet 与主模型的完整接入流程缺失

论文使用 modified WaveNet decoder 将 Mel-spectrogram 转为 waveform。

论文还提到：

```text
WaveNet 不仅条件在 Mel-spectrogram 上，还条件在缺失片段之前的 clean audio 上
```

这是为了让生成音频在边界处更连续。

仓库虽然有 `wavenet_vocoder/`，但缺少：

```text
WaveNet 训练命令
WaveNet 数据准备
每类乐器 WaveNet 训练流程
Mel-to-waveform 推理脚本
past clean audio conditioning 接入方式
VIAI 输出 mel 与 WaveNet 输入的接口
```

因此，WaveNet 部分只能说“有目录和部分代码”，但不是完整复现流程。

---

### 5.12 测试和推理脚本缺失

论文最终需要输出：

```text
补全后的 Mel-spectrogram
重建后的 waveform
audio-video demo
```

但仓库中没有清晰的：

```text
test.py
inference.py
evaluate.py
generate_audio.py
```

也没有 README 级别的测试命令。

因此，不能直接用仓库对新视频做推理。

---

### 5.13 评价指标实现缺失

论文使用了多种指标：

```text
PSNR
SSIM
SDR
OPS
MOS
```

其中：

* PSNR、SSIM 用于谱图质量；
* SDR、OPS 用于音频质量；
* MOS 用于主观听感和音画一致性。

仓库中没有完整的评价脚本来复现论文表格结果。

尤其是：

```text
OPS / PEMO-Q
MOS 主观评价
SDR 批量计算
表 1、表 2、表 3、表 4 对应评估流程
```

都没有完整公开。

---

### 5.14 预训练模型缺失

论文摘要中提到代码、模型、数据集和视频结果可用。

但当前 GitHub 仓库中没有看到清晰的 checkpoint 下载入口，也没有 release。

缺少预训练模型意味着：

1. 无法直接验证论文结果；
2. 无法直接跑 demo；
3. 需要重新训练 VIAI-A、VIAI-AV 和 WaveNet；
4. 每个乐器类别可能都需要单独训练模型。

---

## 6. 逐模块对照表

| 模块                      | 论文中要求                        | GitHub 仓库情况                  | 结论      |
| ----------------------- | ---------------------------- | ---------------------------- | ------- |
| 任务定义                    | 根据视频补全缺失音频                   | README 与论文一致                 | 一致      |
| MUSICES 数据集             | MUSIC 扩展，9 类 solo 乐器         | 提供数据入口/索引                    | 部分提供    |
| YouTube 下载              | 需要批量下载视频                     | 未提供完整脚本                      | 缺失      |
| 视频清洗                    | shot detection、去黑屏、去无声       | 未见完整流程                       | 缺失      |
| train/val/test split    | 视频级别划分                       | 未见完整生成脚本                     | 缺失      |
| 音频采样                    | 16kHz mono                   | 部分 dataloader 可能涉及           | 不完整     |
| Mel-spectrogram         | 80 bins，STFT 1280，hop 320    | 无完整 pipeline                 | 不完整     |
| 视频帧                     | 12.5 fps                     | 无完整抽帧流程                      | 不完整     |
| 光流                      | TV-L1 optical flow           | 无完整提取流程                      | 不完整     |
| Motion crop             | 根据 flow 裁剪运动区域               | 无完整流程                        | 不完整     |
| VIAI-A Encoder          | 5 层卷积 encoder                | `MelEncoder` 存在              | 基本一致    |
| VIAI-A Decoder          | 上采样 decoder + skip           | `MelDecoder` 存在              | 基本一致    |
| 视频 Encoder              | ResNet-18 image/flow encoder | `Image_Embedding.py` 存在      | 基本一致    |
| 视觉特征融合                  | Efuse 时间融合                   | 部分结构可能存在                     | 不完整     |
| Audio-visual Decoder    | 音频特征与视觉特征拼接                  | `MelDecoderImage` 存在         | 基本一致    |
| Contrastive sync loss   | 正负样本对比学习                     | 主模型缺失，无法确认                   | 不完整     |
| Probe loss              | VIAI-AA' 分支                  | 无完整训练主体                      | 缺失或不可确认 |
| η1(t), η2(t)            | 动态 loss 权重                   | 配置和训练逻辑缺失                    | 缺失或不可确认 |
| PatchGAN                | 判别器 + GAN loss               | 有部分网络/loss 文件                | 部分存在    |
| WaveNet vocoder         | Mel 转 waveform               | 有 `wavenet_vocoder/`         | 部分存在    |
| Past audio conditioning | WaveNet 使用前文 clean audio     | 无清晰接入流程                      | 不完整     |
| 训练入口                    | 完整 train pipeline            | `train_whole_sync.py` 依赖缺失文件 | 不可运行    |
| 测试入口                    | 完整 inference pipeline        | 未见完整脚本                       | 缺失      |
| 评估指标                    | PSNR/SSIM/SDR/OPS/MOS        | 未见完整评估脚本                     | 缺失      |
| 预训练权重                   | 论文称有 models                  | GitHub 无清晰 checkpoint        | 缺失      |

---

## 7. 最关键的不一致点

### 7.1 README 明确承认代码不可运行

这是最重要的证据。

GitHub README 中明确写道：

```text
For now it is not complete thus not runable, but the architecture is revealed.
```

因此，该仓库不能视为完整复现代码。

---

### 7.2 训练脚本导入了不存在的文件

`train_whole_sync.py` 需要：

```python
Models.Whole_Sync_inpainting_modify
Options_inpainting
```

但仓库中没有这些文件。

这会导致训练脚本在 import 阶段就失败。

---

### 7.3 论文的关键创新不只是网络结构，而是训练流程

论文真正重要的地方包括：

```text
spectrogram inpainting
PatchGAN
audio-video contrastive sync
probe loss
η1(t), η2(t)
WaveNet past audio conditioning
```

仓库只公开了部分网络结构，训练组织、loss 组合、数据 pipeline、评估 pipeline 都不完整。

因此，仅靠仓库无法完整复现论文结果。

---

## 8. 如果要复现，建议补充的工程路线

### 8.1 第一阶段：复现 VIAI-A

先不要做视频分支，只做音频谱图修复。

需要实现：

```text
1. 下载 MUSICES 视频
2. 抽取 16kHz mono audio
3. 生成 Mel-spectrogram
4. 随机 mask 中间一段 0.4s 到 1.0s 音频
5. 训练 MelEncoder + MelDecoder
6. 使用 L1 reconstruction loss
7. 使用 PSNR / SSIM 评估
```

这一阶段目标是跑通最小 pipeline。

---

### 8.2 第二阶段：加入 PatchGAN

在 VIAI-A 基础上加入：

```text
Discriminator
GAN loss
reconstruction loss
loss 权重 β
η1(t) 权重调整
```

目标是提升生成谱图的局部纹理真实感。

---

### 8.3 第三阶段：加入视频分支 VIAI-AV

补充：

```text
视频抽帧
TV-L1 光流提取
Image ResNet18
Flow ResNet18
Efuse 时间融合
MelDecoderImage 融合解码
```

目标是让模型能够根据视频动作补全对应声音。

---

### 8.4 第四阶段：加入 sync loss 和 probe loss

补充：

```text
正样本/负样本构造
contrastive sync loss
VIAI-AA' probe branch
η2(t) probe loss 衰减
```

这是论文中保证视觉信息真正起作用的关键。

---

### 8.5 第五阶段：WaveNet 或替代 vocoder

最后再考虑从 Mel-spectrogram 还原 waveform。

可以有两种路线：

#### 路线 A：复现论文 WaveNet

需要实现：

```text
WaveNet 数据准备
每类乐器单独训练 WaveNet
Mel condition
past clean audio condition
inference 接口
```

优点：更接近论文。
缺点：工作量大，训练慢。

#### 路线 B：使用现成 vocoder 替代

例如：

```text
Griffin-Lim
HiFi-GAN
WaveGlow
其他 Mel vocoder
```

优点：更容易跑通实验。
缺点：与论文设置不完全一致。

如果是课程汇报或方法验证，建议先用路线 B。

---

## 9. 建议的数据目录结构

为了补齐仓库缺失部分，可以整理如下目录：

```text
data/
├── MUSICES.json
├── raw_videos/
│   ├── flute/
│   │   ├── youtube_id.mp4
│   │   └── ...
│   ├── violin/
│   └── ...
├── processed/
│   ├── flute/
│   │   └── youtube_id/
│   │       ├── source.wav
│   │       ├── raw_audio.npy
│   │       ├── mel.npy
│   │       ├── image/
│   │       ├── flow_x/
│   │       ├── flow_y/
│   │       ├── image_crop/
│   │       ├── flow_x_crop/
│   │       └── flow_y_crop/
│   └── ...
├── train.txt
├── val.txt
└── test.txt
```

---

## 10. 建议的复现优先级

如果时间有限，建议按以下优先级：

```text
优先级 1：数据下载与音频预处理
优先级 2：VIAI-A 音频谱图修复
优先级 3：PSNR / SSIM 指标
优先级 4：PatchGAN
优先级 5：视频帧和光流
优先级 6：VIAI-AV
优先级 7：sync loss 和 probe loss
优先级 8：WaveNet 或 vocoder
优先级 9：SDR / OPS / MOS 评估
```

不要一开始就尝试完整复现所有模块，否则很难判断错误来自：

```text
数据
环境
模型
loss
vocoder
评估
```

中的哪一部分。

---

## 11. 总结

GitHub 项目与论文的关系可以总结为：

```text
一致的部分：
任务定义、主要网络结构雏形、Mel Encoder/Decoder、Image/Flow Encoder、Audio-Visual Decoder 拼接方式、WaveNet 目录。

缺失的部分：
核心 AudioModel 封装、Options_inpainting 配置、完整数据下载/清洗/切片流程、shot detection、TV-L1 光流 pipeline、完整训练 loop、sync loss、probe loss、η 权重策略、测试推理脚本、评价指标脚本、预训练模型。
```

最终判断：

> 该仓库可以作为理解论文结构的参考，但不能作为完整复现工程直接训练。若要复现论文，需要自行补齐数据处理、主模型封装、loss 组合、训练循环、测试推理、指标评估和 vocoder 接口。

```

---

参考来源：GitHub README 明确说明当前代码“不完整、不能运行”，论文/项目页说明该任务是用视觉信息辅助补全缺失音频片段，并强调谱图操作和音频-视频 joint feature space 是关键设计。:contentReference[oaicite:0]{index=0}
::contentReference[oaicite:1]{index=1}
```
