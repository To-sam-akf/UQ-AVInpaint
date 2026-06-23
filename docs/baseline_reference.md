# VIAI-AV-PatchGAN Reference

本文档用于固定本仓库后续 EC-VIAI-AV 扩展的 baseline 参照。本仓库不重新训练
baseline，只复用另一个仓库已经验证过的 VIAI-AV-PatchGAN checkpoint、测试指标和
实验配置。

## 参照范围

后续实验表格只使用这个固定名称：

- `VIAI-AV-PatchGAN reference`

当前优化路线不再单独引入 `VIAI-AV reference`。在这个项目里，PatchGAN 版本已经作为
VIAI-AV 的升级 baseline，EC-VIAI-AV 主实验默认和它比较。除非后续明确做非 GAN 消融，
否则不要把普通 VIAI-AV 结果混入主参照。

## 来源记录

| 字段 | 值 |
| --- | --- |
| 参照名称 | `VIAI-AV-PatchGAN reference` |
| 结果 JSON 记录的来源仓库 | `/root/Vision-Infused-Audio-Inpainter-VIAI` |
| 结果 JSON 记录的 checkpoint | `/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar` |
| checkpoint step | `19000` |
| global step / epoch | `19000` / `47` |
| stage | `VIAI-AV-stage4-sync-probe` |
| PatchGAN | 开启 |
| sync loss | 开启 |
| probe branch/loss | 开启 |
| 本仓库测试 JSON | `checkpoints/viai_av_patchgan_test_results/VIAI-AV-PatchGAN_step000019000_test.json` |
| 本仓库测试 summary CSV | `checkpoints/viai_av_patchgan_test_results/VIAI-AV-PatchGAN_test_summary.csv` |
| 本仓库测试 Mel 图 | `checkpoints/viai_av_patchgan_test_results/mel-image/step000019000/` |
| 本仓库 TensorBoard event | `checkpoints/events_viai_av_patchgan/events.out.tfevents.1779795600.p-250cc0af6776-ackcs-00gjgrzt` |
| 记录本文档时的当前仓库 commit | `abdaf78670b93be0b08b703e0c097e1966ecc3cc` |
| 来源仓库 commit | 当前工作区无法获取；如果来源仓库重新挂载，需要补填 |

当前机器上没有结果 JSON 中记录的来源仓库和 `.pth.tar` checkpoint。因此，本仓库现在拥有
reference 指标、summary CSV、TensorBoard event 和可视化输出，但没有 reference checkpoint
本体。

## 参照指标

后续主比较优先使用 Mel、PSNR、SSIM 和 retrieval 指标。loss 数值保留用于追踪，但测试
JSON 没有保存 `lambda_gan` 字段，所以只有在确认测试命令和 loss 权重一致时，才比较
`loss_total` 一类的聚合 loss。

| 指标 | 值 |
| --- | ---: |
| test samples | `376` |
| mel_l1_full | `0.04007849247848734` |
| mel_l1_missing | `0.06643945501839861` |
| probe_l1_full | `0.034017237775186275` |
| probe_l1_missing | `0.05272802710533142` |
| psnr_full | `29.092075266736618` |
| psnr_missing | `21.87918318078873` |
| ssim | `0.9662416217417361` |
| retrieval_audio_to_video_r1 | `3.723404255319149` |
| retrieval_audio_to_video_r5 | `12.23404255319149` |
| retrieval_audio_to_video_r10 | `22.074468085106382` |
| retrieval_audio_to_video_r50 | `47.340425531914896` |
| retrieval_audio_to_video_medr | `57.0` |
| retrieval_audio_to_video_meanr | `91.11968085106383` |
| retrieval_video_to_audio_r1 | `1.8617021276595744` |
| retrieval_video_to_audio_r5 | `10.638297872340425` |
| retrieval_video_to_audio_r10 | `17.5531914893617` |
| retrieval_video_to_audio_r50 | `41.48936170212766` |
| retrieval_video_to_audio_medr | `71.0` |
| retrieval_video_to_audio_meanr | `97.8563829787234` |

辅助 loss 记录：

| loss | 值 |
| --- | ---: |
| loss_total | `2.2108741973308805` |
| loss_av_gen | `1.4232032375132784` |
| loss_recon | `0.07185346506377484` |
| loss_g_gan | `1.3513497697546126` |
| loss_sync | `0.5973061703621073` |
| loss_probe_gen | `1.4092205509226372` |
| loss_probe_recon | `0.05732325179145691` |
| loss_probe_g_gan | `1.3518972954851516` |
| loss_d | `0.8192217654370247` |
| eta1 / eta2 | `0.1350851717672992` / `0.1350851717672992` |

## 数据与 mask 对齐

除非实验本身就是 mask 长度或数据消融，所有 EC-VIAI-AV 主实验都保持以下设置：

| 设置 | 固定值 |
| --- | --- |
| train split | `train_av_split.txt` |
| val split | `val_av_split.txt` |
| test split | `test_av_split.txt` |
| 本地 split 行数 | train `3196`, val `187`, test `376` |
| reference test samples | `376` |
| audio window | 4 秒，16 kHz 下 `64000` samples |
| Mel shape | `80 x 200` |
| visual input | 50 帧 RGB + 50 帧 optical flow |
| visual frame interval | `0.08` 秒 |
| hop size | `320` |
| blank/missing span | 连续随机 `20` 到 `50` 个 Mel frames |
| AV 样本必需文件 | `raw_audio.npy`, `mel.npy`, `image_crop/`, `flow_x_crop/`, `flow_y_crop/` |

当前机器的数据状态：

- `data/` 下存在同名 split 文件，行数与 reference 记录对齐。
- 当前本地 `data/` 目录没有这些 split 指向的 `processed/.../shot.../clip...` 样本目录。
  因此如果要在本机完整运行 `test-viai-av`，需要重新挂载来源数据根目录或重新生成 AV 数据。

## 记录命令

来源仓库的 shell history 当前不可用。下面命令用于记录当前代码风格下应当复现的 reference
实验形态和输出位置，不等价于已经核验过的原始命令逐字记录。

训练命令：

```bash
python main.py train-viai-av -- \
  --use_gan \
  --name VIAI-AV-PatchGAN \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000006800.pth.tar \
  --checkpoint_dir checkpoints/viai-av_train \
  --log_event_path checkpoints/events_viai_av_patchgan \
  --batch_size 8 \
  --num_workers 4 \
  --lambda_recon 1.0 \
  --lambda_gan 0.001 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0
```

测试命令：

```bash
python main.py test-viai-av -- \
  --use_gan \
  --name VIAI-AV-PatchGAN \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0 \
  --results_dir checkpoints/viai_av_patchgan_test_results
```

如果需要精确比较 loss，请用和 reference 训练一致的 loss 权重重新测试。论文主比较优先看
JSON/CSV 中的质量指标。

## EC-VIAI-AV 初始化规则

如果之后把 reference checkpoint 拷回当前工作区，EC-VIAI-AV 可以从它初始化，但只加载这些
可复用权重：

- `Mel_Encoder`
- `VideoEncoder`
- `Mel_Decoder`

新增 EC 模块，例如 evidence estimator、fusion gate、stochastic bottleneck adapter、
candidate scorer 和 uncertainty head，必须随机初始化。除非是在继续一个已经存在的 EC run，
否则不要加载 optimizer state，也不要继承 baseline 的 `global_step` 或 `global_epoch`。

这个初始化应当被视为 `init_from_reference`，不是 resume baseline。

## 当前验证状态

已完成：

- 已解析 reference JSON 和 CSV。
- 已确认 `VIAI-AV-PatchGAN reference` 开启 PatchGAN、sync loss 和 probe loss。
- 已确认本地 AV split 名称和行数：train `3196`、val `187`、test `376`。
- 已确认 `mel-image/step000019000/` 下有 `376` 张 Mel 对比图。
- 已解析本地 TensorBoard event，确认 PatchGAN 训练曲线至少记录到 step `19604`。

暂未完成，原因是必要产物缺失：

- 当前 `test-viai-av` 加载 reference checkpoint 的兼容性验证。记录中的 `.pth.tar` 不在当前工作区。
- EC-VIAI-AV smoke 初始化。EC 模块尚未实现，reference checkpoint 也不在本机。

当 checkpoint 和完整数据根目录可用后，先运行这个 reference smoke test：

```bash
python main.py test-viai-av -- \
  --use_gan \
  --name VIAI-AV-PatchGAN \
  --resume_path /path/to/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar \
  --data_root /path/to/data \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir /tmp/viai_av_patchgan_reference_smoke
```

等 EC-VIAI-AV 分支实现后，再运行一个单独的 smoke test，只从同一 checkpoint 加载
`Mel_Encoder`、`VideoEncoder` 和 `Mel_Decoder`。

## 后续实验规则

- 新的 EC-VIAI-AV 输出不要写入 `checkpoints/viai_av_patchgan_test_results/`。
- 新实验使用 `EC-VIAI-AV-K4`、`EC-VIAI-AV-K8`、`EC-VIAI-AV-full` 等名称，不要复用
  `VIAI-AV-PatchGAN reference`。
- 主公平比较保持 `train_av_split.txt`、`val_av_split.txt`、`test_av_split.txt` 和
  blank frame 范围 `20-50`。
- 如果后续实验改变 mask 长度、视频扰动、sync/probe 设置、PatchGAN 设置或数据 split，
  需要记录为 ablation，而不是主 reference 对比。
