# UQ-AVInpaint 实施方案与开发步骤

## 1. 文档目标

本文把 `thought/thoUght.md` 中的研究设想映射到当前 VIAI demo 工程，形成可直接执行的开发路线。

目标不是一次性重写整个项目，而是在保留现有 VIAI-A、VIAI-AV 基线的前提下，逐步新增：

1. 可复现的多类型缺失区域生成；
2. Mel latent autoencoder；
3. 条件 latent diffusion；
4. 时序视觉证据估计与 Evidence-Aware Fusion；
5. K 候选采样、候选排序和不确定性校准；
6. 对应的训练、测试、导出和实验指标。

最终输出应为：

```text
K 个修复候选
+ 每个候选的置信度 pi_k
+ 样本级不确定性 u
+ 视觉证据分数 e(t)
```

建议将新路线作为独立实验分支实现，暂定模型名为 `UQ-AVInpaint`。不要直接替换现有 `VIAIAVModel`，否则会破坏基线复现、checkpoint 兼容和后续消融实验。

---

## 2. 当前工程基线

### 2.1 已有能力

当前仓库已经具备：

- `main.py` 统一命令入口；
- VIAI-A 和 VIAI-AV 的训练、测试与 checkpoint；
- 16 kHz、80-bin Mel、4 秒窗口的数据管线；
- 50 帧 RGB 与 optical flow 输入；
- 连续缺失区间生成，`mask=1` 表示缺失；
- Mel encoder、image/flow ResNet-18、AV decoder；
- reconstruction、sync、probe 和可选 PatchGAN loss；
- PSNR、SSIM、Mel L1、AV retrieval 指标；
- 已知区域强制保留：

```python
final = mel_input * (1.0 - mask) + mel_pred * mask
```

- Griffin-Lim wav 导出。

### 2.2 当前关键张量

默认配置下：

```text
mel_target:    [B, 80, 200]
mel_input:     [B, 80, 200]
missing_mask:  [B, 1, 80, 200]
video_batch:   [B, 50, 3, 256, 256]
flow_batch:    [B, 50, 2, 256, 256]
```

当前 Mel encoder 将 `80 x 200` 压缩到约：

```text
[B, 256, 1, 13]
```

当前 VideoEncoder 先逐帧提取 RGB/flow 特征，再将 50 帧下采样到 13 个时序位置，输出同样约为：

```text
[B, 256, 1, 13]
```

这个设计适合原始 VIAI bottleneck concat，但不适合直接输出逐视频帧的 `e(t)`，也不适合作为高保真 diffusion latent。因此新路线需要单独的 Mel autoencoder 和保留时间轴的视觉编码接口。

### 2.3 实施前必须处理的限制

1. `corrupt_mel_spectrogram()` 当前让整个 batch 共用同一个 `start/end`。
2. mask 只支持单一 random gap，尚无 onset-centered、long-gap 等类型。
3. 训练阶段和测试阶段都可能随机生成 gap，不利于跨模型公平比较。
4. 现有 `VideoEncoder` 只返回融合后的 13-step bottleneck，不返回逐帧 RGB/flow token。
5. 现有测试入口只处理单候选，JSON/CSV 和 wav 导出格式不能表示 K 个候选。
6. 当前缺少真正的单元测试目录，主要依赖训练 smoke test。
7. Griffin-Lim 适合打通流程，但不应作为最终主观音质结论的唯一 vocoder。

---

## 3. 总体实施原则

### 3.1 保留现有基线

以下入口及其默认行为必须保持不变：

```text
train-viai-a
test-viai-a
train-viai-av
test-viai-av
```

新增独立入口：

```text
prepare-uq-metadata
train-mel-ae
test-mel-ae
train-uq-av
test-uq-av
```

### 3.2 先固定数据契约，再实现模型

所有新模块统一使用以下语义：

```text
mask=1: 需要生成的缺失区域
mask=0: 必须保留的已知区域
```

测试和导出阶段必须使用同一个 compose 函数，不能让不同模型各自实现一份 mask 拼接逻辑。

### 3.3 逐阶段训练

推荐顺序：

```text
P0 固定基线
P1 数据、mask、测试协议
P2 Mel autoencoder
P3 K=1 AV latent diffusion
P4 K-sampling
P5 visual evidence + uncertainty gate
P6 candidate scorer + calibration
P7 完整评估与消融
```

每一阶段达到验收标准后再进入下一阶段。不要一开始同时训练 autoencoder、diffusion、evidence、scorer 和 calibration head。

---

## 4. 建议目录与文件

建议新增：

```text
Data_loaders/
  uq_av_loader.py
  mask_sampler.py

Models/
  UQ_AV_inpainting.py
  Mel_Autoencoder.py

networks/
  uq/
    __init__.py
    mel_autoencoder.py
    audio_condition_encoder.py
    video_evidence_encoder.py
    diffusion_unet.py
    candidate_scorer.py

utils/
  diffusion.py
  uq_losses.py
  uq_metrics.py
  uq_export.py

tools/
  prepare_uq_metadata.py

tests/
  test_mask_sampler.py
  test_mel_autoencoder.py
  test_diffusion_masking.py
  test_uq_metrics.py
  test_uq_export.py

train_mel_autoencoder.py
test_mel_autoencoder.py
train_uq_av.py
test_uq_av.py
uq_options.py
```

对现有文件只做小范围接入：

```text
main.py              # 注册新入口
pyproject.toml        # 增加可选评估/开发依赖
README.md             # 增加运行命令
utils/viai_a_metrics.py
                     # 复用或迁移 compose_inpainted_mel
```

---

## 5. P0：冻结并记录 VIAI 基线

### 5.1 目标

在新增模型前保存一套可重复的基线结果，防止后续无法判断改进来自哪里。

### 5.2 实施步骤

1. 固定 train/val/test split，保持 video-level 隔离。
2. 固定测试 mask 清单，而不是测试时随机采样。
3. 分别运行 VIAI-A、VIAI-AV 和 VIAI-AA' probe reference。
4. 保存：
   - checkpoint 路径；
   - git commit；
   - 完整命令；
   - 配置 JSON；
   - 测试 mask manifest；
   - JSON/CSV 指标；
   - Mel 图和少量 wav。
5. 将基线结果写入独立实验目录：

```text
experiments/baselines/
  viai_a/
  viai_av/
  viai_aa_probe/
```

### 5.3 验收标准

- 同一 checkpoint 连续测试两次，指标完全一致或只存在可解释的数值误差；
- 三个模型使用同一批样本和同一组 gap；
- 已知区域 compose 后逐元素不变；
- 每个结果文件记录 `sample_id`、`mask_type`、`start`、`end` 和 `gap_frames`。

---

## 6. P1：数据契约、mask 和视觉退化

### 6.1 新 dataloader 输出

新 `uq_av_loader.py` 不再返回难扩展的 8-tuple，改为字典：

```python
{
    "sample_id": list[str],
    "mel_target": Tensor[B, 1, 80, 200],
    "mel_corrupted": Tensor[B, 1, 80, 200],
    "missing_mask": Tensor[B, 1, 80, 200],
    "boundary_map": Tensor[B, 2, 80, 200],
    "video": Tensor[B, 50, 3, H, W],
    "flow": Tensor[B, 50, 2, H, W],
    "audio_target": Tensor[B, 64000],
    "mask_spec": list[MaskSpec],
    "video_condition": list[str],
}
```

`boundary_map` 使用两个归一化通道：

```text
distance_to_left_boundary
distance_to_right_boundary
```

比单一距离图更容易区分缺失区左右边界。

### 6.2 MaskSpec

在 `mask_sampler.py` 定义：

```python
@dataclass
class MaskSpec:
    mask_type: str
    start: int
    end: int
    gap_frames: int
    seed: int
```

第一版支持：

| mask_type | 生成方式 |
| --- | --- |
| `random` | 在合法区间均匀采样 |
| `onset_centered` | 以高 spectral-flux/onset 帧为中心 |
| `boundary_near` | 靠近窗口前部或后部，但保留最小 clean context |
| `long_gap` | 60、80、100 帧，对应约 1.2、1.6、2.0 秒 |

训练时每个样本独立采样。验证和测试从 manifest 读取固定 `MaskSpec`。

### 6.3 固定测试 manifest

`prepare_uq_metadata.py` 生成：

```text
<data_root>/uq_metadata/
  train_onsets/<sample_id>.npy
  val_masks.jsonl
  test_masks.jsonl
  metadata_summary.json
```

`test_masks.jsonl` 每个样本至少包含：

```json
{
  "sample_id": "processed/accordion/xxx",
  "variants": [
    {"mask_type": "random", "start": 50, "end": 90, "seed": 1},
    {"mask_type": "onset_centered", "start": 80, "end": 120, "seed": 2},
    {"mask_type": "long_gap", "start": 50, "end": 130, "seed": 3}
  ]
}
```

### 6.4 视觉退化

训练和测试统一支持：

```text
original
blur
occlusion
frame_drop
temporal_shift
wrong_video
no_video
```

第一版不引入手部关键点检测。`occlusion` 使用可复现的矩形区域遮挡；后续有检测器时再替换为 hand/instrument-aware occlusion。

每种退化必须记录参数，例如：

```text
blur_sigma
occlusion_box
frame_keep_ratio
temporal_shift_frames
wrong_video_sample_id
```

### 6.5 测试

至少覆盖：

- batch 内不同样本使用不同 gap；
- 所有 mask 均为 `1=missing`；
- start/end 与 mask 非零区域一致；
- boundary map 在左右边界附近取值正确；
- compose 后已知区域误差严格为 0；
- temporal shift 和 wrong video 不改变音频输入；
-固定 seed 可复现相同 mask 和视觉退化。

### 6.6 验收标准

- 新 loader 可在现有 AV 数据上完成一个 batch；
- 输出 shape 与数据契约一致；
- 固定 manifest 重跑完全一致；
- 原 VIAI loader 行为不受影响。

---

## 7. P2：Mel Latent Autoencoder

### 7.1 选择

第一版使用确定性 convolutional autoencoder，不先引入 KL-VAE。随机性由 diffusion 提供，先减少一个训练变量。

建议 latent：

```text
input:  [B, 1, 80, 200]
latent: [B, 8, 10, 50]
output: [B, 1, 80, 200]
```

latent 时间维 50 与 50 个视频帧一一对应，每个 latent time token 对应 4 个 Mel frame。

### 7.2 网络

`MelAutoencoder` 提供明确接口：

```python
z = model.encode(mel)
mel_recon = model.decode(z)
mel_recon, z = model(mel)
```

编码器建议使用三次频率下采样、两次时间下采样：

```text
80 x 200
40 x 100
20 x 50
10 x 50
```

decoder 使用对称上采样，并在最终层使用 sigmoid，以兼容当前 `[0, 1]` Mel 归一化。

### 7.3 Loss

```text
L_ae =
  lambda_l1 * L1(mel_recon, mel)
  + lambda_grad * L1(time_gradient(recon), time_gradient(target))
  + lambda_boundary * random_boundary_loss
```

其中 `random_boundary_loss` 在随机时间点比较一阶差分，避免 autoencoder 自身引入时间方向模糊。

### 7.4 训练

1. 使用完整 clean Mel 训练，不注入缺失 mask。
2. 先只优化 L1，稳定后加入 gradient loss。
3. 保存 `encoder`、`decoder`、optimizer、配置和 latent normalization 统计量。
4. 统计训练集 latent 的均值与标准差，在 diffusion 前标准化：

```text
z_norm = (z - mean) / std
```

5. P3 开始后默认冻结 autoencoder。

### 7.5 入口

```bash
python main.py train-mel-ae -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --batch_size 16 \
  --latent_channels 8 \
  --checkpoint_dir checkpoints/mel_ae
```

```bash
python main.py test-mel-ae -- \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --resume_path checkpoints/mel_ae/MelAE_checkpoint_stepXXXXXXXXX.pth.tar
```

### 7.6 验收标准

- encode/decode shape 严格匹配；
- 输出无 NaN/Inf，范围在 `[0, 1]`；
- autoencoder 重建 PSNR/SSIM 显著高于后续 inpainting 模型预期上限；
- random boundary gradient error 足够低；
- Griffin-Lim 导出的 AE 重建音频与原 Mel 导出的音频没有明显额外断裂；
- 冻结后同一输入的 latent 完全确定。

如果 AE 重建质量不足，不进入 diffusion 阶段。

---

## 8. P3：K=1 条件 AV Latent Diffusion

### 8.1 训练变量

训练目标：

```text
z_target = AE.encode(clean_mel)
z_context = AE.encode(corrupted_mel)
```

将 Mel mask 下采样到 latent：

```text
mask_z: [B, 1, 10, 50]
```

mask 下采样使用 max-pooling 或 nearest，保证任何包含缺失 Mel 的 latent cell 都被标记为缺失。

### 8.2 Diffusion 输入

U-Net 输入拼接：

```text
z_t                 [B, 8, 10, 50]
z_context           [B, 8, 10, 50]
mask_z              [B, 1, 10, 50]
boundary_map_z      [B, 2, 10, 50]
```

总输入通道：

```text
8 + 8 + 1 + 2 = 19
```

视觉条件暂时使用 P5 之前的简化版：

```text
video_tokens: [B, 50, 256]
```

在 U-Net 中间层通过 cross-attention 注入。

### 8.3 视觉编码接口

新 `VideoEvidenceEncoder` 在 P3 先实现 token 提取：

```python
{
    "rgb_tokens": Tensor[B, 50, D],
    "flow_tokens": Tensor[B, 50, D],
    "video_tokens": Tensor[B, 50, D],
}
```

可复用当前 RGB/flow ResNet-18 权重，但不要复用当前只返回最终 bottleneck 的 `forward()` 接口。

融合第一版采用：

```text
rgb/flow frame features
-> concat
-> Linear(2D, D)
-> 2 层 temporal Transformer 或 temporal Conv block
-> [B, 50, D]
```

P3 只要求可条件生成，不加入 evidence gate。

### 8.4 Masked diffusion

训练时：

```text
noisy_missing =
  sqrt(alpha_bar_t) * z_target
  + sqrt(1 - alpha_bar_t) * epsilon

z_t =
  mask_z * noisy_missing
  + (1 - mask_z) * z_context
```

diffusion loss 只在缺失 latent 区域计算：

```text
L_diff =
  sum(mask_z * (epsilon - epsilon_pred)^2)
  / sum(mask_z)
```

采样每一步都重新 clamp 已知 latent：

```text
z_t =
  mask_z * z_t_generated
  + (1 - mask_z) * z_context
```

最终 Mel 仍执行一次原分辨率 compose，作为最后的已知区域保护。

### 8.5 Boundary loss

从预测的 `z0_pred` decode 得到 `mel_pred`，只在 gap 左右各 `boundary_width` 帧计算：

```text
L_boundary =
  L1(delta_t(mel_completed), delta_t(mel_target))
```

第一版设置：

```text
boundary_width = 3 Mel frames
```

不要只比较左右单个点，否则指标对局部噪声过于敏感。

### 8.6 训练目标

P3 使用：

```text
L =
  L_diff
  + lambda_boundary * L_boundary
  + lambda_sync * L_sync
```

暂不加入：

```text
L_div
L_score
L_calib
```

`L_sync` 第一版可复用现有全局 AV contrastive 思路，但输入改为预测候选的 audio embedding 与对应 video embedding。

### 8.7 训练策略

1. 冻结 Mel autoencoder。
2. 可从现有 VIAI-AV checkpoint 初始化 RGB/flow ResNet。
3. 先在 `lambda_sync=0` 下确认 diffusion 收敛。
4. 再加入小权重 sync loss。
5. 训练和验证均使用固定的 noise seed 子集生成可视化。
6. 使用 EMA 保存 diffusion 权重，测试默认加载 EMA。

### 8.8 验收标准

- 一步 forward/backward 无 shape 和显存错误；
- 已知 latent 在每个采样 step 后保持不变；
- compose 后已知 Mel 区域严格不变；
- K=1 的 missing Mel L1、PSNR 和 boundary 指标达到可用水平；
- 关闭视频后性能下降，证明模型确实使用视觉条件；
- 固定 seed 时输出可复现。

在 K=1 尚未达到 VIAI-AV 附近水平前，不开始强调多样性。

---

## 9. P4：Multi-Hypothesis Sampling

### 9.1 推理接口

模型统一提供：

```python
result = model.sample(batch, num_candidates=K, seed=seed)
```

返回：

```python
{
    "candidate_mels": Tensor[B, K, 1, 80, 200],
    "completed_mels": Tensor[B, K, 1, 80, 200],
    "candidate_latents": Tensor[B, K, 8, 10, 50],
    "candidate_scores": None,
    "uncertainty": None,
    "visual_evidence": None,
}
```

### 9.2 采样配置

先实现 DDIM：

```text
train_timesteps = 1000
inference_steps = 50
K = 1, 4, 8, 16
```

默认论文主结果使用固定 `K=8`，其他 K 用于消融。

### 9.3 基础多候选指标

实现：

```text
best_of_k_missing_l1
mean_k_missing_l1
best_of_k_boundary_error
mean_k_boundary_error
pairwise_latent_diversity
pairwise_audio_embedding_diversity
```

禁止只报告 Best-of-K。必须同时报告 Mean-K，防止模型靠大量低质量候选取得较好最优值。

### 9.4 导出格式

每个样本目录：

```text
<results_dir>/<sample_id>/
  metadata.json
  input_corrupted.wav
  target.wav
  candidate_00.wav
  candidate_01.wav
  ...
  candidate_07.wav
  candidate_00_mel.png
  ...
```

`metadata.json` 包含：

```json
{
  "mask": {},
  "video_condition": "original",
  "num_candidates": 8,
  "candidate_scores": null,
  "uncertainty": null,
  "metrics": {}
}
```

### 9.5 验收标准

- `K=1` 与单候选入口一致；
- K 个候选的已知区域完全相同；
- 不同 seed 能产生非零但不过度的缺失区差异；
- Best-of-K 随 K 增加改善；
- Mean-K 不出现明显崩溃；
- 候选差异主要出现在缺失区域。

---

## 10. P5：Visual Evidence 与 Evidence-Aware Fusion

### 10.1 Evidence 输出

扩展 `VideoEvidenceEncoder`：

```python
{
    "video_tokens": Tensor[B, 50, D],
    "motion_strength": Tensor[B, 50],
    "sync_logits": Tensor[B, 50],
    "evidence": Tensor[B, 50],
    "visual_logvar": Tensor[B, 50],
}
```

其中：

```text
evidence 越高：视觉越可靠
visual_logvar 越高：视觉不确定性越高
```

### 10.2 Evidence 训练信号

MVP 不使用关键点检测，使用三类信号：

1. optical flow magnitude 和局部 motion peak；
2. 正确对齐与 temporal shift/wrong video 的 sync 分类；
3. 已知视觉退化类型提供的 reliability target。

建议 reliability 初始目标：

| 条件 | reliability target |
| --- | ---: |
| original | 1.0 |
| blur | 0.7 |
| frame_drop | 0.6 |
| occlusion | 0.4 |
| temporal_shift | 0.1 |
| wrong_video | 0.0 |
| no_video | 0.0 |

这只是训练先验，不是最终 evidence 定义。模型仍需结合 motion 和 AV sync 自行预测。

### 10.3 Evidence loss

```text
L_evidence =
  lambda_rel * BCE/Huber(evidence, reliability_target)
  + lambda_sync_cls * sync_classification_loss
  + lambda_temporal * temporal_smoothness
```

`temporal_smoothness` 不应过强，以免抹平真实的瞬时 motion peak。

### 10.4 Audio condition

`AudioConditionEncoder` 输入：

```text
corrupted_mel
missing_mask
left_boundary_map
right_boundary_map
```

输出：

```text
audio_tokens: [B, 50, D]
```

### 10.5 Evidence-Aware Fusion

逐时间位置计算：

```text
g(t) = sigmoid(
  MLP([audio_token(t), video_token(t), evidence(t), visual_logvar(t)])
)

fused(t) =
  audio_token(t)
  + g(t) * video_projection(video_token(t))
```

使用 residual 形式比直接

```text
g * video + (1-g) * audio
```

更容易保持 audio context 主干稳定。

必须记录并可视化：

```text
evidence(t)
gate(t)
motion_strength(t)
sync_confidence(t)
```

### 10.6 Evidence 控制采样

MVP 使用显式的 evidence-conditioned sampling temperature：

```text
e_bar = gap 内 evidence 的平均值
temperature =
  T_min + (T_max - T_min) * (1 - e_bar)
```

建议初值：

```text
T_min = 0.7
T_max = 1.0
```

temperature 作用于缺失 latent 的初始噪声和 DDIM stochasticity。高 evidence 时候选更集中，低 evidence 时保留更多随机性。

正式版本再增加 learnable variance head，并约束 gap 内预测方差与 `1-evidence` 正相关。

### 10.7 Evidence-diversity 约束

训练时每个样本采样 `K_train=2`，在冻结的 audio embedding 空间计算：

```text
D_sample = pairwise_distance(candidate_1, candidate_2)
D_target = alpha * (1 - mean_gap_evidence)
L_div = SmoothL1(D_sample, D_target)
```

该形式同时避免：

- 高 evidence 下候选无意义地发散；
- 低 evidence 下候选完全塌缩。

增加候选质量门槛：只有 boundary、sync 和 reconstruction proxy 达标时才奖励多样性。

### 10.8 验收标准

对同一个 audio gap：

- original video 的平均 evidence 高于 wrong/no-video；
- temporal shift 后 sync confidence 明显下降；
- evidence 与 gate 正相关；
- evidence 高时 pairwise diversity 较低；
- evidence 低时 diversity 上升，但 Mean-K quality 不显著崩溃；
- no-video 时模型仍可依赖 audio context 完成修复。

---

## 11. P6：Candidate Scorer 与 Calibration

### 11.1 Candidate scorer 输入

每个候选提取：

```text
candidate audio embedding
boundary consistency feature
AV sync score
audio-context compatibility
Mel realism feature
mean visual evidence
```

输出：

```text
score_logits: [B, K]
pi = softmax(score_logits)
```

### 11.2 Scorer 监督目标

训练时可以使用 ground truth 构造候选风险：

```text
r_k =
  w_recon * missing_embedding_distance
  + w_boundary * boundary_error
  + w_sync * (1 - av_sync)
```

转为软排序目标：

```text
q_k = softmax(-r_k / tau)
L_score = KL(q || pi)
```

推理时 scorer 不访问 ground truth，只使用候选和已知上下文特征。

### 11.3 不确定性

固定 K 下计算：

```text
u_components =
  normalized_entropy(pi)
  + normalized_pairwise_diversity
  + predicted_risk
```

建议由一个小 head 学习组合，而不是手工永久固定权重：

```text
u = sigmoid(UncertaintyHead(context, evidence, score_stats, diversity))
```

### 11.4 Calibration target

```text
error_target = min_k r_k
```

损失：

```text
L_calib =
  lambda_reg * Huber(u, normalize(error_target))
  + lambda_rank * pairwise_ranking_loss
```

`error_target`、`q_k` 和用于计算它们的候选指标应 `detach()`，避免 calibration head 通过篡改生成器来投机降低监督目标。

### 11.5 后处理校准

训练完成后在 validation set 上拟合轻量的 temperature/isotonic calibration，仅用于校准输出，不使用 test set。

### 11.6 验收标准

- top-1 scorer 候选优于随机候选和固定 candidate 0；
- scorer top-1 接近 oracle best-of-K，但不能用 oracle 信息；
- uncertainty 与实际 error 显著正相关；
- 风险覆盖曲线显示：优先保留低 uncertainty 样本时平均错误下降；
- ECE 优于无 calibration 版本；
- 视觉退化越严重，平均 uncertainty 越高。

---

## 12. 完整 Loss 与训练开关

正式版本：

```text
L_total =
  L_diff
  + lambda_boundary * L_boundary
  + lambda_sync * L_sync
  + lambda_evidence * L_evidence
  + lambda_div * L_div
  + lambda_score * L_score
  + lambda_calib * L_calib
```

`L_known` 主要通过每个 diffusion step 的 latent clamp 和最终 Mel compose 保证。可以保留一个数值检查项，但不应依赖它代替硬约束。

所有 loss 必须支持单独关闭：

```text
--disable_boundary_loss
--disable_sync_loss
--disable_evidence_gate
--disable_diversity_loss
--disable_candidate_scorer
--disable_calibration_loss
```

TensorBoard 同时记录原始值和加权值：

```text
loss/diff
loss/boundary
loss/sync
loss/evidence
loss/diversity
loss/score
loss/calibration
weighted/*
```

---

## 13. Checkpoint 设计

UQ checkpoint 至少包含：

```python
{
    "stage": "UQ-AVInpaint-Px",
    "config": ...,
    "mel_autoencoder": ...,
    "audio_condition_encoder": ...,
    "video_evidence_encoder": ...,
    "diffusion_unet": ...,
    "candidate_scorer": ...,
    "uncertainty_head": ...,
    "ema": ...,
    "optimizers": ...,
    "schedulers": ...,
    "global_step": ...,
    "global_epoch": ...,
    "latent_stats": ...,
}
```

加载时必须校验：

- latent channels；
- Mel shape；
- diffusion schedule；
- 是否启用 evidence/scorer/calibration；
- autoencoder checkpoint 哈希；
- checkpoint stage。

不允许静默加载 shape 不匹配的核心 diffusion 权重。

---

## 14. 测试与指标

### 14.1 单目标指标

复用并扩展：

```text
Mel L1 full/missing
PSNR full/missing
SSIM
Log-spectral distance
```

### 14.2 Boundary 指标

实现：

```text
boundary_mel_l1
boundary_delta_l1
boundary_energy_jump
```

wav 质量稳定后再增加 click rate。

### 14.3 AV Sync

```text
global_av_sync
onset_motion_alignment
temporal_offset_error
```

### 14.4 多候选

```text
best_of_k_error
mean_k_error
top1_scorer_error
oracle_gap = top1_scorer_error - best_of_k_error
pairwise_diversity
quality_diversity_curve
```

### 14.5 Calibration

```text
Pearson/Spearman uncertainty-error correlation
ECE
Brier score
AUROC for high-error detection
risk-coverage curve
AURC
```

### 14.6 统计要求

- 指标按 gap length、mask type、instrument 和 video condition 分组；
- 报告均值和 bootstrap 置信区间；
- 主比较使用完全相同的 sample/mask/video-condition manifest；
- 多候选主结果固定 K，避免因 K 不同造成不可比。

---

## 15. `test-uq-av` 输出

建议命令：

```bash
python main.py test-uq-av -- \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --mask_manifest "$DATA_ROOT/uq_metadata/test_masks.jsonl" \
  --resume_path checkpoints/uq_av/UQ-AVInpaint_checkpoint_stepXXXXXXXXX.pth.tar \
  --num_candidates 8 \
  --ddim_steps 50 \
  --results_dir checkpoints/uq_av_test_results \
  --use_vocoder
```

总结果：

```text
<results_dir>/
  summary.json
  samples.jsonl
  metrics.csv
  metrics_by_gap.csv
  metrics_by_video_condition.csv
  risk_coverage.csv
  calibration_bins.csv
  candidates/
```

`samples.jsonl` 每行保存单个样本、单个 mask、单个视觉条件下的完整结果，便于后续重新聚合而无需再次推理。

---

## 16. 实验矩阵

### 16.1 主模型

```text
VIAI-A
VIAI-AV
Audio-only Diffusion K=1
Audio-only Diffusion K=8
AV Diffusion K=1
AV Diffusion K=8, no evidence
UQ-AVInpaint K=8
```

### 16.2 消融

```text
w/o boundary loss
w/o sync loss
w/o visual evidence gate
w/o visual corruption training
w/o diversity target
w/o candidate scorer
w/o calibration loss
fixed sampling temperature
```

### 16.3 视觉条件

```text
original
blur
occlusion
frame_drop
temporal_shift
wrong_video
no_video
```

### 16.4 Gap

```text
0.4 s / 20 frames
0.8 s / 40 frames
1.0 s / 50 frames
1.2 s / 60 frames
1.6 s / 80 frames
2.0 s / 100 frames
```

2 秒 gap 只剩约 1 秒左右两侧 context，必须单独报告，不能和 0.4-1.0 秒结果直接平均后隐藏难度差异。

---

## 17. MVP 范围

第一轮建议只完成：

1. P0 固定基线；
2. P1 独立 mask 与固定 manifest；
3. P2 Mel autoencoder；
4. P3 K=1 AV latent diffusion；
5. P4 K=8 sampling；
6. P5 简化 evidence：
   - optical flow strength；
   - global/temporal sync；
   - evidence-conditioned temperature；
7. uncertainty 暂定为归一化 pairwise embedding diversity。

MVP 成功标准：

```text
Best-of-8 优于 K=1
Mean-K 质量没有明显崩溃
original video 的 diversity 低于 wrong/no-video
视觉退化后 uncertainty 单调上升
boundary 指标不弱于 VIAI-AV
```

如果这些趋势不成立，不应立即增加复杂 scorer，而应先排查：

- diffusion 是否真正使用视频；
- evidence target 是否只学到退化标签；
- sampling temperature 是否造成分布外 latent；
- AE 是否损失了 onset 和边界细节；
- sync 指标是否与人类感知一致。

---

## 18. 正式版本增量

MVP 成立后再加入：

1. learnable diffusion variance；
2. quality-aware diversity loss；
3. candidate scorer；
4. uncertainty calibration head；
5. validation 后处理校准；
6. onset-motion fine sync；
7. 更强 vocoder；
8. 人类主观实验页面与盲测导出工具。

vocoder 升级时必须同时保留：

```text
same-vocoder comparison
best-quality vocoder demo
```

避免把 vocoder 提升误认为 inpainting 模型提升。

---

## 19. 推荐开发顺序与阶段产物

| 阶段 | 主要改动 | 必须产物 |
| --- | --- | --- |
| P0 | 固定基线 | baseline JSON/CSV、固定 manifest |
| P1 | 新 loader、mask、退化 | 单元测试、数据可视化 |
| P2 | Mel AE | AE checkpoint、重建报告 |
| P3 | K=1 diffusion | deterministic AV diffusion 报告 |
| P4 | K sampling | Best/Mean-K 与 diversity 报告 |
| P5 | evidence gate | 视觉退化趋势图 |
| P6 | scorer/calibration | top-1、ECE、risk-coverage |
| P7 | 全实验 | 主表、消融表、demo 样本 |

每个阶段都建议建立独立实验目录，不复用 TensorBoard 路径，不覆盖旧 checkpoint。

---

## 20. 近期第一批开发任务

按依赖关系，第一批代码任务建议严格按以下顺序执行：

1. 新建 `tests/`，为现有 compose 和新 mask 语义建立测试；
2. 实现 `MaskSpec`、逐样本 mask 和 boundary map；
3. 实现固定 val/test mask manifest；
4. 新建 `uq_av_loader.py`，输出字典并通过 shape 测试；
5. 注册 `prepare-uq-metadata`；
6. 实现 Mel autoencoder 与 shape/reconstruction 测试；
7. 注册 `train-mel-ae`、`test-mel-ae`；
8. 完成 AE smoke test 并检查重建上限；
9. 实现 diffusion schedule、masked noising 和 known-region clamp；
10. 实现最小 conditional U-Net，先用 no-video/audio-only 条件跑通；
11. 接入逐帧 video token，形成 AV Diffusion K=1；
12. 扩展 K-sampling、指标和候选导出；
13. 最后再实现 evidence、scorer 和 calibration。

这批任务中，前 8 项完成前不应开始训练 diffusion；P3 的 K=1 未稳定前不应实现 calibration。

---

## 21. 关键风险与处理

### 风险 1：数据量不足以训练 diffusion

处理：

- 先统计完整 processed clip 和 instrument 数量；
- 扩大 clip 数前先保证 video-level split；
- 使用数据增强和 EMA；
- 必要时先做 audio-only diffusion 预训练，再加入视频条件。

### 风险 2：autoencoder 成为质量瓶颈

处理：

- 把 AE 重建指标作为进入 P3 的硬门槛；
- 增加 latent channel 或减少压缩率；
- 优先保留时间分辨率；
- 不要用 diffusion loss 掩盖 AE 重建失败。

### 风险 3：模型忽略视频

处理：

- 训练时加入 no-video、wrong-video、temporal-shift 对照；
- 记录 cross-attention/gate；
- 比较 original 与 shuffled video；
- 使用候选级 sync loss。

### 风险 4：多样性只是噪声

处理：

- 在感知 embedding 而非 waveform 像素空间衡量；
- 同时约束 boundary、sync 和 Mean-K quality；
- 人类实验区分“合理变化”和“噪声变化”。

### 风险 5：uncertainty 只识别人工退化

处理：

- evidence 输入同时包含 motion、sync 和内容特征；
- 在未见过的退化强度上测试；
- 单独报告自然低运动、遮挡和镜头切换样本；
- calibration target 使用实际生成错误，而不只是退化类别。

### 风险 6：计算量过大

处理：

- AE 冻结并预计算可选 clean latent cache；
- P3 训练使用单样本 diffusion loss；
- 只在 P5/P6 小比例 batch 上做 `K_train=2`；
- 推理用 DDIM 50 steps；
- K=16 只用于离线消融。

---

## 22. 完成定义

UQ-AVInpaint 的工程实现完成，需要同时满足：

1. VIAI-A/VIAI-AV 原入口仍可运行；
2. 新数据协议可复现并有单元测试；
3. AE、diffusion、evidence、scorer 可分阶段训练和恢复；
4. K 候选、置信度和 uncertainty 可通过统一接口输出；
5. 已知区域在最终结果中严格不变；
6. 测试结果包含 Best-of-K、Mean-K、boundary、sync、diversity 和 calibration；
7. 视觉退化实验呈现“证据越弱，不确定性越高”的趋势；
8. scorer top-1 优于随机候选；
9. uncertainty-error correlation、ECE 和 risk-coverage 可计算；
10. 所有主结果都能追溯到配置、checkpoint、manifest、seed 和代码版本。

做到以上十项，`thoUght.md` 中的研究设想才算从概念变成了可训练、可比较、可复现的系统。
