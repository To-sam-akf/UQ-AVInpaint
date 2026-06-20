# UQ-AVInpaint 详细执行计划

本文根据 `UQ_AVInpaint_implementation_plan.md` 整理，用于指导后续工程落地。计划以“先固定数据契约，再分阶段训练模型”为主线，所有阶段都必须满足四类信息：需求、核心逻辑、验证方法、预期结果。

## 0. 总体需求

### 需求

- 在不破坏现有 `train-viai-a`、`test-viai-a`、`train-viai-av`、`test-viai-av` 行为的前提下，新增独立 UQ-AVInpaint 路线。
- 支持多类型缺失区域、固定测试 manifest、视觉退化、Mel latent autoencoder、条件 latent diffusion、K 候选采样、视觉证据估计、候选排序和不确定性校准。
- 最终模型需要输出：
  - `K` 个修复候选；
  - 每个候选置信度 `pi_k`；
  - 样本级不确定性 `u`；
  - 逐时间视觉证据分数 `e(t)`。
- 所有新模块统一遵守 `mask=1` 表示缺失、`mask=0` 表示已知区域。
- 测试、导出和指标计算必须复用同一个 compose 逻辑，保证已知区域严格不被模型覆盖。

### 核心逻辑

- 保留原 VIAI-A/VIAI-AV 作为可复现基线。
- 新增独立入口：
  - `prepare-uq-metadata`
  - `train-mel-ae`
  - `test-mel-ae`
  - `train-uq-av`
  - `test-uq-av`
- 新增 UQ 数据协议，dataloader 返回字典而不是固定 tuple。
- 训练顺序严格采用：
  - P0 固定基线；
  - P1 数据、mask、测试协议；
  - P2 Mel autoencoder；
  - P3 K=1 AV latent diffusion；
  - P4 K-sampling；
  - P5 visual evidence 与 uncertainty gate；
  - P6 candidate scorer 与 calibration；
  - P7 完整评估与消融。
- 每一阶段达到验收标准后再进入下一阶段，避免 AE、diffusion、evidence、scorer 和 calibration 同时耦合调试。

### 验证方法

- 使用单元测试覆盖 mask、compose、shape、已知区域保护、manifest 复现、diffusion clamp、K 候选导出和 UQ 指标。
- 使用 smoke test 验证每个训练入口至少可完成一个 batch 的 forward/backward。
- 使用固定 seed、固定 split、固定 mask manifest 验证结果可复现。
- 对所有主结果记录配置、checkpoint、manifest、seed、git commit 和完整命令。

### 预期结果

- 原 VIAI 路线可继续运行并作为对照。
- UQ-AVInpaint 可以分阶段训练、恢复、测试和导出。
- 每个阶段都有独立实验目录、指标文件和可追溯元数据。
- 最终结果可比较 Best-of-K、Mean-K、boundary、sync、diversity、calibration 和 uncertainty-error correlation。

## 1. P0：冻结并记录 VIAI 基线

### 需求

- 固定 VIAI-A、VIAI-AV、VIAI-AA' probe reference 的可复现实验结果。
- 固定 train/val/test split，保持 video-level 隔离。
- 固定测试 mask 清单，避免测试阶段随机采样导致模型间不可比。
- 将基线结果保存到 `experiments/baselines/` 下的独立目录。

### 核心逻辑

- 使用同一批测试样本和同一组 gap 分别测试 VIAI-A、VIAI-AV、VIAI-AA'。
- 为每个测试样本记录 `sample_id`、`mask_type`、`start`、`end`、`gap_frames`。
- 保存 checkpoint 路径、git commit、命令、配置 JSON、mask manifest、JSON/CSV 指标、Mel 图和少量 wav。
- 已知区域合成统一使用：

```python
final = mel_input * (1.0 - mask) + mel_pred * mask
```

### 验证方法

- 同一 checkpoint 连续测试两次，比较指标、CSV 和 JSON 是否一致。
- 检查三个基线模型是否读取同一 manifest。
- 单元测试确认 compose 后已知区域逐元素误差为 0。
- 检查结果文件是否包含 sample/mask 元数据。

### 预期结果

- 生成：
  - `experiments/baselines/viai_a/`
  - `experiments/baselines/viai_av/`
  - `experiments/baselines/viai_aa_probe/`
- 得到一组可长期复用的基线指标，后续所有 UQ-AVInpaint 结果都能与之公平比较。

### P0 验证结果 (2026-06-18)

#### 单元测试

| 测试文件 | 测试数量 | 结果 |
|---|---|---|
| `tests/test_baseline_protocol.py` | 7 | ✅ 全部通过 |
| `tests/test_baseline_runner.py` | 4 | ✅ 全部通过 |
| `tests/test_baseline_evaluation.py` | 5 | ✅ 全部通过 |
| `tests/test_mask_sampler.py` | 3 | ✅ 全部通过 |
| `tests/test_uq_av_data.py` | 4 | ✅ 全部通过 |

**总计: 23/23 测试通过 (100%)**

#### 验证项逐项确认

| # | 验证项 | 方法 | 结果 |
|---|---|---|---|
| 1 | compose 后已知区域逐元素误差为 0 | `compose_inpainted_mel` 后检查 `|final - input| * (1-mask)` 最大值 | ✅ 严格为 0 |
| 2 | 同一 checkpoint 连续测试两次，指标一致 | `generate_mask_specs(rows, seed)` 两次调用结果完全相等 | ✅ byte-to-byte 一致 |
| 3 | 三个基线模型读取同一 manifest | `build_baseline_commands` 输出 3 条命令均含 `--baseline-mask-manifest` 和 `--baseline-protocol-json` | ✅ 统一指向同一 manifest |
| 4 | 结果文件包含 sample/mask 元数据 | `compute_sample_records` 输出含 `sample_id`, `mask_type`, `start`, `end`, `gap_frames`, `known_region_max_abs_error` | ✅ 所有字段齐全 |
| 5 | split video-level 隔离 | `audit_model_splits` 检测跨 phase 泄漏, `audit_cross_model_assignments` 检测 VIAI-A/VIAI-AV 分配冲突 | ✅ 泄漏/冲突均被拒绝 |
| 6 | Protocol 可复现 | `create_baseline_protocol` 两次调用产生 byte-identical mask manifest + 相同 SHA256 | ✅ 完全可复现 |
| 7 | GAN/probe 标记检测 | `inspect_checkpoint` 从 checkpoint 字典检测 `netD` 和 `enable_probe_loss` | ✅ 检测正确，probe 缺失时 refuse |
| 8 | Mask contract (mask=1 表示缺失) | `build_missing_mask` 中 mask=1 精确在 `[start:end]`，其余为 0 | ✅ 严格遵守 |

#### 命令行

```bash
# ============================================================
# P0 验证命令
# ============================================================

# 1. 运行全部单元测试 (23 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_baseline_protocol.py \
  tests/test_baseline_runner.py \
  tests/test_baseline_evaluation.py \
  tests/test_mask_sampler.py \
  tests/test_uq_av_data.py \
  -v --tb=short

# 2. 验证 CLI 入口点
python main.py freeze-viai-baselines -- --help
python main.py test-viai-a -- --help
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-viai-av -- --help

# 3. 冻结基线 (需要 checkpoint 和 data_root)
# python main.py freeze-viai-baselines -- \
#   --data-root /path/to/data \
#   --viai-a-checkpoint /path/to/viai_a.pth.tar \
#   --viai-av-checkpoint /path/to/viai_av.pth.tar \
#   --output-root experiments/baselines \
#   --batch-size 16 --num-workers 4 --seed 1234
```

## 2. P1：数据契约、Mask 与视觉退化

### 需求

- 新增 `Data_loaders/uq_av_loader.py`，输出可扩展字典。
- 新增 `Data_loaders/mask_sampler.py`，支持逐样本 mask 和固定 manifest。
- 新增 `tools/prepare_uq_metadata.py`，生成 onset 信息、val/test mask manifest 和 summary。
- 支持可复现视觉退化：
  - `original`
  - `blur`
  - `occlusion`
  - `frame_drop`
  - `temporal_shift`
  - `wrong_video`
  - `no_video`

### 核心逻辑

- 新 dataloader 输出：

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

- `MaskSpec` 定义：

```python
@dataclass
class MaskSpec:
    mask_type: str
    start: int
    end: int
    gap_frames: int
    seed: int
```

- 第一版 mask 类型：
  - `random`：合法区间均匀采样；
  - `onset_centered`：围绕高 spectral-flux/onset 帧；
  - `boundary_near`：靠近窗口边缘但保留最小 clean context；
  - `long_gap`：60/80/100 Mel frames。
- `boundary_map` 使用左右边界距离两个归一化通道。
- train 阶段每个样本独立采样；val/test 阶段从 manifest 读取固定 `MaskSpec`。
- 视觉退化记录参数，如 `blur_sigma`、`occlusion_box`、`frame_keep_ratio`、`temporal_shift_frames`、`wrong_video_sample_id`。

### 验证方法

- 测试 batch 内不同样本可使用不同 gap。
- 验证 `mask=1` 的位置与 `start/end` 一致。
- 验证 `boundary_map` 在左右边界附近数值正确。
- 验证固定 seed 可复现相同 mask 和视觉退化。
- 验证 `temporal_shift` 和 `wrong_video` 不改变音频输入。
- 验证新 loader 可在现有 AV 数据上完成一个 batch，shape 与契约一致。
- 回归测试确认原 VIAI loader 行为不变。

### 预期结果

- 生成：
  - `<data_root>/uq_metadata/train_onsets/*.npy`
  - `<data_root>/uq_metadata/val_masks.jsonl`
  - `<data_root>/uq_metadata/test_masks.jsonl`
  - `<data_root>/uq_metadata/metadata_summary.json`
- UQ 数据协议稳定，后续 AE、diffusion、evidence、scorer 共用同一 batch 结构。
- 测试集重跑时 sample/mask/video-condition 完全一致。

### P1 验证结果 (2026-06-18)

#### 单元测试

| 测试文件 | 测试数量 | 结果 |
|---|---|---|
| `tests/test_mask_sampler.py` | 3 | ✅ 全部通过 |
| `tests/test_uq_av_data.py` | 4 | ✅ 全部通过 |
| `tests/test_visual_degradation.py` | 17 | ✅ 全部通过 |
| `tests/test_viai_loader_regression.py` | 12 | ✅ 全部通过 |
| `tests/test_p1_integration.py` | 11 | ✅ 全部通过 |

**P1 总计: 47/47 测试通过 (100%)**
**P0+P1 回归: 63/63 测试通过 (100%)**

#### 验证项逐项确认

| # | 验证项 | 方法 | 测试 | 结果 |
|---|---|---|---|---|
| 1 | batch 内不同样本可使用不同 gap | 训练 batch 中检查 `mask_spec` 的 `gap_frames` 多样性 | `test_batch_varied_mask_types` | ✅ 多种 gap 出现 |
| 2 | mask=1 位置与 start/end 一致 | 对 4 种 mask 类型分别检查 `missing_mask` 的 1 区域 | `test_mask_ones_match_spec_boundaries`, `test_all_mask_types_are_deterministic_and_valid` | ✅ 严格一致 |
| 3 | boundary_map 在左右边界数值正确 | 检查 `boundary_map[0]` 在 start 处为 0、`boundary_map[1]` 在 end-1 处为 0 | `test_boundary_map_at_edges`, `test_mask_boundary_map_and_corruption_follow_contract` | ✅ 数值精确 |
| 4 | 固定 seed 可复现相同 mask | 两个独立 `UQAVDataset` 实例 (相同 seed) 逐样本比对 `mask_spec` | `test_training_masks_are_per_sample_and_epoch_reproducible` | ✅ byte-identical |
| 5 | 固定 seed 可复现视觉退化 | 对所有 `VIDEO_CONDITIONS` 逐元素比对 video/flow tensor | `test_visual_degradations_are_reproducible_and_preserve_audio`, `test_joint_reproducibility` | ✅ 完全一致 |
| 6 | mask + 视觉退化联合复现 | eval 阶段相同 seed 下 `mask_spec` + `video` + `flow` + `video_degradation` 全部相等 | `test_joint_reproducibility` | ✅ 联合一致 |
| 7 | temporal_shift 不改变音频 | 比对 original 与 temporal_shift 的 `mel_target`/`mel_corrupted`/`audio_target` | `test_visual_degradations_are_reproducible_and_preserve_audio` | ✅ 音频不变 |
| 8 | wrong_video 不改变音频 | 比对 original 与 wrong_video 的 `mel_target`/`mel_corrupted`/`audio_target`/`missing_mask` | `test_wrong_video_preserves_audio`, `test_visual_degradations_are_reproducible_and_preserve_audio` | ✅ 音频不变 |
| 9 | loader 完成一个 batch 且 shape 与契约一致 | 使用 `create_uq_av_dataloader` 迭代一个 batch, 检查所有 tensor key 的 shape | `test_loader_contract_manifest_expansion_and_collate`, `test_smoke_all_phases`, `test_collate_fn_smoke` | ✅ 所有 shape 匹配 |
| 10 | train/val/test 三阶段 smoke | 对 train/val/test 各创建 dataloader 并迭代一个 batch | `test_smoke_all_phases` | ✅ 三阶段均成功 |
| 11 | 原 VIAI loader 行为不变 | 测试 `VIAIASplitDataset` 的 split 读取、pad、collate、shape 合约 | `test_viai_loader_regression.py` (12 tests) | ✅ 行为一致 |
| 12 | visual_degradation 参数记录 | 验证 blur_sigma, occlusion_box, frame_keep_ratio, temporal_shift_frames, wrong_video_sample_id 均被记录 | `test_blur_records_sigma`, `test_occlusion_records_box`, `test_frame_drop_records_ratio`, `test_temporal_shift_records_offset`, `test_wrong_video_uses_replacement` | ✅ 参数齐全 |
| 13 | metadata 生成 byte 可复现 | 两次 `prepare_uq_metadata` 调用后逐字节比对所有输出文件 | `test_metadata_is_byte_reproducible_and_complete` | ✅ byte-identical |
| 14 | mask manifest 覆盖率检查 | 验证 manifest 的 sample_id 集合与 split 完全一致 (missing/extra 均被拒绝) | `test_loader_contract_manifest_expansion_and_collate` | ✅ 覆盖一致 |
| 15 | long_gap 边界约束 | 20 次采样验证 `long_gap` 的 start/end 始终在 [3, 197] 内 | `test_long_gap_within_bounds` | ✅ 严格约束 |
| 16 | MaskSpec 序列化 round-trip | `to_dict()` → `from_dict()` 后相等 | `test_mask_spec_serialization` | ✅ |
| 17 | prepare-uq-metadata CLI 入口 | 通过 `main.py prepare-uq-metadata -- --help` 和直接调用 `main()` 验证 | `test_prepare_uq_metadata_cli`, CLI help 输出 | ✅ 入口可用 |

#### 命令行

```bash
# ============================================================
# P1 验证命令
# ============================================================

# 1. 运行全部 P1 单元测试 (47 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_mask_sampler.py \
  tests/test_uq_av_data.py \
  tests/test_visual_degradation.py \
  tests/test_viai_loader_regression.py \
  tests/test_p1_integration.py \
  -v --tb=short

# 2. 运行 P0+P1 回归测试 (63 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_baseline_protocol.py \
  tests/test_baseline_runner.py \
  tests/test_baseline_evaluation.py \
  tests/test_mask_sampler.py \
  tests/test_uq_av_data.py \
  tests/test_visual_degradation.py \
  tests/test_viai_loader_regression.py \
  tests/test_p1_integration.py \
  -v --tb=short

# 3. 验证 CLI 入口点
python main.py prepare-uq-metadata -- --help

# 4. 生成 UQ metadata (需要 data_root)
# python main.py prepare-uq-metadata -- \
#   --data-root /path/to/data \
#   --output-dir /path/to/data/uq_metadata \
#   --seed 1234

# 5. 验证 UQ dataloader 可独立实例化并完成一个 batch
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -c "
from Data_loaders.uq_av_loader import UQAVDataset, create_uq_av_dataloader
print('UQAVDataset and create_uq_av_dataloader import successfully')
from Data_loaders.mask_sampler import MaskSampler, MaskSpec, MASK_TYPES
print(f'Mask types: {MASK_TYPES}')
from Data_loaders.visual_degradation import VIDEO_CONDITIONS
print(f'Video conditions: {VIDEO_CONDITIONS}')
print('All P1 modules import successfully')
"
```

## 3. P2：Mel Latent Autoencoder

### 需求

- 新增确定性 convolutional Mel autoencoder。
- 输入输出保持 `[B, 1, 80, 200]`，latent 建议为 `[B, 8, 10, 50]`。
- latent 时间维为 50，与 50 个视频帧对齐。
- P3 之后默认冻结 autoencoder，并使用训练集 latent 均值方差进行标准化。

### 核心逻辑

- 新增模块：
  - `Models/Mel_Autoencoder.py`
  - `networks/uq/mel_autoencoder.py`
  - `train_mel_autoencoder.py`
  - `test_mel_autoencoder.py`
- 明确接口：

```python
z = model.encode(mel)
mel_recon = model.decode(z)
mel_recon, z = model(mel)
```

- 编码器下采样路线：
  - `80 x 200`
  - `40 x 100`
  - `20 x 50`
  - `10 x 50`
- decoder 对称上采样，最终层使用 sigmoid，兼容当前 `[0, 1]` Mel 归一化。
- loss：

```text
L_ae =
  lambda_l1 * L1(mel_recon, mel)
  + lambda_grad * L1(time_gradient(recon), time_gradient(target))
  + lambda_boundary * random_boundary_loss
```

- 训练策略：
  - 使用完整 clean Mel，不注入缺失 mask；
  - 先只优化 L1；
  - 稳定后加入 gradient/boundary loss；
  - 保存 encoder、decoder、optimizer、配置和 latent normalization 统计量。

### 验证方法

- 单元测试确认 encode/decode shape 严格匹配。
- 检查输出范围在 `[0, 1]`，且无 NaN/Inf。
- 用 test 入口计算 AE 重建 PSNR、SSIM、Mel L1、boundary gradient error。
- 使用 Griffin-Lim 导出少量 AE 重建音频，检查是否引入明显额外断裂。
- 冻结后同一输入重复 encode，latent 必须完全确定。

### 预期结果

- 生成可复用 AE checkpoint：
  - `checkpoints/mel_ae/MelAE_checkpoint_step*.pth.tar`
- AE 重建质量明显高于后续 inpainting 模型预期上限。
- 若 AE 重建质量不足，暂停进入 P3，优先调整 latent channels、压缩率和时间分辨率。

### P2 验证结果 (2026-06-18)

#### 单元测试

| 测试文件 | 测试数量 | 结果 |
|---|---|---|
| `tests/test_mel_autoencoder.py` | 37 | ✅ 全部通过 |

**P0+P1+P2 回归: 100/100 测试通过 (100%)**

#### 测试覆盖明细

| 测试类 | 测试数 | 覆盖内容 |
|---|---|---|
| `TestNetworkShapes` | 8 | encode/decode/forward shape 严格匹配, 3D/4D 输入兼容, 自定义 latent_dim |
| `TestOutputQuality` | 3 | 输出范围 [0, 1], 无 NaN/Inf, 极端输入 (全零/全一/噪声) |
| `TestDeterminism` | 3 | 相同输入 → 相同 latent, 不同输入 → 不同 latent, 10 次重复一致性 |
| `TestTimeGradient` | 4 | shape、常值输入梯度为零、线性输入梯度为 1、阶跃检测 |
| `TestRandomBoundaryLoss` | 4 | 标量输出、相同输入损失为零、不同输入非零、短序列不崩溃 |
| `TestMelAEModel` | 8 | 初始化、set_input/forward、encode/decode 接口、loss 计算、warmup 阶段仅 L1、optimizer step 参数更新、test 模式无梯度、get_current_errors |
| `TestCheckpoint` | 2 | save/load round-trip 权重一致、latent_stats 持久化 |
| `TestLatentStats` | 2 | 统计量确定性、mean/std shape 正确 |
| `TestReconstructionSanity` | 3 | 无 NaN、encode-decode round-trip shape、batch 独立性 |

#### 验证项逐项确认

| # | 验证项 | 方法 | 测试 | 结果 |
|---|---|---|---|---|
| 1 | encode/decode shape 严格匹配 | 检查 encoder 输出 [B,8,10,50], decoder 输出 [B,1,80,200] | `test_encoder_shape_4d`, `test_decoder_shape`, `test_autoencoder_forward_4d` 等 8 项 | ✅ 所有 shape 精确匹配 |
| 2 | 输出范围在 [0, 1] | 检查 recon min/max, 极端输入测试 | `test_output_in_01_range`, `test_output_range_with_extreme_input` | ✅ sigmoid 约束有效 |
| 3 | 无 NaN/Inf | `torch.isnan`/`torch.isinf` 检查 | `test_no_nan_inf`, `test_reconstruction_not_nan` | ✅ 干净 |
| 4 | 冻结后确定性 encode | 相同输入重复 encode 10 次 | `test_same_input_same_latent`, `test_single_sample_determinism` | ✅ 完全一致 |
| 5 | 不同输入不同 latent | 随机不同输入 encode 后 latent 不相等 | `test_different_inputs_different_latents` | ✅ 区分不同输入 |
| 6 | batch 独立性 | 单样本 encode vs 批次 encode 结果一致 | `test_batch_independence` | ✅ 样本独立 |
| 7 | time_gradient 正确性 | 线性/常数/阶跃输入验证 | `test_constant_input_zero_gradient`, `test_linear_input`, `test_step_change` | ✅ 梯度计算正确 |
| 8 | random_boundary_loss 行为 | 相同输入为零、不同输入非零 | `test_identical_inputs_zero_loss`, `test_different_inputs_nonzero_loss` | ✅ 符合预期 |
| 9 | optimizer step 参数更新 | 前后参数对比 | `test_optimizer_step` | ✅ 参数更新 |
| 10 | warmup 阶段仅 L1 | warmup 期 total = lambda_l1 * l1 | `test_warmup_only_l1` | ✅ 仅 L1 |
| 11 | checkpoint save/load | round-trip 权重逐参数比对 | `test_save_load_roundtrip` | ✅ 权重一致 |
| 12 | latent_stats 持久化 | mean/std 写入 checkpoint 并正确恢复 | `test_load_preserves_latent_stats` | ✅ 统计量持久化 |
| 13 | latent_stats 确定性 | 相同数据两次计算一致 | `test_compute_stats_deterministic` | ✅ 确定性 |

#### 新增文件

| 文件 | 用途 |
|---|---|
| `networks/uq/__init__.py` | UQ 子包 |
| `networks/uq/mel_autoencoder.py` | Conv AE 网络：MelEncoder, MelDecoder, MelAutoencoder, time_gradient, random_boundary_loss |
| `Models/Mel_Autoencoder.py` | 训练 wrapper：loss 管理、checkpoint、latent 统计量 |
| `train_mel_autoencoder.py` | 训练入口：Phase 1 L1 warmup → Phase 2 gradient+boundary loss |
| `test_mel_autoencoder.py` | 测试入口：PSNR/SSIM/Mel L1/gradient error/确定性/Griffin-Lim 导出 |
| `tests/test_mel_autoencoder.py` | 37 项单元测试 |

#### 架构确认

```
Encoder: 80×200 → 40×100 → 20×50 → 10×50  (3 downsample: (2,2), (2,2), (2,1))
Decoder: 10×50 → 20×50 → 40×100 → 80×200  (symmetric upsample, final sigmoid)
Latent:  [B, 8, 10, 50]  (50 time frames ↔ 50 video frames aligned)
```

#### 命令行

```bash
# ============================================================
# P2 验证命令
# ============================================================

# 1. 运行 P2 单元测试 (37 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_mel_autoencoder.py \
  -v --tb=short

# 2. 运行 P0+P1+P2 回归测试 (100 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_baseline_protocol.py \
  tests/test_baseline_runner.py \
  tests/test_baseline_evaluation.py \
  tests/test_mask_sampler.py \
  tests/test_uq_av_data.py \
  tests/test_visual_degradation.py \
  tests/test_viai_loader_regression.py \
  tests/test_p1_integration.py \
  tests/test_mel_autoencoder.py \
  -v --tb=short

# 3. 验证 CLI 入口点
python main.py train-mel-ae -- --help
python main.py test-mel-ae -- --help

# 4. 导入验证
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -c "
from networks.uq.mel_autoencoder import MelAutoencoder, MelEncoder, MelDecoder, time_gradient, random_boundary_loss
print('networks.uq.mel_autoencoder — all symbols imported')
from Models.Mel_Autoencoder import MelAEModel
print('Models.Mel_Autoencoder — MelAEModel imported')
print('All P2 modules import successfully')
"

# ============================================================
# P2.5 数据准备 (解决 visual frames 不完整问题)
# ============================================================

# 5a. 创建 processed -> processed_viai_av 软链接 (split 文件引用 'processed/' 前缀)
ln -sf processed_viai_av /root/shared-nvme/data/processed

# 5b. 过滤 visual frames 不足的 clip (17/3762 clips 有缺帧或缺失目录)
# 生成 train_av_split.txt / val_av_split.txt
# train: 3196 -> 3184 (移除12), val: 187 -> 185 (移除2), test: 376 -> 376 (无需修改)

# 5c. 生成 UQ metadata (onset + val/test mask manifest)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py prepare-uq-metadata -- \
  --data-root /root/shared-nvme/data \
  --train-split-name train_av_split.txt \
  --val-split-name val_av_split.txt \
  --test-split-name test_av_split.txt \
  --output-dir /root/shared-nvme/data/uq_metadata \
  --seed 1234

# ============================================================
# P2 训练 & 测试
# ============================================================

# 6. 训练 Mel AE (RTX 4090, ~7-8 小时)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-mel-ae -- \
  --name MelAE \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoints_dir /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints \
  --batch_size 16 --nepochs 50 \
  --ae_latent_dim 8 --ae_base_channels 32 \
  --ae_warmup_steps 2000 \
  --num_workers 4 --lr 1e-4 \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar


# 7. 测试 Mel AE (需要训练好的 checkpoint)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-mel-ae -- \
  --name MelAE_test \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step*.pth.tar \
  --checkpoints_dir /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints \
  --batch_size 16 --use_vocoder \
  --results_dir /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/test_results
```

### P2 训练 & 测试结果 (2026-06-18)

#### 数据准备

| 步骤 | 结果 |
|---|---|
| 软链接 `processed -> processed_viai_av` | ✅ 创建成功 |
| 扫描全部 3762 clips 的 visual frames | 17 clips 不合格 (0.5%) |
| 过滤 split 文件 | train: 3196→3184, val: 187→185, test: 376→376 |
| `prepare-uq-metadata` 生成 | ✅ train=3184, val=185, test=376 |

#### 训练状态 (运行中)

| 项目 | 值 |
|---|---|
| 启动时间 | 2026-06-18 12:55 |
| 设备 | NVIDIA RTX 4090 (24 GB) |
| 训练样本 | 3184 (199 batches/epoch) |
| 验证样本 | 740 eval entries (47 batches) |
| 学习率 | 1e-4 |
| Warmup steps | 2000 (仅 L1 loss) |

**Epoch 1 初期 loss 变化:**

| Step | L1 Loss | PSNR (dB) | SSIM | Warmup |
|---|---|---|---|---|
| 1 | 0.4258 | 6.61 | - | on |
| 10 | 0.3708 | 7.59 | - | on |
| 25 | 0.2046 | 12.38 | - | on |
| 50 | 0.1305 | 15.58 | - | on |
| 75 | 0.1161 | 16.07 | - | on |
| 100 | 0.1021 | 17.32 | 0.552 | on |
| 121 | 0.0911 | 18.72 | - | on |

**预计完成时间:** ~7-8 小时 (50 epochs × ~8 min/epoch)

训练完成后 checkpoint 将保存至:
`/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step*.pth.tar`

测试将在训练完成后自动执行。

## 4. P3：K=1 条件 AV Latent Diffusion

### 需求

- 基于冻结 AE，在 latent 空间训练条件 diffusion。
- 第一阶段只实现 K=1，先确保可用质量和视频条件有效性。
- 已知 latent 在训练和采样过程中必须被硬约束保护。

### 核心逻辑

- 训练变量：

```text
z_target = AE.encode(clean_mel)
z_context = AE.encode(corrupted_mel)
mask_z = downsample(missing_mask) -> [B, 1, 10, 50]
```

- diffusion U-Net 输入拼接：

```text
z_t            [B, 8, 10, 50]
z_context      [B, 8, 10, 50]
mask_z         [B, 1, 10, 50]
boundary_map_z [B, 2, 10, 50]
```

- 总输入通道为 `19`。
- 新增 `VideoEvidenceEncoder` 的 P3 最小版本，输出逐帧 token：

```python
{
    "rgb_tokens": Tensor[B, 50, D],
    "flow_tokens": Tensor[B, 50, D],
    "video_tokens": Tensor[B, 50, D],
}
```

- P3 暂不启用 evidence gate，仅通过 cross-attention 注入 `video_tokens`。
- masked noising：

```text
noisy_missing =
  sqrt(alpha_bar_t) * z_target
  + sqrt(1 - alpha_bar_t) * epsilon

z_t =
  mask_z * noisy_missing
  + (1 - mask_z) * z_context
```

- diffusion loss 只在缺失 latent 区域计算：

```text
L_diff =
  sum(mask_z * (epsilon - epsilon_pred)^2)
  / sum(mask_z)
```

- 每个采样 step 后重新 clamp 已知 latent：

```text
z_t = mask_z * z_t_generated + (1 - mask_z) * z_context
```

- 最终 Mel 再执行一次原分辨率 compose。
- P3 loss：

```text
L = L_diff + lambda_boundary * L_boundary + lambda_sync * L_sync
```

### 验证方法

- 单元测试覆盖 diffusion schedule、mask_z 下采样、masked loss、known-region clamp。
- 一个 batch forward/backward 无 shape 和显存错误。
- 采样每一步后检查已知 latent 不变。
- compose 后检查已知 Mel 区域误差严格为 0。
- 固定 seed 输出可复现。
- 对比 original video、no-video、wrong-video，确认关闭或替换视频后指标下降。
- 验证 K=1 missing Mel L1、PSNR、boundary 指标达到可用水平。

### 预期结果

- 生成 K=1 AV latent diffusion checkpoint。
- K=1 结果接近或超过现有 VIAI-AV 的同类指标。
- 证明模型确实使用视觉条件，而不是退化为 audio-only 修复。
- 在 K=1 稳定前，不进入 scorer/calibration。

### P3 验证结果 (2026-06-19)

#### 单元测试

| 测试文件 | 测试数量 | 结果 |
|---|---|---|
| `tests/test_p3_uq_av_diffusion.py` | 45 | ✅ 全部通过 |

**P0+P1+P2+P3 回归: 145/145 测试通过 (100%)**

#### 测试覆盖明细

| 测试类 | 测试数 | 覆盖内容 |
|---|---|---|
| `TestDiffusionSchedule` | 7 | beta shape/range, alpha_bar 递减, q_sample shape, t=0 低噪声, t=T 纯噪声, cosine schedule, DDIM timestep 到 t=0 |
| `TestMaskDownsampling` | 6 | mask_z shape [B,1,10,50], 二值性, known→0, full→1, boundary map shape, 确定性 |
| `TestMaskedNoising` | 4 | known region 严格保留, missing region 被噪声化, compose_known_region, masked diffusion loss |
| `TestKnownRegionClamp` | 2 | DDPM step 后 known 保留, DDIM step 后 known 保留 |
| `TestLatentDiffusionUNet` | 7 | input/output shape, without-video mode, 确定性, 不同 t 不同输出, sinusoidal embedding, TimeEmbedding, 无 NaN/Inf |
| `TestVideoEncoderP3` | 5 | rgb/flow/video tokens shape, 不同视频不同 tokens, 确定性, dummy encoder, 无 NaN/Inf |
| `TestEndToEndSmoke` | 9 | 1 training step (forward+backward), video encoder 梯度, boundary loss 可导, UQ schedule CLI 参数生效, no-video checkpoint 兼容, test forward (Mel pred shape), K=1 DDIM sampling (shape + null scorer/evidence), known region compose 误差严格 < 1e-5, 固定 seed 可复现 (allclose), 所有模块可 import |
| `TestDiffusionEdgeCases` | 3 | 单步 schedule, deterministic q_sample (same noise), different noise → different sample |
| `TestVideoConditions` | 2 | `--uq_no_video` flag 使用 VideoConditionDummy, loader 支持 no_video/wrong_video 条件 |

#### 验证项逐项确认

| # | 验证项 | 方法 | 测试 | 结果 |
|---|---|---|---|---|
| 1 | diffusion schedule beta/alpha 形状与单调性 | `linear_beta_schedule(1000)` 检查 shape 与递增性; `alpha_bar` 检查非增递减 | `test_beta_shape_and_range`, `test_alpha_bar_decreasing` | ✅ beta∈(0,1) 递增, alpha_bar 非增 |
| 2 | q_sample forward 扩散 shape | 随机 z_0 + 随机 t → z_t shape 不变 | `test_q_sample_shape` | ✅ [B,8,10,50] |
| 3 | mask_z 下采样 shape + 二值 | `downsample_mask_2d` 输出 [B,1,10,50] 且值 ∈{0,1} | `test_downsample_shape`, `test_downsample_binary` | ✅ 严格二值 |
| 4 | 已知 latent 在 masked_q_sample 后严格保留 | `(z_t - z_context) * (1-mask_z)` 最大值 | `test_known_region_preserved` | ✅ 严格 = 0 |
| 5 | DDPM step 后已知 latent 不变 | `compute_previous_z` with clamp → known region error=0 | `test_ddpm_step_clamp` | ✅ 严格 = 0 |
| 6 | DDIM step 后已知 latent 不变 | `ddim_step` with clamp → known region error=0 | `test_ddim_step_clamp` | ✅ 严格 = 0 |
| 7 | 最终 compose 后已知 Mel 区域误差严格 0 | `compose_known_region(mel_pred, mel_corrupted, mask)` 检查 | `test_compose_known_region_is_strict` | ✅ max error < 1e-5 |
| 8 | diffusion loss 仅在 missing 区域计算 | `compute_diffusion_loss(ones, zeros, partial_mask)` | `test_diffusion_loss_only_in_missing_region` | ✅ loss 精确按 mask 计算 |
| 9 | U-Net input/output shape 严格匹配 | [B,19,10,50] → [B,8,10,50]（默认 latent_dim=8） | `test_unet_input_output_shape` | ✅ 所有 shape 匹配 |
| 10 | U-Net 无 NaN/Inf | 随机输入 forward，检查 NaN/Inf | `test_unet_no_nan_inf` | ✅ 干净 |
| 11 | U-Net 确定性 (eval 模式) | 相同输入 forward 两次 | `test_unet_deterministic` | ✅ 完全一致 |
| 12 | U-Net 不同 timestep 不同输出 | t=0 vs t=500 | `test_unet_different_t_gives_different_output` | ✅ 输出不同 |
| 13 | Without-video 模式可用 | video_tokens=None → forward 正常 | `test_unet_without_video` | ✅ shape 正确 |
| 14 | Video encoder shape 合约 | [B,50,3,64,64] + [B,50,2,64,64] → rgb/flow/video tokens [B,50,D] | `test_output_shapes` | ✅ 精确匹配 |
| 15 | Video encoder 确定性 | 相同输入 → 相同输出 | `test_deterministic` | ✅ 完全一致 |
| 16 | Dummy encoder 返回零 tokens | `VideoConditionDummy` → all zeros | `test_dummy_encoder` | ✅ 全零, shape 正确 |
| 17 | 1 步训练 forward/backward 可完成 | `optimize_parameters(0)` 后 loss > 0 且有限 | `test_one_training_step` | ✅ loss_diff > 0, 无 NaN/Inf |
| 18 | Test forward 产生 Mel pred | `model.test()` → `model.mel_pred` shape [B,1,80,200] | `test_test_forward` | ✅ shape 正确 |
| 19 | K=1 DDIM sampling shape | `model.sample(num_candidates=1)` → candidate_mels [B,1,1,80,200] | `test_sample_k1` | ✅ 所有 field 匹配, scorer/evidence=null |
| 20 | 固定 seed 可复现 | 两次 `sample(seed=42)` → allclose | `test_fixed_seed_reproducibility` | ✅ allclose(atol=1e-6) |
| 21 | no-video flag 使用 dummy encoder | `uq_no_video=True` → VideoConditionDummy, use_video=False | `test_no_video_flag` | ✅ dummy 激活 |
| 22 | 视频分支参与训练 | 1 step backward 后 video encoder grad sum > 0 | `test_training_step_backprops_into_video_and_boundary` | ✅ 有非零梯度 |
| 23 | boundary loss 参与反传 | `loss_boundary.requires_grad` 与 `loss_total.requires_grad` 检查 | `test_training_step_backprops_into_video_and_boundary` | ✅ 可导 |
| 24 | DDIM 采样走到 t=0 | `get_ddim_timesteps(50)` → len-1=50 且最后一个 timestep=0 | `test_ddim_timesteps_include_zero_and_requested_transitions` | ✅ 50 次转移 |
| 25 | UQ CLI schedule 参数生效 | `uq_diffusion_timesteps/uq_beta_start/uq_beta_end` 覆盖旧 diff 字段 | `test_uq_cli_schedule_params_take_precedence` | ✅ 优先生效 |
| 26 | no-video 可加载 AV checkpoint | real-video checkpoint → `uq_no_video=True` 模型 load | `test_real_video_checkpoint_loads_in_no_video_mode` | ✅ 跳过 video encoder 权重 |
| 27 | loader 视觉退化条件可用 | `video_conditions=("no_video",)` 与 `("wrong_video",)` | `test_loader_video_degradation_conditions` | ✅ condition 正确 |

#### 新增文件

| 文件 | 用途 |
|---|---|
| `networks/uq/diffusion_schedule.py` | Diffusion noise schedule: linear/cosine beta, DDPM q_sample, masked noising, DDIM step, known-region clamp, masked loss, mask/boundary downsampling |
| `networks/uq/latent_diffusion_unet.py` | Conditional 2D U-Net: residual blocks, time FiLM, cross-attention for video tokens, interpolation-based upsampling |
| `networks/uq/video_evidence_encoder.py` | P3 minimal video encoder: PerFrameCNN for RGB + flow → video_tokens [B,50,D]; VideoConditionDummy for audio-only |
| `Models/UQ_AV_Diffusion.py` | Training wrapper: frozen AE, diffusion schedule, video encoder, U-Net; optimize_parameters, test, sample (K=1 DDIM), checkpoint save/load |
| `train_uq_av.py` | Training entry point: reads UQ-AV data, loads frozen AE, runs training loop with TensorBoard logging |
| `test_uq_av.py` | Testing entry point: K=1 DDIM evaluation, metrics (PSNR/SSIM), Mel images, optional Griffin-Lim wav export, summary.json/samples.jsonl/metrics.csv |
| `tests/test_p3_uq_av_diffusion.py` | 40 项单元测试覆盖 diffusion schedule、mask 下采样、masked noising、known-region clamp、U-Net shape、video encoder、端到端 training/sampling、可复现性、no-video 模式 |

#### 架构确认

```
Data: UQAVDataset → batch dict {mel_target, mel_corrupted, missing_mask, boundary_map, video, flow, ...}

Training:
  1. Frozen AE .encode(mel_target) → z_target  [B, 8, 10, 50]
     Frozen AE .encode(mel_corrupted) → z_context [B, 8, 10, 50]
  2. downsample_mask_2d(missing_mask) → mask_z   [B, 1, 10, 50]
     downsample_boundary_map(boundary_map) → bdy_z [B, 2, 10, 50]
  3. VideoEvidenceEncoderP3(video, flow) → video_tokens [B, 50, D]
  4. t ~ Uniform(0, T-1), ε ~ N(0,I)
  5. masked_q_sample(z_target, z_context, mask_z, t, ε) → z_t
  6. UNet(cat[z_t, z_context, mask_z, bdy_z], t, video_tokens) → ε_pred
  7. L_diff = mean(mask_z * (ε - ε_pred)²)
  8. L = L_diff + λ_bdy * L_boundary + λ_sync * L_sync

Inference (K=1 DDIM, 50 steps):
  1. z_T = mask_z * N(0,I) + (1-mask_z) * z_context
  2. for each DDIM step:
       ε_pred = UNet(cat[z_t, z_context, mask_z, bdy_z], t, video_tokens)
       z_{t-1} = DDIM_step(z_t, ε_pred, t, t_next)
       clamp: z_{t-1} = mask_z * z_{t-1} + (1-mask_z) * z_context
  3. Frozen AE .decode(z_0) → mel_pred [B, 1, 80, 200]
  4. mel_completed = mask * mel_pred + (1-mask) * mel_corrupted

U-Net: input [B, 2*latent_dim+3, 10, 50] → 4 encoder levels (strides 1, (2,1), (1,2), (2,2))
       → bottleneck (time FiLM + cross-attn video tokens)
       → 3 decoder levels (interpolation upsample + skip concat) → output [B, latent_dim, 10, 50]
```

#### 命令行

```bash
# ============================================================
# P3 验证命令
# ============================================================

# 1. 运行 P3 单元测试 (45 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_p3_uq_av_diffusion.py \
  -v --tb=short

# 2. 运行 P0+P1+P2+P3 回归测试 (145 tests)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_baseline_protocol.py \
  tests/test_baseline_runner.py \
  tests/test_baseline_evaluation.py \
  tests/test_mask_sampler.py \
  tests/test_uq_av_data.py \
  tests/test_visual_degradation.py \
  tests/test_viai_loader_regression.py \
  tests/test_p1_integration.py \
  tests/test_mel_autoencoder.py \
  tests/test_p3_uq_av_diffusion.py \
  -v --tb=short

# 3. 验证 CLI 入口点
python main.py train-uq-av -- --help
python main.py test-uq-av -- --help

# 4. 导入验证
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -c "
from networks.uq.diffusion_schedule import (
    DiffusionSchedule, linear_beta_schedule, cosine_beta_schedule,
    downsample_mask_2d, downsample_boundary_map,
    compose_known_region, compute_diffusion_loss,
)
print('diffusion_schedule — OK')
from networks.uq.latent_diffusion_unet import (
    LatentDiffusionUNet, sinusoidal_embedding, TimeEmbedding,
)
print('latent_diffusion_unet — OK')
from networks.uq.video_evidence_encoder import (
    VideoEvidenceEncoderP3, VideoConditionDummy,
)
print('video_evidence_encoder — OK')
from networks.uq.mel_autoencoder import MelAutoencoder
print('mel_autoencoder — OK')
print('All P3 modules import successfully')
"

# ============================================================
# P3 训练 & 测试 (需要 P2 Mel AE checkpoint)
# ============================================================

# 5. 1-step sanity check (确保数据链路 + 模型可训练)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-smoke \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 1 --num_workers 0 --max_train_steps 1 \
  --display_id 0 --print_freq 1

# 6. 正式训练 (RTX 4090, ~300 samples, ~2-3 days for 100 epochs)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1 \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 8 --nepochs 100 \
  --uq_lambda_boundary 0.1 \
  --uq_unet_base_channels 128 \
  --checkpoint_interval 1000 --print_freq 100 \
  --display_id 0 \
  --log_event_path checkpoints/uq_av_k1/events

# 7. 测试 K=1 (需要训练好的 UQ-AV checkpoint)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-test \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_checkpoint_step000040144.pth.tar \
  --uq_unet_base_channels 128 \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir checkpoints/uq_av_k1_test_results \
  --use_vocoder --vocoder_n_iter 32 --vocoder_max_samples 20


# 8. No-video ablation (验证模型确实使用视觉条件)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-no-video \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_checkpoint_step*.pth.tar \
  --uq_video_degradation no_video \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir checkpoints/uq_av_k1_test_results_no_video

# 9. Wrong-video ablation (替换为同乐器/其他视频的视觉条件)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-wrong-video \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_checkpoint_step*.pth.tar \
  --uq_video_degradation wrong_video \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir checkpoints/uq_av_k1_test_results_wrong_video

# 10. Dummy-token ablation (跳过视频 encoder 权重, 直接喂零 token)
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --uq_no_video \
  --name UQ-AV-K1-dummy-token \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_checkpoint_step*.pth.tar \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir checkpoints/uq_av_k1_test_results_dummy_token
```

## 5. P4：Multi-Hypothesis Sampling

### 需求

- 将推理接口扩展为 K 候选采样。
- 支持 `K=1, 4, 8, 16`，主结果默认 `K=8`。
- 导出多候选 wav、Mel 图、metadata 和候选级指标。

### 核心逻辑

- 统一推理接口：

```python
result = model.sample(batch, num_candidates=K, seed=seed)
```

- 返回：

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

- 第一版使用 DDIM：
  - `train_timesteps=1000`
  - `inference_steps=50`
- 基础多候选指标：
  - `best_of_k_missing_l1`
  - `mean_k_missing_l1`
  - `best_of_k_boundary_error`
  - `mean_k_boundary_error`
  - `pairwise_latent_diversity`
  - `pairwise_audio_embedding_diversity`
- 禁止只报告 Best-of-K，必须同时报告 Mean-K。
- 每个样本导出目录包含 `metadata.json`、输入、target、各候选 wav 和 Mel 图。

### 验证方法

- `K=1` 与单候选入口输出一致。
- K 个候选的已知区域完全相同。
- 不同 seed 产生非零但不过度的缺失区差异。
- Best-of-K 随 K 增加改善。
- Mean-K 不出现明显崩溃。
- 候选差异主要集中在缺失区域。
- 导出 metadata 包含 mask、video_condition、num_candidates、metrics。

### 预期结果

- `test-uq-av` 可以导出完整 K 候选结果。
- 得到 Best/Mean-K 与 diversity 报告。
- MVP 中 Best-of-8 应优于 K=1，且 Mean-K 质量不明显崩溃。

## 6. P5：Visual Evidence 与 Evidence-Aware Fusion

### 需求

- 扩展视觉编码器，输出视觉证据、运动强度、sync 信号和视觉不确定性。
- 使用 evidence 控制 audio/video 融合和采样温度。
- 支持视觉退化下的不确定性趋势分析。

### 核心逻辑

- `VideoEvidenceEncoder` 输出：

```python
{
    "video_tokens": Tensor[B, 50, D],
    "motion_strength": Tensor[B, 50],
    "sync_logits": Tensor[B, 50],
    "evidence": Tensor[B, 50],
    "visual_logvar": Tensor[B, 50],
}
```

- `AudioConditionEncoder` 输入 corrupted Mel、missing mask、左右 boundary map，输出 `[B, 50, D]` audio tokens。
- 融合方式：

```text
g(t) = sigmoid(MLP([audio_token(t), video_token(t), evidence(t), visual_logvar(t)]))
fused(t) = audio_token(t) + g(t) * video_projection(video_token(t))
```

- MVP evidence 训练信号：
  - optical flow magnitude 和局部 motion peak；
  - 正确对齐与 temporal shift/wrong video 的 sync 分类；
  - 已知视觉退化类型提供 reliability target。
- 初始 reliability target：
  - `original=1.0`
  - `blur=0.7`
  - `frame_drop=0.6`
  - `occlusion=0.4`
  - `temporal_shift=0.1`
  - `wrong_video=0.0`
  - `no_video=0.0`
- evidence-conditioned sampling temperature：

```text
e_bar = mean(evidence inside gap)
temperature = T_min + (T_max - T_min) * (1 - e_bar)
```

- 建议初值：
  - `T_min=0.7`
  - `T_max=1.0`
- 训练时可用 `K_train=2` 做 evidence-diversity 约束：

```text
D_target = alpha * (1 - mean_gap_evidence)
L_div = SmoothL1(D_sample, D_target)
```

### 验证方法

- 同一 audio gap 下，original video 的平均 evidence 高于 wrong/no-video。
- temporal shift 后 sync confidence 明显下降。
- evidence 与 gate 正相关。
- evidence 高时 pairwise diversity 较低。
- evidence 低时 diversity 上升，但 Mean-K quality 不明显崩溃。
- no-video 时模型仍可依赖 audio context 完成修复。
- TensorBoard 或导出文件记录 evidence、gate、motion_strength、sync_confidence 曲线。

### 预期结果

- original video 的 diversity 低于 wrong/no-video。
- 视觉退化越严重，uncertainty 趋势越高。
- boundary 指标不弱于 VIAI-AV。
- 如果趋势不成立，优先排查 diffusion 是否使用视频、evidence 是否只学到退化标签、AE 是否丢失 onset 细节。

## 7. P6：Candidate Scorer 与 Calibration

### 需求

- 为每个候选输出置信度 `pi_k`。
- 输出样本级不确定性 `u`。
- 在 validation set 上做后处理校准，不使用 test set 拟合校准参数。

### 核心逻辑

- candidate scorer 输入候选级特征：
  - candidate audio embedding；
  - boundary consistency feature；
  - AV sync score；
  - audio-context compatibility；
  - Mel realism feature；
  - mean visual evidence。
- 输出：

```text
score_logits: [B, K]
pi = softmax(score_logits)
```

- 用 ground truth 构造训练期候选风险：

```text
r_k =
  w_recon * missing_embedding_distance
  + w_boundary * boundary_error
  + w_sync * (1 - av_sync)
```

- 转为软排序目标：

```text
q_k = softmax(-r_k / tau)
L_score = KL(q || pi)
```

- uncertainty 由小 head 学习组合：

```text
u = sigmoid(UncertaintyHead(context, evidence, score_stats, diversity))
```

- calibration target：

```text
error_target = min_k r_k
```

- `error_target`、`q_k` 和候选指标应 `detach()`，避免生成器投机优化监督目标。
- 训练完成后在 validation set 上拟合 temperature 或 isotonic calibration。

### 验证方法

- top-1 scorer 候选优于随机候选和固定 candidate 0。
- top-1 scorer 接近 oracle best-of-K，但推理不访问 oracle 信息。
- uncertainty 与实际 error 显著正相关。
- risk-coverage 曲线显示优先保留低 uncertainty 样本时平均错误下降。
- ECE 优于无 calibration 版本。
- 视觉退化越严重，平均 uncertainty 越高。

### 预期结果

- `test-uq-av` 输出 `candidate_scores`、`pi_k`、`uncertainty` 和 calibration 指标。
- scorer top-1 可作为默认导出候选。
- uncertainty 可用于筛选高风险样本和分析视觉证据失效场景。

## 8. P7：完整评估、消融与导出

### 需求

- 建立完整实验矩阵，评估主模型、消融、视觉条件和 gap 长度。
- 所有结果按相同 sample/mask/video-condition manifest 比较。
- 输出可复聚合的 summary、samples、CSV、risk-coverage、calibration bins 和候选目录。

### 核心逻辑

- 主模型：
  - VIAI-A
  - VIAI-AV
  - GACELA 或等价 audio-only latent-variable baseline
  - Audio-only Diffusion K=1
  - Audio-only Diffusion K=8
  - AV Diffusion K=1
  - AV Diffusion K=8, no evidence
  - UQ-AVInpaint K=8
- 消融：
  - w/o boundary loss
  - w/o sync loss
  - w/o visual evidence gate
  - w/o visual corruption training
  - w/o diversity target
  - w/o candidate scorer
  - w/o calibration loss
  - fixed sampling temperature
- 视觉条件：
  - original、blur、occlusion、frame_drop、temporal_shift、wrong_video、no_video
- Gap：
  - 0.4s / 20 frames
  - 0.8s / 40 frames
  - 1.0s / 50 frames
  - 1.2s / 60 frames
  - 1.6s / 80 frames
  - 2.0s / 100 frames
- 指标分组：
  - gap length
  - mask type
  - instrument
  - video condition
- Vocoder 设置：
  - same vocoder：沿用 VIAI/WaveNet，公平比较 inpainting 能力；
  - strong vocoder：可选 HiFi-GAN/BigVGAN，用于展示最佳听感；
  - 所有主表必须标明 vocoder，不能混合比较。

### 验证方法

- 指标：
  - Mel L1 full/missing
  - PSNR full/missing
  - SSIM
  - log-spectral distance
  - FAD
  - PANNs embedding distance
  - CLAP audio-audio similarity（可选）
  - boundary_mel_l1
  - boundary_delta_l1
  - boundary_energy_jump
  - global_av_sync
  - onset_motion_alignment
  - temporal_offset_error
  - best_of_k_error
  - mean_k_error
  - top1_scorer_error
  - oracle_gap
  - pairwise_diversity
  - Pearson/Spearman uncertainty-error correlation
  - ECE
  - Brier score
  - AUROC for high-error detection
  - risk-coverage curve
  - AURC
- 报告均值和 bootstrap 置信区间。
- 2 秒 gap 单独报告，不能与短 gap 平均后隐藏难度差异。
- 检查每个结果可追溯到 config、checkpoint、manifest、seed 和代码版本。

### 预期结果

- `test-uq-av` 输出目录：

```text
<results_dir>/
  summary.json
  samples.jsonl
  metrics.csv
  metrics_by_gap.csv
  metrics_by_video_condition.csv
  risk_coverage.csv
  calibration_bins.csv
  perceptual_metrics.csv
  candidates/
```

- 形成主表、消融表、视觉退化趋势图、risk-coverage 曲线、calibration bins 和 demo 样本。
- 形成 same-vocoder 公平对比表和 strong-vocoder 听感展示表。
- 验证“证据越弱，不确定性越高”的趋势。
- 最终系统满足可训练、可比较、可复现的完成定义。

## 9. P8：感知质量、Vocoder 对照与人类主观实验

### 需求

- 补齐自动指标无法覆盖的听感自然度、同步主观感受和多候选合理性评估。
- 明确 same-vocoder 与 strong-vocoder 两套报告方式，避免把 vocoder 提升误认为 inpainting 提升。
- 将主观实验结果与自动指标、uncertainty 和 visual evidence 趋势对应起来。
- 主观实验只在 P0-P7 自动评估稳定后执行，不参与模型选择和 calibration 拟合。

### 核心逻辑

- 感知质量自动评估：
  - FAD：衡量生成音频分布与真实音频分布距离；
  - PANNs embedding distance：衡量感知语义和音色距离；
  - CLAP audio-audio similarity：可选，用于补充跨模型感知相似度；
  - MOS 或偏好实验：作为最终论文听感证据。
- Vocoder 对照：
  - same-vocoder setting：VIAI-A、VIAI-AV、Diffusion 和 UQ-AVInpaint 使用同一 vocoder；
  - strong-vocoder setting：仅用于展示最佳 demo 和上限听感；
  - 每个导出 wav 的 metadata 必须记录 vocoder 名称、checkpoint 和采样参数。
- Human Study 1：单候选质量偏好。
  - 比较 VIAI-AV、AV Diffusion K=1、UQ-AVInpaint scorer top-1；
  - 问题包括自然度、边界平滑度、与视频同步程度。
- Human Study 2：多候选合理性。
  - 给定同一个视频和 K 个候选；
  - 标注 plausible candidate ratio、best candidate preference、diversity 是否为有意义差异。
- Human Study 3：不确定性感知。
  - 对比 original video 与 blur/occlusion/temporal_shift/wrong_video/no_video；
  - 检查人类认为“更难确定真实声音”的样本是否也有更高模型 uncertainty。

### 验证方法

- 主观实验样本必须来自固定 manifest，并覆盖不同 gap、mask type、instrument 和 video condition。
- 参与者不能看到模型名称、candidate score、uncertainty 或 oracle 信息。
- 每个 pairwise preference 至少包含随机左右顺序，避免展示顺序偏置。
- 汇总：
  - preference rate；
  - plausible candidate ratio；
  - mean opinion score；
  - human uncertainty agreement；
  - 与模型 uncertainty 的 Pearson/Spearman correlation。
- 报告 bootstrap 置信区间，并保存匿名原始打分表。

### 预期结果

- UQ-AVInpaint scorer top-1 在自然度、边界平滑度和 AV sync 偏好上优于确定性基线。
- 多候选不是随机噪声，低视觉证据样本中合理候选的差异被人类认为有意义。
- 人类不确定性感知与模型 uncertainty 呈正相关。
- 输出：
  - `human_study_samples.jsonl`
  - `human_study_responses.csv`
  - `human_study_summary.json`
  - `vocoder_comparison.csv`

## 10. 第一批代码任务顺序

### 需求

- 先完成数据和 AE 的工程基础，再进入 diffusion。
- 前 8 项完成前不开始训练 diffusion。
- P3 的 K=1 未稳定前不实现 calibration。

### 核心逻辑

1. 新建或完善 `tests/`，为现有 compose 和新 mask 语义建立测试。
2. 实现 `MaskSpec`、逐样本 mask 和 boundary map。
3. 实现固定 val/test mask manifest。
4. 新建或完善 `uq_av_loader.py`，输出字典并通过 shape 测试。
5. 注册并验证 `prepare-uq-metadata`。
6. 实现 Mel autoencoder 与 shape/reconstruction 测试。
7. 注册 `train-mel-ae`、`test-mel-ae`。
8. 完成 AE smoke test 并检查重建上限。
9. 实现 diffusion schedule、masked noising 和 known-region clamp。
10. 实现最小 conditional U-Net，先用 no-video/audio-only 条件跑通。
11. 接入逐帧 video token，形成 AV Diffusion K=1。
12. 扩展 K-sampling、指标和候选导出。
13. 最后实现 evidence、scorer 和 calibration。

### 验证方法

- 每个任务完成后运行对应单元测试。
- 每个新入口注册后运行 `python main.py <action> -- --help` 或最小 smoke 命令。
- 每个阶段至少保存一次小规模实验结果，避免只依赖训练日志判断。
- 遇到指标异常时先回退到更简单条件，如 audio-only、K=1、无 sync loss。

### 预期结果

- 工程推进顺序清晰，可避免早期过度耦合。
- 每个子系统都有独立验证入口。
- 后续开发可以按阶段持续提交，而不是一次性大改。

## 11. 风险与处理策略

### 需求

- 提前定义关键风险和停止条件，避免在错误方向上继续堆复杂度。

### 核心逻辑

- 数据量不足：
  - 先统计 processed clip 和 instrument 数量；
  - 保证 video-level split；
  - 必要时 audio-only diffusion 预训练，再加入视频条件。
- AE 成为瓶颈：
  - AE 重建指标作为进入 P3 的硬门槛；
  - 增加 latent channel 或降低压缩率；
  - 优先保留时间分辨率。
- 模型忽略视频：
  - 加入 no-video、wrong-video、temporal-shift 对照；
  - 记录 cross-attention/gate；
  - 比较 original 与 shuffled video。
- 多样性只是噪声：
  - 在 embedding 空间衡量；
  - 同时约束 boundary、sync 和 Mean-K quality；
  - 使用人工听感或盲测区分合理变化与噪声。
- uncertainty 只识别人工退化：
  - evidence 输入包含 motion、sync 和内容特征；
  - 在未见退化强度上测试；
  - calibration target 使用实际生成错误。
- 计算量过大：
  - AE 冻结并可选预计算 latent cache；
  - P3 训练使用单样本 diffusion loss；
  - P5/P6 只在小比例 batch 上做 `K_train=2`；
  - 推理用 DDIM 50 steps。

### 验证方法

- 每个风险都设置对照实验，而不是仅观察单个总指标。
- 将异常样本导出到 candidates 目录，人工检查 Mel 图和 wav。
- 每次模型复杂度增加后，比较是否真正改善目标指标。

### 预期结果

- 能快速定位问题属于数据、AE、diffusion、视频条件、采样、scorer 还是 calibration。
- 不会把 vocoder、Best-of-K 或人工退化标签造成的表面提升误认为 inpainting 模型提升。

## 12. 完成定义

### 需求

- UQ-AVInpaint 被视为完成时，必须同时满足工程完整性、实验可复现性和研究指标有效性。

### 核心逻辑

- 原 VIAI-A/VIAI-AV 入口仍可运行。
- 新数据协议可复现并有单元测试。
- AE、diffusion、evidence、scorer 可分阶段训练和恢复。
- 统一接口输出 K 候选、置信度和 uncertainty。
- 已知区域在最终结果中严格不变。
- 结果包含 Best-of-K、Mean-K、boundary、sync、diversity、perceptual quality 和 calibration。
- 视觉退化实验呈现 evidence 越弱、uncertainty 越高的趋势。
- scorer top-1 优于随机候选。
- uncertainty-error correlation、ECE 和 risk-coverage 可计算。
- same-vocoder 和 strong-vocoder 结果分开报告。
- 主观实验支持自然度、同步、多候选合理性和不确定性感知主张。
- 所有主结果可追溯到配置、checkpoint、manifest、seed 和代码版本。

### 验证方法

- 完整运行 P0-P8 评估矩阵。
- 对主实验和消融结果执行复聚合，确认 `samples.jsonl` 能重建 summary。
- 抽查候选导出，确认 metadata、音频、Mel 图和指标一一对应。
- 抽查主观实验样本，确认不泄露模型名称、score、uncertainty 或 oracle 信息。
- 在固定 checkpoint 和 manifest 下重复测试，确认结果可复现。

### 预期结果

- `thoUght.md` 中的研究设想转化为可训练、可比较、可复现的系统。
- 项目具备论文实验、消融分析、主观评估、demo 导出和后续模型迭代的基础。
