# 第2步：EC-VIAI-AV 配置开关接入

**Summary**
- 只接入命令行、实验命名、运行配置打印和测试结果记录；不改 `VIAIAVModel` 前向、损失、数据加载或 checkpoint 逻辑。
- EC 功能仅由 `--enable_ec_viai_av` 显式开启；默认不带新参数时，VIAI-AV 路径保持现状。
- 若启用 EC 且未手动传 `--name`，默认实验名改为 `EC-VIAI-AV` 或 `EC-VIAI-AV-PatchGAN`，避免覆盖 baseline 输出。

**Key Changes**
- 在 `base_options.py` 新增参数：
  - 模式：`--enable_ec_viai_av`，默认 `False`
  - 候选：`--num_candidates` 默认 `1`
  - adapter：`--stochastic_adapter`、`--deterministic_adapter`，默认 `False`
  - 损失权重：`--lambda_min_k`、`--lambda_mean_k`、`--lambda_boundary`、`--lambda_diversity`、`--lambda_calib`，默认 `0.0`
  - evidence：`--enable_evidence_gate` 默认 `False`，`--evidence_source` 默认 `"none"`，`--sigma_min` 默认 `0.0`，`--sigma_max` 默认 `1.0`
  - 测试：`--test_num_candidates` 默认跟随 `--num_candidates`，`--save_candidates` 默认 `False`，`--video_perturbation` 默认 `"none"`
- 在解析阶段做轻量校验：
  - `num_candidates >= 1`
  - `test_num_candidates >= 1`
  - `sigma_min >= 0`
  - `sigma_max >= sigma_min`
  - `--stochastic_adapter` 与 `--deterministic_adapter` 不能同时开启。
- 在 `train_viai_av.py`：
  - 扩展 `configure_viai_av_defaults()`，EC 模式使用独立默认 `name` 和 `log_event_path`。
  - 扩展 `print_viai_av_run_config()`，打印所有 EC、loss、evidence、candidate/test 参数。
- 在 `test_viai_av.py`：
  - 扩展 `configure_viai_av_defaults()`，EC 模式使用独立默认 `results_dir`。
  - 新增测试运行配置打印，覆盖同一组 EC 参数。
  - 扩展 `RESULT_FIELDS`、`build_result_record()`、`coerce_csv_record()`，把 EC 参数写入 JSON/CSV，便于云端实验追踪。

**Test Plan**
- 在已安装 PyTorch 的云端环境运行：
  - `python main.py train-viai-av -- --help`
  - `python main.py test-viai-av -- --help`
  - 确认新增参数全部可见。
- Baseline 回归：
  - 使用原 VIAI-AV 或 VIAI-AV-PatchGAN 命令，不带任何 EC 参数。
  - 确认默认 `name`、`log_event_path`、`results_dir`、stage 字段和训练/测试行为保持原状。
- EC 配置 smoke：
  - `python main.py train-viai-av -- --enable_ec_viai_av --deterministic_adapter --num_candidates 1 ... --max_train_steps 1`
  - 确认配置打印中 EC 参数正确，默认实验名为 `EC-VIAI-AV` 或 `EC-VIAI-AV-PatchGAN`。
- 测试记录 smoke：
  - `python main.py test-viai-av -- --enable_ec_viai_av --deterministic_adapter --num_candidates 1 --test_num_candidates 1 ...`
  - 确认 JSON/CSV 包含新增 EC 字段。

**Concrete Commands**

运行前先按云端实际路径设置变量：

```bash
export DATA_ROOT=/root/shared-nvme/data
export VIAI_A_CKPT=checkpoints/VIAI-A_checkpoint_step000006800.pth.tar
export VIAI_AV_CKPT=checkpoints/viai-av_train/VIAI-AV-PatchGAN_checkpoint_step000019000.pth.tar
```

确认新增参数已经出现在 help 中：

```bash
python main.py train-viai-av -- --help | grep -E "enable_ec_viai_av|num_candidates|stochastic_adapter|deterministic_adapter|lambda_min_k|lambda_mean_k|lambda_boundary|lambda_diversity|lambda_calib|enable_evidence_gate|evidence_source|sigma_min|sigma_max|test_num_candidates|save_candidates|video_perturbation"
```

```bash
python main.py test-viai-av -- --help | grep -E "enable_ec_viai_av|num_candidates|stochastic_adapter|deterministic_adapter|lambda_min_k|lambda_mean_k|lambda_boundary|lambda_diversity|lambda_calib|enable_evidence_gate|evidence_source|sigma_min|sigma_max|test_num_candidates|save_candidates|video_perturbation"
```

Baseline VIAI-AV 训练 smoke，不带任何 EC 参数，用于确认默认路径不变：

```bash
python main.py train-viai-av -- \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a "$VIAI_A_CKPT" \
  --checkpoint_dir /tmp/viai_av_baseline_smoke \
  --log_event_path /tmp/viai_av_baseline_smoke/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 1 \
  --print_freq 1 \
  --display_id 0
```

Baseline VIAI-AV-PatchGAN 测试 smoke，不带任何 EC 参数，用于确认原结果目录和记录字段兼容：

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
  --results_dir /tmp/viai_av_patchgan_baseline_test_smoke
```

EC-VIAI-AV deterministic adapter 训练 smoke。当前第 2 步只接配置开关，所以该命令用于验证配置打印、默认实验名和参数解析：

```bash
python main.py train-viai-av -- \
  --enable_ec_viai_av \
  --deterministic_adapter \
  --num_candidates 1 \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a "$VIAI_A_CKPT" \
  --checkpoint_dir /tmp/ec_viai_av_smoke \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 1 \
  --print_freq 1 \
  --display_id 0
```

EC-VIAI-AV-PatchGAN 测试记录 smoke，用于确认 JSON/CSV 写出新增 EC 字段：

```bash
python main.py test-viai-av -- \
  --enable_ec_viai_av \
  --deterministic_adapter \
  --num_candidates 1 \
  --test_num_candidates 1 \
  --use_gan \
  --name EC-VIAI-AV-PatchGAN \
  --resume_path "$VIAI_AV_CKPT" \
  --data_root "$DATA_ROOT" \
  --test_split_name test_av_split.txt \
  --batch_size 1 \
  --num_workers 0 \
  --display_id 0 \
  --results_dir /tmp/ec_viai_av_patchgan_test_smoke
```

检查测试 JSON/CSV 中是否包含新增字段：

```bash
python - <<'PY'
import csv
import glob
import json
import os

results_dir = "/tmp/ec_viai_av_patchgan_test_smoke"
json_path = sorted(glob.glob(os.path.join(results_dir, "*_test.json")))[-1]
csv_path = sorted(glob.glob(os.path.join(results_dir, "*_test_summary.csv")))[-1]
keys = [
    "enable_ec_viai_av",
    "num_candidates",
    "test_num_candidates",
    "deterministic_adapter",
    "lambda_min_k",
    "enable_evidence_gate",
    "evidence_source",
    "sigma_min",
    "sigma_max",
    "save_candidates",
    "video_perturbation",
]
with open(json_path, "r", encoding="utf-8") as handle:
    record = json.load(handle)
print("json:", {key: record.get(key) for key in keys})
with open(csv_path, "r", encoding="utf-8", newline="") as handle:
    row = next(csv.DictReader(handle))
print("csv:", {key: row.get(key) for key in keys})
PY
```

**Assumptions**
- 第 2 步不实现 stochastic adapter、evidence gate、candidate 保存或视频扰动的实际模型/数据逻辑；这些参数先作为后续阶段的显式配置入口。
- `--video_perturbation` 和 `--evidence_source` 暂时作为开放字符串保存，不在第 2 步限制枚举，方便后续阶段扩展。
- 本地当前缺少 `torch`，所以 `main.py train-viai-av -- --help` 需要在云端依赖完整环境验证。
