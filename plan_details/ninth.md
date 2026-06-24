# 第 9 步：EC-VIAI-AV 测试协议扩展

## Summary
- 在现有 Stage8 基础上扩展测试协议，不改训练主链路。
- 当前代码已具备 `mel_candidates`、`candidate_pi`、`candidate_top1_index`、`uncertainty_score`、`evidence_score` 等张量；`save_candidates` 和 `video_perturbation` 目前主要是占位，需要补实际行为。
- 已确认当前检查通过：`uv run python -m py_compile ...` 和 `uv run pytest -q tests/test_multi_candidate_losses.py tests/test_evidence_fusion_gate.py`，19 个测试通过。

## Public API / Interfaces
- `base_options.py` 保留并收紧 `--video_perturbation`，支持：
  `none, blur, flow_zero, frame_drop, temporal_shift, wrong_video, no_video`
- 新增测试扰动参数：
  `--video_blur_kernel 9`、`--video_frame_drop_stride 2`、`--video_temporal_shift_frames 6`、`--calibration_bins 10`。
- `utils/viai_a_metrics.py` 新增多候选评估函数：
  `compute_multi_candidate_metrics(...)`、`compute_boundary_delta_error(...)`、`compute_risk_coverage_curve(...)`、`compute_calibration_bins(...)`。
- `test_viai_av.py` 新增输出字段：
  `evidence_mean/min/max`、`candidate_pairwise_mel_l1`、`boundary_delta_error_top1/best/mean`、`risk_coverage_path`、`calibration_bins_path`、`per_sample_metrics_path`、`candidate_image_dir`、`candidate_vocoder_dir`、扰动参数字段。
- JSON 文件名包含扰动名：`<name>_stepXXXXXXXXX_perturb-<mode>_test.json`；summary CSV 去重 key 改为 `(checkpoint_step, video_perturbation, test_num_candidates)`，避免不同扰动互相覆盖。

## Implementation Changes
- 多候选指标统一从 `model.mel_candidates`、`model.mel_completed_candidates`、`model.candidate_top1_index`、`model.candidate_pi`、`model.uncertainty_score` 计算；K=1 时退化为 top1/best/mean/candidate0 相同、pairwise 为 0。
- 候选质量指标按 sample 加权汇总，不再用 batch 平均累加；loss 字段继续保持现有 batch 平均语义。
- `boundary_delta_error` 使用 completed mel 在 missing span 左右边界的一阶时间差分与 target 对齐；边界越界时跳过该侧。
- 测试扰动在 `model.set_inputs(data)` 后、`model.test(...)` 前应用，保证 evidence、gate、retrieval 都基于扰动后视频：
  `blur` 对 RGB 帧做均值/高斯式模糊；`flow_zero` 清零 flow；`frame_drop` 每隔 stride 保留一帧并复制到被 drop 帧、对应 flow 清零；`temporal_shift` 沿视觉时间轴平移 6 帧，空位用边界帧/零 flow 填充；`no_video` 清零 RGB 和 flow。
- `wrong_video` 从同一 test split 的第 0 列样本目录中按 `+1` offset 确定性取另一个样本，用 `sample_data_new(..., train=False)` 读取 video/flow；支持 `batch_size=1`。若 split 少于 2 个有效 AV 样本，直接报错。
- `--save_candidates` 时保存候选图到：
  `<results_dir>/mel-candidates/stepXXXXXXXXX/perturb-<mode>/candidate_00/*.png`
  并保留现有 top1 图目录。
- 同时启用 `--save_candidates --use_vocoder` 时，保存每个候选 wav 到：
  `<results_dir>/wav-candidates/stepXXXXXXXXX/perturb-<mode>/candidate_00/*.wav`；
  `--vocoder_max_samples` 表示保存前 N 个样本的所有 K 个候选。
- 每次测试写出 per-sample CSV，包含 sample path、top1 index、uncertainty、evidence、gate、sigma scale、candidate pi、每个 candidate missing L1、top1/best/mean error、pairwise、boundary delta。
- risk-coverage curve 按 uncertainty 从低到高排序，输出 20 个 coverage 点的 retained count、threshold、mean top1 error。
- calibration bins 按 uncertainty 等宽分 10 桶，输出 count、avg uncertainty、avg top1 error、avg best error、avg oracle gain、avg evidence、avg pairwise。

## Test Plan
- 静态检查：
  `uv run python -m py_compile utils/viai_a_metrics.py test_viai_av.py base_options.py Models/VIAI_AV_inpainting.py`
- 单元测试：
  扩展 `tests/test_multi_candidate_losses.py` 或新增 `tests/test_stage9_eval_protocol.py`，覆盖多候选指标、boundary delta、risk-coverage、calibration bins、K=1 退化、扰动 shape 与效果、CSV 去重 key。
- 云端 smoke：
  对 `none blur flow_zero frame_drop temporal_shift wrong_video no_video` 各跑一个小 test split，确认 JSON/CSV/per-sample/risk/calibration 文件存在且字段齐全。
- 人工验收：
  检查 candidate mel 图中 known 区域与输入一致，missing 区域候选有差异；比较 `none` 与 `wrong_video/no_video` 的 evidence、retrieval、uncertainty、pairwise 是否出现合理变化。

## Assumptions
- 第 9 步不改变训练损失、checkpoint 结构或 Stage8 scorer/head 行为。
- `wrong_video` 采用同 split 确定性错配，不跨 split 采样，保证复现实验可追踪。
- risk-coverage 默认认为 uncertainty 越低越可信，risk 使用 top1 missing L1。
- 本地已有未提交改动会被保留，实施时只在相关文件上增量修改，不回滚已有 Stage8 工作。
