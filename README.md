# EC-VIAI-AV

EC-VIAI-AV 是在 **Vision-Infused Deep Audio Inpainting (VIAI-AV)** 复现工程基础上扩展出的音视频联合音频修复框架。当前主线不再只是复现原始 VIAI-AV，而是研究一个更实际的问题：视频条件并不总是可靠，模型应当根据视觉证据决定何时信任视频。

本仓库当前支持：

- Original VIAI-AV / VIAI-A 复现入口。
- EC-VIAI-AV multi-candidate 音频修复。
- Evidence-Aware Fusion Gate 与 evidence-scaled sigma。
- Candidate Scorer 与 Uncertainty Head。
- 基于冻结 CLIP 的 target-specific semantic evidence。
- Semantic Perturbation Training：训练时显式加入 `wrong_video_cross_instrument`、`no_video`、`flow_zero`。
- 标准扰动测试、per-sample metrics、risk-coverage、calibration bins、TensorBoard 日志和 Mel 可视化。

原始 VIAI 项目：

- [Project](https://hangz-nju-cuhk.github.io/projects/AudioInpainting)
- [Paper](https://arxiv.org/abs/1910.10997)
- [Demo](https://www.youtube.com/watch?v=2C8s_YuRRxk)

## 1. 方法概览
![](./docs/img/pipeline.png)

EC-VIAI-AV 保留原始 VIAI-AV 的 Mel Encoder、RGB/Flow Video Encoder 和音视频融合解码器，并加入以下模块：

| 模块 | 作用 |
| --- | --- |
| Visual Evidence Estimator | 估计视频可靠性，包括 heuristic evidence 与 semantic evidence。 |
| Evidence-Aware Fusion Gate | 根据 evidence score 动态调整视频特征与 audio prior 的融合比例。 |
| Stochastic Adapter | 在 bottleneck 处生成多个候选，支持 multi-hypothesis prediction。 |
| Candidate Scorer | 测试时从多个候选中选择 Top-1 输出。 |
| Uncertainty Head | 输出样本级不确定性，用于风险排序和 calibration 分析。 |
| Semantic Perturbation Training | 训练时加入错配/缺失/退化视频，使 gate 和 uncertainty 真正响应低 evidence。 |

Semantic evidence 使用 target-specific 定义：

```text
semantic evidence = P(source instrument | current video frames)
```

这与普通视频自分类不同。对于源音频是 xylophone、错配视频是 accordion 的样本，普通自分类会给 accordion video 高分；本项目查询的是该 accordion video 对 xylophone 的概率，因此能够识别跨乐器错配。

## 2. 环境准备

推荐 Python 3.10。CUDA 版 PyTorch 建议按目标机器单独安装。

基础依赖：

```bash
python -m pip install --upgrade pip
python -m pip install imageio-ffmpeg librosa nnmnkwii opencv-contrib-python pillow scikit-image tensorboard tensorboardX tqdm "yt-dlp[default]"
```

云端 CUDA 环境可使用 headless OpenCV：

```bash
python -m pip install --upgrade pip
python -m pip install imageio-ffmpeg librosa nnmnkwii "numpy==1.22.4" opencv-contrib-python-headless pillow scikit-image tensorboard tensorboardX tqdm "yt-dlp[default]"
python -c "import torch, cv2; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
```

确认 TV-L1 optical flow 可用：

```bash
python -c "import cv2; print(hasattr(cv2, 'optflow') or hasattr(cv2, 'DualTVL1OpticalFlow_create'))"
```

如果要预计算 CLIP semantic evidence，需要安装 optional semantic 依赖：

```bash
uv sync --extra semantic
```

统一设置数据目录：

```bash
export DATA_ROOT=/root/shared-nvme/data
```

本地小规模调试可以用仓库内数据目录：

```bash
export DATA_ROOT=data
```

## 3. 数据准备

### 3.1 下载 MUSICES 视频

```bash
python main.py prepare-data -- download \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing
```

如果 YouTube 需要 cookies：

```bash
python main.py prepare-data -- download \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/absolute/path/to/youtube_cookies.txt
```

视频目录约定：

```text
$DATA_ROOT/raw_videos/<instrument>/<youtube_id>.mp4
```

### 3.2 生成 AV 样本

生成 4 秒 clip、RGB frames、optical flow、Mel/audio：

```bash
python main.py prepare-data -- process \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --skip-existing
```

只做链路 smoke test 时，可临时加 `--flow-method farneback` 提速。

生成 AV split：

```bash
python main.py split-data -- \
  --data-root "$DATA_ROOT" \
  --train-split-name train_av_split.txt \
  --val-split-name val_av_split.txt \
  --test-split-name test_av_split.txt

wc -l "$DATA_ROOT/train_av_split.txt" "$DATA_ROOT/val_av_split.txt" "$DATA_ROOT/test_av_split.txt"
```

AV 样本需包含：

```text
raw_audio.npy
mel.npy
image_crop/
flow_x_crop/
flow_y_crop/
```

不合格样本会被跳过，并记录到：

```text
$DATA_ROOT/viai_av_bad_samples.csv
$DATA_ROOT/musices_process_failures.csv
```

### 3.3 可选：生成 VIAI-A audio-only 样本

如果要复现实验中的 audio-only baseline：

```bash
python main.py prepare-viai-a -- \
  --json "$DATA_ROOT/MUSICES.json" \
  --data-root "$DATA_ROOT" \
  --processed-dir processed_viai_a \
  --skip-existing

python main.py split-data -- \
  --data-root "$DATA_ROOT" \
  --processed-dir processed_viai_a \
  --train-split-name train_viai_a_split.txt \
  --val-split-name val_viai_a_split.txt \
  --test-split-name test_viai_a_split.txt \
  --audio-only
```

## 4. Semantic Evidence 预计算

EC-VIAI-AV final model 使用离线 CLIP sidecar。JSONL 会保存每个视频对所有乐器 prompt 的概率分布，并在训练/测试时按 source instrument 读取 target-specific score。

先做 5 个样本 smoke test：

```bash
export SEM_DIR="$DATA_ROOT/semantic_evidence/clip_vit_b32"

python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --output-dir "$SEM_DIR/smoke" \
  --model ViT-B-32 \
  --pretrained openai \
  --frames-per-sample 8 \
  --limit 5

head -n 2 "$SEM_DIR/smoke/train_av_split.jsonl"
```

正式预计算 train / val / test：

```bash
python tools/precompute_semantic_evidence.py \
  --data-root "$DATA_ROOT" \
  --split-name train_av_split.txt \
  --split-name val_av_split.txt \
  --split-name test_av_split.txt \
  --output-dir "$SEM_DIR" \
  --model ViT-B-32 \
  --pretrained openai \
  --frames-per-sample 8

wc -l "$SEM_DIR/train_av_split.jsonl" "$SEM_DIR/val_av_split.jsonl" "$SEM_DIR/test_av_split.jsonl"
```

JSONL 关键字段：

```text
sample_dir
instrument
semantic_score
probs_by_instrument
top1_instrument
top1_prob
target_rank
frame_consistency
frame_top1_instruments
num_frames
```

诊断 wrong-video semantic 是否有效：

```bash
python tools/diagnose_semantic_evidence.py \
  --semantic-jsonl "$SEM_DIR/test_av_split.jsonl" \
  --sample-metrics-csv checkpoints/<eval_dir>/wrong_video_cross_instrument/sample-metrics/<samples.csv>
```

期望现象：`wrong_source_score_mean` 明显低于 `original_source_score_mean`。

## 5. 训练

### 5.1 Original VIAI-AV / PatchGAN baseline

原始 VIAI-AV 仍可使用 `train-viai-av` 训练。开启 PatchGAN 时传 `--use_gan`。

```bash
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000006800.pth.tar \
  --checkpoint_dir checkpoints/viai_av_patchgan_reference \
  --log_event_path checkpoints/viai_av_patchgan_reference/events \
  --batch_size 8 \
  --num_workers 4 \
  --checkpoint_interval 5000 \
  --print_freq 100 \
  --display_id 0
```

### 5.2 EC-VIAI-AV Candidate-Scorer Baseline

这是 strong baseline：multi-candidate + evidence gate + evidence-scaled sigma + Candidate Scorer + Uncertainty Head，但不做 semantic perturbation training。

```bash
export BASELINE_CKPT=checkpoints/viai_av_patchgan_reference/VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export EC_BASELINE_DIR=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.1 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$BASELINE_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$EC_BASELINE_DIR" \
  --log_event_path "$EC_BASELINE_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 90000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

### 5.3 EC-VIAI-AV Final：Semantic Perturbation Training

当前推荐的 final robustness-oriented 模型：从 EC-VIAI-AV Candidate-Scorer Baseline checkpoint 开始，使用 corrected fused semantic evidence，并在训练中加入 `wrong_video_cross_instrument,no_video,flow_zero`。

```bash
export SEM_DIR="$DATA_ROOT/semantic_evidence/clip_vit_b32"
export TRAIN_SEM="$SEM_DIR/train_av_split.jsonl"
export STAGE8_CKPT=checkpoints/formal_ec_viai_av/stage8_candidate_scorer_uncertainty/EC-VIAI-AV-PatchGAN_checkpoint_step000090000.pth.tar
export FINAL_DIR=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected

python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_source fused \
  --semantic_evidence_path "$TRAIN_SEM" \
  --semantic_evidence_weight 0.35 \
  --semantic_missing_score 0.0 \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.35 \
  --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.2 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$STAGE8_CKPT" \
  --reset_optimizer \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$FINAL_DIR" \
  --log_event_path "$FINAL_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

继续从 95000 到 100000：

```bash
python main.py train-viai-av -- \
  --use_gan \
  --lambda_gan 0.001 \
  --enable_ec_viai_av \
  --stochastic_adapter \
  --enable_evidence_gate \
  --freeze_gate_evidence_backbone \
  --enable_evidence_scaled_sigma \
  --enable_candidate_scorer \
  --evidence_source fused \
  --semantic_evidence_path "$TRAIN_SEM" \
  --semantic_evidence_weight 0.35 \
  --semantic_missing_score 0.0 \
  --enable_visual_evidence_aug \
  --visual_evidence_aug_prob 0.35 \
  --visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero \
  --evidence_sigma_scale_min 0.5 \
  --evidence_sigma_scale_max 2.0 \
  --sigma_max 2.0 \
  --num_candidates 4 \
  --test_num_candidates 4 \
  --lambda_min_k 1.0 \
  --lambda_mean_k 0.1 \
  --lambda_boundary 0.05 \
  --lambda_diversity 0.05 \
  --lambda_gate_evidence 0.2 \
  --lambda_calib 0.05 \
  --calib_error_tau 0.1 \
  --resume \
  --resume_path "$FINAL_DIR/EC-VIAI-AV-PatchGAN_checkpoint_step000095000.pth.tar" \
  --data_root "$DATA_ROOT" \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --checkpoint_dir "$FINAL_DIR" \
  --log_event_path "$FINAL_DIR/events" \
  --batch_size 2 \
  --num_workers 4 \
  --lr 0.00002 \
  --max_train_steps 100000 \
  --checkpoint_interval 5000 \
  --print_freq 50 \
  --display_id 0
```

## 6. 测试

### 6.1 Final model 四种扰动测试

```bash
export SEM_DIR="$DATA_ROOT/semantic_evidence/clip_vit_b32"
export TEST_SEM="$SEM_DIR/test_av_split.jsonl"
export FINAL_CKPT=checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/EC-VIAI-AV-PatchGAN_checkpoint_step000100000.pth.tar
export TEST_ROOT=checkpoints/stage10d_eval_100000_perturbation

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  python test_viai_av.py \
    --use_gan \
    --lambda_gan 0.001 \
    --enable_ec_viai_av \
    --stochastic_adapter \
    --enable_evidence_gate \
    --freeze_gate_evidence_backbone \
    --enable_evidence_scaled_sigma \
    --enable_candidate_scorer \
    --evidence_source fused \
    --semantic_evidence_path "$TEST_SEM" \
    --semantic_evidence_weight 0.35 \
    --semantic_missing_score 0.0 \
    --evidence_sigma_scale_min 0.5 \
    --evidence_sigma_scale_max 2.0 \
    --sigma_max 2.0 \
    --num_candidates 4 \
    --test_num_candidates 4 \
    --lambda_min_k 1.0 \
    --lambda_mean_k 0.1 \
    --lambda_boundary 0.05 \
    --lambda_diversity 0.05 \
    --lambda_gate_evidence 0.2 \
    --lambda_calib 0.05 \
    --calib_error_tau 0.1 \
    --resume_path "$FINAL_CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --results_dir "$TEST_ROOT/$MODE" \
    --video_perturbation "$MODE" \
    --calibration_bins 10 \
    --display_id 0
done
```

### 6.2 汇总测试 JSON

```bash
python - <<'PY'
import glob
import json
import os

root = os.environ["TEST_ROOT"]
modes = ["none", "flow_zero", "no_video", "wrong_video_cross_instrument"]
fields = [
    "top1_missing_l1",
    "best_of_k_missing_l1",
    "evidence_mean",
    "heuristic_evidence_mean",
    "semantic_evidence_mean",
    "gate_mean",
    "gate_target_mean",
    "gate_target_gap",
    "uncertainty_mean",
    "uncertainty_error_spearman",
    "psnr_missing",
]

print("mode," + ",".join(fields))
for mode in modes:
    path = glob.glob(f"{root}/{mode}/*_test.json")[0]
    data = json.load(open(path, encoding="utf-8"))
    vals = []
    for field in fields:
        value = data.get(field, "")
        vals.append(f"{value:.6f}" if isinstance(value, float) else str(value))
    print(mode + "," + ",".join(vals))
PY
```

### 6.3 Original VIAI-AV baseline 测试

原始 VIAI-AV 没有 multi-candidate、learned gate 或 uncertainty。测试时不要传 EC 参数。

```bash
export ORIG_CKPT=checkpoints/viai_av_patchgan_reference/VIAI-AV-PatchGAN_checkpoint_stepXXXXXXXXX.pth.tar
export ORIG_TEST_ROOT=checkpoints/original_viai_av_baseline_eval

for MODE in none flow_zero no_video wrong_video_cross_instrument
do
  python test_viai_av.py \
    --use_gan \
    --lambda_gan 0.001 \
    --resume_path "$ORIG_CKPT" \
    --data_root "$DATA_ROOT" \
    --test_split_name test_av_split.txt \
    --batch_size 1 \
    --num_workers 0 \
    --results_dir "$ORIG_TEST_ROOT/$MODE" \
    --video_perturbation "$MODE" \
    --display_id 0
done
```

## 7. 关键指标与当前结果

测试 JSON/CSV 中最常用字段：

```text
top1_missing_l1
best_of_k_missing_l1
psnr_missing
ssim
evidence_mean
heuristic_evidence_mean
semantic_evidence_mean
gate_mean
gate_target_mean
uncertainty_mean
uncertainty_error_spearman
retrieval_audio_to_video_r1
retrieval_video_to_audio_r1
```

当前主实验结论：

| Method | none Top-1 ↓ | wrong Top-1 ↓ | wrong gate ↓ | no-video gate ↓ | wrong Spearman ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original VIAI-AV | 0.063576 | 0.067543 | 1.000000 | 1.000000 | 0.000000 |
| EC-VIAI-AV Candidate-Scorer Baseline | 0.059799 | 0.061106 | 0.713195 | 0.367402 | 0.326222 |
| Semantic Evidence Fine-Tuning | 0.062668 | 0.064093 | 0.866814 | 0.528362 | 0.354754 |
| Heuristic-Only Perturbation | 0.060096 | 0.060119 | 0.537373 | 0.053071 | 0.455732 |
| Semantic Perturbation Training, Final | 0.061787 | 0.060251 | 0.413393 | 0.089303 | 0.548191 |

更多表格与图：

```text
docs/paper_draft.md
docs/exp_res_ana.md
docs/stage10d_final_tables.md
docs/figures/stage10d_ablation/
```

## 8. 输出文件

测试输出：

```text
<results_dir>/*_test.json
<results_dir>/*_test_summary.csv
<results_dir>/sample-metrics/*_samples.csv
<results_dir>/risk-coverage/*_risk_coverage.csv
<results_dir>/calibration/*_bins.csv
<results_dir>/mel-image/stepXXXXXXXXX/
```

可选保存多候选与 vocoder wav：

```bash
# 在 6.1 的测试命令基础上额外加入：
--save_candidates \
--use_vocoder \
--vocoder_max_samples 8
```

TensorBoard：

```bash
tensorboard --logdir checkpoints/formal_ec_viai_av/stage10d_semantic_perturb_fused_corrected/events --port 6006
```

重点看：

```text
train/evidence/mean
train/semantic_evidence/mean
train/heuristic_evidence/mean
train/gate/mean
train/gate/target
train/candidate/top1_missing_l1
train/candidate/best_of_k_missing_l1
train/visual_evidence_aug/applied
train/visual_evidence_aug/wrong_video_cross_instrument
train/visual_evidence_aug/no_video
train/visual_evidence_aug/flow_zero
```

## 9. 常见问题

### OpenCV 无法读取 mp4

```bash
python - <<'PY'
import cv2
print(cv2.__version__, cv2.__file__)
for line in cv2.getBuildInformation().splitlines():
    if "FFMPEG" in line or "GStreamer" in line:
        print(line)
PY
```

如果 `FFMPEG: NO`，建议重装 headless contrib：

```bash
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
python -m pip install --no-cache-dir "opencv-contrib-python-headless==4.10.0.84"
```

### 数据 split 为空

- AV：确认样本目录里有 `raw_audio.npy`、`mel.npy`、`image_crop/`、`flow_x_crop/`、`flow_y_crop/`。
- Audio-only：确认样本目录里有 `raw_audio.npy` 和 `mel.npy`。
- 查看 `$DATA_ROOT/viai_av_bad_samples.csv` 和 `$DATA_ROOT/musices_process_failures.csv`。

### semantic evidence 没有效果

先确认 JSONL 是修正后的 schema，必须包含：

```text
probs_by_instrument
frame_top1_instruments
```

再用诊断脚本确认：

```text
wrong_source_score_mean << original_source_score_mean
```

如果 wrong-video 的 source score 仍然高，说明 CLIP prompt 或视觉语义模型不足，需要重新设计 prompt、换 larger CLIP，或考虑 ImageBind / AudioCLIP。

### wrong-video gate 不下降

只接入 semantic evidence 不够。训练时必须打开：

```text
--enable_visual_evidence_aug
--visual_evidence_aug_modes wrong_video_cross_instrument,no_video,flow_zero
--lambda_gate_evidence 0.2
```

### 显存不足

- 先降低 `--batch_size`。
- smoke test 用 `--batch_size 1 --num_workers 0 --max_train_steps 1`。
- 测试建议 `--batch_size 1 --num_workers 0`，尤其是保存图片和 per-sample metrics 时。

## 10. 代码入口速查

```text
main.py                              # 统一命令入口
train_viai_av.py / test_viai_av.py   # VIAI-AV / EC-VIAI-AV 训练测试
train_viai_a.py / test_viai_a.py     # VIAI-A audio-only 训练测试
Models/VIAI_AV_inpainting.py         # EC-VIAI-AV 模型封装
networks/EC_VIAI_Modules.py          # evidence gate / candidate scorer / uncertainty 等模块
utils/semantic_evidence.py           # semantic evidence lookup / fusion
utils/wrong_video_sampler.py         # wrong-video sampler
tools/precompute_semantic_evidence.py
tools/diagnose_semantic_evidence.py
tools/make_stage10d_ablation_figures.py
tools/prepare_musices.py
tools/prepare_viai_a.py
tools/split_musices.py
utils/vocoder.py
```

## 11. 云端同步

上传到当前云端：

```bash
rsync -avz --progress \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='data/' \
  --exclude='checkpoints/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pth.tar' \
  --exclude='*.Zone.Identifier' \
  --exclude='.agents/' \
  --exclude='.codex' \
  --exclude='.python-version' \
  --exclude='NUL' \
  --exclude='uv.lock' \
  --exclude='VIAI.pdf' \
  -e "ssh -p 2233" \
  ~/VIAIpro/ \
  root@ackcs-00gjh78a@ssh.bj8.bz1.paratera.com:/root/EC-ViAv-vgpu/
```

## License and Citation

The use of the original VIAI software is restricted to **non-commercial research and educational purposes**. This repository is a research extension of VIAI-AV.

```bibtex
@InProceedings{Zhou_2019_ICCV,
  author = {Zhou, Hang and Liu, Ziwei and Xu, Xudong and Luo, Ping and Wang, Xiaogang},
  title = {Vision-Infused Deep Audio Inpainting},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  month = {October},
  year = {2019}
}
```

## Acknowledgement

The original codebase structure follows the VIAI implementation, which borrowed from [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [wavenet_vocoder](https://github.com/r9y9/wavenet_vocoder).
