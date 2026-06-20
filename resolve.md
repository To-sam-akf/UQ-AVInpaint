# 问题：

P3 K=1 diffusion 的测试结果有两个信号：

1. 视频条件无效：original/no-video/wrong-video/zero-token 几乎一样。
2. 训练/val 指标看起来很好，但真实 `test-uq-av` 采样指标明显低于 PatchGAN。

原因：
最核心原因不是“视频稍弱”，而是当前 P3 实际退化成了 audio-only latent inpainter，并且训练日志里的好指标不等价于真实推理指标。

1. 训练/val 指标偏乐观  
   [Models/UQ_AV_Diffusion.py](/home/sanmu/VIAIpro/Models/UQ_AV_Diffusion.py:390) 里的 `test()` 用的是 `z_target` 在 `t=0` 附近做 one-step teacher-forced 预测，模型几乎直接看到 clean latent。  
   但 [test_uq_av.py](/home/sanmu/VIAIpro/test_uq_av.py:244) 里真实测试是 `sample()`，从 missing 区域随机噪声开始跑 50-step DDIM。两者难度完全不同，所以训练日志 `psnr_miss≈27.7` 不能和测试 `psnr_miss≈17.4` 直接比。

2. 视频分支没有被强制使用  
   P3 只有 bottleneck 一处 cross-attention，没有 sync/evidence/contrastive/negative-video loss。主损失是 masked latent noise MSE，模型靠 `z_context + mask + boundary_map` 就能降低 loss，最省力路径就是忽略视频。

3. 视频 token 本身信息弱  
   [video_evidence_encoder.py](/home/sanmu/VIAIpro/networks/uq/video_evidence_encoder.py:48) 是从零训练的 per-frame CNN + global pooling，没有预训练 AV 表征，也没有明确的同步监督。对 300 多个样本来说，很难学出“画面动作 -> mel 缺失内容”的强映射。

4. 时间对齐注入不足  
   video tokens 没有显式 frame positional embedding；cross-attention 又只在 bottleneck 一次注入。即使视频有信息，模型也不容易知道第几帧视觉应该影响 mel 的第几段时间。

5. K=1 本来就不利于 diffusion  
   K=1 只取一个 deterministic DDIM 样本，PSNR/L1 往往不如直接回归式 PatchGAN。diffusion 的优势通常在 Best-of-K、多样性、不确定性和候选排序，不在单样本 PSNR。

解决方案：
1. 先修验证口径  
   把训练/val 阶段的指标改成 `model.sample()` 指标，或者至少新增 `val_sample_psnr_missing`。当前 `model.test()` 可以保留为 denoising debug，但不要当最终质量指标。

2. 做分层的视频有效性对照  
   若训练预算紧张，可以先用同一个 AV checkpoint 在 original video、no-video、wrong-video、zero-token、shuffled-video 上做 inference-time ablation，判断“已训练 AV 模型在推理时是否依赖视频条件”。  
   但这不能完全替代真正的 `--uq_no_video` checkpoint，因为测试时去掉视频属于分布外消融，无法回答“同等训练预算和模型容量下，audio-only 模型是否已经能达到同样效果”。因此 audio-only checkpoint 应作为最终强基线保留，可在关键结构确定后只训练一次。

3. 加强视频使用压力  
   加入 negative conditioning loss：original video 应优于 wrong/no-video；或者加 sync/contrastive loss，让 video encoder 先学会区分正确视频和错误视频。

4. 改 cross-attention 注入  
   给 video tokens 加 frame positional embedding，并在多个 U-Net 分辨率注入 cross-attention，而不是只在 bottleneck 一处。最好加一个 gate，并记录 gate/attention 权重，确认视频分支不是常年接近 0。

5. 让 P3 先追上基本采样质量  
   尝试更多 DDIM steps、cosine schedule、预测 `x0` 或 `v`、EMA、latent clipping/normalization检查；同时报告 `sample()` 的 val 曲线，避免再被 one-step 指标误导。

6. 若目标是超过 PatchGAN，进入 K>1/p8  
   用 K=8/16 + scorer/evidence 做 Best-of-K 或 rerank。单个 K=1 diffusion 不太可能在 PSNR 上自然压过直接监督的 PatchGAN。

7. classifier-free / modality dropout 训练  
   训练时随机 drop 条件模态，不只 drop 视频，也可以 partial/drop 音频 context。推荐覆盖 keep audio + keep video、keep audio + drop video、drop/partial audio + keep video、wrong/shuffled video 等组合，让模型在音频信息不足时被迫利用视频，同时保留无视频鲁棒性。
   
8. teacher-student 机制  
   可用 audio-only 或强 PatchGAN 作为 teacher，帮助 AV diffusion student 维持基础重建质量；同时配合 video margin / contrastive loss，让 student 在 original video 条件下优于 wrong/no-video。该机制更适合稳定训练和提升质量，不能单独作为“视觉有效性”的充分证据。

# plan

## p0:
需求：
修正 P3 训练/验证阶段的指标口径，避免继续用 one-step teacher-forced denoising 指标误判真实采样质量。
在此基础上加入 validation sampling 指标驱动的 early stopping。

核心逻辑：
保留 `model.test()` 作为 denoising debug 指标，但新增基于 `model.sample()` 的 validation sampling 指标，例如 `val_sample_psnr_missing`、`val_sample_mel_l1_missing`、`val_sample_ssim` 和 `val_sample_boundary_l1`。训练日志、TensorBoard 和 checkpoint 选择优先看 sampling 指标，而不是 `t=0` one-step 指标。
默认使用 `val_sample_psnr_missing_db` 选择 `UQ-AV_best.pth.tar`；若 10 个 validation epoch 内该指标没有超过 `uq_early_stop_min_delta` 的提升，则自动停止训练。可用 `--uq_disable_early_stop` 关闭自动停止但继续保存 best checkpoint。

验证方法：
1. 在 val split 上用同一个 checkpoint 同时跑 `model.test()` 和 `model.sample()`。
2. 对比训练日志中的 one-step 指标与 `test-uq-av` 输出的 summary 指标。
3. 确认新增的 val sampling 指标与正式 `test-uq-av` 指标在同一数量级。
4. 确认日志中同时出现 `val_sample_psnr_missing_db`/`val_sample_psnr_miss` 和 `val_denoise_psnr_missing`/`val_denoise_psnr_miss`，并确认 best checkpoint 写入 `UQ-AV_best.pth.tar`。

```bash
# 1. 低成本 smoke：验证 train-uq-av 会执行 validation sampling、
#    写入 val_sample/val_denoise TensorBoard 指标，并启用 patience=10。
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p0-smoke \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 1 --num_workers 0 --max_train_steps 1 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_val_inference_steps 5 \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p0_smoke.log

# 2. 检查日志里 one-step debug 和 sampling validation 同时存在。
grep -E "val_sample_psnr_miss|val_sample_psnr_missing_db|val_denoise_psnr_miss|val_denoise_psnr_missing" \
  checkpoints/uq_av_k1/p0_smoke.log

# 3. 用同一个 best checkpoint 在 val split 上跑正式 sample() 测试，
#    对比 summary.json 与训练日志中的 val_sample_* 是否同量级。
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-val-sample-check \
  --data_root /root/shared-nvme/data \
  --test_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --resume_path /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_best.pth.tar \
  --uq_unet_base_channels 128 \
  --uq_inference_steps 50 \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir checkpoints/uq_av_k1_val_sample_check

cat checkpoints/uq_av_k1_val_sample_check/summary.json
```

预期结果：
训练/验证日志不再出现 `psnr_miss≈27 dB` 但正式测试只有 `17 dB` 的误导性差异；后续调参可以直接根据真实采样质量判断是否有效。
当 validation sampling 指标连续 10 个 validation epoch 没有明显提升时，训练自动停止并保留此前最优的 `UQ-AV_best.pth.tar`。

## p1:
需求：
先做低成本 AV inference-time ablation，快速判断现有 checkpoint 是否真的使用视频条件。

核心逻辑：
不立刻新增一轮 audio-only 训练，而是用同一个 AV checkpoint 在同一 test manifest 上跑多种视频条件：

`original video / no-video / wrong-video / zero-token / shuffled-video`

这个实验回答的是：“当前已训练 AV 模型的采样结果是否对视频条件敏感？”如果 original 明显优于 no/wrong/zero/shuffled，说明视频在推理时有贡献；如果几乎一致，说明当前模型仍在走 audio-only 捷径，应先改训练目标和注入结构。

验证方法：
1. 固定 checkpoint、test manifest、mask manifest、DDIM steps 和随机种子。
2. 跑 original/no-video/wrong-video/zero-token/shuffled-video 五组测试。
3. 对比 PSNR missing、Mel L1 missing、SSIM、Boundary L1。
4. 同时记录每组指标的均值、方差和逐样本差值，避免只看平均数误判。

预期结果：
若 original 相比其他条件有稳定优势，说明视频条件在当前 AV checkpoint 中已经产生可测影响；若差异仍只有噪声级别，则暂不投入 audio-only 强基线训练，优先进入 p2/p3。

## p2:
需求：
加入 classifier-free / modality dropout，减少模型对完整音频 context 的依赖，并为 no-video 推理保留鲁棒性。

核心逻辑：
训练时随机采样条件组合，而不是始终使用 full audio + full video：

`C ∈ {audio+video, audio+drop_video, partial/drop_audio+video, audio+wrong_video, audio+shuffled_video}`

其中 drop video 用于 classifier-free 式条件 dropout；partial/drop audio 用于制造“音频不够确定，视频有机会补充”的训练场景；wrong/shuffled video 用于后续 negative conditioning。

验证方法：
1. 训练日志记录各条件组合的采样比例和 loss。
2. validation sampling 分别报告 original、drop-video、drop-audio、wrong/shuffled-video 指标。
3. 检查 drop-video 时质量不要崩，drop/partial-audio + original-video 时应优于 drop/partial-audio + wrong-video。
4. 继续保留 p1 的五组 inference-time ablation。

```bash
# 1. 单元测试：覆盖 p2 采样、drop-audio、drop-video、CLI 参数和 ablation 汇总。
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_uq_av_data.py \
  tests/test_p3_uq_av_diffusion.py \
  tests/test_uq_av_ablation_summary.py \
  -v --tb=short

# 2. CLI 参数检查：确认 p2 开关已经挂到 train/test 入口。
python main.py train-uq-av -- --help | grep -E "uq_enable_modality_dropout|uq_p_|uq_condition_override"
python main.py test-uq-av -- --help | grep -E "uq_condition_override|uq_audio_context_drop"

# 3. p2 smoke train：确认训练日志出现各 condition count/ratio/loss，
#    并且 val 侧同时输出 original / drop-video / drop-audio / wrong / shuffled 指标。
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p2-smoke \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 2 --num_workers 0 --max_train_steps 2 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_val_inference_steps 5 \
  --uq_enable_modality_dropout \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p2_smoke.log

grep -E "cond_(audio_video|drop_video|partial_audio_video|wrong_video|shuffled_video)|val_sample_(original|drop_video|drop_audio_original|drop_audio_wrong_video|wrong_video|shuffled_video)" \
  checkpoints/uq_av_k1/p2_smoke.log

# 4. 完整 p2 训练：使用保守的 40/20/20/10/10 采样比例。
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p2 \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 8 --num_workers 4 --nepochs 100 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_val_inference_steps 50 \
  --uq_enable_modality_dropout \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 50

# 5. p1 五组 inference-time ablation：继续保留。
CKPT=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_best.pth.tar
AE=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar

for COND in original no_video wrong_video shuffled_video; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
    --name UQ-AV-K1-p2-${COND} \
    --data_root /root/shared-nvme/data \
    --test_split_name test_av_split.txt \
    --ae_checkpoint $AE --resume_path $CKPT \
    --uq_unet_base_channels 128 \
    --uq_inference_steps 50 \
    --uq_video_degradation $COND \
    --batch_size 8 --num_workers 4 --display_id 0 \
    --results_dir checkpoints/uq_av_k1_p2_ablation/${COND}
done

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-p2-zero-token \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint $AE --resume_path $CKPT \
  --uq_unet_base_channels 128 \
  --uq_inference_steps 50 \
  --uq_no_video \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir checkpoints/uq_av_k1_p2_ablation/zero_token

# 6. p2 专项 drop-audio 对照：original vs wrong-video。
for SPEC in "original drop_audio_original" "wrong_video drop_audio_wrong_video"; do
  set -- $SPEC
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
    --name UQ-AV-K1-p2-$2 \
    --data_root /root/shared-nvme/data \
    --test_split_name test_av_split.txt \
    --ae_checkpoint $AE --resume_path $CKPT \
    --uq_unet_base_channels 128 \
    --uq_inference_steps 50 \
    --uq_video_degradation $1 \
    --uq_condition_override drop_audio \
    --batch_size 8 --num_workers 4 --display_id 0 \
    --results_dir checkpoints/uq_av_k1_p2_ablation/$2
done
```

预期结果：
模型在无视频条件下仍可用，但在音频 context 不完整或不确定时能从视频获得增益；original video 相比 wrong/shuffled/no-video 出现稳定优势。

## p3:
需求：
给视频分支加入可学习压力，避免 U-Net 只依赖 audio context 绕过 cross-attention。

核心逻辑：
加入 negative conditioning 或 sync/contrastive 辅助目标。训练时同一音频样本构造 original video 与 wrong/shuffled/no-video 条件，使模型在 original 条件下的 missing-region 重建误差低于错误视频条件。可先实现轻量 margin loss：

`L_video_margin = max(0, margin + L_original - L_wrong)`

总损失变为：

`L = L_diff + lambda_boundary * L_boundary + lambda_video_margin * L_video_margin`

验证方法：
1. 单元测试确认 wrong/shuffled-video batch 不改变音频和 mask，只替换 video/flow。
2. 训练日志记录 `loss_video_margin`、`L_original`、`L_wrong`。
3. validation/test 跑 original、wrong-video、shuffled-video、no-video、zero-token 五组指标。
4. 检查 original 相比 wrong/shuffled/no-video 至少有稳定正向差异。

预期结果：
视频条件不再完全可替换；wrong/shuffled/no-video 的 missing PSNR 或 Mel L1 应稳定差于 original，而不是只有 ±0.01 dB 的噪声级波动。

## P3 命令行

### 1. 单元测试

```bash
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_uq_av_data.py \
  tests/test_p3_uq_av_diffusion.py \
  tests/test_uq_av_ablation_summary.py \
  -v --tb=short
```

### 2. CLI 参数检查

```bash
python main.py train-uq-av -- --help | grep -E \
  "uq_lambda_video_margin|uq_video_margin|uq_video_margin_negative"
```

### 3. P3 smoke train

```bash
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p3-smoke \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 2 --num_workers 0 --max_train_steps 2 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_val_inference_steps 5 \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p3_smoke.log

grep -E "loss_video_margin|video_margin_l_original|video_margin_l_wrong" \
  checkpoints/uq_av_k1/p3_smoke.log
```

### 4. P3 正式训练

```bash
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p3 \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 8 --num_workers 4 --nepochs 100 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_val_inference_steps 50 \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 50
```

### 5. 五组视频条件评测

```bash
CKPT=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_best.pth.tar
AE=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar
OUT=checkpoints/uq_av_k1_p3_ablation

for COND in original no_video wrong_video shuffled_video; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
    --name UQ-AV-K1-p3-${COND} \
    --data_root /root/shared-nvme/data \
    --test_split_name test_av_split.txt \
    --ae_checkpoint $AE --resume_path $CKPT \
    --uq_unet_base_channels 128 \
    --uq_inference_steps 50 \
    --uq_video_degradation $COND \
    --batch_size 8 --num_workers 4 --display_id 0 \
    --results_dir $OUT/${COND}
done

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-p3-zero-token \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint $AE --resume_path $CKPT \
  --uq_unet_base_channels 128 \
  --uq_inference_steps 50 \
  --uq_no_video \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir $OUT/zero_token

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python tools/summarize_uq_av_ablation.py \
  --original $OUT/original \
  --no-video $OUT/no_video \
  --wrong-video $OUT/wrong_video \
  --zero-token $OUT/zero_token \
  --shuffled-video $OUT/shuffled_video \
  --output-dir $OUT/summary
```

## p4:
需求：
增强视频条件注入能力，让模型具备使用时间对齐视觉信息的结构基础。

核心逻辑：
给 video tokens 加 frame positional embedding，并把 cross-attention 从 bottleneck 单点注入扩展到多个 U-Net 层级。优先采用 gated cross-attention：

`h = h + sigmoid(gate) * CrossAttention(h, video_tokens)`

同时记录 gate 均值、attention 输出范数、video token 范数，作为视频分支是否被使用的诊断指标。

验证方法：
1. 单元测试覆盖 positional embedding shape、multi-level attention shape、无视频模式兼容。
2. 训练时记录 `video_gate_mean`、`video_attn_norm`、`video_token_norm`。
3. 对同一 batch 分别输入 original、wrong-video、zero-token，检查 U-Net 中间激活或最终预测是否产生可测差异。
4. 重新跑 p1 的五组 ablation。

预期结果：
模型输出对视频条件产生可观测敏感性；zero-token、wrong-video、shuffled-video 不再与 original 完全重合。

### p4 训练与验证命令

#### 1. 单元测试

```bash
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH pytest \
  tests/test_p3_uq_av_diffusion.py \
  tests/test_uq_av_ablation_summary.py \
  -v --tb=short
```

#### 2. P4 smoke train + 诊断指标检查

```bash
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p4-smoke \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 2 --num_workers 0 --max_train_steps 2 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_val_inference_steps 5 \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p4_smoke.log

grep -E "video_gate_mean|video_attn_norm|video_token_norm|loss_video_margin" \
  checkpoints/uq_av_k1/p4_smoke.log
```

#### 3. P4 正式训练

```bash
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p4 \
  --data_root /root/shared-nvme/data \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint /root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar \
  --batch_size 8 --num_workers 4 --nepochs 100 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_val_inference_steps 50 \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 50
```

#### 4. p1 五组 inference-time ablation

```bash
CKPT=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_best.pth.tar
AE=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar
OUT=checkpoints/uq_av_k1_p4_ablation

for COND in original no_video wrong_video shuffled_video; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
    --name UQ-AV-K1-p4-${COND} \
    --data_root /root/shared-nvme/data \
    --test_split_name test_av_split.txt \
    --ae_checkpoint $AE --resume_path $CKPT \
    --uq_unet_base_channels 128 \
    --uq_inference_steps 50 \
    --uq_video_degradation $COND \
    --batch_size 8 --num_workers 4 --display_id 0 \
    --results_dir $OUT/${COND}
done

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-p4-zero-token \
  --data_root /root/shared-nvme/data \
  --test_split_name test_av_split.txt \
  --ae_checkpoint $AE --resume_path $CKPT \
  --uq_unet_base_channels 128 \
  --uq_inference_steps 50 \
  --uq_no_video \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir $OUT/zero_token

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python tools/summarize_uq_av_ablation.py \
  --original $OUT/original \
  --no-video $OUT/no_video \
  --wrong-video $OUT/wrong_video \
  --zero-token $OUT/zero_token \
  --shuffled-video $OUT/shuffled_video \
  --output-dir $OUT/summary
```

## p5:
需求：
引入 teacher-student 稳定训练，在强化视频使用的同时避免基础重建质量下降。

核心逻辑：
可用强 PatchGAN 或后续 audio-only diffusion 作为 teacher，给 AV diffusion student 提供基础重建蒸馏目标。teacher 负责稳定音频补全质量，video margin / contrastive loss 负责拉开 original 与 wrong/no-video 的差距。需要注意：teacher-student 不能单独证明视觉有效性，只能作为质量和稳定性辅助。

验证方法：
1. 分别记录 `L_diff`、`L_distill`、`L_video_margin`。
2. 对比无 teacher、PatchGAN teacher、audio-only teacher 三种配置。
3. 检查 student 不只是复制 teacher：original video 应优于 wrong/no-video。
4. 继续使用 `model.sample()` 指标，而不是 one-step teacher-forced 指标。

预期结果：
AV diffusion 在基础 PSNR/L1 上不明显退化，同时仍保留对 original video 的正向敏感性。

### p5 训练与验证命令

```bash
# ============================================================
# P5 teacher-student distillation 验证命令
# ============================================================

DATA_ROOT=/root/shared-nvme/data
AE=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar
PATCHGAN_TEACHER=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/viai-a_patchfromscratch/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar
AUDIO_TEACHER=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1_audio_only/UQ-AV_best.pth.tar

# 1. P5/P6 单元测试
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_p3_uq_av_diffusion.py \
  tests/test_uq_av_sampling_sweep.py \
  -v --tb=short

# 2. CLI 参数检查
python main.py train-uq-av -- --help | grep -E "uq_teacher_type|uq_lambda_distill|uq_prediction_type|uq_use_ema"
python main.py test-uq-av -- --help | grep -E "uq_ema_eval|uq_prediction_type|uq_latent_clip_value"

# 3. smoke: no teacher
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p5-smoke-no-teacher \
  --data_root $DATA_ROOT \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint $AE \
  --batch_size 2 --num_workers 0 --max_train_steps 2 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_val_inference_steps 5 \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p5_smoke_no_teacher.log

# 4. smoke: PatchGAN teacher
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p5-smoke-patchgan-teacher \
  --data_root $DATA_ROOT \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint $AE \
  --batch_size 2 --num_workers 0 --max_train_steps 2 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_teacher_type patchgan \
  --uq_teacher_checkpoint $PATCHGAN_TEACHER \
  --uq_lambda_distill 0.5 \
  --uq_val_inference_steps 5 \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p5_smoke_patchgan_teacher.log

# 5. smoke: audio-only diffusion teacher
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p5-smoke-audio-teacher \
  --data_root $DATA_ROOT \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint $AE \
  --batch_size 2 --num_workers 0 --max_train_steps 2 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_teacher_type audio_only_diffusion \
  --uq_teacher_checkpoint $AUDIO_TEACHER \
  --uq_teacher_ae_checkpoint $AE \
  --uq_teacher_inference_steps 25 \
  --uq_teacher_ddim_eta 0.0 \
  --uq_lambda_distill 0.5 \
  --uq_val_inference_steps 5 \
  --display_id 0 --print_freq 1 2>&1 | tee checkpoints/uq_av_k1/p5_smoke_audio_teacher.log

grep -E "loss_distill|loss_video_margin|diff=" checkpoints/uq_av_k1/p5_smoke_*.log

# 6. full train: no teacher 对照
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p5-no-teacher \
  --data_root $DATA_ROOT \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint $AE \
  --batch_size 8 --num_workers 4 --nepochs 100 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_val_inference_steps 50 \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 50

# 7. full train: PatchGAN teacher
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p5-patchgan-teacher \
  --data_root $DATA_ROOT \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint $AE \
  --batch_size 8 --num_workers 4 --nepochs 100 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_teacher_type patchgan \
  --uq_teacher_checkpoint $PATCHGAN_TEACHER \
  --uq_lambda_distill 0.5 \
  --uq_val_inference_steps 50 \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 50

# 8. full train: audio-only diffusion teacher
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p5-audio-teacher \
  --data_root $DATA_ROOT \
  --train_split_name train_av_split.txt \
  --val_split_name val_av_split.txt \
  --ae_checkpoint $AE \
  --batch_size 8 --num_workers 4 --nepochs 100 \
  --uq_unet_base_channels 128 \
  --uq_lambda_boundary 0.1 \
  --uq_enable_modality_dropout \
  --uq_lambda_video_margin 0.1 \
  --uq_video_margin 0.02 \
  --uq_video_margin_negative cycle \
  --uq_teacher_type audio_only_diffusion \
  --uq_teacher_checkpoint $AUDIO_TEACHER \
  --uq_teacher_ae_checkpoint $AE \
  --uq_teacher_inference_steps 50 \
  --uq_teacher_ddim_eta 0.0 \
  --uq_lambda_distill 0.5 \
  --uq_val_inference_steps 50 \
  --uq_early_stop_patience 10 \
  --display_id 0 --print_freq 50

# 9. 对最终 student 跑 original/no/wrong/shuffled/zero-token ablation
CKPT=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_best.pth.tar
OUT=checkpoints/uq_av_k1_p5_ablation

for COND in original no_video wrong_video shuffled_video; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
    --name UQ-AV-K1-p5-${COND} \
    --data_root $DATA_ROOT \
    --test_split_name test_av_split.txt \
    --ae_checkpoint $AE --resume_path $CKPT \
    --uq_unet_base_channels 128 \
    --uq_inference_steps 50 \
    --uq_video_degradation $COND \
    --batch_size 8 --num_workers 4 --display_id 0 \
    --results_dir $OUT/${COND}
done

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-p5-zero-token \
  --data_root $DATA_ROOT \
  --test_split_name test_av_split.txt \
  --ae_checkpoint $AE --resume_path $CKPT \
  --uq_unet_base_channels 128 \
  --uq_inference_steps 50 \
  --uq_no_video \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir $OUT/zero_token

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python tools/summarize_uq_av_ablation.py \
  --original $OUT/original \
  --no-video $OUT/no_video \
  --wrong-video $OUT/wrong_video \
  --zero-token $OUT/zero_token \
  --shuffled-video $OUT/shuffled_video \
  --output-dir $OUT/summary
```

## p6:
需求：
提升 K=1 diffusion 的基础采样质量，先缩小与 PatchGAN 的硬指标差距。

核心逻辑：
围绕采样链路做稳定性改进：尝试 cosine beta schedule、更多 DDIM steps、EMA checkpoint、预测 `x0` 或 `v`、检查 latent clamp 与 latent normalization。所有改动都必须用 `model.sample()` 指标评估，不能用 one-step denoising 指标代替。

验证方法：
1. 固定 test manifest，对比 linear/cosine schedule。
2. 对比 50/100/200 DDIM steps 的 PSNR missing、Mel L1 missing 和耗时。
3. 若加入 EMA，比较 raw checkpoint 与 EMA checkpoint。
4. 检查 known region max abs error 仍为 0。

预期结果：
K=1 的 missing-region 指标明显优于当前 `17.4 dB` 水平，并尽量接近 PatchGAN；如果仍无法接近，说明 K=1 diffusion 不适合作为直接 PSNR 竞争方案。

### p6 训练与验证命令

```bash
# ============================================================
# P6 K=1 sampling quality sweep
# ============================================================

DATA_ROOT=/root/shared-nvme/data
AE=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/mel_ae/MelAE_checkpoint_step000009950.pth.tar
BASE_ARGS="--data_root $DATA_ROOT --train_split_name train_av_split.txt --val_split_name val_av_split.txt --ae_checkpoint $AE --batch_size 8 --num_workers 4 --nepochs 100 --uq_unet_base_channels 128 --uq_lambda_boundary 0.1 --uq_enable_modality_dropout --uq_lambda_video_margin 0.1 --uq_video_margin 0.02 --uq_video_margin_negative cycle --uq_val_inference_steps 50 --uq_early_stop_patience 10 --display_id 0 --print_freq 50"

# 1. 单元测试 + CLI 参数检查
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python -m pytest \
  tests/test_p3_uq_av_diffusion.py \
  tests/test_uq_av_sampling_sweep.py \
  -v --tb=short

python main.py train-uq-av -- --help | grep -E "uq_beta_schedule|uq_prediction_type|uq_latent_clip_value|uq_use_ema|uq_require_latent_stats"
python main.py test-uq-av -- --help | grep -E "uq_inference_steps|uq_ema_eval|uq_prediction_type|uq_latent_clip_value"

# 2. linear vs cosine schedule
for SCHEDULE in linear cosine; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
    --name UQ-AV-K1-p6-${SCHEDULE} \
    $BASE_ARGS \
    --uq_beta_schedule $SCHEDULE \
    --uq_prediction_type epsilon \
    --uq_latent_clip_value 4.0
done

# 3. epsilon/x0/v prediction objective
for PRED in epsilon x0 v; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
    --name UQ-AV-K1-p6-pred-${PRED} \
    $BASE_ARGS \
    --uq_beta_schedule cosine \
    --uq_prediction_type $PRED \
    --uq_latent_clip_value 4.0
done

# 4. EMA 训练
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py train-uq-av -- \
  --name UQ-AV-K1-p6-cosine-v-ema \
  $BASE_ARGS \
  --uq_beta_schedule cosine \
  --uq_prediction_type v \
  --uq_latent_clip_value 4.0 \
  --uq_use_ema \
  --uq_ema_decay 0.999 \
  --uq_ema_start_step 100

# 5. 固定 checkpoint 对比 50/100/200 DDIM steps
CKPT=/root/Vision-Infused-Audio-Inpainter-VIAI/checkpoints/uq_av_k1/UQ-AV_best.pth.tar
OUT=checkpoints/uq_av_k1_p6_sampling_sweep

for STEPS in 50 100 200; do
  PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
    --name UQ-AV-K1-p6-steps-${STEPS} \
    --data_root $DATA_ROOT \
    --test_split_name test_av_split.txt \
    --ae_checkpoint $AE --resume_path $CKPT \
    --uq_unet_base_channels 128 \
    --uq_beta_schedule cosine \
    --uq_prediction_type v \
    --uq_latent_clip_value 4.0 \
    --uq_inference_steps $STEPS \
    --batch_size 8 --num_workers 4 --display_id 0 \
    --results_dir $OUT/steps_${STEPS}
done

# 6. raw vs EMA checkpoint 对比
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-p6-raw \
  --data_root $DATA_ROOT \
  --test_split_name test_av_split.txt \
  --ae_checkpoint $AE --resume_path $CKPT \
  --uq_unet_base_channels 128 \
  --uq_beta_schedule cosine \
  --uq_prediction_type v \
  --uq_latent_clip_value 4.0 \
  --uq_inference_steps 100 \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir $OUT/raw_100

PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python main.py test-uq-av -- \
  --name UQ-AV-K1-p6-ema \
  --data_root $DATA_ROOT \
  --test_split_name test_av_split.txt \
  --ae_checkpoint $AE --resume_path $CKPT \
  --uq_unet_base_channels 128 \
  --uq_beta_schedule cosine \
  --uq_prediction_type v \
  --uq_latent_clip_value 4.0 \
  --uq_inference_steps 100 \
  --uq_ema_eval \
  --batch_size 8 --num_workers 4 --display_id 0 \
  --results_dir $OUT/ema_100

# 7. known region 必须仍严格为 0
python - <<'PY'
import json
from pathlib import Path
paths = sorted(Path("checkpoints/uq_av_k1_p6_sampling_sweep").glob("*/summary.json"))
for path in paths:
    summary = json.loads(path.read_text())
    value = float(summary["known_region_max_abs_error_max"])
    print(path, value)
    assert value == 0.0, f"known region changed for {path}: {value}"
PY

# 8. 汇总 schedule / DDIM steps / raw-vs-EMA sweep
PYTHONPATH=/tmp/opencv_fix:$PYTHONPATH python tools/summarize_uq_av_sampling_sweep.py \
  --run steps_50 $OUT/steps_50 \
  --run steps_100 $OUT/steps_100 \
  --run steps_200 $OUT/steps_200 \
  --run raw_100 $OUT/raw_100 \
  --run ema_100 $OUT/ema_100 \
  --output-dir $OUT/summary
```

## p7:
需求：
在关键结构确定后，补一次真正的 audio-only diffusion 强基线，用于最终论文/报告结论。

核心逻辑：
训练一个真正的 `--uq_no_video` checkpoint，而不是只在测试时 zero token。该步骤后置执行，避免每轮调参都增加训练成本。AV 与 audio-only 必须使用相同 split、mask manifest、AE checkpoint、U-Net 容量、diffusion schedule、训练步数和测试命令，只改变是否启用视频 encoder/cross-attention。

验证方法：
1. 训练 `UQ-AV-K1-audio-only`。
2. 在同一 test manifest 上分别测试最终 AV checkpoint 和 audio-only checkpoint。
3. 对比 PSNR missing、Mel L1 missing、SSIM、Boundary L1。
4. 同时跑 AV checkpoint 的 original/no-video/wrong-video/zero-token/shuffled-video ablation。

预期结果：
若最终 AV checkpoint 明显优于真正 audio-only checkpoint，且 original video 明显优于 no/wrong/zero/shuffled，才能给出强结论：视频模态不仅在推理时被使用，也在训练后的最终性能上带来了有效增益。

## p8:
需求：
进入 K>1 候选生成与重排序，让 diffusion 发挥多样性和不确定性优势。

核心逻辑：
扩展 `test-uq-av` 支持 K 个候选，导出每个候选的 Mel、latent、指标和候选分数。先实现 oracle Best-of-K 作为上界，再实现 scorer/evidence rerank，比较 `K=1`、`Mean-K`、`Best-of-K` 和 `Scorer-K`。

验证方法：
1. 用 K=4/8/16 在同一 test manifest 上采样。
2. 输出 `best_of_k_psnr_missing`、`mean_k_psnr_missing`、`diversity`、`boundary_l1`。
3. 加入 no-video/wrong-video/shuffled-video ablation，观察 original video 是否提升 Best-of-K 或 rerank 命中率。
4. 与 VIAI-AV PatchGAN 在同一 376 样本测试集上对齐比较。

预期结果：
即使单个 K=1 不超过 PatchGAN，Best-of-K 或 scorer rerank 应该体现 diffusion 的候选优势；同时 original video 应在候选选择或不确定性校准上优于 wrong/no-video。
