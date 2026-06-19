## VIAI-AV train metrics
VIAI-AV 训练时 TensorBoard 里主要有这些指标，前缀通常是 train/ 或 val/。

重建质量指标

psnr_full：完整 Mel 频谱上的 PSNR。越高越好，表示整体重建结果越接近真实 Mel。

psnr_missing：只在被 mask / 缺失区域上计算的 PSNR。越高越好，这个更关键，因为 VIAI 的目标就是修复缺失音频。

ssim_full：完整 Mel 频谱上的 SSIM，越高越好。训练时不是每步都算，默认每 metric_freq 步算一次，因为比较耗 CPU。

总损失

loss_total：生成器总损失，训练主要优化它。VIAI-AV 中大致由重建损失、GAN loss、sync loss、probe loss 组合而来。越低通常越好，但不能只看它，要结合
psnr_missing。

音频修复损失

loss_av_gen：VIAI-AV 生成器主分支损失，包含音视频融合后的修复目标。

loss_recon：重建损失，主要衡量预测 Mel 和目标 Mel 的差异。

loss_full_l1：完整 Mel 区域的 L1 loss。

loss_missing_l1：缺失区域的 L1 loss。这个很重要，直接反映被挖空部分修得怎么样。越低越好。

GAN 相关损失

loss_g_gan：生成器骗过判别器的 GAN loss。训练 PatchGAN 时使用。

loss_d：判别器总损失。

loss_d_real：判别器判断真实 Mel 的 loss。

loss_d_fake：判别器判断生成 Mel 的 loss。

GAN 指标不要简单理解成越低越好。判别器和生成器是对抗关系，重点看是否稳定，不要剧烈爆炸或长期失衡。

音视频同步 / probe 分支

loss_sync：audio-video sync loss，用来约束音频特征和视频特征对齐。越低表示音视频表征更一致。

loss_probe_gen：probe branch 的总生成损失。这个分支用于辅助 VIAI-AA' 式的音频修复监督。

loss_probe_recon：probe 分支的重建损失。

loss_probe_full_l1：probe 分支完整 Mel 的 L1 loss。

loss_probe_missing_l1：probe 分支缺失区域的 L1 loss。

loss_probe_g_gan：probe 分支对应的 GAN loss。

训练状态指标

lr：当前学习率。

blank_frames：当前训练 step 使用的缺失帧长度，也就是 mask 掉多少 Mel frames。

eta1：重建损失权重系数，代码里会随 step 衰减。

eta2：sync/probe 相关权重系数，代码里也会随 step 衰减。

重点看哪些

训练 VIAI-AV 时建议优先看：

train/loss_total
train/loss_missing_l1
train/psnr_missing
val/loss_missing_l1
val/psnr_missing
train/loss_sync
train/loss_d
train/loss_g_gan

如果 train 指标变好但 val/psnr_missing 不升或下降，可能过拟合。
如果 loss_d 很快接近 0，而 loss_g_gan 很高，说明判别器可能太强。
如果 psnr_missing 稳定上升，通常说明缺失区域修复质量在改善。