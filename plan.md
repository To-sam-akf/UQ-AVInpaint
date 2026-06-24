第1步：
需求：
接入已完成的 VIAI-AV baseline 结果，作为本仓库优化工作的固定参照。
核心逻辑：
不在本仓库重复 baseline 固化工作，而是复用另一个仓库已经得到的 VIAI-AV baseline checkpoint、测试指标和实验配置。本仓库只负责在当前 VIAI-AV 架构上做 EC-VIAI-AV 扩展，并保证所有新实验都能和既有 baseline 在同一数据、同一 split、同一 mask 设置下对齐比较。
具体改动：
- 从另一个仓库拷贝或记录已验证的 VIAI-AV baseline checkpoint 路径、训练命令、测试 JSON/CSV 和关键指标。
- 在本仓库建立 `docs/baseline_reference.md` 或实验记录表，写清楚 baseline 来源、commit、数据 split、mask 设置和 checkpoint step。
- 确认当前仓库的数据准备方式与 baseline 仓库一致：`train_av_split.txt`、`val_av_split.txt`、`test_av_split.txt`，输入仍为 4 秒、80 x 200 Mel、50 帧 RGB + flow。
- 如果需要从 baseline checkpoint 初始化 EC-VIAI-AV，只加载可复用权重：`Mel_Encoder`、`VideoEncoder`、`Mel_Decoder`，新增模块随机初始化。
- 固定参照名称：`VIAI-AV reference`、`VIAI-AV-PatchGAN reference`，避免把本仓库的新训练误认为 baseline 结果。
验证方法：
- 用当前仓库的 `test-viai-av` 尝试加载 reference checkpoint，确认结构兼容；如果 checkpoint 来自不同代码版本，只记录指标，不强制加载。
- 对比 baseline reference 的 split 名称、blank frame 范围、是否启用 sync/probe/PatchGAN，确保后续实验配置一致。
- 跑一次 EC-VIAI-AV smoke test，确认新分支可以从 reference checkpoint 初始化。
预期结果：
本仓库不重复训练 baseline，但拥有清晰、可追踪、可对比的 VIAI-AV reference，后续优化结果可以直接与它公平比较。

第2步：
需求：
补齐新模型的命令行开关和实验配置，保证优化路线可开关、可消融、可回退。
核心逻辑：
不要直接替换 VIAI-AV，而是在现有训练入口上新增 `EC-VIAI-AV` 模式。默认关闭新增随机分支时，行为应等价于当前 VIAI-AV。
具体改动：
- 在 `base_options.py` 新增参数：`--enable_ec_viai_av`、`--num_candidates`、`--stochastic_adapter`、`--deterministic_adapter`。
- 新增损失权重：`--lambda_min_k`、`--lambda_mean_k`、`--lambda_boundary`、`--lambda_diversity`、`--lambda_calib`。
- 新增 evidence 相关参数：`--enable_evidence_gate`、`--evidence_source`、`--sigma_min`、`--sigma_max`。
- 新增测试参数：`--test_num_candidates`、`--save_candidates`、`--video_perturbation`。
- 在 `train_viai_av.py` 和 `test_viai_av.py` 的运行配置打印中加入这些参数。
验证方法：
- 运行 `python main.py train-viai-av -- --help`，确认新增参数可见。
- 运行不带新参数的 VIAI-AV 前向或测试命令，确认新增参数默认关闭时不改变现有逻辑。
- 运行带 `--enable_ec_viai_av --deterministic_adapter --num_candidates 1` 的 smoke test。
预期结果：
新功能通过参数显式启用，baseline 路径保持干净，便于后续逐步消融。

第3步：
需求：
实现 Evidence Estimator，估计视频证据强弱，为后续门控和不确定性提供输入。
核心逻辑：
第一版不引入关键点检测或额外预训练模型，只复用已有 RGB、flow、audio-video embedding 信号。证据强度 `e` 越高，说明视频越可信；`e` 越低，说明模型应该少依赖视频并允许更多候选。
具体改动：
- 新建 `networks/EC_VIAI_Modules.py`，加入 `VisualEvidenceEstimator`。
- 输入包括 `video_feature`、`flow_batch` 的 magnitude/temporal variance、`mel_target_feature_flat` 与 `video_feature_flat` 的 sync distance。
- 输出 batch 级 evidence：`e`，形状为 `[B, 1]`，范围用 sigmoid 限制到 `[0, 1]`。
- 在 `Models/VIAI_AV_inpainting.py` 的 `_forward_inpainter` 中计算并保存 `self.evidence_score`。
- 在 TensorBoard 中记录 `evidence/mean`、`evidence/min`、`evidence/max`。
验证方法：
- 用正常视频、flow 置零、wrong video 或 temporal shift 的小 batch 做前向测试。
- 检查正常视频的 evidence 均值应高于 flow 置零或错位视频。
- 确认 evidence 计算不会改变 baseline 输出，除非启用 evidence gate。
预期结果：
得到一个轻量、可解释、可记录的视觉证据分数，支撑“视觉证据控制不确定性”的论文主线。

第4步：
需求：
加入可退化的 deterministic bottleneck adapter，先确认模型扩展不会破坏 baseline。
核心逻辑：
在 MelEncoder 和 MelDecoderImage 之间插入 adapter，但先关闭随机性，让它学习近似恒等映射。这样可以验证工程改动本身不会导致指标明显下降。
具体改动：
- 在 `networks/EC_VIAI_Modules.py` 新增 `BottleneckAdapter`。
- 输入 `mel_features[-1]` 和 `video_feature`，输出与瓶颈特征同 shape 的 residual。
- 初始 residual scale 设为 0 或极小值，使模型初始等价于 VIAI-AV。
- 修改 `VIAIAVModel` 的 generator 参数列表，把 adapter 参数加入 `optimizer_G`。
- 修改 checkpoint 保存和加载，增加 `EvidenceEstimator`、`BottleneckAdapter` 的 state_dict；老 checkpoint 加载时允许新模块随机初始化。
验证方法：
- 从 VIAI-AV checkpoint resume 或 init，运行 `--enable_ec_viai_av --deterministic_adapter --num_candidates 1`。
- 对比同一 batch 上 adapter 关闭与开启后的 mel L1、PSNR、SSIM。
- 训练少量 step 后确认 loss 不爆炸，checkpoint 可保存和加载。
预期结果：
得到可稳定训练的 EC-VIAI-AV 基础结构，并在 K=1、无随机性时接近 VIAI-AV baseline。

第5步：
需求：
实现 stochastic bottleneck adapter，让模型生成 K 个缺失音频候选。
核心逻辑：
把当前确定性输出 `missing_mel = f(audio_context, video, mask)` 扩展为 K 个候选。随机变量作用在 bottleneck residual 上，MelDecoderImage 权重继续共享，最大限度复用 VIAI-AV。
具体改动：
- 在 `BottleneckAdapter` 中新增 `mu_head`、`logvar_head` 或 `sigma_head`。
- 采样 `z_k = mu + sigma * eps_k`，生成 `fused_k = fused + Adapter(z_k)`。
- 在 `VIAIAVModel._forward_inpainter` 中根据 `num_candidates` 生成 `self.mel_candidates`，形状建议为 `[B, K, 1, 80, 200]`。
- 保留 `self.mel_pred` 作为 top-1 或第一个候选，兼容现有训练和测试代码。
- 推理和指标计算时使用 mask compose：已知区域保持 `mel_input`，只替换 missing 区域。
验证方法：
- `K=1` 时输出 shape 与当前 `mel_pred` 一致。
- `K=4/8` 时确认每个候选 shape 正确，且显存占用可接受。
- 固定随机种子时采样可重复，不固定种子时候选之间存在非零 pairwise distance。
预期结果：
模型可以一次前向生成多个合理候选，为 best-of-K、diversity 和 calibration 指标打基础。

第6步：
需求：
加入多候选训练损失，使 K 个候选既贴近真实缺失片段，又不退化成无意义噪声。
核心逻辑：
用 baseline anchoring 保住质量，用 best-of-K 鼓励候选池覆盖真实答案，用 mean-K 防止只有一个候选好，用 boundary loss 改善缺失边界听感。
具体改动：
- 在 `VIAIAVModel` 中新增 `_multi_candidate_losses`。
- 实现 `L_anchor`：候选均值或 candidate 0 与 ground truth 的 reconstruction loss。
- 实现 `L_minK`：missing region 上 `min_k L1(mel_candidate_k, mel_target)`。
- 实现 `L_meanK`：所有候选 missing L1 的平均。
- 实现 `L_boundary`：缺失片段左右边界的一阶差分误差。
- 将总损失扩展为 `loss_total = baseline losses + lambda_min_k * L_minK + lambda_mean_k * L_meanK + lambda_boundary * L_boundary`。
- 在 `get_loss_items` 和 `TF_writer` 中记录新增 loss。
验证方法：
- 用 `K=4` 跑 1 step backward，确认所有新增 loss 为 finite。
- 人工构造 mask 边界，单元测试或脚本检查 boundary loss 不越界。
- 比较 `K=1` 与 `K=4` 的 `best_of_k_missing_l1`，确认统计逻辑正确。
预期结果：
多候选训练稳定，best-of-K 有机会优于单输出，同时 mean quality 不明显崩坏。

第7步：
需求：
实现 Evidence-Aware Fusion Gate，让模型根据视觉证据强弱决定依赖视频的程度。
核心逻辑：
当视频清晰、同步、flow 明显时提高视频分支权重；当视频模糊、遮挡、错位或无视频时降低视频分支权重，并使用 audio prior 兜底。
具体改动：
- 在 `EC_VIAI_Modules.py` 新增 `EvidenceFusionGate`。
- 计算 `g = sigmoid(MLP([audio_bottleneck_pool, video_feature_pool, evidence]))`。
- 得到 `video_feature_calibrated = g * video_feature + (1 - g) * learned_audio_prior`。
- 将 `MelDecoderImage(..., video_feature)` 改为使用 calibrated video feature。
- 新增 `L_evidence_div`：让 pairwise candidate distance 接近 `d_min + alpha * (1 - evidence)`。
- TensorBoard 记录 `gate/mean`、`candidate_pairwise_distance`、`evidence_diversity_gap`。
验证方法：
- 正常视频、flow zero、wrong video、temporal shift 条件下比较 gate 均值。
- 检查低 evidence 样本的候选 pairwise distance 高于高 evidence 样本。
- 消融 `--enable_evidence_gate`，比较有无 gate 的鲁棒性指标。
预期结果：
模型不再盲目信任视频，视觉证据弱时不确定性和候选多样性上升，视觉证据强时候选更集中。

第8步：
需求：
加入 candidate scorer 和 uncertainty head，实现 top-1 选择与不确定性校准。
核心逻辑：
K 个候选不是只用于 oracle best-of-K，模型还要学会自己选择最可信候选，并输出整体不确定性 `u`。`u` 应与真实错误正相关。
具体改动：
- 在 `EC_VIAI_Modules.py` 新增 `CandidateScorer` 和 `UncertaintyHead`。
- scorer 输入每个候选的 missing L1 proxy、boundary feature、sync score、audio context feature、evidence score，输出 `pi_k`。
- top-1 输出设为 `argmax(pi_k)` 或 soft-weighted candidate，并保存为 `self.mel_pred`。
- 实现 `L_calib`：用 `min_k error.detach()` 作为难度目标，训练 `u` 与错误正相关。
- 测试 CSV/JSON 新增 `top1_missing_l1`、`best_of_k_missing_l1`、`oracle_gain`、`uncertainty_error_corr`。
验证方法：
- 检查 `pi_k` 每行 sum 为 1，`u` shape 为 `[B, 1]`。
- 在 validation 上比较 scorer top-1 与随机候选、candidate 0 的 missing L1。
- 计算 uncertainty-error Pearson/Spearman，确认相关性为正。
预期结果：
模型不仅能生成多个候选，还能给出可用的 top-1 结果和有意义的不确定性分数。

第9步：
需求：
扩展测试协议，评估多候选质量、多样性、校准、边界连续性和视觉扰动鲁棒性。
核心逻辑：
论文主实验不能只看单一 Mel L1。需要证明：K 候选提升 oracle 上限，scorer 能选好候选，低视觉证据产生更高不确定性，错位或错误视频不会被盲信。
具体改动：
- 在 utils/viai_a_metrics.py 新增多候选指标函数：top-1 error、best-of-K error、mean-K quality、pairwise Mel distance、boundary delta error。
- 在 test_viai_av.py 扩展 RESULT_FIELDS，写出新增指标。
- 增加测试时视频扰动：blur、flow_zero、frame_drop、temporal_shift、wrong_video、no_video。
- 支持保存每个样本的 candidate mel 图和可选 vocoder wav，目录按 candidate_00、candidate_01 分开。
- 汇总 risk-coverage curve 和 calibration bin 统计，便于画图。
验证方法：
- 用小测试集运行每种 perturbation，确认 JSON/CSV 字段齐全。
- 手动检查候选 mel 图：已知区域不被改变，missing 区域有差异。
- 对比 original video 与 wrong/no video，确认 sync/retrieval 或 evidence 指标发生合理变化。
预期结果：
形成完整的 EC-VIAI-AV 评价协议，能支撑论文里的质量、多样性和不确定性校准分析。

第10步：
需求：
设计正式实验矩阵和消融实验，验证每个新增模块的贡献。
核心逻辑：
主对比必须围绕同一 VIAI-AV backbone 展开，证明收益来自 evidence-calibrated multi-hypothesis，而不是换数据、换主干或换训练技巧。
具体改动：
- 引用另一个仓库已完成的 baseline：VIAI-A、VIAI-A + PatchGAN、VIAI-AV、VIAI-AV + PatchGAN。
- 跑同架构变体：`VIAI-AV-K`、`VIAI-AV-K + scorer`、`VIAI-AV-K + evidence`、`EC-VIAI-AV full`。
- K 值实验：`K=1,4,8,16`。
- mask 实验：20-50 frames、60 frames、80/100 frames、onset-centered gap。
- 消融实验：no evidence gate、fixed sigma、no scorer、no calibration loss、no diversity target、no sync loss、no probe branch。
- 将每个实验的命令、checkpoint、结果 CSV 和图表路径整理成固定表格。
验证方法：
- 每组实验至少跑一次完整 test split，并确保记录同一套指标。
- 比较 `best_of_k_missing_l1`、`top1_missing_l1`、`pairwise_distance`、`uncertainty_error_corr`。
- 检查 full model 在 original video 上 top-1 不弱于 baseline，在低 evidence 条件下 uncertainty 更高。
预期结果：
得到一套可写论文的实验结论：EC-VIAI-AV 复用 VIAI-AV baseline，在保持重建质量的同时提供多候选生成、视觉证据感知和不确定性校准能力。

第11步：
需求：
整理工程文档和实验入口，方便后续训练、测试和论文写作。
核心逻辑：
所有新增功能都要能被 README 或单独实验文档追踪和复跑，避免代码可跑但实验不可追踪。
具体改动：
- 在 README 增加 EC-VIAI-AV 训练和测试命令示例。
- 新增 `docs/ec_viai_av_experiments.md`，记录 baseline、K-sampling、perturbation、ablation 的推荐命令。
- 在 checkpoint 中保存新参数、stage 名称、K 值、是否启用 evidence/scorer/calibration。
- 结果文件命名包含模型名、K 值、扰动类型和 checkpoint step。
验证方法：
- 从空白 shell 按文档命令跑通 smoke test。
- 确认 checkpoint resume、test、CSV 追加、mel image 保存均正常。
- 复查 README 中不会把 EC-VIAI-AV 描述成替换 baseline，而是描述为 baseline 上的可退化扩展。
预期结果：
项目从 VIAI-AV baseline 平滑升级为可追踪、可复跑的 Evidence-Calibrated Multi-Hypothesis VIAI-AV 实验平台。
