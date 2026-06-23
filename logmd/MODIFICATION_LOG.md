# 修改日志

## 2026-04-20 main.py 补全

### 修改内容
1. 将 `main.py` 从占位脚本改为统一入口脚本。
2. 新增 `MODULE_MAP`，支持两类动作：
   - `train` -> `train_whole_sync`
   - `preprocess` -> `Data_loaders.Image_preprocess`
3. 新增参数解析：
   - 位置参数 `action`（默认 `train`）
   - 透传参数 `extra_args`（支持 `--` 后参数原样转发）
4. 新增 `_run_module`，通过 `runpy.run_module` 运行目标模块，并在运行后恢复 `sys.argv`。
5. 新增启动日志输出（目标模块与透传参数）。

### 本次使用的命令行
```bash
sed -n '1,200p' main.py
python -m py_compile main.py
python main.py --help
git diff -- main.py
git status --short main.py && sed -n '1,220p' main.py
```

## 2026-05-01 VIAI 本地跑通测试与云端训练适配

### 背景与目标
1. 本地机器为 RTX 3060 级别显卡，不作为完整论文训练环境，只用于 smoke test。
2. 本地 smoke test 只验证依赖、数据格式、预处理、dataloader、模型 forward/backward 和一次 optimizer update。
3. 正式训练迁移到云端 GPU，云端恢复论文/默认配置后再做长训练和指标核验。
4. 本节同时修正 2026-04-20 日志中的入口描述：当前 `main.py` 实际支持 `prepare-data -> tools.prepare_musices`，不是 `Data_loaders.Image_preprocess`。

### 论文要求核对
1. 数据处理：16kHz audio、STFT frame length 1280、hop size 320、80 mel bins、125Hz-7.6kHz、Mel 归一化到 0-1。
2. 输入形状：4 秒音频、80x200 Mel-spectrogram、对应 50 个视频帧。
3. 缺失区域：训练时随机裁剪 0.4s-1.0s，对应 20-50 个 Mel frame，并用相邻 clean spectrum bins 插值初始化。
4. 视频处理：论文使用 TV-L1 optical flow，flow clipping 到 20 pixels，motion salient crop 后 padding 成正方形，图像和 flow 归一化到 -1 到 1。
5. 训练目标：Adam lr=1e-4；VIAI-AV batch size 16；contrastive synchronization margin `gamma=1`；sync loss 只更新 video encoder。

### 已执行/拟执行代码调整
1. `pyproject.toml`
   - 基础依赖不包含 `torch`，避免云端被本仓库固定到不匹配的 CUDA wheel。
   - 新增 `local-cuda` extra，本地 3060 smoke test 使用 CUDA 版 torch。
   - 云端服务器按实际驱动/CUDA 版本自行安装 PyTorch。
   - 增加训练辅助依赖：`tensorboardX`、`nnmnkwii`。
   - 将 `opencv-python` 替换为 `opencv-contrib-python`，用于 TV-L1 optical flow。
2. `tools/prepare_musices.py`
   - 新增 `--flow-method {tvl1,farneback}`，默认 `tvl1`。
   - 默认使用 OpenCV contrib 的 TV-L1 接口；如环境缺失，提示安装/sync contrib OpenCV。
   - `farneback` 仅保留为非论文 smoke-test fallback。
3. `Data_loaders/audio_loader.py`
   - 移除对 `keras.utils.np_utils` 的依赖，改为本地 `to_categorical()`，减少本地和云端环境负担。
4. `base_options.py` / `train_whole_sync.py`
   - 新增 `--max_train_steps`，用于本地 3060 smoke test 只跑 1 个或少量 optimizer step。
   - 新增 `--sync_margin`，默认 1.0，对齐论文 `gamma=1`。
5. `Models/Whole_Sync_inpainting_modify.py` / `loss_functions.py`
   - `L2ContrastiveLoss` 默认 margin 改为 1.0。
   - sync loss 计算时对 target audio feature 使用 `detach()`，使 sync loss 只推动 video encoder，与论文训练稳定化描述一致。
6. `README.md`
   - 增加本地 3060 smoke-test 命令。
   - 增加云端训练前检查和训练命令。
   - 明确本地降配参数不用于论文指标。

### 本地 3060 smoke-test 命令
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
uv run --extra local-cuda python main.py --help
uv run --extra local-cuda python main.py prepare-data -- --help
uv run --extra local-cuda python -c "import torch, cv2, librosa, nnmnkwii, tensorboardX; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 1 --skip-existing
uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

如 3060 显存仍不足，可临时追加：
```bash
uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --image_size 128
```

注意：任何降低 `image_size`、`load_num` 或其他输入规模的设置，只用于本地功能测试，不用于论文指标。

### 云端训练建议命令
```bash
nvidia-smi
UV_CACHE_DIR=/tmp/uv-cache uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch, cv2; print(torch.cuda.is_available(), torch.version.cuda, hasattr(cv2, 'optflow'))"
.venv/bin/python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
.venv/bin/python main.py split-data -- --data-root data
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

云端训练前需要重新确认：
1. CUDA 版 PyTorch 与云端驱动匹配。
2. `opencv-contrib-python` 的 TV-L1 接口可用。
3. 完整数据已经完成 `process` 和 `split-data`，默认生成 85% train / 5% val / 10% test。
4. checkpoint、TensorBoard 日志、retrieval 指标可以正常写入。
5. 先跑 100-500 step sanity training，再启动长训练。

注意：上面的 `cu121` 只是示例，云端需按服务器实际 CUDA/驱动选择 PyTorch wheel。手动安装云端 PyTorch 后，建议用 `.venv/bin/python` 或 `uv run --no-sync` 运行，避免 `uv run` 自动同步时移除手动安装的云端 torch。

### 已知仍未完整复现的部分
1. 论文中的 shot detection、去除非演奏/黑场片段、裁掉每个视频前 6 秒，当前 pipeline 尚未完整自动化。
2. 论文的 10% fixed test + 5% held-out validation 协议尚未完整接入训练 loader。
3. VIAI-AA' probe loss 和 WaveNet spectrogram-to-audio 端到端评估仍需后续补齐。
4. 本地 smoke test 只证明链路可运行，不代表模型收敛或论文指标。

## 2026-05-01 独立 split 工具与 50 帧对齐

### 修改内容
1. 新增 `tools/split_musices.py`，从 `data/processed/<instrument>/<youtube_id>/` 扫描已处理样本并生成：
   - `train_new_split.txt`
   - `val_new_split.txt`
   - `test_new_split.txt`
2. 新 split 工具默认 `test_size=0.10`、`val_size=0.05`，剩余为 train；支持 `--max-samples` 与 `--allow-empty-eval` 方便本地 smoke test。
3. `main.py` 新增 `split-data -> tools.split_musices`。
4. `prepare_musices.py` 原 `splits` action 仅保留 legacy 兼容，并打印提示推荐使用 `split-data`。
5. `base_options.py` 新增 `train_split_name`、`val_split_name`、`test_split_name`，并将 `image_hope_size` 默认改为 2。
6. `Data_loaders/audio_loader.py` 默认读取 `train+val`；空的 val/test split 会跳过，test 保留给最终评估。
7. 4 秒视频窗口改为默认 50 帧：从 25fps 原始帧中按 `image_hope_size=2` 抽帧，并用时间换算对齐 200 个 mel frames。

### 推荐命令
```bash
uv run --extra local-cuda python main.py split-data -- --data-root data
uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

### split 数据处理测试运行命令
前置条件：`split-data` 只扫描已经完成预处理的样本，必须先存在至少一个完整目录：
`data/processed/<instrument>/<youtube_id>/`，且其中包含 `raw_audio.npy`、`mel.npy`、`image_crop/`、`flow_x_crop/`、`flow_y_crop/`。

1. 检查 split 命令参数：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --help
```

2. 检查当前是否已有完整 processed 样本目录：
```bash
find data/processed -mindepth 2 -maxdepth 2 -type d | head
find data/processed -mindepth 3 -maxdepth 3 \( -name raw_audio.npy -o -name mel.npy -o -name image_crop -o -name flow_x_crop -o -name flow_y_crop \) | head -30
```

3. 如果还没有完整 processed 样本，先处理一批本地已有视频。不要把 `--max-videos` 设得太小，否则可能只扫到缺失 mp4 并直接跳过：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
```

4. 本地 smoke test：只取 1 个可用样本，允许 val/test 为空，用来验证 split 工具、进度条和输出格式：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

5. 检查 smoke split 输出：
```bash
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
sed -n '1,5p' data/train_new_split.txt
```

6. 完整数据划分：默认生成 85% train、5% val、10% test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data
```

7. 完整 split 后再次检查行数：
```bash
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
```

常见报错：
1. `Processed data directory not found: data/processed`：还没有成功运行 `prepare-data -- process`。
2. `No processed samples found. Run the process stage first.`：`data/processed` 存在，但样本不完整，通常缺少 `mel.npy`、`raw_audio.npy` 或 crop 后的 `image_crop/flow_x_crop/flow_y_crop`。
3. `--skip-existing: command not found`：多行命令中反斜杠 `\` 后面有空格，需保证 `\` 是该行最后一个字符。

## 2026-05-01 论文对齐的数据管线、split 和测试入口

### 背景
前一版 `split-data` 只做样本级 train/val/test 划分，尚未处理论文协议中的几个关键点：
1. 论文先做 video-level train/test split，再从对应视频生成样本，避免同一视频片段跨集合泄漏。
2. 论文会裁掉每个视频前 6 秒，并按 shot 处理视频，去除黑场/静音或非演奏片段。
3. 公开 `MUSICES.json` 只有 instrument 和 YouTube ID，没有原论文内部 shot 边界标注，因此本仓库只能用可复现的 OpenCV frame-difference heuristic 近似 shot detection。
4. 原 `main.py` 没有真实测试入口，README 也缺少完整的“数据准备/划分、训练、测试”命令。

### 本次修改内容
1. `tools/prepare_musices.py`
   - 新增论文风格预处理参数：
     - `--trim-start-sec`，默认 6.0，对齐论文裁掉视频开头 6 秒。
     - `--min-segment-sec`，默认 4.0，对齐 4 秒输入窗口。
     - `--shot-detection` / `--no-shot-detection`，默认开启 shot detection。
     - `--shot-diff-threshold`，默认 35.0，用于 OpenCV 灰度帧差 shot boundary 近似。
     - `--black-frame-threshold`、`--max-black-ratio`，用于过滤黑场片段。
     - `--min-audio-rms`，用于过滤近似静音片段。
   - 默认将每个有效 shot 写到：
     - `data/processed/<instrument>/<youtube_id>/shot_000000/`
     - `data/processed/<instrument>/<youtube_id>/shot_000001/`
   - 每个 shot 样本仍生成 `source.wav`、`raw_audio.npy`、`mel.npy`、`image/`、`flow_x/`、`flow_y/`、`image_crop/`、`flow_x_crop/`、`flow_y_crop/`。
   - 保留旧式整视频处理兼容模式：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing --no-shot-detection --trim-start-sec 0
```
   - `prepare-data -- all` 不再自动写 legacy two-way split，完成处理后提示继续运行 `main.py split-data`。

2. `tools/split_musices.py`
   - 递归发现 processed 样本，兼容旧 flat 样本和新 `shot_*` 样本。
   - 对样本按源视频 key `<instrument>/<youtube_id>` 分组，再做 train/val/test split，避免同一视频的不同 shot 泄漏到不同 split。
   - 默认仍为论文比例：85% train、5% val、10% test。
   - split 输出格式保持不变：`sample_dir|mel_path|audio_path|mel_frames`，兼容当前 dataloader。
   - split summary 新增样本数和源视频数，例如 `train=1 samples/1 videos`。
   - 若同一视频目录里同时存在旧 flat 样本和新 `shot_*` 样本，优先使用 shot 样本，跳过父目录旧样本。

3. 测试入口
   - `main.py` 新增 `test -> test_whole_sync`。
   - 新增 `test_whole_sync.py`，加载 `test_new_split.txt`，恢复 checkpoint，仅执行 evaluation，不做 optimizer update。
   - 输出 test reconstruction loss、Mel L1 loss、sync loss，以及 audio-video retrieval 指标。
   - 如果传入的 `--resume_path` 不存在，会在同目录或 `--checkpoint_dir` 下查找最新的 `VIAI-AV_checkpoint_step*.pth.tar`。

4. README
   - 重写数据准备和 dataset split 流程命令。
   - 写入训练 smoke test、云端完整训练、测试集评估命令。
   - 明确说明 shot detection 是公开数据条件下的 OpenCV 近似，不是论文内部人工/原始 shot 标注。
   - 明确当前 `main.py test` 输出当前模型路径下的 loss 和 retrieval 指标；完整 SDR/OPS 仍需要后续补 WaveNet/audio-generation evaluation。

5. 为通过 smoke test 顺手修正的兼容问题
   - `train_whole_sync.py`：`matplotlib` 改为可选导入，缺包时不阻塞训练。
   - `utils/util.py`：`PIL.Image` 改为 `save_image()` 内部懒加载，避免非图片保存路径强依赖 Pillow。
   - `visdom_utils/visualizer.py`：移除顶层 `html` 导入，避免 `display_id=0` 时被 `dominate` 缺包阻塞。
   - `networks/Image_Embedding.py`：将固定 `AvgPool2d(7)` 改为 `AdaptiveAvgPool2d((1, 1))`，修复默认 `image_size=256` 下 `mat1 and mat2 shapes cannot be multiplied` 的视觉 encoder 维度错误。

### README 中记录的完整流程命令
1. 环境检查：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python -c "import torch, cv2, librosa, nnmnkwii, tensorboardX, tqdm; print(torch.__version__, torch.cuda.is_available(), hasattr(cv2, 'optflow'))"
```

2. 数据准备：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- manifest --json data/MUSICES.json --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- stats --json data/MUSICES.json --data-root data
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- download --json data/MUSICES.json --data-root data --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing
```

3. 数据集划分：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data
```

4. 本地 smoke split：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- process --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
wc -l data/train_new_split.txt data/val_new_split.txt data/test_new_split.txt
sed -n '1,5p' data/train_new_split.txt
```

5. 本地训练 smoke test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

6. 云端完整训练：
```bash
nvidia-smi
UV_CACHE_DIR=/tmp/uv-cache uv sync
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -c "import torch, cv2; print(torch.cuda.is_available(), torch.version.cuda, hasattr(cv2, 'optflow'))"
.venv/bin/python main.py prepare-data -- all --json data/MUSICES.json --data-root data --skip-existing
.venv/bin/python main.py split-data -- --data-root data
.venv/bin/python main.py train -- --batch_size 16 --num_workers 4 --display_id 0
```

7. 测试集评估：
```bash
.venv/bin/python main.py test -- --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0
```

### 本次实际验证结果
已通过：
```bash
.venv/bin/python -m py_compile tools/prepare_musices.py tools/split_musices.py main.py test_whole_sync.py
.venv/bin/python main.py prepare-data -- process --help
.venv/bin/python main.py split-data -- --help
.venv/bin/python main.py test -- --help
.venv/bin/python main.py split-data -- --data-root data --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

本地训练 smoke test 已通过：
```bash
.venv/bin/python main.py train -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

关键输出：
```text
Reached local smoke-test max_train_steps=1
Step 1 L1_loss [train] Loss: 0.5894665718078613
VIAI-AV Step 1 [train] EmbeddingL2_loss: 1.0002461671829224
Saved checkpoint: ./checkpoints/VIAI-AV_checkpoint_step000000001.pth.tar
Finished
```

测试入口已通过 smoke 验证。由于本地 smoke split 的 `test_new_split.txt` 为空，使用 `--test_split_name train_new_split.txt` 临时验证测试代码路径：
```bash
.venv/bin/python main.py test -- --resume_path ./checkpoints/VIAI-AV_checkpoint_step000000001.pth.tar --test_split_name train_new_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

关键输出：
```text
[test] losses: reconstruction=1.380574, mel_l1=0.250176, sync=1.061548
[test] Video Retrieval (1 samples): R@1: 100.00, R@5: 100.00, R@10: 100.00, R@50: 100.00, MedR: 1.0, MeanR: 1.0
[test] Audio Retrieval (1 samples): R@1: 100.00, R@5: 100.00, R@10: 100.00, R@50: 100.00, MedR: 1.0, MeanR: 1.0
```

### 仍需后续补齐
1. 当前 shot detection 是 OpenCV frame-difference heuristic，无法等价论文内部原始 shot 边界标注。
2. 黑场/静音过滤是可复现近似规则，尚未完全覆盖论文中的“非演奏片段”人工/规则清洗。
3. `main.py test` 当前评估当前 VIAI 模型路径下的 loss 和 retrieval 指标；论文完整 SDR/OPS 仍需要 WaveNet/audio-generation evaluation 链路。

## 2026-05-01 11.00 pm VIAI-A 第一阶段 audio-only 复现入口

### 背景与目标
根据 `information.md` 中“8.1 第一阶段：复现 VIAI-A”的要求，本次新增一个独立的 audio-only 最小复现链路。该阶段只做 Mel-spectrogram inpainting，不使用视频帧、光流、visual encoder、sync loss、GAN loss 或 WaveNet，目标是先跑通：

```text
MUSICES raw video -> 16kHz mono audio -> 80-bin Mel -> random 0.4s-1.0s mask -> MelEncoder + MelDecoder -> L1 / PSNR / SSIM
```

### 本次修改内容
1. `main.py`
   - 新增 `prepare-viai-a -> tools.prepare_viai_a`。
   - 新增 `train-viai-a -> train_viai_a`。
   - 新增 `test-viai-a -> test_viai_a`。

2. `tools/prepare_viai_a.py`
   - 新增 VIAI-A audio-only 处理脚本。
   - 复用 `tools.prepare_musices` 中的 MUSICES 记录读取、raw video 路径解析、ffmpeg 解析、音频抽取和 Mel 生成函数。
   - 从 `data/raw_videos/<instrument>/<youtube_id>.mp4` 抽取 `source.wav`，并生成 `raw_audio.npy` 与 `mel.npy`。
   - 默认使用论文参数：16kHz mono、STFT length 1280、hop size 320、80 Mel bins、125Hz-7.6kHz。
   - 输出目录复用 `data/processed/<instrument>/<youtube_id>/`，不要求存在 `image_crop/flow_x_crop/flow_y_crop`。

3. `tools/split_musices.py`
   - 新增 `--audio-only`。
   - audio-only 模式只检查 `raw_audio.npy` 和 `mel.npy`。
   - 若未显式指定 split 文件名，默认输出：
     - `train_viai_a_split.txt`
     - `val_viai_a_split.txt`
     - `test_viai_a_split.txt`
   - 仍按 source video key 分组，避免同一原视频跨 train/val/test 泄漏。

4. `Data_loaders/viai_a_loader.py`
   - 新增 VIAI-A 专用 dataloader。
   - 从 split 文件读取 `mel.npy` 与 `raw_audio.npy`。
   - 训练阶段随机裁剪 4 秒窗口，即 200 个 Mel frames。
   - 验证/测试阶段使用居中窗口。
   - batch 只返回 `mel`、`audio`、`path`，不读取图片或光流。

5. `Models/VIAI_A_inpainting.py`
   - 新增 `VIAIAModel`。
   - 只包含 `MelEncoder`、`MelDecoder` 和 Adam optimizer。
   - 使用 `mel_loader.corrupt_mel_spectrogram()` 随机 mask 20-50 个 Mel frames。
   - 优化目标为 `eta1 * full_l1 + missing_l1`。
   - checkpoint 命名为 `VIAI-A_checkpoint_step*.pth.tar`。

6. `train_viai_a.py`
   - 新增 VIAI-A 训练入口。
   - 支持 `--max_train_steps`，用于本地 smoke test。
   - 写入 TensorBoard 标量：total loss、full Mel L1、missing-region Mel L1。

7. `test_viai_a.py`
   - 新增 VIAI-A 测试入口。
   - 自动查找或加载 `VIAI-A_checkpoint_step*.pth.tar`。
   - 在 `test_viai_a_split.txt` 上输出：
     - total loss
     - full Mel L1
     - missing-region Mel L1
     - full PSNR
     - missing-region PSNR
     - SSIM

8. `pyproject.toml`
   - 新增 `scikit-image`，用于 `skimage.metrics.structural_similarity` 计算 SSIM。

9. `README.md`
   - 新增 “Stage 1: VIAI-A Audio-Only” 小节。
   - 写入下载、audio-only 处理、audio-only split、训练 smoke test、完整训练和测试命令。

### README 中记录的 VIAI-A 命令
1. 下载 MUSICES raw videos：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-data -- download --json data/MUSICES.json --data-root data --skip-existing
```

2. 生成 audio-only 样本：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --skip-existing
```

3. 本地 smoke test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --audio-only --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

4. 完整 audio-only split / train / test：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py split-data -- --data-root data --audio-only
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py train-viai-a -- --batch_size 16 --num_workers 4 --display_id 0
UV_CACHE_DIR=/tmp/uv-cache uv run --extra local-cuda python main.py test-viai-a -- --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0
```

### 后续验证计划
1. 静态检查：
```bash
.venv/bin/python -m py_compile main.py tools/prepare_viai_a.py tools/split_musices.py train_viai_a.py test_viai_a.py Models/VIAI_A_inpainting.py Data_loaders/viai_a_loader.py
```

2. 数据准备 smoke test：
```bash
.venv/bin/python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --max-videos 100 --skip-existing
.venv/bin/python main.py split-data -- --data-root data --audio-only --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

3. 训练 smoke test：
```bash
.venv/bin/python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

4. 测试 smoke test：
```bash
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

### 本次实际验证结果
已通过静态检查：
```bash
.venv/bin/python -m py_compile main.py tools/prepare_viai_a.py tools/split_musices.py train_viai_a.py test_viai_a.py Models/VIAI_A_inpainting.py Data_loaders/viai_a_loader.py
```

新入口 help 已通过：
```bash
.venv/bin/python main.py --help
.venv/bin/python main.py prepare-viai-a -- --help
.venv/bin/python main.py split-data -- --help
```

audio-only 数据准备 smoke test 已通过：
```bash
.venv/bin/python main.py prepare-viai-a -- --json data/MUSICES.json --data-root data --max-videos 4 --skip-existing
```

关键输出：
```text
[prepare_viai_a] skipped existing: accordion/yy2vL2RUiPI -> data/processed/accordion/yy2vL2RUiPI
[prepare_viai_a] processed: accordion/A2p8VW61RGc mel_frames=4517 -> data/processed/accordion/A2p8VW61RGc
[prepare_viai_a] summary: missing=2, processed=1, skipped_existing=1
```

audio-only split smoke test 已通过：
```bash
.venv/bin/python main.py split-data -- --data-root data --audio-only --max-samples 1 --test-size 0 --val-size 0 --allow-empty-eval
```

关键输出：
```text
[split_musices] wrote splits: train=1 samples/1 videos (data/train_viai_a_split.txt), val=0 samples/0 videos (data/val_viai_a_split.txt), test=0 samples/0 videos (data/test_viai_a_split.txt)
```

VIAI-A 训练 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-a -- --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

关键输出：
```text
Reached VIAI-A smoke-test max_train_steps=1
[VIAI-A train] loss=0.591076 full_l1=0.288049 missing_l1=0.303027
Saved VIAI-A checkpoint: ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar
Finished VIAI-A training
```

VIAI-A 测试 smoke test 已通过。由于 smoke split 的 `test_viai_a_split.txt` 为空，本次临时使用 `train_viai_a_split.txt` 验证测试路径：
```bash
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.320611 mel_l1_full=0.152091 mel_l1_missing=0.168520 psnr_full=14.181 psnr_missing=13.249 ssim=0.0129
```

依赖锁更新已通过：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv lock
```

关键输出：
```text
Added scikit-image v0.25.2, v0.26.0
```

## 2026-05-02 VIAI-A 训练实时监督增强

### 修改内容
1. 新增 `utils/viai_a_metrics.py`：
   - 抽出 VIAI-A normalized Mel `[0, 1]` 指标计算。
   - 统一计算 full PSNR、missing-region PSNR、SSIM。
   - 保留 `skimage.metrics.structural_similarity` 缺失时的 SSIM fallback。
   - 提供 TensorBoard Mel 图像写入工具：masked input、prediction、target、abs error。
   - 如果当前环境缺少 `Pillow`，Mel 图像写入会给出一次 warning，但不打断训练。
2. `base_options.py`
   - 新增 `--metric_freq`，默认 100，用于训练阶段间隔计算 SSIM。
   - 新增 `--tb_image_freq`，默认 500，用于间隔写入 Mel 对比图。
   - 新增 `--tb_image_count`，默认 4，限制每次写入 TensorBoard 的样本数量。
3. `train_viai_a.py`
   - `tqdm` 实时显示 loss、full/missing PSNR、mask blank length，并在间隔 step 显示 SSIM。
   - TensorBoard 继续写入 loss，同时新增 PSNR、SSIM、blank frames、learning rate。
   - 训练按 `--tb_image_freq` 写入 Mel 对比图；验证每轮第一批写入一组 Mel 对比图。
4. `test_viai_a.py`
   - 删除本地重复 PSNR/SSIM 实现，改用 `utils.viai_a_metrics`。
   - 保持原输出字段兼容：`mel_l1_full`、`mel_l1_missing`、`psnr_full`、`psnr_missing`、`ssim`。
5. `README.md`
   - VIAI-A 训练章节新增 TensorBoard 启动命令。
   - 记录 `--metric_freq`、`--tb_image_freq`、`--tb_image_count` 常用监督参数。
6. `pyproject.toml`
   - 新增 `pillow` 依赖；`tensorboardX` 写 image panel 时需要 `PIL`。
   - 新增 `tensorboard` 依赖；提供 `tensorboard --logdir` CLI 和 event tag 检查工具。

### 验证命令
```bash
python -m py_compile train_viai_a.py test_viai_a.py utils/viai_a_metrics.py
python main.py train-viai-a -- --data_root data --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
tensorboard --logdir checkpoints/events_viai_a
python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

### 待记录 smoke test 关键输出
```text
[VIAI-A train] step=... loss=... full_l1=... missing_l1=... psnr=... psnr_missing=... ssim=...
TensorBoard event log path: ./checkpoints/events_viai_a
[VIAI-A test] samples=... loss=... mel_l1_full=... mel_l1_missing=... psnr_full=... psnr_missing=... ssim=...
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile train_viai_a.py test_viai_a.py utils/viai_a_metrics.py
```

依赖锁和本地环境同步已通过：
```bash
UV_CACHE_DIR=/tmp/uv-cache uv lock
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra local-cuda
```

VIAI-A 训练 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0
```

关键输出：
```text
[VIAI-A train] epoch=1 ... loss=0.5728 ... psnr=9.11 psnr_miss=9.31 ssim=-0.0016 step=1
Reached VIAI-A smoke-test max_train_steps=1
[VIAI-A train] loss=0.572797 full_l1=0.292290 missing_l1=0.280507 psnr=9.112 psnr_missing=9.310 ssim=-0.001626
Saved VIAI-A checkpoint: ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar
Finished VIAI-A training
```

TensorBoard event tag 检查已通过：
```text
scalars ['train/blank_frames', 'train/loss_full_l1', 'train/loss_missing_l1', 'train/loss_total', 'train/lr', 'train/psnr_full', 'train/psnr_missing', 'train/ssim_full']
images ['train/mel_abs_error', 'train/mel_input_masked', 'train/mel_prediction', 'train/mel_target']
```

TensorBoard CLI 可用：
```bash
.venv/bin/tensorboard --version
```

关键输出：
```text
2.20.0
```

VIAI-A 测试入口已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.356585 mel_l1_full=0.163478 mel_l1_missing=0.193108 psnr_full=13.548 psnr_missing=12.252 ssim=0.1417
```

## 2026-05-02 VIAI-A 测试结果 JSON/CSV 持久化

### 修改内容
1. `base_options.py`
   - 新增 `--results_dir`，默认 `./checkpoints/viai_a_test_results`。
2. `test_viai_a.py`
   - 每次测试后保存 checkpoint 专属 JSON：
     - `VIAI-A_step000001000_test.json`
   - 同步维护 CSV 总表：
     - `VIAI-A_test_summary.csv`
   - CSV 按 `checkpoint_step` 升序排列。
   - 同一个 checkpoint 重复测试会覆盖旧行，不会追加重复行。
   - 记录字段包括 checkpoint path/step、global step/epoch、test split、样本数、loss、L1、PSNR、SSIM。
3. `README.md`
   - VIAI-A 测试命令增加 `--results_dir` 示例。
   - 说明逐个测试多个 checkpoint 后可用 `VIAI-A_test_summary.csv` 横向比较。

### 验证命令
```bash
.venv/bin/python -m py_compile test_viai_a.py base_options.py
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python main.py test-viai-a -- --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

### 待记录 smoke test 关键输出
```text
[VIAI-A test] wrote json: checkpoints/viai_a_test_results/VIAI-A_step000000001_test.json
[VIAI-A test] wrote summary csv: checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile test_viai_a.py base_options.py
```

重复测试同一个 checkpoint 已通过，CSV 未产生重复行：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
wc -l checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
```

关键输出：
```text
[VIAI-A test] wrote json: checkpoints/viai_a_test_results/VIAI-A_step000000001_test.json
[VIAI-A test] wrote summary csv: checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
2 checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
```

本地 smoke test 使用随机 mask，同一 checkpoint 重测的 loss/PSNR/SSIM 可能略有变化；CSV 会保留最近一次该 checkpoint 的结果。

## 2026-05-02 VIAI-A 测试 Mel 图片保存

### 修改内容
1. `utils/viai_a_metrics.py`
   - 新增 `save_mel_comparison_png()`，保存单个样本四联图：
     - masked input
     - prediction
     - target
     - abs error
   - 新增 `save_mel_comparison_batch()`，按 batch 批量保存测试样本图片。
   - PNG 使用 normalized Mel `[0, 1]` 映射到灰度 `[0, 255]`。
2. `test_viai_a.py`
   - 测试时为每个样本保存一张 Mel 对比 PNG。
   - 输出目录为：
     - `<results_dir>/mel-image/stepXXXXXXXXX/`
   - 文件名包含全局样本序号和安全化后的 split sample path。
   - 保留原有 JSON/CSV 指标输出不变。
3. `README.md`
   - VIAI-A 测试结果说明中新增 Mel 图片输出路径和四联图内容。

### 验证命令
```bash
.venv/bin/python -m py_compile test_viai_a.py utils/viai_a_metrics.py
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python -c "from PIL import Image; import glob; path=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; Image.open(path).verify(); print(path)"
```

### 待记录 smoke test 关键输出
```text
[VIAI-A test] wrote mel images: checkpoints/viai_a_test_results/mel-image/step000000001
checkpoints/viai_a_test_results/mel-image/step000000001/000000_*.png
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile test_viai_a.py utils/viai_a_metrics.py
```

VIAI-A 测试 smoke test 已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.328249 mel_l1_full=0.163457 mel_l1_missing=0.164792 psnr_full=13.550 psnr_missing=13.643 ssim=0.1415
[VIAI-A test] wrote json: checkpoints/viai_a_test_results/VIAI-A_step000000001_test.json
[VIAI-A test] wrote summary csv: checkpoints/viai_a_test_results/VIAI-A_test_summary.csv
[VIAI-A test] wrote mel images: checkpoints/viai_a_test_results/mel-image/step000000001
```

PNG 输出和 PIL 校验已通过：
```bash
find checkpoints/viai_a_test_results/mel-image/step000000001 -maxdepth 1 -type f -name '*.png'
.venv/bin/python -c "from PIL import Image; import glob; path=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; Image.open(path).verify(); print(path)"
```

关键输出：
```text
000000_processed_accordion_A2p8VW61RGc.png
checkpoints/viai_a_test_results/mel-image/step000000001/000000_processed_accordion_A2p8VW61RGc.png
```

## 2026-05-03 VIAI-A 测试 Mel PNG 改为 RGB 热力图

### 修改内容
1. `utils/viai_a_metrics.py`
   - `_mel_to_uint8_image()` 从单通道灰度输出改为 `H x W x 3` RGB 输出。
   - 使用内置 magma-like colormap，不新增 `matplotlib` 等依赖。
   - `save_mel_comparison_png()` 改为 `Image.new("RGB", ...)` 和 `Image.fromarray(..., mode="RGB")`。
2. `README.md`
   - 明确 `mel-image/stepXXXXXXXXX/` 下保存的是 RGB 热力图四联图。

### 验证命令
```bash
.venv/bin/python -m py_compile utils/viai_a_metrics.py test_viai_a.py
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python -c "from PIL import Image; import glob; p=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; img=Image.open(p); print(img.mode, img.size)"
```

### 待记录 smoke test 关键输出
```text
RGB (...)
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile utils/viai_a_metrics.py test_viai_a.py
```

VIAI-A 测试 smoke test 已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

PNG mode 检查已通过：
```bash
.venv/bin/python -c "from PIL import Image; import glob; p=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; img=Image.open(p); print(img.mode, img.size, p)"
```

关键输出：
```text
RGB (812, 98) checkpoints/viai_a_test_results/mel-image/step000000001/000000_processed_accordion_A2p8VW61RGc.png
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile utils/viai_a_metrics.py test_viai_a.py
```

VIAI-A 测试 smoke test 已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
```

PNG mode 检查已通过：
```bash
.venv/bin/python -c "from PIL import Image; import glob; p=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; img=Image.open(p); print(img.mode, img.size, p)"
```

关键输出：
```text
RGB (812, 98) checkpoints/viai_a_test_results/mel-image/step000000001/000000_processed_accordion_A2p8VW61RGc.png
```

## 2026-05-03 VIAI-A 测试 Mel 四联图面板调整

### 修改内容
1. `utils/viai_a_metrics.py`
   - `save_mel_comparison_png()` 的四联图从：
     - masked input / prediction / target / abs error
   - 改为：
     - masked / interpolated / prediction / groundtruth
   - `masked` 使用 `missing_mask` 将缺失区域置黑，便于肉眼确认缺失区。
   - `interpolated` 保留当前模型实际输入，即边界插值后的 corrupted Mel。
2. `test_viai_a.py`
   - 调用 `save_mel_comparison_batch()` 时传入 `model.missing_mask`。
3. `README.md`
   - 更新测试 Mel PNG 四联图说明。

### 验证命令
```bash
.venv/bin/python -m py_compile utils/viai_a_metrics.py test_viai_a.py
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path ./checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir checkpoints/viai_a_test_results
.venv/bin/python -c "from PIL import Image; import glob; p=glob.glob('checkpoints/viai_a_test_results/mel-image/step000000001/*.png')[0]; img=Image.open(p); print(img.mode, img.size, p)"
```

### 待记录 smoke test 关键输出
```text
RGB (...)
```

## 2026-05-03 VIAI-A 第二阶段 PatchGAN

### 背景与目标
根据 `information.md` 中“8.2 第二阶段：加入 PatchGAN”的要求，本次在 VIAI-A audio-only 基础上加入可选 PatchGAN 训练。默认 `train-viai-a` 仍保持 8.1 的 L1-only baseline；显式传入 `--use_gan` 后启用判别器和 GAN loss，用于提升生成 Mel-spectrogram 的局部纹理真实感。

### 本次修改内容
1. `base_options.py`
   - 新增 `--use_gan`，默认关闭。
   - 继续复用 `--beta_gan`、`--lambda_recon` 和 `--recon_decay_*`。
2. `Models/VIAI_A_inpainting.py`
   - `--use_gan` 开启时实例化 `Discriminator_Networks.MelDiscriminator()`、`GANLoss(use_lsgan=False)` 和 `optimizer_D`。
   - 将 reconstruction loss 拆为 `loss_recon = eta1 * loss_full_l1 + loss_missing_l1`。
   - PatchGAN 训练目标初版误写为 `loss_total = lambda_recon * loss_recon + beta_gan * loss_G_GAN`，后续已按论文第 4 页式 (3) 修正为 `loss_total = loss_G_GAN + beta_gan * loss_recon`。
   - 判别器目标为 `loss_D = 0.5 * (loss_D_real + loss_D_fake)`，fake 分支使用 `mel_pred.detach()`。
   - checkpoint 新增可选 `netD`、`optimizer_D`、`use_gan` 字段。
   - 从 8.1 checkpoint 热启动时允许缺少 `netD/optimizer_D`，判别器随机初始化。
3. `train_viai_a.py`
   - `--use_gan` 且未传 `--name` 时默认使用 `VIAI-A-PatchGAN`。
   - `--use_gan` 且未传 `--log_event_path` 时默认写到 `checkpoints/events_viai_a_patchgan`。
   - 修复 resume 后 `global_step/global_epoch` 被重置的问题。
   - 训练日志和 TensorBoard 增加 `loss_recon`、`loss_g_gan`、`loss_d`、`loss_d_real`、`loss_d_fake`、`eta1`。
4. `test_viai_a.py`
   - `--use_gan` 时测试也默认使用 `VIAI-A-PatchGAN`。
   - JSON/CSV 新增 `use_gan`、`loss_recon`、`loss_g_gan`、`loss_d`、`eta1`、`beta_gan`、`lambda_recon`。
5. `README.md`
   - 新增“第二阶段：加入 PatchGAN”操作流程。
   - 保留 8.1 baseline 命令，并补充 stage2 smoke train、正式 train、test 命令。

### README 中记录的第二阶段命令
1. 8.1 baseline 训练不变：
```bash
python main.py train-viai-a -- --batch_size 16 --num_workers 4 --display_id 0
```

2. 从 8.1 checkpoint 热启动 stage2 smoke test：
```bash
python main.py train-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --resume \
  --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar \
  --reset_optimizer \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 2 \
  --display_id 0
```

3. Stage2 正式训练：
```bash
python main.py train-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --resume \
  --resume_path checkpoints/VIAI-A_checkpoint_step000001000.pth.tar \
  --reset_optimizer \
  --batch_size 16 \
  --num_workers 4 \
  --beta_gan 0.1 \
  --checkpoint_interval 1000 \
  --print_freq 100 \
  --display_id 0
```

4. Stage2 测试：
```bash
python main.py test-viai-a -- \
  --use_gan \
  --name VIAI-A-PatchGAN \
  --resume_path checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar \
  --batch_size 16 \
  --num_workers 4 \
  --display_id 0 \
  --results_dir checkpoints/viai_a_patchgan_test_results
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile base_options.py Models/VIAI_A_inpainting.py train_viai_a.py test_viai_a.py
```

8.1 baseline 回归 smoke test 已通过。为避免覆盖仓库内 checkpoint，本次验证写入 `/tmp/viai_patchgan_smoke_baseline`：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --checkpoint_dir /tmp/viai_patchgan_smoke_baseline --log_event_path /tmp/viai_patchgan_smoke_baseline/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A train] step=1 loss=0.590856 full_l1=0.293740 missing_l1=0.297116 eta1=1.000000 psnr=9.163 psnr_missing=9.029 ssim=0.033071
Saved VIAI-A checkpoint: /tmp/viai_patchgan_smoke_baseline/VIAI-A_checkpoint_step000000001.pth.tar
```

PatchGAN 从旧 8.1 checkpoint 热启动 smoke test 已通过，验证了旧 checkpoint 缺少 `netD/optimizer_D` 时不会报错：
```bash
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_patchgan_smoke_gan --log_event_path /tmp/viai_patchgan_smoke_gan/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A] resumed checkpoint step=1 epoch=0
[VIAI-A train] step=2 loss=0.786060 full_l1=0.358661 missing_l1=0.349788 eta1=0.999895 recon=0.708411 g_gan=0.776496 d=0.714494 psnr=7.454 psnr_missing=7.864
Saved VIAI-A checkpoint: /tmp/viai_patchgan_smoke_gan/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar
```

PatchGAN 测试入口已通过：
```bash
.venv/bin/python main.py test-viai-a -- --use_gan --data_root data --resume_path /tmp/viai_patchgan_smoke_gan/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_patchgan_smoke_results
```

关键输出：
```text
[VIAI-A test] samples=1 loss=0.400465 recon=0.332187 g_gan=0.682783 d=0.692535 eta1=1.000000 mel_l1_full=0.168928 mel_l1_missing=0.163259 psnr_full=13.279 psnr_missing=13.602 ssim=0.1412
[VIAI-A test] wrote json: /tmp/viai_patchgan_smoke_results/VIAI-A-PatchGAN_step000000002_test.json
[VIAI-A test] wrote summary csv: /tmp/viai_patchgan_smoke_results/VIAI-A-PatchGAN_test_summary.csv
```

测试 JSON 字段检查已通过，包含：
```text
use_gan=true, loss_recon, loss_g_gan, loss_d, eta1, beta_gan, lambda_recon
```

TensorBoard event tag 检查已通过，PatchGAN 标量包含：
```text
train/loss_recon, train/loss_g_gan, train/loss_d, train/loss_d_real, train/loss_d_fake, train/eta1
```

PatchGAN checkpoint 字段检查已通过：
```text
netD, optimizer_D, use_gan, global_step, global_epoch
```

## 2026-05-03 VIAI-A PatchGAN loss 权重方向修正

### 背景
论文第 4 页式 (3) 写作：
```text
L_total^a = L_Gen^a = L_GAN^a + beta * L_re^a
```

因此 VIAI-A PatchGAN 中 β 应当乘在 reconstruction loss 上，而不是乘在 GAN loss 上。当前代码保留 `--beta_gan` 这个历史参数名，但在 VIAI-A PatchGAN 中它对应论文公式里的 β。

### 修改内容
1. `Models/VIAI_A_inpainting.py`
   - 将 `--use_gan` 分支的生成器总损失从：
```python
self.loss_total = lambda_recon * self.loss_recon + beta_gan * self.loss_G_GAN
```
   - 修正为：
```python
self.loss_total = self.loss_G_GAN + beta_gan * self.loss_recon
```
   - L1-only baseline 分支仍保持 `self.loss_total = self.loss_recon`。
2. `README.md`
   - 第二阶段公式改为 `loss_total = loss_g_gan + beta_gan * loss_recon`。
   - 移除 VIAI-A PatchGAN 训练命令中的 `--lambda_recon 1.0`。
   - 增加 `--beta_gan` 在 VIAI-A PatchGAN 中对应论文 β 的说明。

### 验证命令
```bash
.venv/bin/python -m py_compile Models/VIAI_A_inpainting.py train_viai_a.py test_viai_a.py
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_patchgan_formula_smoke --log_event_path /tmp/viai_patchgan_formula_smoke/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

### 待记录 smoke test 关键输出
已通过 PatchGAN smoke test：
```bash
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_patchgan_formula_smoke --log_event_path /tmp/viai_patchgan_formula_smoke/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A] resumed checkpoint step=1 epoch=0
[VIAI-A train] step=2 loss=0.683178 full_l1=0.264623 missing_l1=0.249545 eta1=0.999895 recon=0.514140 g_gan=0.631764 d=0.701329 psnr=9.798 psnr_missing=10.176
Saved VIAI-A checkpoint: /tmp/viai_patchgan_formula_smoke/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar
```

公式核对：
```text
0.631764 + 0.1 * 0.514140 = 0.683178
```

8.1 baseline 回归 smoke test 已通过，不传 `--use_gan` 时日志不包含 `g_gan/d`，仍然只使用 reconstruction loss：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --checkpoint_dir /tmp/viai_patchgan_formula_baseline --log_event_path /tmp/viai_patchgan_formula_baseline/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A train] step=1 loss=0.560637 full_l1=0.289882 missing_l1=0.270755 eta1=1.000000 psnr=9.138 psnr_missing=9.589 ssim=0.009980
Saved VIAI-A checkpoint: /tmp/viai_patchgan_formula_baseline/VIAI-A_checkpoint_step000000001.pth.tar
```

## 2026-05-03 VIAI-AV 第三阶段视频分支复现

### 背景与目标
根据 `information.md` 中“8.3 第三阶段：加入视频分支 VIAI-AV”的要求，本次只补齐视频条件分支，不加入 8.4 的 `contrastive sync loss`、`VIAI-AA' probe loss`、`η2(t)` 或 WaveNet。目标链路为：

```text
视频抽帧 -> TV-L1 光流 -> Image ResNet18 + Flow ResNet18 -> Efuse 时间融合 -> MelDecoderImage 融合解码
```

训练默认从第二阶段 `VIAI-A-PatchGAN` checkpoint 热启动音频侧权重，视觉分支随机初始化。

### 本次修改内容
1. `main.py`
   - 新增 `train-viai-av -> train_viai_av`。
   - 新增 `test-viai-av -> test_viai_av`。
2. `base_options.py`
   - 新增 `--init_from_viai_a`，用于从 `VIAI-A` 或 `VIAI-A-PatchGAN` checkpoint 初始化 Stage3 音频侧权重。
   - 未显式传入时，`train-viai-av` 只自动寻找 `VIAI-A-PatchGAN_checkpoint_step*.pth.tar`；找不到会报错，不静默随机初始化。
3. `Models/VIAI_AV_inpainting.py`
   - 新增 `VIAIAVModel`，包含 `MelEncoder`、`ImageEmbedding`、`MelDecoderImage`、`MelDiscriminator`。
   - 仅计算 reconstruction + GAN loss：
```text
loss_recon = eta1(t) * full_l1 + missing_l1
loss_total = loss_g_gan + beta_gan * loss_recon
loss_d = 0.5 * (loss_d_real + loss_d_fake)
```
   - 不计算 sync/probe loss。
   - 从 VIAI-A checkpoint 部分加载 `Mel_Encoder`、`Mel_Decoder` 和可用的 `netD`。
   - 使用 audio decoder stem 初始化 `MelDecoderImage` 的融合 stem。
   - AV checkpoint 保存 `VideoEncoder`、`Mel_Encoder`、`Mel_Decoder`、`netD`、optimizer 和 step/epoch。
4. `train_viai_av.py`
   - 新增第三阶段训练入口，不使用旧 `Visualizer`，避免写入固定 `checkpoints/VIAI-AV/loss_log.txt`。
   - TensorBoard 写入 `loss_total/loss_recon/loss_g_gan/loss_d/eta1/PSNR/SSIM/blank_frames/lr` 和 Mel 对比图。
5. `test_viai_av.py`
   - 新增第三阶段测试入口。
   - 输出 normalized Mel `[0, 1]` 上的 L1、PSNR、SSIM、GAN/reconstruction loss。
   - 写入 JSON、CSV 和 Mel RGB 热力图四联图。
6. `networks/Image_Embedding.py`
   - 修复 Efuse 中 `BN/ReLU` 调用未赋值的问题，两层 stride-2 1D conv 现在实际执行 `conv -> BN -> ReLU -> conv -> BN -> ReLU`。
7. `networks/New_Inpainting_Networks.py`
   - `MelDecoderImage.init_deconv_1_1_1()` 同步复制 bias，便于从 Stage2 decoder 初始化融合 stem。
8. `Data_loaders/audio_loader.py`
   - 修复 AV window 起点采样只受视频帧数约束的问题。
   - 现在起点同时受 `mel.npy` 长度约束，避免 test 阶段随机采到视频末端导致 Mel/audio 越界。
9. `utils/util.py`
   - 旧 `save_inpainting_checkpoint()` 也会在模型存在 `VideoEncoder` 时保存视觉分支，避免旧 `main.py train` 路径丢失视频 encoder。
10. `README.md`
   - 新增“第三阶段：加入视频分支 VIAI-AV”完整操作流程。
11. `checkpoints/VIAI-AV/loss_log.txt`
   - 清理计划阶段探测命令追加的两行 smoke 日志，工作树不保留该探测副作用。

### README 中记录的第三阶段命令
```bash
python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing
python main.py split-data -- --data-root data
python main.py train-viai-av -- --data_root data --init_from_viai_a checkpoints/VIAI-A-PatchGAN_checkpoint_step000002000.pth.tar --batch_size 16 --num_workers 4 --beta_gan 0.1 --checkpoint_interval 1000 --print_freq 100 --display_id 0
python main.py test-viai-av -- --resume_path checkpoints/VIAI-AV_checkpoint_step000001000.pth.tar --batch_size 16 --num_workers 4 --display_id 0 --results_dir checkpoints/viai_av_test_results
```

### 本次实际验证结果
静态检查已通过：
```bash
.venv/bin/python -m py_compile main.py train_viai_av.py test_viai_av.py Models/VIAI_AV_inpainting.py Data_loaders/audio_loader.py networks/Image_Embedding.py networks/New_Inpainting_Networks.py utils/util.py
```

AV dataloader 与 Efuse 形状检查已通过：
```text
mel (1, 80, 200)
video (1, 50, 3, 256, 256)
flow (1, 50, 2, 256, 256)
efuse (1, 256, 1, 13)
```

为 Stage3 smoke test 临时生成 `/tmp` 下的 Stage2 checkpoint 已通过：
```bash
.venv/bin/python main.py train-viai-a -- --use_gan --data_root data --checkpoint_dir /tmp/viai_av_stage3_patchgan_init --log_event_path /tmp/viai_av_stage3_patchgan_init/events --resume --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --reset_optimizer --batch_size 1 --num_workers 0 --max_train_steps 2 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-A train] step=2 loss=0.749014 full_l1=0.275098 missing_l1=0.233697 eta1=0.999895 recon=0.508765 g_gan=0.698137 d=0.710336 psnr=9.551 psnr_missing=10.769
Saved VIAI-A checkpoint: /tmp/viai_av_stage3_patchgan_init/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar
```

VIAI-AV 训练 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-av -- --data_root data --init_from_viai_a /tmp/viai_av_stage3_patchgan_init/VIAI-A-PatchGAN_checkpoint_step000000002.pth.tar --checkpoint_dir /tmp/viai_av_smoke --log_event_path /tmp/viai_av_smoke/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-AV] loaded 30 tensors into Mel_Encoder; skipped_shape=0
[VIAI-AV] loaded 101 tensors into MelDecoderImage; skipped_shape=0
[VIAI-AV] initialized MelDecoderImage fusion stem from audio decoder stem
[VIAI-AV] loaded 25 tensors into MelDiscriminator; skipped_shape=0
[VIAI-AV train] step=1 loss=0.810526 recon=0.521095 full_l1=0.259729 missing_l1=0.261366 g_gan=0.758417 d=0.693089 eta1=1.000000 psnr=10.241 psnr_missing=10.299 ssim=0.010216
Saved VIAI-AV checkpoint: /tmp/viai_av_smoke/VIAI-AV_checkpoint_step000000001.pth.tar
```

VIAI-AV checkpoint 字段检查已通过，包含：
```text
Mel_Decoder, Mel_Encoder, VideoEncoder, global_epoch, global_step, netD, optimizer_D, optimizer_G, stage, use_gan
```

VIAI-AV 测试入口已通过。由于本地 smoke split 的 `test_new_split.txt` 为空，本次临时使用 `train_new_split.txt` 验证测试路径：
```bash
.venv/bin/python main.py test-viai-av -- --data_root data --resume_path /tmp/viai_av_smoke/VIAI-AV_checkpoint_step000000001.pth.tar --test_split_name train_new_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_av_smoke_results
```

关键输出：
```text
[VIAI-AV test] samples=1 loss=0.743333 recon=0.378604 g_gan=0.705472 d=0.690175 eta1=1.000000 mel_l1_full=0.221759 mel_l1_missing=0.156844 psnr_full=11.780 psnr_missing=14.807 ssim=0.1668
[VIAI-AV test] wrote json: /tmp/viai_av_smoke_results/VIAI-AV_step000000001_test.json
[VIAI-AV test] wrote summary csv: /tmp/viai_av_smoke_results/VIAI-AV_test_summary.csv
[VIAI-AV test] wrote mel images: /tmp/viai_av_smoke_results/mel-image/step000000001
```

VIAI-A 第一/第二阶段入口回归已通过：
```bash
.venv/bin/python main.py train-viai-a -- --data_root data --checkpoint_dir /tmp/viai_av_stage3_regress_a --log_event_path /tmp/viai_av_stage3_regress_a/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_av_stage3_regress_a_results
```

关键输出：
```text
[VIAI-A train] step=1 loss=0.569536 full_l1=0.269511 missing_l1=0.300025 eta1=1.000000 psnr=9.682 psnr_missing=9.013 ssim=0.018781
[VIAI-A test] samples=1 loss=0.337543 recon=0.337543 g_gan=0.000000 d=0.000000 eta1=1.000000 mel_l1_full=0.163488 mel_l1_missing=0.174055 psnr_full=13.548 psnr_missing=12.967 ssim=0.1415
```

## 2026-05-08 VIAI-AV 坏 clip 跳过与记录

### 修改内容
1. 新增 `utils/av_sample_validation.py`，统一计算 AV 训练窗口需求，并写入 `viai_av_bad_samples.csv`。
2. `tools/prepare_musices.py`：当实际抽取视觉帧数少于 `--visual-frame-count` 时跳过该 clip；`--skip-existing` 遇到已存在但不合格的样本会重新处理。
3. `tools/split_musices.py`：AV split 生成前过滤短帧/短音频/短 Mel 样本，并把排除原因追加到 `$DATA_ROOT/viai_av_bad_samples.csv`。
4. `Data_loaders/audio_loader.py` / `train_viai_av.py`：训练时发现旧 split 残留坏样本会记录并跳过；`--strict_av_samples` 可恢复 fail-fast。
5. `base_options.py`：新增 `--bad_sample_log` 和 `--strict_av_samples`。

### 验证命令
```bash
.venv/bin/python -m py_compile Data_loaders/audio_loader.py tools/split_musices.py tools/prepare_musices.py train_viai_av.py base_options.py utils/av_sample_validation.py
.venv/bin/python main.py split-data -- --data-root /tmp/viai_bad_data_liu8w1ar --test-size 0 --val-size 0 --allow-empty-eval
.venv/bin/python main.py train-viai-av -- --data_root /tmp/viai_bad_data_liu8w1ar --train_split_name train_mixed_split.txt --init_from_viai_a checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --checkpoint_dir /tmp/viai_bad_train_smoke --log_event_path /tmp/viai_bad_train_smoke/events --batch_size 2 --num_workers 0 --max_train_steps 1 --print_freq 1 --display_id 0
```

## 2026-05-13 VIAI-AV 第四阶段 sync/probe 复现

### 修改内容
1. `Models/VIAI_AV_inpainting.py`
   - 在现有 `VIAIAVModel` 中默认启用第四阶段。
   - 新增 batch 内 `L2ContrastiveLoss` sync loss：`fa_t = Ea(s_t)` 与 `fv = Ev(video, flow)` 先 L2 normalize，同 index 为正样本，batch 内不同 index 为负样本，`sync_margin=1.0`。
   - sync loss 对 `fa_t` 使用 `detach()`，保证该项只更新 `VideoEncoder`。
   - 新增 VIAI-AA' probe branch：`s_aa' = Gav(Ea(s_i), Ea(s_t))`，复用同一个 `MelDecoderImage`、`MelDiscriminator`，不新增网络参数。
   - 总损失改为：
     ```text
     L_total = L_av_gen + lambda_sync * L_sync + lambda_probe * eta2(t) * L_probe_gen
     L_av_gen = L_av_gan + beta_gan * L_av_recon
     L_probe_gen = L_probe_gan + beta_gan * L_probe_recon
     ```
   - 判别器 fake loss 在 probe 开启时对主 AV fake 和 AA' fake 取平均。
   - checkpoint `stage` 改为 `VIAI-AV-stage4-sync-probe`，并写入 `enable_sync_loss` / `enable_probe_loss`。

2. `base_options.py`
   - 新增 `--disable_sync_loss`、`--disable_probe_loss` 用于 ablation 或退回第三阶段式损失。
   - 新增 `--lambda_probe`，默认 `1.0`。
   - 新增 `--probe_decay_base`、`--probe_decay_interval`、`--probe_decay_floor`；未显式传入时沿用 `sync_decay_*` 作为兼容默认值。

3. `train_viai_av.py`
   - tqdm、stdout、TensorBoard 增加 `sync`、`probe`、`eta2`、`probe_full_l1`、`probe_missing_l1`、`probe_g_gan` 等监控项。
   - `--disable_sync_loss --disable_probe_loss` 下 `sync=0`、`probe=0`，总损失退回主 AV generator loss。

4. `test_viai_av.py`
   - JSON/CSV 增加 sync/probe loss、`eta2`、probe L1、audio->video / video->audio retrieval 指标。
   - 测试仍只保存主 AV 输出的 Mel 对比图，AA' probe 只作为 loss/metric 输出。

5. `README.md`
   - 新增“第四阶段：加入 sync loss 和 probe loss”说明，记录默认启用、ablation 参数、推荐命令和输出字段。

### 验证命令
静态检查已通过：
```bash
python3 -m py_compile base_options.py train_viai_av.py test_viai_av.py Models/VIAI_AV_inpainting.py loss_functions.py utils/util.py
```

第四阶段训练 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-av -- --data_root data --init_from_viai_a checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --checkpoint_dir /tmp/viai_av_stage4_smoke --log_event_path /tmp/viai_av_stage4_smoke/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
```

关键输出：
```text
[VIAI-AV train] step=1 loss=2.451999 av_gen=0.916690 recon=0.574608 full_l1=0.292652 missing_l1=0.281956 sync=0.618531 probe=0.916778 probe_full_l1=0.290695 probe_missing_l1=0.244090 g_gan=0.859229 probe_g_gan=0.863299 d=0.718636 eta1=1.000000 eta2=1.000000 psnr=9.198 psnr_missing=9.529 ssim=0.011838
Saved VIAI-AV checkpoint: /tmp/viai_av_stage4_smoke/VIAI-AV_checkpoint_step000000001.pth.tar
```

第三阶段兼容 smoke test 已通过：
```bash
.venv/bin/python main.py train-viai-av -- --data_root data --init_from_viai_a checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --checkpoint_dir /tmp/viai_av_stage4_stage3_compat --log_event_path /tmp/viai_av_stage4_stage3_compat/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1 --disable_sync_loss --disable_probe_loss
```

关键输出：
```text
[VIAI-AV train] step=1 loss=0.889254 av_gen=0.889254 recon=0.671244 full_l1=0.319207 missing_l1=0.352037 sync=0.000000 probe=0.000000 probe_full_l1=0.000000 probe_missing_l1=0.000000 g_gan=0.822129 probe_g_gan=0.000000 d=0.718065 eta1=1.000000 eta2=1.000000
```

第四阶段测试入口已通过。由于本地 `test_new_split.txt` 为空，本次临时使用 `train_new_split.txt` 验证测试路径：
```bash
.venv/bin/python main.py test-viai-av -- --data_root data --resume_path /tmp/viai_av_stage4_smoke/VIAI-AV_checkpoint_step000000001.pth.tar --test_split_name train_new_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_av_stage4_smoke_results
```

关键输出：
```text
[VIAI-AV test] samples=1 loss=2.314855 av_gen=0.760267 recon=0.297125 sync=0.794257 probe=0.760332 g_gan=0.730555 probe_g_gan=0.730565 d=0.690305 eta1=1.000000 eta2=1.000000 mel_l1_full=0.154027 mel_l1_missing=0.143098 probe_l1_full=0.154349 probe_l1_missing=0.143321 psnr_full=14.853 psnr_missing=15.439 ssim=0.1227
[VIAI-AV test] audio->video retrieval R@1=100.00 R@5=100.00 R@10=100.00 R@50=100.00 MedR=1.0 MeanR=1.0
[VIAI-AV test] video->audio retrieval R@1=100.00 R@5=100.00 R@10=100.00 R@50=100.00 MedR=1.0 MeanR=1.0
```

checkpoint / JSON 字段检查已通过：
```text
checkpoint stage: VIAI-AV-stage4-sync-probe
checkpoint flags: True True
checkpoint has: Mel_Encoder, VideoEncoder, Mel_Decoder, netD, optimizer_G, optimizer_D
stage: VIAI-AV-stage4-sync-probe
enable_sync_loss: True
enable_probe_loss: True
loss_sync: 0.7942565083503723
loss_probe_gen: 0.7603315711021423
eta2: 1.0
retrieval_audio_to_video_r1: 100.0
retrieval_video_to_audio_r1: 100.0
```

sync loss 梯度约束检查已通过：
```text
mel_encoder_sync_grad_sum=0.000000
video_encoder_sync_grad_sum=9112.837492
```

## 2026-05-17 23:12:07 CST VIAI loss 权重命名与 TensorBoard 加权项修正

### 背景
VIAI-AV 的 beta=0.1 / beta=1 / beta=10 对照实验显示，原 `--beta_gan`
参数名容易误导：在 VIAI-A/VIAI-AV 当前论文公式实现中，该参数实际乘在
reconstruction loss 上，而不是 GAN loss 上。为避免后续训练继续混淆，本次统一改用
`--lambda_recon` 表示 reconstruction loss 权重，并补充 TensorBoard 加权 loss 监控。

### 修改内容
1. `base_options.py`
   - 移除 `--beta_gan` 参数入口。
   - 正式使用 `--lambda_recon` 作为 reconstruction loss 权重，默认 `1.0`。
   - 新增 `--lambda_gan`，用于旧 `Whole_Sync_inpainting_modify.py` 中原先的 GAN loss 权重语义。
   - 若命令行继续传入 `--beta_gan`，会直接报错提示改用 `--lambda_recon`。

2. `Models/VIAI_A_inpainting.py` / `Models/VIAI_AV_inpainting.py`
   - 将 GAN 分支损失从 `loss_g_gan + beta_gan * loss_recon` 改为
     `loss_g_gan + lambda_recon * loss_recon`。
   - VIAI-AV probe 分支同步改为
     `loss_probe_g_gan + lambda_recon * loss_probe_recon`。
   - 新增运行时加权项：
     `weighted_loss_recon`、`weighted_loss_gan`，VIAI-AV 额外新增
     `weighted_loss_probe_gen = eta2 * loss_probe_gen`。
   - `test()` 增加 `global_step` 参数，验证/测试阶段按传入 step 计算 `eta1/eta2`。

3. `train_viai_a.py` / `train_viai_av.py`
   - 验证阶段调用 `model.test(global_step=global_step)`，修复 `val/eta1` 永远为 1.0 的问题。
   - TensorBoard 新增：
     `train|val/weighted_loss_recon`、`train|val/weighted_loss_gan`；
     VIAI-AV probe 启用时新增 `train|val/weighted_loss_probe_gen`。

4. `test_viai_a.py` / `test_viai_av.py`
   - 测试 JSON/CSV 字段移除 `beta_gan`，保留/新增 `lambda_recon`。
   - 测试阶段按 checkpoint step 调用 `model.test(global_step=checkpoint_step)`。

5. `README.md`
   - 训练命令示例从 `--beta_gan` 改为 `--lambda_recon`。
   - 指标说明补充 TensorBoard 加权 loss tags。

### 验证结果
```bash
.venv/bin/python -m py_compile \
  base_options.py \
  Models/VIAI_A_inpainting.py \
  Models/VIAI_AV_inpainting.py \
  Models/Whole_Sync_inpainting_modify.py \
  train_viai_a.py \
  train_viai_av.py \
  test_viai_a.py \
  test_viai_av.py
```

静态检查已通过。

VIAI-AV smoke test 已通过。由于本地没有 `data/train_av_split.txt`，验证时使用
`train_new_split.txt` 作为本地最小 AV split，并使用同一 split 临时充当 val split：

```bash
.venv/bin/python main.py train-viai-av -- \
  --data_root data \
  --train_split_name train_new_split.txt \
  --val_split_name train_new_split.txt \
  --init_from_viai_a checkpoints/VIAI-A_checkpoint_step000000001.pth.tar \
  --checkpoint_dir /tmp/viai_lambda_recon_smoke_val \
  --log_event_path /tmp/viai_lambda_recon_smoke_val/events \
  --batch_size 1 \
  --num_workers 0 \
  --max_train_steps 2 \
  --lambda_recon 1.0 \
  --display_id 0 \
  --print_freq 1
```

关键输出：

```text
[VIAI-AV val] ... eta2=0.9999 ... step=1 ...
[VIAI-AV val] loss=3.313268 av_gen=1.286148 recon=0.591638 ...
Saved VIAI-AV checkpoint: /tmp/viai_lambda_recon_smoke_val/VIAI-AV_checkpoint_step000000002.pth.tar
```

TensorBoard tag 检查已通过，包含：

```text
train/weighted_loss_recon
train/weighted_loss_gan
train/weighted_loss_probe_gen
val/weighted_loss_recon
val/weighted_loss_gan
val/weighted_loss_probe_gen
val/eta1 = 0.999895 at step 1
```

旧参数拦截已通过：

```bash
.venv/bin/python main.py train-viai-av -- --beta_gan 0.1 --max_train_steps 0
```

输出会报错：

```text
train_viai_av: error: --beta_gan has been removed. Use --lambda_recon for the reconstruction loss weight.
```

## 2026-05-16 VIAI 第五阶段 Griffin-Lim vocoder 输出

根据 `information.md` 8.5 的路线 B，本次先接入轻量 Griffin-Lim vocoder，用于从测试阶段的修复 Mel-spectrogram 导出 waveform。该路线不训练 WaveNet，也不依赖 HiFi-GAN / WaveGlow 预训练 checkpoint；目标是先跑通端到端 demo 音频输出。

### 主要改动
1. `utils/vocoder.py`
   - 新增 normalized Mel `[0, 1]` 到 waveform 的反变换工具。
   - 复用当前工程 Mel 参数：16kHz、80 Mel bins、`fft_size=1280`、`hop_size=320`、`fmin=125`、`fmax=7600`、`min_level_db=-100`、`ref_level_db=20`。
   - 使用 `librosa.feature.inverse.mel_to_audio(..., power=1.0, center=False)`，并把输出裁剪/补零到 `mel_frames * hop_size`。
   - 批量保存 `*_reconstructed.wav` 和 `*_target.wav`。

2. `base_options.py`
   - 新增 `--use_vocoder`，默认关闭。
   - 新增 `--vocoder_backend griffin_lim`、`--vocoder_n_iter`、`--vocoder_max_samples`、`--vocoder_output_dir`。

3. `test_viai_a.py` / `test_viai_av.py`
   - 测试指标、Mel 图默认行为保持不变。
   - 开启 `--use_vocoder` 后，输出目录默认为 `<results_dir>/wav/stepXXXXXXXXX/`。
   - reconstructed Mel 使用 `mel_input * (1 - missing_mask) + mel_pred * missing_mask`，只替换缺失区域。
   - JSON/CSV 增加 `use_vocoder`、`vocoder_backend`、`vocoder_n_iter`、`vocoder_output_dir`、`vocoder_num_samples`。

4. `README.md`
   - 新增“第五阶段：Griffin-Lim vocoder 输出 waveform”说明和命令示例。

### 验证命令
静态检查已通过：
```bash
.venv/bin/python -m py_compile base_options.py utils/vocoder.py test_viai_a.py test_viai_av.py
```

Griffin-Lim 单条 Mel 反变换 smoke test 已通过：
```bash
.venv/bin/python -c "import numpy as np; from Options_inpainting import Inpainting_Config; from utils.vocoder import mel_to_waveform; hp=Inpainting_Config(force_reload=True, args=[]); mel=np.load('data/processed/accordion/A2p8VW61RGc/mel.npy').astype('float32')[:200].T; wav=mel_to_waveform(mel, hp, n_iter=1); print(wav.shape, wav.dtype, bool(np.isfinite(wav).all()), float(wav.min()), float(wav.max()))"
```

关键输出：
```text
(64000,) float32 True -0.48688557744026184 10.033004760742188
```

VIAI-A vocoder smoke test 已通过：
```bash
.venv/bin/python main.py test-viai-a -- --data_root data --resume_path checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --test_split_name train_viai_a_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_vocoder_a_results --use_vocoder --vocoder_max_samples 1 --vocoder_n_iter 1
```

关键输出：
```text
[VIAI-A test] wrote vocoder wavs: /tmp/viai_vocoder_a_results/wav/step000000001 (1 samples)
```

VIAI-AV vocoder smoke test 已通过。由于仓库内旧 `checkpoints/VIAI-AV_checkpoint_step000000001.pth.tar` 缺少当前 `VideoEncoder` 字段，本次先用当前代码训练 1 step 到 `/tmp`，再验证测试入口：
```bash
.venv/bin/python main.py train-viai-av -- --data_root data --init_from_viai_a checkpoints/VIAI-A_checkpoint_step000000001.pth.tar --checkpoint_dir /tmp/viai_vocoder_av_smoke --log_event_path /tmp/viai_vocoder_av_smoke/events --batch_size 1 --num_workers 0 --max_train_steps 1 --display_id 0 --print_freq 1
.venv/bin/python main.py test-viai-av -- --data_root data --resume_path /tmp/viai_vocoder_av_smoke/VIAI-AV_checkpoint_step000000001.pth.tar --test_split_name train_new_split.txt --batch_size 1 --num_workers 0 --display_id 0 --results_dir /tmp/viai_vocoder_av_results --use_vocoder --vocoder_max_samples 1 --vocoder_n_iter 1
```

关键输出：
```text
[VIAI-AV test] wrote vocoder wavs: /tmp/viai_vocoder_av_results/wav/step000000001 (1 samples)
```
