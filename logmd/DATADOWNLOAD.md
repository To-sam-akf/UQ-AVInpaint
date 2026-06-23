# MUSICES 数据下载排查记录

## 1. 这次报错是什么

你在运行：

```bash
bash tools/export_windows_edge_cookies.sh
```

时看到：

```text
Extracting cookies from edge
[Cookies] Loading cookie ...
ERROR: Failed to decrypt with DPAPI
```

这不是“复制/迁移 cookies 把 Edge 搞坏了”，而是 Windows 在解密 Edge 的 Chromium cookies 时失败了。

## 2. 当前脚本会不会影响 Edge 使用

不会。

当前脚本的行为是：

1. 调用 Windows 版 `yt-dlp.exe`
2. 让它从 Edge 读取 cookies
3. 尝试导出一份新的 `cookies.txt`
4. 再把导出的文件复制到项目目录

也就是说，它做的是 export/copy，不是 move，不会迁移或删除 Edge 原始 cookies 数据库。

当前脚本失败时，也只是“导不出来”，不会改坏 Edge 本身。

## 3. 为什么不再推荐自动从 Edge 导出

目前已经确认两点：

1. 关闭 Edge 后，自动导出依然可能被 `DPAPI` 解密失败卡住。
2. `yt-dlp` 官方文档对 YouTube 更推荐“私密/无痕窗口 + 手动导出新鲜 cookies.txt”，而不是长期依赖浏览器 cookies 自动提取。

因此现在的策略是：

1. `tools/export_windows_edge_cookies.sh` 只保留为备用工具
2. 主推荐方案改为手动导出 Netscape 格式的 `youtube_cookies.txt`

## 4. 现在的主推荐方案

### 方案 A：手动导出 `youtube_cookies.txt`（主路径）

推荐浏览器：

1. Windows Firefox + `cookies.txt` 扩展
2. Windows Edge / Chrome + `Get cookies.txt LOCALLY`

推荐导出步骤：

1. 打开一个新的私密/无痕窗口
2. 在该窗口里登录 YouTube
3. 在同一个标签页访问：

```text
https://www.youtube.com/robots.txt
```

4. 只导出 `youtube.com` 相关 cookies
5. 导出为 Netscape 格式
6. 保存到：

```bash
/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

注意：

1. 导出后尽量不要继续在同一个私密窗口里刷 YouTube，避免 cookies 被轮换
2. 这个 `cookies.txt` 最好是“刚导出不久”的新鲜文件

## 5. 下载时统一使用的命令

### 当前新问题：`n challenge solving failed`

如果看到下面这类报错：

```text
n challenge solving failed
Only images are available for download
Requested format is not available
```

说明 cookies 已经被读取到了，但 `yt-dlp` 没有成功解决 YouTube 的 JavaScript challenge。  
现在脚本已经支持 `--yt-dlp-js-runtime auto`，会自动优先使用本机可用的 `deno`、`node`、`bun` 或 `qjs`。当前机器已经检测到 `node`，所以默认命令会自动使用 Node。

本次已经验证成功：

```text
[youtube] [jsc:node] Solving JS challenges using node
```

并且已成功下载第一条样本：

```text
data/raw_videos/accordion/iB17xqmFw3A.mp4
```

先验证 1 条样本：

```bash
uv run python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --skip-existing \
  --max-videos 1 \
  --abort-on-download-error \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

成功标准：

1. 不再报 `Cookie file not found`
2. 不再报 `Failed to decrypt with DPAPI`
3. 不再报 `Sign in to confirm you're not a bot`
4. 开始出现真实下载进度

如果你想显式指定 Node，可以加：

```bash
--yt-dlp-js-runtime node
```

如果自动模式失效，完整命令可以写成：

```bash
uv run python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --skip-existing \
  --max-videos 1 \
  --abort-on-download-error \
  --yt-dlp-js-runtime node \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

单条成功后，再跑全量：

```bash
uv run python main.py prepare-data -- download \
  --json data/MUSICES.json \
  --data-root data \
  --skip-existing \
  --yt-dlp-extra-arg=--cookies \
  --yt-dlp-extra-arg=/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt
```

## 6. 如果 YouTube 下载仍然不稳定

MUSICES 官方公开的是 YouTube ID 清单，而不是一个稳定的完整视频压缩包。因此批量下载会受 YouTube 可用性、地区、账号状态、cookies 新鲜度和 challenge 规则影响。

可行替代路线：

1. 继续使用当前脚本批量下载，失败样本会写入 `data/musices_download_failures.csv`，后续可以换网络或更新 cookies 后重跑。
2. 先用 `--max-videos` 下载小批量样本，确认训练管线能跑通，再逐步扩大数据量。
3. 如果某些 YouTube ID 永久失效，只保留成功下载并处理的样本生成 splits；脚本会跳过缺失视频。
4. 如果需要完全稳定的数据源，需要改造数据入口，换成你本地已有的音视频数据集，并生成与 `data/processed/<instrument>/<youtube_id>/` 兼容的目录结构。

## 7. 自动 Edge 导出现在怎么定位

`bash tools/export_windows_edge_cookies.sh` 现在只是备用方案。

它的定位是：

1. 能成功则省一步
2. 失败时直接提示你切回手动导出
3. 不再把它当成默认主路径

如果它报 `DPAPI`，不要反复重试，直接改走“手动导出 Netscape cookies.txt”。

## 8. 下载完成后的后续步骤

```bash
uv run python main.py prepare-data -- process --json data/MUSICES.json --data-root data --skip-existing
uv run python main.py prepare-data -- splits --json data/MUSICES.json --data-root data
```

## 9. 相关文件

1. [data/youtube_cookies.txt](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/youtube_cookies.txt)
   - 手动导出的 Netscape cookies 文件
2. [data/musices_download_failures.csv](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/musices_download_failures.csv)
   - 记录失败样本
3. [data/musices_download_stats.csv](/home/sanmu/Vision-Infused-Audio-Inpainter-VIAI/data/musices_download_stats.csv)
   - 记录大小估算结果
4. `data/musices_downloaded.txt`
   - 记录已成功下载样本，支持断点续跑
