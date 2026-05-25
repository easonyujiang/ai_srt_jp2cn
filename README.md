# AI 日语字幕生成与翻译流水线

全自动日语视频字幕生成 + 日译中翻译流水线，从视频文件一键生成带时间轴的中文字幕（SRT/ASS 格式）。

## 流水线架构

```
视频文件 → M1 解复用 → M2 人声分离 → M3 ASR 转写对齐 → M4 文本规范化 → M5 字幕生成 → M6 翻译 → 中文字幕
```

### 模块说明

| 模块 | 功能 | 核心技术 | 依赖 API |
|------|------|---------|---------|
| **M1** 解复用 | 视频提取为 16kHz 单声道 WAV | ffmpeg | - |
| **M2** 人声分离 | 分离背景音/人声 | Demucs (htdemucs) | - |
| **M3** ASR 转写 | 语音识别 + 词级时间戳 + 多说话人分离 | WhisperX (large-v2) + pyannote + ECAPA-TDNN | HuggingFace |
| **M4** 规范化 | 修正 ASR 错误、口语整理、保留说话人标签 | DeepSeek | DeepSeek API |
| **M5** 字幕生成 | 轴文重构，输出 SRT/ASS | Python 标准库 | - |
| **M6** 翻译 | 日译中，保留口语化表达 | DeepSeek | DeepSeek API |

## 环境要求

- **操作系统**：Windows / Linux（config.py 自动适配）
- **Python**：3.10
- **包管理**：Anaconda / Miniconda
- **GPU**：推荐 NVIDIA GPU（≥8GB 显存，RTX 3060/4060 即可），使用 CUDA int8 推理
- **ffmpeg**：需可通过命令行调用

### 0. Windows 用户：下载 cuDNN DLL（约 900MB）

`ctranslate2` 的 PyPI wheel 自带 cuDNN 8 DLL 不完整，GPU 推理需要完整的 cuDNN 8.9.x。

```bash
# 一键下载（从 HuggingFace 镜像拉取）
python scripts/download_cudnn.py

# 或手动从 NVIDIA 下载 cuDNN 8.9.x for CUDA 12.x
# https://developer.nvidia.com/cudnn
# 解压后将 bin/ 下所有 .dll 放入 libs/
```

> **Linux 云端用户**：`conda install -c conda-forge cudnn=8.9.*` 即可，不需要 DLL。`deploy_cloud.sh` 已自动处理。

## 快速开始

### 1. 克隆仓库

```bash
git clone <仓库地址>
cd ai_srt_jp2cn
```

### 2. 配置 API 密钥

复制示例配置文件并填入你的密钥：

```bash
cp .env.example .env
```

编辑 `.env`，替换为你的真实密钥：

```
HF_TOKEN=hf_你的HuggingFace_Token
DEEPSEEK_API_KEY=sk-你的DeepSeek_API密钥
```

> **获取密钥**：
> - HuggingFace Token：前往 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 创建，**务必勾选 "Read access to content of gated repos"**
> - DeepSeek API Key：前往 [platform.deepseek.com](https://platform.deepseek.com/) 获取

#### ⚠️ 门控模型授权（必须）

`pyannote/speaker-diarization-3.1` 和 `pyannote/segmentation-3.0` 是门控模型，**即使有 HF_TOKEN 也必须先在浏览器中接受协议**：

1. 登录 [hf-mirror.com](https://hf-mirror.com)（或 [huggingface.co](https://huggingface.co)）
2. 打开 [speaker-diarization-3.1](https://hf-mirror.com/pyannote/speaker-diarization-3.1)，点击 **"Agree and access repository"**
3. 打开 [segmentation-3.0](https://hf-mirror.com/pyannote/segmentation-3.0)，点击 **"Agree and access repository"**
4. 填写表单（Company/university、Website）后提交，即可下载

> 未完成授权的话，download_models.py 会自动检测并给出提示。

### 3. 预下载 M3 模型（推荐）

M3（ASR 转写对齐）首次运行需从 HuggingFace 下载约 **13 GB** 模型文件（6 个仓库）。国内网络环境建议提前下载。

```bash
# 使用 M3 环境运行（已有 huggingface_hub）
conda activate mod3_asr

# 如需国内镜像加速，先在 .env 中设置
# HF_ENDPOINT=https://hf-mirror.com

# 下载全部 5 个模型（需 .env 中配置 HF_TOKEN 并完成门控授权）
python download_models.py

# 可选参数
python download_models.py --skip-gated              # 跳过门控模型
python download_models.py --repo openai/whisper-large-v2  # 指定仓库
python download_models.py --no-mirror               # 直连官方源
python download_models.py --retry 5                 # 重试次数
python download_models.py --cache-dir ./models      # 指定缓存目录
```

模型下载到 `models/` 目录（HF 标准缓存格式），M3 运行时 WhisperX 会**自动识别**，无需额外配置。

> **原理**：`download_models.py` 使用 `huggingface_hub.snapshot_download()` —— HuggingFace 官方下载接口，自动拉取仓库中所有模型文件并写入标准缓存结构。WhisperX 在运行时通过 `HF_HOME` 读取同一目录，直接命中缓存。下载完成后还会自动修复 `speaker-diarization-3.1` 的 config.yaml 格式（添加 `pipeline` 和 `params` 键，兼容 pyannote 3.x）。

| 模型 | 大小 | 说明 |
|------|------|------|
| `Systran/faster-whisper-large-v2` | ~3 GB | Faster-Whisper CT2 (WhisperX 实际调用) |
| `openai/whisper-large-v2` | ~6.5 GB | Whisper 原始模型 (tokenizer 等，备用参考) |
| `wav2vec2-large-xlsr-53-japanese` | ~1.2 GB | 词级对齐 |
| `pyannote/segmentation-3.0` | ~380 MB | 语音活动检测（门控）|
| `pyannote/speaker-diarization-3.1` | 数 MB | 说话人分割 Pipeline（门控）|
| `speechbrain/spkrec-ecapa-voxceleb` | ~34 MB | ECAPA-TDNN 说话人嵌入 |
| `pyannote/wespeaker-voxceleb-resnet34-LM` | ~120 MB | WeSpeaker 说话人嵌入（pyannote 3.x 新版必需，门控）|

> **注意**：如果之前用旧版下载过模型（扁平结构 `hub/pytorch_model.bin`），需清理后重下：
> ```bash
> rm -rf models/hub
> python download_models.py
> ```

#### 备用方案：手动浏览器下载 + 导入

如果网络环境导致 `download_models.py` 下载失败，可以使用浏览器手动下载模型文件，再用 `import_models.py` 导入到标准缓存：

```bash
# 1. 查看需要下载的文件清单和下载地址
python import_models.py --list-missing

# 2. 用浏览器逐个下载文件，放入 models_download/ 目录

# 3. 检查文件完整性
python import_models.py --check-only

# 4. 导入到 models/ 标准缓存
python import_models.py
```

### 4. 创建 Conda 环境

每个模块有独立的 conda 环境，使用环境名分别为：`mod1_demux`、`mod2_separate`、`mod3_asr`、`mod4_normalize`、`mod5_subtitle`、`mod6_translate`。

```bash
# 依次创建各模块环境
cd Module1_demux && conda env create -f environment.yml && cd ..
cd Module2_separate && conda env create -f environment.yml && cd ..
cd Module3_asr && conda env create -f environment.yml && cd ..
cd Module4_normalize && conda env create -f environment.yml && cd ..
cd Module5_subtitle && conda env create -f environment.yml && cd ..
cd Module6_translate && conda env create -f environment.yml && cd ..
```

### 5. 安装 pip 依赖

```bash
# 流水线主环境
pip install -r requirements_pipeline.txt
```

#### M2 - 音源分离（需要手动安装 PyTorch）

**重要：PyTorch 必须用官方 pip 安装，避免 Windows DLL 错误。**

先安装基础依赖：
```bash
conda activate mod2_separate
pip install -r Module2_separate/requirements.txt
```

然后根据你的硬件选择 PyTorch 版本：

- **NVIDIA GPU（CUDA 12.1，推荐）**
  ```bash
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
  ```

- **NVIDIA GPU（CUDA 11.8）**
  ```bash
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
  ```

- **仅 CPU**
  ```bash
  pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
  ```

验证安装：
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); import demucs; print('Module 2 ready')"
```

```bash
# M3 - ASR
conda activate mod3_asr
pip install -r Module3_asr/requirements.txt

# M4 - 规范化
conda activate mod4_normalize
pip install -r Module4_normalize/requirements.txt

# M6 - 翻译
conda activate mod6_translate
pip install -r Module6_translate/requirements.txt
```

### 6. 运行（config.py 已自动适配平台，无需手动配置路径）

#### GUI 模式

```bash
python pipeline.py
```

启动图形界面，选择视频文件后一键运行全流程。

#### 命令行模式

```bash
python pipeline.py video.mp4
```

#### Web 前端模式

Web 前端通过 Flask + SSE（Server-Sent Events）提供可视化操作界面，支持拖拽上传、实时进度监控、参数配置和字幕下载。所有后端逻辑复用 `PipelineRunner`，与 CLI/GUI 模式完全一致。

##### 前置条件

Web 服务器依赖流水线主环境的 pip 包。如果还未安装：

```bash
pip install -r requirements_pipeline.txt
```

`requirements_pipeline.txt` 已包含 Flask 和 python-dotenv，无需额外安装。

##### 启动 Web 服务器

```bash
# 本地访问（仅本机可访问）
python web_server.py

# 局域网 / 公网访问
python web_server.py --host 0.0.0.0 --port 5000

# 自定义端口
python web_server.py --port 8080
```

启动后终端会打印访问地址，例如 `http://127.0.0.1:5000`。

##### 使用流程

**1. 上传视频**

打开浏览器访问 Web 界面，可以通过以下两种方式添加视频：

- **拖拽上传**：将视频文件直接拖入页面中央的「拖拽或点击」区域
- **点击上传**：点击上传区域选择本地视频文件
- **已有视频**：如果 `videos/` 目录下已有视频文件，页面左侧会自动列出

支持格式：`.mp4` `.mkv` `.avi` `.mov` `.flv`

**2. 配置参数**

点击已上传的视频，右侧会展开参数配置面板，可调整：

| 参数 | 说明 | 可选值 |
|------|------|--------|
| Whisper 模型 | ASR 模型大小 | large-v2 / large-v3 |
| 最大说话人数 | pyannote 聚类上限 | 1-10 |
| 规范化风格 | M4 文本清洗策略 | retain（保留口语）/ clean（正式） |
| 字幕格式 | 输出格式 | ass（彩色说话人）/ srt（通用） |
| 翻译风格 | M6 翻译策略 | creative（本地化）/ literal（直译） |
| 计算精度 | GPU 推理精度 | int8（低显存）/ float16（高精度） |
| 计算设备 | 推理后端 | cuda / cpu |

**3. 运行流水线**

点击「开始处理」按钮，流水线会依次执行 M1→M6 六个模块。界面实时显示：

- **模块进度条**：6 个模块依次高亮，当前模块显示百分比（如「M3 ASR转写 · 67%」）
- **实时日志流**：底部日志面板通过 SSE 实时推送，支持 info / debug / warn / error 彩色分级
- **状态指示**：顶部显示当前状态（运行中 / 完成 / 错误）

> **注意**：同一时间只能运行一个流水线任务。任务运行期间「开始处理」按钮会禁用，完成后恢复。

**4. 下载字幕**

流水线完成后：

- 页面自动显示结果摘要（包含日文和中文两个字幕文件）
- 点击「下载字幕」可下载 `.ass` 或 `.srt` 文件
- 也可在「字幕列表」面板中随时下载历史生成的字幕

##### 查看技术讲解页面

Web 服务器还内置了一个模块化技术讲解幻灯片页面（presentation）：

```
http://127.0.0.1:5000/presentation
```

使用键盘 **↑↓←→** 或鼠标滚轮翻页，共 20 页，涵盖 M1-M6 每个模块的技术细节、Prompt 设计和容错机制。

##### Web 界面预览

![Web UI 截图](https://trae-api-cn.mchost.guru/api/ide/v1/text_to_image?prompt=A%20dark%20theme%20web%20application%20interface%20for%20an%20AI%20subtitle%20generation%20pipeline%2C%20modern%20and%20clean%20design%2C%20showing%20video%20upload%20area%20on%20the%20left%20panel%2C%20progress%20bars%20for%206%20pipeline%20modules%20in%20the%20center%2C%20real-time%20log%20panel%20at%20the%20bottom%2C%20with%20Chinese%20text%20labels%2C%20professional%20developer%20tool%20aesthetic&image_size=landscape_16_9)

> **技术架构**：Web 前端使用 Flask 作为 HTTP 服务器，SSE（Server-Sent Events）实现服务端到浏览器的单向实时推送。后端 `WebPipelineManager` 管理事件队列和流水线生命周期。代码入口：[web_server.py](web_server.py)，前端页面：[templates/index.html](templates/index.html)，技术讲解：[templates/presentation.html](templates/presentation.html)。

#### 单独运行某模块

```bash
# 仅提取音频
conda activate mod1_demux
python Module1_demux/demux.py video.mp4 output.wav

# 仅人声分离
conda activate mod2_separate
python Module2_separate/separate.py input.wav vocals.wav
```

## ☁️ CloudStudio / AutoDL 云端一键部署

适用于 腾讯 CloudStudio、AutoDL 等 GPU 算力平台。推荐 V100 32GB / RTX 3090 24GB / RTX 4090 24GB。

### 云端拉取命令

```bash
# 从 GitHub 拉取最新代码
git clone https://github.com/easonyujiang/ai_srt_jp2cn.git
cd ai_srt_jp2cn

# 如果已克隆，拉取更新
git pull origin master

# 复制环境变量模板并配置密钥
cp .env.example .env
vim .env   # 或 nano .env，填入 HF_TOKEN 和 DEEPSEEK_API_KEY
```

### 一键部署

```bash
# 1. 终端输入以下唯一命令
bash <(curl -sSL https://raw.githubusercontent.com/easonyujiang/ai_srt_jp2cn/master/deploy_cloud.sh)

# 2. 按提示输入 API 密钥，等待安装完成

# 3. 上传视频
# 用 JupyterLab 文件管理器拖拽上传到 videos/ 目录

# 4. 运行
python pipeline.py videos/你的视频.mp4

# 5. 用完关机
sudo shutdown -h now
```

> 脚本自动完成：GPU 检测 → 安装系统依赖 → 克隆仓库 → 创建 6 个独立环境 → 安装 PyTorch/Demucs/WhisperX（含 cudatoolkit） → 自动检测 HF 直连/镜像 → 模型预下载（含 wespeaker） → 验证安装

### 云端已知问题与处理

| 问题 | 现象 | 方案 |
|------|------|------|
| cuBLAS 缺失 | `libcublas.so.11 not found` | 部署脚本已自动安装 `cudatoolkit=11.8` |
| 显存检测为 0GB | M2 显示 `(0.0 GB)` 低显存模式 | 不影响运行，V100 自动降级为分片处理 |
| 模型缓存路径 | pyannote 找不到本地文件 | 部署脚本自动创建 `models/hub → ~/.cache/huggingface/hub` 软链接 |
| faster-whisper API 变更 | `TranscriptionOptions missing arguments` | `transcribe_align.py` 已内置 monkey-patch 兼容 ctranslate2 3.20 |
| PyAV 编译失败 | 缺少 ffmpeg 开发库 | 部署脚本已自动安装 `libav*-dev` |

## 配置参数

在 `config.py` 中可以调整以下默认值：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEFAULT_LANGUAGE` | `"ja"` | ASR 语言 |
| `DEFAULT_WHISPER_MODEL` | `"large-v2"` | Whisper 模型大小 |
| `DEFAULT_MAX_SPEAKERS` | `5` | 最大说话人数 |
| `DEFAULT_NORM_STYLE` | `"retain"` | 规范化风格 (retain/clean) |
| `DEFAULT_SUBTITLE_FORMAT` | `"ass"` | 字幕格式 (ass/srt) |
| `DEFAULT_TRANSLATION_STYLE` | `"creative"` | 翻译风格 |
| `DEFAULT_TRANSLATION_CHUNK_SIZE` | `20` | 翻译切片大小（过大易导致 API 返回格式异常） |

### 命令行参数

```bash
python pipeline.py video.mp4 \
  --device cuda \           # cuda / cpu（默认 cuda）
  --compute-type int8 \     # int8 / float16（默认 int8，8GB 显存推荐）
  --max-speakers 5 \        # 最大说话人数
  --subtitle-format ass \    # ass / srt
  --quiet                    # 减少输出
```

## 输出

- **源语言字幕**：`subtitles/` 目录
- **翻译后中文字幕**：`subtitles_translated/` 目录
- **中间文件**：`temp/` 目录（运行后可清理）

## 注意事项

- **`.env` 文件包含 API 密钥，切勿提交到公开仓库**
- 首次运行 M3 会从 HuggingFace 下载约 10GB 模型文件，建议先用 [`download_models.py`](#3-预下载-m3-模型推荐) 预下载；如网络受限，可用 [`import_models.py`](#备用方案手动浏览器下载--导入) 手动导入
- 腾讯云 / AutoDL 等国内算力平台通常可直接访问 HuggingFace。如果下载失败，在 `.env` 中添加 `HF_ENDPOINT=https://hf-mirror.com` 或使用 `python download_models.py --retry 10`

## 常见问题

### ImportError: DLL load failed (fbgemm.dll)

Windows 缺少 Visual C++ 运行库或 PyTorch 安装不完整。

**解决**：下载安装 [VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)，然后确保使用官方 pip 安装 PyTorch（见上方 M2 安装步骤）。

### 内存不足（OOM）

处理大文件（>1小时）时进程被杀死。

**缓解**：确保系统有 8GB 以上可用内存，关闭其他占用内存的程序。

### 人声分离效果不理想

模型默认使用 `htdemucs`，可在 `separate.py` 中将 `pretrained.get_model('htdemucs')` 改为 `htdemucs_ft` 或 `mdx_extra`（需先 `pip install demucs[all]`）。调整 `shifts` 参数（如 `shifts=3`）可提高质量，但更慢。

### CUDA 崩溃：cudnn_ops_infer64_8.dll 缺失（0xC0000409）

`ctranslate2` 的 PyPI wheel 自带 cuDNN 8 DLL 不完整，在 CUDA + float16 模式下会崩溃。

**症状**：M3 阶段立即退出，返回码 `0xC0000409` 或 `-1073740791`，stderr 显示 `Could not locate cudnn_ops_infer64_8.dll`。

**解决**（已内置）：项目 `transcribe_align.py` 启动时自动检查 `libs/` 目录（通过 `os.add_dll_directory()` 加载 cuDNN DLL）。`libs/` 目录**不随仓库分发**（约 900MB），首次使用前运行：

```bash
python scripts/download_cudnn.py
```

或手动从 [NVIDIA cuDNN Archive](https://developer.nvidia.com/cudnn) 下载 cuDNN 8.9.x for CUDA 12.x，将 `bin/` 下的 DLL 放入 `libs/`。

> 注意：默认使用 CUDA int8 模式，可缓解 cuDNN 缺失问题。如果你是云端 Linux 环境，可通过 `conda install -c conda-forge cudnn` 安装，不需要 `libs/`。

### M3 对齐阶段卡住 / HuggingFace 连接超时

国内网络环境访问 huggingface.co 可能超时。

**解决**（已内置）：在 `.env` 中添加 `HF_ENDPOINT=https://hf-mirror.com`，脚本已自动设置离线模式（`HF_HUB_OFFLINE=1`）和镜像端点回退。确保模型已完整导入到 `models/hub/` 目录。

### 无 GPU 时速度慢

CPU 处理 1 小时音频约需 10~30 分钟。可考虑将输入音频切割为 10 分钟片段再处理。

## 许可证

MIT License
