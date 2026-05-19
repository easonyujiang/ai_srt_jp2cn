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
| **M3** ASR 转写 | 语音识别 + 词级对齐 + 说话人分离 | WhisperX (large-v2) | HuggingFace |
| **M4** 规范化 | 修正 ASR 错误、口语整理 | DeepSeek | DeepSeek API |
| **M5** 字幕生成 | 轴文重构，输出 SRT/ASS | Python 标准库 | - |
| **M6** 翻译 | 日译中，保留口语化表达 | DeepSeek | DeepSeek API |

## 环境要求

- **操作系统**：Windows / Linux（config.py 自动适配）
- **Python**：3.10
- **包管理**：Anaconda / Miniconda
- **GPU**：推荐 NVIDIA GPU（≥16GB 显存），低显存自动分片处理
- **ffmpeg**：需可通过命令行调用

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

M3（ASR 转写对齐）首次运行需从 HuggingFace 下载约 **10 GB** 模型文件。国内网络环境建议提前下载。

```bash
# 使用 M3 环境运行（已有 huggingface_hub）
conda activate mod3_asr

# 如需国内镜像加速，先在 .env 中设置
# HF_ENDPOINT=https://hf-mirror.com

# 下载全部 4 个模型（需 .env 中配置 HF_TOKEN 并完成门控授权）
python download_models.py

# 可选参数
python download_models.py --skip-gated              # 跳过门控模型
python download_models.py --repo openai/whisper-large-v2  # 指定仓库
python download_models.py --no-mirror               # 直连官方源
python download_models.py --retry 5                 # 重试次数
python download_models.py --cache-dir ./models      # 指定缓存目录
```

模型下载到 `models/` 目录（HF 标准缓存格式），M3 运行时 WhisperX 会**自动识别**，无需额外配置。

> **原理**：`download_models.py` 使用 `huggingface_hub.snapshot_download()` —— HuggingFace 官方下载接口，自动拉取仓库中所有模型文件并写入标准缓存结构。WhisperX 在运行时通过 `HF_HOME` 读取同一目录，直接命中缓存。

| 模型 | 大小 | 说明 |
|------|------|------|
| `openai/whisper-large-v2` | ~6.5 GB | 语音识别 |
| `pyannote/speaker-diarization-3.1` | ~800 MB | 说话人分离（门控）|
| `pyannote/segmentation-3.0` | ~380 MB | 语音活动检测（门控）|
| `wav2vec2-large-xlsr-53-japanese` | ~1.2 GB | 词级对齐 |

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

> 脚本自动完成：GPU 检测 → 安装系统依赖 → 克隆仓库 → 创建 6 个独立环境 → 安装 PyTorch/Demucs/WhisperX → 自动检测 HF 直连/镜像 → 验证安装

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

## 输出

- **源语言字幕**：`subtitles/` 目录
- **翻译后中文字幕**：`subtitles_translated/` 目录
- **中间文件**：`temp/` 目录（运行后可清理）

## 注意事项

- **`.env` 文件包含 API 密钥，切勿提交到公开仓库**
- 首次运行 M3 会从 HuggingFace 下载约 10GB 模型文件，建议先用 [`download_models.py`](#3-预下载-m3-模型推荐) 预下载
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

### 无 GPU 时速度慢

CPU 处理 1 小时音频约需 10~30 分钟。可考虑将输入音频切割为 10 分钟片段再处理。

## 许可证

MIT License
