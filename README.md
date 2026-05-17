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

- **操作系统**：Windows（其他平台需调整 conda 路径配置）
- **Python**：3.10
- **包管理**：Anaconda / Miniconda
- **GPU**：推荐 NVIDIA GPU（≥8GB 显存），低显存自动分片处理
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
> - HuggingFace Token：前往 [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) 创建
> - DeepSeek API Key：前往 [platform.deepseek.com](https://platform.deepseek.com/) 获取

### 3. 创建 Conda 环境

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

### 4. 安装 pip 依赖

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

### 5. 配置路径

编辑 `config.py` 中的 `CONDA_ENVS_DIR`，指向你的 Anaconda 安装目录：

```python
CONDA_ENVS_DIR = Path(r"C:\ProgramData\anaconda3\envs")
```

### 6. 运行

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
- 首次运行 M3 会下载 WhisperX 模型（约 3GB），请保持网络畅通
- 支持 HuggingFace 镜像站，在 `.env` 中设置 `HF_ENDPOINT=https://hf-mirror.com` 可加速下载

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
