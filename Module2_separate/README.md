# Module 2：音源分离（人声提取）

基于 **Demucs (htdemucs)** 模型，从 16kHz 单声道 WAV 中提取人声并保存为 16kHz 单声道 WAV。

---

## 环境配置（可迁移）

### 1. 创建 conda 环境

确保已安装 [Anaconda/Miniconda](https://docs.conda.io/en/latest/miniconda.html)。

```bash
# 进入项目根目录（包含 environment.yml 和 requirements.txt）
cd Module2_separate

# 创建环境
conda env create -f environment.yml
```

### 2. 激活环境

```bash
conda activate mod2_separate
```

### 3. 安装 PyTorch（根据硬件选择一条命令）

**重要：PyTorch 必须用官方 pip 安装，避免 Windows DLL 错误。**

- **NVIDIA GPU（CUDA 12.1）**

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

### 4. 安装其余模块依赖

```bash
pip install -r requirements.txt
```

### 5. 验证安装

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); import demucs; print('Module 2 ready')"
```

输出应显示相应信息且无报错。

---

## 使用方法

### 命令行

```bash
python separate.py <输入音频> [输出人声] [--temp-dir <临时目录>]
```

#### 示例

- **输出自动生成到 temp 目录**

  ```bash
  python separate.py audio_16k.wav --temp-dir ../temp
  ```

- **指定输出路径**

  ```bash
  python separate.py audio_16k.wav vocals.wav
  ```

#### 参数说明

| 参数       | 必须 | 说明                                      |
|------------|------|-------------------------------------------|
| input      | 是   | 16kHz 单声道 WAV 文件路径（通常由 M1 生成） |
| output     | 否   | 输出人声 WAV 路径，默认为 `输入文件名_vocals.wav` 自动放入临时目录 |
| --temp-dir | 否   | 临时文件存放目录，优先级高于系统默认临时目录 |

---

## 测试流程（独立调试）

### 准备测试音频

使用 M1 生成的 `xxx_16k.wav` 文件。

### 运行

```bash
python separate.py E:\...\309jp_16k.wav E:\...\309jp_vocals.wav --temp-dir E:\...\temp
```

### 预期输出

```text
[GPU] NVIDIA GeForce RTX XXXX
[Module 2] 加载 htdemucs 模型...
[Module 2] 音源分离中...
[Module 2] 完成。输出: E:\...\309jp_vocals.wav
输出文件: E:\...\309jp_vocals.wav
```

---

## 常见问题

### 1. ImportError: DLL load failed while importing ... fbgemm.dll

**原因**：Windows 缺少 Visual C++ 运行库或 PyTorch 安装不完整。

**解决**：

- 下载安装 [VC++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)
- 确保使用 **官方 pip** 安装 PyTorch，而非 conda 的 PyTorch（见上方安装步骤）
- 如仍报错，可尝试 CPU 版 PyTorch

### 2. 内存不足（OOM）

**现象**：处理大文件（>1小时）时进程被杀死。

**缓解**：

- 确保系统有 8GB 以上可用内存
- 关闭其他占用内存的程序
- 可将输入音频先切割为 10 分钟片段再处理（未来可脚本化）

### 3. 分离效果不理想

- 模型默认使用 `htdemucs`，可尝试在代码中将 `pretrained.get_model('htdemucs')` 改为 `htdemucs_ft` 或 `mdx_extra`（需先 `pip install demucs[all]`，可能引入额外依赖）
- 调整 `shifts` 参数（如 `shifts=3`）可提高质量，但更慢

### 4. 无 GPU 时速度慢

- CPU 处理 1 小时音频约需 10~30 分钟（视硬件而定）
- 可考虑升级代码中的 `segment` 大小（默认为 `apply_model` 内部自适应），一般无需修改

---

## 文件结构

```text
Module2_separate/
├── separate.py              # 主脚本
├── environment.yml          # conda 环境定义（仅 Python）
├── requirements.txt         # pip 依赖
└── README.md                # 本文件
```

---

## 与流水线集成

该模块接受 M1 输出的 16k WAV，输出人声 WAV 供 M3 使用。

在主控 `pipeline.py` 中已配置好调用逻辑，无需手动干预。