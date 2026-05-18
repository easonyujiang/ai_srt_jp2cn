#!/usr/bin/env bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

section()   { echo -e "\n${GREEN}━━━ $1 ━━━${NC}"; }
warn()      { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()      { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
check_ok()  { echo -e "  ${GREEN}✓${NC} $1"; }
check_bad() { echo -e "  ${RED}✗${NC} $1"; }

# ============================================================
section "ai_srt_jp2cn 云端部署 (Linux V100 + 6独立环境)"
# ============================================================

# ----- 0. GPU -----
section "[0/8] GPU 检测"
if nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
    check_ok "GPU 可用"
else
    fail "未检测到 NVIDIA GPU"
fi

# ----- 1. 系统依赖 -----
section "[1/8] 系统依赖"
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq ffmpeg git wget bzip2
elif command -v yum &>/dev/null; then
    sudo yum install -y ffmpeg git wget bzip2
fi
check_ok "ffmpeg / git / wget / bzip2"

# ----- 2. Miniconda -----
section "[2/8] Miniconda"
if command -v conda &>/dev/null; then
    check_ok "conda 已安装 ($(conda --version 2>&1))"
else
    if [ ! -f "$HOME/miniconda3/bin/conda" ]; then
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
        rm -f /tmp/miniconda.sh
    fi
    eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
    conda init bash &>/dev/null
    check_ok "Miniconda 安装完成"
fi
eval "$("$HOME/miniconda3/bin/conda" shell.bash hook 2>/dev/null || conda shell.bash hook)"

# ----- 3. 克隆项目 -----
section "[3/8] 项目仓库"
# CloudStudio 默认工作目录为 /workspace，其他平台用 $HOME/ai_srt_jp2cn
if [ -d "/workspace" ] && [ -w "/workspace" ]; then
    PROJECT_DIR="/workspace"
else
    PROJECT_DIR="$HOME/ai_srt_jp2cn"
fi
if [ -d "$PROJECT_DIR/.git" ]; then
    cd "$PROJECT_DIR"
    git pull --ff-only
    check_ok "已更新到最新"
else
    git clone https://github.com/easonyujiang/ai_srt_jp2cn.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    check_ok "克隆完成"
fi
mkdir -p videos subtitles subtitles_translated temp models

# ----- 4. 配置 .env -----
section "[4/8] API 密钥"
if [ -f ".env" ] && grep -q "sk-" .env 2>/dev/null; then
    check_ok ".env 已配置"
else
    echo ""
    echo -e "${YELLOW}══════════════════════════════════════${NC}"
    echo -e "${YELLOW}  请先填入你的 API 密钥${NC}"
    echo -e "${YELLOW}══════════════════════════════════════${NC}"
    echo ""
    read -rp "  HuggingFace Token (hf_...):  " HF_TOKEN
    read -rp "  DeepSeek API Key  (sk-...): " DS_KEY
    cat > .env <<EOF
HF_TOKEN=${HF_TOKEN}
DEEPSEEK_API_KEY=${DS_KEY}
EOF
    check_ok ".env 写入完成"
fi

# ----- 5. 创建环境 -----
section "[5/8] 创建 6 个独立 conda 环境"

create_env() {
    local name=$1; local module_dir=$2; local req_file=$3; local extra_pip=$4
    if conda env list | grep -q "^${name} "; then
        echo "  ${name}  已存在，跳过"
        return
    fi
    echo "  >>> 创建 ${name}..."

    conda create -n "$name" python=3.10 -y -q -c conda-forge

    conda run -n "$name" pip install -q -r "$module_dir/requirements.txt" 2>/dev/null || true

    if [ -n "$extra_pip" ]; then
        eval "conda run -n $name pip install -q $extra_pip"
    fi

    echo "  ${GREEN}✓${NC} ${name} 完成"
}

# M1: 解复用  (无 Python 依赖，仅 ffmpeg)
conda create -n mod1_demux python=3.10 -y -q -c conda-forge 2>/dev/null || true
conda run -n mod1_demux conda install -y -q -c conda-forge ffmpeg 2>/dev/null || true
check_ok "mod1_demux  (ffmpeg 解复用)"

# M2: 人声分离 Demucs + PyTorch
create_env "mod2_separate" "Module2_separate" \
    "Module2_separate/requirements.txt" \
    "torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121"

# M3: ASR WhisperX + PyTorch（注意顺序：先装 torch，再装 whisperx，最后锁定 numpy）
conda create -n mod3_asr python=3.10 -y -q -c conda-forge
conda run -n mod3_asr pip install -q \
    torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
conda run -n mod3_asr pip install -q whisperx==3.1.1 python-dotenv
conda run -n mod3_asr pip install -q "numpy<2" "transformers>=4.36.0,<4.46.0"
check_ok "mod3_asr     (WhisperX ASR)"

# M4: 文本规范化 (OpenAI SDK)
create_env "mod4_normalize" "Module4_normalize" \
    "Module4_normalize/requirements.txt" ""

# M5: 字幕重建 (纯标准库)
conda create -n mod5_subtitle python=3.10 -y -q -c conda-forge 2>/dev/null || true
check_ok "mod5_subtitle (无第三方依赖)"

# M6: 翻译 (OpenAI SDK)
create_env "mod6_translate" "Module6_translate" \
    "Module6_translate/requirements.txt" ""

# Pipeline 主环境
pip install -q -r requirements_pipeline.txt 2>/dev/null || true
check_ok "pipeline 主环境"

# ----- 6. 验证 -----
section "[6/8] 验证各模块"

FAIL=0

verify_env() {
    local name=$1; local code=$2; local label=$3
    echo -n "  ${name}: "
    if conda run -n "$name" python -c "$code" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $label"
    else
        echo -e "${RED}✗${NC} $label"
        FAIL=1
    fi
}

verify_env "mod1_demux"   "import subprocess; print(subprocess.run(['ffmpeg','-version'],capture_output=True).returncode==0 and 'ffmpeg OK')" "ffmpeg"
verify_env "mod2_separate" "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}, {torch.cuda.get_device_name(0)}')" "Demucs"
verify_env "mod3_asr"     "import torch, whisperx; print(f'WhisperX {whisperx.__version__}, GPU: {torch.cuda.get_device_name(0)}')" "WhisperX"
verify_env "mod4_normalize" "from openai import OpenAI; print('openai OK')" "OpenAI SDK"
verify_env "mod5_subtitle" "print('stdlib OK')" "标准库"
verify_env "mod6_translate" "from openai import OpenAI; print('openai OK')" "OpenAI SDK"

if [ "$FAIL" -eq 1 ]; then
    warn "部分模块验证失败，请检查上方输出"
fi

# ----- 7. HuggingFace 镜像 -----
section "[7/8] HuggingFace 镜像"
if ! grep -q "HF_ENDPOINT" .env 2>/dev/null; then
    cat >> .env <<'EOF'

# 国内 HuggingFace 镜像加速
HF_ENDPOINT=https://hf-mirror.com
EOF
    check_ok "已添加 HF_ENDPOINT=https://hf-mirror.com"
else
    check_ok "HF_ENDPOINT 已配置"
fi

# ----- 8. 完成 -----
section "[8/8] 部署完成！"

echo ""
echo "  使用方法："
echo "  ──────────────────────────────────────"
echo "  # 上传视频"
echo "  scp video.mp4 root@<ip>:$PROJECT_DIR/videos/"
echo ""
echo "  # 运行流水线（命令行）"
echo "  cd $PROJECT_DIR"
echo "  python pipeline.py videos/你的视频.mp4"
echo ""
echo "  # 运行流水线（GUI）"
echo "  python pipeline.py"
echo "  ──────────────────────────────────────"
echo ""
echo "  输出目录："
echo "    源字幕:  $PROJECT_DIR/subtitles/"
echo "    中文字幕: $PROJECT_DIR/subtitles_translated/"
echo "    中间文件: $PROJECT_DIR/temp/"
echo ""
echo "  ⚠️ 用完记得关机：sudo shutdown -h now"
echo ""
