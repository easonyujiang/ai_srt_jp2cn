"""
项目全局配置文件
定义路径、环境名、模块脚本等，供 pipeline.py 使用
兼容 Windows / Linux 双平台
"""
import platform
import subprocess
from pathlib import Path

_IS_WINDOWS = platform.system() == "Windows"

# 项目根目录 (假设 config.py 位于项目根)
ROOT_DIR = Path(__file__).resolve().parent

# 公共临时文件夹（存放所有中间音频、JSON 等）
TEMP_DIR = ROOT_DIR / "temp"

# 模型缓存目录（M3 下载的 WhisperX 模型等）
MODEL_CACHE_DIR = ROOT_DIR / "models"

# 视频源文件目录（可选，也可直接传入视频路径）
VIDEO_DIR = ROOT_DIR / "videos"

# 输出字幕目录（M5 生成的源语言字幕）
SUBTITLE_OUTPUT_DIR = ROOT_DIR / "subtitles"

# 翻译后字幕输出目录（M6 生成）
TRANSLATED_SUBTITLE_DIR = ROOT_DIR / "subtitles_translated"

# ---------- Conda 环境信息（跨平台适配） ----------

def _detect_conda_envs_dir() -> Path:
    if _IS_WINDOWS:
        for candidate in [
            r"C:\ProgramData\anaconda3\envs",
            r"C:\ProgramData\miniconda3\envs",
        ]:
            p = Path(candidate)
            if p.exists():
                return p
        return Path.home() / "anaconda3" / "envs"
    else:
        try:
            result = subprocess.run(
                ["conda", "info", "--base"],
                capture_output=True, text=True, timeout=10
            )
            base = result.stdout.strip()
            if base:
                return Path(base) / "envs"
        except Exception:
            pass
        for candidate in [
            Path.home() / "miniconda3" / "envs",
            Path.home() / "anaconda3" / "envs",
            Path("/opt/conda/envs"),
            Path("/root/miniconda3/envs"),
        ]:
            if candidate.exists():
                return candidate
        return Path.home() / "miniconda3" / "envs"

CONDA_ENVS_DIR = _detect_conda_envs_dir()

def _env_python(env_name: str) -> Path:
    if _IS_WINDOWS:
        return CONDA_ENVS_DIR / env_name / "python.exe"
    else:
        return CONDA_ENVS_DIR / env_name / "bin" / "python"

# =============== 模块环境与脚本 ===============

# 格式：{"env": "conda环境名", "script": "脚本路径", "python": "环境内的python路径"}
MODULES = {
    "M1": {
        "env": "mod1_demux",
        "script": ROOT_DIR / "Module1_demux" / "demux.py",
        "python": _env_python("mod1_demux"),
    },
    "M2": {
        "env": "mod2_separate",
        "script": ROOT_DIR / "Module2_separate" / "separate.py",
        "python": _env_python("mod2_separate"),
    },
    "M3": {
        "env": "mod3_asr",
        "script": ROOT_DIR / "Module3_asr" / "transcribe_align.py",
        "python": _env_python("mod3_asr"),
    },
    "M4": {
        "env": "mod4_normalize",
        "script": ROOT_DIR / "Module4_normalize" / "normalize_text.py",
        "python": _env_python("mod4_normalize"),
    },
    "M5": {
        "env": "mod5_subtitle",
        "script": ROOT_DIR / "Module5_subtitle" / "rebuild_subtitle.py",
        "python": _env_python("mod5_subtitle"),
    },
    "M6": {
        "env": "mod6_translate",
        "script": ROOT_DIR / "Module6_translate" / "translate_subtitle.py",
        "python": _env_python("mod6_translate"),
    },
}

# =============== 默认参数（可在命令行覆盖） ===============

# M3: ASR & 对齐
DEFAULT_LANGUAGE = "ja"
DEFAULT_WHISPER_MODEL = "large-v2"
DEFAULT_MAX_SPEAKERS = 5           # 最多说话人数，可根据需要修改

# M4: 文本规范化
DEFAULT_NORM_STYLE = "retain"      # 娱乐直播风格，可选 "clean"
DEFAULT_NORM_CHUNK_SIZE = 50       # 切片大小

# M5: 字幕格式与过滤
DEFAULT_SUBTITLE_FORMAT = "ass"    # 输出格式，ass 或 srt
DEFAULT_MIN_DURATION = 0.5         # 过滤极短片段

# M6: 翻译
DEFAULT_TRANSLATION_STYLE = "creative"   # 网络化口语翻译
DEFAULT_TRANSLATION_CHUNK_SIZE = 50      # 翻译切片条数

# ---------- Conda 可执行文件路径（旧方式，已废弃） ----------
# CONDA_EXE = r"C:\ProgramData\anaconda3\Scripts\conda.exe"