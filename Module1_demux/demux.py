"""
Module 1: 音视频解复用（支持公共临时文件夹）
用法：python demux.py <输入视频> [输出音频] [--temp-dir <临时目录>] [--no-progress] [--quiet] [--progress-style {auto,line,anim}]
说明：若未指定输出路径，则自动在临时目录生成 16k 单声道 WAV。
     临时目录优先级：--temp-dir 参数 > 项目 config.TEMP_DIR > 系统临时文件夹。
     默认显示提取进度条；可使用 --no-progress 禁用。
"""
import sys
import io

# ---------- 强制使用 UTF-8 输出，避免 Windows 下编码乱码 ----------
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import tempfile
import re
import subprocess
from pathlib import Path

TARGET_SR = 16000
TARGET_CHANNELS = 1
TARGET_CODEC = "pcm_s16le"

# ---------- 尝试读取项目全局配置的临时目录 ----------
try:
    _project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_project_root))
    from config import TEMP_DIR as _CONFIG_TEMP_DIR
    DEFAULT_TEMP_DIR = Path(_CONFIG_TEMP_DIR)
except (ImportError, AttributeError):
    DEFAULT_TEMP_DIR = None


def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def check_ffprobe():
    """检查 ffprobe 是否可用"""
    try:
        subprocess.run(
            ["ffprobe", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception:
        return False


def get_duration(input_path: Path) -> float:
    """使用 ffprobe 获取媒体时长（秒），失败返回 None"""
    if not check_ffprobe():
        return None
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(input_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return None
        duration_str = result.stdout.strip()
        if duration_str and duration_str != "N/A":
            return float(duration_str)
    except Exception:
        pass
    return None


def format_time(seconds: float) -> str:
    """将秒数转换为 HH:MM:SS 格式"""
    total_secs = int(seconds)
    h, rem = divmod(total_secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_time(time_str: str) -> float:
    """将 HH:MM:SS.mm 格式的时间字符串转换为秒"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    return 0.0


def extract_audio(input_video: str,
                  output_audio: str = None,
                  temp_dir: Path = None,
                  verbose: bool = True,
                  show_progress: bool = True,
                  progress_style: str = "auto") -> Path:
    """
    从视频提取音频，返回输出文件完整路径。
    若未指定 output_audio，则自动生成文件名；
    temp_dir 指定了存放位置（优先级：参数 > config > 系统默认）。
    show_progress: 是否显示进度（受 --no-progress 控制）。
    progress_style: "auto" -> 终端动画；"line" -> 可解析行；"anim" -> 强制动画。
    """
    input_path = Path(input_video).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {input_path}")

    # 确定临时目录
    if temp_dir is None:
        if DEFAULT_TEMP_DIR is not None:
            temp_dir = DEFAULT_TEMP_DIR
        else:
            temp_dir = Path(tempfile.gettempdir()) / "mod1_demux"
    else:
        temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 确定输出文件路径
    if output_audio is None:
        output_path = temp_dir / (input_path.stem + "_16k.wav")
    else:
        output_path = Path(output_audio).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # 获取视频总时长（用于进度计算）
    total_duration = get_duration(input_path)

    # 决定实际显示进度方式
    is_tty = sys.stderr.isatty()
    if show_progress:
        if progress_style == "auto":
            use_anim = is_tty
            use_line = not is_tty
        elif progress_style == "anim":
            use_anim = True
            use_line = False
        elif progress_style == "line":
            use_anim = False
            use_line = True
        else:
            use_anim = use_line = False
    else:
        use_anim = use_line = False

    # 构建 ffmpeg 命令
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-vn",
        "-acodec", TARGET_CODEC,
        "-ar", str(TARGET_SR),
        "-ac", str(TARGET_CHANNELS),
        "-y",
        str(output_path),
    ]
    if verbose and not show_progress:
        print(f"[Module 1] 音频提取: {input_path.name} → {output_path}")

    # 启动 ffmpeg，实时读取 stderr 以获取进度
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)

    time_pattern = re.compile(r"time=(\d+:\d+:\d+\.\d+)")
    stderr_lines = []
    last_progress_line = ""

    try:
        while True:
            line = proc.stderr.readline()
            if not line and proc.poll() is not None:
                break
            stderr_lines.append(line)

            match = time_pattern.search(line) if show_progress else None
            if match:
                current_time = parse_time(match.group(1))
                if total_duration and total_duration > 0:
                    percent = min(current_time / total_duration * 100, 100)
                    bar_length = 30
                    filled = int(bar_length * percent / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    progress_str = (f"[{bar}] {percent:.1f}%  "
                                    f"{format_time(current_time)}/{format_time(total_duration)}")
                else:
                    progress_str = f"处理时间: {format_time(current_time)}"

                if use_anim:
                    sys.stderr.write("\r" + progress_str)
                    sys.stderr.flush()
                elif use_line:
                    # 只在进度内容变化时输出新行，防止刷屏
                    if progress_str != last_progress_line:
                        sys.stderr.write(f"PROGRESS: {progress_str}\n")
                        sys.stderr.flush()
                        last_progress_line = progress_str
    finally:
        returncode = proc.wait()
        if use_anim:
            sys.stderr.write("\n")
            sys.stderr.flush()

    if returncode != 0 or not output_path.is_file():
        error_msg = "".join(stderr_lines)
        raise RuntimeError(f"FFmpeg 错误:\n{error_msg.strip()}")

    if verbose and not show_progress:
        print("[Module 1] 完成。")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 1: 音视频解复用")
    parser.add_argument("input", help="输入视频文件路径")
    parser.add_argument("output", nargs="?", default=None, help="输出 WAV 路径（可选）")
    parser.add_argument("--temp-dir", default=None, help="临时文件存放目录（可选）")
    parser.add_argument("--no-progress", action="store_true", help="禁用进度条显示")
    parser.add_argument("--quiet", action="store_true", help="减少提示输出")
    parser.add_argument("--progress-style", choices=["auto", "line", "anim"], default="auto",
                        help="进度显示风格：auto(自动) / line(可解析行) / anim(终端动画)")
    args = parser.parse_args()

    if not check_ffmpeg():
        print("错误：未找到 FFmpeg，请先安装。")
        sys.exit(1)

    try:
        result = extract_audio(
            args.input,
            args.output,
            temp_dir=args.temp_dir,
            verbose=not args.quiet,
            show_progress=not args.no_progress,
            progress_style=args.progress_style
        )
        # 即使 quiet 模式，最终输出文件路径也要打印，方便管道解析
        print(f"输出文件: {result}")
    except Exception as e:
        print(f"失败: {e}")
        sys.exit(1)