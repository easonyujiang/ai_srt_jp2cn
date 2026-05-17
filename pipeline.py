#!/usr/bin/env python3
"""
全自动字幕生成流水线（带 GUI + 日志）
用法：
    python pipeline.py               # 启动 GUI（固定日语，遍历视频目录）
    python pipeline.py video.mp4 ... # 命令行模式（保留原参数）
"""
import argparse
import subprocess
import sys
import threading
import queue
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from config import *

# ---------- 日志系统 ----------
LOG_FILE = ROOT_DIR / "pipeline.log"

logger = logging.getLogger("Pipeline")
logger.setLevel(logging.DEBUG)

# 每次启动覆盖旧日志，编码 UTF-8
fh = logging.FileHandler(LOG_FILE, encoding="utf-8", mode="w")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.WARNING)
ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
logger.addHandler(ch)


class QueueHandler(logging.Handler):
    """将日志记录放入队列，供 GUI 线程读取"""
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)


# ---------- 流水线执行器 ----------
class PipelineRunner:
    def __init__(self, video_path, **kwargs):
        self.video_path = Path(video_path).resolve()
        self.args = kwargs

    @staticmethod
    def validate_environments():
        errors = []
        for key, mod in MODULES.items():
            python_exe = Path(mod["python"])
            script = Path(mod["script"])
            if not python_exe.exists():
                errors.append(f"{key}: Python 不存在 ({python_exe})")
            if not script.is_file():
                errors.append(f"{key}: 脚本不存在 ({script})")
        if errors:
            raise RuntimeError("环境检查失败:\n  " + "\n  ".join(errors))
        logger.info("所有模块环境检查通过。")

    def run(self, progress_callback=None):
        try:
            self.validate_environments()
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            SUBTITLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            final_output_dir = Path(self.args.get("output_dir")) if self.args.get("output_dir") else TRANSLATED_SUBTITLE_DIR
            final_output_dir.mkdir(parents=True, exist_ok=True)

            base_name = self.video_path.stem
            steps = 6

            def report(step_idx, step_progress=0.0):
                if progress_callback:
                    progress_callback(step_idx, steps, step_progress)

            # M1: 音频提取
            report(1, 0.0)
            wav_16k = TEMP_DIR / f"{base_name}_16k.wav"
            m1_progress = {"value": 0.0}

            def m1_line_handler(line):
                if line.startswith("PROGRESS:"):
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match:
                        m1_progress["value"] = float(match.group(1)) / 100.0
                        report(1, m1_progress["value"])

            self._run_module("M1", [
                str(self.video_path), str(wav_16k),
                "--temp-dir", str(TEMP_DIR),
                "--progress-style", "line",
                *(["--quiet"] if self.args.get("quiet") else [])
            ], line_handler=m1_line_handler)
            report(1, 1.0)

            # M2: 人声分离
            report(2, 0.0)
            vocals_wav = TEMP_DIR / f"{base_name}_vocals.wav"
            m2_progress = {"value": 0.0}

            def m2_line_handler(line):
                if line.startswith("PROGRESS:"):
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match:
                        m2_progress["value"] = float(match.group(1)) / 100.0
                        report(2, m2_progress["value"])

            self._run_module("M2", [
                str(wav_16k), str(vocals_wav),
                "--temp-dir", str(TEMP_DIR),
                *(["--quiet"] if self.args.get("quiet") else [])
            ], line_handler=m2_line_handler)
            report(2, 1.0)

            # M3: ASR + 对齐
            report(3, 0.0)
            asr_json = TEMP_DIR / f"{base_name}_asr.json"
            m3_progress = {"value": 0.0}

            def m3_line_handler(line):
                if line.startswith("PROGRESS:"):
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match:
                        m3_progress["value"] = float(match.group(1)) / 100.0
                        report(3, m3_progress["value"])

            self._run_module("M3", [
                str(vocals_wav),
                "-o", str(asr_json),
                "--language", self.args.get("language", "ja"),
                "--model", self.args.get("whisper_model", DEFAULT_WHISPER_MODEL),
                "--model-cache-dir", str(MODEL_CACHE_DIR),
                "--temp-dir", str(TEMP_DIR),
                "--min-speakers", str(self.args.get("min_speakers", 1)),
                "--max-speakers", str(self.args.get("max_speakers", DEFAULT_MAX_SPEAKERS)),
                *(["--quiet"] if self.args.get("quiet") else [])
            ], line_handler=m3_line_handler)
            report(3, 1.0)

            # M4: 文本规范化
            report(4, 0.0)
            normalized_json = TEMP_DIR / f"{base_name}_normalized.json"
            m4_progress = {"value": 0.0}

            def m4_line_handler(line):
                if line.startswith("PROGRESS:"):
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match:
                        m4_progress["value"] = float(match.group(1)) / 100.0
                        report(4, m4_progress["value"])

            self._run_module("M4", [
                str(asr_json),
                "-o", str(normalized_json),
                "--temp-dir", str(TEMP_DIR),
                "--style", self.args.get("norm_style", DEFAULT_NORM_STYLE),
                "--max-segments-per-chunk", str(DEFAULT_NORM_CHUNK_SIZE),
                *(["--quiet"] if self.args.get("quiet") else [])
            ], line_handler=m4_line_handler)
            report(4, 1.0)

            # M5: 字幕生成
            report(5, 0.0)
            subtitle_path = SUBTITLE_OUTPUT_DIR / f"{base_name}.{self.args.get('subtitle_format', DEFAULT_SUBTITLE_FORMAT)}"
            m5_progress = {"value": 0.0}

            def m5_line_handler(line):
                if line.startswith("PROGRESS:"):
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match:
                        m5_progress["value"] = float(match.group(1)) / 100.0
                        report(5, m5_progress["value"])

            self._run_module("M5", [
                str(asr_json), str(normalized_json),
                "-o", str(SUBTITLE_OUTPUT_DIR),
                "--format", self.args.get("subtitle_format", DEFAULT_SUBTITLE_FORMAT),
                "--min-duration", str(DEFAULT_MIN_DURATION),
                *(["--quiet"] if self.args.get("quiet") else [])
            ], line_handler=m5_line_handler)

            # 文件名修正
            raw_stem = asr_json.stem
            expected_subtitle = SUBTITLE_OUTPUT_DIR / f"{raw_stem}.{self.args.get('subtitle_format', DEFAULT_SUBTITLE_FORMAT)}"
            if expected_subtitle != subtitle_path:
                if expected_subtitle.exists():
                    expected_subtitle.rename(subtitle_path)
                    logger.info(f"M5 输出已重命名为: {subtitle_path}")
                else:
                    logger.warning(f"未找到 M5 输出文件: {expected_subtitle}")
            report(5, 1.0)

            # M6: 翻译字幕
            report(6, 0.0)
            translated_subtitle = final_output_dir / f"{base_name}_cn.{self.args.get('subtitle_format', DEFAULT_SUBTITLE_FORMAT)}"
            m6_progress = {"value": 0.0}

            def m6_line_handler(line):
                if line.startswith("PROGRESS:"):
                    match = re.search(r"(\d+\.?\d*)%", line)
                    if match:
                        m6_progress["value"] = float(match.group(1)) / 100.0
                        report(6, m6_progress["value"])

            self._run_module("M6", [
                str(subtitle_path),
                "-o", str(translated_subtitle),
                "--format", self.args.get("subtitle_format", DEFAULT_SUBTITLE_FORMAT),
                "--style", self.args.get("translation_style", DEFAULT_TRANSLATION_STYLE),
                "--max-lines-per-chunk", str(DEFAULT_TRANSLATION_CHUNK_SIZE),
                *(["--quiet"] if self.args.get("quiet") else [])
            ], line_handler=m6_line_handler)
            report(6, 1.0)

            logger.info(f"流水线成功完成！最终字幕: {translated_subtitle}")
            return True, str(translated_subtitle)

        except Exception as e:
            logger.error(f"流水线执行失败: {e}", exc_info=True)
            return False, str(e)

    def _run_module(self, module_key, args, line_handler=None):
        mod = MODULES[module_key]
        env_name = mod["env"]
        script = mod["script"]
        # 直接使用环境内的 python.exe，不再用 conda run
        python_exe = str(mod["python"])

        cmd = [python_exe, str(script)] + args

        logger.debug(f"启动 {module_key}: {' '.join(cmd)}")
        try:
            # 强制子进程使用 UTF-8 输出，并复制当前环境变量
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",       # 用 UTF-8 解码子进程输出
                bufsize=1,
                env=env,
            )

            # 同时读取 stdout 和 stderr，避免管道阻塞
            def read_stream(stream, stream_name):
                try:
                    for line in iter(stream.readline, ''):
                        if not line:
                            break
                        line = line.rstrip('\n')
                        if line:
                            logger.debug(f"[{module_key} {stream_name}] {line}")
                            if line_handler:
                                line_handler(line)
                except Exception:
                    logger.exception(f"读取 {module_key} {stream_name} 时出错")

            t_stdout = threading.Thread(target=read_stream, args=(proc.stdout, "stdout"), daemon=True)
            t_stderr = threading.Thread(target=read_stream, args=(proc.stderr, "stderr"), daemon=True)
            t_stdout.start()
            t_stderr.start()

            returncode = proc.wait()
            t_stdout.join(timeout=2)
            t_stderr.join(timeout=2)

            if returncode != 0:
                raise RuntimeError(
                    f"{module_key} 返回码 {returncode}，请查看上方日志了解详情。"
                )
            logger.info(f"{module_key} 正常结束")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{module_key} 执行超时")
        except Exception as e:
            logger.error(f"{module_key} 异常: {e}")
            raise


# ---------- GUI 部分 ----------
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


class PipelineGUI:
    def __init__(self):
        if not GUI_AVAILABLE:
            print("tkinter 不可用，无法启动 GUI")
            sys.exit(1)

        self.root = tk.Tk()
        self.root.title("AI 字幕生成流水线（日语→中文）")
        self.root.geometry("700x550")

        self.log_queue = queue.Queue()
        self.runner_thread = None
        self.progress_var = tk.DoubleVar()

        self.video_dir = VIDEO_DIR
        if not self.video_dir.exists():
            messagebox.showwarning("注意", f"视频目录不存在：{self.video_dir}")

        try:
            PipelineRunner.validate_environments()
        except RuntimeError as e:
            messagebox.showwarning("环境警告", str(e))

        self.setup_ui()
        self.setup_logging()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # 视频列表区域
        frame_list = ttk.LabelFrame(self.root, text="选择视频", padding=5)
        frame_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)

        self.video_listbox = tk.Listbox(frame_list, exportselection=False)
        scrollbar = ttk.Scrollbar(frame_list, orient=tk.VERTICAL, command=self.video_listbox.yview)
        self.video_listbox.configure(yscrollcommand=scrollbar.set)
        self.video_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(frame_list, text="刷新列表", command=self.refresh_list).pack(anchor=tk.NE, pady=2)

        # 参数设置
        frame_params = ttk.LabelFrame(self.root, text="参数设置", padding=5)
        frame_params.pack(fill=tk.X, padx=5, pady=2)

        row1 = ttk.Frame(frame_params)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="语言: 日本語").pack(side=tk.LEFT)
        ttk.Label(row1, text="Whisper模型:").pack(side=tk.LEFT, padx=10)
        self.model_var = tk.StringVar(value=DEFAULT_WHISPER_MODEL)
        ttk.Combobox(row1, textvariable=self.model_var,
                     values=["tiny", "base", "small", "medium", "large-v2"], width=10).pack(side=tk.LEFT)

        row2 = ttk.Frame(frame_params)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="最大说话人数:").pack(side=tk.LEFT)
        self.speakers_var = tk.IntVar(value=DEFAULT_MAX_SPEAKERS)
        ttk.Spinbox(row2, from_=1, to=10, textvariable=self.speakers_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Label(row2, text="字幕格式:").pack(side=tk.LEFT, padx=5)
        self.format_var = tk.StringVar(value=DEFAULT_SUBTITLE_FORMAT)
        ttk.Combobox(row2, textvariable=self.format_var, values=["ass", "srt"], width=5).pack(side=tk.LEFT)

        # 进度条
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=6)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

        # 日志区域
        frame_log = ttk.LabelFrame(self.root, text="运行日志", padding=5)
        frame_log.pack(fill=tk.BOTH, expand=True, padx=5, pady=2)
        self.log_text = tk.Text(frame_log, wrap=tk.WORD, state=tk.DISABLED, bg="black", fg="white")
        scrollbar_log = ttk.Scrollbar(frame_log, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar_log.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_log.pack(side=tk.RIGHT, fill=tk.Y)

        # 按钮区
        frame_btn = ttk.Frame(self.root, padding=5)
        frame_btn.pack(fill=tk.X)
        self.start_btn = ttk.Button(frame_btn, text="开始处理", command=self.start_pipeline)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(frame_btn, text="停止", command=self.stop_pipeline, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame_btn, text="清除日志", command=self.clear_log).pack(side=tk.RIGHT, padx=5)

        self.refresh_list()

    def setup_logging(self):
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setLevel(logging.DEBUG)
        logger.addHandler(self.queue_handler)
        self.poll_log_queue()

    def poll_log_queue(self):
        while not self.log_queue.empty():
            record = self.log_queue.get()
            msg = self.queue_handler.format(record)
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)
        self.root.after(200, self.poll_log_queue)

    def refresh_list(self):
        self.video_listbox.delete(0, tk.END)
        if not self.video_dir.exists():
            return
        video_exts = (".mp4", ".mkv", ".avi", ".mov", ".flv")
        files = [f.name for f in self.video_dir.iterdir() if f.is_file() and f.suffix.lower() in video_exts]
        for f in sorted(files):
            self.video_listbox.insert(tk.END, f)

    def get_selected_video(self):
        sel = self.video_listbox.curselection()
        if not sel:
            messagebox.showerror("错误", "请先从列表中选择一个视频文件")
            return None
        filename = self.video_listbox.get(sel[0])
        return self.video_dir / filename

    def start_pipeline(self):
        video_path = self.get_selected_video()
        if not video_path:
            return

        if not video_path.is_file():
            messagebox.showerror("错误", f"视频文件不存在: {video_path}")
            return

        args = {
            "language": "ja",
            "whisper_model": self.model_var.get(),
            "max_speakers": self.speakers_var.get(),
            "norm_style": DEFAULT_NORM_STYLE,
            "subtitle_format": self.format_var.get(),
            "translation_style": DEFAULT_TRANSLATION_STYLE,
            "output_dir": str(TRANSLATED_SUBTITLE_DIR),
            "quiet": False
        }

        self.start_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.progress_var.set(0)

        self.runner_thread = threading.Thread(
            target=self.execute_in_thread,
            args=(video_path, args),
            daemon=True
        )
        self.runner_thread.start()

    def execute_in_thread(self, video_path, args):
        runner = PipelineRunner(video_path, **args)

        def progress_callback(current, total, step_progress=0.0):
            self.root.after(0, self.update_progress, current, total, step_progress)

        success, msg = runner.run(progress_callback=progress_callback)
        self.root.after(0, self.pipeline_finished, success, msg)

    def update_progress(self, current_step, total_steps, step_progress=0.0):
        value = current_step - 1 + step_progress
        self.progress_var.set(value)

    def pipeline_finished(self, success, msg):
        self.start_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        if success:
            messagebox.showinfo("完成", f"字幕生成成功！\n文件: {msg}")
        else:
            messagebox.showerror("失败", f"流水线执行失败:\n{msg}")

    def stop_pipeline(self):
        messagebox.showwarning("注意", "无法直接中断正在运行的模块，请等待当前步骤完成。")

    def clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def on_close(self):
        if self.runner_thread and self.runner_thread.is_alive():
            if not messagebox.askokcancel("退出", "流水线正在运行，确定要退出吗？"):
                return
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ---------- 命令行入口 ----------
def run_cli(args):
    video_input = Path(args.video).resolve()
    if not video_input.is_file():
        logger.error(f"视频文件不存在: {video_input}")
        sys.exit(1)

    runner = PipelineRunner(
        video_input,
        language=args.language,
        whisper_model=args.whisper_model,
        max_speakers=args.max_speakers,
        norm_style=args.norm_style,
        subtitle_format=args.subtitle_format,
        translation_style=args.translation_style,
        output_dir=args.output_dir,
        quiet=args.quiet
    )

    def progress_cb(current, total, step_progress=0.0):
        print(f"进度: 步骤 {current}/{total} 完成 {step_progress*100:.1f}%")

    success, msg = runner.run(progress_callback=progress_cb)
    if success:
        print(f"\n[成功] 最终翻译字幕: {msg}")
    else:
        print(f"\n[错误] {msg}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="AI 字幕生成流水线 (M1-M6)",
            formatter_class=argparse.RawTextHelpFormatter,
        )
        parser.add_argument("video", help="输入视频文件路径")
        parser.add_argument("--language", default="ja", help="语言代码 (默认: ja)")
        parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL, help="Whisper 模型")
        parser.add_argument("--max-speakers", type=int, default=DEFAULT_MAX_SPEAKERS, help="最大说话人数")
        parser.add_argument("--norm-style", default=DEFAULT_NORM_STYLE, choices=["retain", "clean"])
        parser.add_argument("--subtitle-format", default=DEFAULT_SUBTITLE_FORMAT, choices=["ass", "srt"])
        parser.add_argument("--translation-style", default=DEFAULT_TRANSLATION_STYLE, choices=["creative", "literal"])
        parser.add_argument("--quiet", action="store_true", help="减少输出")
        parser.add_argument("--output-dir", default=None, help="最终输出目录")
        cli_args = parser.parse_args()

        logger.info("命令行模式启动")
        run_cli(cli_args)
    else:
        if not GUI_AVAILABLE:
            print("无法启动 GUI，请安装 tkinter 或使用命令行模式")
            sys.exit(1)
        logger.info("GUI 模式启动")
        app = PipelineGUI()
        app.run()