#!/usr/bin/env python3
"""
Web 前端服务器 - 为 AI 字幕生成流水线提供 Web 界面
启动方式: python web_server.py [--port 5000] [--host 0.0.0.0]
"""
import argparse
import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

try:
    from flask import Flask, render_template, request, jsonify, Response, send_from_directory
except ImportError:
    print("请先安装 Flask: pip install flask")
    sys.exit(1)

from config import *
from pipeline import PipelineRunner, logger

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"))


class WebPipelineManager:
    def __init__(self):
        self.event_queues = []
        self.lock = threading.Lock()
        self.current_task = None
        self.current_progress = {"step": 0, "total": 6, "step_progress": 0.0, "status": "idle"}
        self.running = False

    def broadcast(self, event_type, data):
        msg = json.dumps({"type": event_type, "data": data}, ensure_ascii=False)
        with self.lock:
            for q in self.event_queues:
                try:
                    q.put(msg)
                except Exception:
                    pass

    def subscribe(self):
        q = queue.Queue(maxsize=500)
        with self.lock:
            self.event_queues.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.event_queues:
                self.event_queues.remove(q)

    def run_pipeline(self, video_path, params):
        self.running = True
        self.current_progress = {"step": 0, "total": 6, "step_progress": 0.0, "status": "running",
                                 "video": str(Path(video_path).name)}
        self.broadcast("status", self.current_progress)

        try:
            runner = PipelineRunner(video_path, **params)

            def progress_callback(current, total, step_progress=0.0):
                step_names = ["", "音频提取", "人声分离", "ASR转写", "文本规范化", "字幕生成", "翻译"]
                self.current_progress = {
                    "step": current, "total": total, "step_progress": step_progress,
                    "status": "running",
                    "step_name": step_names[current] if current < len(step_names) else "",
                    "video": str(Path(video_path).name)
                }
                self.broadcast("progress", self.current_progress)

            PipelineRunner.validate_environments()
            self.broadcast("log", "环境检查通过，开始流水线...")
            success, msg = runner.run(progress_callback=progress_callback)

            if success:
                self.current_progress = {"step": 6, "total": 6, "step_progress": 1.0, "status": "completed",
                                         "result": msg, "video": str(Path(video_path).name)}
                self.broadcast("status", self.current_progress)
                self.broadcast("log", f"流水线成功完成！最终字幕: {msg}")
            else:
                self.current_progress = {"step": 0, "total": 6, "step_progress": 0.0, "status": "error",
                                         "error": str(msg), "video": str(Path(video_path).name)}
                self.broadcast("status", self.current_progress)
                self.broadcast("log", f"流水线执行失败: {msg}")

        except Exception as e:
            err_msg = f"{e}\n{traceback.format_exc()}"
            self.current_progress = {"step": 0, "total": 6, "step_progress": 0.0, "status": "error",
                                     "error": str(e), "video": str(Path(video_path).name)}
            self.broadcast("status", self.current_progress)
            self.broadcast("log", f"错误: {err_msg}")
        finally:
            self.running = False


manager = WebPipelineManager()


class PipelineLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        manager.broadcast("log", msg)


log_handler = PipelineLogHandler()
log_handler.setLevel(logging.DEBUG)
log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(log_handler)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/videos")
def list_videos():
    video_exts = (".mp4", ".mkv", ".avi", ".mov", ".flv")
    videos = []
    video_dir = VIDEO_DIR
    if video_dir.exists():
        for f in sorted(video_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in video_exts:
                stat = f.stat()
                size_mb = stat.st_size / (1024 * 1024)
                videos.append({
                    "name": f.name,
                    "path": str(f),
                    "size_mb": round(size_mb, 1),
                    "mtime": stat.st_mtime
                })
    return jsonify({"videos": videos, "video_dir": str(video_dir)})


@app.route("/api/upload", methods=["POST"])
def upload_video():
    if "file" not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "文件名为空"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".mkv", ".avi", ".mov", ".flv"):
        return jsonify({"error": f"不支持的视频格式: {ext}"}), 400

    video_dir = VIDEO_DIR
    video_dir.mkdir(parents=True, exist_ok=True)
    save_path = video_dir / file.filename
    file.save(str(save_path))
    manager.broadcast("log", f"上传成功: {file.filename}")
    return jsonify({"success": True, "name": file.filename, "path": str(save_path)})


@app.route("/api/run", methods=["POST"])
def run_pipeline():
    if manager.running:
        return jsonify({"error": "流水线正在运行中，请等待完成"}), 409

    data = request.get_json()
    if not data:
        return jsonify({"error": "无效的请求数据"}), 400

    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"error": "未指定视频路径"}), 400

    video_file = Path(video_path)
    if not video_file.is_file():
        return jsonify({"error": f"视频文件不存在: {video_path}"}), 404

    params = {
        "language": data.get("language", DEFAULT_LANGUAGE),
        "whisper_model": data.get("whisper_model", DEFAULT_WHISPER_MODEL),
        "max_speakers": int(data.get("max_speakers", DEFAULT_MAX_SPEAKERS)),
        "norm_style": data.get("norm_style", DEFAULT_NORM_STYLE),
        "subtitle_format": data.get("subtitle_format", DEFAULT_SUBTITLE_FORMAT),
        "translation_style": data.get("translation_style", DEFAULT_TRANSLATION_STYLE),
        "output_dir": str(TRANSLATED_SUBTITLE_DIR) if not data.get("output_dir") else data["output_dir"],
        "quiet": data.get("quiet", False),
        "device": data.get("device", "cuda"),
        "compute_type": data.get("compute_type", "int8"),
    }

    manager.broadcast("log", f"开始处理视频: {video_file.name}")
    thread = threading.Thread(target=manager.run_pipeline, args=(video_path, params), daemon=True)
    thread.start()

    return jsonify({"success": True, "message": "流水线已启动"})


@app.route("/api/status")
def get_status():
    return jsonify(manager.current_progress)


@app.route("/api/stream")
def stream_events():
    q = manager.subscribe()

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'connected', 'data': manager.current_progress}, ensure_ascii=False)}\n\n"
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield f"data: {msg}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'type': 'heartbeat', 'data': None})}\n\n"
        except GeneratorExit:
            pass
        finally:
            manager.unsubscribe(q)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/subtitles")
def list_subtitles():
    subtitles = []
    for d in [SUBTITLE_OUTPUT_DIR, TRANSLATED_SUBTITLE_DIR]:
        if d.exists():
            for f in sorted(d.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                if f.is_file() and f.suffix.lower() in (".srt", ".ass"):
                    subtitles.append({
                        "name": f.name,
                        "path": str(f),
                        "size_kb": round(f.stat().st_size / 1024, 1),
                        "dir": d.name,
                        "mtime": f.stat().st_mtime
                    })
    return jsonify({"subtitles": subtitles})


@app.route("/api/download")
def download_subtitle():
    name = request.args.get("name", "")
    if not name:
        return jsonify({"error": "缺少文件名参数"}), 400

    for dir_path in (SUBTITLE_OUTPUT_DIR, TRANSLATED_SUBTITLE_DIR):
        file_path = dir_path / name
        if file_path.is_file():
            return send_from_directory(str(dir_path), name, as_attachment=True)

    return jsonify({"error": "文件不存在"}), 404


@app.route("/api/env-check")
def env_check():
    result = {"valid": True, "errors": [], "video_dir_exists": VIDEO_DIR.exists()}
    try:
        PipelineRunner.validate_environments()
    except RuntimeError as e:
        result["valid"] = False
        result["errors"] = str(e).split("\n")
    return jsonify(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="字幕流水线 Web 服务器")
    parser.add_argument("--port", type=int, default=5000, help="服务端口 (默认: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="绑定地址 (默认: 127.0.0.1)")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  AI 字幕生成流水线 Web 服务器")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  按 Ctrl+C 停止服务")
    print(f"{'='*60}\n")

    app.run(host=args.host, port=args.port, debug=False, threaded=True)
