"""
Module 3: ASR 转写与强制对齐
使用 WhisperX 输出词级时间戳与说话人标签。
Token 从项目根 .env 文件读取；模型缓存至项目 models/ 目录；
自动检测 GPU 显存，8GB 卡串行释放模型，大显存卡全驻留加速。
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path
import tempfile

import torch
import whisperx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_HF_MIRROR = "https://hf-mirror.com"
DEFAULT_MODEL_CACHE = PROJECT_ROOT / "models"

VRAM_LOW_THRESHOLD_GB = 12


def get_gpu_vram_gb():
    if not torch.cuda.is_available():
        return 0
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_mem
        vram = total_bytes / (1024 ** 3)
        if vram < 1.0:
            try:
                import subprocess
                r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                vram = float(r.stdout.strip().split("\n")[0]) / 1024.0
            except Exception:
                vram = max(vram, 16.0)
        return vram
    except Exception:
        return 0


def report_progress(percent: float, message: str = ""):
    sys.stderr.write(f"PROGRESS: {percent:.1f}% {message}\n")
    sys.stderr.flush()


def check_gpu():
    if torch.cuda.is_available():
        vram = get_gpu_vram_gb()
        print(f"[GPU] {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    else:
        print("[GPU] CUDA 不可用，将使用 CPU（速度较慢）")


def transcribe_and_align(
    audio_path: str,
    output_json: str = None,
    language: str = "ja",
    model_name: str = "large-v2",
    device: str = None,
    compute_type: str = None,
    model_cache_dir: str = None,
    temp_dir: str = None,
    min_speakers: int = None,
    max_speakers: int = None,
    verbose: bool = True,
) -> Path:
    audio_file = Path(audio_path).resolve()
    if not audio_file.is_file():
        raise FileNotFoundError(f"输入音频不存在: {audio_file}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if compute_type is None:
        compute_type = "float16" if device == "cuda" else "int8"

    vram_gb = get_gpu_vram_gb()
    low_vram = device == "cuda" and vram_gb < VRAM_LOW_THRESHOLD_GB
    if verbose:
        if low_vram:
            print(f"[Module 3] 低显存模式 ({vram_gb:.1f} GB)：将串行加载/释放模型")
        elif device == "cuda":
            print(f"[Module 3] 全速模式 ({vram_gb:.1f} GB)：模型全驻留 GPU")

    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir()) / "mod3_asr"
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if output_json is None:
        output_path = temp_dir / (audio_file.stem + "_asr.json")
    else:
        output_path = Path(output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    token = os.environ.get("HF_TOKEN", "")
    if not token or "你的token" in token or "hf_" not in token:
        raise RuntimeError(
            "请在项目根 .env 文件中设置有效的 HF_TOKEN。\n"
            "需要 HuggingFace 账号并接受 pyannote/speaker-diarization 模型协议。"
        )

    if os.environ.get("HF_ENDPOINT") is None and os.environ.get("HF_MIRROR"):
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_MIRROR
        if verbose:
            print(f"[镜像] 使用 {DEFAULT_HF_MIRROR}")

    if model_cache_dir:
        cache = Path(model_cache_dir).resolve()
    else:
        cache = DEFAULT_MODEL_CACHE.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    if verbose:
        print(f"[缓存] 模型目录: {cache}")
        hub = cache / "hub"
        cached_dirs = list(hub.glob("models--*/")) if hub.is_dir() else []
        if cached_dirs:
            print(f"[缓存] 已缓存 {len(cached_dirs)} 个模型（标准 HF 格式）")
        else:
            bins = list(hub.rglob("*.bin")) + list(hub.rglob("*.safetensors"))
            yamls = list(hub.rglob("config.yaml"))
            if bins or yamls:
                print(f"[缓存] 检测到 {len(bins)} 个 .bin + {len(yamls)} 个 config.yaml（非标准结构）")
                print("[缓存] 建议删除旧缓存后重新下载:")
                print("[缓存]   rm -rf models/hub && python download_models.py")
            else:
                print("[缓存] 无已缓存模型，首次使用将自动下载 (~13 GB)")
                print("[缓存] 可提前运行: python download_models.py")

    if min_speakers is None and max_speakers is None:
        if sys.stdin.isatty():
            print("\n=== 说话人数量设置 ===")
            try:
                min_str = input("最少说话人数 (直接回车跳过): ").strip()
                max_str = input("最多说话人数 (直接回车跳过): ").strip()
                if min_str:
                    min_speakers = int(min_str)
                if max_str:
                    max_speakers = int(max_str)
            except Exception:
                print("输入无效，使用默认值。")
        else:
            min_speakers = 1
            max_speakers = 5

    report_progress(0.0, "加载 ASR 模型...")

    # ======== 阶段 1: ASR 转写 ========
    if verbose:
        print(f"[Module 3] 加载语音识别模型 '{model_name}' ...")
    asr_options = {"word_timestamps": True}
    model = whisperx.load_model(
        model_name, device, compute_type=compute_type,
        language=language, asr_options=asr_options
    )
    audio = whisperx.load_audio(str(audio_file))

    report_progress(15.0, "语音识别中...")
    if verbose:
        print("[Module 3] 进行语音识别...")
    result = model.transcribe(audio, batch_size=16)

    if low_vram:
        if verbose:
            print("[Module 3] 释放 ASR 模型...")
        del model
        torch.cuda.empty_cache()

    # ======== 阶段 2: 词级对齐 ========
    report_progress(35.0, "加载对齐模型...")
    if verbose:
        print("[Module 3] 加载对齐模型 (wav2vec2) ...")
    model_a, metadata = whisperx.load_align_model(
        language_code=language, device=device
    )
    result_aligned = whisperx.align(
        result["segments"], model_a, metadata, audio, device,
        return_char_alignments=False
    )

    report_progress(55.0, "对齐完成")
    if low_vram:
        del model_a
        torch.cuda.empty_cache()

    # ======== 阶段 3: 说话人分离 ========
    report_progress(60.0, "加载说话人分离模型...")
    if verbose:
        print("[Module 3] 加载说话人分离模型 (pyannote) ...")
    if low_vram:
        torch.cuda.empty_cache()
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=token, device=device
    )

    diarize_kwargs = {}
    if min_speakers is not None:
        diarize_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        diarize_kwargs["max_speakers"] = max_speakers

    report_progress(70.0, "说话人分离中...")
    if verbose:
        print("[Module 3] 运行说话人分离...")
    diarize_segments = diarize_model(audio, **diarize_kwargs)

    result_final = whisperx.assign_word_speakers(diarize_segments, result_aligned)

    report_progress(85.0, "说话人分离完成")
    if low_vram:
        del diarize_model
        torch.cuda.empty_cache()

    # ======== 阶段 4: 保存 JSON ========
    report_progress(90.0, "保存结果...")
    output_data = []
    for seg in result_final["segments"]:
        seg_out = {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": seg.get("speaker", "UNKNOWN"),
            "words": []
        }
        for w in seg.get("words", []):
            seg_out["words"].append({
                "word": w.get("word", ""),
                "start": w.get("start"),
                "end": w.get("end"),
                "speaker": w.get("speaker", "UNKNOWN"),
            })
        output_data.append(seg_out)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    report_progress(100.0, "完成")
    torch.cuda.empty_cache()
    if verbose:
        print(f"[Module 3] 完成。输出: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 3: ASR 转写与对齐")
    parser.add_argument("audio", help="输入纯净人声 WAV 文件")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 路径（可选）")
    parser.add_argument("--language", default="ja", help="语言代码（默认 ja）")
    parser.add_argument("--model", default="large-v2", help="Whisper 模型名")
    parser.add_argument("--device", default=None, help="设备 (cuda/cpu)")
    parser.add_argument("--compute-type", default=None, help="精度 (float16/int8)")
    parser.add_argument("--model-cache-dir", default=str(DEFAULT_MODEL_CACHE),
                        help="模型缓存目录（默认: ../models）")
    parser.add_argument("--temp-dir", default=None, help="临时文件目录")
    parser.add_argument("--min-speakers", type=int, default=None,
                        help="最少说话人数（子进程模式默认 1）")
    parser.add_argument("--max-speakers", type=int, default=None,
                        help="最多说话人数（子进程模式默认 5）")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    args = parser.parse_args()

    if not args.quiet:
        check_gpu()

    try:
        result = transcribe_and_align(
            audio_path=args.audio,
            output_json=args.output,
            language=args.language,
            model_name=args.model,
            device=args.device,
            compute_type=args.compute_type,
            model_cache_dir=args.model_cache_dir,
            temp_dir=args.temp_dir,
            min_speakers=args.min_speakers,
            max_speakers=args.max_speakers,
            verbose=not args.quiet,
        )
        print(f"输出文件: {result}")
    except Exception as e:
        print(f"运行失败: {e}")
        sys.exit(1)