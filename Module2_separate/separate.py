"""
Module 2: 音源分离（自动适配显存）
用法：python separate.py <输入音频> [输出人声] [--temp-dir <临时目录>]
低显存卡 (<12GB) 自动分片处理，大显存卡全速单次通过。
"""
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf
from demucs import pretrained
from demucs.apply import apply_model

DEMUCS_SR = 44100
VRAM_LOW_THRESHOLD_GB = 12
CHUNK_SECONDS = 8 * 60


def get_gpu_vram_gb():
    if not torch.cuda.is_available():
        return 0
    try:
        total_bytes = torch.cuda.get_device_properties(0).total_mem
    except Exception:
        total_bytes = 0
    if not total_bytes:
        try:
            total_bytes = torch.cuda.mem_get_info()[1]
        except Exception:
            total_bytes = 0
    return total_bytes / (1024 ** 3) if total_bytes else 0


def check_gpu():
    if torch.cuda.is_available():
        vram = get_gpu_vram_gb()
        print(f"[GPU] {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    else:
        print("[GPU] CUDA 不可用，将使用 CPU")


def report_progress(percent: float, message: str = ""):
    sys.stderr.write(f"PROGRESS: {percent:.1f}% {message}\n")
    sys.stderr.flush()


def _process_single(vocals_tensor_441, model, device, verbose):
    audio_input = vocals_tensor_441.unsqueeze(0).to(device)
    with torch.no_grad():
        sources = apply_model(model, audio_input, shifts=1, split=True, overlap=0.25, progress=verbose)
    vocals = sources[0, model.sources.index('vocals')]
    return vocals.mean(dim=0, keepdim=True)


def _prepare_audio(audio_tensor, sr):
    if sr != DEMUCS_SR:
        resampler = torchaudio.transforms.Resample(sr, DEMUCS_SR)
        audio_tensor = resampler(audio_tensor)
    if audio_tensor.shape[0] == 1:
        audio_tensor = audio_tensor.repeat(2, 1)
    return audio_tensor


def separate_audio(input_wav: str, output_vocals: str = None, temp_dir: Path = None, verbose: bool = True) -> Path:
    input_path = Path(input_wav).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入音频不存在: {input_path}")

    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir()) / "mod2_vocals"
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    if output_vocals is None:
        output_path = temp_dir / (input_path.stem + "_vocals.wav")
    else:
        output_path = Path(output_vocals).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

    report_progress(0.0, "加载音频...")
    raw_audio, sr = sf.read(input_path)
    if verbose and sr != 16000:
        print(f"[警告] 输入采样率 {sr}Hz，非 16kHz，将自动重采样。")

    audio_tensor = torch.from_numpy(raw_audio).float()
    if audio_tensor.ndim == 1:
        audio_tensor = audio_tensor.unsqueeze(0)

    audio_tensor = _prepare_audio(audio_tensor, sr)
    report_progress(5.0, "重采样完成")

    # 模型
    if verbose:
        print("[Module 2] 加载 htdemucs 模型...")
    report_progress(10.0, "加载 Demucs 模型...")
    model = pretrained.get_model('htdemucs')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()

    vram_gb = get_gpu_vram_gb()
    total_samples = audio_tensor.shape[1]
    chunk_samples = CHUNK_SECONDS * DEMUCS_SR
    use_chunks = device == 'cuda' and vram_gb < VRAM_LOW_THRESHOLD_GB and total_samples > chunk_samples

    if use_chunks:
        n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        if verbose:
            print(f"[Module 2] 低显存模式 ({vram_gb:.1f} GB)：将音频分为 {n_chunks} 个片段处理 ({CHUNK_SECONDS//60} 分钟/段)")

        vocals_chunks = []
        for i in range(n_chunks):
            start = i * chunk_samples
            end = min(start + chunk_samples, total_samples)
            seg = audio_tensor[:, start:end]

            report_progress(15.0 + (i / n_chunks) * 60.0, f"分离片段 {i+1}/{n_chunks} ...")
            if verbose:
                print(f"[Module 2] 处理片段 {i+1}/{n_chunks} ({seg.shape[1]/DEMUCS_SR:.0f}s)...")

            chunk_vocals = _process_single(seg, model, device, verbose)
            if chunk_vocals.shape[0] == 2:
                chunk_vocals = chunk_vocals.mean(dim=0, keepdim=True)
            vocals_chunks.append(chunk_vocals.cpu())
            torch.cuda.empty_cache()

        report_progress(80.0, "拼接片段...")
        vocals_441 = torch.cat(vocals_chunks, dim=1)
    else:
        if verbose and device == 'cuda':
            print(f"[Module 2] 全速模式 ({vram_gb:.1f} GB)")
        report_progress(15.0, f"音源分离中 (设备: {device})...")

        vocals_441 = _process_single(audio_tensor, model, device, verbose)

        report_progress(75.0, "提取人声...")
        if vocals_441.shape[0] == 2:
            vocals_441 = vocals_441.mean(dim=0, keepdim=True)

    del model
    torch.cuda.empty_cache()

    report_progress(85.0, "降采样...")
    resampler_back = torchaudio.transforms.Resample(DEMUCS_SR, 16000)
    vocals_16k = resampler_back(vocals_441.cpu())
    vocals_16k_np = vocals_16k.squeeze().numpy()

    report_progress(95.0, "写入文件...")
    sf.write(str(output_path), vocals_16k_np, 16000, subtype='PCM_16')
    report_progress(100.0, "完成")
    if verbose:
        print(f"[Module 2] 完成。输出: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 2: 音源分离")
    parser.add_argument("input", help="输入 16kHz 单声道 WAV 路径")
    parser.add_argument("output", nargs="?", default=None, help="输出人声 WAV 路径（可选）")
    parser.add_argument("--temp-dir", default=None, help="临时文件存放目录（可选）")
    parser.add_argument("--quiet", action="store_true", help="减少提示输出")
    args = parser.parse_args()

    check_gpu()
    temp_dir = Path(args.temp_dir) if args.temp_dir else None
    try:
        result = separate_audio(args.input, args.output, temp_dir=temp_dir, verbose=not args.quiet)
        print(f"输出文件: {result}")
    except Exception as e:
        print(f"失败: {e}")
        sys.exit(1)