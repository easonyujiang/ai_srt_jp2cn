#!/usr/bin/env python3
"""
cuDNN 8.9.x DLL 一键下载（Windows）

NVIDIA 要求登录才能下载 cuDNN，因此本脚本从 HuggingFace 镜像拉取
预打包的 cuDNN 8.9.7 DLL 文件。下载后自动放入项目 libs/ 目录，
transcribe_align.py 启动时会通过 os.add_dll_directory() 自动加载。

用法:
  python scripts/download_cudnn.py
  python scripts/download_cudnn.py --no-mirror   # 直连 huggingface.co

文件列表 (7 个 DLL, 约 900MB):
  cudnn64_8.dll
  cudnn_adv_infer64_8.dll
  cudnn_adv_train64_8.dll
  cudnn_cnn_infer64_8.dll
  cudnn_cnn_train64_8.dll
  cudnn_ops_infer64_8.dll
  cudnn_ops_train64_8.dll

Linux 用户无需此脚本:
  conda install -c conda-forge cudnn=8.9.*
"""
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LIBS = PROJECT / "libs"

DLL_FILES = [
    "cudnn64_8.dll",
    "cudnn_adv_infer64_8.dll",
    "cudnn_adv_train64_8.dll",
    "cudnn_cnn_infer64_8.dll",
    "cudnn_cnn_train64_8.dll",
    "cudnn_ops_infer64_8.dll",
    "cudnn_ops_train64_8.dll",
]

HF_REPO = "easonyujiang/cudnn8-win64"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="下载 cuDNN 8.9 DLL (Windows)")
    parser.add_argument("--no-mirror", action="store_true", help="直连 huggingface.co")
    args = parser.parse_args()

    if args.no_mirror:
        os.environ.pop("HF_ENDPOINT", None)
    else:
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("请先安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)

    LIBS.mkdir(parents=True, exist_ok=True)

    print(f"libs/ 目录: {LIBS}")
    print(f"共需下载 {len(DLL_FILES)} 个文件\n")

    ok = 0
    fail = 0

    for i, fname in enumerate(DLL_FILES, 1):
        dst = LIBS / fname
        if dst.is_file():
            size_mb = dst.stat().st_size / (1024 * 1024)
            print(f"  [{i}/{len(DLL_FILES)}] {fname}  ✓ 已存在 ({size_mb:.0f} MB)")
            ok += 1
            continue

        print(f"  [{i}/{len(DLL_FILES)}] {fname}  下载中...", end=" ", flush=True)
        try:
            path = hf_hub_download(
                HF_REPO,
                fname,
                cache_dir=str(LIBS),
                local_dir=str(LIBS),
                local_dir_use_symlinks=False,
            )
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"✓ ({size_mb:.0f} MB)")
            ok += 1
        except Exception as e:
            print(f"✗ {e}")
            fail += 1

    print(f"\n完成: 成功 {ok}, 失败 {fail}")

    if fail > 0:
        print("\n可手动下载:")
        print("  1. 访问 https://developer.nvidia.com/cudnn (需注册)")
        print("  2. 下载 cuDNN 8.9.x for CUDA 12.x (Windows)")
        print(f"  3. 解压后将 bin/ 下所有 .dll 放入 {LIBS}")
        sys.exit(1)

    if ok == len(DLL_FILES):
        print(f"\n全部就绪。transcribe_align.py 启动时将自动加载 {LIBS}")


if __name__ == "__main__":
    main()
