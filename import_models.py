#!/usr/bin/env python3
"""
手动模型导入脚本 — 将浏览器下载的模型文件导入标准 HF 缓存结构

用法:
  python import_models.py
  python import_models.py --source E:\path\to\models_download
  python import_models.py --check-only       # 仅检查，不导入
  python import_models.py --list-missing      # 列出需要下载的全部文件

  浏览器下载地址见脚本末尾注释。
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT / "models_download"
DEFAULT_TARGET = PROJECT / "models"

REPOS = {
    "Systran/faster-whisper-large-v2": {
        "dir": "models--Systran--faster-whisper-large-v2",
        "desc": "Faster-Whisper CT2 ASR 模型",
        "files": [
            "model.bin",
            "config.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "vocabulary.txt",
        ],
    },
    "jonatasgrosman/wav2vec2-large-xlsr-53-japanese": {
        "dir": "models--jonatasgrosman--wav2vec2-large-xlsr-53-japanese",
        "desc": "Wav2Vec2 词级对齐 (日语)",
        "files": [
            "pytorch_model.bin",
            "config.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "vocab.json",
        ],
    },
    "pyannote/segmentation-3.0": {
        "dir": "models--pyannote--segmentation-3.0",
        "desc": "Pyannote 语音活动检测",
        "files": [
            "pytorch_model.bin",
            "config.yaml",
        ],
    },
    "pyannote/speaker-diarization-3.1": {
        "dir": "models--pyannote--speaker-diarization-3.1",
        "desc": "Pyannote 说话人分割",
        "files": [
            "config.yaml",
        ],
    },
}

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

def _color(code, text):
    if sys.stdout.isatty():
        return f"{code}{text}{RESET}"
    return text

def _size_str(path):
    size = os.path.getsize(path)
    if size >= 1 << 30:
        return f"{size / (1 << 30):.2f} GB"
    if size >= 1 << 20:
        return f"{size / (1 << 20):.1f} MB"
    if size >= 1 << 10:
        return f"{size / (1 << 10):.1f} KB"
    return f"{size} B"


def _find_file(source_dir, repo_files):
    found = {}
    for fname in repo_files:
        candidates = list(source_dir.rglob(fname))
        if len(candidates) == 1:
            found[fname] = candidates[0]
        elif len(candidates) > 1:
            found[fname] = sorted(candidates, key=lambda p: len(p.parts))[0]
    return found


def _find_files_by_subdir(source_dir, repo_files):
    for d in sorted(source_dir.iterdir()):
        if not d.is_dir():
            continue
        found = {}
        for fname in repo_files:
            p = d / fname
            if p.is_file():
                found[fname] = p
        if found:
            yield d.name, found


def list_missing():
    print("\n需要下载的文件清单:\n")
    print(f"{'=' * 70}")
    for repo_id, info in REPOS.items():
        print(f"\n  {_color(CYAN, repo_id)}")
        print(f"  {info['desc']}")
        print(f"  {_color(YELLOW, 'https://hf-mirror.com/' + repo_id + '/tree/main')}")
        print()
        for fname in info["files"]:
            print(f"    {fname}")
    print(f"\n{'=' * 70}")
    print(f"\n下载后放入: {DEFAULT_SOURCE}")
    print("可以按仓库分子目录，也可以全部混放（脚本自动识别）。")


def scan_source(source_dir):
    print(f"\n扫描: {source_dir}\n")
    if not source_dir.is_dir():
        print(_color(RED, f"目录不存在: {source_dir}"))
        print("请先用浏览器下载模型文件放入该目录，或指定 --source 参数。")
        return None

    results = {}
    for repo_id, info in REPOS.items():
        found = _find_file(source_dir, info["files"])
        results[repo_id] = found

    subdir_results = {}
    for sub_name, sub_found in _find_files_by_subdir(source_dir, REPOS[next(iter(REPOS))]["files"]):
        for rid, info in REPOS.items():
            s = _find_file(Path(source_dir) / sub_name, info["files"])
            if len(s) >= len(info["files"]) - 1:
                subdir_results[rid] = s

    for rid, s in subdir_results.items():
        if rid not in results or len(s) > len(results[rid]):
            results[rid] = s

    return results


def show_status(results):
    print(f"{'模型仓库':<48} {'状态':<10} {'文件数'}")
    print("-" * 70)
    all_ok = True
    for repo_id, info in REPOS.items():
        found = results.get(repo_id, {})
        missing = [f for f in info["files"] if f not in found]
        if not missing:
            status = _color(GREEN, "✓ 完整")
        elif len(found) == 0:
            status = _color(RED, "✗ 未找到")
            all_ok = False
        else:
            status = _color(YELLOW, "△ 缺文件")
            all_ok = False
        print(f"  {repo_id:<46} {status:<16} {len(found)}/{len(info['files'])}")
        for mf in missing:
            print(f"    {_color(RED, '✗')} 缺少: {mf}")
    print()
    return all_ok


def import_models(results, target_dir):
    hub = target_dir / "hub"
    hub.mkdir(parents=True, exist_ok=True)

    snapshot_hash = "imported"
    print(f"导入到: {hub}\n")

    for repo_id, info in REPOS.items():
        found = results.get(repo_id, {})
        missing = [f for f in info["files"] if f not in found]
        if missing:
            print(f"  {_color(YELLOW, '跳过')} {info['dir']} (缺 {len(missing)} 文件)")
            for mf in missing:
                print(f"       缺少: {mf}")
            continue

        repo_dir = hub / info["dir"]
        snapshot_dir = repo_dir / "snapshots" / snapshot_hash
        refs_dir = repo_dir / "refs"

        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        refs_dir.mkdir(parents=True, exist_ok=True)

        for fname in info["files"]:
            src = found[fname]
            dst = snapshot_dir / fname
            shutil.copy2(src, dst)

        ref_file = refs_dir / "main"
        ref_file.write_text(snapshot_hash, encoding="utf-8")

        total_size = sum(
            os.path.getsize(snapshot_dir / f)
            for f in info["files"]
            if (snapshot_dir / f).is_file()
        )
        print(f"  {_color(GREEN, '✓')} {info['dir']}  ({_size_str(snapshot_dir)})")


def main():
    parser = argparse.ArgumentParser(
        description="手动模型导入 — 浏览器下载模型 → HF 标准缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python import_models.py                  # 扫描 models_download/ → 导入到 models/
  python import_models.py --list-missing   # 列出所有需要下载的文件
  python import_models.py --check-only     # 仅检查，不导入
  python import_models.py --source D:\\downloads   # 指定下载目录
        """,
    )
    parser.add_argument("--source", type=str, default=str(DEFAULT_SOURCE))
    parser.add_argument("--target", type=str, default=str(DEFAULT_TARGET))
    parser.add_argument("--list-missing", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    if args.list_missing:
        list_missing()
        return

    source_dir = Path(args.source)
    target_dir = Path(args.target)

    results = scan_source(source_dir)
    if results is None:
        sys.exit(1)

    show_status(results)

    if args.check_only:
        return

    import_models(results, target_dir)

    print(f"\n{'=' * 70}")
    print(f"导入完成。运行 pipeline 时 M3 将直接命中缓存。")
    print(f"HF_HOME = {target_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()


"""
浏览器下载地址 (hf-mirror.com):

1. Systran/faster-whisper-large-v2  (~2.9 GB):
   https://hf-mirror.com/Systran/faster-whisper-large-v2/tree/main

   文件: model.bin, config.json, tokenizer.json, preprocessor_config.json, vocabulary.txt

2. jonatasgrosman/wav2vec2-large-xlsr-53-japanese  (~1.2 GB):
   https://hf-mirror.com/jonatasgrosman/wav2vec2-large-xlsr-53-japanese/tree/main

   文件: pytorch_model.bin, config.json, preprocessor_config.json,
         special_tokens_map.json, vocab.json

3. pyannote/segmentation-3.0  (~380 MB, 门控):
   https://hf-mirror.com/pyannote/segmentation-3.0/tree/main

   文件: pytorch_model.bin, config.yaml

4. pyannote/speaker-diarization-3.1  (~1 KB, 门控):
   https://hf-mirror.com/pyannote/speaker-diarization-3.1/tree/main

   文件: config.yaml

提示:
   - 下载后放到 models_download/ 目录下（可以混放，脚本自动识别）
   - 也可以按仓库分子目录存放（脚本优先使用子目录匹配）
   - 门控模型 (pyannote) 需要在 huggingface 上手动接受协议后才能下载
"""
