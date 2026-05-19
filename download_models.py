#!/usr/bin/env python3
"""
备用模型下载器 - 适用于复杂网络环境

特性：
  - 多镜像自动切换（国内镜像 → 官方源）
  - 自动重试 + 指数退避（最多 5 次）
  - tqdm 实时进度条（传输速度 + 预计剩余时间）
  - 详细时间戳日志
  - HTTP 超时保护，不会假死
  - 断点续传 / 跳过已下载且校验通过的文件
  - 下载完成后自动校验文件大小
  - 支持公开模型和门控模型（HF_TOKEN）

用法：
  python download_models.py                     # 下载全部模型
  python download_models.py --skip-gated        # 跳过门控模型（无需 HF_TOKEN）
  python download_models.py --model whisper     # 只下载 whisper
  python download_models.py --retry 3           # 最大重试 3 次
  python download_models.py --no-mirror         # 禁用镜像，直连 huggingface.co
"""
import hashlib
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
except ImportError:
    pass

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("downloader")

# ============================================================
# 配置
# ============================================================

MIRRORS = [
    "https://hf-mirror.com",
    "https://huggingface.co",
]

DEFAULT_MAX_RETRIES = 5
DEFAULT_CONNECT_TIMEOUT = 15
DEFAULT_READ_TIMEOUT = 60

CHUNK_SIZE = 8 * 1024 * 1024

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
if HF_TOKEN and "你的" not in HF_TOKEN and "hf_" in HF_TOKEN:
    logger.info("HF_TOKEN 已加载: %s...", HF_TOKEN[:12])
    HAS_TOKEN = True
else:
    HF_TOKEN = ""
    HAS_TOKEN = False


def _auth_headers():
    if HAS_TOKEN:
        return {"Authorization": f"Bearer {HF_TOKEN}"}
    return {}

# ============================================================
# 模型清单
# ============================================================
MODELS = {
    "whisper-large-v2": {
        "repo": "openai/whisper-large-v2",
        "dir": "whisper",
        "gated": False,
        "description": "Whisper 语音识别 (large-v2, ~6.5 GB)",
        "files": [
            "pytorch_model.bin",
            "config.json",
            "tokenizer.json",
            "preprocessor_config.json",
            "added_tokens.json",
            "normalizer.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "generation_config.json",
        ],
    },
    "speaker-diarization-3.1": {
        "repo": "pyannote/speaker-diarization-3.1",
        "dir": "diarization",
        "gated": True,
        "description": "Pyannote 说话人分割 (diarization-3.1, ~800 MB)",
        "files": [
            "pytorch_model.bin",
            "config.json",
        ],
    },
    "segmentation-3.0": {
        "repo": "pyannote/segmentation-3.0",
        "dir": "segmentation",
        "gated": True,
        "description": "Pyannote 语音活动检测 (segmentation-3.0, ~380 MB)",
        "files": [
            "pytorch_model.bin",
            "config.json",
        ],
    },
    "wav2vec2-xlsr-53-japanese": {
        "repo": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "dir": "wav2vec2",
        "gated": False,
        "description": "Wav2Vec2 词级对齐 (日语, ~1.2 GB)",
        "files": [
            "pytorch_model.bin",
            "config.json",
            "preprocessor_config.json",
            "vocab.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ],
    },
}

# ============================================================
# 工具函数
# ============================================================
def format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 ** 3:
        return f"{size_bytes / (1024**3):.2f} GB"
    if size_bytes >= 1024 ** 2:
        return f"{size_bytes / (1024**2):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def get_file_size(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def md5_hex(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ============================================================
# 核心下载逻辑
# ============================================================
def _fetch_url(url: str, stream: bool = False, timeout: tuple = None):
    if timeout is None:
        timeout = (DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT)
    return requests.get(
        url,
        headers=_auth_headers(),
        stream=stream,
        timeout=timeout,
    )


def _head_url(url: str) -> Optional[requests.Response]:
    for attempt in range(1, 4):
        try:
            resp = requests.head(
                url,
                headers=_auth_headers(),
                timeout=(DEFAULT_CONNECT_TIMEOUT, 30),
                allow_redirects=True,
            )
            if resp.status_code == 200:
                return resp
            logger.debug("HEAD %s → HTTP %d (attempt %d/3)", url, resp.status_code, attempt)
        except requests.RequestException as e:
            logger.debug("HEAD %s error: %s (attempt %d/3)", url, e, attempt)
        time.sleep(1)
    return None


def _get_expected_size(mirror: str, repo: str, filename: str) -> int:
    url = f"{mirror}/{repo}/resolve/main/{filename}"
    resp = _head_url(url)
    if resp is not None:
        size = resp.headers.get("content-length")
        if size:
            return int(size)
    return 0


def _try_download_file(
    mirror: str,
    repo: str,
    filename: str,
    dest_path: Path,
    expected_size: int = 0,
    attempt: int = 1,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> bool:
    url = f"{mirror}/{repo}/resolve/main/{filename}"

    if attempt > max_retries:
        return False

    if attempt > 1:
        backoff = min(2 ** (attempt - 1), 60)
        logger.warning("第 %d/%d 次重试，等待 %ds …", attempt, max_retries, backoff)
        time.sleep(backoff)

    resp = _head_url(url)
    remote_size = 0
    if resp is not None:
        remote_size = int(resp.headers.get("content-length", 0))
        if remote_size == 0:
            logger.warning("无法获取远程文件大小，跳过校验")

    local_size = get_file_size(dest_path)

    if remote_size > 0 and local_size == remote_size:
        logger.info("  ✓ 已存在且大小一致: %s (%s)", filename, format_size(local_size))
        return True

    if local_size > 0:
        logger.info("  → 文件不完整 (%s / %s)，续传下载…",
                     format_size(local_size),
                     format_size(remote_size) if remote_size else "?")

    headers = _auth_headers()
    if local_size > 0 and remote_size > 0:
        headers["Range"] = f"bytes={local_size}-"
    elif local_size > 0:
        dest_path.unlink(missing_ok=True)
        local_size = 0

    try:
        with requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=(DEFAULT_CONNECT_TIMEOUT, DEFAULT_READ_TIMEOUT),
        ) as r:
            if r.status_code not in (200, 206):
                if r.status_code == 401 or r.status_code == 403:
                    logger.error("  ✗ HTTP %d - 门控模型需有效的 HF_TOKEN", r.status_code)
                    return False
                logger.warning("  HTTP %d (attempt %d/%d)", r.status_code, attempt, max_retries)
                return _try_download_file(
                    mirror, repo, filename, dest_path,
                    expected_size=expected_size,
                    attempt=attempt + 1, max_retries=max_retries,
                )

            total_size = remote_size
            resume_pos = local_size if r.status_code == 206 else 0

            if r.status_code == 206 and not total_size:
                content_range = r.headers.get("content-range", "")
                if "/" in content_range:
                    total_size = int(content_range.split("/")[-1])

            mode = "ab" if r.status_code == 206 else "wb"
            initial_pos = resume_pos

            if total_size == 0:
                total_size = int(r.headers.get("content-length", 0))

            desc = filename if len(filename) < 30 else filename[:27] + "…"
            with open(dest_path, mode) as f, \
                 tqdm(
                    total=total_size,
                    initial=initial_pos,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=desc,
                    mininterval=0.5,
                    maxinterval=2.0,
                    smoothing=0.1,
                    leave=False,
                ) as pbar:
                start_time = time.time()
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            elapsed = time.time() - start_time
            final_size = get_file_size(dest_path)
            speed = final_size / elapsed if elapsed > 0 else 0

            if total_size > 0 and final_size != total_size:
                logger.warning("  ⚠ 文件大小不匹配: 本地 %s ≠ 远程 %s",
                               format_size(final_size), format_size(total_size))
                return _try_download_file(
                    mirror, repo, filename, dest_path,
                    expected_size=total_size,
                    attempt=attempt + 1, max_retries=max_retries,
                )

            logger.info("  ✓ %s (%s, %s/s, 耗时 %s)",
                         filename,
                         format_size(final_size),
                         format_size(int(speed)),
                         format_duration(elapsed))
            return True

    except requests.ConnectionError as e:
        logger.warning("  ⚠ 连接失败: %s", e)
        if attempt < max_retries:
            return _try_download_file(
                mirror, repo, filename, dest_path,
                expected_size=expected_size,
                attempt=attempt + 1, max_retries=max_retries,
            )
        return False
    except requests.ReadTimeout:
        logger.warning("  ⚠ 读取超时（%ds 无数据）", DEFAULT_READ_TIMEOUT)
        if attempt < max_retries:
            return _try_download_file(
                mirror, repo, filename, dest_path,
                expected_size=expected_size,
                attempt=attempt + 1, max_retries=max_retries,
            )
        return False
    except Exception as e:
        logger.error("  ✗ 意外错误: %s", e)
        if attempt < max_retries:
            return _try_download_file(
                mirror, repo, filename, dest_path,
                expected_size=expected_size,
                attempt=attempt + 1, max_retries=max_retries,
            )
        return False


def download_file_with_mirrors(
    repo: str,
    filename: str,
    dest_path: Path,
    gated: bool = False,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_mirror: bool = True,
) -> bool:
    mirrors = MIRRORS if use_mirror else ["https://huggingface.co"]

    for mi, mirror in enumerate(mirrors):
        label = mirror.replace("https://", "")
        if mi == 0:
            logger.debug("  尝试镜像: %s", label)
        else:
            logger.info("  → 切换镜像: %s", label)

        success = _try_download_file(
            mirror=mirror,
            repo=repo,
            filename=filename,
            dest_path=dest_path,
            max_retries=max_retries,
        )
        if success:
            return True

    return False


# ============================================================
# 模型下载入口
# ============================================================
def download_model(
    model_key: str,
    model_info: dict,
    dest_base: Path,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_mirror: bool = True,
) -> tuple:
    repo = model_info["repo"]
    gated = model_info["gated"]
    description = model_info["description"]
    files = model_info["files"]
    sub_dir = model_info["dir"]

    model_dest = dest_base / sub_dir
    model_dest.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("模型: %s", model_key)
    logger.info("仓库: %s", repo)
    logger.info("描述: %s", description)
    logger.info("文件: %d 个", len(files))
    logger.info("存储: %s", model_dest)
    if gated and not HAS_TOKEN:
        logger.info("门控: 是（未配置 HF_TOKEN，将跳过）")
    elif gated:
        logger.info("门控: 是（已配置 HF_TOKEN）")
    logger.info("-" * 60)

    success_count = 0
    fail_count = 0
    total_start = time.time()

    for fi, filename in enumerate(files, 1):
        logger.info("[%d/%d] %s", fi, len(files), filename)
        dest_path = model_dest / filename

        success = download_file_with_mirrors(
            repo=repo,
            filename=filename,
            dest_path=dest_path,
            gated=gated,
            max_retries=max_retries,
            use_mirror=use_mirror,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            logger.error("  ✗ %s 下载失败", filename)

    elapsed = time.time() - total_start
    logger.info("-" * 60)
    logger.info("模型 %s 完成: 成功 %d, 失败 %d, 耗时 %s",
                 model_key, success_count, fail_count, format_duration(elapsed))
    logger.info("")

    return success_count, fail_count


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="备用模型下载器 - M3 ASR 所需全部模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_models.py                        # 下载全部
  python download_models.py --model whisper        # 只下载 whisper
  python download_models.py --skip-gated           # 跳过门控模型
  python download_models.py --retry 3              # 最多重试 3 次
  python download_models.py --no-mirror            # 禁用国内镜像
  python download_models.py --list                 # 列出模型清单
        """,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="只下载指定模型（key 名）")
    parser.add_argument("--skip-gated", action="store_true",
                        help="跳过门控模型（无需 HF_TOKEN）")
    parser.add_argument("--retry", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"最大重试次数（默认 {DEFAULT_MAX_RETRIES}）")
    parser.add_argument("--no-mirror", action="store_true",
                        help="禁用国内镜像，直连 huggingface.co")
    parser.add_argument("--list", action="store_true",
                        help="列出所有模型清单后退出")
    parser.add_argument("--dest", type=str, default=str(PROJECT / "models_download"),
                        help="下载目标目录")
    args = parser.parse_args()

    if args.list:
        print("\n模型清单:")
        print("-" * 60)
        for key, info in MODELS.items():
            gate_label = "🔒门控" if info["gated"] else "🌐公开"
            print(f"  {key}")
            print(f"    仓库: {info['repo']}")
            print(f"    描述: {info['description']}")
            print(f"    文件: {len(info['files'])} 个  {gate_label}")
            print()
        return

    dest_dir = Path(args.dest)

    if args.skip_gated and not HAS_TOKEN:
        logger.info("门控模型: 已跳过（--skip-gated + 无 HF_TOKEN）")
    elif not HAS_TOKEN:
        logger.warning("未配置 HF_TOKEN，门控模型将跳过")
        logger.warning("如需下载门控模型，请在项目根 .env 中设置 HF_TOKEN")
        logger.warning("或使用 --skip-gated 明确跳过")

    logger.info("目标目录: %s", dest_dir)
    logger.info("镜像模式: %s", "禁用" if args.no_mirror else "启用 (%d 个镜像)" % len(MIRRORS))
    logger.info("最大重试: %d", args.retry)
    logger.info("")

    total_ok = 0
    total_fail = 0
    skipped_gated = []

    overall_start = time.time()

    for key, info in MODELS.items():
        if args.model and key != args.model:
            continue
        if info["gated"] and (args.skip_gated or not HAS_TOKEN):
            logger.info("=" * 60)
            logger.info("模型: %s — 跳过（门控 / 无 token）", key)
            logger.info("")
            skipped_gated.append(key)
            continue

        ok, fail = download_model(
            model_key=key,
            model_info=info,
            dest_base=dest_dir,
            max_retries=args.retry,
            use_mirror=not args.no_mirror,
        )
        total_ok += ok
        total_fail += fail

    total_elapsed = time.time() - overall_start

    logger.info("=" * 60)
    logger.info("全部完成！")
    logger.info("  成功: %d 个文件", total_ok)
    if total_fail:
        logger.error("  失败: %d 个文件", total_fail)
    if skipped_gated:
        logger.info("  跳过(门控): %s", ", ".join(skipped_gated))
    logger.info("  总耗时: %s", format_duration(total_elapsed))
    logger.info("  存储目录: %s", dest_dir)
    logger.info("=" * 60)

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
