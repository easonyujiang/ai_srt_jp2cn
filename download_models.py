#!/usr/bin/env python3
"""
备用模型下载器 - 适用于复杂网络环境

特性：
  - 通过 HF API 实时获取仓库文件列表（不依赖硬编码文件名）
  - 多镜像自动切换（国内镜像 → 官方源）
  - 自动重试 + 指数退避（最多 5 次）
  - tqdm 实时进度条（传输速度 + 预计剩余时间）
  - 详细时间戳日志
  - HTTP 超时保护，不会假死
  - 断点续传 + 下载完成自动校验
  - 门控模型权限预检 + 友好提示

用法：
  python download_models.py                     # 下载全部模型
  python download_models.py --skip-gated        # 跳过门控模型（无需 HF_TOKEN）
  python download_models.py --repo openai/whisper-large-v2  # 下载指定仓库
  python download_models.py --retry 3           # 最大重试 3 次
  python download_models.py --no-mirror         # 禁用镜像，直连 huggingface.co
"""
import logging
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

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


SKIP_FILE_PREFIXES = (".git", "README.md", ".md", ".lock")
SKIP_DIR_PREFIXES = ("reproducible_research/", ".github/")
SKIP_FILE_SUFFIXES = (".eval", ".rttm")


def _should_download(filename: str) -> bool:
    if filename.startswith(SKIP_FILE_PREFIXES):
        return False
    if filename.endswith(SKIP_FILE_SUFFIXES):
        return False
    for prefix in SKIP_DIR_PREFIXES:
        if filename.startswith(prefix):
            return False
    return True

# ============================================================
# 模型清单（文件列表从 HF API 动态获取）
# ============================================================
MODELS = [
    {
        "key": "whisper-large-v2",
        "repo": "openai/whisper-large-v2",
        "sub_dir": "whisper",
        "gated": False,
        "description": "Whisper 语音识别 (large-v2, ~6.5 GB)",
    },
    {
        "key": "wav2vec2-xlsr-53-japanese",
        "repo": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "sub_dir": "wav2vec2",
        "gated": False,
        "description": "Wav2Vec2 词级对齐 (日语, ~1.2 GB)",
    },
    {
        "key": "segmentation-3.0",
        "repo": "pyannote/segmentation-3.0",
        "sub_dir": "segmentation",
        "gated": True,
        "gated_accept_url": "https://hf-mirror.com/pyannote/segmentation-3.0",
        "description": "Pyannote 语音活动检测 (segmentation-3.0, ~380 MB)",
    },
    {
        "key": "speaker-diarization-3.1",
        "repo": "pyannote/speaker-diarization-3.1",
        "sub_dir": "diarization",
        "gated": True,
        "gated_accept_url": "https://hf-mirror.com/pyannote/speaker-diarization-3.1",
        "description": "Pyannote 说话人分割 Pipeline (diarization-3.1, 小文件)",
    },
]

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

# ============================================================
# 从 HF API 获取仓库文件列表
# ============================================================
def _api_url(mirror: str, repo: str) -> str:
    return f"{mirror}/api/models/{repo}"


def fetch_repo_files(repo: str, use_mirror: bool = True) -> List[str]:
    mirrors = MIRRORS if use_mirror else ["https://huggingface.co"]

    for mirror in mirrors:
        label = mirror.replace("https://", "")
        try:
            resp = requests.get(
                _api_url(mirror, repo),
                headers=_auth_headers(),
                timeout=(DEFAULT_CONNECT_TIMEOUT, 30),
            )
            if resp.status_code == 200:
                data = resp.json()
                siblings = data.get("siblings", [])
                files = [
                    s["rfilename"]
                    for s in siblings
                    if _should_download(s["rfilename"])
                ]
                logger.debug("  %s → %d 个文件 (%s)", label, len(files), repo)
                return files
            else:
                logger.debug("  %s API 返回 HTTP %d", label, resp.status_code)
        except Exception as e:
            logger.debug("  %s API 请求失败: %s", label, e)

    logger.warning("  ⚠ 无法从任何镜像获取文件列表，回退到空列表")
    return []


# ============================================================
# 核心下载逻辑
# ============================================================
def _head_file(mirror: str, repo: str, filename: str) -> Optional[requests.Response]:
    url = f"{mirror}/{repo}/resolve/main/{filename}"
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
            logger.debug("HEAD %s → HTTP %d (attempt %d)", url, resp.status_code, attempt)
        except requests.RequestException as e:
            logger.debug("HEAD %s error: %s (attempt %d)", url, e, attempt)
        time.sleep(1)
    return None


def _download_single_file(
    mirror: str,
    repo: str,
    filename: str,
    dest_path: Path,
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

    head_resp = _head_file(mirror, repo, filename)
    remote_size = int(head_resp.headers.get("content-length", 0)) if head_resp else 0
    local_size = get_file_size(dest_path)

    if remote_size > 0 and local_size == remote_size:
        logger.info("  ✓ 已存在: %s (%s)", filename, format_size(local_size))
        return True

    if local_size > 0:
        logger.info("  → 续传: %s (%s / %s)",
                     filename, format_size(local_size),
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
            if r.status_code == 401:
                logger.error("  ✗ HTTP 401 - 未授权（门控模型需要有效 HF_TOKEN 并接受协议）")
                return False
            if r.status_code == 403:
                logger.error("  ✗ HTTP 403 - 禁止访问")
                return False
            if r.status_code not in (200, 206):
                logger.warning("  HTTP %d (attempt %d/%d)", r.status_code, attempt, max_retries)
                return _download_single_file(
                    mirror, repo, filename, dest_path,
                    attempt=attempt + 1, max_retries=max_retries,
                )

            total_size = remote_size
            resume_pos = local_size if r.status_code == 206 else 0

            if r.status_code == 206 and not total_size:
                cr = r.headers.get("content-range", "")
                if "/" in cr:
                    total_size = int(cr.split("/")[-1])

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
                logger.warning("  ⚠ 大小不匹配: 本地 %s ≠ 远程 %s",
                               format_size(final_size), format_size(total_size))
                return _download_single_file(
                    mirror, repo, filename, dest_path,
                    attempt=attempt + 1, max_retries=max_retries,
                )

            logger.info("  ✓ %s (%s, %s/s, %s)",
                         filename, format_size(final_size),
                         format_size(int(speed)), format_duration(elapsed))
            return True

    except requests.ConnectionError as e:
        logger.warning("  ⚠ 连接失败: %s", e)
        if attempt < max_retries:
            return _download_single_file(
                mirror, repo, filename, dest_path,
                attempt=attempt + 1, max_retries=max_retries,
            )
        return False
    except requests.ReadTimeout:
        logger.warning("  ⚠ 读取超时")
        if attempt < max_retries:
            return _download_single_file(
                mirror, repo, filename, dest_path,
                attempt=attempt + 1, max_retries=max_retries,
            )
        return False
    except Exception as e:
        logger.error("  ✗ 意外错误: %s", e)
        if attempt < max_retries:
            return _download_single_file(
                mirror, repo, filename, dest_path,
                attempt=attempt + 1, max_retries=max_retries,
            )
        return False


def download_file_with_mirrors(
    repo: str,
    filename: str,
    dest_path: Path,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_mirror: bool = True,
) -> bool:
    mirrors = MIRRORS if use_mirror else ["https://huggingface.co"]

    for mi, mirror in enumerate(mirrors):
        label = mirror.replace("https://", "")
        if mi > 0:
            logger.info("  → 切换镜像: %s", label)

        success = _download_single_file(
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
# 门控模型权限预检
# ============================================================
def check_gated_access(repo: str, gated_accept_url: str) -> bool:
    if not HAS_TOKEN:
        return False

    files = fetch_repo_files(repo, use_mirror=False)
    if not files:
        logger.warning("  无法获取文件列表，跳过权限检查")
        return True

    test_file = files[0]
    try:
        resp = requests.head(
            f"https://huggingface.co/{repo}/resolve/main/{test_file}",
            headers=_auth_headers(),
            timeout=(DEFAULT_CONNECT_TIMEOUT, 30),
            allow_redirects=True,
        )
        if resp.status_code == 401:
            logger.error("  ✗ 无权访问门控模型！请先在浏览器中接受协议：")
            logger.error("    1. 打开 %s", gated_accept_url)
            logger.error("    2. 点击 'Agree and access repository'")
            logger.error("    3. 同时接受 segmentation-3.0 的协议（如尚未接受）")
            logger.info("  → 跳过此模型，完成授权后重试")
            return False
        elif resp.status_code == 200:
            logger.info("  ✓ 门控模型访问权限正常")
            return True
        else:
            logger.warning("  ⚠ 预检 HTTP %d，继续尝试下载", resp.status_code)
            return True
    except Exception as e:
        logger.warning("  ⚠ 权限预检失败: %s，继续尝试", e)
        return True


# ============================================================
# 模型下载入口
# ============================================================
def download_model(
    model_info: dict,
    dest_base: Path,
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_mirror: bool = True,
) -> tuple:
    repo = model_info["repo"]
    key = model_info["key"]
    gated = model_info["gated"]
    description = model_info["description"]
    sub_dir = model_info["sub_dir"]
    gated_accept_url = model_info.get("gated_accept_url", "")

    model_dest = dest_base / sub_dir
    model_dest.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("模型: %s", key)
    logger.info("仓库: %s", repo)
    logger.info("描述: %s", description)
    gate_label = "是（已配置 HF_TOKEN）" if gated and HAS_TOKEN else ("是（无 token，将跳过）" if gated else "否")
    logger.info("门控: %s", gate_label)
    logger.info("-" * 60)

    if gated and not HAS_TOKEN:
        logger.info("→ 跳过（需要 HF_TOKEN）")
        return 0, 0

    if gated and gated_accept_url:
        logger.info("验证门控访问权限...")
        if not check_gated_access(repo, gated_accept_url):
            return 0, 0

    logger.info("获取仓库文件列表...")
    files = fetch_repo_files(repo, use_mirror=use_mirror)
    if not files:
        logger.error("✗ 无法获取文件列表")
        return 0, 0

    logger.info("将下载 %d 个文件", len(files))
    for f in files[:5]:
        logger.debug("  - %s", f)
    if len(files) > 5:
        logger.debug("  ... 共 %d 个", len(files))
    logger.info("")

    success_count = 0
    fail_count = 0
    total_start = time.time()

    for fi, filename in enumerate(files, 1):
        logger.info("[%d/%d] %s", fi, len(files), filename)
        dest_path = model_dest / filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        success = download_file_with_mirrors(
            repo=repo,
            filename=filename,
            dest_path=dest_path,
            max_retries=max_retries,
            use_mirror=use_mirror,
        )

        if success:
            success_count += 1
        else:
            fail_count += 1
            logger.error("  ✗ 下载失败: %s", filename)

    elapsed = time.time() - total_start
    logger.info("-" * 60)
    logger.info("%s 完成: 成功 %d/%d, 耗时 %s",
                 key, success_count, len(files), format_duration(elapsed))

    if fail_count > 0:
        if gated:
            logger.warning("⚠ 门控模型失败排查：")
            logger.warning("  1. 接受协议: %s", gated_accept_url or f"https://hf-mirror.com/{repo}")
            logger.warning("  2. Token 需勾选 'Read access to content of gated repos'")
            logger.warning("  3. 中国 IP 试试: python download_models.py --no-mirror")
        else:
            logger.warning("⚠ 公开模型失败：检查网络 / VPN，或加 --retry 10 --no-mirror")
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
  python download_models.py --skip-gated           # 跳过门控模型
  python download_models.py --repo openai/whisper-large-v2  # 指定仓库
  python download_models.py --retry 10 --no-mirror # 直连 + 更多重试
  python download_models.py --list                 # 列出模型
        """,
    )
    parser.add_argument("--repo", type=str, default=None,
                        help="只下载指定仓库（如 openai/whisper-large-v2）")
    parser.add_argument("--skip-gated", action="store_true",
                        help="跳过门控模型")
    parser.add_argument("--retry", type=int, default=DEFAULT_MAX_RETRIES,
                        help=f"最大重试次数（默认 {DEFAULT_MAX_RETRIES}）")
    parser.add_argument("--no-mirror", action="store_true",
                        help="禁用国内镜像，直连 huggingface.co")
    parser.add_argument("--list", action="store_true",
                        help="列出模型清单后退出")
    parser.add_argument("--dest", type=str, default=str(PROJECT / "models_download"),
                        help="下载目标目录")
    args = parser.parse_args()

    if args.list:
        print("\n模型清单（文件列表动态获取，下为各仓库描述）:")
        print("-" * 60)
        for m in MODELS:
            gate_label = "🔒门控" if m["gated"] else "🌐公开"
            print(f"  {m['key']}")
            print(f"    仓库: {m['repo']}")
            print(f"    描述: {m['description']}")
            print(f"    类型: {gate_label}")
            print()
        return

    dest_dir = Path(args.dest)

    if not HAS_TOKEN:
        logger.warning("未配置 HF_TOKEN，门控模型将跳过")
        logger.warning("在 .env 中设置 HF_TOKEN 即可下载门控模型")

    logger.info("目标目录: %s", dest_dir)
    logger.info("镜像模式: %s", "禁用" if args.no_mirror else "启用 (%d 个镜像)" % len(MIRRORS))
    logger.info("最大重试: %d", args.retry)
    logger.info("")

    total_ok = 0
    total_fail = 0
    skipped_gated = []
    overall_start = time.time()

    for model_info in MODELS:
        if args.repo and model_info["repo"] != args.repo:
            continue
        if model_info["gated"] and (args.skip_gated or not HAS_TOKEN):
            logger.info("=" * 60)
            logger.info("模型: %s — 跳过（门控）", model_info["key"])
            logger.info("")
            skipped_gated.append(model_info["key"])
            continue

        ok, fail = download_model(
            model_info=model_info,
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
