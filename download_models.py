#!/usr/bin/env python3
"""
备用模型下载器 - 使用 huggingface_hub.snapshot_download 写入标准 HF 缓存

  下载的每个仓库自动写入 {HF_HOME}/hub/models--org--repo/snapshots/xxx/
  WhisperX / faster-whisper / transformers / pyannote 运行时同一套缓存，零配置

用法:
  python download_models.py                          # 下载全部 5 个模型
  python download_models.py --skip-gated             # 跳过门控模型
  python download_models.py --repo Systran/faster-whisper-large-v2  # 指定仓库
  python download_models.py --no-mirror              # 直连 huggingface.co
  python download_models.py --list                   # 列出模型
"""
import logging
import os
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
except ImportError:
    pass

os.environ.pop("HTTP_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("https_proxy", None)

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
logger = logging.getLogger("downloader")

DEFAULT_CACHE_DIR = PROJECT / "models"

HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HAS_TOKEN = bool(HF_TOKEN and "你的" not in HF_TOKEN and "hf_" in HF_TOKEN)
if HAS_TOKEN:
    logger.info("HF_TOKEN 已加载: %s...", HF_TOKEN[:12])
else:
    HF_TOKEN = None

MIRROR = os.environ.get("HF_ENDPOINT", "")
if MIRROR:
    logger.info("HF 镜像: %s", MIRROR)
else:
    logger.info("HF 镜像: 未设置（直连 huggingface.co）")

MODELS = [
    {
        "key": "faster-whisper-large-v2",
        "repo": "Systran/faster-whisper-large-v2",
        "gated": False,
        "description": "Faster-Whisper CT2 模型 (WhisperX 核心, ~3 GB)",
        "allow": ["model.bin", "config.json", "tokenizer.json",
                  "preprocessor_config.json", "vocabulary.txt"],
    },
    {
        "key": "wav2vec2-xlsr-53-japanese",
        "repo": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
        "gated": False,
        "description": "Wav2Vec2 词级对齐 (日语, ~1.2 GB)",
        "allow": ["pytorch_model.bin", "config.json",
                  "preprocessor_config.json", "special_tokens_map.json",
                  "vocab.json"],
    },
    {
        "key": "segmentation-3.0",
        "repo": "pyannote/segmentation-3.0",
        "gated": True,
        "gated_prompt": "https://hf-mirror.com/pyannote/segmentation-3.0",
        "description": "Pyannote 语音活动检测 (segmentation-3.0, ~380 MB)",
        "allow": ["pytorch_model.bin", "config.yaml"],
    },
    {
        "key": "speaker-diarization-3.1",
        "repo": "pyannote/speaker-diarization-3.1",
        "gated": True,
        "gated_prompt": "https://hf-mirror.com/pyannote/speaker-diarization-3.1",
        "description": "Pyannote 说话人分割 Pipeline",
        "allow": ["config.yaml", "handler.py", "requirements.txt"],
    },
    {
        "key": "whisper-large-v2",
        "repo": "openai/whisper-large-v2",
        "gated": False,
        "description": "Whisper 原始模型 (tokenizer/pyannote 参考, ~6.5 GB)",
        "allow": ["pytorch_model.bin", "config.json", "tokenizer.json",
                  "preprocessor_config.json", "added_tokens.json",
                  "normalizer.json", "vocab.json", "merges.txt",
                  "special_tokens_map.json", "tokenizer_config.json",
                  "generation_config.json"],
    },
    {
        "key": "spkrec-ecapa-voxceleb",
        "repo": "speechbrain/spkrec-ecapa-voxceleb",
        "gated": False,
        "description": "SpeechBrain ECAPA-TDNN 说话人嵌入 (~34 MB)",
        "allow": ["classifier.ckpt", "embedding_model.ckpt",
                  "hyperparams.yaml", "config.json",
                  "label_encoder.txt", "mean_var_norm_emb.ckpt"],
    },
]


def format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def _check_gated(repo: str, gated_prompt: str) -> bool:
    if not HAS_TOKEN:
        return False
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, token=HF_TOKEN)
        if hasattr(info, "gated") and info.gated and info.gated != "auto":
            logger.warning("  HF API 返回 gated=%s，可能未接受协议", info.gated)
    except Exception:
        pass
    return True


BASE_IGNORE = [
    "*.eval", "*.rttm", ".gitattributes", ".github/**",
    "reproducible_research/**", "*.msgpack", "*.h5",
    "*.safetensors", "*.png", "LICENSE", "*.md",
    "example.*", "flax_model.*", "tf_model.*",
]


def download_repo(repo: str, cache_dir: Path, allow_patterns: list = None,
                  gated: bool = False, prompt_url: str = "",
                  max_retries: int = 3) -> bool:
    logger.info("  仓库: %s", repo)
    logger.info("  目录: %s", cache_dir)
    if allow_patterns:
        logger.info("  文件: %s", ", ".join(allow_patterns))

    if gated and not HAS_TOKEN:
        logger.info("  → 跳过（需要 HF_TOKEN）")
        return False
    if gated and prompt_url and not _check_gated(repo, prompt_url):
        logger.info("  → 跳过（权限检查未通过）")
        return False

    os.environ["HF_HOME"] = str(cache_dir.resolve())

    last_error = None
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            wait = min(2 ** (attempt - 1), 30)
            logger.warning("  第 %d/%d 次重试，等待 %ds …", attempt, max_retries, wait)
            time.sleep(wait)

        try:
            kwargs = dict(
                repo_id=repo,
                token=HF_TOKEN,
                max_workers=4,
            )
            if allow_patterns:
                kwargs["allow_patterns"] = allow_patterns
            else:
                kwargs["ignore_patterns"] = BASE_IGNORE

            path = snapshot_download(**kwargs)
            logger.info("  ✓ 下载完成 → %s", path)
            return True

        except HfHubHTTPError as e:
            last_error = e
            code = getattr(getattr(e, "response", None), "status_code", 0)
            if code == 401:
                logger.error("  ✗ 401 未授权")
                if gated and prompt_url:
                    logger.error("     请先接受协议: %s", prompt_url)
                return False
            if code in (403, 404):
                logger.error("  ✗ HTTP %d", code)
                return False
            logger.warning("  ⚠ HTTP %d (attempt %d/%d)", code, attempt, max_retries)

        except Exception as e:
            last_error = e
            logger.warning("  ⚠ %s (attempt %d/%d)", e, attempt, max_retries)

    logger.error("  ✗ 失败: %s", last_error)
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="备用模型下载器 - M3 ASR 所需全部模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python download_models.py                        # 下载全部
  python download_models.py --skip-gated           # 跳过门控模型
  python download_models.py --repo Systran/faster-whisper-large-v2
  python download_models.py --no-mirror            # 直连 huggingface.co
  python download_models.py --list                 # 列出模型

环境变量:
  HF_TOKEN        HuggingFace 访问令牌（门控模型必须）
  HF_ENDPOINT     镜像地址（如 https://hf-mirror.com）
        """,
    )
    parser.add_argument("--repo", type=str, default=None)
    parser.add_argument("--skip-gated", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        print("\n模型清单:")
        print("-" * 60)
        for m in MODELS:
            gate = "🔒门控" if m["gated"] else "🌐公开"
            print(f"  {m['key']}")
            print(f"    仓库: {m['repo']}")
            print(f"    描述: {m['description']}")
            print(f"    类型: {gate}")
            print()
        return

    if args.no_mirror:
        os.environ.pop("HF_ENDPOINT", None)
        logger.info("已禁用镜像")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_TOKEN:
        logger.warning("未配置 HF_TOKEN，门控模型将跳过")

    logger.info("缓存目录: %s", cache_dir)
    logger.info("缓存结构: hub/models--org--repo/snapshots/xxx/ (标准 HF 格式)")
    logger.info("")

    total_ok = 0
    total_fail = 0
    overall_start = time.time()

    for m in MODELS:
        if args.repo and m["repo"] != args.repo:
            continue
        if m["gated"] and (args.skip_gated or not HAS_TOKEN):
            logger.info("[SKIP] %s — %s", m["key"], m["description"])
            logger.info("")
            continue

        logger.info("=" * 60)
        logger.info("%s — %s", m["key"], m["description"])
        logger.info("-" * 60)

        success = download_repo(
            repo=m["repo"],
            cache_dir=cache_dir,
            allow_patterns=m.get("allow"),
            gated=m["gated"],
            prompt_url=m.get("gated_prompt", ""),
            max_retries=args.retry,
        )
        if success:
            total_ok += 1
        else:
            total_fail += 1

    total_elapsed = time.time() - overall_start

    logger.info("=" * 60)
    logger.info("全部完成: 成功 %d, 失败 %d, 耗时 %s",
                 total_ok, total_fail, format_duration(total_elapsed))
    logger.info("缓存目录: %s", cache_dir)
    logger.info("下一步: 运行 pipeline.py 时 M3 会自动命中缓存")
    logger.info("=" * 60)

    if total_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
