"""
Module 6: 日译中翻译
通过 DeepSeek API 将日语字幕翻译为中文，保留口语化、网络化表达。
输入：M5 生成的字幕文件（SRT 或 ASS）
输出：翻译后的字幕文件（同格式），输出到指定目录。
Token 从项目根 .env 读取，支持切片翻译防止注意力分散。
API 超时重试，自动修复 JSON 解析回退。
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
API_TIMEOUT = 180
API_MAX_RETRIES = 3


def report_progress(percent: float, message: str = ""):
    sys.stderr.write(f"PROGRESS: {percent:.1f}% {message}\n")
    sys.stderr.flush()


def load_srt(filepath: str) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read().strip()
    blocks = re.split(r'\n\s*\n', content)
    entries = []
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        idx = lines[0].strip()
        timestamp = lines[1].strip()
        text_lines = lines[2:]
        text = "\n".join(text_lines).strip()
        speaker = None
        speaker_match = re.match(r'^\[([^\]]+)\]\s*(.*)', text)
        if speaker_match:
            speaker = speaker_match.group(1)
            text = speaker_match.group(2)
        entries.append({
            "index": idx,
            "start": timestamp.split(" --> ")[0],
            "end": timestamp.split(" --> ")[1],
            "text": text,
            "speaker": speaker
        })
    return entries


def load_ass(filepath: str) -> List[Dict]:
    entries = []
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith("Dialogue:"):
            parts = line.strip().split(',', 9)
            if len(parts) < 10:
                continue
            start = parts[1].strip()
            end = parts[2].strip()
            style = parts[3].strip()
            text = parts[9].strip()
            speaker = None
            speaker_match = re.match(r'^([^:]+):\s*(.*)', text)
            if speaker_match:
                speaker = speaker_match.group(1)
                text = speaker_match.group(2)
            entries.append({
                "style": style,
                "start": start,
                "end": end,
                "text": text.replace("\\N", "\n"),
                "speaker": speaker
            })
    return entries


def save_srt(entries: List[Dict], filepath: str):
    lines = []
    for i, entry in enumerate(entries, 1):
        start = entry["start"]
        end = entry["end"]
        text = entry["text"]
        speaker = entry.get("speaker")
        if speaker:
            text = f"[{speaker}] {text}"
        lines.append(str(i))
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_ass(entries: List[Dict], original_ass_path: str, filepath: str):
    with open(original_ass_path, "r", encoding="utf-8") as f:
        original_lines = f.readlines()
    header_lines = []
    for line in original_lines:
        if line.startswith("Dialogue:"):
            break
        header_lines.append(line)
    output_lines = header_lines.copy()
    for entry in entries:
        style = entry.get("style", "Default")
        start = entry["start"]
        end = entry["end"]
        text = entry["text"].replace("\n", "\\N")
        speaker = entry.get("speaker")
        if speaker:
            text = f"{speaker}: {text}"
        output_lines.append(f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text}\n")
    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(output_lines)


def build_translation_prompt(texts: List[str], style: str) -> str:
    combined = "\n".join([f"{i}: {t}" for i, t in enumerate(texts)])
    if style == "literal":
        instruction = "请将以下日语字幕逐句直译为中文，保持原有结构。"
    else:
        instruction = (
            "你是一个精通中日文的本地化专家，专为多人娱乐直播录像翻译字幕。"
            "请将以下日语字幕翻译为中文，务必：\n"
            "1. 信达雅：准确传达原意，如遇双关、梗、语气词等使用最贴切的中文网络用语或口语表达。\n"
            "2. 保留风格：保留原文的情感强度、语气（如震惊、吐槽、卖萌），不得平淡化。\n"
            "3. 适配字幕：每段译文尽量简洁，适合字幕滚动速度，必要时可断句或合并短句。\n"
            "4. 不翻译角色口癖、专有名词、固定名称（如「にゃんぱすー」可保留或创造对应中文梗）。\n"
            "5. 输出格式：必须返回一个 JSON 数组，每个元素包含 text 字段（对应序号的译文），"
            "与输入序号一一对应，不要遗漏或合并。"
        )
    return f"""{instruction}
输入字幕序号和原文：
{combined}
请输出 JSON 数组（不要附加任何解释）：
"""


def call_deepseek(prompt: str):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请在 .env 文件中设置 DEEPSEEK_API_KEY。")
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=API_TIMEOUT)
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "只输出要求的 JSON 数组。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=8000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < API_MAX_RETRIES:
                wait = 2 ** attempt
                sys.stderr.write(f"[Module 6] API 调用失败 (第{attempt}次)，{wait}秒后重试: {e}\n")
                time.sleep(wait)
            else:
                raise


def parse_translation_response(text: str, expected_count: int) -> List[str]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(json)?', '', text)
        text = re.sub(r'```$', '', text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                raise ValueError(f"无法解析模型输出（JSON 语法错误），原文前200字: {text[:200]}")
        else:
            raise ValueError(f"无法解析模型输出（未找到JSON数组），原文前200字: {text[:200]}")
    translations = []
    for item in data:
        if isinstance(item, dict) and "text" in item:
            translations.append(item["text"])
        elif isinstance(item, str):
            translations.append(item)
        else:
            translations.append(str(item))
    if len(translations) != expected_count:
        raise ValueError(
            f"译文数量 ({len(translations)}) 与原文数量 ({expected_count}) 不匹配。"
            f"\n原始响应前300字: {text[:300]}"
        )
    return translations


def translate_subtitle(
    input_file: str,
    output_file: str = None,
    output_dir: str = None,
    format: str = None,
    style: str = "creative",
    max_lines_per_chunk: int = 0,
    verbose: bool = True,
):
    input_path = Path(input_file).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or "你的" in api_key or "sk-" not in api_key:
        raise RuntimeError("请在项目根 .env 文件中设置有效的 DEEPSEEK_API_KEY。")

    if output_file:
        output_path = Path(output_file).resolve()
    else:
        if output_dir is None:
            output_dir = Path("..") / "subtitles_translated"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_path.stem
        ext = input_path.suffix
        output_path = output_dir / f"{stem}_cn{ext}"

    ext = input_path.suffix.lower()
    if format is None:
        format = "srt" if ext == ".srt" else "ass"

    report_progress(0.0, "解析字幕...")
    if format == "srt":
        entries = load_srt(str(input_path))
    else:
        entries = load_ass(str(input_path))

    if not entries:
        raise ValueError("未解析到任何字幕条目。")

    texts = [entry["text"] for entry in entries]
    total = len(texts)
    if verbose:
        print(f"[Module 6] 共 {total} 条字幕待翻译。风格: {style}")

    report_progress(10.0, f"翻译中 ({total} 条)...")
    if max_lines_per_chunk > 0 and total > max_lines_per_chunk:
        if verbose:
            print(f"[Module 6] 将按 {max_lines_per_chunk} 条切片翻译...")
        translations = []
        for i in range(0, total, max_lines_per_chunk):
            chunk = texts[i:i + max_lines_per_chunk]

            report_progress(
                10.0 + (i / total) * 80.0,
                f"翻译中 {i+1}-{min(i+len(chunk), total)}/{total} ..."
            )
            prompt = build_translation_prompt(chunk, style)
            resp = call_deepseek(prompt)
            chunk_trans = parse_translation_response(resp, len(chunk))
            translations.extend(chunk_trans)
            if verbose:
                print(f"  已完成 {min(i + max_lines_per_chunk, total)}/{total}")
    else:
        report_progress(15.0, "调用 DeepSeek API...")
        prompt = build_translation_prompt(texts, style)
        resp = call_deepseek(prompt)
        translations = parse_translation_response(resp, total)

    for entry, trans in zip(entries, translations):
        entry["text"] = trans

    report_progress(90.0, "保存中...")
    if format == "srt":
        save_srt(entries, str(output_path))
    else:
        save_ass(entries, str(input_path), str(output_path))

    report_progress(100.0, "完成")
    if verbose:
        print(f"[Module 6] 翻译完成，输出: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 6: 日译中字幕翻译")
    parser.add_argument("input", help="输入字幕文件 (SRT 或 ASS)")
    parser.add_argument("-o", "--output", default=None, help="输出文件路径")
    parser.add_argument("--output-dir", default=None,
                        help="输出目录（若未指定 output）")
    parser.add_argument("--format", choices=["srt", "ass"], default=None,
                        help="输出格式（默认与输入相同）")
    parser.add_argument("--style", choices=["creative", "literal"], default="creative",
                        help="翻译风格 (creative: 网络化/口语化, literal: 直译)")
    parser.add_argument("--max-lines-per-chunk", type=int, default=0,
                        help="每次翻译的字幕条数上限（建议 50）")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.quiet:
        print("[Module 6] 启动翻译...")

    try:
        output_path = translate_subtitle(
            input_file=args.input,
            output_file=args.output,
            output_dir=args.output_dir,
            format=args.format,
            style=args.style,
            max_lines_per_chunk=args.max_lines_per_chunk,
            verbose=not args.quiet,
        )
        print(f"输出: {output_path}")
    except Exception as e:
        print(f"翻译失败: {e}")
        sys.exit(1)
