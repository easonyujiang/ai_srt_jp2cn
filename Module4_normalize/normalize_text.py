"""
Module 4: 文本规范化（完整版）
通过 DeepSeek API 对日语语音转写文本进行清洗。
Token 从项目根 .env 读取；支持娱乐直播风格保留、切片处理。
自动修复切片 original_index 偏移；API 超时重试。
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
import tempfile
import re

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


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, filepath):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(segments, style="retain"):
    combined = "\n".join([
        f"{i}: [{seg.get('speaker', 'UNKNOWN')}] {seg['text']}"
        for i, seg in enumerate(segments)
    ])

    if style == "clean":
        return f"""你是一个专业的日语文本后处理引擎。请对以下语音识别得到的日语文本进行规范化处理，严格遵循：

1. 去除无意义内容：删除填充词（如「えーと」「あのー」）、重复和结巴。
2. 假名转汉字：将应当写成汉字的纯假名转为标准汉字，保留助词和惯用假名。
3. 自动断句：若一段文本超过30字符，按语义拆分为多个片段，时间在原区间按比例分配。
4. 输出格式：返回 JSON 数组，每个元素必须包含：
   - start: 起始时间（继承或按比例分割）
   - end: 结束时间
   - text: 规范化后的文本
   - speaker: 说话人 ID（必须与输入片段中对应的 [说话人ID] 完全相同，一字不改）
   - original_index: 该片段对应的原始片段索引（整数，从0开始）。如一个原始片段拆分为多个，它们都使用同一个 original_index。
   - words: [] （空数组）
5. 不能遗漏任何原始片段。

原始片段（索引: [说话人] 文本）：
{combined}

请输出处理后的 JSON 数组（不要附加任何解释）：
"""
    else:
        return f"""
你是一个专为**多人娱乐直播录像**设计的日语文本后处理引擎。
在保证字幕可读性的前提下，**尽量保留原始对话的风格、语气和情感表达**。

规则：
1. **保留语气与拟声词**：不删除震惊、笑声、哭腔等情感词（如「えええ」「うそー」「あはは」），不合并有意的重复。
2. **网络梗与口癖**：保留角色用语和梗（「○○だお」「～にゃ」「草」「ワロタ」），不强行替换。
3. **假名转汉字**：仅在影响理解时转换（如「けっこうです」→「結構です」），保留日常口语的假名习惯，不转换专有名词和外号。
4. **最小化清洗**：仅删除纯赘字（无意义的「えーと」「あのー」和明显结巴），保留表达情绪的重复。
5. **自然断句**：建议每句不超过 40 字符，但不要割裂完整的情感爆发或梗的包袱。
6. **输出格式**：返回一个 JSON 数组，每个元素必须包含以下字段：
   - start: 起始时间（从原始对应片段继承，若拆分则按文本长度比例分配）
   - end: 结束时间
   - text: 清洗后的文本
   - speaker: 说话人 ID（必须与输入片段中对应的 [说话人ID] 完全相同，一字不改）
   - original_index: 该片段在输入原始片段列表中的索引（整数，从0开始）。如果原始一个片段被拆分为多个，则它们都使用同一个 original_index。
   - words: [] （空数组）
7. 确保原始所有片段都有对应输出，分段数可以变化。

原始片段（索引: [说话人] 文本）：
{combined}

请输出处理后的 JSON 数组（不要附加任何解释）：
"""


def call_deepseek(prompt, api_key):
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=API_TIMEOUT)
    for attempt in range(1, API_MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": "只输出要求的 JSON 数组，不添加任何额外文字。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=8000,
                extra_body={"thinking": {"type": "disabled"}},
            )
            return resp.choices[0].message.content
        except Exception as e:
            if attempt < API_MAX_RETRIES:
                wait = 2 ** attempt
                sys.stderr.write(f"[Module 4] API 调用失败 (第{attempt}次)，{wait}秒后重试: {e}\n")
                time.sleep(wait)
            else:
                raise


def parse_response(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r'^```(json)?', '', text)
        text = re.sub(r'```$', '', text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError("无法解析模型输出。")


def normalize_text(
    input_json,
    output_json=None,
    temp_dir=None,
    style="retain",
    max_segments_per_chunk=0,
    verbose=True,
):
    inp = Path(input_json).resolve()
    if not inp.is_file():
        raise FileNotFoundError(f"输入文件不存在: {inp}")

    if output_json is None:
        if temp_dir is None:
            temp_dir = Path(tempfile.gettempdir()) / "mod4_normalize"
        temp_dir = Path(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        outp = temp_dir / (inp.stem + "_normalized.json")
    else:
        outp = Path(output_json).resolve()
        outp.parent.mkdir(parents=True, exist_ok=True)

    segments = load_json(str(inp))
    if not segments:
        save_json([], str(outp))
        return outp

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key or "你的" in api_key or "sk-" not in api_key:
        raise RuntimeError("请在项目根 .env 文件中设置有效的 DEEPSEEK_API_KEY。")

    report_progress(0.0, "准备规范化...")

    if max_segments_per_chunk > 0 and len(segments) > max_segments_per_chunk:
        if verbose:
            print(f"[Module 4] 片段过多 ({len(segments)})，将按 {max_segments_per_chunk} 个一组切片处理。")
        normalized = []
        for i in range(0, len(segments), max_segments_per_chunk):
            chunk_start = i
            chunk = segments[chunk_start:chunk_start + max_segments_per_chunk]

            report_progress(
                10.0 + (chunk_start / len(segments)) * 80.0,
                f"处理中 {chunk_start+1}-{min(chunk_start+len(chunk), len(segments))}/{len(segments)} ..."
            )
            prompt = build_prompt(chunk, style=style)
            resp = call_deepseek(prompt, api_key)
            partial = parse_response(resp)

            for item in partial:
                if "original_index" in item and isinstance(item["original_index"], int):
                    item["original_index"] += chunk_start

            normalized.extend(partial)
            if verbose:
                print(f"  已完成 {min(chunk_start + max_segments_per_chunk, len(segments))}/{len(segments)} 片段")
    else:
        report_progress(15.0, f"调用 DeepSeek API (风格: {style})...")
        if verbose:
            print(f"[Module 4] 风格:{style}，调用 DeepSeek API...")
        prompt = build_prompt(segments, style=style)
        resp = call_deepseek(prompt, api_key)
        normalized = parse_response(resp)

    if verbose and len(normalized) != len(segments):
        print(f"  片段数量变化: {len(segments)} -> {len(normalized)}")

    report_progress(95.0, "保存结果...")
    save_json(normalized, str(outp))
    report_progress(100.0, "完成")
    if verbose:
        print(f"[Module 4] 完成。输出: {outp}")
    return outp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Module 4: 文本规范化")
    parser.add_argument("input", help="原始 ASR JSON 文件")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 路径")
    parser.add_argument("--temp-dir", default=None, help="临时目录")
    parser.add_argument("--style", default="retain", choices=["clean", "retain"],
                        help="清洗风格 (retain: 保留娱乐性, clean: 强规范化)")
    parser.add_argument("--max-segments-per-chunk", type=int, default=0,
                        help="切片大小，0 表示不切片；推荐值 50")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        outp = normalize_text(
            args.input, args.output, args.temp_dir, args.style,
            args.max_segments_per_chunk, verbose=not args.quiet,
        )
        print(f"输出: {outp}")
    except Exception as e:
        print(f"失败: {e}")
        sys.exit(1)
