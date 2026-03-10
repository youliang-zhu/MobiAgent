#!/usr/bin/env python3
"""
阶段 3 验证脚本：Grounder API 连通性与响应格式测试

用法:
  conda run -n grpo python workspace/training/grpo/test_grounder_api.py
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import re
import time
from pathlib import Path

from PIL import Image
from openai import OpenAI

from src.constants import GROUNDER_PROMPT_TEMPLATE


def img_to_b64(path: Path) -> str:
    img = Image.open(path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_bbox(text: str):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(description="Grounder API smoke test")
    parser.add_argument("--grounder_url", default="http://localhost:8001/v1", help="OpenAI-compatible base URL")
    parser.add_argument(
        "--data_dir",
        default="workspace/data/training_data/grpo_data",
        help="Directory containing GRPO images (*.jpg)",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    test_img_path = next(data_dir.glob("*.jpg"), None)
    if test_img_path is None:
        raise FileNotFoundError(f"No jpg found in {data_dir}")

    test_reasoning = "我需要点击页面上的搜索框"
    test_element = "位于屏幕顶部的搜索输入框"
    prompt = GROUNDER_PROMPT_TEMPLATE.format(reasoning=test_reasoning, description=test_element)

    print(f"Test image : {test_img_path}")
    client = OpenAI(api_key="0", base_url=args.grounder_url, timeout=args.timeout)
    b64 = img_to_b64(test_img_path)

    t0 = time.time()
    resp = client.chat.completions.create(
        model="",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0,
    ).choices[0].message.content
    elapsed = time.time() - t0

    print(f"Raw response ({elapsed:.1f}s): {str(resp)[:300]}")
    parsed = parse_bbox(resp if isinstance(resp, str) else str(resp))
    bbox = parsed["bbox"]
    print(f"Parsed bbox : {bbox}")

    assert len(bbox) == 4, "bbox 必须是 4 个元素"
    assert all(isinstance(v, (int, float)) for v in bbox), "bbox 元素必须是数字"
    assert elapsed < args.timeout, f"响应超时（{elapsed:.1f}s > {args.timeout:.1f}s）"
    print("\n✓ Grounder API 测试通过")


if __name__ == "__main__":
    main()

