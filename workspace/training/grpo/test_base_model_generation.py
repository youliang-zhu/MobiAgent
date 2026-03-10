#!/usr/bin/env python3
"""
阶段 6 前验证：SFT 起点模型的基础生成能力检查

用法:
  conda run -n grpo python workspace/training/grpo/test_base_model_generation.py \
    --model_path /scratch/youliang/models/decider_lora_2_merged \
    --data_path workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json \
    --num_samples 5
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


def strip_image_prefix(instruction: str) -> str:
    text = (instruction or "").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n").strip()
    return text


def parse_action_type(text: str):
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[-1]
    cleaned = cleaned.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
    if m:
        cleaned = m.group(1).strip()
    else:
        # 取第一个 JSON 对象
        start = cleaned.find("{")
        if start >= 0:
            depth = 0
            end = None
            for i in range(start, len(cleaned)):
                if cleaned[i] == "{":
                    depth += 1
                elif cleaned[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            if end:
                cleaned = cleaned[start:end]

    try:
        obj = json.loads(cleaned)
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    # decider 常见格式: {"action":"click","parameters":{...}}
    if isinstance(obj.get("action"), str):
        action = obj.get("action", "").strip().lower()
        return action or None

    if isinstance(obj.get("function"), dict):
        return str(obj["function"].get("name", "")).lower() or None
    if isinstance(obj.get("action"), dict):
        return str(obj["action"].get("type") or obj["action"].get("name") or "").lower() or None
    return str(obj.get("type") or obj.get("name") or "").lower() or None


def main():
    parser = argparse.ArgumentParser(description="Base model generation sanity check for GRPO.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("data_path 不是有效的非空 JSON list")

    n = min(args.num_samples, len(data))
    samples = random.sample(data, n)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=dtype,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "sdpa",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    valid_json = 0
    action_counter = Counter()

    for idx, sample in enumerate(samples, 1):
        instruction = strip_image_prefix(sample.get("instruction", ""))
        img_path = Path(sample["images"][0])
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in inputs.items()}

        with torch.no_grad():
            # 显式关闭采样相关参数，避免 generation config 的 temperature 警告
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )

        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[:, prompt_len:]
        output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        action_type = parse_action_type(output_text)
        if action_type is not None:
            valid_json += 1
            action_counter[action_type] += 1

        print(f"[{idx}/{n}] action_type={action_type} output={output_text[:160]!r}")

    json_rate = valid_json / n
    print("\n===== Summary =====")
    print(f"num_samples: {n}")
    print(f"json_valid_rate: {json_rate:.2%}")
    print(f"action_type_distribution: {dict(action_counter)}")


if __name__ == "__main__":
    main()
