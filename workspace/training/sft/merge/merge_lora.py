#!/usr/bin/env python3
"""
将 LoRA 权重合并到基础模型，生成完整的模型用于部署。

示例:
  python merge_lora.py \
    --base-model-path /scratch/youliang/models/decider_lora_2_merged \
    --lora-path /scratch/youliang/models/decider_grpo_1_500steps \
    --output-path /scratch/youliang/models/decider_grpo_1_500steps_merged
"""

import argparse
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    save_full_precision: bool = False,
) -> None:
    """合并 LoRA 权重到基础模型。"""
    print(f"正在加载基础模型: {base_model_path}")

    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    print(f"检测到模型类型: {config.model_type}")

    dtype = torch.float32 if save_full_precision else torch.bfloat16
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",  # 在 CPU 上操作以节省显存
        trust_remote_code=True,
    )

    print(f"正在加载 LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        torch_dtype=dtype,
    )

    print("正在合并 LoRA 权重...")
    model = model.merge_and_unload()

    print(f"正在保存合并后的模型: {output_path}")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)

    # 复制 processor/tokenizer 等文件，确保 vLLM 可以直接加载。
    processor = AutoProcessor.from_pretrained(lora_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    print("完成！合并后的模型已保存。")
    print(f"\n加载合并后模型的方式:")
    print(f"  model = Qwen2VLForConditionalGeneration.from_pretrained('{output_path}')")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基础模型")
    parser.add_argument("--base-model-path", required=True, help="基础模型路径")
    parser.add_argument("--lora-path", required=True, help="LoRA adapter 路径")
    parser.add_argument("--output-path", required=True, help="合并后模型输出目录")
    parser.add_argument(
        "--save-full-precision",
        action="store_true",
        help="保存为 fp32（默认保存为 bf16）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.base_model_path):
        print(f"错误：基础模型路径不存在: {args.base_model_path}")
        sys.exit(1)

    if not os.path.exists(args.lora_path):
        print(f"错误：LoRA 源路径不存在: {args.lora_path}")
        sys.exit(1)

    if os.path.exists(args.output_path):
        print(f"错误：目标路径已存在: {args.output_path}")
        print("请改用新的输出目录。")
        sys.exit(1)

    print("=" * 50)
    print("LoRA 权重合并")
    print("=" * 50)
    print(f"基础模型: {args.base_model_path}")
    print(f"LoRA 路径: {args.lora_path}")
    print(f"输出路径: {args.output_path}")
    print("=" * 50 + "\n")

    merge_lora_weights(
        args.base_model_path,
        args.lora_path,
        args.output_path,
        save_full_precision=args.save_full_precision,
    )


if __name__ == "__main__":
    main()
