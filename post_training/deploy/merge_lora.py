#!/usr/bin/env python3
"""
将 LoRA 权重合并到基础模型，生成完整的模型用于部署
所有配置从 config.json 读取
"""

import torch
import json
import sys
import os
from pathlib import Path
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"错误：配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        return json.load(f)


def merge_lora_weights(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    save_full_precision: bool = False
):
    """
    合并 LoRA 权重到基础模型
    
    Args:
        base_model_path: 基础模型路径
        lora_path: LoRA adapter 路径
        output_path: 输出合并后模型的路径
        save_full_precision: 是否保存 fp32 精度（默认 bf16）
    """
    print(f"正在加载基础模型: {base_model_path}")
    
    # 加载基础模型
    dtype = torch.float32 if save_full_precision else torch.bfloat16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        device_map="cpu",  # 在 CPU 上操作以节省显存
        trust_remote_code=True,
    )
    
    print(f"正在加载 LoRA adapter: {lora_path}")
    
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(
        model,
        lora_path,
        torch_dtype=dtype,
    )
    
    print("正在合并 LoRA 权重...")
    
    # 合并权重
    model = model.merge_and_unload()
    
    print(f"正在保存合并后的模型: {output_path}")
    
    # 保存合并后的模型
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path, safe_serialization=True)
    
    # 复制 processor 文件
    processor = AutoProcessor.from_pretrained(lora_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    print("完成！合并后的模型已保存。")
    print(f"\n加载合并后模型的方式:")
    print(f"  model = Qwen2VLForConditionalGeneration.from_pretrained('{output_path}')")


def main():
    # 加载配置
    config = load_config()
    
    base_model_path = config["base_model_path"]
    lora_path = config["lora_source_path"]
    models_dir = config["models_dir"]
    merged_save_name = config["merged_save_name"]
    
    # 目标路径
    output_path = os.path.join(models_dir, merged_save_name)
    
    # 检查源路径是否存在
    if not os.path.exists(base_model_path):
        print(f"错误：基础模型路径不存在: {base_model_path}")
        sys.exit(1)
    
    if not os.path.exists(lora_path):
        print(f"错误：LoRA 源路径不存在: {lora_path}")
        sys.exit(1)
    
    # 检查目标路径是否已存在
    if os.path.exists(output_path):
        print(f"错误：目标路径已存在: {output_path}")
        print("请修改 config.json 中的 merged_save_name 为新的名称。")
        sys.exit(1)
    
    # 确保目标目录存在
    os.makedirs(models_dir, exist_ok=True)
    
    print("="*50)
    print("LoRA 权重合并")
    print("="*50)
    print(f"基础模型: {base_model_path}")
    print(f"LoRA 路径: {lora_path}")
    print(f"输出路径: {output_path}")
    print("="*50 + "\n")
    
    # 执行合并
    merge_lora_weights(
        base_model_path,
        lora_path,
        output_path,
        save_full_precision=False  # 使用 bf16
    )


if __name__ == "__main__":
    main()
