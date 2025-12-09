#!/usr/bin/env python3
"""
LoRA 部署脚本
模式1 (--save): 将训练好的 LoRA 权重（不含 checkpoint）复制到指定目录
模式2 (--deploy): 使用 vLLM 部署模型

使用方法:
  python deploy_lora.py --save    # 仅保存 LoRA 权重
  python deploy_lora.py --deploy  # 仅部署模型（需要先保存）
  python deploy_lora.py --save --deploy  # 保存并部署

注意：
  - 训练脚本需要启用 --load_best_model_at_end True
  - 这样主目录的 adapter_model.safetensors 就是 best 模型
  - checkpoint-X 文件夹是用于恢复训练的，不需要保存
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_config():
    """加载配置文件"""
    config_path = Path(__file__).parent / "config.json"
    if not config_path.exists():
        print(f"错误：配置文件不存在: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        return json.load(f)


def save_lora(config):
    """
    保存 LoRA 权重到指定目录（不含 checkpoint 文件夹）
    
    主目录的 adapter_model.safetensors 就是 best 模型
    （前提是训练时启用了 --load_best_model_at_end True）
    """
    lora_source_path = config["lora_source_path"]
    models_save_dir = config["models_save_dir"]
    lora_save_name = config["lora_save_name"]
    
    # 目标路径
    lora_dest_path = os.path.join(models_save_dir, lora_save_name)
    
    # 检查源路径是否存在
    if not os.path.exists(lora_source_path):
        print(f"错误：LoRA 源路径不存在: {lora_source_path}")
        sys.exit(1)
    
    # 检查目标路径是否已存在
    if os.path.exists(lora_dest_path):
        print(f"错误：目标路径已存在: {lora_dest_path}")
        print("请修改 config.json 中的 lora_save_name 为新的名称。")
        sys.exit(1)
    
    # 确保目标目录存在
    os.makedirs(lora_dest_path, exist_ok=True)
    
    # 需要复制的文件（不包含 checkpoint-X 文件夹）
    files_to_copy = [
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "config.json",
        "README.md",
        "trainer_state.json",  # 保留训练日志
    ]
    
    print(f"正在复制 LoRA 权重（不含 checkpoint）...")
    print(f"  源路径: {lora_source_path}")
    print(f"  目标路径: {lora_dest_path}")
    
    copied_count = 0
    for fname in files_to_copy:
        src = os.path.join(lora_source_path, fname)
        dst = os.path.join(lora_dest_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied_count += 1
            print(f"    复制: {fname}")
    
    # 复制 runs/ 目录（TensorBoard 日志）
    runs_src = os.path.join(lora_source_path, "runs")
    runs_dst = os.path.join(lora_dest_path, "runs")
    if os.path.exists(runs_src):
        shutil.copytree(runs_src, runs_dst)
        print(f"    复制: runs/")
    
    print(f"\nLoRA 权重复制完成！共复制 {copied_count} 个文件")
    
    # 显示保存的大小
    total_size = sum(
        os.path.getsize(os.path.join(lora_dest_path, f))
        for f in os.listdir(lora_dest_path)
        if os.path.isfile(os.path.join(lora_dest_path, f))
    )
    print(f"总大小: {total_size / 1024 / 1024:.1f} MB")
    
    return lora_dest_path


def deploy_model(config):
    """使用 vLLM 部署模型"""
    base_model_path = config["base_model_path"]
    models_save_dir = config["models_save_dir"]
    lora_save_name = config["lora_save_name"]
    
    # LoRA 路径
    lora_path = os.path.join(models_save_dir, lora_save_name)
    
    # 检查 LoRA 是否存在
    if not os.path.exists(lora_path):
        print(f"错误：LoRA 权重不存在: {lora_path}")
        print("请先运行 --save 模式保存 LoRA 权重。")
        sys.exit(1)
    
    # 检查 adapter_model.safetensors 是否存在
    adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_path):
        print(f"错误：adapter_model.safetensors 不存在: {adapter_path}")
        sys.exit(1)
    
    # 构建 vLLM 部署命令
    vllm_cmd = [
        "vllm", "serve", base_model_path,
        "--port", "8000",
        "--dtype", "float16",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.63",
        "--enforce-eager",
        "--enable-lora",
        "--max-lora-rank", "64",
        "--lora-modules", f"{lora_save_name}={lora_path}"
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    print("\n正在启动 vLLM 服务...")
    print(f"命令: CUDA_VISIBLE_DEVICES=0 {' '.join(vllm_cmd)}")
    print("\n" + "="*50)
    print("vLLM 服务已启动，按 Ctrl+C 停止")
    print("="*50 + "\n")
    
    # 启动 vLLM 服务
    try:
        subprocess.run(vllm_cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n服务已停止")
    except subprocess.CalledProcessError as e:
        print(f"vLLM 启动失败: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="LoRA 部署脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python deploy_lora.py --save           # 仅保存 LoRA 权重（不含 checkpoint）
  python deploy_lora.py --deploy         # 仅部署模型
  python deploy_lora.py --save --deploy  # 保存并部署
        """
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="保存 LoRA 权重到指定目录（不含 checkpoint 文件夹）"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="使用 vLLM 部署模型"
    )
    
    args = parser.parse_args()
    
    # 检查是否指定了操作模式
    if not args.save and not args.deploy:
        parser.print_help()
        print("\n错误：请至少指定一个操作模式 (--save 或 --deploy)")
        sys.exit(1)
    
    # 加载配置
    config = load_config()
    
    # 执行操作
    if args.save:
        save_lora(config)
    
    if args.deploy:
        deploy_model(config)


if __name__ == "__main__":
    main()
