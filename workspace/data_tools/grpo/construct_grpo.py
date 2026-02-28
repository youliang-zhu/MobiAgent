import os
import json
import random
import argparse
import sys
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from PIL import Image
from tqdm import tqdm

# 允许直接脚本运行时导入项目模块
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.load_md_prompt import load_prompt


@dataclass
class GRPOEntry:
    instruction: str
    images: List[str]
    gt_action: Dict[str, Any]


# 加载 prompt 模板
decider_prompt = load_prompt("decider.md")


def history_str(history):
    """将历史记录转换为字符串格式"""
    if len(history) == 0:
        return "(No history)"
    else:
        return "\n".join(f"{idx}. {h}" for idx, h in enumerate(history, 1))


def resize_and_copy_image(img_path, data_path, out_path, factor, do_copy=True):
    """缩放并复制图片到输出目录"""
    relative_path = os.path.relpath(img_path, data_path)
    safe_filename = relative_path.replace(os.sep, "_").replace(":", "_")
    safe_filename = f"grpo_{safe_filename}"
    out_relpath = os.path.join(out_path, safe_filename)

    if do_copy:
        pil_img = Image.open(img_path)
        
        # 如果是 RGBA 模式，转换为 RGB
        if pil_img.mode == 'RGBA':
            pil_img = pil_img.convert('RGB')
        
        width, height = pil_img.size
        new_width = int(width * factor)
        new_height = int(height * factor)
        resized_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
        resized_img.save(out_relpath)

    out_abspath = os.path.abspath(out_relpath)
    return out_abspath


def create_gt_action(action: Dict, react: Dict, factor: float) -> Dict[str, Any]:
    """
    根据 action 类型创建 ground truth action
    
    Args:
        action: actions.json 中的单个 action
        react: react.json 中对应的 react 数据
        factor: 缩放因子
    
    Returns:
        gt_action 字典
    """
    action_type = react["function"]["name"]
    params = react["function"]["parameters"]
    
    gt_action = {"type": action_type}
    
    if action_type == "click":
        # click 需要 position 和 bounds
        gt_action["position_x"] = int(action["position_x"] * factor)
        gt_action["position_y"] = int(action["position_y"] * factor)
        if "bounds" in action and action["bounds"]:
            gt_action["bounds"] = [int(x * factor) for x in action["bounds"]]
        gt_action["target_element"] = params.get("target_element", "")
        
    elif action_type == "swipe":
        # swipe 需要方向
        gt_action["direction"] = params.get("direction", "")
        
    elif action_type == "input":
        # input 需要输入文本
        gt_action["text"] = params.get("text", "")
        
    elif action_type == "done":
        # done 无额外参数
        pass
        
    elif action_type == "wait":
        # wait 无额外参数
        pass
    
    return gt_action


def collect_all_steps(data_path: str) -> List[Dict]:
    """
    遍历所有轨迹数据，收集所有单步数据
    
    Returns:
        List of step dictionaries, each containing:
        - root: 轨迹目录路径
        - step_idx: 步骤索引 (1-based)
        - task: 任务描述
        - action: actions.json 中的 action
        - react: react.json 中的 react
        - is_input: 是否是 input 类型
        - is_pre_input: 是否是 input 前一步
    """
    all_steps = []
    
    for root, dirs, files in tqdm(os.walk(data_path), desc="Scanning trajectories"):
        if "actions.json" not in files or "react.json" not in files or "parse.error" in files:
            continue
        
        # 读取 actions.json
        actions_json = os.path.join(root, "actions.json")
        with open(actions_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        task_description = data.get("task_description")
        actions = data.get("actions")
        
        # 读取 react.json
        react_json = os.path.join(root, "react.json")
        with open(react_json, "r", encoding="UTF-8") as f:
            react_data = json.load(f)
        
        # 检查图片数量与 react 数量是否匹配
        index = 1
        while f"{index}.jpg" in files:
            index += 1
        num_img = index - 1
        
        # 补充 done 动作（如果需要）
        if num_img == len(react_data) + 1:
            done_reasoning = "我已经完成了目标任务，任务已结束。"
            react_data.append({
                "reasoning": done_reasoning,
                "function": {
                    "name": "done",
                    "parameters": {}
                }
            })
        elif num_img != len(react_data):
            print(f"Warning: Image count ({num_img}) != React count ({len(react_data)}) in {root}. Skipping.")
            continue
        
        # 处理 task_description
        if isinstance(task_description, list):
            task = task_description[0]  # 使用原始描述
        else:
            task = task_description
        
        # 标记 input 步骤和 input 前一步
        input_indices = set()
        for i, react in enumerate(react_data):
            if react["function"]["name"] == "input":
                input_indices.add(i)
        
        pre_input_indices = {i - 1 for i in input_indices if i > 0}
        
        # 收集每一步
        for i, react in enumerate(react_data):
            # 确保有对应的 action（done 动作可能没有对应的 action）
            action = actions[i] if i < len(actions) else {}
            
            step = {
                "root": root,
                "step_idx": i + 1,  # 1-based
                "task": task,
                "action": action,
                "react": react,
                "react_data": react_data,  # 用于构建 history
                "is_input": i in input_indices,
                "is_pre_input": i in pre_input_indices,
            }
            all_steps.append(step)
    
    return all_steps


def create_grpo_entry(step: Dict, data_path: str, out_path: str, factor: float, 
                      copied_images: set, history_cache: Dict) -> GRPOEntry:
    """
    为单个步骤创建 GRPO 数据条目
    """
    root = step["root"]
    step_idx = step["step_idx"]
    task = step["task"]
    action = step["action"]
    react = step["react"]
    react_data = step["react_data"]
    
    # 构建 history（当前步骤之前的所有动作）
    cache_key = (root, step_idx)
    if cache_key not in history_cache:
        history = []
        for j in range(step_idx - 1):
            prev_react = react_data[j]
            prev_reasoning = prev_react["reasoning"]
            prev_action_type = prev_react["function"]["name"]
            prev_param = prev_react["function"]["parameters"]
            output_dict = dict(reasoning=prev_reasoning, action=prev_action_type, parameters=prev_param)
            history.append(json.dumps(output_dict, ensure_ascii=False))
        history_cache[cache_key] = history
    
    history = history_cache[cache_key]
    
    # 构建 instruction
    instruction = decider_prompt.format(task=task, history=history_str(history))
    
    # 处理图片
    img_path = os.path.join(root, f"{step_idx}.jpg")
    do_copy = img_path not in copied_images
    out_abspath = resize_and_copy_image(img_path, data_path, out_path, factor, do_copy=do_copy)
    if do_copy:
        copied_images.add(img_path)
    
    # 创建 gt_action
    gt_action = create_gt_action(action, react, factor)
    
    return GRPOEntry(
        instruction=instruction,
        images=[out_abspath],
        gt_action=gt_action
    )


def sample_steps(all_steps: List[Dict], total_samples: int, 
                 input_ratio: float = 0.2, pre_input_ratio: float = 0.1, swipe_ratio: float = 0.05) -> List[Dict]:
    """按比例采样: input 20%, pre_input 10%, swipe 5%, 其余随机"""
    # 分类步骤
    input_steps = [s for s in all_steps if s["is_input"]]
    pre_input_steps = [s for s in all_steps if s["is_pre_input"]]
    swipe_steps = [s for s in all_steps if s["react"]["function"]["name"] == "swipe" and not s["is_input"] and not s["is_pre_input"]]
    other_steps = [s for s in all_steps if not s["is_input"] and not s["is_pre_input"] and s["react"]["function"]["name"] != "swipe"]
    
    print(f"\nStep statistics: Total={len(all_steps)}, Input={len(input_steps)}, PreInput={len(pre_input_steps)}, Swipe={len(swipe_steps)}, Other={len(other_steps)}")
    
    # 计算目标采样数
    target_input = int(total_samples * input_ratio)
    target_pre_input = int(total_samples * pre_input_ratio)
    target_swipe = int(total_samples * swipe_ratio)
    
    # 实际采样（有多少用多少）
    sampled_input = random.sample(input_steps, min(len(input_steps), target_input))
    sampled_pre_input = random.sample(pre_input_steps, min(len(pre_input_steps), target_pre_input))
    sampled_swipe = random.sample(swipe_steps, min(len(swipe_steps), target_swipe))
    
    # 剩余从 other_steps 随机采样
    remaining = total_samples - len(sampled_input) - len(sampled_pre_input) - len(sampled_swipe)
    sampled_other = random.sample(other_steps, min(len(other_steps), remaining))
    
    sampled_steps = sampled_input + sampled_pre_input + sampled_swipe + sampled_other
    random.shuffle(sampled_steps)
    
    print(f"Sampling: Input={len(sampled_input)}/{target_input}, PreInput={len(sampled_pre_input)}/{target_pre_input}, Swipe={len(sampled_swipe)}/{target_swipe}, Other={len(sampled_other)}, Total={len(sampled_steps)}")
    
    return sampled_steps


def construct_grpo_dataset(data_path: str, out_path: str, factor: float = 0.5,
                           train_ratio: float = 0.9, total_samples: int = 1500):
    """
    构建 GRPO 数据集
    """
    os.makedirs(out_path, exist_ok=True)
    
    # 收集所有步骤
    print("Collecting all steps from trajectories...")
    all_steps = collect_all_steps(data_path)
    
    if len(all_steps) == 0:
        print("Error: No valid steps found!")
        return
    
    # 采样
    print(f"\nSampling {total_samples} steps...")
    sampled_steps = sample_steps(all_steps, total_samples)
    
    # 划分训练集和验证集
    random.shuffle(sampled_steps)
    split_idx = int(len(sampled_steps) * train_ratio)
    train_steps = sampled_steps[:split_idx]
    val_steps = sampled_steps[split_idx:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_steps)}")
    print(f"  Val: {len(val_steps)}")
    
    # 创建数据条目
    copied_images = set()
    history_cache = {}
    
    train_entries = []
    print("\nCreating training entries...")
    for step in tqdm(train_steps, desc="Processing train"):
        entry = create_grpo_entry(step, data_path, out_path, factor, copied_images, history_cache)
        train_entries.append(asdict(entry))
    
    val_entries = []
    print("\nCreating validation entries...")
    for step in tqdm(val_steps, desc="Processing val"):
        entry = create_grpo_entry(step, data_path, out_path, factor, copied_images, history_cache)
        val_entries.append(asdict(entry))
    
    # 保存数据集
    train_path = os.path.join(out_path, "mobimind_decider_grpo_train.json")
    val_path = os.path.join(out_path, "mobimind_decider_grpo_val.json")
    
    with open(train_path, "w", encoding="UTF-8") as f:
        json.dump(train_entries, f, ensure_ascii=False, indent=2)
    
    with open(val_path, "w", encoding="UTF-8") as f:
        json.dump(val_entries, f, ensure_ascii=False, indent=2)
    
    # 保存元数据
    metadata = {
        "total_steps_available": len(all_steps),
        "total_sampled": len(sampled_steps),
        "train_count": len(train_entries),
        "val_count": len(val_entries),
        "factor": factor,
        "train_ratio": train_ratio,
    }
    
    metadata_path = os.path.join(out_path, "metadata.json")
    with open(metadata_path, "w", encoding="UTF-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\nDataset saved to:")
    print(f"  Train: {train_path}")
    print(f"  Val: {val_path}")
    print(f"  Metadata: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRPO dataset construction")
    parser.add_argument("--data_path", type=str, 
                        default=os.path.expanduser("~/mobiAgent/MobiAgent/collect/manual/data/淘宝"),
                        help="Root path of trajectory data")
    parser.add_argument("--out_path", type=str,
                        default="/home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data",
                        help="Output path for GRPO dataset")
    parser.add_argument("--factor", type=float, default=0.5,
                        help="Resize factor for images (default: 0.5)")
    parser.add_argument("--train_ratio", type=float, default=0.9,
                        help="Ratio of training data (default: 0.9)")
    parser.add_argument("--total_samples", type=int, default=1500,
                        help="Total number of samples to collect (default: 1500)")
    
    args = parser.parse_args()
    
    construct_grpo_dataset(
        data_path=args.data_path,
        out_path=args.out_path,
        factor=args.factor,
        train_ratio=args.train_ratio,
        total_samples=args.total_samples,
    )
