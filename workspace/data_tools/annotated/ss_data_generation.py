import os
import json
import random
import shutil
import argparse
from tqdm import tqdm
from pathlib import Path
try:
    from google import genai
except Exception:
    genai = None

# 配置
HARD_STEPS_COUNT = 140
RANDOM_STEPS_COUNT = 60
TOTAL_STEPS = 200
LIVESTREAM_SAMPLE_COUNT = 10

def setup_gemini(api_key, model_name):
    """初始化 Gemini API"""
    if genai is None:
        raise ImportError("Gemini support requires google-genai. Please install it first.")
    client = genai.Client(api_key=api_key)
    return client, model_name

def load_step_data(root, step_idx):
    """加载单步数据"""
    actions_path = os.path.join(root, "actions.json")
    react_path = os.path.join(root, "react.json")
    img_path = os.path.join(root, f"{step_idx}.jpg")
    
    if not all(os.path.exists(p) for p in [actions_path, react_path, img_path]):
        return None
    
    with open(actions_path, 'r', encoding='utf-8') as f:
        actions_data = json.load(f)
    with open(react_path, 'r', encoding='utf-8') as f:
        react_data = json.load(f)
    
    if step_idx > len(react_data):
        return None
    
    return {
        'root': root,
        'step_idx': step_idx,
        'img_path': img_path,
        'task_description': actions_data.get('task_description'),
        'react': react_data[step_idx - 1],
        'action': react_data[step_idx - 1]['function']['name']
    }

def collect_candidate_steps(data_path, livestream_path):
    """收集候选步骤"""
    swipe_steps = []
    click_steps = []
    livestream_steps = []
    other_steps = []
    
    # 收集 livestream 前10个任务的第一步
    if os.path.exists(livestream_path):
        livestream_tasks = sorted([d for d in os.listdir(livestream_path) 
                                  if os.path.isdir(os.path.join(livestream_path, d))])[:LIVESTREAM_SAMPLE_COUNT]
        for task_dir in livestream_tasks:
            task_path = os.path.join(livestream_path, task_dir)
            step_data = load_step_data(task_path, 1)
            if step_data:
                livestream_steps.append(step_data)
    
    # 遍历所有任务收集 click/swipe/other
    for root, dirs, files in os.walk(data_path):
        if "actions.json" not in files or "react.json" not in files:
            continue
        
        react_path = os.path.join(root, "react.json")
        with open(react_path, 'r', encoding='utf-8') as f:
            react_data = json.load(f)
        
        for i, react in enumerate(react_data, 1):
            step_data = load_step_data(root, i)
            if not step_data:
                continue
            
            action = react['function']['name']
            if action == 'swipe':
                swipe_steps.append(step_data)
            elif action == 'click':
                click_steps.append(step_data)
            else:
                other_steps.append(step_data)
    
    return swipe_steps, click_steps, livestream_steps, other_steps

def generate_tasks_with_gemini(client, model_name, step_data, log_file):
    """调用 Gemini 生成任务描述和 reasoning"""
    original_task = step_data['task_description']
    if isinstance(original_task, list):
        original_task = original_task[0]
    
    action = step_data['react']['function']['name']
    params = step_data['react']['function']['parameters']
    original_reasoning = step_data['react']['reasoning']
    
    prompt = f"""你是一个任务描述生成器。给定一个手机操作步骤，生成5-10个不同的独立任务描述，并改写原始推理使该推理单独的成为当前步骤执行的通用理由。

原始完整任务：{original_task}
操作历史记录：{action}
操作参数：{json.dumps(params, ensure_ascii=False)}
原始推理：{original_reasoning}

要求：
1. 生成5-10个独立的任务描述变体，要多样化（命令式、目标式、功能式等）
2. 改写推理，使其脱离原任务上下文，更加通用
3. 任务描述从简短到详细都要有
4. 必须严格按照以下JSON格式输出，不要有任何额外文字：

{{"tasks": ["任务1", "任务2", ...], "reasoning": "适用于单独这一步的通用化的推理"}}
"""
    
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        result_text = response.text.strip()
        
        # 清理可能的 markdown 代码块标记
        if result_text.startswith('```'):
            result_text = result_text.split('```')[1]
            if result_text.startswith('json'):
                result_text = result_text[4:]
        result_text = result_text.strip()
        
        result = json.loads(result_text)
        
        # 记录日志
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Step: {step_data['root']}/{step_data['step_idx']}\n")
            f.write(f"Success: {json.dumps(result, ensure_ascii=False)}\n\n")
        
        return result
    
    except Exception as e:
        # 记录错误并返回默认值
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"Step: {step_data['root']}/{step_data['step_idx']}\n")
            f.write(f"Error: {str(e)}\n\n")
        
        # 返回默认值
        return {
            'tasks': [original_task] * 5,
            'reasoning': original_reasoning
        }

def generate_ss_data(data_path, livestream_path, output_path, gemini_api_key, model_name):
    """生成单步数据"""
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(os.path.dirname(output_path), 'api_calls.log')
    
    # 初始化 Gemini
    client, model_name = setup_gemini(gemini_api_key, model_name)
    
    print("收集候选步骤...")
    swipe_steps, click_steps, livestream_steps, other_steps = collect_candidate_steps(data_path, livestream_path)
    
    print(f"收集到: swipe={len(swipe_steps)}, click={len(click_steps)}, livestream={len(livestream_steps)}, other={len(other_steps)}")
    
    # 按顺序构建: 1. livestream第一步 2. swipe全部 3. click 4. 随机
    selected_steps = []
    
    # 1. 先加入所有 livestream 第一步
    selected_steps.extend(livestream_steps)
    remaining_hard = HARD_STEPS_COUNT - len(selected_steps)
    
    # 2. 加入所有 swipe（优先保证全部包括）
    if len(swipe_steps) <= remaining_hard:
        selected_steps.extend(swipe_steps)
        remaining_hard -= len(swipe_steps)
    else:
        selected_steps.extend(random.sample(swipe_steps, remaining_hard))
        remaining_hard = 0
    
    # 3. 用 click 补充到 140
    if remaining_hard > 0:
        if len(click_steps) <= remaining_hard:
            selected_steps.extend(click_steps)
            remaining_hard -= len(click_steps)
        else:
            selected_steps.extend(random.sample(click_steps, remaining_hard))
            remaining_hard = 0
    
    # 4. 如果困难步骤不足 140，从 other 补充
    if remaining_hard > 0:
        补充数量 = min(remaining_hard, len(other_steps))
        selected_steps.extend(random.sample(other_steps, 补充数量))
    
    # 5. 添加随机步骤到 200
    random_steps_needed = TOTAL_STEPS - len(selected_steps)
    # 排除已选择的步骤
    available_random = [s for s in other_steps if s not in selected_steps]
    if len(available_random) >= random_steps_needed:
        selected_steps.extend(random.sample(available_random, random_steps_needed))
    else:
        selected_steps.extend(available_random)
    
    final_steps = selected_steps
    print(f"最终选择: livestream={len(livestream_steps)}, swipe={min(len(swipe_steps), HARD_STEPS_COUNT-len(livestream_steps))}, "
          f"总困难步骤={min(len(selected_steps), HARD_STEPS_COUNT)}, 随机步骤={len(selected_steps)-min(len(selected_steps), HARD_STEPS_COUNT)}, "
          f"总计={len(final_steps)}")
    
    # 生成单步数据
    for idx, step_data in enumerate(tqdm(final_steps, desc="生成单步数据"), 1):
        step_dir = os.path.join(output_path, f"step_{idx:03d}")
        os.makedirs(step_dir, exist_ok=True)
        
        # 复制截图
        shutil.copy(step_data['img_path'], os.path.join(step_dir, '1.jpg'))
        
        # 调用 Gemini 生成
        generated = generate_tasks_with_gemini(client, model_name, step_data, log_file)
        
        # 保存 tasks.json
        with open(os.path.join(step_dir, 'tasks.json'), 'w', encoding='utf-8') as f:
            json.dump(generated['tasks'], f, ensure_ascii=False, indent=2)
        
        # 保存 react.json（新 reasoning + 原 action/parameters）
        new_react = {
            'reasoning': generated['reasoning'],
            'function': step_data['react']['function']
        }
        with open(os.path.join(step_dir, 'react.json'), 'w', encoding='utf-8') as f:
            json.dump(new_react, f, ensure_ascii=False, indent=2)
    
    print(f"完成！生成了 {len(final_steps)} 条单步数据")
    print(f"输出目录: {output_path}")
    print(f"API 日志: {log_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成单步训练数据")
    parser.add_argument("--data_path", type=str, required=True,
                       help="主数据路径")
    parser.add_argument("--livestream_path", type=str, required=True,
                       help="livestream 数据路径")
    parser.add_argument("--output_path", type=str, required=True,
                       help="输出路径")
    parser.add_argument("--gemini_api_key", type=str, required=True,
                       help="Gemini API Key")
    parser.add_argument("--model_name", type=str, required=True,
                       help="Gemini 模型名称 (如 gemini-2.0-flash-exp)")
    
    args = parser.parse_args()
    
    generate_ss_data(
        data_path=args.data_path,
        livestream_path=args.livestream_path,
        output_path=args.output_path,
        gemini_api_key=args.gemini_api_key,
        model_name=args.model_name
    )
