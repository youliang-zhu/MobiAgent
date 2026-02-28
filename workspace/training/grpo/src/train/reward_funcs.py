"""Reward functions for GRPO training of MobiMind Decider"""

import json
import re
import logging
from typing import List, Dict, Any, Optional

from .grounder_client import get_grounder_client

logger = logging.getLogger(__name__)


def parse_completion(completion) -> Optional[Dict[str, Any]]:
    """Parse model completion to extract action info
    
    Args:
        completion: Can be:
            - str: Direct completion text
            - dict: {'role': 'assistant', 'content': '...'}
            - list: [{'role': 'assistant', 'content': '...'}]
    """
    try:
        # Extract text content from various formats
        if isinstance(completion, list):
            # Handle list format: [{'role': 'assistant', 'content': '...'}]
            if len(completion) > 0 and isinstance(completion[0], dict):
                completion = completion[0].get("content", "")
            else:
                completion = str(completion)
        elif isinstance(completion, dict):
            # Handle dict format: {'role': 'assistant', 'content': '...'}
            completion = completion.get("content", "")
        elif not isinstance(completion, str):
            completion = str(completion)
        
        # 移除 thinking 标签
        completion = re.sub(r'<think>.*?</think>', '', completion, flags=re.DOTALL)
        if '</think>' in completion:
            completion = completion.split('</think>', 1)[-1]
        completion = completion.strip()
        
        # 提取 JSON
        if "```json" in completion:
            match = re.search(r"```json\n(.*?)\n```", completion, re.DOTALL)
            if match:
                completion = match.group(1)
        
        # 尝试解析 JSON
        parsed = json.loads(completion)
        return parsed
    except json.JSONDecodeError:
        # 尝试手动提取关键字段
        try:
            action_match = re.search(r'"action"\s*:\s*"(\w+)"', completion)
            if action_match:
                action = action_match.group(1)
                result = {"action": action, "parameters": {}}
                
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', completion)
                if reasoning_match:
                    result["reasoning"] = reasoning_match.group(1)
                
                if action == "click":
                    target_match = re.search(r'"target_element"\s*:\s*"([^"]*)"', completion)
                    if target_match:
                        result["parameters"]["target_element"] = target_match.group(1)
                elif action == "input":
                    text_match = re.search(r'"text"\s*:\s*"([^"]*)"', completion)
                    if text_match:
                        result["parameters"]["text"] = text_match.group(1)
                elif action == "swipe":
                    dir_match = re.search(r'"direction"\s*:\s*"(\w+)"', completion)
                    if dir_match:
                        result["parameters"]["direction"] = dir_match.group(1)
                
                return result
        except Exception:
            pass
    except Exception:
        pass
    
    return None


def is_point_in_bounds(x: int, y: int, bounds: List[int]) -> bool:
    """Check if point (x, y) is within bounds [x1, y1, x2, y2]"""
    if len(bounds) != 4:
        return False
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def text_contains_match(pred_text: str, gt_text: str) -> bool:
    """Check if texts match (contains match)"""
    if not pred_text or not gt_text:
        return False
    pred_text = pred_text.strip().lower()
    gt_text = gt_text.strip().lower()
    return gt_text in pred_text or pred_text in gt_text


def decider_reward(
    prompts: List[str],
    completions: List[str],
    gt_action: List[Dict[str, Any]] = None,
    images: List = None,
    **kwargs
) -> List[float]:
    """
    Reward function for MobiMind Decider GRPO training
    
    Args:
        prompts: List of prompts (not used, but required by TRL)
        completions: List of model completions (JSON strings)
        gt_action: List of ground truth actions
        images: List of images (for Grounder calls)
        **kwargs: Additional fields
    
    Returns:
        List of rewards [0.0, 1.0]
    """
    # Debug logging (first batch only)
    if not hasattr(decider_reward, '_logged_once'):
        logger.info(f"=== Reward Function Debug ===")
        logger.info(f"Batch size: {len(completions)}")
        logger.info(f"Sample completion (first 200 chars): {completions[0][:200] if completions else 'None'}")
        logger.info(f"GT action sample: {gt_action[0] if gt_action else 'None'}")
        logger.info(f"Images provided: {images is not None and len(images) > 0}")
        decider_reward._logged_once = True
    
    if gt_action is None:
        gt_action = [{}] * len(completions)
    else:
        # Unwrap gt_action if it's wrapped in lists (from dataset format)
        # Dataset returns gt_action=[{...}] per sample, so batch is [[{...}], [{...}], ...]
        if gt_action and isinstance(gt_action[0], list):
            gt_action = [item[0] if item else {} for item in gt_action]
    
    if images is None:
        images = [None] * len(completions)
    
    rewards = []
    grounder_requests = []  # Batch grounder requests
    grounder_indices = []   # Track which rewards need grounder results
    
    # Debug logging (first 3 samples of first batch)
    if not hasattr(decider_reward, '_sample_logged'):
        logger.info("="*80)
        logger.info("REWARD FUNCTION DEBUG - First Batch Samples")
        logger.info("="*80)
        for idx in range(min(3, len(completions))):
            logger.info(f"\n--- Sample {idx} ---")
            logger.info(f"Completion (first 300 chars):\n{completions[idx][:300]}")
            logger.info(f"GT Action: {gt_action[idx]}")
            logger.info(f"Has image: {images[idx] is not None}")
        decider_reward._sample_logged = True
    
    # First pass: compute rewards for non-click actions, collect click requests
    for i, (completion, gt, img) in enumerate(zip(completions, gt_action, images)):
        parsed = parse_completion(completion)
        
        # Debug: log first few parsing results
        if not hasattr(decider_reward, '_parse_logged') and i < 3:
            logger.info(f"\n[Parse Debug {i}] Completion length: {len(completion)}")
            logger.info(f"[Parse Debug {i}] Parsed result: {parsed}")
            if i == 2:
                decider_reward._parse_logged = True
        
        # 解析失败
        if parsed is None:
            rewards.append(0.0)
            continue
        
        pred_action = parsed.get("action", "")
        gt_type = gt.get("type", "")
        
        # action type 不匹配
        if pred_action != gt_type:
            rewards.append(0.0)
            continue
        
        # action type 匹配，根据类型计算 reward
        if pred_action == "click":
            # 需要调用 Grounder，先占位
            rewards.append(None)  # Placeholder
            
            reasoning = parsed.get("reasoning", "")
            target_element = parsed.get("parameters", {}).get("target_element", "")
            
            # Debug: log click action details
            if not hasattr(decider_reward, '_click_logged') and len(grounder_requests) < 3:
                logger.info(f"\n[Click Debug {i}] Reasoning: {reasoning[:100]}")
                logger.info(f"[Click Debug {i}] Target: {target_element}")
                logger.info(f"[Click Debug {i}] Has image: {img is not None}")
                logger.info(f"[Click Debug {i}] GT bounds: {gt.get('bounds', 'N/A')}")
            
            if target_element and img is not None:
                grounder_requests.append((img, reasoning, target_element))
                grounder_indices.append(i)
            else:
                # 无法调用 Grounder，给部分分
                rewards[i] = 0.3
                
        elif pred_action == "input":
            pred_text = parsed.get("parameters", {}).get("text", "")
            gt_text = gt.get("text", "")
            if text_contains_match(pred_text, gt_text):
                rewards.append(1.0)
            else:
                rewards.append(0.3)  # action type 对但 text 不对
                
        elif pred_action == "swipe":
            pred_dir = parsed.get("parameters", {}).get("direction", "").upper()
            gt_dir = gt.get("direction", "").upper()
            if pred_dir == gt_dir:
                rewards.append(1.0)
            else:
                rewards.append(0.3)  # action type 对但 direction 不对
                
        elif pred_action in ["done", "wait"]:
            # 无额外参数，type 对就是对
            rewards.append(1.0)
            
        else:
            # 未知 action type
            rewards.append(0.0)
    
    # Batch call Grounder for click actions
    if grounder_requests:
        logger.info(f"\n[Grounder] Calling Grounder with {len(grounder_requests)} requests")
        if not hasattr(decider_reward, '_click_logged'):
            decider_reward._click_logged = True
        
        try:
            client = get_grounder_client()
            bbox_results = client.batch_predict_bbox(grounder_requests)
            
            logger.info(f"[Grounder] Received {len(bbox_results)} bbox results")
            
            for idx, bbox in zip(grounder_indices, bbox_results):
                gt = gt_action[idx]
                gt_bounds = gt.get("bounds", [])
                
                if bbox is None:
                    # Grounder 调用失败
                    rewards[idx] = 0.3
                elif not gt_bounds:
                    # 没有 GT bounds
                    rewards[idx] = 0.3
                else:
                    # 计算中心点是否在 bounds 内
                    center_x = (bbox[0] + bbox[2]) // 2
                    center_y = (bbox[1] + bbox[3]) // 2
                    if is_point_in_bounds(center_x, center_y, gt_bounds):
                        rewards[idx] = 1.0
                    else:
                        rewards[idx] = 0.3
                        
        except Exception as e:
            logger.warning(f"Grounder batch call failed: {e}")
            # 所有待处理的 click 给部分分
            for idx in grounder_indices:
                if rewards[idx] is None:
                    rewards[idx] = 0.3
    
    # 确保没有 None 值
    rewards = [r if r is not None else 0.0 for r in rewards]
    
    # Debug: log reward statistics
    if not hasattr(decider_reward, '_reward_stats_logged'):
        reward_mean = sum(rewards) / len(rewards) if rewards else 0
        reward_nonzero = sum(1 for r in rewards if r > 0)
        logger.info(f"\n[Reward Stats] Batch size: {len(rewards)}")
        logger.info(f"[Reward Stats] Mean reward: {reward_mean:.3f}")
        logger.info(f"[Reward Stats] Non-zero rewards: {reward_nonzero}/{len(rewards)}")
        logger.info(f"[Reward Stats] First 10 rewards: {rewards[:10]}")
        decider_reward._reward_stats_logged = True
    
    return rewards
