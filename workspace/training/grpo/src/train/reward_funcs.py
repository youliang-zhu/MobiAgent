"""Reward functions for Decider GRPO (strict binary reward)."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional

from .grounder_client import call_grounder


_FIRST_BATCH_LOGGED = False
_LAST_REWARD_STATS: Dict[str, float] = {
    "reward_mean": 0.0,
    "json_valid_rate": 0.0,
    "action_type_acc": 0.0,
    "click_hit_rate": 0.0,
    "num_samples": 0.0,
}


def get_last_reward_stats() -> Dict[str, float]:
    return dict(_LAST_REWARD_STATS)


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, dict):
        content = completion.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            if parts:
                return "\n".join(parts)

    return str(completion)


def _clean_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def _extract_json_text(text: str) -> str:
    cleaned = _clean_text(text)
    md_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if md_block:
        return md_block.group(1).strip()

    start = cleaned.find("{")
    if start < 0:
        return cleaned

    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1].strip()

    return cleaned[start:].strip()


def parse_decider_action(completion: Any) -> Optional[Dict[str, Any]]:
    """Parse completion text to normalized action dict."""
    text = _completion_to_text(completion)
    json_text = _extract_json_text(text)
    try:
        data = json.loads(json_text)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    reasoning = data.get("reasoning", "")

    # Case A: {"function": {"name": "...", "parameters": {...}}, "reasoning": "..."}
    if isinstance(data.get("function"), dict):
        fn = data["function"]
        params = fn.get("parameters") or {}
        if not isinstance(params, dict):
            params = {}
        return {
            "type": fn.get("name"),
            "direction": params.get("direction"),
            "text": params.get("text"),
            "target_element": params.get("target_element"),
            "reasoning": reasoning,
        }

    # Case B: {"action": {"type": "...", ...}, "reasoning": "..."}
    if isinstance(data.get("action"), dict):
        action = dict(data["action"])
        if isinstance(action.get("parameters"), dict):
            for k, v in action["parameters"].items():
                action.setdefault(k, v)
        return {
            "type": action.get("type") or action.get("name"),
            "direction": action.get("direction"),
            "text": action.get("text"),
            "target_element": action.get("target_element"),
            "reasoning": reasoning or action.get("reasoning", ""),
        }

    # Case C: flat JSON
    flat = dict(data)
    if isinstance(flat.get("parameters"), dict):
        for k, v in flat["parameters"].items():
            flat.setdefault(k, v)
    return {
        "type": flat.get("type") or flat.get("name"),
        "direction": flat.get("direction"),
        "text": flat.get("text"),
        "target_element": flat.get("target_element"),
        "reasoning": reasoning,
    }


def _norm(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _norm_lower(v: Any) -> str:
    return _norm(v).lower()


def _center_in_bounds(pred_bbox: Any, gt_bounds: Any) -> bool:
    if not isinstance(pred_bbox, list) or len(pred_bbox) != 4:
        return False
    if not isinstance(gt_bounds, list) or len(gt_bounds) != 4:
        return False

    try:
        px1, py1, px2, py2 = [float(v) for v in pred_bbox]
        gx1, gy1, gx2, gy2 = [float(v) for v in gt_bounds]
    except Exception:
        return False

    cx = (px1 + px2) / 2.0
    cy = (py1 + py2) / 2.0
    lx, rx = sorted([gx1, gx2])
    ty, by = sorted([gy1, gy2])
    return lx <= cx <= rx and ty <= cy <= by


def _single_reward(
    completion: Any,
    gt_action: Dict[str, Any],
    image: Any,
    grounder_fn: Optional[Callable[..., Optional[List[int]]]],
) -> float:
    pred = parse_decider_action(completion)
    if pred is None:
        return 0.0

    gt_type = _norm_lower(gt_action.get("type"))
    pred_type = _norm_lower(pred.get("type"))
    if not gt_type or pred_type != gt_type:
        return 0.0

    if gt_type in {"done", "wait"}:
        return 1.0

    if gt_type == "swipe":
        gt_dir = _norm_lower(gt_action.get("direction"))
        pred_dir = _norm_lower(pred.get("direction"))
        return 1.0 if gt_dir and (gt_dir == pred_dir) else 0.0

    if gt_type == "input":
        gt_text = _norm(gt_action.get("text"))
        pred_text = _norm(pred.get("text"))
        return 1.0 if gt_text == pred_text else 0.0

    if gt_type == "click":
        gt_bounds = gt_action.get("bounds")
        target_element = _norm(pred.get("target_element"))
        if not target_element:
            return 0.0
        if not isinstance(gt_bounds, list) or len(gt_bounds) != 4:
            return 0.0

        reasoning = _norm(pred.get("reasoning"))
        fn = grounder_fn or call_grounder
        pred_bbox = fn(
            image=image,
            reasoning=reasoning,
            target_element=target_element,
            url=os.getenv("GROUNDER_URL", "http://localhost:8001/v1"),
            timeout=float(os.getenv("GROUNDER_TIMEOUT", "30")),
        )
        if pred_bbox is None:
            return 0.0
        return 1.0 if _center_in_bounds(pred_bbox, gt_bounds) else 0.0

    return 0.0


def decider_reward(
    prompts: List[Any],
    completions: List[Any],
    gt_action: List[Dict[str, Any]],
    images: List[Any],
    **kwargs: Any,
) -> List[float]:
    """
    Strict binary reward for decider actions.

    Always returns only 0.0 or 1.0.
    """
    del prompts  # unused by design

    grounder_fn = kwargs.get("grounder_fn")
    log_details = kwargs.get("log_details", os.getenv("LOG_REWARD_DETAILS", "1") == "1")

    rewards: List[float] = []
    type_counter: Counter[str] = Counter()
    json_valid = 0
    action_type_hit = 0
    click_total = 0
    click_hit = 0

    n_gt = len(gt_action) if gt_action else 0
    n_img = len(images) if images else 0

    global _FIRST_BATCH_LOGGED

    for i, completion in enumerate(completions):
        gt = gt_action[i % n_gt] if n_gt else {}
        image = images[i % n_img] if n_img else None
        gt_type = _norm_lower(gt.get("type"))
        parsed = parse_decider_action(completion)
        parsed_type = _norm_lower(parsed.get("type")) if parsed else ""

        if parsed is not None:
            json_valid += 1
        if gt_type and parsed_type == gt_type:
            action_type_hit += 1

        reward = _single_reward(
            completion=completion,
            gt_action=gt,
            image=image,
            grounder_fn=grounder_fn,
        )
        reward = 1.0 if reward >= 1.0 else 0.0
        rewards.append(reward)
        type_counter[gt_type or "unknown"] += 1

        if gt_type == "click" and parsed_type == "click":
            click_total += 1
            if reward > 0.0:
                click_hit += 1

        if log_details and not _FIRST_BATCH_LOGGED:
            text = _completion_to_text(completion)
            print(
                "[reward-debug]"
                f" idx={i}"
                f" gt={gt}"
                f" parsed={parsed}"
                f" reward={reward}"
                f" completion={text[:300]}"
            )

    if log_details and not _FIRST_BATCH_LOGGED:
        _FIRST_BATCH_LOGGED = True

    if rewards:
        mean_reward = sum(rewards) / len(rewards)
        nonzero = sum(1 for r in rewards if r > 0.0)
    else:
        mean_reward = 0.0
        nonzero = 0

    sample_count = len(rewards)
    _LAST_REWARD_STATS["reward_mean"] = float(mean_reward)
    _LAST_REWARD_STATS["json_valid_rate"] = float(json_valid / sample_count) if sample_count else 0.0
    _LAST_REWARD_STATS["action_type_acc"] = float(action_type_hit / sample_count) if sample_count else 0.0
    _LAST_REWARD_STATS["click_hit_rate"] = float(click_hit / click_total) if click_total else 0.0
    _LAST_REWARD_STATS["num_samples"] = float(sample_count)

    print(
        "[reward-summary]"
        f" batch={len(rewards)}"
        f" mean={mean_reward:.4f}"
        f" nonzero={nonzero}"
        f" action_types={dict(type_counter)}"
        f" json_valid_rate={_LAST_REWARD_STATS['json_valid_rate']:.4f}"
        f" action_type_acc={_LAST_REWARD_STATS['action_type_acc']:.4f}"
        f" click_hit_rate={_LAST_REWARD_STATS['click_hit_rate']:.4f}"
    )

    return rewards
