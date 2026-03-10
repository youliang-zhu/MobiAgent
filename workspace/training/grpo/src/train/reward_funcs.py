"""Reward functions for Decider GRPO."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .grounder_client import call_grounder


_FIRST_BATCH_LOGGED = False
_REWARD_BATCH_ID = 0
_LAST_REWARD_STATS: Dict[str, float] = {
    "reward_mean": 0.0,
    "json_valid_rate": 0.0,
    "action_type_acc": 0.0,
    "click_hit_rate": 0.0,
    "click_iou_mean": 0.0,
    "num_samples": 0.0,
}


def get_last_reward_stats() -> Dict[str, float]:
    return dict(_LAST_REWARD_STATS)


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return repr(obj)


def _image_brief(image: Any) -> str:
    if isinstance(image, (list, tuple)):
        if not image:
            return "[]"
        return f"list[{len(image)}]->{_image_brief(image[0])}"
    if image is None:
        return "None"
    filename = getattr(image, "filename", None)
    if isinstance(filename, str) and filename:
        return filename
    return f"<{type(image).__name__}>"


def _unwrap_image(image: Any) -> Any:
    if isinstance(image, (list, tuple)):
        if not image:
            return None
        return _unwrap_image(image[0])
    return image


def _write_reward_batch_log(
    batch_id: int,
    trainer_state: Any,
    gt_action: List[Dict[str, Any]],
    images: List[Any],
    item_logs: List[Dict[str, Any]],
):
    if os.getenv("ENABLE_REWARD_LOGGER", "1") != "1":
        return

    log_root = Path(os.getenv("REWARD_LOG_DIR", "./reward_logs"))
    log_root.mkdir(parents=True, exist_ok=True)

    global_step = getattr(trainer_state, "global_step", -1) if trainer_state is not None else -1
    num_generations = int(os.getenv("GRPO_NUM_GENERATIONS", "1") or "1")
    num_generations = max(1, num_generations)
    sample_count = (len(item_logs) + num_generations - 1) // num_generations

    out_file = log_root / f"batch_{batch_id:07d}_step_{int(global_step):07d}.log"
    with out_file.open("w", encoding="utf-8") as f:
        f.write(
            f"[reward-batch] id={batch_id} step={global_step} "
            f"items={len(item_logs)} num_generations={num_generations} inferred_samples={sample_count}\n\n"
        )

        for sample_idx in range(sample_count):
            s = sample_idx * num_generations
            e = min((sample_idx + 1) * num_generations, len(item_logs))
            sample_items = item_logs[s:e]
            if not sample_items:
                continue

            gt_idx = sample_items[0]["gt_index"]
            dataset_idx = sample_items[0].get("dataset_idx")
            sample_task = sample_items[0].get("task")
            sample_image_path = sample_items[0].get("image_path")
            gt = gt_action[gt_idx] if gt_action and 0 <= gt_idx < len(gt_action) else {}
            img = images[gt_idx] if images and 0 <= gt_idx < len(images) else None

            f.write("=" * 80 + "\n")
            f.write(
                f"[sample {sample_idx:04d}] gt_index={gt_idx} "
                f"dataset_idx={dataset_idx} image={_image_brief(img)}\n"
            )
            f.write(f"task: {sample_task or ''}\n")
            f.write(f"image_path: {sample_image_path or ''}\n")
            f.write(f"gt_action:\n{_safe_json(gt)}\n")

            for row in sample_items:
                completion_text = row.get("completion_text", "")
                f.write("-" * 80 + "\n")
                f.write(
                f"[rollout {row['rollout_index']:02d}] item_index={row['item_index']} "
                f"reward={row['reward']:.4f} parsed_type={row['parsed_type']}\n"
            )
                if row.get("gt_type") == "click":
                    f.write(
                        "click_detail: "
                        f"target={row.get('target_element')} "
                        f"gt_bounds={row.get('gt_bounds')} "
                        f"pred_bbox={row.get('pred_bbox')} "
                        f"iou={row.get('click_iou')}\n"
                    )
                f.write(f"parsed_action:\n{_safe_json(row.get('parsed'))}\n")
                f.write(f"completion:\n{completion_text}\n")
            f.write("\n")


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion

    if isinstance(completion, list):
        # TRL conversational format may return a list of messages, e.g.
        # [{"role": "assistant", "content": "..."}]
        parts: List[str] = []
        for item in completion:
            txt = _completion_to_text(item)
            if isinstance(txt, str) and txt.strip():
                parts.append(txt)
        if parts:
            return "\n".join(parts)

    if isinstance(completion, dict):
        # Common chat-format message object
        role = completion.get("role")
        if role is not None and "content" in completion:
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

    # Case C: {"action": "input", "parameters": {...}, "reasoning": "..."}
    if isinstance(data.get("action"), str):
        params = data.get("parameters") if isinstance(data.get("parameters"), dict) else {}
        return {
            "type": data.get("action"),
            "direction": params.get("direction") or data.get("direction"),
            "text": params.get("text") or data.get("text"),
            "target_element": params.get("target_element") or data.get("target_element"),
            "reasoning": reasoning,
        }

    # Case D: flat JSON
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


def _normalize_bbox(bbox: Any) -> Optional[List[float]]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    lx, rx = sorted([x1, x2])
    ty, by = sorted([y1, y2])
    return [lx, ty, rx, by]


def _bbox_iou(pred_bbox: Any, gt_bbox: Any) -> float:
    p = _normalize_bbox(pred_bbox)
    g = _normalize_bbox(gt_bbox)
    if p is None or g is None:
        return 0.0

    px1, py1, px2, py2 = p
    gx1, gy1, gx2, gy2 = g
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    pa = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)
    union = pa + ga - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _single_reward(
    pred: Optional[Dict[str, Any]],
    gt_action: Dict[str, Any],
    image: Any,
    grounder_fn: Optional[Callable[..., Optional[List[int]]]],
) -> tuple[float, Dict[str, Any]]:
    detail: Dict[str, Any] = {}
    if pred is None:
        return 0.0, detail

    gt_type = _norm_lower(gt_action.get("type"))
    pred_type = _norm_lower(pred.get("type"))
    detail["gt_type"] = gt_type
    detail["pred_type"] = pred_type
    if not gt_type or pred_type != gt_type:
        return 0.0, detail

    if gt_type in {"done", "wait"}:
        return 1.0, detail

    if gt_type == "swipe":
        gt_dir = _norm_lower(gt_action.get("direction"))
        pred_dir = _norm_lower(pred.get("direction"))
        return (1.0 if gt_dir and (gt_dir == pred_dir) else 0.0), detail

    if gt_type == "input":
        gt_text = _norm(gt_action.get("text"))
        pred_text = _norm(pred.get("text"))
        return (1.0 if gt_text == pred_text else 0.0), detail

    if gt_type == "click":
        gt_bounds = gt_action.get("bounds")
        target_element = _norm(pred.get("target_element"))
        detail["gt_bounds"] = gt_bounds
        detail["target_element"] = target_element
        if not target_element:
            return 0.0, detail
        if not isinstance(gt_bounds, list) or len(gt_bounds) != 4:
            return 0.0, detail

        reasoning = _norm(pred.get("reasoning"))
        fn = grounder_fn or call_grounder
        image_for_grounder = _unwrap_image(image)
        pred_bbox = fn(
            image=image_for_grounder,
            reasoning=reasoning,
            target_element=target_element,
            url=os.getenv("GROUNDER_URL", "http://localhost:8001/v1"),
            timeout=float(os.getenv("GROUNDER_TIMEOUT", "30")),
        )
        detail["pred_bbox"] = pred_bbox
        if pred_bbox is None:
            detail["click_iou"] = 0.0
            return 0.0, detail

        # Continuous click reward: IoU in [0, 1].
        iou = _bbox_iou(pred_bbox, gt_bounds)
        detail["click_iou"] = iou
        return max(0.0, min(1.0, float(iou))), detail

    return 0.0, detail


def decider_reward(
    prompts: List[Any],
    completions: List[Any],
    gt_action: List[Dict[str, Any]],
    images: List[Any],
    **kwargs: Any,
) -> List[float]:
    """
    Mixed reward:
    - done/wait/swipe/input: binary 0/1
    - click: continuous IoU reward in [0, 1]
    """
    del prompts  # unused by design

    grounder_fn = kwargs.get("grounder_fn")
    log_details = kwargs.get("log_details", os.getenv("LOG_REWARD_DETAILS", "1") == "1")

    rewards: List[float] = []
    item_logs: List[Dict[str, Any]] = []
    type_counter: Counter[str] = Counter()
    json_valid = 0
    action_type_hit = 0
    click_total = 0
    click_hit = 0
    click_iou_sum = 0.0
    click_iou_count = 0

    n_gt = len(gt_action) if gt_action else 0
    n_img = len(images) if images else 0
    dataset_indices = kwargs.get("dataset_idx")
    image_paths = kwargs.get("image_path")
    tasks = kwargs.get("task")

    if not isinstance(dataset_indices, list):
        dataset_indices = []
    if not isinstance(image_paths, list):
        image_paths = []
    if not isinstance(tasks, list):
        tasks = []

    n_dataset_idx = len(dataset_indices)
    n_image_paths = len(image_paths)
    n_tasks = len(tasks)

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

        reward, detail = _single_reward(
            pred=parsed,
            gt_action=gt,
            image=image,
            grounder_fn=grounder_fn,
        )
        rewards.append(reward)
        type_counter[gt_type or "unknown"] += 1

        num_generations = int(os.getenv("GRPO_NUM_GENERATIONS", "1") or "1")
        num_generations = max(1, num_generations)
        item_logs.append(
            {
                "item_index": i,
                "sample_index": i // num_generations,
                "rollout_index": i % num_generations,
                "gt_index": i % n_gt if n_gt else -1,
                "reward": reward,
                "gt_type": gt_type,
                "parsed_type": parsed_type,
                "parsed": parsed,
                "target_element": detail.get("target_element"),
                "gt_bounds": detail.get("gt_bounds"),
                "pred_bbox": detail.get("pred_bbox"),
                "click_iou": detail.get("click_iou"),
                "completion_text": _completion_to_text(completion),
                "dataset_idx": dataset_indices[i % n_dataset_idx] if n_dataset_idx else None,
                "image_path": image_paths[i % n_image_paths] if n_image_paths else None,
                "task": tasks[i % n_tasks] if n_tasks else None,
            }
        )

        if gt_type == "click" and parsed_type == "click":
            click_total += 1
            click_iou = float(detail.get("click_iou") or 0.0)
            click_iou_sum += click_iou
            click_iou_count += 1
            if click_iou > 0.0:
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
    _LAST_REWARD_STATS["click_iou_mean"] = float(click_iou_sum / click_iou_count) if click_iou_count else 0.0
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
        f" click_iou_mean={_LAST_REWARD_STATS['click_iou_mean']:.4f}"
    )

    global _REWARD_BATCH_ID
    _REWARD_BATCH_ID += 1
    _write_reward_batch_log(
        batch_id=_REWARD_BATCH_ID,
        trainer_state=kwargs.get("trainer_state"),
        gt_action=gt_action,
        images=images,
        item_logs=item_logs,
    )

    return rewards
