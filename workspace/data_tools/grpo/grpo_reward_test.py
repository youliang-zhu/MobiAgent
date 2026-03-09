#!/usr/bin/env python3
"""
GRPO Reward 函数离线单测（阶段 4）

unit 模式（默认）：手工 22+ 用例，mock grounder
  conda run -n GRPO python workspace/data_tools/grpo/grpo_reward_test.py

dataset 模式：从 GRPO 训练集随机抽样，构造正负 completion 验证 reward
  conda run -n GRPO python workspace/data_tools/grpo/grpo_reward_test.py --mode dataset --sample_size 8 --seed 42

both 模式：unit + dataset 一起跑
  conda run -n GRPO python workspace/data_tools/grpo/grpo_reward_test.py --mode both --sample_size 8 --seed 42

可选：附加真实 grounder 集成探针（不影响主用例）
  conda run -n GRPO python workspace/data_tools/grpo/grpo_reward_test.py --with_grounder
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[3]
GRPO_ROOT = PROJECT_ROOT / "workspace" / "training" / "grpo"
if str(GRPO_ROOT) not in sys.path:
    sys.path.insert(0, str(GRPO_ROOT))

from src.train.reward_funcs import decider_reward  # noqa: E402


@dataclass
class RewardCase:
    name: str
    completion: Any
    gt_action: Dict[str, Any]
    expected: float
    mock_bbox: Optional[List[int]] = None


def _dummy_image() -> Image.Image:
    return Image.new("RGB", (128, 128), color="white")


def _mock_grounder_factory(bbox: Optional[List[int]]) -> Callable[..., Optional[List[int]]]:
    def _mock_grounder(**_: Any) -> Optional[List[int]]:
        return bbox

    return _mock_grounder


def _run_case(case: RewardCase, image: Image.Image) -> float:
    rewards = decider_reward(
        prompts=["unused"],
        completions=[case.completion],
        gt_action=[case.gt_action],
        images=[image],
        grounder_fn=_mock_grounder_factory(case.mock_bbox),
        log_details=False,
    )
    return float(rewards[0])


def _build_cases() -> List[RewardCase]:
    # Group 1: JSON parsing (4)
    cases = [
        RewardCase(
            name="G1-legal-json",
            completion='{"type":"done"}',
            gt_action={"type": "done"},
            expected=1.0,
        ),
        RewardCase(
            name="G1-truncated-json",
            completion='{"type":"done"',
            gt_action={"type": "done"},
            expected=0.0,
        ),
        RewardCase(
            name="G1-think-tag-json",
            completion='<think>analysis</think>{"type":"wait"}',
            gt_action={"type": "wait"},
            expected=1.0,
        ),
        RewardCase(
            name="G1-markdown-json",
            completion='```json\n{"type":"done"}\n```',
            gt_action={"type": "done"},
            expected=1.0,
        ),
        # Group 2: type mismatch (3)
        RewardCase(
            name="G2-click-vs-swipe",
            completion='{"type":"swipe","direction":"up"}',
            gt_action={"type": "click", "bounds": [0, 0, 100, 100]},
            expected=0.0,
        ),
        RewardCase(
            name="G2-input-vs-done",
            completion='{"type":"done"}',
            gt_action={"type": "input", "text": "abc"},
            expected=0.0,
        ),
        RewardCase(
            name="G2-done-vs-click",
            completion='{"type":"click","target_element":"搜索框"}',
            gt_action={"type": "done"},
            expected=0.0,
        ),
        # Group 3: done/wait (2)
        RewardCase(
            name="G3-done-match",
            completion='{"type":"done"}',
            gt_action={"type": "done"},
            expected=1.0,
        ),
        RewardCase(
            name="G3-wait-match",
            completion='{"type":"wait"}',
            gt_action={"type": "wait"},
            expected=1.0,
        ),
        # Group 4: swipe (3)
        RewardCase(
            name="G4-swipe-same",
            completion='{"type":"swipe","direction":"UP"}',
            gt_action={"type": "swipe", "direction": "UP"},
            expected=1.0,
        ),
        RewardCase(
            name="G4-swipe-diff",
            completion='{"type":"swipe","direction":"DOWN"}',
            gt_action={"type": "swipe", "direction": "UP"},
            expected=0.0,
        ),
        RewardCase(
            name="G4-swipe-case-insensitive",
            completion='{"type":"swipe","direction":"up"}',
            gt_action={"type": "swipe", "direction": "UP"},
            expected=1.0,
        ),
        # Group 5: input (3)
        RewardCase(
            name="G5-input-equal",
            completion='{"type":"input","text":"hello"}',
            gt_action={"type": "input", "text": "hello"},
            expected=1.0,
        ),
        RewardCase(
            name="G5-input-diff",
            completion='{"type":"input","text":"world"}',
            gt_action={"type": "input", "text": "hello"},
            expected=0.0,
        ),
        RewardCase(
            name="G5-input-strip-equal",
            completion='{"type":"input","text":"  hello  "}',
            gt_action={"type": "input", "text": "hello"},
            expected=1.0,
        ),
        # Group 6: click with mock grounder (5)
        RewardCase(
            name="G6-click-center-in-bounds",
            completion='{"type":"click","target_element":"搜索框","reasoning":"我要点击搜索"}',
            gt_action={"type": "click", "bounds": [0, 0, 40, 40]},
            expected=1.0,
            mock_bbox=[10, 10, 30, 30],
        ),
        RewardCase(
            name="G6-click-center-outside",
            completion='{"type":"click","target_element":"搜索框","reasoning":"我要点击搜索"}',
            gt_action={"type": "click", "bounds": [0, 0, 15, 15]},
            expected=0.0,
            mock_bbox=[10, 10, 30, 30],
        ),
        RewardCase(
            name="G6-click-grounder-none",
            completion='{"type":"click","target_element":"搜索框","reasoning":"我要点击搜索"}',
            gt_action={"type": "click", "bounds": [0, 0, 40, 40]},
            expected=0.0,
            mock_bbox=None,
        ),
        RewardCase(
            name="G6-click-empty-target",
            completion='{"type":"click","target_element":"","reasoning":"我要点击搜索"}',
            gt_action={"type": "click", "bounds": [0, 0, 40, 40]},
            expected=0.0,
            mock_bbox=[10, 10, 30, 30],
        ),
        RewardCase(
            name="G6-click-missing-gt-bounds",
            completion='{"type":"click","target_element":"搜索框","reasoning":"我要点击搜索"}',
            gt_action={"type": "click"},
            expected=0.0,
            mock_bbox=[10, 10, 30, 30],
        ),
        # Additional cases to exceed 20+
        RewardCase(
            name="G7-function-style-json",
            completion='{"reasoning":"输入关键词","function":{"name":"input","parameters":{"text":"abc"}}}',
            gt_action={"type": "input", "text": "abc"},
            expected=1.0,
        ),
        RewardCase(
            name="G7-action-nested-style-json",
            completion='{"action":{"type":"swipe","direction":"left"}}',
            gt_action={"type": "swipe", "direction": "LEFT"},
            expected=1.0,
        ),
    ]
    return cases


def _alt_type(gt_type: str) -> str:
    candidates = ["click", "input", "swipe", "done", "wait"]
    for t in candidates:
        if t != gt_type:
            return t
    return "done"


def _valid_bounds(bounds: Any) -> bool:
    return isinstance(bounds, list) and len(bounds) == 4


def _build_positive_completion(gt_action: Dict[str, Any]) -> str:
    t = str(gt_action.get("type", "")).strip().lower()
    if t in {"done", "wait"}:
        return json.dumps({"type": t}, ensure_ascii=False)
    if t == "swipe":
        return json.dumps({"type": "swipe", "direction": gt_action.get("direction", "")}, ensure_ascii=False)
    if t == "input":
        return json.dumps({"type": "input", "text": gt_action.get("text", "")}, ensure_ascii=False)
    if t == "click":
        return json.dumps(
            {
                "type": "click",
                "target_element": gt_action.get("target_element", "页面元素"),
                "reasoning": "点击目标元素",
            },
            ensure_ascii=False,
        )
    return json.dumps({"type": "done"}, ensure_ascii=False)


def _expected_positive(gt_action: Dict[str, Any]) -> float:
    t = str(gt_action.get("type", "")).strip().lower()
    if t in {"done", "wait"}:
        return 1.0
    if t == "swipe":
        return 1.0 if str(gt_action.get("direction", "")).strip() else 0.0
    if t == "input":
        return 1.0
    if t == "click":
        target = str(gt_action.get("target_element", "")).strip()
        return 1.0 if target and _valid_bounds(gt_action.get("bounds")) else 0.0
    return 0.0


def _build_negative_completion(gt_action: Dict[str, Any]) -> str:
    t = str(gt_action.get("type", "")).strip().lower()
    neg_t = _alt_type(t)
    if neg_t == "click":
        return json.dumps({"type": "click", "target_element": "无关元素", "reasoning": "错误动作"}, ensure_ascii=False)
    if neg_t == "input":
        return json.dumps({"type": "input", "text": "__WRONG_TEXT__"}, ensure_ascii=False)
    if neg_t == "swipe":
        return json.dumps({"type": "swipe", "direction": "down"}, ensure_ascii=False)
    return json.dumps({"type": neg_t}, ensure_ascii=False)


def _load_image_from_sample(sample: Dict[str, Any]) -> Image.Image:
    image_paths = sample.get("images") or []
    if image_paths:
        p = Path(image_paths[0])
        if p.exists():
            try:
                return Image.open(p).convert("RGB")
            except Exception:
                pass
    return _dummy_image()


def _run_unit_cases(with_grounder: bool) -> Tuple[List[float], List[str], int]:
    cases = _build_cases()
    image = _dummy_image()
    reward_values: List[float] = []
    failed: List[str] = []

    print("\n=== 模式: unit（手工用例） ===")
    for case in cases:
        reward = _run_case(case, image)
        reward_values.append(reward)
        if reward != case.expected:
            failed.append(f"{case.name}: expected {case.expected}, got {reward}")
        else:
            print(f"✓ {case.name} -> {reward}")

    total = len(cases)
    if with_grounder:
        print("\n[with_grounder] running real grounder probe...")
        try:
            probe_reward = _run_with_grounder_probe(image)
            reward_values.append(probe_reward)
            total += 1
            print(f"✓ with_grounder probe -> {probe_reward}")
        except Exception as e:
            total += 1
            failed.append(f"with_grounder probe failed: {e}")

    return reward_values, failed, total


def _run_dataset_cases(
    dataset_path: Path,
    sample_size: int,
    seed: int,
) -> Tuple[List[float], List[str], int]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise RuntimeError("dataset is empty or invalid")

    rng = random.Random(seed)
    k = min(sample_size, len(data))
    indices = sorted(rng.sample(range(len(data)), k=k))

    reward_values: List[float] = []
    failed: List[str] = []
    total = 0

    print("\n=== 模式: dataset（随机抽样） ===")
    print(f"dataset: {dataset_path}")
    print(f"sample_size={k}, seed={seed}, indices={indices}")

    for idx in indices:
        sample = data[idx]
        gt_action = sample.get("gt_action") or {}
        image = _load_image_from_sample(sample)
        gt_type = str(gt_action.get("type", "")).strip().lower()

        positive_completion = _build_positive_completion(gt_action)
        positive_expected = _expected_positive(gt_action)
        positive_bbox = gt_action.get("bounds") if gt_type == "click" and _valid_bounds(gt_action.get("bounds")) else None
        positive_reward = decider_reward(
            prompts=["unused"],
            completions=[positive_completion],
            gt_action=[gt_action],
            images=[image],
            grounder_fn=_mock_grounder_factory(positive_bbox),
            log_details=False,
        )[0]
        reward_values.append(float(positive_reward))
        total += 1
        if float(positive_reward) != positive_expected:
            failed.append(
                f"dataset idx={idx} positive expected={positive_expected} got={positive_reward} gt={gt_action}"
            )
        else:
            print(f"✓ idx={idx} positive({gt_type}) -> {positive_reward}")

        negative_completion = _build_negative_completion(gt_action)
        negative_reward = decider_reward(
            prompts=["unused"],
            completions=[negative_completion],
            gt_action=[gt_action],
            images=[image],
            grounder_fn=_mock_grounder_factory(positive_bbox),
            log_details=False,
        )[0]
        reward_values.append(float(negative_reward))
        total += 1
        if float(negative_reward) != 0.0:
            failed.append(f"dataset idx={idx} negative expected=0.0 got={negative_reward} gt={gt_action}")
        else:
            print(f"✓ idx={idx} negative({gt_type}) -> {negative_reward}")

    return reward_values, failed, total


def _run_with_grounder_probe(image: Image.Image) -> float:
    """Optional integration probe for real grounder call path."""
    rewards = decider_reward(
        prompts=["unused"],
        completions=['{"type":"click","target_element":"页面顶部搜索框","reasoning":"点击搜索"}'],
        gt_action=[{"type": "click", "bounds": [0, 0, 2000, 2000]}],
        images=[image],
        log_details=False,
    )
    reward = float(rewards[0])
    if reward not in (0.0, 1.0):
        raise RuntimeError(f"with_grounder probe reward 非法: {reward}")
    return reward


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO reward tests (unit/dataset/both)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["unit", "dataset", "both"],
        default="unit",
        help="测试模式：unit=手工用例，dataset=随机抽样，both=两者都跑",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(
            PROJECT_ROOT
            / "workspace"
            / "data"
            / "training_data"
            / "grpo_data"
            / "mobimind_decider_grpo_train.json"
        ),
        help="dataset 模式使用的 GRPO train JSON 路径",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=8,
        help="dataset 模式随机抽样条数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="dataset 模式随机种子",
    )
    parser.add_argument(
        "--with_grounder",
        action="store_true",
        help="附加真实 grounder 连通性探针（仅在 unit/both 下执行）",
    )
    args = parser.parse_args()

    all_rewards: List[float] = []
    failed: List[str] = []
    total = 0

    if args.mode in {"unit", "both"}:
        rewards, errs, cnt = _run_unit_cases(with_grounder=args.with_grounder)
        all_rewards.extend(rewards)
        failed.extend(errs)
        total += cnt

    if args.mode in {"dataset", "both"}:
        rewards, errs, cnt = _run_dataset_cases(
            dataset_path=Path(args.dataset_path),
            sample_size=max(1, args.sample_size),
            seed=args.seed,
        )
        all_rewards.extend(rewards)
        failed.extend(errs)
        total += cnt

    fail_cnt = len(failed)
    pass_cnt = total - fail_cnt

    print(f"\n总用例: {total}   通过: {pass_cnt}   失败: {fail_cnt}")
    print(f"所有 reward 值: {all_rewards}")

    unique_values = set(all_rewards)
    if not unique_values.issubset({0.0, 1.0}):
        failed.append(f"发现非法 reward 值: {sorted(unique_values)}")

    if failed:
        print("\n失败明细:")
        for msg in failed:
            print(f"- {msg}")
        return 1

    print("✓ 全部测试通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
