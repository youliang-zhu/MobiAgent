import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from src.params import DataArguments


def _strip_image_prefix(instruction: str) -> str:
    text = (instruction or "").strip()
    if text.startswith("<image>"):
        text = text[len("<image>") :].lstrip("\n").strip()
    return text


def _load_image(path: str) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image file not found: {p}")
    return Image.open(p).convert("RGB")


def _extract_task(instruction: str) -> str:
    text = instruction or ""
    m = re.search(r'Now your task is\\s+"([^"]+)"', text)
    if m:
        return m.group(1).strip()
    return ""


class GRPODataset(Dataset):
    """
    Dataset format:
    {
      "instruction": "<image>\\n...",
      "images": ["/abs/path.jpg"],
      "gt_action": {...}
    }
    """

    def __init__(
        self,
        data_path: str | List[Dict[str, Any]],
        data_args: DataArguments,
        model_id: str,
        processor: Optional[Any] = None,
    ):
        del data_args  # reserved for future preprocessing controls
        del model_id
        del processor

        if isinstance(data_path, str):
            with open(data_path, "r", encoding="utf-8") as f:
                self.samples = json.load(f)
        else:
            self.samples = data_path

        if not isinstance(self.samples, list):
            raise ValueError("GRPO dataset must be a JSON list.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        instruction = _strip_image_prefix(sample.get("instruction", ""))
        image_paths = sample.get("images") or []
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if not image_paths:
            raise ValueError(f"Sample index {index} has no images.")
        images = [_load_image(p) for p in image_paths]

        # TRL conversational multimodal format
        prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        return {
            "prompt": prompt,
            "images": images,
            "gt_action": sample.get("gt_action", {}),
            "dataset_idx": int(index),
            "image_path": image_paths[0],
            "task": _extract_task(instruction),
        }


def grpo_data_collator(features: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    if not features:
        return {}
    keys = set()
    for feat in features:
        keys.update(feat.keys())
    return {k: [feat.get(k) for feat in features] for k in keys}


def make_grpo_data_module(
    processor: Any,
    model_id: str,
    data_args: DataArguments,
) -> Dict[str, Any]:
    train_dataset = GRPODataset(
        data_path=data_args.data_path,
        data_args=data_args,
        model_id=model_id,
        processor=processor,
    )
    eval_dataset = None
    if data_args.eval_path:
        eval_dataset = GRPODataset(
            data_path=data_args.eval_path,
            data_args=data_args,
            model_id=model_id,
            processor=processor,
        )

    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": grpo_data_collator,
    }
