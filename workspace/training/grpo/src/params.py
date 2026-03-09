from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:  # pragma: no cover - compatibility shim
        pass

import transformers.training_args as _ta

if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from trl import GRPOConfig

from src.constants import DEFAULT_GROUNDER_URL


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    freeze_vision_tower: bool = field(default=True)
    freeze_merger: bool = field(default=False)
    freeze_llm: bool = field(default=True)
    disable_flash_attn2: bool = field(default=False)

    lora_enable: bool = field(default=True)
    lora_rank: int = field(default=64)
    lora_alpha: int = field(default=64)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_namespan_exclude: str = field(default="['visual']")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to GRPO train json"})
    eval_path: Optional[str] = field(default=None, metadata={"help": "Path to GRPO val json"})
    image_min_pixels: int = field(default=200704)  # 256*28*28
    image_max_pixels: int = field(default=501760)  # 640*28*28


@dataclass
class GRPOArguments(GRPOConfig):
    grounder_url: str = field(default=DEFAULT_GROUNDER_URL)
    grounder_timeout: float = field(default=30.0)
    log_reward_details: bool = field(default=True)

