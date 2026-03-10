import ast
import importlib.util
import os
import shutil
from pathlib import Path
from typing import List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, HfArgumentParser, Qwen2_5_VLForConditionalGeneration

from src.constants import DEFAULT_LORA_TARGET_SUFFIXES
from src.dataset import make_grpo_data_module
from src.params import DataArguments, GRPOArguments, ModelArguments
from src.train.reward_funcs import decider_reward
from src.trainer import QwenGRPOTrainer
from src.trainer.grpo_trainer import get_peft_state_non_lora_maybe_zero_3

local_rank = None


def rank0_print(*args):
    if local_rank in (0, "0", None, -1):
        print(*args)


def _checkpoint_step(checkpoint_dir: Path) -> int:
    name = checkpoint_dir.name
    if not name.startswith("checkpoint-"):
        return -1
    try:
        return int(name.split("-", 1)[1])
    except Exception:
        return -1


def _list_checkpoints(output_dir: Path) -> List[Path]:
    checkpoints = [
        p for p in output_dir.glob("checkpoint-*")
        if p.is_dir() and _checkpoint_step(p) >= 0
    ]
    return sorted(checkpoints, key=_checkpoint_step)


def _select_best_checkpoint(trainer: QwenGRPOTrainer, output_dir: Path):
    # Prefer explicit eval metric if available in log history.
    candidate_metric_keys = ("eval_reward/mean", "eval_reward_mean", "eval_reward")
    best_step = None
    best_metric = None
    best_value = None
    for row in (trainer.state.log_history or []):
        if not isinstance(row, dict):
            continue
        step = row.get("step")
        if step is None:
            continue
        for metric_key in candidate_metric_keys:
            value = row.get(metric_key)
            if isinstance(value, (int, float)):
                if best_value is None or float(value) > float(best_value):
                    best_value = float(value)
                    best_step = int(step)
                    best_metric = metric_key
                break

    if best_step is not None:
        metric_ckpt = output_dir / f"checkpoint-{best_step}"
        if metric_ckpt.exists():
            return metric_ckpt, best_metric, best_value

    # Fallback to Trainer best pointer if configured.
    state_best = getattr(trainer.state, "best_model_checkpoint", None)
    if state_best:
        state_best_path = Path(state_best)
        if state_best_path.exists():
            return state_best_path, "state.best_model_checkpoint", None

    # Final fallback: latest checkpoint.
    checkpoints = _list_checkpoints(output_dir)
    if checkpoints:
        return checkpoints[-1], "latest_checkpoint_fallback", None

    return None, None, None


def _materialize_best_alias(output_dir: Path, best_ckpt: Path) -> str:
    best_path = output_dir / "best"
    if best_path.exists() or best_path.is_symlink():
        if best_path.is_symlink() or best_path.is_file():
            best_path.unlink()
        else:
            shutil.rmtree(best_path)

    try:
        # Keep folder size stable by default via symlink.
        link_target = best_ckpt.name if best_ckpt.parent == output_dir else str(best_ckpt.resolve())
        best_path.symlink_to(link_target)
        return f"symlink -> {link_target}"
    except Exception:
        shutil.copytree(best_ckpt, best_path)
        return f"copytree from {best_ckpt.name}"


def _load_sft_monkey_patch(module_name: str, file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"SFT monkey patch not found: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_sft_monkey_patches():
    sft_train_dir = Path(__file__).resolve().parents[3] / "sft" / "src" / "train"
    mp_forward = _load_sft_monkey_patch("sft_monkey_patch_forward", sft_train_dir / "monkey_patch_forward.py")
    mp_vision = _load_sft_monkey_patch("sft_monkey_patch_vision", sft_train_dir / "monkey_patch_vision.py")

    mp_forward.replace_qwen2_5_with_mixed_modality_forward()
    mp_vision.replace_qwen2_5_vision()
    rank0_print("Applied SFT monkey patches for Qwen2.5-VL.")


def set_requires_grad(parameters, requires_grad: bool):
    for p in parameters:
        p.requires_grad = requires_grad


def configure_vision_tower(model, model_args, compute_dtype, device):
    vision_tower = model.visual
    vision_tower.to(dtype=compute_dtype, device=device)
    set_requires_grad(vision_tower.parameters(), not model_args.freeze_vision_tower)
    set_requires_grad(model.visual.merger.parameters(), not model_args.freeze_merger)


def configure_llm(model, model_args):
    set_requires_grad(model.lm_head.parameters(), not model_args.freeze_llm)
    set_requires_grad(model.model.parameters(), not model_args.freeze_llm)


def find_target_linear_names(
    model,
    target_suffixes: List[str],
    lora_namespan_exclude: List[str],
):
    linear_cls = torch.nn.Linear
    target_modules = []
    for name, module in model.named_modules():
        if any(ex in name for ex in lora_namespan_exclude):
            continue
        if not isinstance(module, linear_cls):
            continue
        if any(name.endswith(sfx) for sfx in target_suffixes):
            target_modules.append(name)
    return sorted(set(target_modules))


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, GRPOArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank

    if model_args.lora_enable and not model_args.freeze_llm:
        raise ValueError("When lora_enable=True, freeze_llm must be True.")

    try:
        lora_namespan_exclude = ast.literal_eval(model_args.lora_namespan_exclude)
        if not isinstance(lora_namespan_exclude, list):
            raise ValueError
    except Exception:
        raise ValueError(f"Invalid lora_namespan_exclude: {model_args.lora_namespan_exclude}")

    compute_dtype = torch.float32
    if training_args.bf16:
        compute_dtype = torch.bfloat16
    elif training_args.fp16:
        compute_dtype = torch.float16

    apply_sft_monkey_patches()

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_id,
        dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not model_args.disable_flash_attn2 else "sdpa",
    )
    model.config.use_cache = False

    configure_llm(model, model_args)
    target_device = training_args.device if hasattr(training_args, "device") else ("cuda" if torch.cuda.is_available() else "cpu")
    configure_vision_tower(model, model_args, compute_dtype, target_device)

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    if model_args.lora_enable:
        target_modules = find_target_linear_names(
            model=model,
            target_suffixes=list(DEFAULT_LORA_TARGET_SUFFIXES),
            lora_namespan_exclude=lora_namespan_exclude,
        )
        if not target_modules:
            raise RuntimeError("No LoRA target modules found. Check lora_namespan_exclude.")

        rank0_print(f"LoRA target module count: {len(target_modules)}")
        rank0_print(f"LoRA target modules (first 20): {target_modules[:20]}")

        peft_config = LoraConfig(
            r=model_args.lora_rank,
            lora_alpha=model_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
        )
        model = get_peft_model(model, peft_config)

        # PEFT may override trainable flags, re-apply explicit settings.
        if not model_args.freeze_merger:
            for name, param in model.named_parameters():
                if "merger" in name:
                    param.requires_grad = True
        if not model_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "visual" in name:
                    param.requires_grad = True

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_id,
            min_pixels=data_args.image_min_pixels,
            max_pixels=data_args.image_max_pixels,
        )
    except TypeError:
        processor = AutoProcessor.from_pretrained(model_args.model_id)
        if hasattr(processor, "image_processor"):
            if hasattr(processor.image_processor, "min_pixels"):
                processor.image_processor.min_pixels = data_args.image_min_pixels
            if hasattr(processor.image_processor, "max_pixels"):
                processor.image_processor.max_pixels = data_args.image_max_pixels

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    os.environ["GROUNDER_URL"] = training_args.grounder_url
    os.environ["GROUNDER_TIMEOUT"] = str(training_args.grounder_timeout)
    os.environ["LOG_REWARD_DETAILS"] = "1" if training_args.log_reward_details else "0"
    os.environ["GRPO_NUM_GENERATIONS"] = str(training_args.num_generations)
    os.environ["REWARD_LOG_DIR"] = str(Path(training_args.output_dir) / "reward_logs")
    os.environ.setdefault("ENABLE_REWARD_LOGGER", "1")
    rank0_print(f"GROUNDER_URL={os.environ['GROUNDER_URL']}")
    rank0_print(f"GROUNDER_TIMEOUT={os.environ['GROUNDER_TIMEOUT']}")
    rank0_print(f"REWARD_LOG_DIR={os.environ['REWARD_LOG_DIR']}")

    data_module = make_grpo_data_module(
        processor=processor,
        model_id=model_args.model_id,
        data_args=data_args,
    )
    rank0_print(f"Train dataset size: {len(data_module['train_dataset'])}")
    if data_module["eval_dataset"] is not None:
        rank0_print(f"Eval dataset size: {len(data_module['eval_dataset'])}")

    trainer = QwenGRPOTrainer(
        model=model,
        reward_funcs=[decider_reward],
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        processing_class=processor,
    )

    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if _list_checkpoints(output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    trainer.save_model(training_args.output_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(training_args.output_dir)

    if model_args.lora_enable and trainer.args.should_save:
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            trainer.model.named_parameters(), require_grad_only=False
        )
        torch.save(non_lora_state_dict, output_dir / "non_lora_state_dict.bin")

    best_ckpt, best_metric, best_value = _select_best_checkpoint(trainer, output_dir)
    if best_ckpt is None:
        rank0_print("No checkpoint found. Skip creating output_dir/best alias.")
    else:
        alias_desc = _materialize_best_alias(output_dir, best_ckpt)
        if best_value is None:
            rank0_print(
                f"Best checkpoint alias created: {output_dir / 'best'} ({alias_desc}, source={best_metric})."
            )
        else:
            rank0_print(
                f"Best checkpoint alias created: {output_dir / 'best'} ({alias_desc}, metric={best_metric}, value={best_value:.6f})."
            )


if __name__ == "__main__":
    train()
