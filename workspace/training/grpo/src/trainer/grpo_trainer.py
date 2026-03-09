import os
from typing import Dict

import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR, logger
from trl import GRPOTrainer

from src.train.reward_funcs import get_last_reward_stats


def maybe_zero_3(param, ignore_status: bool = False, name: str | None = None):
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    except Exception:
        return param.detach().cpu().clone()

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE and not ignore_status:
            logger.warning("%s no ignore status", name)
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only: bool = True) -> Dict[str, torch.Tensor]:
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    return {k: maybe_zero_3(v, ignore_status=True, name=k) for k, v in to_return.items()}


class QwenGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        custom_collator = kwargs.pop("data_collator", None)
        super().__init__(*args, **kwargs)
        if custom_collator is not None:
            self.data_collator = custom_collator

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        mode = "train" if self.model.training else "eval"
        stats = get_last_reward_stats()
        # Custom metrics required by grpo_plan.md
        self._metrics[mode]["reward/mean"].append(float(stats.get("reward_mean", 0.0)))
        self._metrics[mode]["json_valid_rate"].append(float(stats.get("json_valid_rate", 0.0)))
        self._metrics[mode]["action_type_acc"].append(float(stats.get("action_type_acc", 0.0)))
        self._metrics[mode]["click_hit_rate"].append(float(stats.get("click_hit_rate", 0.0)))

        return rewards

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)

        if not getattr(self.args, "lora_enable", False):
            return
        if not self.args.should_save:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        try:
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(), require_grad_only=False
            )
            torch.save(non_lora_state_dict, os.path.join(output_dir, "non_lora_state_dict.bin"))
        except Exception as e:
            logger.warning("Failed to save non_lora_state_dict.bin at step %s: %s", self.state.global_step, e)

