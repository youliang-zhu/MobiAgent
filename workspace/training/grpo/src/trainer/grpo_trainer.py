import os
from typing import Dict

import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR, logger
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import entropy_from_logits, selective_log_softmax

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
            logger.warning(
                "Ignoring custom data_collator for GRPOTrainer. "
                "TRL GRPO expects identity collation (list[dict])."
            )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        rewards = super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        mode = "train" if self.model.training else "eval"
        stats = get_last_reward_stats()
        # Custom metrics required by grpo_plan.md
        self._metrics[mode]["reward/mean"].append(float(stats.get("reward_mean", 0.0)))
        self._metrics[mode]["json_valid_rate"].append(float(stats.get("json_valid_rate", 0.0)))
        self._metrics[mode]["action_type_acc"].append(float(stats.get("action_type_acc", 0.0)))
        self._metrics[mode]["click_hit_rate"].append(float(stats.get("click_hit_rate", 0.0)))
        self._metrics[mode]["click_iou_mean"].append(float(stats.get("click_iou_mean", 0.0)))

        return rewards

    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        pixel_values=None,
        image_grid_thw=None,
        num_images=None,
        pixel_attention_mask=None,
        image_sizes=None,
        token_type_ids=None,
    ):
        """
        Same logic as TRL, but handles the final short chunk safely.
        Upstream uses `start + batch_size` directly for multimodal row slicing,
        which can go out-of-bounds on the last eval chunk.
        """
        batch_size = batch_size or input_ids.size(0)
        all_logps = []
        all_entropies = []
        total = input_ids.size(0)

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            input_ids_batch = input_ids[start:end]
            attention_mask_batch = attention_mask[start:end]

            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat(
                    [torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)]
                )
                row_start, row_end = cum_rows[start].item(), cum_rows[end].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]

                cum_imgs = torch.tensor([0] + list(num_images), device=rows_per_sample.device).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[end]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start:end]
            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start:end]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start:end]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start:end]

            if "logits_to_keep" in self.model_kwarg_keys:
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False

            logits = model(**model_inputs).logits
            logits = logits[:, :-1, :]
            logits = logits[:, -logits_to_keep:, :]
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

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
