import os
import torch
from pathlib import Path
import torch.nn as nn
from typing import Any

import bitsandbytes

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    TRAINER_STATE_NAME,
    PREFIX_CHECKPOINT_DIR,
    logger,
    ExportableState,
    SaveStrategy
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from trl import GRPOTrainer
from trl.data_utils import is_conversational
from trl.trainer.utils import (
    pad,
    nanmax,
    nanmin,
    nanstd,
    selective_log_softmax,
    entropy_from_logits,
)
from trl.extras.profiling import profiling_decorator
from accelerate.utils import gather_object, is_peft_model
from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3


class QwenGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        # GRPOTrainer 不接受 data_collator 参数，需要拦截并在初始化后覆盖
        # 从 kwargs 中提取 data_collator（如果存在）
        custom_data_collator = kwargs.pop('data_collator', None)
        
        # 调用父类初始化（不传 data_collator）
        super(QwenGRPOTrainer, self).__init__(*args, **kwargs)
        
        # 如果提供了自定义 data_collator，覆盖父类默认的 collator
        if custom_data_collator is not None:
            self.data_collator = custom_data_collator
        
        # Log generation_config for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Generation Config: {self.generation_config}")
        logger.info(f"max_completion_length from args: {self.max_completion_length}")

    @profiling_decorator
    def _prepare_inputs(self, generation_batch: dict[str, torch.Tensor | Any]) -> dict[str, torch.Tensor | Any]:
        """
        Override to handle scalar metadata fields that cannot be shuffled.
        
        TRL's shuffle_sequence_dict assumes all values are sequences, but _generate_and_score_completions
        returns some scalar fields like 'num_items_in_batch'. We need to extract these before shuffling
        and restore them after.
        """
        from trl.trainer.utils import shuffle_sequence_dict, split_tensor_dict
        
        mode = "train" if self.model.training else "eval"
        if mode == "train":
            generate_every = self.args.steps_per_generation * self.num_iterations
            if self._step % generate_every == 0 or self._buffered_inputs is None:
                # Generate and score completions
                generation_batch = self._generate_and_score_completions(generation_batch)
                
                # Extract non-sequence (scalar) metadata before shuffling
                scalar_fields = {}
                sequence_fields = {}
                for key, val in generation_batch.items():
                    # Check if value is a sequence (list, tuple, or tensor with batch dimension)
                    if isinstance(val, (list, tuple)):
                        sequence_fields[key] = val
                    elif isinstance(val, torch.Tensor):
                        # Tensors with more than 0 dimensions are sequences
                        if val.ndim > 0:
                            sequence_fields[key] = val
                        else:
                            scalar_fields[key] = val
                    else:
                        # Scalars (int, float, etc.)
                        scalar_fields[key] = val
                
                # Shuffle only sequence fields
                from trl.trainer.utils import split_pixel_values_by_grid, unsplit_pixel_values_by_grid
                
                sequence_fields = split_pixel_values_by_grid(sequence_fields)
                sequence_fields = shuffle_sequence_dict(sequence_fields)
                generation_batches = split_tensor_dict(sequence_fields, self.args.steps_per_generation)
                
                # Restore scalar fields to each batch
                self._buffered_inputs = []
                for batch in generation_batches:
                    # Add scalar fields back to each split batch
                    for key, val in scalar_fields.items():
                        batch[key] = val
                    self._buffered_inputs.append(unsplit_pixel_values_by_grid(batch))
            
            inputs = self._buffered_inputs[self._step % self.args.steps_per_generation]
            self._step += 1
        else:
            # In evaluation, there is neither batch grouping for generation, nor multiple iterations
            inputs = self._generate_and_score_completions(generation_batch)
        
        return inputs

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        # Note: gt_action is wrapped in a list in the dataset to allow TRL's shuffle_sequence_dict to work
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "assistant", "image", "images", "video", "videos", "video_kwargs", "gt_action"]

    def _generate(self, prompts: list, images: list = None):
        """
        Override _generate to support VLM by passing images to processor.
        This fixes TRL's incomplete VLM support where images are not passed during generation.
        """
        from trl.extras.profiling import profiling_context
        from contextlib import nullcontext
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Only support regular generation path (not vLLM or paged)
        # CRITICAL: For VLM, do NOT truncate to avoid cutting image_pad tokens
        # Image tokens can be 4600+ per image, truncation causes misalignment
        processor_kwargs = {
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            # "max_length": self.max_prompt_length,  # Removed: causes image_pad truncation
            # "truncation": True,  # Removed: must NOT truncate VLM inputs
            "add_special_tokens": False,
        }
        
        if is_conversational({"prompt": prompts[0]}):
            import logging
            logger = logging.getLogger(__name__)
            logger.info("CONV PATH: Using conversational format")
            
            # ======================================================================
            # EXPERIMENT CONTROL: Test different strategies
            # ======================================================================
            # Strategy A: Original (remove EOS + manual assistant header + mask remaining EOS)
            # Strategy B: Tokenizer native (add_generation_prompt=True, NO modifications)
            # Strategy C: Hybrid (remove EOS + manual header, NO masking)
            # Set TEST_STRATEGY = 'A', 'B', or 'C'
            TEST_STRATEGY = 'B'  # Change this to test different approaches
            
            logger.info("=" * 80)
            logger.info(f"EXPERIMENT: Testing Strategy {TEST_STRATEGY}")
            if TEST_STRATEGY == 'A':
                logger.info("Strategy A: Remove EOS + Manual Assistant Header + Mask Remaining EOS")
            elif TEST_STRATEGY == 'B':
                logger.info("Strategy B: Tokenizer Native (add_generation_prompt=True, NO mods)")
            elif TEST_STRATEGY == 'C':
                logger.info("Strategy C: Remove EOS + Manual Header, NO Masking")
            logger.info("=" * 80)
            
            # DEBUG: Check message structure AND image content
            first_prompt = prompts[0]
            roles = [msg['role'] for msg in first_prompt]
            has_system = 'system' in roles
            logger.info(f"CONV PATH: Message roles in prompt: {roles}")
            logger.info(f"CONV PATH: Has system role: {has_system}")
            
            # Check if images are present in message content
            for msg in first_prompt:
                if msg['role'] == 'user' and isinstance(msg['content'], list):
                    image_items = [item for item in msg['content'] if item.get('type') == 'image']
                    text_items = [item for item in msg['content'] if item.get('type') == 'text']
                    logger.info(f"CONV PATH: User message has {len(image_items)} image items and {len(text_items)} text items")
                    if image_items:
                        # Check first image object
                        img = image_items[0].get('image')
                        if img is not None:
                            logger.info(f"CONV PATH: First image type: {type(img)}, shape: {getattr(img, 'size', 'N/A')}")
                        else:
                            logger.error("CONV PATH: Image item exists but image is None!")
            
            if has_system:
                logger.warning("CONV PATH: WARNING - System role found! This will add EOS in middle of prompt!")
            
            # For conversational format, use apply_chat_template
            # Images are already embedded in the messages content, no need to pass separately
            # Strategy B uses add_generation_prompt=True (tokenizer handles everything)
            # Strategy A/C use add_generation_prompt=False (manual handling)
            use_generation_prompt = (TEST_STRATEGY == 'B')
            
            # CRITICAL FIX: Process each sample independently to generate correct image_pad tokens
            # Batch processing causes only 1 image worth of image_pad tokens to be generated
            logger.info(f"CONV PATH: Processing {len(prompts)} samples independently...")
            all_input_ids = []
            all_attention_masks = []
            all_pixel_values = []
            all_image_grid_thw = []
            
            image_pad_token_id = 151655
            
            for i, prompt in enumerate(prompts):
                sample_inputs = self.processing_class.apply_chat_template(
                    conversation=[prompt],  # Process single sample
                    add_generation_prompt=use_generation_prompt,
                    tokenize=True,
                    return_dict=True,
                    **processor_kwargs,
                    **self.chat_template_kwargs,
                )
                
                # DEBUG: Check each sample's image_pad count
                single_image_pads = (sample_inputs['input_ids'] == image_pad_token_id).sum().item()
                expected_pads = sample_inputs['image_grid_thw'].prod(dim=1).sum().item() if 'image_grid_thw' in sample_inputs else 0
                match_status = "✓ MATCH" if single_image_pads == expected_pads else f"✗ MISMATCH (expected {expected_pads})"
                logger.info(f"CONV PATH: Sample {i} - input_ids shape: {sample_inputs['input_ids'].shape}, image_pad count: {single_image_pads} {match_status}")
                if 'pixel_values' in sample_inputs:
                    logger.info(f"CONV PATH: Sample {i} - pixel_values shape: {sample_inputs['pixel_values'].shape}")
                if 'image_grid_thw' in sample_inputs:
                    logger.info(f"CONV PATH: Sample {i} - image_grid_thw: {sample_inputs['image_grid_thw'].tolist()}")
                
                all_input_ids.append(sample_inputs['input_ids'])
                all_attention_masks.append(sample_inputs['attention_mask'])
                if 'pixel_values' in sample_inputs:
                    all_pixel_values.append(sample_inputs['pixel_values'])
                if 'image_grid_thw' in sample_inputs:
                    all_image_grid_thw.append(sample_inputs['image_grid_thw'])
            
            # DEBUG: Check before concatenation
            logger.info(f"CONV PATH: all_input_ids length: {len(all_input_ids)}")
            logger.info(f"CONV PATH: all_input_ids[0] shape: {all_input_ids[0].shape}")
            logger.info(f"CONV PATH: all_pixel_values length: {len(all_pixel_values) if all_pixel_values else 0}")
            logger.info(f"CONV PATH: all_image_grid_thw length: {len(all_image_grid_thw) if all_image_grid_thw else 0}")
            
            # Concatenate all samples
            generate_inputs = {
                'input_ids': torch.cat(all_input_ids, dim=0),
                'attention_mask': torch.cat(all_attention_masks, dim=0),
            }
            if all_pixel_values:
                generate_inputs['pixel_values'] = torch.cat(all_pixel_values, dim=0)
            if all_image_grid_thw:
                generate_inputs['image_grid_thw'] = torch.cat(all_image_grid_thw, dim=0)
            
            # DEBUG: Check after concatenation
            total_image_pads_after_cat = (generate_inputs['input_ids'] == image_pad_token_id).sum().item()
            logger.info(f"CONV PATH: After concatenation - input_ids shape: {generate_inputs['input_ids'].shape}")
            logger.info(f"CONV PATH: After concatenation - total image_pad count: {total_image_pads_after_cat}")
            
            logger.info(f"CONV PATH: add_generation_prompt={use_generation_prompt}")
            
            logger.info(f"CONV PATH: After apply_chat_template, input_ids shape: {generate_inputs['input_ids'].shape}")
            logger.info(f"CONV PATH: Last 10 tokens before removal: {generate_inputs['input_ids'][0, -10:].tolist()}")
            
            batch_size = generate_inputs["input_ids"].size(0)
            device = generate_inputs["input_ids"].device
            eos_token_id = self.processing_class.tokenizer.eos_token_id
            
            # === DEEP DEBUG: Check ALL EOS positions in prompt ===
            first_prompt_ids = generate_inputs["input_ids"][0]
            eos_mask = (first_prompt_ids == eos_token_id)
            eos_positions = eos_mask.nonzero(as_tuple=True)[0].tolist()
            logger.info(f"CONV PATH: Found {len(eos_positions)} EOS tokens (151645) in prompt at positions: {eos_positions}")
            
            # Print context around each EOS
            for i, pos in enumerate(eos_positions[:5]):  # Print first 5 EOS tokens
                start = max(0, pos - 5)
                end = min(len(first_prompt_ids), pos + 6)
                context_tokens = first_prompt_ids[start:end].tolist()
                logger.info(f"CONV PATH: EOS #{i+1} at position {pos}, context: {context_tokens}")
            
            # === Print full prompt tokens (first 50 and last 50) ===
            prompt_len = first_prompt_ids.size(0)
            logger.info(f"CONV PATH: Full prompt length: {prompt_len}")
            logger.info(f"CONV PATH: First 50 tokens: {first_prompt_ids[:50].tolist()}")
            logger.info(f"CONV PATH: Last 50 tokens: {first_prompt_ids[-50:].tolist()}")
            
            # Decode first and last parts for readability
            first_text = self.processing_class.tokenizer.decode(first_prompt_ids[:50].tolist(), skip_special_tokens=False)
            last_text = self.processing_class.tokenizer.decode(first_prompt_ids[-50:].tolist(), skip_special_tokens=False)
            logger.info(f"CONV PATH: First 50 tokens decoded: {repr(first_text)}")
            logger.info(f"CONV PATH: Last 50 tokens decoded: {repr(last_text)}")
            
            # ======================================================================
            # STRATEGY-SPECIFIC PROCESSING
            # ======================================================================
            
            if TEST_STRATEGY == 'B':
                # Strategy B: Do NOTHING - use tokenizer output as-is
                logger.info("Strategy B: NO modifications to tokenizer output")
                logger.info(f"Strategy B: Using prompt as-is, length: {generate_inputs['input_ids'].shape[1]}")
                
            else:
                # Strategy A & C: Remove trailing EOS and add manual assistant header
                logger.info(f"Strategy {TEST_STRATEGY}: Removing trailing EOS and adding manual assistant header")
                
                # Remove trailing <|im_end|>\n (EOS token 151645 + newline 198)
                if generate_inputs["input_ids"][0, -2] == eos_token_id and generate_inputs["input_ids"][0, -1] == 198:
                    logger.info("Removing trailing EOS (151645) and newline (198)")
                    generate_inputs["input_ids"] = generate_inputs["input_ids"][:, :-2]
                    generate_inputs["attention_mask"] = generate_inputs["attention_mask"][:, :-2]
                
                logger.info(f"After EOS removal, last 10 tokens: {generate_inputs['input_ids'][0, -10:].tolist()}")
                
                # Check how many EOS tokens remain
                remaining_eos = (generate_inputs["input_ids"][0] == eos_token_id).sum().item()
                logger.info(f"Remaining EOS tokens after removal: {remaining_eos}")
            
            # ======================================================================
            # MASKING (Strategy A only) or NO-OP (Strategy B/C)
            # ======================================================================
            
            if TEST_STRATEGY == 'A':
                # Strategy A: Mask ALL remaining EOS tokens
                logger.info("=" * 80)
                logger.info("Strategy A: Masking ALL EOS tokens in attention")
                logger.info("=" * 80)
                
                eos_mask_full = (generate_inputs["input_ids"] == eos_token_id)
                total_eos_to_mask = eos_mask_full.sum().item()
                eos_per_sample = eos_mask_full.sum(dim=1).tolist()
                
                logger.info(f"MASK DEBUG: Total EOS tokens to mask: {total_eos_to_mask}")
                logger.info(f"MASK DEBUG: EOS per sample: {eos_per_sample}")
                
                first_sample_eos_positions = eos_mask_full[0].nonzero(as_tuple=True)[0].tolist()
                logger.info(f"MASK DEBUG: First sample EOS positions: {first_sample_eos_positions}")
                
                masked_before = (generate_inputs["attention_mask"] == 0).sum().item()
                generate_inputs["attention_mask"][eos_mask_full] = 0
                masked_after = (generate_inputs["attention_mask"] == 0).sum().item()
                
                logger.info(f"MASK DEBUG: Newly masked: {masked_after - masked_before}")
                logger.info("=" * 80)
                
            elif TEST_STRATEGY == 'C':
                logger.info("Strategy C: NO masking (testing if masking is the problem)")
                
            else:  # Strategy B
                logger.info("Strategy B: NO masking needed (using tokenizer native format)")
            
            # ======================================================================
            # ADD ASSISTANT HEADER (Strategy A/C only)
            # ======================================================================
            
            if TEST_STRATEGY in ['A', 'C']:
                # Manually add the generation prompt tokens: <|im_start|>assistant\n
                # Token IDs: 151644 (im_start), 77091 (assistant), 198 (newline)
                logger.info(f"Strategy {TEST_STRATEGY}: Adding manual assistant header tokens")
                gen_prompt_tokens = torch.tensor([[151644, 77091, 198]], device=device).repeat(batch_size, 1)
                generate_inputs["input_ids"] = torch.cat([generate_inputs["input_ids"], gen_prompt_tokens], dim=1)
                generate_inputs["attention_mask"] = torch.cat([
                    generate_inputs["attention_mask"],
                    torch.ones(batch_size, 3, dtype=torch.long, device=device)
                ], dim=1)
                logger.info(f"After manual addition, input_ids shape: {generate_inputs['input_ids'].shape}")
                logger.info(f"Last 10 tokens after addition: {generate_inputs['input_ids'][0, -10:].tolist()}")
            else:
                # Strategy B: No manual addition needed
                logger.info(f"Strategy B: Using tokenizer-generated prompt as-is")
                logger.info(f"Final prompt shape: {generate_inputs['input_ids'].shape}")
                logger.info(f"Last 10 tokens: {generate_inputs['input_ids'][0, -10:].tolist()}")
        else:
            # For non-conversational format, add images to processor kwargs
            if images is not None:
                processor_kwargs["images"] = images
            generate_inputs = self.processing_class(text=prompts, **processor_kwargs)
        
        generate_inputs = Trainer._prepare_inputs(self, generate_inputs)

        # ======================================================================
        # CRITICAL DEBUG: Check if vision inputs are passed to generate()
        # ======================================================================
        import logging
        logger = logging.getLogger(__name__)
        logger.info("=" * 80)
        logger.info("VISION INPUTS CHECK BEFORE generate()")
        logger.info("=" * 80)
        logger.info(f"generate_inputs keys: {list(generate_inputs.keys())}")
        logger.info(f"Has pixel_values: {'pixel_values' in generate_inputs}")
        logger.info(f"Has image_grid_thw: {'image_grid_thw' in generate_inputs}")
        
        if 'pixel_values' in generate_inputs:
            logger.info(f"pixel_values shape: {generate_inputs['pixel_values'].shape}")
            logger.info(f"pixel_values device: {generate_inputs['pixel_values'].device}")
        else:
            logger.error("CRITICAL: pixel_values NOT in generate_inputs!")
            logger.error("Model will NOT see images during generation!")
            
        if 'image_grid_thw' in generate_inputs:
            logger.info(f"image_grid_thw shape: {generate_inputs['image_grid_thw'].shape}")
            logger.info(f"image_grid_thw values (first sample): {generate_inputs['image_grid_thw'][0].tolist()}")
        else:
            logger.error("CRITICAL: image_grid_thw NOT in generate_inputs!")
            logger.error("Model will NOT understand image structure!")
            
        logger.info(f"input_ids shape: {generate_inputs['input_ids'].shape}")
        logger.info(f"attention_mask shape: {generate_inputs['attention_mask'].shape}")
        
        # ======================================================================
        # CRITICAL DEBUG: Verify image_pad token count matches pixel_values
        # ======================================================================
        image_pad_token_id = 151655  # <|image_pad|>
        num_image_pads = (generate_inputs["input_ids"] == image_pad_token_id).sum().item()
        expected_pads = generate_inputs["image_grid_thw"].prod(dim=1).sum().item()
        logger.info(f"IMAGE_PAD VERIFICATION:")
        logger.info(f"  Actual image_pad count in input_ids: {num_image_pads}")
        logger.info(f"  Expected image_pad count (from image_grid_thw): {expected_pads}")
        logger.info(f"  Match: {num_image_pads == expected_pads}")
        
        if num_image_pads != expected_pads:
            logger.error("=" * 80)
            logger.error("CRITICAL MISMATCH: image_pad token count != expected count!")
            logger.error(f"  This will cause Vision Encoder misalignment!")
            logger.error(f"  Difference: {num_image_pads - expected_pads}")
            logger.error(f"  image_grid_thw per sample: {generate_inputs['image_grid_thw'].tolist()}")
            logger.error("=" * 80)
        
        logger.info("=" * 80)

        # Unwrap model for generation - compatible with newer TRL versions
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Ensure generation config is used correctly
        # Pass generation parameters explicitly to avoid model overriding them
        gen_kwargs = {
            "max_new_tokens": self.max_completion_length,
            "do_sample": True,
            "temperature": getattr(self.args, 'temperature', 0.9),
            "top_p": getattr(self.args, 'top_p', 1.0),
            "top_k": getattr(self.args, 'top_k', 50),
            "eos_token_id": self.processing_class.tokenizer.eos_token_id,
            "pad_token_id": self.processing_class.tokenizer.pad_token_id,
        }
        
        logger.info(f"gen_kwargs: {gen_kwargs}")
        logger.info("Calling unwrapped_model.generate()...")
        
        with (
            profiling_context(self, "transformers.generate"),
            torch.no_grad(),
            FSDP.summon_full_params(self.model_wrapped, recurse=False) if self.is_fsdp_enabled else nullcontext(),
        ):
            prompt_completion_ids = unwrapped_model.generate(
                **generate_inputs, **gen_kwargs, disable_compile=True
            )
        
        logger.info(f"Generation complete. Output shape: {prompt_completion_ids.shape}")
        
        # Compute prompt length and extract completion ids
        prompt_ids, prompt_mask = generate_inputs["input_ids"], generate_inputs["attention_mask"]
        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        
        # Convert to list format expected by parent class
        prompt_ids_list = prompt_ids.tolist()
        completion_ids_list = completion_ids.tolist()
        
        # Debug: log generated completions (first batch only)
        if not hasattr(self, '_generation_logged'):
            import logging
            logger = logging.getLogger(__name__)
            logger.info("="*80)
            logger.info("GENERATION DEBUG - First Batch")
            logger.info("="*80)
            logger.info(f"gen_kwargs: {gen_kwargs}")
            logger.info(f"prompt_ids shape: {prompt_ids.shape}")
            logger.info(f"prompt_completion_ids shape: {prompt_completion_ids.shape}")
            logger.info(f"prompt_length (from input_ids): {prompt_length}")
            logger.info(f"completion_ids shape: {completion_ids.shape}")
            # Check if generate_inputs has pixel_values
            if "pixel_values" in generate_inputs:
                logger.info(f"pixel_values shape: {generate_inputs['pixel_values'].shape}")
            else:
                logger.info("WARNING: No pixel_values in generate_inputs!")
            if "image_grid_thw" in generate_inputs:
                logger.info(f"image_grid_thw shape: {generate_inputs['image_grid_thw'].shape}")
            else:
                logger.info("WARNING: No image_grid_thw in generate_inputs!")
            
            # Check the last few tokens of prompt and first few of completion
            logger.info(f"Last 10 prompt tokens: {prompt_ids[0, -10:].tolist()}")
            logger.info(f"First completion (all tokens): {completion_ids[0].tolist()}")
            logger.info(f"EOS token ID: {self.processing_class.tokenizer.eos_token_id}")
            logger.info(f"PAD token ID: {self.processing_class.tokenizer.pad_token_id}")
            # Check if completion contains EOS
            contains_eos = [(self.processing_class.tokenizer.eos_token_id in comp_ids.tolist()) for comp_ids in completion_ids]
            logger.info(f"Completions contain EOS: {contains_eos}")
            
            # Decode first 3 completions
            for idx in range(min(3, len(completion_ids_list))):
                completion_text = self.processing_class.decode(completion_ids_list[idx], skip_special_tokens=True)
                logger.info(f"\n[Generation {idx}]")
                logger.info(f"Completion IDs length: {len(completion_ids_list[idx])}")
                logger.info(f"Completion text:\n{completion_text}")
            self._generation_logged = True
        
        # Get completion length per sequence
        prompt_lengths = torch.tensor([len(ids) for ids in prompt_ids_list], device=device)
        completion_lengths = torch.tensor([len(ids) for ids in completion_ids_list], device=device)
        
        # Calculate num_items_in_batch
        num_items_in_batch = len(prompts) // self.num_generations
        
        # Log generation stats (simplified, no detailed metrics)
        mean_prompt_length = prompt_lengths.float().mean().item()
        mean_completion_length = completion_lengths.float().mean().item()
        
        self._metrics[mode]["generation/prompt_length/mean"].append(
            self.accelerator.gather(torch.tensor(mean_prompt_length, device=device)).mean().item()
        )
        self._metrics[mode]["generation/completion_length/mean"].append(
            self.accelerator.gather(torch.tensor(mean_completion_length, device=device)).mean().item()
        )
        
        # Return in expected format: (prompt_ids_list, completion_ids_list, num_items_in_batch, logprobs, extra_fields)
        return prompt_ids_list, completion_ids_list, num_items_in_batch, None, {}

    def _generate_and_score_completions(
        self, inputs: dict[str, list]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Modified to handle inputs as Dict[str, List] format (from data_collator).
        """
        device = self.accelerator.device
        mode = "train" if self.model.training else "eval"

        # Extract data from Dict[str, List] format
        prompts = inputs["prompt"]
        
        # Handle images
        if "images" in inputs:
            images = inputs["images"]
        elif "image" in inputs:
            images = inputs["image"]
        else:
            images = None
        # Transformers requires at least one image in the batch, otherwise it throws an error
        if images is not None and all(img is None or (isinstance(img, list) and len(img) == 0) for img in images):
            images = None

        # Handle videos
        if "videos" in inputs:
            videos = inputs["videos"]
            video_kwargs = inputs.get("video_kwargs")
        elif "video" in inputs:
            videos = inputs["video"]
            video_kwargs = inputs.get("video_kwargs")
        else:
            videos = None

        if videos is not None and all(v is None or (isinstance(v, list) and len(v) == 0) for v in videos):
            videos = None

        # Pass images to _generate for proper VLM support
        prompt_ids_list, completion_ids_list, num_items_in_batch, sampling_per_token_logps_list, extra_fields = (
            self._generate(prompts, images=images)
        )

        # Convert lists of token IDs to padded tensors
        prompt_ids = [torch.tensor(ids, device=device) for ids in prompt_ids_list]
        prompt_mask = [torch.ones_like(ids, dtype=torch.long) for ids in prompt_ids]
        prompt_ids = pad(prompt_ids, padding_value=self.pad_token_id, padding_side="left")
        prompt_mask = pad(prompt_mask, padding_value=0, padding_side="left")
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids_list]
        completion_mask = [torch.ones_like(ids, dtype=torch.long) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.pad_token_id, padding_side="right")
        completion_mask = pad(completion_mask, padding_value=0, padding_side="right")
        if sampling_per_token_logps_list is not None:
            sampling_per_token_logps = [torch.tensor(logps, device=device) for logps in sampling_per_token_logps_list]
            sampling_per_token_logps = pad(sampling_per_token_logps, padding_value=0.0, padding_side="right")
        else:
            sampling_per_token_logps = None

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            eos_and_pad = [self.eos_token_id, self.pad_token_id]
            is_truncated = torch.tensor([ids[-1] not in eos_and_pad for ids in completion_ids_list], device=device)
            completion_mask = completion_mask * (~is_truncated).unsqueeze(1).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)  # (B, P+C)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        batch_size = self.args.per_device_train_batch_size if mode == "train" else self.args.per_device_eval_batch_size

        num_images = [len(img_list) for img_list in images] if images is not None else None

        model_id = getattr(self.model.config, "_name_or_path", "")

        # Get forward_kwargs for models with multimodal inputs
        if images is not None or videos is not None:
            # 使用与 _generate 完全相同的参数确保 input_ids 一致
            processor_kwargs = dict(
                return_tensors="pt",
                padding=True,
                padding_side="left",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=False,
                do_resize=False
            )
            
            # 检查是否为 conversational format
            if is_conversational({"prompt": prompts[0]}):
                # For conversational format, use apply_chat_template
                # Images are already embedded in the messages content
                prompt_inputs = self.processing_class.apply_chat_template(
                    conversation=prompts,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **processor_kwargs,
                    **self.chat_template_kwargs,
                )
            else:
                # For non-conversational format (legacy text mode)
                processor_kwargs["text"] = prompts
                
                if images is not None:
                    processor_kwargs["images"] = images
                if videos is not None:
                    if "Qwen2.5" in model_id:
                        common_vk = video_kwargs[0] if isinstance(video_kwargs, list) else video_kwargs
                        processor_kwargs["videos"] = videos
                        if common_vk is not None:
                            processor_kwargs.update(common_vk)
                    elif "Qwen3" in model_id:
                        batched_video_datas = []
                        batched_video_metadatas = []
                        for sample_videos in videos:
                            if sample_videos is None:
                                batched_video_datas.append(None)
                                batched_video_metadatas.append(None)
                            else:
                                datas, metas = zip(*sample_videos)
                                batched_video_datas.append(list(datas))
                                batched_video_metadatas.append(list(metas))
                        processor_kwargs["videos"] = batched_video_datas
                        processor_kwargs["video_metadata"] = batched_video_metadatas
                        
                        common_vk = video_kwargs[0] if isinstance(video_kwargs, list) else video_kwargs
                        if common_vk is not None:
                            processor_kwargs.update(common_vk)
                    else:
                        processor_kwargs["videos"] = videos
                
                prompt_inputs = self.processing_class(**processor_kwargs)
            # 使用 Trainer._prepare_inputs 而不是 super()._prepare_inputs
            # 以避免调用 GRPOTrainer._prepare_inputs 导致递归
            prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
            forward_kwargs = {k: v for k, v in prompt_inputs.items() if k not in ["input_ids", "attention_mask"]}
        else:
            forward_kwargs = {}

        # If token_type_ids are used, extend them with zeros for the completion part
        if "token_type_ids" in forward_kwargs:
            token_type_ids = forward_kwargs["token_type_ids"]
            forward_kwargs["token_type_ids"] = torch.cat(
                [token_type_ids, token_type_ids.new_zeros(completion_ids.shape)], dim=1
            )

        with torch.no_grad():
            # If the generation and optimization steps are misaligned—i.e., if generation does not occur at the end of
            # a full optimizer step (when gradient_accumulation_steps is not a multiple of generate_every)—then the
            # samples may come from an earlier version of the model. In that case, we need to track old_per_token_logps
            # for importance sampling. If the steps are aligned, importance sampling isn't necessary and we set
            # old_per_token_logps to None.
            # When using vLLM, we always compute old_per_token_logps for importance sampling, it was shown that the
            # distribution mismatch between vLLM and the training model can be large and harm the training.
            generate_every = self.args.steps_per_generation * self.num_iterations  # generation frequency
            if self.args.gradient_accumulation_steps % generate_every != 0 or (
                self.use_vllm and self.vllm_importance_sampling_correction
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                    num_images=num_images,
                    **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                )
            else:
                old_per_token_logps = None

            # Compute the importance sampling ratio when using vLLM, to correct for potential distribution mismatch
            if self.use_vllm and self.vllm_importance_sampling_correction:
                importance_sampling_ratio = torch.exp(old_per_token_logps - sampling_per_token_logps)
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio, max=self.vllm_importance_sampling_cap
                )

            # Compute the per-token log probabilities for the reference model
            if self.beta != 0.0:
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size=batch_size,
                        num_images=num_images,
                        **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                            self.model,
                            prompt_completion_ids,
                            attention_mask,
                            logits_to_keep,
                            batch_size=batch_size,
                            num_images=num_images,
                            **forward_kwargs,  # may contain pixel_values, image_grid_thw, pixel_attention_mask and image_sizes
                        )
            else:
                ref_per_token_logps = None

        # Decode
        prompts_text = self.processing_class.batch_decode(prompt_ids, skip_special_tokens=True)
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Check if prompts are conversational (check first prompt)
        if is_conversational({"prompt": prompts[0]}):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=True):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        # Convert inputs from Dict[str, List] to List[Dict] for reward function
        # Reward functions expect inputs as List[Dict]
        batch_size = len(prompts)
        inputs_list = []
        for i in range(batch_size):
            sample_dict = {}
            for key, values in inputs.items():
                if isinstance(values, list) and i < len(values):
                    sample_dict[key] = values[i]
                else:
                    sample_dict[key] = values
            inputs_list.append(sample_dict)
        
        # Merge extra_fields from rollout_func into inputs for reward functions
        if extra_fields:
            for i, inp in enumerate(inputs_list):
                for key, values in extra_fields.items():
                    if isinstance(values, list) and i < len(values):
                        inp[key] = values[i]
                    elif not isinstance(values, list):
                        inp[key] = values

        # Calculate rewards for each reward function. rewards_per_func aggregates rewards across all processes. This is
        # important because rewards will be normalized per group, and completions are distributed. We will later slice
        # rewards_per_func to extract each process's subset.
        rewards_per_func = self._calculate_rewards(inputs_list, prompts, completions, completion_ids_list)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards in ["group", "none"]:
            # If self.scale_rewards = "none", we'll still log group level std
            std_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            std_rewards = std_rewards.repeat_interleave(self.num_generations, dim=0)
        elif self.scale_rewards == "batch":
            # Compute global std
            std_rewards = rewards.std().expand_as(rewards)
        else:
            raise ValueError(
                f"Invalid value for scale_rewards: {self.scale_rewards}. Must be one of 'batch', 'group', or 'none'."
            )

        is_std_zero = torch.isclose(std_rewards, torch.zeros_like(std_rewards))
        if self.scale_rewards != "none":
            advantages = advantages / (std_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        all_process_advantages = advantages.clone()  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_func_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_func_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(is_std_zero.float().mean().item())

        # Log prompt and completion texts
        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())
        self._logs["advantages"].extend(all_process_advantages.tolist())

        if images is not None:
            self._logs["images"].extend(gather_object(images))

        if self.use_vllm and self.vllm_importance_sampling_correction:
            delta = torch.abs(old_per_token_logps - sampling_per_token_logps)
            delta = delta[completion_mask.bool()]
            mean_delta = torch.mean(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            max_delta = torch.max(delta) if delta.numel() > 0 else torch.tensor(0.0, device=device)
            self._metrics[mode]["sampling/sampling_logp_difference/mean"].append(
                self.accelerator.gather(mean_delta).mean().item()
            )
            self._metrics[mode]["sampling/sampling_logp_difference/max"].append(
                self.accelerator.gather(max_delta).max().item()
            )

            flat_is_ratio = importance_sampling_ratio[completion_mask.bool()]
            min_importance_sampling_ratio = (
                torch.min(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            mean_importance_sampling_ratio = (
                torch.mean(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            max_importance_sampling_ratio = (
                torch.max(flat_is_ratio) if flat_is_ratio.numel() > 0 else torch.tensor(0.0, device=device)
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/min"].append(
                nanmin(self.accelerator.gather(min_importance_sampling_ratio)).item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/mean"].append(
                self.accelerator.gather(mean_importance_sampling_ratio).nanmean().item()
            )
            self._metrics[mode]["sampling/importance_sampling_ratio/max"].append(
                nanmax(self.accelerator.gather(max_importance_sampling_ratio)).item()
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "num_items_in_batch": num_items_in_batch,
        }
        if old_per_token_logps is not None:
            output["old_per_token_logps"] = old_per_token_logps
        if self.use_vllm and self.vllm_importance_sampling_correction:
            output["importance_sampling_ratio"] = importance_sampling_ratio
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        if "pixel_values" in forward_kwargs:
            output["pixel_values"] = forward_kwargs["pixel_values"]
        if "image_grid_thw" in forward_kwargs:
            output["image_grid_thw"] = forward_kwargs["image_grid_thw"]
        if "pixel_attention_mask" in forward_kwargs:
            output["pixel_attention_mask"] = forward_kwargs["pixel_attention_mask"]
        if "image_sizes" in forward_kwargs:
            output["image_sizes"] = forward_kwargs["image_sizes"]

        if "pixel_values_videos" in forward_kwargs:
            output["pixel_values_videos"] = forward_kwargs["pixel_values_videos"]
        if "video_grid_thw" in forward_kwargs:
            output["video_grid_thw"] = forward_kwargs["video_grid_thw"]
        if "second_per_grid_ts" in forward_kwargs:
            output["second_per_grid_ts"] = forward_kwargs["second_per_grid_ts"]

        if "token_type_ids" in forward_kwargs:
            output["token_type_ids"] = forward_kwargs["token_type_ids"]
        if images is not None:
            output["num_images"] = num_images
        return output
    
    @profiling_decorator
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
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ) -> dict[str, torch.Tensor | None]:
        """Compute log-probs and (optionally) entropies for each token."""
        batch_size = batch_size or input_ids.size(0)  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        all_entropies = []
        for start in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[start : start + batch_size]
            attention_mask_batch = attention_mask[start : start + batch_size]

            # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
            model_inputs = {"input_ids": input_ids_batch, "attention_mask": attention_mask_batch}
            if image_grid_thw is not None and pixel_values is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                cum_rows = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                row_start, row_end = cum_rows[start].item(), cum_rows[start + batch_size].item()
                model_inputs["pixel_values"] = pixel_values[row_start:row_end]
                cum_imgs = torch.tensor([0] + num_images).cumsum(0)
                img_start, img_end = cum_imgs[start], cum_imgs[start + batch_size]
                model_inputs["image_grid_thw"] = image_grid_thw[img_start:img_end]
            elif pixel_values is not None:
                model_inputs["pixel_values"] = pixel_values[start : start + batch_size]

            if pixel_values_videos is not None:
                model_inputs["pixel_values_videos"] = pixel_values_videos[start : start + batch_size]
            if video_grid_thw is not None:
                model_inputs["video_grid_thw"] = video_grid_thw[start : start + batch_size]
            if second_per_grid_ts is not None:
                model_inputs["second_per_grid_ts"] = second_per_grid_ts[start : start + batch_size]

            if pixel_attention_mask is not None:
                model_inputs["pixel_attention_mask"] = pixel_attention_mask[start : start + batch_size]
            if image_sizes is not None:
                model_inputs["image_sizes"] = image_sizes[start : start + batch_size]
            if token_type_ids is not None:
                model_inputs["token_type_ids"] = token_type_ids[start : start + batch_size]

            # Only add logits_to_keep if the model supports it
            if "logits_to_keep" in self.model_kwarg_keys:
                # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
                model_inputs["logits_to_keep"] = logits_to_keep + 1

            model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

            logits = model(**model_inputs).logits
            # Exclude the last value: it corresponds to the next token pred
            logits = logits[:, :-1, :]  # (B, L-1, H)
            # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
            logits = logits[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            completion_ids = input_ids_batch[:, -logits_to_keep:]
            logps = selective_log_softmax(logits, completion_ids)  # compute logprobs
            all_logps.append(logps)

            if compute_entropy:
                with torch.no_grad():
                    entropies = entropy_from_logits(logits)
                all_entropies.append(entropies)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies
    
    @profiling_decorator
    def _get_last_hidden_state(
        self,
        unwrapped_model,
        input_ids,
        attention_mask,
        logits_to_keep,
        pixel_values=None,
        image_grid_thw=None,
        pixel_attention_mask=None,
        image_sizes=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
    ):
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model

        # Build model inputs - check if the model supports logits_to_keep (some models and VLMs don't)
        model_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

        # For Qwen models:
        if image_grid_thw is not None and pixel_values is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        # For Gemma, SmolVLM2, LLaVa-Next etc.:
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values

        if video_grid_thw is not None and pixel_values_videos is not None:
            model_inputs["video_grid_thw"] = video_grid_thw
            model_inputs["pixel_values_videos"] = pixel_values_videos
        if second_per_grid_ts is not None:
            model_inputs["second_per_grid_ts"] = second_per_grid_ts

        # For SmolVLM2
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        # For LLaVa-Next
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes

        # Only add logits_to_keep if the model supports it
        if "logits_to_keep" in self.model_kwarg_keys:
            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            model_inputs["logits_to_keep"] = logits_to_keep + 1

        model_inputs["use_cache"] = False  # only used in generation; set False to suppress warnings

        last_hidden_state = unwrapped_model.model(**model_inputs).last_hidden_state
        # Exclude the last value: it corresponds to the next token pred
        last_hidden_state = last_hidden_state[:, :-1, :]  # (B, L-1, H)
        # Only keep the last logits_to_keep. For model that support logits_to_keep, this is a no-op.
        last_hidden_state = last_hidden_state[:, -logits_to_keep:, :]  # (B, logits_to_keep, H)
        return last_hidden_state
    
    def compute_liger_loss(self, unwrapped_model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Get the last hidden state of the model
        last_hidden_state = self._get_last_hidden_state(
            unwrapped_model,
            input_ids,
            attention_mask,
            logits_to_keep,
            inputs.get("pixel_values"),
            inputs.get("image_grid_thw"),
            inputs.get("pixel_attention_mask"),
            inputs.get("image_sizes"),
            inputs.get("pixel_values_videos"),
            inputs.get("video_grid_thw"),
            inputs.get("second_per_grid_ts"),
        )

        # compute loss and metrics using liger grpo loss
        loss, metrics = self.liger_grpo_loss(
            _input=last_hidden_state,
            lin_weight=unwrapped_model.lm_head.weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=inputs["advantages"],
            bias=unwrapped_model.lm_head.bias,
            old_per_token_logps=inputs.get("old_per_token_logps"),
            ref_per_token_logps=inputs.get("ref_per_token_logps"),
        )
        # Extract metrics from the liger_grpo_loss output
        # KL divergence is the first metric when beta is non-zero
        mean_kl = metrics[0] if self.beta != 0.0 else None
        clip_ratio = metrics[-1]

        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).mean().item())
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).mean().item())
        return loss / self.current_gradient_accumulation_steps

    
    def _compute_loss(self, model, inputs):
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        # Compute the per_token_logps and the entropy at each position in the completion
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
            pixel_values_videos=inputs.get("pixel_values_videos"),
            video_grid_thw=inputs.get("video_grid_thw"),
            second_per_grid_ts=inputs.get("second_per_grid_ts"),
        )

        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps,
        # old_per_token_logps == per_token_logps. In this case we can skip its computation
        # (see _generate_and_score_completions) and instead use per_token_logps.detach().
        # The exception is when using vLLM, where we always compute old_per_token_logps
        # for importance sampling
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        if self.importance_sampling_level == "token":
            log_importance_weights = log_ratio
        elif self.importance_sampling_level == "sequence":
            log_importance_weights = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
            log_importance_weights = log_importance_weights.unsqueeze(-1)
        else:
            raise ValueError(
                f"Unknown importance sampling level: {self.importance_sampling_level}. Possible values are 'token' "
                "and 'sequence'."
            )
        # From here, log_importance_weights (and all subsequent tensors, coef_1, coef_2, etc.) shape depends on
        # importance_sampling_level: "token" level: (B, T); "sequence" level: (B, 1)

        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        # Two-sided clipping
        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask

        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_batch_mean(x):
            if x.shape[1] == 1:  # when importance_sampling_level == "sequence"
                return x.mean()
            else:
                return (x * completion_mask).sum() / completion_token_count

        if self.beta != 0.0:
            mean_kl = masked_batch_mean(per_token_kl)
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        mean_entropy = masked_batch_mean(entropies)
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages.unsqueeze(1) > 0)
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = masked_batch_mean(is_low_clipped.float())
        high_clip = masked_batch_mean(is_high_clipped.float())
        clip_ratio = masked_batch_mean(is_region_clipped.float())

        gathered_low_clip = self.accelerator.gather(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(gathered_low_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/low_min"].append(nanmin(gathered_low_clip).item())
        gathered_high_clip = self.accelerator.gather(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(gathered_high_clip.nanmean().item())
        self._metrics[mode]["clip_ratio/high_max"].append(nanmax(gathered_high_clip).item())
        gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(gathered_clip_ratio.nanmean().item())
        return loss

    def create_optimizer(self):
        """
        Setup the optimizer.
        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            lr_mapper = {}
            visual_parameters = []
            merger_parameters = []

            if self.args.vision_lr is not None:
                lr_mapper["visual"] = self.args.vision_lr
                visual_parameters = [name for name, _ in opt_model.named_parameters() if "visual" in name and "merger" not in name]
            if self.args.merger_lr is not None:
                lr_mapper["merger"] = self.args.merger_lr
                merger_parameters = [name for name, _ in opt_model.named_parameters() if "merger" in name]

            if len(lr_mapper) > 0:
                special_lr_parameters = merger_parameters + visual_parameters
                
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in special_lr_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
                
                if visual_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.vision_lr,
                                "param_group_name": "visaul_decay"
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in visual_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.vision_lr,
                                "param_group_name": "visaul_non_decay"
                            },
                        ]
                    )
                
                if merger_parameters: 
                    optimizer_grouped_parameters.extend(
                        [
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": self.args.weight_decay,
                                "lr": self.args.merger_lr,
                                "param_group_name": "merger_decay",
                            },
                            {
                                "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in merger_parameters and p.requires_grad)],
                                "weight_decay": 0.0,
                                "lr": self.args.merger_lr,
                                "param_group_name": "merger_non_decay",
                            },
                        ]
                    )
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    },
                ]
            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial):
        if self.args.hub_model_id is None:
            model_name = Path(self.args.output_dir).name
        else:
            model_name = self.args.hub_model_id.split("/")[-1]
        self.create_model_card(model_name=model_name)
        
        if self.args.lora_enable:
            # Only rank that should save does the filesystem work
            if not self.args.should_save:
                return

            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            os.makedirs(output_dir, exist_ok=True)
            self.save_model(output_dir, _internal_call=True)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=False)
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH] and self.state.best_global_step:
                best_checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.best_global_step}"
                best_checkpoint_dir = os.path.join(run_dir, best_checkpoint_folder)

                if os.path.exists(best_checkpoint_dir):
                    self.state.best_model_checkpoint = best_checkpoint_dir

            if not self.args.save_only_model:
                # Save optimizer and scheduler
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                # Save RNG state
                self._save_rng_state(output_dir)

            # Save the Trainer state
            if self.args.should_save:
                # Update `ExportableState` callbacks and `TrainerControl` state to where we are currently
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
                self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

        else:
            super(QwenGRPOTrainer, self)._save_checkpoint(model, trial)