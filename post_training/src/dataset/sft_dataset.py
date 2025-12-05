import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    IGNORE_INDEX,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IMAGE_TOKEN,
)

from .data_utils import get_image_info, replace_image_tokens, pad_sequence


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with MobiAgent data format."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id,
        padding=True,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height

        # Qwen2.5-VL uses 14x14 patches
        self.image_patch_size = 14

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Process a single sample.
        
        Expected data format:
        {
            "instruction": "<image>\n...(prompt with <image> token)...",
            "output": "...(model response)...",
            "images": ["absolute/path/to/image.jpg"],
            "input": ""  # unused
        }
        """
        sample = self.list_data_dict[i]
        processor = self.processor

        # === Process images ===
        image_files = sample["images"]
        if isinstance(image_files, str):
            image_files = [image_files]

        images = []
        for image_file in image_files:
            # Require absolute path to exist
            if not os.path.exists(image_file):
                raise FileNotFoundError(f"Image file not found: {image_file}")
            
            image_input = get_image_info(
                image_file,
                self.image_min_pixel,
                self.image_max_pixel,
                self.image_resized_w,
                self.image_resized_h,
                self.image_patch_size
            )
            images.append(image_input)

        # === Process instruction (user input) ===
        # Replace <image> with <|vision_start|><|image_pad|><|vision_end|>
        instruction = replace_image_tokens(sample["instruction"])
        
        # Format as Qwen chat format
        # <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n
        user_input = f"{DEFAULT_IM_START_TOKEN}user\n{instruction}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
        
        # === Process output (model response) ===
        output = sample["output"]
        gpt_response = f"{output}{DEFAULT_IM_END_TOKEN}\n"

        # === Tokenization ===
        # Process instruction with images
        if DEFAULT_IMAGE_TOKEN in user_input:
            inputs = processor(
                text=[user_input],
                images=images,
                padding=False,
                do_resize=False,
                return_tensors='pt'
            )
            prompt_input_ids = inputs['input_ids']
            pixel_values = inputs['pixel_values']
            image_grid_thw = inputs['image_grid_thw']
        else:
            prompt_input_ids = processor.tokenizer(
                user_input,
                add_special_tokens=False,
                padding=False,
                return_tensors='pt'
            )['input_ids']
            pixel_values = None
            image_grid_thw = None

        # Tokenize response
        response_input_ids = processor.tokenizer(
            gpt_response,
            add_special_tokens=False,
            padding=False,
            return_tensors='pt'
        )['input_ids']

        # === Concatenate and create labels ===
        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        
        # Labels: IGNORE_INDEX for instruction, actual token ids for output
        labels = torch.cat(
            [
                torch.full((len(prompt_input_ids[0]),), IGNORE_INDEX, dtype=torch.long),
                response_input_ids.squeeze(0),
            ],
            dim=0,
        )

        # Attention mask: all 1s
        attention_mask = torch.ones_like(input_ids)

        data_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_grid_thw

        return data_dict


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_thw = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            
            if "pixel_values" in example:
                batch_pixel_values.append(example["pixel_values"])
                batch_image_thw.append(example["image_grid_thw"])

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )

        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)

        data_dict = {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }

        if len(batch_pixel_values) > 0:
            pixel_values = torch.cat(batch_pixel_values, dim=0)
            image_thw = torch.cat(batch_image_thw, dim=0)
            data_dict["pixel_values"] = pixel_values
            data_dict["image_grid_thw"] = image_thw

        return data_dict


def make_supervised_data_module(model_id, processor, data_args):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id
    )
    
    eval_dataset = None
    if data_args.eval_path is not None:
        eval_dataset = SupervisedDataset(
            data_path=data_args.eval_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id
        )

    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id
    )

    return dict(
        train_dataset=sft_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
