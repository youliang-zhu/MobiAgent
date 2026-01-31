import os
from typing import Dict
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset

from src.params import DataArguments
from src.constants import (
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    SYSTEM_MESSAGE,
)

from .data_utils import get_image_info


class GRPODataset(Dataset):
    """Dataset for GRPO training with instruction + images format"""

    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        model_id: str,
    ):
        super().__init__()
        if isinstance(data_path, str):
            with open(data_path, "r", encoding="utf-8") as f:
                self.list_data_dict = json.load(f)
        else:
            self.list_data_dict = data_path

        self.model_id = model_id
        self.processor = processor
        self.data_args = data_args
        
        self.image_min_pixel = data_args.image_min_pixels
        self.image_max_pixel = data_args.image_max_pixels
        self.image_resized_w = data_args.image_resized_width
        self.image_resized_h = data_args.image_resized_height

        self.image_patch_size = 16 if "Qwen3" in model_id else 14
        self.processor.image_processor.do_resize = False

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]

        # 处理图片
        image_files = sources.get("images", [])
        images = []
        for image_file in image_files:
            if not os.path.exists(image_file):
                image_folder = self.data_args.image_folder
                if image_folder and not image_file.startswith("http"):
                    image_file = os.path.join(image_folder, image_file)
            image_input = get_image_info(
                image_file,
                self.image_min_pixel,
                self.image_max_pixel,
                self.image_resized_w,
                self.image_resized_h,
                self.image_patch_size
            )
            images.append(image_input)

        # 构建 OpenAI 标准的 conversational format
        # 原始 instruction 格式: "<image>\nYou are a phone-use AI agent..."
        instruction = sources["instruction"]
        
        # 移除 <image> 占位符（processor 会自动处理）
        instruction_text = instruction.replace("<image>\n", "").replace("<image>", "")
        
        # 构建 messages（OpenAI 标准格式）
        user_content = []
        
        # 如果有图像，添加图像对象到 content
        # Qwen2.5-VL 要求图像对象直接嵌入在 content 中
        if images:
            for img in images:
                user_content.append({"type": "image", "image": img})
        
        # 添加文本内容
        user_content.append({"type": "text", "text": instruction_text})
        
        # 构建完整的 messages - 不使用 system role 避免中间 EOS token
        messages = [
            {"role": "user", "content": user_content}
        ]
        
        # DEBUG: Log message structure for first few samples
        import logging
        logger = logging.getLogger(__name__)
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 1
        
        if self._debug_count <= 3:
            logger.info(f"[Dataset Debug {self._debug_count}] Message roles: {[m['role'] for m in messages]}")
            logger.info(f"[Dataset Debug {self._debug_count}] Has system role: {any(m['role'] == 'system' for m in messages)}")

        # gt_action 用于 reward 计算
        # 包装在列表中，以便 TRL 的 shuffle_sequence_dict 可以正常处理
        gt_action = sources.get("gt_action", {})

        return dict(
            prompt=messages,  # 返回 conversational format（list of dicts）
            images=images if images else None,
            gt_action=[gt_action],  # 包装在列表中
        )


def make_grpo_data_module(model_id, processor, data_args):
    """Make dataset for GRPO training."""
    train_dataset = GRPODataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        model_id=model_id
    )
    
    eval_dataset = None
    if data_args.eval_path:
        eval_dataset = GRPODataset(
            data_path=data_args.eval_path,
            processor=processor,
            data_args=data_args,
            model_id=model_id
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )