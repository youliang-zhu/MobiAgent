import re
import torch

from qwen_vl_utils import process_vision_info

from src.constants import (
    DEFAULT_IMAGE_TOKEN,
    LLAVA_IMAGE_TOKEN,
    VISION_START_TOKEN,
    VISION_END_TOKEN,
)


def replace_image_tokens(input_string):
    """
    Replace <image> token with Qwen2.5-VL format: <|vision_start|><|image_pad|><|vision_end|>
    """
    pattern = r'\n?' + re.escape(LLAVA_IMAGE_TOKEN) + r'\n?'
    replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output


def get_image_info(image_path, min_pixel, max_pixel, width, height, image_patch_size):
    """
    Process image using qwen_vl_utils.
    """
    content = {
        "type": "image", 
        "image": image_path,
        "min_pixels": min_pixel,
        "max_pixels": max_pixel
    }

    if width is not None and height is not None:
        content["resized_width"] = width
        content["resized_height"] = height
    
    messages = [
        {
            "role": "user", 
            "content": [content]
        }
    ]

    image_input, _ = process_vision_info(messages, image_patch_size=image_patch_size)

    return image_input[0]
