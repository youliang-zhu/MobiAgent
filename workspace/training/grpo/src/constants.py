"""Shared constants for GRPO training."""

GROUNDER_PROMPT_TEMPLATE = (
    "Based on the screenshot, user's intent and the description of the target UI element, "
    "provide the bounding box of the element using **absolute coordinates**.\n"
    "User's intent: {reasoning}\n"
    "Target element's description: {description}\n"
    "Your output should be a JSON object with the following format:\n"
    '{{"bbox": [x1, y1, x2, y2]}}'
)

# Keep this value aligned with grpo_plan.md. grounder_client will normalize it to OpenAI base_url.
DEFAULT_GROUNDER_URL = "http://localhost:8001/v1/chat/completions"

# LoRA target suffixes for Qwen2.5-VL LLM blocks.
DEFAULT_LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
