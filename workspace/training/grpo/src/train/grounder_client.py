"""Synchronous grounder client used by GRPO reward functions."""

from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import List, Optional, Union

from PIL import Image

from src.constants import DEFAULT_GROUNDER_URL, GROUNDER_PROMPT_TEMPLATE

DEFAULT_GROUNDER_TIMEOUT = 30.0


def _normalize_base_url(url: str) -> str:
    """Normalize endpoint/path to OpenAI client base_url."""
    u = (url or "").strip().rstrip("/")
    if not u:
        return DEFAULT_GROUNDER_URL

    for suffix in ("/chat/completions", "/completions"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
            break
    return u


def _image_to_b64(image: Union[Image.Image, str, Path]) -> Optional[str]:
    try:
        if isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            img = Image.open(str(image)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        import base64

        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


def _clean_model_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    return text.strip()


def _extract_json_text(text: str) -> str:
    cleaned = _clean_model_text(text)
    block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    if block:
        return block.group(1).strip()

    # Extract first complete JSON object.
    start = cleaned.find("{")
    if start < 0:
        return cleaned

    depth = 0
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1].strip()
    return cleaned[start:].strip()


def _parse_bbox_from_response(text: str) -> Optional[List[int]]:
    try:
        obj = json.loads(_extract_json_text(text))
    except Exception:
        return None

    bbox = obj.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None

    try:
        return [int(float(v)) for v in bbox]
    except Exception:
        return None


def call_grounder(
    image: Union[Image.Image, str, Path],
    reasoning: str,
    target_element: str,
    url: str = DEFAULT_GROUNDER_URL,
    timeout: float = DEFAULT_GROUNDER_TIMEOUT,
) -> Optional[List[int]]:
    """
    Call grounder service and return bbox [x1, y1, x2, y2].

    Returns None on any failure (HTTP/timeout/parse/etc.).
    """
    if not target_element or not str(target_element).strip():
        return None

    image_b64 = _image_to_b64(image)
    if not image_b64:
        return None

    try:
        from openai import OpenAI
    except Exception:
        return None

    prompt = GROUNDER_PROMPT_TEMPLATE.format(
        reasoning=(reasoning or "").strip(),
        description=target_element.strip(),
    )

    try:
        client = OpenAI(api_key="0", base_url=_normalize_base_url(url), timeout=float(timeout))
        completion = client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0,
        )
        message = completion.choices[0].message.content
        text = message if isinstance(message, str) else str(message)
        return _parse_bbox_from_response(text)
    except Exception:
        return None
