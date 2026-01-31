"""Grounder HTTP Client for GRPO training reward computation"""

import asyncio
import aiohttp
import base64
import json
import logging
from typing import List, Dict, Optional, Tuple
from PIL import Image
import io

logger = logging.getLogger(__name__)

GROUNDER_PROMPT_TEMPLATE = '''Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''


class GrounderClient:
    """Async HTTP client for Grounder service"""
    
    def __init__(self, base_url: str = "http://localhost:8001/v1/chat/completions", timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        
    def _image_to_base64(self, image_input) -> str:
        """Convert image input to base64 string"""
        if isinstance(image_input, str):
            # 如果是文件路径
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, Image.Image):
            # 如果是 PIL Image
            buffered = io.BytesIO()
            image_input.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        elif isinstance(image_input, dict) and "image" in image_input:
            # 如果是 dataset 返回的格式 {"image": PIL.Image}
            return self._image_to_base64(image_input["image"])
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    async def predict_bbox_async(
        self, 
        session: aiohttp.ClientSession,
        image_input,
        reasoning: str,
        description: str
    ) -> Optional[List[int]]:
        """Async call to Grounder service"""
        try:
            image_base64 = self._image_to_base64(image_input)
            prompt = GROUNDER_PROMPT_TEMPLATE.format(reasoning=reasoning, description=description)
            
            payload = {
                "model": "",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                            {"type": "text", "text": prompt},
                        ]
                    }
                ],
                "temperature": 0
            }
            
            async with session.post(self.base_url, json=payload) as response:
                if response.status != 200:
                    logger.warning(f"Grounder returned status {response.status}")
                    return None
                    
                result = await response.json()
                content = result["choices"][0]["message"]["content"]
                
                # 解析 bbox
                # 处理可能的 markdown 包裹
                if "```json" in content:
                    import re
                    match = re.search(r"```json\n(.*?)\n```", content, re.DOTALL)
                    if match:
                        content = match.group(1)
                
                parsed = json.loads(content)
                bbox = parsed.get("bbox")
                if bbox and len(bbox) == 4:
                    return [int(x) for x in bbox]
                return None
                
        except Exception as e:
            logger.warning(f"Grounder call failed: {e}")
            return None
    
    async def batch_predict_bbox_async(
        self,
        requests: List[Tuple]  # List of (image_input, reasoning, description)
    ) -> List[Optional[List[int]]]:
        """Batch async calls to Grounder service"""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            tasks = [
                self.predict_bbox_async(session, img, reasoning, desc)
                for img, reasoning, desc in requests
            ]
            return await asyncio.gather(*tasks)
    
    def batch_predict_bbox(
        self,
        requests: List[Tuple]
    ) -> List[Optional[List[int]]]:
        """Synchronous wrapper for batch prediction"""
        return asyncio.run(self.batch_predict_bbox_async(requests))
    
    def predict_bbox(
        self,
        image_input,
        reasoning: str,
        description: str
    ) -> Optional[List[int]]:
        """Synchronous single prediction"""
        results = self.batch_predict_bbox([(image_input, reasoning, description)])
        return results[0] if results else None


# Global client instance (initialized lazily)
_grounder_client: Optional[GrounderClient] = None


def get_grounder_client(base_url: str = "http://localhost:8001/v1/chat/completions") -> GrounderClient:
    """Get or create global Grounder client"""
    global _grounder_client
    if _grounder_client is None:
        _grounder_client = GrounderClient(base_url=base_url)
    return _grounder_client


def set_grounder_url(base_url: str):
    """Set Grounder service URL"""
    global _grounder_client
    _grounder_client = GrounderClient(base_url=base_url)
