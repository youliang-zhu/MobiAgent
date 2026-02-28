"""
Gemini API 适配器 - 将 Gemini API 包装成 OpenAI 兼容接口
支持纯文本和多模态（文本+图片）输入
"""

import base64
from typing import List, Dict, Union, Any
from google.genai import types


class GeminiAdapter:
    """将 Gemini API 包装成 OpenAI 兼容接口"""
    
    def __init__(self, api_key: str, **kwargs):
        """
        初始化 Gemini 客户端
        
        Args:
            api_key: Gemini API 密钥
            **kwargs: 兼容参数（如 base_url），会被忽略
        """
        from google import genai
        self.client = genai.Client(api_key=api_key)
    
    @property
    def chat(self):
        """模拟 OpenAI 的 client.chat 属性"""
        return self
    
    @property
    def completions(self):
        """模拟 OpenAI 的 client.chat.completions 属性"""
        return self
    
    def create(self, model: str, messages: List[Dict[str, Any]], **kwargs) -> 'MockResponse':
        """
        模拟 OpenAI 的 chat.completions.create 方法
        
        Args:
            model: 模型名称（如 "gemini-2.5-flash"）
            messages: OpenAI 格式的消息列表
            **kwargs: 其他参数（兼容性，会被忽略）
        
        Returns:
            MockResponse 对象，兼容 OpenAI 响应格式
        """
        # 转换消息格式：OpenAI → Gemini
        contents = self._convert_messages(messages)
        
        # 调用 Gemini API
        response = self.client.models.generate_content(
            model=model,
            contents=contents
        )
        
        # 包装响应为 OpenAI 格式
        return MockResponse(response.text)
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> Union[str, List]:
        """
        转换消息格式：OpenAI → Gemini
        
        Args:
            messages: OpenAI 格式的消息列表
                [{"role": "user", "content": "text" 或 [{"type": "text", ...}, {"type": "image_url", ...}]}]
        
        Returns:
            Gemini 格式的内容：字符串（纯文本）或列表（多模态）
        """
        if not messages or len(messages) == 0:
            return ""
        
        # 提取用户消息内容
        user_message = messages[0]
        content = user_message.get("content", "")
        
        # 纯文本消息
        if isinstance(content, str):
            return content
        
        # 多模态消息（文本 + 图片）
        if isinstance(content, list):
            parts = []
            for item in content:
                item_type = item.get("type", "")
                
                if item_type == "text":
                    # 文本部分
                    parts.append(item.get("text", ""))
                
                elif item_type == "image_url":
                    # 图片部分
                    image_url = item.get("image_url", {})
                    url = image_url.get("url", "")
                    
                    # 解析 base64 图片：data:image/jpeg;base64,/9j/4AAQ...
                    if url.startswith("data:"):
                        try:
                            # 提取 MIME 类型和 base64 数据
                            header, base64_data = url.split(",", 1)
                            mime_type = header.split(";")[0].split(":")[1]  # 提取 "image/jpeg"
                            
                            # 解码 base64
                            image_bytes = base64.b64decode(base64_data)
                            
                            # 使用 Gemini 的 Part 对象
                            part = types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            )
                            parts.append(part)
                        except Exception as e:
                            print(f"警告：解析图片 URL 失败: {e}")
                            continue
            
            return parts
        
        # 其他情况返回空字符串
        return ""


class MockResponse:
    """模拟 OpenAI 响应结构"""
    
    def __init__(self, text: str):
        self.choices = [MockChoice(text)]


class MockChoice:
    """模拟 OpenAI 的 Choice 对象"""
    
    def __init__(self, text: str):
        self.message = MockMessage(text)


class MockMessage:
    """模拟 OpenAI 的 Message 对象"""
    
    def __init__(self, text: str):
        self.content = text
