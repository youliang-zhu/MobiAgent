"""
workspace/utils/test_grounder.py - Grounder 服务单步调试工具

功能：
    连接真实 Android 设备，抓取当前屏幕，将截图和一个固定的查找目标 Prompt 发送给
    Grounder 服务，返回元素的 bounding box 坐标并实际点击该位置。
    全部过程的截图、prompt/response、可视化结果保存在 workspace/utils/test_records/<timestamp>/。

使用先决条件：
    1. Android 设备已由 adb 连接
    2. Grounder 服务已在目标端口启动

使用方法：
    python -m workspace.utils.test_grounder --service_ip <IP> --grounder_port <PORT>

示例：
    python -m workspace.utils.test_grounder --service_ip localhost --grounder_port 8001
"""

import os
import time
import base64
from PIL import Image, ImageDraw
import json
import cv2
import argparse
import re
from openai import OpenAI
import io
import datetime
import zoneinfo

from ...runner.mobiagent.mobiagent import AndroidDevice

def get_base64_screenshot(img_path, factor=0.5):
    img = Image.open(img_path)
    img = img.resize((int(img.width * factor), int(img.height * factor)), Image.Resampling.LANCZOS)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    screenshot = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return screenshot

def save_visualizations(img_path, bbox, point, save_dir):
    # 保存原始截图
    img = Image.open(img_path)
    img.save(os.path.join(save_dir, "screenshot.png"))

    # 画bbox
    img_bbox = img.copy()
    draw = ImageDraw.Draw(img_bbox)
    x1, y1, x2, y2 = bbox
    draw.rectangle([x1, y1, x2, y2], outline='red', width=5)
    img_bbox.save(os.path.join(save_dir, "result_bbox.png"))

    # 画点击点
    cv_img = cv2.imread(os.path.join(save_dir, "result_bbox.png"))
    if cv_img is not None:
        cv2.circle(cv_img, point, 15, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(save_dir, "result_point.png"), cv_img)

def main():

    device = AndroidDevice()
    parser = argparse.ArgumentParser()
    parser.add_argument("--service_ip", type=str, default="localhost")
    parser.add_argument("--grounder_port", type=int, default=8001)
    args = parser.parse_args()

    grounder_client = OpenAI(
        api_key = "0",
        base_url = f"http://{args.service_ip}:{args.grounder_port}/v1",
    )

    # 读取截图并转base64
    device.screenshot("tmp_screenshot.jpg")
    screenshot = get_base64_screenshot("tmp_screenshot.jpg")

    reasoning = "我已经进入到了淘宝主界面，现在我需要点击“搜索框”图标按钮。"
    target_element = "上方的搜索框图标按钮"

    grounder_prompt_template_bbox = '''
    Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
    User's intent: {reasoning}
    Target element's description: {description}
    Your output should be a JSON object with the following format:
    {{"bbox": [x1, y1, x2, y2]}}'''
    grounder_prompt = grounder_prompt_template_bbox.format(reasoning=reasoning, description=target_element)

    # 创建测试记录文件夹
    swiss_tz = zoneinfo.ZoneInfo("Europe/Zurich")
    ts = datetime.datetime.now(swiss_tz).strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(os.path.dirname(__file__), "test_records", ts)

    os.makedirs(save_dir, exist_ok=True)

    # 调用grounder
    grounder_response_str = grounder_client.chat.completions.create(
        model="",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
                    {"type": "text", "text": grounder_prompt},
                ]
            }
        ],
        temperature=0
    ).choices[0].message.content

    print("Grounder response:", grounder_response_str)
    with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
        f.write("PROMPT:\n" + grounder_prompt + "\n\nRESPONSE:\n" + grounder_response_str)

    # 处理 thinking 标签和 Markdown 格式（与 mobiagent.py 保持一致）
    # 移除 thinking 标签和内容
    grounder_response_str = re.sub(r'<think>.*?</think>', '', grounder_response_str, flags=re.DOTALL)
    if '</think>' in grounder_response_str:
        grounder_response_str = grounder_response_str.split('</think>', 1)[-1]
    grounder_response_str = grounder_response_str.strip()
    
    # 提取 Markdown 代码块中的 JSON
    pattern = re.compile(r"```json\n(.*?)\n```", re.DOTALL)
    match = pattern.search(grounder_response_str)
    if match:
        json_str = match.group(1)
    else:
        json_str = grounder_response_str
    
    # 解析bbox
    grounder_response = json.loads(json_str)
    factor = 0.5
    bbox = [int(coord / factor) for coord in grounder_response["bbox"]]  # 除以 factor 还原
    x1, y1, x2, y2 = bbox
    position_x = (x1 + x2) // 2
    position_y = (y1 + y2) // 2

    device.click(position_x, position_y)

    # 保存可视化图片
    save_visualizations("tmp_screenshot.jpg", bbox, (position_x, position_y), save_dir)

if __name__ == "__main__":
    main()
    # 代码调用方法，首先在8000端口部署grounder，然后cmd执行： 
    # python -m runner.mobiagent.test_grounder --service_ip localhost --grounder_port 8001