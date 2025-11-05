import os
import time
import base64
from PIL import Image, ImageDraw
import json
import cv2
import argparse
from openai import OpenAI
import io
import datetime
import zoneinfo

from .mobiagent import AndroidDevice

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

    reasoning = "我已经进入到了uber eat主界面，现在我需要点击“寿司”图标按钮进入到寿司相关的外卖菜单。"
    target_element = "寿司图标按钮"

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

    # 解析bbox
    grounder_response = json.loads(grounder_response_str)
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