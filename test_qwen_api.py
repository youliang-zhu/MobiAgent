# import os
# from openai import OpenAI


# client = OpenAI(
#     # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
#     api_key="sk-4201f908ffb241d0b4f2eaaf81048add",
#     base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
# )

# completion = client.chat.completions.create(
#     # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
#     model="Qwen-VL",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "你好！请用中文介绍一下自己。"},
#     ],
#     # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
#     # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
#     # extra_body={"enable_thinking": False},
# )
# print(completion.model_dump_json())



from openai import OpenAI
import base64

client = OpenAI(
    api_key="sk-4201f908ffb241d0b4f2eaaf81048add",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)

with open("collect/manual/data/淘宝/点击搜索栏/1/1.jpg", "rb") as f:
    img_base64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="qwen-vl-plus",  # 或 qwen-vl-plus
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}
                },
                {
                    "type": "text",
                    "text": "描述这张图片"
                }
            ]
        }
    ]
)

print(response.choices[0].message.content)