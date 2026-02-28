**研究资料**

**基础Language Modeling from Scratch课程**

[Stanford CS336 | Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)



**部署MLLM到手机端，实现终端（on-device）高效、可用的语言模型推理：**

1.降低 LLM 在资源受限设备（如智能手机）上的推理成本和延迟，利用专用硬件（如NPU）提升推理性能

**PowerInfer-2: Fast Large Language Model Inference on a Smartphone**
**Fast On-device LLM Inference with NPUs**

**PowerInfer-2** 提出了一个结合 CPU+GPU+NPU 的异构执行策略，充分发挥各类处理器的性能。

**Fast On-device LLM Inference with NPUs** 则聚焦在最大化 NPU 的使用效率，通过模型改造使其更适配于 NPU 的计算特点。



2.模型训练与设计导向：关注如何预训练出结构小、性能强、适合终端推理的小模型。

**PhoneLM: an Efficient and Capable Small Language Model Family through Principled Pre-training**



**千问模型研究，了解千问架构，方便未来优化自己的架构：**

1. qwen2.5技术报告：[2502.13923](https://arxiv.org/pdf/2502.13923)
2. huggingface transformer，qwen视觉语言模型（VLM），多模态源码实现https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/qwen2_5_vl.py



**diffusion model手机端的部署应用，纯探索阶段**

**openai api:** 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



**idea**

1. planner需要模板才能给出详细步骤，而有时候decider如果没有具体的步骤指导，只有一个user目标指令，不一定能做出正确的判断。能不能让planner对所有的用户输入都生成详细的执行步骤？初步想法是对planner进行微调或者直接做上下文工程。
1. planner只能打开planner.md里面定义的app，也就是能打开的app是被写死的。优化planner，实现根据用户输入的提示词自动判断要打开哪个app，通过adb连接判断有哪些可打开的软件，从系统层面解决这个问题。
1. pipline优化：planner规划，然后先用YOLO识别一遍屏幕可点击的地方，decider分析YOLO提供的可点击的所有地方的图标以及整体语义，并只能点击YOLO提供的位置。这样就可以直接省略掉grounder，YOLO提供坐标，decider决定点哪个。可以设计回退算法机制，直接用adb返回实现回退。
1. in the flow强化学习：现在decider输出的目标点击元素是唯一的，可以加一个机制：从可能性从高到低列举出可以点击的按钮位置，先尝试点击最高可能性的位置，如果成功，就对该条路径进行奖励。如果失效，就回退选择第二可能性按钮，再次尝试，如果正确就奖励，如果失效就返回任务失败。
1. pipline优化：当我想要让agent实现：搜索最新上市的平板电脑。但是淘宝没有最新平板的筛选功能，所以可以让planner规划阶段先调用外部搜索工具，搜索出3款最新的平板，让用户给出具体的选择，然后再到淘宝里面进行具体的搜索。所以agent效果不好其实很多时候是用户输入指令不清晰，如果让planner把用户的指令转化为清晰具体的动作。或者说如果当对应的app里面没有这样的接口，就可以尝试调用搜索工具进行回退再入，比如：搜索最轻便的折叠椅。淘宝没有重量这个选项，那么可以调用搜索工具查询到市面上最轻便的折叠椅可能是哪一款，然后再输入给agent搜索具体的这一款产品。或者说如果用户输入的是一种抽象的主观的标注，比如：最耐用的书包，最保暖的羽绒服，材质最好的床单，最静音的加湿器，最防滑的瑜伽垫。这些是不能量化的，可以直接让planner把这些抽象描述加入到搜索词，直接搜索，比如：书包--耐用的书包，羽绒服--保暖的羽绒服...
1. 回退机制的判定
1. 应该以图标ui和文字作为训练依据，因为人能知道怎么用手机是因为图标和文字的逻辑都是通的。



**mobiagent存在的问题**

1.因为数据标注react是用vlm推理做的，大批量标注很难保证全部都是对的，很多数据会发生一些最基本的格式错误，最常见的是target_element字段缺失。

2.如果一直卡在某一步，没有退出或者回退机制，会一直处在这个页面，模型陷入推理循环

```
你的操作历史如下：
1. {"reasoning": "首先需要点击进入淘宝直播页面，以便后续搜索自然堂并进入其直播间。截图中显示了淘宝直播的入口，因此需要点击该入口。",
"action": "click",
"parameters": {"target_element": "淘宝直播"}}
2. {"reasoning": "当前页面显示的是淘宝直播首页，需要在搜索框中输入“自然堂”以查找自然堂的官方直播间。搜索框位于页面顶部，因此需要点击搜索框并输入关键词。",
"action": "click",
"parameters": {"target_element": "搜索框"}}
3. {"reasoning": "当前页面显示的是淘宝直播的搜索界面，需要在搜索框中输入关键词“自然堂”。搜索框位于页面顶部，因此需要点击搜索框并输入关键词。",
"action": "input",
"parameters": {"text": "自然堂"}}
4. {"reasoning": "当前搜索结果已经显示了多个自然堂相关的直播间选项，下一步需要选择其中一个直播间进入。由于目标是进入自然堂的官方直播间，可以点击第一个选项来进入直播间。", "action": "click", "parameters": {"target_element": "自然堂旗舰店"}}
5. {"reasoning": "当前页面显示了自然堂旗舰店的直播间信息，包括关注按钮、充值购物金活动等。为了进入直播间，需要点击关注按钮，这样可以进入直播间观看直播内容。", "action": "click", "parameters": {"target_element": "关注"}}
6. {"reasoning": "当前页面显示了自然堂旗舰店的直播间信息，包括关注按钮、充值购物金活动等。为了进入直播间观看直播内容，需要点击关注按钮。", "action": "click", "parameters": {"target_element": "关注成功"}}
7. {"reasoning": "当前页面显示了自然堂旗舰店的直播间信息，包括关注按钮、充值购物金活动等。为了进入直播间观看直播内容，需要点击关注按钮。", "action": "click", "parameters": {"target_element": "关注成功"}}
8. {"reasoning": "当前页面显示了自然堂旗舰店的直播间信息，包括关注按钮、充值购物金活动等。为了进入直播间观看直播内容，需要点击关注按钮。", "action": "click", "parameters": {"target_element": "关注成功"}}
```

3.由于有些情况是购物车图标没有加入购物车几个字，且decider输出是没问题的："target_element": "加入购物车按钮"，但是最后grounder点击就很容易犯错。这里能不能搞一个模式切换，让decider自己来点击，就不用grounder了。

![image-20251209210756283](D:\A_MyApps\Typora\pictures_zyl\image-20251209210756283.png)

4.没有足够多的操作选项，只有点击等待滑动结束，可以参考AutoGLM，它的操作会全很多。



# 设备配置与基本Demo运行

## 环境配置

**实验设服务器连接， 密码：agent**

```
ssh agent@rs3labsrv8.iccluster.epfl.ch -v
conda activate MobiMind
cd mobiAgent/
```



**手机连接到实验室服务器**

首先连接手机到laptop，Windows查看手机设备：Powershell

```
usbipd list
```

记录下显示的BUS ID，然后**以管理员身份运行**PowerShell，把手机设备绑定到wsl

```
usbipd.exe attach --busid 5-1 --wsl
```

在wsl中，查看手机设备连接情况

```
adb devices
## List of devices attached
## 2e3fc77c        device
```

继续在wsl中查看手机IP地址v4，或者直接打开手机wifi查看ip也行

```
adb shell ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1
# 当有多个设备时会报错，需要在-s后面指定设备，用adb device查看
adb -s 128.179.189.127:5555 shell ip addr show wlan0 | grep "inet " | awk '{print $2}' | cut -d'/' -f1
```

记录下ip地址，然后在wsl里面给手机固定网络端口

```
adb tcpip 5555
```

最后打开服务器的cmd

```
adb connect $PHONE_IP:5555
# 比如 adb connect 128.179.189.127:5555。然后通过adb devices验证连接
192.168.31.140
```

简单测试连接，观察手机屏幕反应

```
adb shell input tap 500 500
adb shell input text "HelloMobiAgent"
```



## MobiAgent

**模型启动**

```
# 第二步数据自动标注部署decider
CUDA_VISIBLE_DEVICES=0 vllm serve /scratch/youliang/models/decider \
    --port 8000 \
    --dtype float16 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.95 \
    --enforce-eager

# 要分3个不同的命令行窗口运行，不能关闭保持后台响应
# pipline运行部署模型
CUDA_VISIBLE_DEVICES=0 vllm serve /scratch/youliang/models/decider \
    --port 8000 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager

CUDA_VISIBLE_DEVICES=1 vllm serve /scratch/youliang/models/grounder \
    --port 8001 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager
    
CUDA_VISIBLE_DEVICES=0,1 vllm serve /scratch/youliang/models/planner \
    --port 8002 \
    --tensor-parallel-size 2 \
    --max-model-len 10240 \
    --dtype float16 \
    --gpu-memory-utilization 0.35 \
    --enforce-eager

# 留一整块0卡给decider 7B
CUDA_VISIBLE_DEVICES=1 vllm serve /scratch/youliang/models/grounder \
    --port 8001 \
    --dtype float16 \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.5 \
    --enforce-eager
CUDA_VISIBLE_DEVICES=1 vllm serve /scratch/youliang/models/planner \
    --port 8002 \
    --max-model-len 10240 \
    --dtype float16 \
    --gpu-memory-utilization 0.45 \
    --enforce-eager
```



**模型调用调试**

```
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Hello, what is the result of 15 by 15"}]}'

curl -X POST http://localhost:8001/v1/chat/completions   -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Hello, what is the result of 15 by 15"}]}'

curl -X POST http://localhost:8002/v1/chat/completions   -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Hello, what is the result of 15 by 15"}]}'
```

**停止所有vllm部署**

```
pkill -f "vllm serve"
```

**运行程序**

在task.json指定任务，然后执行

```
cd ~/mobiAgent/MobiAgent
python -m runner.mobiagent.mobiagent --service_ip localhost --decider_port 8000 --grounder_port 8001 --planner_port 8002
```



## UI-TARS 模型

模型部署

```
python -m vllm.entrypoints.openai.api_server \
    --model /scratch/youliang/UI-TARS-models \
    --served-model-name UI-TARS-7B-SFT \
    --host 0.0.0.0 \
    --port 8000
```

**模型调用调试**

```
curl -X POST http://localhost:8000/v1/chat/completions   -H "Content-Type: application/json"   -d '{"messages":[{"role":"user","content":"Hello, what is the result of 15 by 15"}]}'
```

demo运行

```
cd ~/mobiAgent/MobiAgent/runner/UI-TARS-agent
python quick_start.py
```



## Mobiagent Pipline原理

进入mobiagent.py

**planner**

```python
app_name, package_name, new_task_description, template_name = get_app_package_name(task_description)
```

```
app_selection_prompt = app_selection_prompt_template.format(
        task_description=task_description,
        task_templates=template_list_text
    )
    # print(app_selection_prompt)
    while True:
        response_str = planner_client.chat.completions.create(
            model="",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": app_selection_prompt},
                    ]
                }
            ]
        ).choices[0].message.content
```

**app_selection_prompt**从planner.md加载过来，把里面的task_description和template_list_text替换，构成prompt输入到planner模型，planner输出new_task_description，以及template_name。如果planner经过推理认为experience/store文件夹下的模板相关可以调用，就会选择合适的template_name，进入阶段二；否则直接，template_name": null。如下是planner的输出指定prompt

````
```json
{{
    "reasoning": "分析任务内容，说明为什么选择这个应用最合适，并说明所选模板理由。若不适用模板也需说明",
    "app_name": "选择的应用名称",
    "package_name": "选择的应用包名",
    "template_name": "建议使用的模板文件名或 null",
    "task_description": "优化后的任务描述（保持语义完全一致，仅自然表达，不做模板字段扩展）"
}}
```
````

如果找到模板，则会根据第一阶段给出的new_task_description和任务模板（预定义好的prompt），重新构建prompt。最终输出final_task_description

````
```json
{{
  "reasoning": "说明你如何映射模板步骤与用户需求以及如何裁剪",
  "final_task_description": "最终的完整任务描述文本"
}}
```
````

**一个完整的示例**

````
connect to device
- Bilibili_play.md: 搜索并播放视频的任务
- Alipay_collect.md: 支付宝蚂蚁森林收集能量
- Bilibili_follow.md: 关注UP主
- Alipay_autopay.md: 关闭支付宝自动扣费
- Add_cart.md: 针对电商加入购物车（购物类默认模版，如未要求下单时）
- Place_order.md: 针对电商下单购买操作
- Bilibili_comment.md: 视频评论与弹幕发送
- Bilibili_sanlian.md: 视频点赞、投币和收藏
- Hotel_reservation.md: 针对酒店查找、预定类任务的操作
- Ticket_reservation.md: 针对机票、火车票预定任务的操作
2025-10-06 14:44:44,101 - INFO - HTTP Request: POST http://localhost:8002/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-06 14:44:44,107 - INFO - 阶段1 应用/模板选择响应: 
```json
{
    "reasoning": "用户希望在哔哩哔哩视频软件中播放一个视频，因此选择哔哩哔哩应用是最合适的。Bilibili_play.md模板正好适用于搜索并播放视频的任务。",
    "app_name": "哔哩哔哩",
    "package_name": "tv.danmaku.bili",
    "template_name": "Bilibili_play.md",
    "task_description": "打开哔哩哔哩视频软件，然后播放一个视频"
}
```
2025-10-06 14:44:46,159 - INFO - HTTP Request: POST http://localhost:8002/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-06 14:44:46,160 - INFO - 阶段2 模板填充响应: 
```json
{
  "reasoning": "根据用户的需求，任务描述中不需要包含搜索关键词和视频标题等额外信息，因为用户没有提供具体的搜索关键词或视频标题。因此，直接保留模板中的关键步骤，并裁剪掉不必要的占位符。",
  "final_task_description": "请你使用哔哩哔哩App，帮我完成任务“打开哔哩哔哩视频软件，然后播放一个视频”。主要操作流程为：\n\n1. 打开哔哩哔哩App。\n2. 浏览首页，找到与需求匹配的视频。\n3. 点击进入视频播放页。\n4. 播放视频，等待页面加载完成。\n5. 可选：全屏观看、调整清晰度、开启弹幕或关闭弹幕。"
}
```
````

在planner结束规划后，开始迭代执行每一步，每一步由decider决定点哪个图标，输出目标的自然语言描述。然后由grounder来识别图标的具体坐标bbox，并做出点击动作。



**decider**

```
screenshot = get_screenshot(device)
decider_prompt = decider_prompt_template_zh.format(
    task=task,
    history=history_str
)
# logging.info(f"Decider prompt: \n{decider_prompt}")
decider_response_str = decider_client.chat.completions.create(
    model="",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{screenshot}"}},
                {"type": "text", "text": decider_prompt},
            ]
        }
    ],
    temperature=0
).choices[0].message.content
```



用任务描述task与这项任务执行的历史步骤history构建decider的prompt

```
decider_prompt_template_zh = """
你是一个手机使用AI代理。现在你的任务是“{task}”。
你的操作历史如下：
{history}
请根据截图和你的操作历史提供下一步操作。在提供操作之前，你需要进行仔细的推理。
你的操作范围包括：
- 名称：点击（click），参数：目标元素（target_element，对要点击的UI元素的高级描述）。
- 名称：滑动（swipe），参数：方向（direction，UP、DOWN、LEFT、RIGHT中的一个）。
- 名称：输入（input），参数：文本（text，要输入的文本）。
- 名称：等待（wait），参数：（无参数，将等待1秒）。
- 名称：完成（done），参数：（无参数）。
你的输出应该是一个如下格式的JSON对象：
{{"reasoning": "你的推理分析过程在此", "action": "下一步操作（click、input、swipe、done中的一个）", "parameters": {{"param1": "value1", ...}}}}"""
```

**播放b站视频任务的第一步decider输出示例：target_element，对要点击的UI元素的高级描述**

```
{"reasoning": "为了完成任务，我需要选择一个视频来播放。我将点击第一个视频“女人乘坐电梯的时候发现了不对劲...”来开始播放。", "action": "click", "parameters": {"target_element": "标题为“女人乘坐电梯的时候发现了不对劲...”的视频"}}
```

然后把输出转换为结构化的json加入到react.json文件



**grounder**

```
reasoning = decider_response["reasoning"]
            target_element = decider_response["parameters"]["target_element"]
            grounder_prompt = (grounder_prompt_template_bbox if bbox_flag else grounder_prompt_template_no_bbox).format(reasoning=reasoning, description=target_element)
            # logging.info(f"Grounder prompt: \n{grounder_prompt}")  
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
```

提取decider模型输出的reasoning和target_element，注入到grounder的prompt里面

```
grounder_prompt_template_bbox = '''
Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: {reasoning}
Target element's description: {description}
Your output should be a JSON object with the following format:
{{"bbox": [x1, y1, x2, y2]}}'''
```

由于设定bbox_flag=True，所以选择的prompt固定输出为GUI元素描述图案对应的bbox，如果想要选择让grounder输出具体的绝对坐标，改为bbox_flag=False

```
def task_in_app(app, old_task, task, device, data_dir, bbox_flag=True):
```

输出格式示例

```
{"bbox": [125, 86, 618, 149]}
```

然后会读取这个输出的bbox，计算bbox的中心点坐标，操控手机点击。最后把操作记录加入到actions.json.

同时会把这一步操作的过程记录截图，用红色框可视化，中心点用绿点可视化，这些每一步的过程记录放在mobiagent/data/n/文件夹下面，记录每一步的操作。

最后在完成整个任务后，把执行过的action加入到actions.json，可以在react.json和actions.json查看操作记录



**Planner不再参与**：一旦生成了最终的[new_task_description]，planner就完成了它的使命

**Decider根据task和history，根据当前状态生成下一步决策，直到下一步是done为止**



# 数据集制作方法

## **数据采集与标注**

**1.manual手动使用数据标注网页进行人工采集**

启动数据收集服务器

```
cd ~/mobiAgent/MobiAgent
python -m collect.manual.server
```

完成这一步之后得到一个文件夹，下面放着每一步的截图以及一个描述每一步骤的action.json，但是这一步只是对操作的简单描述，没有高级信息，比如具体点了哪里，为什么要这样点。



**2.auto自动采集**

**使用 OpenAI 兼容 API（如 Qwen）**：

```
cd ~/mobiAgent/MobiAgent
python -m collect.auto.server \
    --model "qwen-vl-plus" \
    --api_key "sk-4201f908ffb241d0b4f2eaaf81048add" \
    --base_url "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" \
    --max_steps 20
```

**使用 Gemini API**：

```
python -m collect.auto.server \
    --model "gemini-2.5-pro" \
    --api_key "AIzaSyB2nteoAPuuH2De5opj8iTzjMdzyb2yU-I" \
    --api_type gemini \
    --max_steps 20
```

**使用 Openai API**：

```
python -m collect.auto.server \
    --model "gpt-4o" \
    --api_key "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" \
    --api_type openai \
    --base_url "https://api.openai.com/v1" \
    --max_steps 20
```

**本地模型api**

```
python -m collect.auto.server \
    --base_url "http://localhost:8000/v1" \
    --api_type local \
    --max_steps 20
```

步骤1: 任务初始化，读取task.json中的任务描述：调用Planner选择应用:

步骤2: 截取当前屏幕：screenshot.jpg 保存UI层级树：hierarchy.xml。OmniParser分析，YOLOv8检测UI元素（图标、按钮等），PaddleOCR检测文本框，返回bounds_list（所有元素的边界框坐标）  

步骤3: 多层标注图生成（draw_bounds.py）贪心算法分配元素到不同图层（避免重叠），绘制红色边框 + 索引数字 ，输出：layer_1.jpg, layer_2.jpg, ...                       

步骤4: VLM决策（auto_decider.md prompt）输入给VLM：任务描述， 操作历史，屏幕截图（完整原图）， layer_1.jpg, layer_2.jpg， VLM输出点击坐标和原因                                       

步骤5: 执行操作并记录 



**数据标注**

采集完的数据为截图list和action.json，运行**annotate.py**

注意这里默认目前所有的数据都是手动标注的，都是具有完整的框和坐标对应的，所以可以执行**annotate_without_omniparser.py**

```
sys_prompt = load_prompt("annotation_en_general.md")
chain = prompt | model
```

这里说明了如何导入提示词，让调用另外的视觉模型，生成对应每一步的推理信息，得到react.json

**qwen api**

```
python -m collect.annotate_without_omniparser \
    --data_path collect/manual/data \
    --api_type openai \
    --model qwen-vl-max \
    --api_key sk-4201f908ffb241d0b4f2eaaf81048add \
    --base_url https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

**gemini api**

```
python -m collect.annotate_without_omniparser \
    --data_path collect/manual/data \
    --api_type gemini \
    --model gemini-2.5-flash \
    --api_key AIzaSyB2nteoAPuuH2De5opj8iTzjMdzyb2yU-I \
    --base_url none
```

**local api**

```
python -m collect.annotate_without_omniparser \
    --data_path collect/manual/data \
    --api_type local \
    --model none \
    --api_key none \
    --base_url http://localhost:8000/v1
```

调用视觉模型api，结合三个截图和对应的action.json，得到具体的三个截图对应的操作，以及react.json，里面记录了为什么要这么操作的分析。



**数据标注提示词：**

type4：2，7

livestream: 18

1. 注意搜索结果，如果是“闲鱼”的二手买卖产品就需要跳过，具体操作是往下滑，找到不是闲鱼的第一个商品，然后再点击进入。
2. 关于加入购物车任务的成功判定方法，看又右上角购物车图标的右上角的橙色小数字，如果小数字加一，就说明我刚刚把这一件商品加入了购物车，使得购物车的总数加一，这就说明加入购物车成功了。
3. 关于带具体参数的商品加入购物车，需要在加入购物车前看到明确的图标才能说明加入了用户指定参数的商品。具体操作可以上下滑动等操作查看当前页面的信息，来确认是否真的已经选定了参数规格。比如加入45码的鞋子到购物车，必须在看到“45”这样明确指示鞋码的图标后才能说明已经选定了45码的鞋子，才可以加入购物车。
4. 如果点击进入某一个商品页面后，在查看商品具体信息后，发现没有达到用户指令的要求，比如：没有指定的参数规格（比如手机内存大小，产品颜色等）。则需要退出当前商品，退回到搜索结果界面，按照搜索结果顺序重新选择别的商品，进一步查看别的商品是否符合用户指令要求，直到找到符合用户指令描述的商品才可以加入购物车。
5. 进入直播间只需要点击商店的头像图标就可以了，该商户可能在直播或者不在直播，只要最后进入到正确的页面，带有商户名字和图标，或者“直播间”的字样。即可算作进入直播间



**标注后数据检查**

```
# 只检查，不删除
python -m workspace.data_tools.annotated.labeled_data_checker

# 检查并删除 parse.error
python -m workspace.data_tools.annotated.labeled_data_checker --clean-errors

# 指定数据目录
python -m workspace.data_tools.annotated.labeled_data_checker --data-path $path$ --clean-errors

# 删除 livestream 类型下所有 react.json
python -m workspace.data_tools.annotated.labeled_data_checker --clean-react livestream
```



## **SFT数据集构建**

**ss_data单步数据集生成**

原理是从完整的多步数据中拆分成单步数据

gemini api重写单步数据

```
python workspace/data_tools/annotated/ss_data_generation.py \
    --gemini_api_key AIzaSyB2nteoAPuuH2De5opj8iTzjMdzyb2yU-I \
    --model_name gemini-2.5-pro
```



**unexpected_data意外图片生成**

直接点击运行unexpected_data_generation.py



**construct_sft.py**

这里默认没有单步数据ss_data_not_exist

```
python -m collect.construct_sft \
    --data_path ~/mobiAgent/MobiAgent/collect/manual/data/淘宝 \
    --ss_data_path ~/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/ss_data \
    --unexpected_img_path ~/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/unexpected_data \
    --out_path ~/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/sft_data \
    --factor 0.5 \
    --train_ratio 0.9
```

![image-20251027113601002](D:\A_MyApps\Typora\pictures_zyl\image-20251027113601002.png)

**第一步：打开actions.json和react.json把task_description和actions和react_data提取出来**

```python
actions_json = os.path.join(root, "actions.json")
with open(actions_json, 'r', encoding='utf-8') as file:
    data = json.load(file)

# 提取信息
task_description = data.get("task_description")  # 你的7个任务描述变体
actions = data.get("actions")                     # 你的3个动作

# 读取 react.json
react_json = os.path.join(root, "react.json")
with open(react_json, "r", encoding="UTF-8") as f:
    react_data = json.load(f)  # 你的3个推理-动作对
```

**第二步：在`tasks = [task_description[0]]`这里，从多个task_description里面挑选出3个，增强泛化能力**

**第三步：为每个任务描述生成Decider数据 - `create_decider_entries_for_one_task()`**

对于`tasks`中的每一个任务描述，都会调用这个函数。3个tasks就调用3次

每一个子任务，循环处理每一步`for i, react in enumerate(react_data, 1):`

对每一步生成3个不同的数据：

**normal_entries, no_history_entries, terminate_entries** = create_decider_entries_for_one_task

a. 每一步都会生成normal_entries，记录历史步骤和当前步骤输入输出。

b. 对于非done和input，则会生成no_history_entries，这是不带历史的单步状态输入输出数据。因为这两个需要前面的history作为上下文，否则模型学习 不到为什么要done或者input。对于click这种只需要根据当前状态就能够确定，单步数据。

c. 对于`synthesize_terminate = action_type != "wait" and action_type != "done" and action_type != "swipe"`，会生成terminate_entries，这是与预期截图不符的错误截图数据，让模型学习到与预期不符的状态

![image-20251027113616843](D:\A_MyApps\Typora\pictures_zyl\image-20251027113616843.png)

对于decider，这一个数据集的3步可以生成6个数据，总共有3个tasks。考虑3个任务描述变体，实际是第一个任务6个 + 后两个任务9，第一个任务保留所有数据，后续任务采样75%

**第四步：为每个任务生成Grounder数据 - `create_grounder_entries_for_one_trace()`**

在react.json里面遍历检查有没有click，如果有就把里面的bbox和带点击的截图拿出来，根据：`instruction=grounder_prompt.format(reasoning=reasoning, description=param["target_element"]),` 构建单步click数据。

action.json对于click会保存bbox和coordinate两种形式，保持其它instruction不变，两种不同的坐标形式对于两个数据。所以对于一个task，有n个click就有2n个数据。



**额外单步数据：** 代码中并没有给出单步数据的制作代码，单步数据可以从多步中拆出来，也可以单独手动制作。但是input和done这种需要历史背景的操作不可以做单步数据。单步数据的目的是为了保证模型的泛化能力。



**第五步：数据整合**，把decider的**normal_entries, no_history_entries, terminate_entries**加上单步数据放到一起，按照91分为训练集和验证集。grounder有grounder_entries加上单步数据，整合成grounder数据集。



**SFT数据质量检查**

--sft_data_path指定数据集路径；--verbose指定是否详细输出信息

```
python -m workspace.data_tools.sft.sft_data_check \
--sft_data_path /scratch/youliang/mobidata/sft_data/sft_taobao_100

python -m workspace.data_tools.sft.sft_data_check \
--sft_data_path ~/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/sft_data/ \
--verbose
```





# SFT监督微调

## **训练策略与过程**

一、模型与框架选择

**基础模型**: Qwen/Qwen2.5-VL-7B-Instruct
理由: 这是 MobiAgent 项目指定的视觉语言模型，7B 参数量在 2×A100 40GB 上可行。

**训练框架**: DeepSpeed ZeRO-2 + HuggingFace Transformers + PEFT
理由: ZeRO-2 对梯度和优化器状态进行分片，在保持训练效率的同时显著降低显存占用，适合多卡 LoRA 微调场景。

**精度**: bf16
理由: A100 原生支持 bf16，相比 fp16 有更大的动态范围，不易溢出，训练更稳定。

------

二、冻结策略

**freeze_vision_tower**: True
理由: Vision Tower 约 625M 参数，冻结它可大幅减少显存占用和训练时间。Decider 任务主要依赖语言理解能力，视觉编码器已在预训练中学到足够的图像表征。

**freeze_llm**: True
理由: 使用 LoRA 微调时必须冻结 LLM 主干，只训练低秩适配器。这是 PEFT 框架的标准用法。

**freeze_merger**: False
理由: Merger 是连接视觉特征和语言模型的桥梁，仅约 26M 参数。解冻 Merger 可以让模型更好地适应手机 UI 截图的视觉特征与决策任务之间的对齐关系。

**vision_lora**: False
理由: 已经冻结了 Vision Tower，不需要在视觉编码器上应用 LoRA。这样可以进一步节省显存。

------

三、LoRA 配置

**lora_rank**: 64
理由: 对于 7B 模型和接近 1 万条数据的微调任务，rank=64 提供了足够的表达能力。太小容易欠拟合，太大则显存压力增加且容易过拟合。

**lora_alpha**: 64
理由: 采用 alpha 等于 rank 的设置，使得缩放因子 alpha/rank 等于 1，是一个平衡的选择。既不会过度放大 LoRA 的影响，也不会削弱其作用。

**lora_dropout**: 0.05
理由: 轻微的 dropout 可以提供一定的正则化效果，防止过拟合，同时不会过度影响训练效率。

------

四、学习率配置

**learning_rate**: 5e-5
理由: 这是 LoRA 微调的常用学习率范围。相比全量微调，LoRA 通常使用略高的学习率，5e-5 是一个稳健的起点。

**merger_lr**: 5e-5
理由: Merger 参数量较小，与 LoRA 使用相同的学习率可以保持训练的同步性和稳定性。

**lr_scheduler_type**: cosine
理由: 余弦退火调度器在训练后期平滑降低学习率，有助于模型收敛到更好的局部最优。

**warmup_ratio**: 0.03
理由: 3% 的 warmup 步数可以让模型在初期稳定地适应训练，避免初始阶段的梯度震荡。

------

五、图像处理配置

**image_min_pixels**: 256×28×28 = 200,704
理由: 设置一个合理的下限，避免过小的图像丢失太多信息。

**image_max_pixels**: 1280×28×28 = 1,003,520
理由: 你的训练图像分辨率为 632×1390，共计 878,480 像素。设置 max 为约 100 万像素可以确保图像无需降采样，保留完整的 UI 细节信息。若设置为 896×28×28 约 70 万像素，则会强制对所有图像进行约 20% 的压缩，损失信息。

------

六、批次与梯度配置

**batch_per_device**: 2
理由: 保守选择，确保在 40GB 显存下有充足的安全边际。考虑到高分辨率图像（约 4500 个 vision tokens/图）的激活值内存占用，batch=2 可以稳定运行。

**gradient_accumulation_steps**: 16
理由: 使用 2 张卡、每卡 batch=2，累积 16 步，得到全局 batch size = 2×2×16 = 64。这是一个常见的有效批次大小，既能提供稳定的梯度估计，又不会使单步更新间隔过长。

**gradient_checkpointing**: True
理由: 这是高分辨率 VLM 训练的必选项。通过重计算中间激活值来换取显存，可以大幅降低峰值显存占用。原脚本设置为 False 是危险的，必须修改为 True。

------

七、训练轮次与保存策略

**num_train_epochs**: 3
理由: 对于约 9200 条数据的监督微调，3 个 epoch 是一个合理的起点。可以让模型充分学习数据，同时不至于过拟合。

**save_steps**: 500
理由: 按照 global_batch_size=64 计算，9197 条数据每个 epoch 约 144 步，3 个 epoch 共约 432 步。设置 save_steps=500 意味着在训练结束时保存一次。如果想要中间检查点，可以适当减小这个值。

**save_total_limit**: 3
理由: 限制保存的检查点数量为 3 个，节省磁盘空间，同时保留最近的几个版本以便回滚。

------

八、其他配置

**use_liger_kernel**: True
理由: Liger Kernel 是一个优化的 CUDA 内核库，可以加速 Transformer 的前向和反向传播，提升训练效率。

**disable_flash_attn2**: False
理由: 启用 Flash Attention 2 可以显著减少注意力计算的显存占用和计算时间，A100 完全支持。

**tf32**: True
理由: A100 支持 TF32 张量核心，可以在保持精度的同时加速矩阵运算。

**weight_decay**: 0.1
理由: 适度的权重衰减提供正则化效果，防止过拟合，0.1 是 Transformer 微调的常用值。

**dataloader_num_workers**: 4
理由: 4 个数据加载工作进程可以充分利用 CPU 进行数据预处理，避免 GPU 等待数据。

**report_to**: tensorboard
理由: 使用 TensorBoard 记录训练指标，便于监控训练进度和调试。



**训练问题与解决方案**

**1.多卡通信死锁问题**

```
# 原代码（有问题）
def _save_checkpoint(self, ...):
    if not self.args.should_save:  # Rank 1: should_save=False
        return                      # Rank 1 直接退出！
    # 只有 Rank 0 执行下面的代码
    self.deepspeed.save_checkpoint(...)  # 这是分布式操作，需要所有 rank 参与
```

在单卡训练或普通的 PyTorch DDP（Data Distributed Parallel）中，保存模型通常只需要主进程（Rank 0）把数据写入硬盘即可，其他进程不需要参与。但是在 **DeepSpeed ZeRO** 模式下，情况完全变了：为了节省显存，DeepSpeed ZeRO 将模型参数、梯度和优化器状态（Optimizer States）切分成小块，分散存储在 Rank 0 和 Rank 1 上。Rank 0 自己并没有完整的模型状态。保存需要“全员集合：当需要保存 Checkpoint 时，DeepSpeed 必须指挥所有显卡：大家把数据汇总发给 Rank 0。这意味着：**`self.deepspeed.save_checkpoint(...)` 是一个“集体通信操作”（Collective Operation）**。它就像一次“全员点名”，所有人都必须喊“到”，流程才能继续。所以R1跳出这个函数return之后就不会再回来了，不会再运行self.deepspeed.save_checkpoint来响应R1的号召。



## 微调训练与模型部署

首先在finetune_lora_vision.sh把参数和数据导入，保存的路径配置好，然后运行脚本开始训练

```
cd /home/agent/mobiAgent/MobiAgent/workspace/training/post_training
bash scripts/finetune_lora_vision.sh
```

**tensorboard训练效果查看**

```
cd ~/mobiAgent/MobiAgent/workspace/training/post_training
tensorboard --logdir output/mobimind_decider_lora_sft/runs/$具体的run case$/
```



**模型部署与保存**

1.deploy_lora.py

先保存训练结果权重：按照配置文件config.json定义，保存lora训练的输出文件夹`lora_source_path`到`models_dir/lora_save_name`的指定位置

```
cd /home/agent/mobiAgent/MobiAgent/workspace/training/post_training/deploy
python deploy_lora.py --save
```

保存完权重之后，可以把output目录下的checkpoint删除，这个文件很占空间，因为里面含了基础模型的权重

然后再部署：先在config.json里面配置好基础模型`base_model_path`加lora模型`models_save_dir/lora_save_name`

```
cd /home/agent/mobiAgent/MobiAgent/workspace/training/post_training/deploy
python deploy_lora.py --deploy
```

2.merge_lora.py

合并权重lora adpter和基础模型权重（仅保存，不部署），保存在`models_save_dir/merged_save_name`

```
cd /home/agent/mobiAgent/MobiAgent/workspace/training/post_training/deploy
python merge_lora.py
```

测试部署后的模型

```
CUDA_VISIBLE_DEVICES=0 vllm serve /scratch/youliang/models/decider_lora_2_merged --port 8000 --dtype float16 --max-model-len 32768 --gpu-memory-utilization 0.9 --enforce-eager
```



训练中断问题测试方法

```
# 单卡基线测试
cd ~/mobiAgent/MobiAgent/workspace/training/post_training
python scripts/debug_eval_timeout.py --mode single

# 双卡通信测试
torchrun --nproc_per_node=2 scripts/debug_eval_timeout.py --mode multi

# 双卡 + 实际验证数据分布测试
torchrun --nproc_per_node=2 scripts/debug_eval_timeout.py --mode eval_data
```



**部署后的测试**

首先运行`python deploy_lora.py --deploy`把decider模型部署到8000端口之后，然后再部署planner和grounder，运行demo mobiagent.py查看效果。[查看前面mobiagent部署章节](# MobiAgent)



## 结论

1. 对于淘宝直播任务，只给出了20条训练数据，训练效果很差，所以不要指望20条数据能够训练好，问题是实现目标的路径是错误的





# GRPO

## 训练策略

一、训练目标
基于SFT微调后的Decider模型，使用GRPO强化学习进一步优化模型的动作决策能力，特别是点击位置的准确性。

二、硬件配置
使用2张A100 80GB显卡，GPU0用于GRPO训练，GPU1部署Grounder服务提供点击位置验证。

三、模型配置
基础模型为已merge的SFT Decider模型（Qwen2.5-VL-7B），采用LoRA微调方式，rank和alpha均为64，dropout为0.05。冻结视觉编码器和LLM主干，仅训练LoRA adapter和merger模块。

四、数据配置
训练集约1350条，验证集约150条。数据格式包含instruction（已含图片token和历史动作）、images（缩放后的截图路径）、gt_action（包含动作类型、位置坐标、bounds边界框等）。数据采样策略为input类型占20%、input前一步占10%、swipe占5%、其余随机采样。

五、训练超参数
每卡batch size为1，每个prompt生成4个候选completion，梯度累积4步，有效batch为4。最大训练步数200步用于初步验证效果。学习率5e-6，余弦调度，warmup比例10%。KL惩罚系数beta设为0.01以保持探索能力。开启gradient checkpointing节省显存，使用bf16混合精度。每10步记录日志，每50步评估验证集，每100步保存checkpoint。

六、奖励函数设计
采用分层奖励机制，范围0到1。解析失败或动作类型错误得0分。动作类型正确但细节错误（如文本不匹配、方向错误、点击位置偏离）得0.3分。完全正确得1分。对于click动作，通过调用Grounder服务预测点击bbox，若预测bbox中心点落在GT bounds内则判定正确。对于input动作采用包含匹配，即预测文本包含GT或GT包含预测均视为匹配。奖励归一化采用group方式，在每组生成的4个候选中计算相对优势。

七、Grounder调用
Grounder服务部署在本地8001端口，采用asyncio并发调用提高效率。输入为截图、模型输出的reasoning和target_element描述，输出为预测的bbox坐标。调用失败时给予0.3的部分奖励而非0分。

八、生成参数
最大completion长度256 token，最大prompt长度4096 token。生成温度0.9，top_p为1.0，使用liger loss优化。



## 强化学习数据集制作

```
cd /home/agent/mobiAgent/MobiAgent/
python -m workspace.data_tools.grpo.construct_grpo \
    --data_path ~/mobiAgent/MobiAgent/collect/manual/data/淘宝 \
    --out_path /home/agent/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/grpo_data \
    --factor 0.5 \
    --train_ratio 0.9 \
    --total_samples 1500
```

检查数据集是否合格

```
cd /home/agent/mobiAgent/MobiAgent/workspace/data_tools/grpo

## 仅检查
python grpo_data_check.py --grpo_data_path /home/agent/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/grpo_data

## 检查并自动清理无效样本
python grpo_data_check.py --grpo_data_path /home/agent/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/grpo_data --auto_fix
```



## GRPO训练方法

**先部署 Grounder 服务（GPU1）**

```
CUDA_VISIBLE_DEVICES=1 vllm serve /scratch/youliang/models/grounder \
    --port 8001 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager
```

**修改脚本中的模型路径：**编辑 `scripts/finetune_grpo.sh` 中的 `BASE_MODEL` 为 merge 后的lora SFT 模型路径，然后**运行训练**

```
cd /home/agent/mobiAgent/MobiAgent/workspace/training/post_training
bash scripts/finetune_grpo.sh
```





# Benchmark与模型测试

## DAG框架原理

```
1.需要提前写好测试的app和配置，预先写好每一步可能的流程和分岔路径，是人为手动写的 
2.使用ocr和llm两种测评方法，且据我观察，如果上一步ocr判断为false且llm判断为true，则下一步只会使用llm判断，直到下一步再恢复ocr判断。检查代码验证我的观点 3.只要有一个环节错误，后面就不会执行，且任务判定为失败
```

```
 初始化阶段 ─────────────────────────────────┐
│ 1. 加载YAML配置,构建DAG图                   │
│ 2. 验证无环性、依赖一致性                   │
│ 3. 发现所有成功路径                         │
│ 4. 初始化OCR/LLM检查器                      │
└─────────────────────────────────────────────┘
           ↓
┌─ 路径感知的帧验证 ────────────────────────────┐
│ FOR EACH 节点 (按拓扑序):                    │
│   IF 节点不可达: SKIP                        │
│   ELSE:                                      │
│     起始帧 = max(父节点匹配帧) + 1           │
│     FOR 帧i FROM 起始帧 TO 最后一帧:         │
│       ├─ 检查帧独占约束(已用帧集合)          │
│       ├─ 执行escalation检查器链:             │
│       │   text → regex → ui → action →      │
│       │   dynamic_match → ocr → llm         │
│       │   ↓ 任意一个返回True立即成功         │
│       ├─ IF 匹配成功:                        │
│       │   ├─ 记录候选帧                      │
│       │   ├─ 更新后继节点可达性              │
│       │   └─ IF 帧独占: BREAK(早停)         │
│       └─ ELSE: 继续下一帧                    │
└─────────────────────────────────────────────┘
           ↓
┌─ 强制LLM模式触发 ────────────────────────────┐
│ IF 节点N的OCR失败但LLM成功:                 │
│   节点N+1进入force_llm_verification模式     │
│   ↓ 只执行LLM检查,跳过OCR                   │
└─────────────────────────────────────────────┘
           ↓
┌─ 失败传播机制 ──────────────────────────────┐
│ IF 节点X找到0个候选帧:                      │
│   FOR 所有依赖节点X的后继节点Y:             │
│     标记Y为"不可达"                         │
│     跳过Y及其所有子孙节点的检查             │
└─────────────────────────────────────────────┘
           ↓
┌─ 最终判定 ──────────────────────────────────┐
│ 成功匹配节点集 ∩ 成功节点要求 ≠ ∅?         │
│   YES → PASS                                │
│   NO  → FAIL       
```



## DAG测试方法（仅用于预定义task）

**单独测试少量任务，可以测试已知或未知任务：**

进入MobiFlow/task_configs，这里面存放不同app运行测试的配置，以taobao.json为例：

```
  # 存放运行结果数据的路径
  "data_base_dir": "../runner/mobiagent/data", 
  #对每一个步骤的截图，可以选择启用OCR和视觉llm进行评估
   "enable_ocr": true,
   "enable_llm": true,
```

运行structural_test_runner.py的少量测试模式，需要指定某个app的测试配置和测试type

```
# 进入 MobiFlow 目录
cd /home/agent/mobiAgent/MobiAgent/MobiFlow

# 测试单个 trace
python structural_test_runner.py task_configs/taobao.json type3:1 \
    --data-base ../benchmark_test_data/mobiagent/taobao
    
# 测试某个type下的所有 trace， 比如type3
python structural_test_runner.py task_configs/taobao.json type3 \
    --data-base ../benchmark_test_data/mobiagent/taobao
```

测试框架默认的层级结构：

![image-20251104152533613](D:\A_MyApps\Typora\pictures_zyl\image-20251104152533613.png)

实际可以简单建立一个比如type3的文件夹，把某一条数据放在里面，比如type3/7。运行方法1，来具体测试某一条数据的结果



**批量测试已知任务，结果作为benchmark，可用于对比模型性能**

在task_list.json定义结构化的任务测试列表，可用命令行执行脚本选择模型，输出到对应模型的路径保存运行结果。

```
cd ~/mobiAgent/MobiAgent/workspace/benchmark
python run_task_list.py --model $模型结果文件夹名称$
```

![image-20251106183459504](D:\A_MyApps\Typora\pictures_zyl\image-20251106183459504.png)

运行structural_test_runner.py的批量测试模式，一次性运行所有的app和所有的type，timestamp参数自定义需要测试的时间戳文件夹。model参数为对应的模型文件夹。

```
cd /home/agent/mobiAgent/MobiAgent/MobiFlow
python structural_test_runner.py --batch-mode \
    --timestamp $时间$ \
    --model $模型结果文件夹名称$ \
	--workers 2
```



**替换模型进行测试**

首先部署待测试的模型

```
# decider
CUDA_VISIBLE_DEVICES=0 vllm serve /scratch/youliang/$待测模型$ \
    --port 8000 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager

# grounder
CUDA_VISIBLE_DEVICES=1 vllm serve /scratch/youliang/$待测模型$ \
    --port 8001 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager

# planner 对最后一个模型使用张量并行，把显存分给两个gpu
CUDA_VISIBLE_DEVICES=0,1 vllm serve ~/mobiAgent/models/planner \
    --port 8002 \
    --tensor-parallel-size 2 \
    --max-model-len 10240 \
    --dtype float16 \
    --gpu-memory-utilization 0.35 \
    --enforce-eager
```

## grounder单独测试

首先在终端部署需要测试的模型，注意端口要对上（比如这里是8001）.可以通过修改reasoning后面的任务描述，来测试目标识别点击能力。注意手机要提前切换到描述的页面

```
cd ~/mobiAgent/MobiAgent
python -m runner.mobiagent.test_grounder --service_ip localhost --grounder_port 8001
```



## 不同模型测试结果

**qwen2.5-vl-7b为decider，qwen2.5-vl-3b为grounder：**由于decider未微调，导致decider输出格式不对，不符合程序设定，使得甚至连任务都无法执行成功，很多task只有一张截图，在第一步就结束了运行



**grounder性能测试：**

**qwen2.5-vl-7b微调版为decider，qwen2.5-vl-3b为grounder：** 没微调过的grounder也很准确，跑task list的结果准确率与总分和微调过的grounder效果相当，甚至有时候更好。用加购物车task作为测试。测试结果指向任务执行的准确率主要由decider决定，如果decider每一步都能准确说出点击的目标，grounder的输出坐标是十分准确的，即使没有微调过。

**qwen2.5-vl-7b微调版为decider，qwen3-vl-4b-instruct为grounder：** **qwen2.5-vl-7b微调版为decider，qwen3-vl-4b-thinking为grounder：** **qwen2.5-vl-7b微调版为decider，qwen3-vl-8b-instruct为grounder：** 效果都很差，qwen3可能不适合做grounder。单独运行test.grounder.py发现qwen3作为grounder很差



**decider性能测试：**

**qwen3-vl-4binstruct为decider，qwen2.5-vl-3b微调版为grounder：**

未训练的decider不能按照预期模板输出格式，直接丢失了target element，对于任务成功执行的路径不熟悉，执行路径是错的。

{"reasoning": "我需要在淘宝App中找到价格最低的运动鞋并加入购物车。从截图中可以看到，有一个运动鞋商品，价格为¥99.34，这是当前页面上最便宜的运动鞋。为了将它加入购物车，我需要点击该商品的购买按钮或直接进入商品详情页。由于截图中没有直接显示加入购物车的按钮，我需要点击该商品以进入详情页，然后找到加入购物车的选项。", "action": "click", "parameters": {"param1": "潮鞋优惠 ¥99.34"}}

**qwen3-vl-4bthinking为decider，qwen2.5-vl-3b微调版为grounder：**

thinking版本能够输出target element，但是推理时间太久，甚至有时候会无限的长时间推理。可能微调过后会好一些。且由于没有微调，不知道正确的点击的地方在哪里。比如淘宝搜索，它会直接点搜索框右侧的搜索按钮，如果这时候搜索框有默认内容，就直接搜索默认内容了。正确步骤应该是点击搜索框，然后输入目标商品。



**decider换为通用模型api：**

```
python workspace/runners/mobiagent_change_decider_api.py \
    --service_ip localhost \
    --grounder_port 8001 \
    --planner_port 8002 \
    --decider_api_type openai \
    --decider_model "gpt-4o-mini" \
    --decider_api_key "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"
```

**gemini-2.5-flash**





## 未知任务定义

**taobao**

​	special：未知任务

​			s.在淘宝App中打开淘宝直播，点击进入任意一个直播间

​			s.在淘宝App中打开淘宝直播，然后在搜索栏搜索bilibili，最后进入bilibili直播间

**bilibili**

​	special：未知任务

​			s.点击首页的直播，随便进入一个直播间

​			s.点击首页的直播，随便进入一个直播间，发送弹幕“666”

**weixin**

​	special：未知任务

​			s.打开微信视频号，对所展示的第一个视频视频的作者点击关注

​			s.点击微信发现，进入微信视频号，对所展示的第一个视频的视频作者点击关注

**未知app**

**Uber Eats**：unknown未知app

​			u.打开uber eats，打开披萨界面，选择一家店，进入这家店

​			u.打开uber eats，打开披萨界面选择一家店，任意选择一块披萨，点击加入购物车

**youtube**：unknown未知app

​			u. 打开youtube，播放首页任意一个视频

​			u. 打开youtube，搜索ishowspeed，进入他的个人主页



# Progress

### 9.29

解决了demo跑不了的问题，原因是vllm部署指令配置的model len上下文长度太少	

已经能跑部分demo，比如[

  "打开哔哩哔哩软件随机播放一个视频"

]，但是对于打开b占放视频，报错显示：ValueError: Unknown action: 等待

可以尝试debug，也可以再看看UI-TARS是什么，跑一下它的demo

### 10.4

mobiagent和uitars的demo都能跑了，但是目前只能把模型部署到服务器，手机要adb连接服务器才能跑。

下一步：

1. **搞清楚pipline的输入输出，实际的跑一个demo任务**
2. 搞清楚数据集怎么做的，decider和grounder的sft数据是怎么做的
3. 看看uitars的原理与mobiagent有什么不一样
4. 自己尝试基于qwen做一下sft
5. 对一个跑不通的demo debug
6. **看看grounder的输入Target element's description: {description}长什么样**
7. **看看grounder泛化能力**

最终目标：找到一个细分领域的需求，做一个手机端agent解决这个需求

模型问题：可以部署模型到server或者云端，手机软件调用。或者尝试研究如何把模型部署到手机端

### 10.6

搞清楚了pipline每一步的输入输出，重点了解到了grounder输入的target element就是对ui的描述，那这样就可以很容易单独调试grounder

下一步：看看grounder泛化能力，单独写一个脚本，实现：首先读取当前截屏，然后构建prompt，尝试让grounder点击任何我们想点的ui



### 10.8

测试了grounder模型，能够很准确的识别图标ui，即使在一些没有微调过的场景也有不错的准确度，说明qwen基础模型的理解能力本身就很强。

下一步：

1. 找找垂直领域的idea，看看手机agent的具体实现场景
2. 看看现在做agent的paper是如何应用到市场的，有什么创新，包括算法或者软件架构上的创新



### 10.17

大概搞了一遍数据集的manual制作过程。下一步尝试手动标注10个数据，主题选在淘宝购买苹果笔记本电脑。把千问开源模型下载下来，**先直接尝试用没有微调过的模型测试看看效果，包括qwen2.5和最新的qwen3**。然后再尝试做10个数据的简单微调。



### 10.28

~/mobiAgent/MobiAgent/collect/manual/data/淘宝/点击搜索栏/1 ，该目录下存有数据

尝试运行，不能正常生成数据集

```
cd ~/mobiAgent/MobiAgent
python -m collect.construct_sft \
    --data_path ./data/淘宝/点击搜索栏 \
    --ss_data_path ./ss_data_not_exist \
    --unexpected_img_path ./unexpected_img \
    --out_path ./sft_output \
    --factor 0.5 \
    --train_ratio 0.9
```



### 11.4

1. 学习如何benchmark，DAG框架

2. 部署未微调的qwen模型，与微调模型对比效果

3. 尝试构建完整的sft数据集

4. 如何sft？尝试sft

5. 对比sft与未sft模型的效果

6. GRPO训练，如何训？效果是否又有提升？

7. agent memory机制，agentrr推理优化框架

   搞清楚了DAG这个benchmark使用原理，它不适合用来做一些通用性的评估，因为评估方法都是需要预先写好写死了的，包括得分。但是可以用来对比不同模型的能力，看不同模型在固定的任务上的表现。下一步，用官方模型跑一些预定义好的测试任务或者自定义一些没见过的app任务作为标准，最好是选择一些严重依赖微调的任务。可以再看看论文是怎么做对比实验的。然后换qwen3，gemini api，gpt api做对比实验。

### 11.5

建好了运行结果数据的保存路径与结构，测试结果的路径，运行测试方法的管理，包括3种不同情况的管理方式

### 11.6

列了一个task_list，测试5个app的多个任务，写了一个run_task_list的批量自动运行脚本，能够批量自动运行并保存测试数据到指定位置

优化了structural_test_runner.py，支持批量运行评估

### 11.7

尝试修正基准模型测试结果，调整tasklist，目的是尽量让30个任务执行成功，作为benchmark。11.8必须完成的任务，把固定的30个任务benchmark做好定义好，部署qwen2.5vl和qwen3作为grounder，单独测试作为grounder的效果。

### 11.8

下载了qwen系列模型，继续调试tasklist的任务执行，完善运行和测试的log

### 11.9

完成了源微调版本的模型基准测试，做了4组benchmark测试结果作为基准参考。发现了planner需要模板才能给出详细步骤，能否优化planner？测试发现主要瓶颈为decider模型，grounder即使没有微调表现也很好。

### 11.10

根据测试结果，planner不支持未知任务的运行；grounder可以不微调，且2.5比qwen3更好用；decider一定要微调，因为：1.输出格式不对 2.任务执行的路径不对；或许可以尝试微调decider thinking模式，但可能陷入长时间推理无法退出。可以开始准备微调数据集，先训练decider。另外关于planner或许可以尝试使用qwen3.

### 11.12

运行了自动采集数据的pipline

### 11.13

调试了gemini api，下一步测试不同vl模型的自动采集能力。读了agentflow在线rl的论文。

### 11.15

测试了Gemini2.5flash，作为数据采集的视觉模型确实比qwen要好，更加准确。需要进一步确认具体在哪个app上面做训练，然后开始定义task.json，批量进行数据采集。

### 11.17

尝试使用openai，gemini，微调后的decider作为自动采集数据集的模型，但是只有genimi表现较好，然而gemini api很容易过载，且自动采集精度不是特别高，决定手动标注数据。下一步，标注300-400条淘宝数据，构建微调数据集，然后尝试训练。未来展望，改进思路目前比较看好planner制定计划，基于xml识别ui元素，decider只在ui元素选取点击位置，进行概率评估，如果执行失败就退回，然后尝试实时强化学习来训练模型。

### 11.18

标注了20条淘宝数据，搜索商品进入具体的商品页面，主要是价格最低销量最高。下一步标注：搜索将<商品描述>加入购物车；将<选择条件>的<商品描述>加入购物车；在淘宝App中将<参数规格>的<商品描述>加入购物车，需要把规格加入搜索词；

### 11.26

优化了自动标注reasoning的提示词，加入了fewshot prompt的技巧，重新生成数据集的react.json。顺利的话明天可以构造sft数据集。

### 11.27

终于把数据集做完了，成功运行了construct_sft，把微调数据集做好了。这个过程中遇到了很多困难：

1.第一步是数据采集，尝试过使用llm api进行自动数据采集，qwen效果很差，Gemini之前尝试的时候服务器总是崩溃，所以最后选择手动标注了100条数据。不过现在Gemini api可以用了，后面需要更多数据可以尝试Gemini api自动采集，但是还是需要人工检验。

2.第二部数据标注，这一步必须要用一个视觉模型进行自动标注，这里还是用的gemini api，这说明在视觉模型gemini还是表现最好。但是会有很多问题，api调用间隔要设置sleep，不然会报错调用频率过高。自动标注的数据有些问题，不可能所有东西都要人工检查，写了一个脚本labeled_data_checker.py来检查数据，是否格式正确，有没有一些最基本的问题。但其实这样还是不能保证数据100%是正确的，还是要人工检查。

3.构建sft还需要单步数据和意外数据，写了脚本ss_data_generation.py来把完整步骤的数据拆分为单步。写了unexpected_data_generation.py脚本来手动截取保存一些意外图片。

最后查看这个sft数据集构建完了可能还需要进一步处理才能sft，所以复现训练的工作比想象的难一些。

### 12.2

整理好了微调数据集，准备开始写训练代码。

### 12.5

写好了微调代码，开始尝试训练。学习qwen2.5架构，lora，deepspeed zero2和3的区别，不同超参数对训练速度的影响

### 12.6

解决了deep speed多卡通信死锁问题。使用lora微调了3个epoch，顺利完成了训练，初步查看loss效果不错。准备调试微调后的模型。

### 12.7

测试了sft模型效果。模型学习到了输出格式，以及部分的执行路径，但是绝大多数时候不能表现出很好效果。可能的愿意有：1.这是sft，本来就是用来规范模型输出的，不能指望一步到位，按论文的说法是为了不让RL奖励稀疏。 2.lora参数配置有问题，应该只给K,V,MLP层加lora，不给Q加lora，且给全block加lora。要给模型验证加的更密，才能确认是否过拟合。 3.不应该用lora，还是要用全量微调。

下一步行动：进行全量微调，查看grpo具体如何实施

### 12.9

优化了lora策略，进行了第二次微调，效果略好于之前的模型。写完了第一版grpo训练代码。下一步，理解grpo原理与参数设置，进行grpo训练。













## 输出记录

````
connect to device
- Bilibili_play.md: 搜索并播放视频的任务
- Alipay_collect.md: 支付宝蚂蚁森林收集能量
- Bilibili_follow.md: 关注UP主
- Alipay_autopay.md: 关闭支付宝自动扣费
- Add_cart.md: 针对电商加入购物车（购物类默认模版，如未要求下单时）
- Place_order.md: 针对电商下单购买操作
- Bilibili_comment.md: 视频评论与弹幕发送
- Bilibili_sanlian.md: 视频点赞、投币和收藏
- Hotel_reservation.md: 针对酒店查找、预定类任务的操作
- Ticket_reservation.md: 针对机票、火车票预定任务的操作
2025-10-07 16:11:58,204 - INFO - HTTP Request: POST http://localhost:8002/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:11:58,211 - INFO - 阶段1 应用/模板选择响应: 
```json
{
    "reasoning": "用户想要在淘宝上搜索苹果笔记本电脑，因此选择淘宝应用是最合适的。虽然任务模板列表中没有直接针对电商搜索的模板，但可以使用Add_cart作为基础模板进行自然化改写。",
    "app_name": "淘宝",
    "package_name": "com.taobao.taobao",
    "template_name": "Add_cart",
    "task_description": "打开淘宝，搜索苹果笔记本电脑"
}
```
2025-10-07 16:11:58,211 - WARNING - 模板 Add_cart 不存在，跳过填充。
Starting task '打开淘宝，搜索苹果笔记本电脑' in app '淘宝' (模板: Add_cart)
2025-10-07 16:12:01,567 - INFO - HTTP Request: POST http://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:12:01,567 - INFO - Decider response: 
{"reasoning": "为了搜索“苹果笔记本电脑”，我首先需要点击屏幕顶部的搜索框以激活它并调出键盘。", "action": "click", "parameters": {"target_element": "位于屏幕顶部的搜索框"}}
2025-10-07 16:12:02,128 - INFO - Grounder prompt: 

Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: 为了搜索“苹果笔记本电脑”，我首先需要点击屏幕顶部的搜索框以激活它并调出键盘。
Target element's description: 位于屏幕顶部的搜索框
Your output should be a JSON object with the following format:
{"bbox": [x1, y1, x2, y2]}
2025-10-07 16:12:02,831 - INFO - HTTP Request: POST http://localhost:8001/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:12:02,832 - INFO - Grounder response: 
{"bbox": [105, 139, 484, 187]}
2025-10-07 16:12:05,452 - INFO - HTTP Request: POST http://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:12:05,452 - INFO - Decider response: 
{"reasoning": "我已经激活了搜索框，现在我需要输入我的搜索关键词“苹果笔记本电脑”。", "action": "input", "parameters": {"text": "苹果笔记本电脑"}}
2025-10-07 16:12:17,529 - INFO - HTTP Request: POST http://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:12:17,530 - INFO - Decider response: 
{"reasoning": "我已经输入了完整的搜索词“苹果笔记本电脑”，现在我需要点击搜索按钮来执行搜索。", "action": "click", "parameters": {"target_element": "搜索框右侧的“搜索”按钮"}}
2025-10-07 16:12:20,772 - INFO - Grounder prompt: 

Based on the screenshot, user's intent and the description of the target UI element, provide the bounding box of the element using **absolute coordinates**.
User's intent: 我已经输入了完整的搜索词“苹果笔记本电脑”，现在我需要点击搜索按钮来执行搜索。
Target element's description: 搜索框右侧的“搜索”按钮
Your output should be a JSON object with the following format:
{"bbox": [x1, y1, x2, y2]}
2025-10-07 16:12:21,488 - INFO - HTTP Request: POST http://localhost:8001/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:12:21,489 - INFO - Grounder response: 
{"bbox": [526, 163, 619, 218]}
2025-10-07 16:12:33,025 - INFO - HTTP Request: POST http://localhost:8000/v1/chat/completions "HTTP/1.1 200 OK"
2025-10-07 16:12:33,026 - INFO - Decider response: 
{"reasoning": "我已经成功搜索了“苹果笔记本电脑”，并且搜索结果已经显示在屏幕上。任务已完成。", "action": "done", "parameters": {}}
Task completed.
````





## **配置问题记录**

wsl无法code .打开vscode

首先找到windows下vscode的安装位置

```
D:\Microsoft VS Code\bin
```

在wsl对应的路径就是

```
/mnt/d/Microsoft VS Code/bin
```

进入bashrc修改

```
nano ~/.bashrc
# 在末尾加入
export PATH=$PATH:"/mnt/d/Microsoft VS Code/bin"
#最后
source ~/.bashrc
```





## 测试失败案例

**taobao**

<img src="D:\A_MyApps\Typora\pictures_zyl\image-20251106200501267.png" alt="image-20251106200501267" style="zoom: 15%;" />

任务：

在淘宝App中将红色32GB的手机加入购物车

现象：

"reasoning": "我已经成功将指定颜色和规格的手机加入了购物车，完成了所有任务步骤。"

"reason": "当前界面显示仍为选择机身颜色和存储容量的选项，且底部的'加入购物车'按钮未被激活或显示已添加状态，表明物品尚未成功加入购物车。"

原因分析：

decider有问题，认为已经完成任务

<img src="D:\A_MyApps\Typora\pictures_zyl\image-20251107164316068.png" alt="image-20251107164316068" style="zoom:15%;" />

任务：

在淘宝App中将销量最高的牙刷加入购物车

现象：decider是对的

"reasoning": "我已经成功找到了销量最高的牙刷，并且进入了商品详情页。至此，任务已经完成。"

原因分析：

decider有问题，认为已经完成任务，还没加入购物车

<img src="D:\A_MyApps\Typora\pictures_zyl\image-20251107162330978.png" alt="image-20251107162330978" style="zoom:15%;" />

任务：

在淘宝App中将大号白色的T恤加入购物车

现象：decider是对的

 "reasoning": "我现在在商品详情页。为了将商品加入购物车，我需要点击页面底部的“加入购物车”按钮。","target_element": "位于屏幕底部的“加入购物车”按钮"

"reason": "当前界面显示仍为选择机身颜色和存储容量的选项，且底部的'加入购物车'按钮未被激活或显示已添加状态，表明物品尚未成功加入购物车。"

但是grounder的点击位置输出错了："type": "click", "position_x": 878,"position_y": 2665,

在另一个正确的任务中grounder输出是正确的："type": "click", "position_x": 585,"position_y": 2654,

原因分析：

输出的坐标错了，但是grounder不总是出问题，存在正确输出坐标的case，说明不是坐标转换和输出的问题，而是模型理解的有问题，没有正确理解到购物车图标。

<img src="D:\A_MyApps\Typora\pictures_zyl\image-20251107164526011.png" alt="image-20251107164526011" style="zoom:15%;" />

任务：

在淘宝App中将价格最低的运动鞋加入购物车

现象：decider是对的

"reasoning": "我现在在商品详情页。为了完成任务，我需要将此商品加入购物车。我将点击页面底部的“加入购物车”按钮。","target_element": "位于屏幕底部的“加入购物车”按钮"

"reason": "当前界面显示仍为选择机身颜色和存储容量的选项，且底部的'加入购物车'按钮未被激活或显示已添加状态，表明物品尚未成功加入购物车。"

"reasoning": "我已经成功将价格最低的运动鞋加入了购物车，并且可以看到购物车里已经有两件商品。任务已完成。",

原因分析：

decider对，grounder点错了，同上。但是最后一步decider认为已经加入购物车

**有时候由于grounder输出的坐标在屏幕外，所以手机没有点击反应，对应的步骤截图也没有bbox**























