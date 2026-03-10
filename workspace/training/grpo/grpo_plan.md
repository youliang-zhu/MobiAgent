# MobiMind Decider grpo 复现方案

**版本**：v1.0
**日期**：2026-03-06
**执行者**：youliang
**框架**：TRL GRPOTrainer（全新实现，不沿用旧代码）

---

## 一、总体架构

### 1.1 目标

在 SFT checkpoint（`decider_lora_2_merged`）基础上，通过 grpo 强化训练提升 Decider 的决策准确率，以 MobiMind-Grounder-3B 作为 click 动作的奖励信号来源。

### 1.2 硬件分配

```
GPU 0（A100 40GB）：GRPO 训练（TRL GRPOTrainer + LoRA，单卡）
GPU 1（A100 40GB）：Grounder vLLM 服务（奖励信号来源，长期运行）
```

### 1.3 关键路径（每阶段须通过验收才能进入下一阶段）

```
阶段 0  ->  阶段 1  ->  阶段 2  ->  阶段 3  ->  阶段 4  ->  阶段 5  ->  阶段 6  ->  阶段 7  ->  阶段 8
基线冻结    环境搭建   数据修复   Grounder   Reward    训练代码   冒烟训练   Pilot     正式训练
                              验证       单测       开发      (50步)    (500步)
```

### 1.4 全局决策清单

| 决策项 | 已确认方案 |
|--------|-----------|
| 训练框架 | TRL 0.26.0 GRPOTrainer（全新代码，非旧版恢复）|
| 初始模型 | `/scratch/youliang/models/decider_lora_2_merged`（merged SFT 模型）|
| 奖励类型 | 严格二值（0.0 / 1.0，无任何部分分）|
| click 奖励 | Grounder bbox 中心点在 gt_bounds 内 -> 1，否则 0（含 grounder 失败）|
| GPU 分配 | GPU 0 训练，GPU 1 Grounder |
| 数据 seed | 必须先修复，使用 `--seed 42` 重建数据 |
| conda 环境 | 新建 `grpo` 环境，删除旧 `MobiMindGRPO` |

---

## 二、阶段 0：基线冻结

**目标**：在任何改动前记录项目起点状态，供后续对比。

### 2.1 记录关键路径

```
Decider 起点模型 ：/scratch/youliang/models/decider_lora_2_merged
Grounder 模型   ：/scratch/youliang/models/grounder
基础模型        ：/scratch/youliang/qwen2.5-vl-7b
grpo 数据目录   ：workspace/data/training_data/grpo_data
```

### 2.2 记录当前数据状态

```bash
# 在项目根目录执行
python workspace/data_tools/grpo/grpo_data_check.py \
    --grpo_data_path workspace/data/training_data/grpo_data \
    --no_check_images
```

预期输出：Train 735 / Val 82（记录下来，本阶段数据会被重建）

### 2.3 记录 git 状态

```bash
git log --oneline -5
git status
git stash list
```

### 2.4 通过标准

- [ ] 上述命令均正常执行，输出已记录

---

## 三、阶段 1：环境搭建

### 3.1 删除旧环境（MobiMindGRPO）

```bash
# 先导出依赖快照留存
conda run -n MobiMindGRPO pip list > /tmp/MobiMindGRPO_snapshot_$(date +%Y%m%d).txt
echo "Snapshot saved"

# 确认无活跃训练任务后执行删除
conda env remove -n MobiMindGRPO -y
```

### 3.2 创建新环境 grpo

```bash
conda create -n grpo python=3.10 -y
conda activate grpo
```

### 3.3 安装核心依赖

```bash
# 1. PyTorch（与机器 CUDA 12.8 匹配，和 MobiMind 环境保持一致）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 2. 训练框架
pip install transformers==4.57.3
pip install trl==0.26.0
pip install peft==0.18.0
pip install accelerate==1.12.0

# 3. Qwen2.5-VL 图像处理
pip install qwen-vl-utils

# 4. Grounder HTTP 客户端
pip install openai

# 5. 工具库
pip install ujson tqdm pillow "numpy<2.0.0"

# 6. 训练监控
pip install tensorboard
```

### 3.4 验证安装

```bash
conda run -n grpo python -c "
import torch
import trl
import transformers
import peft
from trl import GRPOTrainer, GRPOConfig

print(f'torch: {torch.__version__}')
print(f'cuda available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
print(f'trl: {trl.__version__}')
print(f'transformers: {transformers.__version__}')
print(f'peft: {peft.__version__}')
print('GRPOTrainer import: OK')
print('GRPOConfig import: OK')
"
```

### 3.5 通过标准

- [ ] torch 2.8.0，cuda=True，GPU count=2
- [ ] trl 0.26.0
- [ ] `GRPOTrainer` 和 `GRPOConfig` 可正常导入

---

## 四、阶段 2：数据修复与重建

**问题**：`construct_grpo.py` 没有 seed 控制，每次运行结果不同，不可复现。
**目标**：修复 seed 控制，重新生成 grpo 数据集，保证可复现。

### 4.1 需要修改的内容（construct_grpo.py）

在 `workspace/data_tools/grpo/construct_grpo.py` 中进行以下 4 处修改：

**修改 1**：在 `argparse` 中增加 `--seed` 参数

```python
# 在 if __name__ == "__main__": 的 argparser 中加入：
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
```

**修改 2**：在 `construct_grpo_dataset` 函数签名中增加 seed 参数并在函数开头设置 seed

```python
def construct_grpo_dataset(data_path, out_path, factor, train_ratio, total_samples, seed=42):
    random.seed(seed)
    # ... 其余代码不变
```

同时在 `__main__` 末尾调用时传入 `seed=args.seed`。

**修改 3**：对 `os.walk` 的结果排序，保证目录遍历顺序确定

在 `collect_all_steps` 函数中，`os.walk` 循环第一行加入 `dirs.sort()`：

```python
for root, dirs, files in tqdm(os.walk(data_path), desc="Scanning trajectories"):
    dirs.sort()  # 关键：就地排序，控制子目录遍历顺序
    ...
```

**修改 4**：对 `all_steps` 在采样前排序

在 `construct_grpo_dataset` 中，`all_steps = collect_all_steps(data_path)` 之后立即加：

```python
# 采样前排序，保证 random.sample 在相同 seed 下结果确定
all_steps.sort(key=lambda x: (x["root"], x["step_idx"]))
```

### 4.2 可复现性验证（两次 md5 必须完全一致）

```bash
# 第一次运行
python workspace/data_tools/grpo/construct_grpo.py \
    --data_path collect/manual/data/淘宝 \
    --out_path /tmp/grpo_verify_1 \
    --factor 0.5 \
    --train_ratio 0.9 \
    --total_samples 817 \
    --seed 42

# 第二次运行（不同 out_path，相同 seed）
python workspace/data_tools/grpo/construct_grpo.py \
    --data_path collect/manual/data/淘宝 \
    --out_path /tmp/grpo_verify_2 \
    --factor 0.5 \
    --train_ratio 0.9 \
    --total_samples 817 \
    --seed 42

# 对比（train 和 val 均必须完全一致）
md5sum /tmp/grpo_verify_1/mobimind_decider_grpo_train.json
md5sum /tmp/grpo_verify_2/mobimind_decider_grpo_train.json
md5sum /tmp/grpo_verify_1/mobimind_decider_grpo_val.json
md5sum /tmp/grpo_verify_2/mobimind_decider_grpo_val.json
```

### 4.3 生成正式数据集

验证一致后，覆盖当前 grpo_data 目录：

```bash
# 备份旧数据元数据
cp workspace/data/training_data/grpo_data/metadata.json \
   /tmp/grpo_metadata_old_backup.json

# 生成新数据（seed=42，覆盖旧数据）
python workspace/data_tools/grpo/construct_grpo.py \
    --data_path collect/manual/data/淘宝 \
    --out_path workspace/data/training_data/grpo_data \
    --factor 0.5 \
    --train_ratio 0.9 \
    --total_samples 817 \
    --seed 42
```

### 4.4 数据质检

```bash
python workspace/data_tools/grpo/grpo_data_check.py \
    --grpo_data_path workspace/data/training_data/grpo_data \
    --auto_fix
```

### 4.5 通过标准

- [ ] 两次运行 md5sum 完全一致（train 和 val 均相同）
- [ ] 数据质检 0 个问题
- [ ] `metadata.json` 中 `train_count + val_count == total_sampled`
- [ ] action 分布合理（click > 50%，input/done 均 > 10%）

---

## 五、阶段 3：Grounder 服务验证

**目标**：在正式训练前，独立验证 Grounder API 可用、响应格式正确。

### 5.1 Grounder 部署脚本

创建 `workspace/training/grpo/scripts/deploy_grounder.sh`：

```bash
#!/bin/bash
# 在 GPU 1 上部署 Grounder vLLM 服务
# 用法: bash workspace/training/grpo/scripts/deploy_grounder.sh

GROUNDER_MODEL="/scratch/youliang/models/grounder"
GROUNDER_PORT=8001

echo "Starting Grounder service on GPU 1, port $GROUNDER_PORT..."
echo "Model: $GROUNDER_MODEL"

CUDA_VISIBLE_DEVICES=1 conda run -n MobiMind vllm serve "$GROUNDER_MODEL" \
    --port $GROUNDER_PORT \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager

echo "Grounder service stopped."
```

启动（统一使用 tmux，保持长期运行）：

```bash
# 1) 新建 grounder 会话（后台）
tmux new -d -s grpo_grounder \
  "cd /home/agent/mobiAgent/MobiAgent && bash workspace/training/grpo/scripts/deploy_grounder.sh"

# 2) 查看 grounder 日志（可选）
tmux attach -t grpo_grounder
# 退出但不停止：Ctrl+b 然后按 d
```

结束 Grounder（训练结束后执行）：

```bash
tmux kill-session -t grpo_grounder
```

等待出现 `INFO:     Application startup complete.`

### 5.2 API 连通性快速验证

```bash
curl -s http://localhost:8001/v1/models | python3 -m json.tool
```

### 5.3 Grounder 端到端测试

创建并运行 `workspace/training/grpo/test_grounder_api.py`：

```python
"""
阶段 3 验证脚本：Grounder API 连通性与响应格式测试
用法: conda run -n grpo python workspace/training/grpo/test_grounder_api.py
"""
import base64, io, json, re, time
from pathlib import Path
from PIL import Image
from openai import OpenAI

GROUNDER_URL  = "http://localhost:8001/v1"
DATA_DIR      = Path("workspace/data/training_data/grpo_data")
TEST_IMG_PATH = next(DATA_DIR.glob("*.jpg"))
TEST_REASONING = "我需要点击页面上的搜索框"
TEST_ELEMENT   = "页面顶部的搜索输入框"

GROUNDER_PROMPT = (
    "Based on the screenshot, user's intent and the description of the target UI element, "
    "provide the bounding box of the element using **absolute coordinates**.\n"
    "User's intent: {reasoning}\n"
    "Target element's description: {description}\n"
    "Your output should be a JSON object with the following format:\n"
    '{{"bbox": [x1, y1, x2, y2]}}'
)

def img_to_b64(path):
    img = Image.open(path)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()

def parse_bbox(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = text.strip()
    m = re.search(r"```json\n(.*?)\n```", text, re.DOTALL)
    if m:
        text = m.group(1)
    return json.loads(text)

def main():
    print(f"Test image : {TEST_IMG_PATH}")
    client = OpenAI(api_key="0", base_url=GROUNDER_URL)
    b64 = img_to_b64(TEST_IMG_PATH)
    prompt = GROUNDER_PROMPT.format(
        reasoning=TEST_REASONING, description=TEST_ELEMENT
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model="",
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
        temperature=0,
    ).choices[0].message.content
    elapsed = time.time() - t0

    print(f"Raw response ({elapsed:.1f}s): {resp[:300]}")
    parsed = parse_bbox(resp)
    bbox   = parsed["bbox"]
    print(f"Parsed bbox : {bbox}")

    assert len(bbox) == 4,                                  "bbox 必须是 4 个元素"
    assert all(isinstance(v, (int, float)) for v in bbox),  "bbox 元素必须是数字"
    assert elapsed < 10,                                    f"响应超时（{elapsed:.1f}s > 10s）"
    print("\n✓ Grounder API 测试通过")

if __name__ == "__main__":
    main()
```

```bash
conda run -n grpo python workspace/training/grpo/test_grounder_api.py
```

### 5.4 通过标准

- [ ] `curl` 返回 models 列表，HTTP 200
- [ ] bbox 解析成功，是长度为 4 的数字列表
- [ ] 响应时间 < 10 秒
- [ ] 无报错或异常

---

## 六、阶段 4：Reward 函数开发与单测

**目标**：实现严格二值奖励函数，并通过离线单测验证正确性。

### 6.1 奖励规则（严格二值）

| 情况 | Reward |
|------|--------|
| JSON 解析失败 | 0.0 |
| action type 不匹配 | 0.0 |
| done / wait，type 匹配 | 1.0 |
| swipe，type 匹配 且 direction 完全一致（大小写不敏感）| 1.0 |
| swipe，type 匹配 但 direction 不一致 | 0.0 |
| input，type 匹配 且 text strip 后完全一致 | 1.0 |
| input，type 匹配 但 text 不一致 | 0.0 |
| click，type 匹配 且 grounder 成功 且中心点在 gt_bounds 内 | 1.0 |
| click，其他所有情况（grounder 失败/超时/target 为空/bounds 缺失）| 0.0 |

**无任何部分分，reward 只有 0.0 或 1.0。**

### 6.2 实现 grpo_reward_test.py

`workspace/data_tools/grpo/grpo_reward_test.py` 需实现为完整可执行测试，最少 20 条用例，分组如下：

```
分组 1：JSON 解析（4 条）
  - 合法 JSON -> 不报错
  - 截断 JSON -> reward=0
  - 含 <think> 标签 -> 剥离后正确解析
  - JSON 外包 markdown 代码块 -> 正确提取

分组 2：action type 不匹配（3 条）
  - gt=click, pred=swipe -> 0
  - gt=input, pred=done -> 0
  - gt=done, pred=click -> 0

分组 3：done / wait（2 条）
  - gt=done, pred=done -> 1
  - gt=wait, pred=wait -> 1

分组 4：swipe（3 条）
  - 方向一致（UP/UP）-> 1
  - 方向不一致（UP/DOWN）-> 0
  - 大小写不敏感（"up"/"UP"）-> 1

分组 5：input（3 条）
  - text 完全一致（strip 后）-> 1
  - text 不一致 -> 0
  - text 有多余空格（strip 后一致）-> 1

分组 6：click（5 条，使用 mock grounder）
  - 中心点在 bounds 内（mock 返回合法 bbox）-> 1
  - 中心点在 bounds 外（mock 返回合法 bbox）-> 0
  - grounder 返回 None（模拟失败）-> 0
  - target_element 为空 -> 0
  - gt_bounds 字段缺失 -> 0
```

脚本末尾打印：
```
总用例: N   通过: N   失败: 0
所有 reward 值: [0.0, 1.0, ...]
✓ 全部测试通过
```

默认使用 mock grounder（不依赖服务在线）。提供 `--with_grounder` 标志启用真实调用：

```bash
# 默认（mock 模式）
conda run -n grpo python workspace/data_tools/grpo/grpo_reward_test.py

# 真实 grounder（需要服务在线）
conda run -n grpo python workspace/data_tools/grpo/grpo_reward_test.py --with_grounder
```

### 6.3 通过标准

- [ ] 所有 20+ 测试用例通过
- [ ] reward 输出全部为 0.0 或 1.0，无其他值
- [ ] `--with_grounder` 模式下 click 完整路径走通
- [ ] 测试运行无报错、无异常退出

---

## 七、阶段 5：GRPO 训练代码开发

### 7.1 最终目录结构

```
workspace/training/grpo/
├── scripts/
│   ├── finetune_grpo.sh              # 主训练入口
│   └── deploy_grounder.sh            # Grounder 部署（阶段 3 已创建）
├── src/
│   ├── __init__.py
│   ├── constants.py                  # Grounder prompt 模板等常量
│   ├── params.py                     # 参数定义（三个 dataclass）
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── grpo_dataset.py           # GRPODataset + collator + 工厂函数
│   ├── train/
│   │   ├── __init__.py
│   │   ├── train_grpo.py             # 主训练逻辑
│   │   ├── grounder_client.py        # Grounder HTTP 同步客户端
│   │   └── reward_funcs.py           # 严格二值奖励函数
│   └── trainer/
│       ├── __init__.py
│       └── grpo_trainer.py           # QwenGRPOTrainer（扩展 trl.GRPOTrainer）
├── test_grounder_api.py              # 阶段 3 已创建
├── test_base_model_generation.py     # 阶段 6 前验证
└── output/                           # 本地调试输出
```

### 7.2 各模块设计规范

#### constants.py

```python
GROUNDER_PROMPT_TEMPLATE = (
    "Based on the screenshot, user's intent and the description of the target UI element, "
    "provide the bounding box of the element using **absolute coordinates**.\n"
    "User's intent: {reasoning}\n"
    "Target element's description: {description}\n"
    "Your output should be a JSON object with the following format:\n"
    '{{"bbox": [x1, y1, x2, y2]}}'
)

DEFAULT_GROUNDER_URL = "http://localhost:8001/v1/chat/completions"
```

#### params.py

三个 dataclass：

```python
@dataclass
class ModelArguments:
    model_id: str                    # 模型路径
    freeze_vision_tower: bool = True
    freeze_merger: bool = False      # merger 参与训练
    freeze_llm: bool = True          # 仅 LoRA 训练
    lora_enable: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_namespan_exclude: str = "['visual']"  # 排除视觉编码器

@dataclass
class DataArguments:
    data_path: str                   # train JSON 路径
    eval_path: str                   # val JSON 路径
    image_min_pixels: int = 200704   # 256*28*28
    image_max_pixels: int = 501760   # 640*28*28（比 SFT 小，节省 rollout 显存）

@dataclass
class GRPOArguments(GRPOConfig):     # 继承 trl.GRPOConfig
    grounder_url: str = DEFAULT_GROUNDER_URL
    grounder_timeout: float = 30.0
    log_reward_details: bool = True
```

#### grpo_dataset.py

**输入**（每条 JSON 样本）：
```json
{
  "instruction": "<image>\nYou are a phone-use AI agent...",
  "images": ["/absolute/path/to/image.jpg"],
  "gt_action": {"type": "click", "bounds": [...], ...}
}
```

**`__getitem__` 输出**：
```python
{
    "prompt": [                           # 会话格式，供 TRL 处理
        {
            "role": "user",
            "content": [
                {"type": "image"},        # 图像占位符
                {"type": "text", "text": instruction_without_image_token}
            ]
        }
    ],
    "images": [PIL.Image],               # 原始图像，供 reward function 传给 grounder
    "gt_action": {...},                  # 传递给 reward function
}
```

**关键细节**：
- `instruction` 开头的 `<image>\n` 需剥除，其余文本放入 text content
- `grpo_data_collator` 将 list of dicts 聚合为 `{key: [val, val, ...]}` 列表格式，不做 Tensor 转换
- `make_grpo_data_module(processor, model_id, data_args)` 返回 `dict(train_dataset, eval_dataset, data_collator)`

#### grounder_client.py

```python
def call_grounder(
    image,               # PIL.Image 或文件路径（str）
    reasoning: str,      # decider 输出的 reasoning
    target_element: str, # decider 预测的 target_element 参数
    url: str,
    timeout: float = 30.0,
) -> Optional[List[int]]:
    """
    同步调用 Grounder，返回 [x1, y1, x2, y2] 或 None。
    任何失败（HTTP 错误、解析失败、超时）均返回 None，不抛异常。
    调用方将 None 解释为 reward=0。
    """
```

实现要点：
- 使用 `openai.OpenAI` 客户端（GRPO 环境已安装）
- 图像转 base64 逻辑与 `test_grounder_api.py` 保持完全一致
- 响应解析逻辑（剥离 `<think>` 标签、markdown 代码块）与 `test_grounder_api.py` 保持完全一致
- 从模块内全局变量读取 url/timeout（由 `train_grpo.py` 在启动时通过环境变量设置）

#### reward_funcs.py

```python
def decider_reward(
    prompts: List,           # TRL 传入，本函数不使用
    completions: List[str],  # 模型生成的文本列表（batch_size * num_generations）
    gt_action: List[dict],   # 来自 dataset 的 gt 动作列表
    images: List,            # 来自 dataset 的 PIL 图像列表
    **kwargs
) -> List[float]:
```

实现要点：
- 完全遵循 6.1 节二值规则，无任何部分分
- grounder_url / grounder_timeout 从 `os.environ` 读取
- 第一个 batch 详细打印调试日志（completion 内容、gt_action、解析结果、reward 值）
- 每个 batch 结束后打印统计摘要：batch 大小、mean reward、非零数量、action type 分布

#### grpo_trainer.py

```python
class QwenGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, **kwargs):
        # 拦截 data_collator（TRL 默认不支持自定义 collator 注入）
        custom_collator = kwargs.pop("data_collator", None)
        super().__init__(*args, **kwargs)
        if custom_collator is not None:
            self.data_collator = custom_collator
```

额外功能：
- checkpoint 保存时额外保存 LoRA adapter weights 和 non_lora_state_dict.bin
- 训练日志中增加 4 个自定义指标（参见阶段 7.2）

#### train_grpo.py

主要流程：
1. 解析三组参数（`ModelArguments`, `DataArguments`, `GRPOArguments`）
2. 调用 SFT 的 monkey patch（见 7.3 节）
3. 加载模型：`Qwen2_5_VLForConditionalGeneration.from_pretrained`，bf16，flash_attention_2
4. 配置 freeze：freeze_vision_tower=True，freeze_llm=True，freeze_merger=False
5. 配置 LoRA：target_modules 覆盖 LLM 的 7 个投影层，通过 lora_namespan_exclude=["visual"] 排除视觉层
6. 加载 processor，配置 image_min/max_pixels
7. 设置环境变量 `GROUNDER_URL` 和 `GROUNDER_TIMEOUT`
8. 调用 `make_grpo_data_module` 获取 dataset 和 collator
9. 初始化 `QwenGRPOTrainer`，传入 `reward_funcs=[decider_reward]`
10. 启动训练

#### finetune_grpo.sh

```bash
#!/bin/bash
# MobiMind Decider grpo 训练脚本
# 前提：Grounder 服务已在 GPU 1 的 8001 端口运行
# 用法：bash workspace/training/grpo/scripts/finetune_grpo.sh

set -e

# GPU 分配：GPU 0 训练，GPU 1 已被 Grounder 占用
export CUDA_VISIBLE_DEVICES=0

# Grounder 配置
export GROUNDER_URL="http://localhost:8001/v1/chat/completions"
export GROUNDER_TIMEOUT="30"

# 模型与数据路径
BASE_MODEL="/scratch/youliang/models/decider_lora_2_merged"
DATA_PATH="workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json"
EVAL_PATH="workspace/data/training_data/grpo_data/mobimind_decider_grpo_val.json"
OUTPUT_DIR="/scratch/youliang/models/decider_grpo_1"
LOG_DIR="$OUTPUT_DIR/runs"

mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
echo "Training log: $TRAIN_LOG"

export PYTHONUNBUFFERED=1

# 切换到 grpo 训练目录，并加入 SFT src（用于 monkey patch）
cd "$(dirname "$0")/.."
export PYTHONPATH="src:../sft/src:$PYTHONPATH"

conda run -n grpo python -m src.train.train_grpo \
    --model_id               "$BASE_MODEL" \
    --data_path              "$DATA_PATH" \
    --eval_path              "$EVAL_PATH" \
    --output_dir             "$OUTPUT_DIR" \
    --freeze_vision_tower    True \
    --freeze_merger          False \
    --freeze_llm             True \
    --lora_enable            True \
    --lora_rank              64 \
    --lora_alpha             64 \
    --lora_dropout           0.05 \
    --lora_namespan_exclude  "['visual']" \
    --image_min_pixels       200704 \
    --image_max_pixels       501760 \
    --grounder_url           "$GROUNDER_URL" \
    --grounder_timeout       "$GROUNDER_TIMEOUT" \
    --per_device_train_batch_size 1 \
    --num_generations        4 \
    --gradient_accumulation_steps 4 \
    --max_new_tokens         512 \
    --max_prompt_length      2048 \
    --learning_rate          5e-6 \
    --lr_scheduler_type      cosine \
    --warmup_ratio           0.05 \
    --beta                   0.01 \
    --bf16                   True \
    --gradient_checkpointing True \
    --logging_steps          1 \
    --eval_strategy          steps \
    --eval_steps             50 \
    --save_strategy          steps \
    --save_steps             50 \
    --save_total_limit       5 \
    --max_steps              500 \
    --report_to              tensorboard \
    --logging_dir            "$LOG_DIR" \
    --remove_unused_columns  False \
    2>&1 | tee "$TRAIN_LOG"

echo "Training complete. Output: $OUTPUT_DIR"
```

### 7.3 关于 Monkey Patch

grpo 训练需要 SFT 已有的 Qwen2.5-VL monkey patch（混合模态 forward + 视觉编码器修复）。
处理方式：`finetune_grpo.sh` 已将 `../sft/src` 加入 PYTHONPATH，在 `train_grpo.py` 顶部直接 import：

```python
# train_grpo.py 顶部（调用 from_pretrained 之前）
from monkey_patch_forward import replace_qwen2_5_with_mixed_modality_forward
from monkey_patch_vision import replace_qwen2_5_vision

replace_qwen2_5_with_mixed_modality_forward()
replace_qwen2_5_vision()
```

这不是引用旧 grpo 代码，而是复用 SFT 生产级 patch。

### 7.4 代码完成后验证（不运行训练）

```bash
cd workspace/training/grpo
export PYTHONPATH="src:../sft/src:$PYTHONPATH"

conda run -n grpo python -c "
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '../sft/src')
from src.params import ModelArguments, DataArguments
from src.dataset import GRPODataset, grpo_data_collator, make_grpo_data_module
from src.train.reward_funcs import decider_reward
from src.train.grounder_client import call_grounder
from src.trainer import QwenGRPOTrainer
print('✓ 所有模块导入成功')
"

bash -n scripts/finetune_grpo.sh && echo '✓ shell 语法检查通过'
```

### 7.5 通过标准

- [ ] 所有模块可正常导入，无报错
- [ ] shell 语法检查通过

---

## 八、阶段 6：冒烟训练（50 步）

**目标**：验证"训练流程能从头走到尾"，不关注模型效果。 模型能输出可解析的结构化 JSON。JSON 里的 action 字段被正确识别为有效动作类型（click/input/done）。文本内容和动作在语义上是匹配的（例如“激活搜索框后输入关键词”对应 input）。
  这一步的目的是什么，这一步不是为了评估“模型最终效果好不好”，而是做GRPO开跑前的工程健康检查，验证三件事：
  模型能正常推理（权重、processor、多模态输入链路没坏）。
  输出格式符合你后续 reward/训练脚本的解析预期。动作空间是可用的（至少 click/input/done 能稳定产出）。先确认“管道通了”，再上 GRPO。

  成功了能说明什么，不能说明什么，能说明：
  decider_lora_2_merged 作为 GRPO 起点是可用的。你的数据样本 + 推理模板 + 解析代码三者是对齐的。后续 reward 计算不会因为“连 action 都解析不出来”而全崩。

  不能说明：不代表策略已经最优。不代表 reward 设计一定正确。不代表 GRPO 训练一定稳定收敛（那是下一阶段要验证的）
  如果不做这一步，会怎样。直接进 GRPO 后才发现 JSON 解析失败，reward 全是 0 或随机噪声。训练跑了很久才发现动作字段格式不匹配，白烧 GPU。问题定位困难：你分不清是模型问题、数据问题、还是 reward 代码问题。
  所以这一步的价值是：用很小成本（几条样本）提前排除高概率工程事故。

### 8.1 训练前模型验证

创建 `workspace/training/grpo/test_base_model_generation.py`，功能：
- 加载 `decider_lora_2_merged`
- 从 grpo_data train 中随机抽取 N 条样本
- 对每条样本做推理（max_new_tokens=256）
- 统计 JSON 解析成功率和 action type 分布

```bash
conda run -n grpo python workspace/training/grpo/test_base_model_generation.py \
    --model_path /scratch/youliang/models/decider_lora_2_merged \
    --data_path workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json \
    --num_samples 5
```

预期：JSON 解析成功率 >= 60%（SFT 模型基础能力验证）

### 8.2 启动冒烟训练

在 `finetune_grpo.sh` 中临时将 `--max_steps` 改为 `50`，然后（统一使用 tmux）：

```bash
# 确认 Grounder 在线
curl -s http://localhost:8001/v1/models

# 启动冒烟训练（后台）
tmux new -d -s grpo_smoke_train \
  "cd /home/agent/mobiAgent/MobiAgent && RUN_NAME=decider_grpo_smoke MAX_STEPS=50 bash workspace/training/grpo/scripts/finetune_grpo.sh"

# 查看训练日志（可选）
tmux attach -t grpo_smoke_train
# 退出但不停止：Ctrl+b 然后按 d

# 冒烟结束后关闭会话
tmux kill-session -t grpo_smoke_train
```

### 8.3 实时监控

```bash
tail -f /scratch/youliang/models/decider_grpo_1/runs/train_*.log \
    | grep -E "reward|loss|step|OOM|Error"
```

### 8.4 通过标准

- [ ] 无 OOM 错误（`CUDA out of memory`）
- [ ] 无 NaN loss
- [ ] reward/mean > 0（done/wait 等类型提供正向信号）
- [ ] checkpoint-25 和 checkpoint-50 目录均存在
- [ ] TensorBoard 日志可正常写入

---

## 九、阶段 7：Pilot 训练（500 步）

**目标**：验证"模型在学习，不只是能跑"。

### 9.1 配置

- 使用脚本默认 `--max_steps 500`
- 其他训练参数保持不变
- 使用最新路径约定：输出统一写入 `workspace/training/grpo/output/<run_name>`
- 本轮 pilot 建议：`run_name=decider_grpo_1`

执行命令（推荐，统一 tmux）：

```bash
# 1) 先启动 grounder（后台）
tmux new -d -s grpo_grounder \
  "cd /home/agent/mobiAgent/MobiAgent && bash workspace/training/grpo/scripts/deploy_grounder.sh"

# 2) 健康检查
curl -s http://localhost:8001/v1/models

# 3) 跑一次 API 测试
conda run -n grpo python workspace/training/grpo/test_grounder_api.py

# 4) 启动 Pilot 训练（后台）
tmux new -d -s grpo_pilot_train \
  "cd /home/agent/mobiAgent/MobiAgent && RUN_NAME=decider_grpo_1 MAX_STEPS=500 bash workspace/training/grpo/scripts/finetune_grpo.sh"

# 5) 查看训练日志（可选）
tmux attach -t grpo_pilot_train
# 退出但不停止：Ctrl+b 然后按 d

# 6) Pilot 结束后关闭 tmux
tmux kill-session -t grpo_pilot_train
tmux kill-session -t grpo_grounder
```

本阶段目录结构预期：

```text
workspace/training/grpo/output/decider_grpo_1/
├── checkpoint-*
├── best                 # 指向最佳 checkpoint 的别名（若无法软链则复制目录）
├── logs/
│   └── train_*.log
└── tensorboard/
    └── events.*
```

### 9.2 关键监控指标

| 指标 | 含义 | 预期趋势 |
|------|------|---------|
| `reward/mean` | batch 平均奖励 | 不塌陷（> 0.1），后期有上升 |
| `json_valid_rate` | JSON 可解析率 | 高位稳定（SFT 起点良好）|
| `action_type_acc` | action type 准确率 | 不下降 |
| `click_hit_rate` | click 中心点命中率 | 有上升趋势 |

以上 4 个指标由 `QwenGRPOTrainer` 的日志回调计算并写入 TensorBoard。

### 9.3 中途检查（step=100 时）

```bash
conda run -n grpo python workspace/training/grpo/test_base_model_generation.py \
    --model_path /home/agent/mobiAgent/MobiAgent/workspace/training/grpo/output/decider_grpo_1/checkpoint-100 \
    --data_path workspace/data/training_data/grpo_data/mobimind_decider_grpo_val.json \
    --num_samples 10
```

对比 step=0（SFT 起点）和 step=100 的 JSON 解析率、action type 准确率是否有变化。

### 9.4 通过标准（500 步结束时）

- [ ] reward/mean 相比前 50 步有提升或维持
- [ ] json_valid_rate >= 0.80
- [ ] action_type_acc >= 0.70
- [ ] 训练无异常中断
- [ ] `workspace/training/grpo/output/decider_grpo_1/best` 已生成

说明：`workspace/training/grpo/output/decider_grpo_1` 是训练过程目录。待阶段 7/8 验证通过后，再将最终可用模型归档到 `/scratch/youliang/models/<final_name>`。

### 遇到过的一个问题
这个报错是参数约束导致的，不是模型问题：

  - ValueError: global eval batch size (1*1) must be divisible by num_generations (4)
  - 你当时设置了 --per_device_eval_batch_size 1，而 num_generations=4，不满足 TRL 的硬性要求。

  我已经做了两层修复：

  1. 把 eval batch size 改回 4（满足可整除）

  - finetune_grpo.sh:64

  2. 修复了你之前在 eval 末尾遇到的 TRL 越界 bug（最后一个不满 batch）

  - 在自定义 trainer 里覆盖了 _get_per_token_logps_and_entropies，对最后 chunk 用 end=min(start+batch_size,total) 防越界
  - grpo_trainer.py:59

2.click保持二值rewards，不用IOU会怎么样，虽然看起来没问题，但是对模型的学习是不利的，IOU可以让模型知道要覆盖的正确点击范围

3.4条rollout完全一样，一起对一起错，导致训练练不动，改成6也没用。也许尝试改温度？

4.温度调到1.5直接开始放飞自我，韩文阿拉伯文都来了，输出的东西完全没有逻辑

---

## 十、阶段 8：正式训练

**触发条件**：阶段 7 全部通过标准达成。

### 10.1 配置

在 `finetune_grpo.sh` 中修改：

```bash
OUTPUT_DIR="/scratch/youliang/models/decider_grpo_1"
--max_steps    2000
--save_steps   200
--eval_steps   100
```

推荐从头开始（不在 Pilot checkpoint 续训），因为 Pilot 仅用于链路验证。

### 10.2 完成后评估

```bash
# 1) 确保 Grounder 服务运行（GPU 1，tmux 后台）
tmux new -d -s grpo_grounder \
  "cd /home/agent/mobiAgent/MobiAgent && bash workspace/training/grpo/scripts/deploy_grounder.sh"

# 2) 正式训练（tmux 后台）
tmux new -d -s grpo_formal_train \
  "cd /home/agent/mobiAgent/MobiAgent && RUN_NAME=decider_grpo_formal MAX_STEPS=2000 SAVE_STEPS=200 EVAL_STEPS=100 bash workspace/training/grpo/scripts/finetune_grpo.sh"

# 3) 部署 grpo 训练后的 decider（参考 SFT deploy 流程，配置好 config.json）

# 4) 跑 benchmark
python workspace/benchmark/runners/run_task_list.py \
    --save_raw_data_path workspace/data/raw_runs/grpo_eval \
    --service_ip localhost \
    --decider_port 8000 \
    --grounder_port 8001 \
    --planner_port 8002

# 5) 结构化评估
python workspace/benchmark/evaluators/batch_structural_evaluator.py \
    --batch-mode \
    --raw-data-path workspace/data/raw_runs/grpo_eval \
    --eval-result-path workspace/data/benchmark_results/grpo_eval \
    --workers 2

# 6) 正式训练结束后关闭 tmux
tmux kill-session -t grpo_formal_train
tmux kill-session -t grpo_grounder
```

---

## 十一、已知限制与后续改进

| 限制 | 影响 | 后续方向 |
|------|------|---------|
| 数据 split 为步骤级（非轨迹级）| val 指标略偏乐观 | 改为轨迹级 split |
| grounder 调用为同步串行 | click 密集 batch 每步多花 1-3s | 改为 async 批量调用 |
| 训练样本 735 条，数量较少 | RL 对样本量敏感 | Pilot 验证后扩充 |
| 无 curriculum | 所有难度混合训练 | 后续按任务复杂度分 stage |
| 无 self-evolution | 无闭环迭代 | 正式训练成功后加入 |

---

## 十二、快速检查清单（每阶段执行前确认）

```
[ ] Grounder 服务在线：curl http://localhost:8001/v1/models
[ ] 训练环境激活：conda activate grpo
[ ] GPU 分配正确：CUDA_VISIBLE_DEVICES=0
[ ] 数据路径存在：ls workspace/data/training_data/grpo_data/*.json
[ ] 模型路径存在：ls /scratch/youliang/models/decider_lora_2_merged/
[ ] 磁盘空间充足：df -h /scratch （需 > 50GB 空余）
```

---

## 十三、附录：关键参数速查

### 训练超参（Pilot 默认值）

| 参数 | 值 |
|------|----|
| `per_device_train_batch_size` | 1 |
| `num_generations` | 4 |
| `gradient_accumulation_steps` | 4 |
| `max_new_tokens` | 512 |
| `max_prompt_length` | 2048 |
| `learning_rate` | 5e-6 |
| `lr_scheduler_type` | cosine |
| `warmup_ratio` | 0.05 |
| `beta`（KL 系数）| 0.01 |
| `max_steps`（Pilot）| 500 |
| `max_steps`（正式）| 2000 |

### LoRA 配置

| 参数 | 值 |
|------|----|
| `lora_rank` | 64 |
| `lora_alpha` | 64 |
| `lora_dropout` | 0.05 |
| `target_modules` | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| `lora_namespan_exclude` | ['visual']（排除视觉编码器所有层）|

### 图像处理

| 参数 | 值 | 说明 |
|------|----|------|
| `image_min_pixels` | 200704（=256×28×28）| 与 SFT 一致 |
| `image_max_pixels` | 501760（=640×28×28）| 比 SFT 小，节省 rollout 显存 |

### 服务端口

| 服务 | GPU | 端口 | 启动命令 |
|------|-----|------|---------|
| Grounder（常驻）| GPU 1 | 8001 | `tmux new -d -s grpo_grounder "bash scripts/deploy_grounder.sh"` |
| grpo 训练 | GPU 0 | — | `tmux new -d -s <train_session> "bash scripts/finetune_grpo.sh"` |
| Decider 推理（评估用）| GPU 0 | 8000 | 参考 SFT deploy 流程 |
