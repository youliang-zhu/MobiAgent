# GRPO 环境配置调研报告

**作者**: Claude (基于公开仓库调研)
**日期**: 2026-03-07
**适用项目**: MobiAgent GRPO 训练 (Qwen2.5-VL-7B Decider)

---

## 一、调研目标

本次调研旨在回答以下核心问题：

1. 哪些开源 GRPO 仓库适合作为我们的代码参考？
2. torch / transformers / trl / peft 这几个关键库之间的兼容版本是什么？
3. 上一版 GRPO 代码"倒在依赖"的根本原因是什么，如何规避？
4. 应该新建什么版本的 conda 环境？

---

## 二、调研范围 — 开源 GRPO 仓库

### 2.1 仓库列表与评级

| 仓库 | Stars 量级 | 与我们的相关度 | 评级 |
|------|-----------|--------------|------|
| [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) | 高 | 最高 — 原生支持 Qwen2.5-VL GRPO+LoRA，TRL 实现，有完整 env | ★★★★★ |
| [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) | 高 | 高 — 基于 HuggingFace open-r1，reward 设计有参考价值 | ★★★★☆ |
| [huggingface/trl 官方 grpo_vlm.py](https://github.com/huggingface/trl/blob/main/examples/scripts/grpo_vlm.py) | 极高 | 中 — API 最新但版本较激进 | ★★★☆☆ |
| [tulerfeng/Video-R1](https://github.com/tulerfeng/Video-R1) | 高 | 低 — 使用自定义 transformers fork，架构过于复杂 | ★★☆☆☆ |

---

## 三、核心依赖版本分析

### 3.1 关键库版本时间线

#### transformers
```
4.49.0  (2025-02-17)  ← Qwen2.5-VL 首次进入 transformers
4.55.x  (2025-08-xx)
4.57.1  (2025-10-14)  ← 2U1 仓库使用，证明可用
4.57.2  (2025-11-24)
4.57.3  (2025-11-25)  ← 原计划版本
4.57.4  (2026-01-13)
4.57.6  (2026-01-16)  ← 最后一个 4.x 稳定版
5.0.0   (2026-01-26)  ← 重大 breaking change（VLM 架构重构）
5.3.0   (2026-03-04)  ← 当前最新
```

#### trl
```
0.17.0  (2025-xx-xx)  ← VLM-R1 使用（较老 API）
0.25.0  (2025-11-05)  ← 2U1 仓库使用，证明可用
0.25.1  (2025-11-12)
0.26.0  (2025-12-09)  ← 原计划版本，增加 agent+tool 特性
0.26.2  (2025-12-18)
0.27.0  (2026-01-16)  ← 增加 async rewards（我们不需要）
0.29.0  (2026-02-25)  ← 当前最新
```

#### peft
```
0.15.2  (2025-04-15)  ← 2U1 仓库使用，证明与 trl 0.25 完全兼容
0.16.0  (2025-07-03)  ← API 有变动，与 trl 兼容性未充分验证
0.17.1  (2025-08-21)
0.18.0  (2025-11-13)  ← 原计划版本，与 trl 0.25 兼容性未知
0.18.1  (2026-01-09)  ← 当前最新
```

#### 关键辅助库
```
accelerate  1.10.1   (2U1 使用)
deepspeed   0.17.5   (2U1 使用)
tokenizers  0.22.0   (2U1 使用，关键 pin)
liger-kernel 0.6.4   (2U1 使用)
```

### 3.2 各仓库实际使用的版本对比

| 依赖 | 原计划 | 2U1（实测可用） | VLM-R1 | 差异说明 |
|------|--------|----------------|---------|---------|
| Python | 3.10 | **3.11** | 3.10 | 建议改 3.11 |
| torch | 2.8.0+cu128 | **2.8.0** | >=2.5.1 | 一致 ✅ |
| CUDA | 12.8 | 12.8 | 12.x | 一致 ✅ |
| transformers | 4.57.3 | **4.57.1** | 4.49.0 | 小版本差异，均安全 |
| trl | 0.26.0 | **0.25.0** | 0.17.0 | 见风险分析 |
| peft | **0.18.0** | 0.15.2 | 未锁定 | **高风险，见下节** |
| accelerate | 未指定 | **1.10.1** | >=1.2.1 | 需明确指定 |
| deepspeed | 未指定 | **0.17.5** | 0.15.4 | 需明确指定 |
| liger-kernel | 未安装 | 0.6.4 | 0.5.2 | 见风险分析 |
| tokenizers | 未指定 | **0.22.0** | 未锁定 | **需明确锁定** |

---

## 四、风险分析 — 上一版 GRPO 代码失败根因推断

### 风险 A：peft 版本不兼容（高概率根因）

**问题**: 原计划 `peft==0.18.0` 是 2025-11-13 发布的，而 2U1 仓库明确使用 `peft==0.15.2`（2025-04-15），即便该仓库有 `torch==2.8.0`（2026 年），也故意保持 peft 在旧版。

**原因推断**: peft 0.16.0 起（2025-07-03）对 `LoraConfig`、`get_peft_model` 等 API 有破坏性变更，与 TRL 的 `GRPOConfig` / `get_peft_config` 接口产生冲突，导致 LoRA 挂载失败或训练时报错。

**规避方法**: 使用 `peft==0.15.2`，与 `trl==0.25.0` 的组合已经过 2U1 在实际 Qwen2.5-VL GRPO 训练中验证。

### 风险 B：transformers 5.x 导致的 `language_model` AttributeError（已知 bug）

**问题**: TRL issue [#4601](https://github.com/huggingface/trl/issues/4601)（2025-11-28 报告）：

```
AttributeError: 'Qwen2_5_VLForConditionalGeneration' object has no attribute 'language_model'
```

**根因**: transformers 5.x 的某个 PR 重构了 VLM 的内部架构，将 `model.language_model` 属性移除。liger-kernel 的 Qwen2.5-VL monkey-patch 依赖这个属性，因此在 transformers 5.x + liger-kernel 的组合下会崩溃。

**触发条件**:
- `transformers >= 5.0.0` + `liger-kernel` 已安装
- 且 TRL GRPOTrainer 开启了 `use_liger_kernel=True`

**我们的情况**: 只要保持 `transformers==4.57.x`（4.x 系列），此 bug **不会触发**。transformers 5.x 截至 2026-03 仍有大量兼容性问题，强烈不建议用于 GRPO 训练。

### 风险 C：tokenizers 版本与 Qwen2.5-VL 分词器不匹配

**问题**: Qwen2.5-VL 使用 tiktoken 风格的词表，tokenizers 的主版本差异（0.20 vs 0.21 vs 0.22）可能导致 fast tokenizer 行为不一致，引发 input_ids 长度计算错误，进而导致 `max_prompt_length` 越界或 attention mask 对齐问题。

**规避方法**: 明确锁定 `tokenizers==0.22.0`，与 2U1 的实测环境一致。

### 风险 D：trl 0.26.0 vs 0.25.0 的细微差异

**问题**: `trl 0.26.0`（2025-12-09）主要增加了 GRPO agent training with tools、ScaleRL、SAPO 等特性，这些与我们的场景无关。但版本升级可能改变 `GRPOConfig` 的某些默认参数行为（如 `beta`、`loss_type` 的默认值），导致训练行为静默改变。

**规避方法**: 使用经过实测的 `trl==0.25.0` 或 `trl==0.25.1`。0.25.1 仅是 bug fix，更稳健。

---

## 五、推荐仓库分析

### 5.1 首选参考：2U1/Qwen-VL-Series-Finetune

**推荐理由**:

1. **原生支持 Qwen2.5-VL GRPO + LoRA**，不需要自定义 monkey-patch 或 custom trainer
2. **环境有完整 freeze 记录**（`environment.yaml`），可逐字照抄依赖版本
3. **同样使用标准 TRL GRPOTrainer**，无自定义 fork
4. **同样使用 CUDA 12.8 + torch 2.8.0**，硬件路径完全一致
5. **无需 vLLM 做 rollout**（vLLM 仅在需要加速生成时可选），与我们的单卡计划匹配

**应参考的内容**:
- `environment.yaml` 中的精确依赖版本（直接复用）
- GRPO dataset 格式（prompt + image 的组织方式）
- GRPOTrainer 的初始化参数（GRPOConfig 字段）
- LoRA 应用到 Qwen2.5-VL 的方式（target_modules 的选择）

**不应复制的内容**:
- Reward function（他们的 reward 针对的是一般 VQA/REC 任务，我们是 mobile agent）
- 数据加载逻辑（我们有自己的 grpo_dataset.py）

### 5.2 次要参考：VLM-R1 (om-ai-lab)

**参考价值**: Reward function 设计模式，尤其是如何结构化 `reward_funcs` 列表传给 `GRPOTrainer`，以及如何 parse 模型输出的 JSON 并计算奖励。

**不推荐照搬的原因**:
- 使用旧版 `trl==0.17.0` + `transformers==4.49.0`（API 与 0.25.0 有差异）
- 依赖 `vllm==0.6.6.post1` 做 rollout 生成，单卡场景下额外占用显存
- 部分代码来自 huggingface/open-r1，针对文本推理任务设计

### 5.3 不推荐参考：Video-R1 (tulerfeng)

**原因**: 使用自定义修改的 transformers（非官方 fork），无法直接 `pip install transformers` 获得相同行为，维护成本高，与 MobiAgent 的场景也相差较大。

---

## 六、推荐的环境配置方案

### 6.1 核心原则

1. **锚定 2U1 的实测版本组合**，这是当前最权威的 Qwen2.5-VL + GRPO 成功案例
2. **不用 transformers 5.x**，坚守 4.57.x 系列
3. **明确锁定每个关键库版本**，避免 `pip install trl` 拉取最新版引入的隐患
4. **不启用 liger-kernel**（安装了也不在 trainer 里开启），消除 issue #4601 的风险

### 6.2 推荐版本矩阵（最终版）

| 组件 | 推荐版本 | 来源依据 |
|------|---------|---------|
| Python | 3.11 | 2U1 实测 |
| CUDA | 12.8 | 服务器已有 |
| torch | 2.8.0+cu128 | 2U1 + 计划一致 |
| torchvision | 0.23.0 | 2U1 实测 |
| transformers | **4.57.1** | 2U1 实测（最稳定 4.x） |
| tokenizers | **0.22.0** | 2U1 实测（关键锁定） |
| trl | **0.25.1** | 2U1 用 0.25.0，取 patch 版 0.25.1 更稳 |
| peft | **0.15.2** | 2U1 实测（不升级！） |
| accelerate | **1.10.1** | 2U1 实测 |
| deepspeed | **0.17.5** | 2U1 实测 |
| datasets | >=3.2.0 | VLM-R1 要求 |
| qwen-vl-utils | latest | Qwen 官方工具包 |
| openai | latest | Grounder 客户端 |
| ujson | latest | JSON 解析加速 |
| tensorboard | latest | 训练监控 |
| flash-attn | latest（最后装） | 需在其他包之后安装 |

> **注意**: liger-kernel **不安装**，彻底规避风险。

### 6.3 完整安装脚本

```bash
#!/bin/bash
# 新建 grpo conda 环境（Qwen2.5-VL GRPO 训练专用）
# 基于 2U1/Qwen-VL-Series-Finetune 的实测版本组合

set -e

CONDA_ENV_NAME="grpo"

# Step 1: 清理旧环境（如有）
conda env remove -n $CONDA_ENV_NAME --yes 2>/dev/null || true

# Step 2: 新建环境
conda create -n $CONDA_ENV_NAME python=3.11 -y
conda activate $CONDA_ENV_NAME

# Step 3: 安装 PyTorch（CUDA 12.8）
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 4: 安装核心训练依赖（严格锁版本）
pip install \
    transformers==4.57.1 \
    tokenizers==0.22.0 \
    trl==0.25.1 \
    peft==0.15.2 \
    accelerate==1.10.1 \
    deepspeed==0.17.5

# Step 5: 安装数据/模型加载依赖
pip install \
    datasets>=3.2.0 \
    qwen-vl-utils \
    openai \
    ujson \
    tensorboard \
    tensorboardx \
    pillow \
    numpy \
    safetensors

# Step 6: 安装 flash-attn（最后安装，需在其他包就位后编译）
pip install flash-attn --no-build-isolation

# Step 7: 验证关键版本
python -c "
import torch, transformers, trl, peft, accelerate
print(f'torch:          {torch.__version__}')
print(f'transformers:   {transformers.__version__}')
print(f'trl:            {trl.__version__}')
print(f'peft:           {peft.__version__}')
print(f'accelerate:     {accelerate.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count:      {torch.cuda.device_count()}')
"

# Step 8: 验证 Qwen2.5-VL 可加载
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print('Qwen2.5-VL import OK')
"

echo "Environment setup complete: $CONDA_ENV_NAME"
```

### 6.4 版本检查脚本（安装后验证）

```python
# workspace/training/grpo/scripts/check_env.py
"""安装后运行此脚本，确认环境满足所有要求。"""

REQUIRED = {
    "torch": "2.8.0",
    "transformers": "4.57.1",
    "tokenizers": "0.22.0",
    "trl": "0.25.1",
    "peft": "0.15.2",
    "accelerate": "1.10.1",
}

import sys

def check():
    all_ok = True
    for pkg, expected in REQUIRED.items():
        try:
            mod = __import__(pkg.replace("-", "_"))
            actual = mod.__version__
            ok = actual == expected
            status = "OK" if ok else "MISMATCH"
            print(f"  {pkg:20s} expected={expected:10s} actual={actual:10s} [{status}]")
            if not ok:
                all_ok = False
        except ImportError:
            print(f"  {pkg:20s} NOT INSTALLED")
            all_ok = False

    # Check CUDA
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"\n  CUDA available:  {cuda_ok}")
    if cuda_ok:
        print(f"  GPU count:       {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory // (1024**3)
            print(f"  GPU {i}:           {name} ({mem}GB)")

    # Check liger-kernel is NOT installed (we don't want it)
    try:
        import liger_kernel
        print(f"\n  WARNING: liger-kernel is installed ({liger_kernel.__version__})")
        print(f"           Make sure NOT to use use_liger_kernel=True in GRPOTrainer")
    except ImportError:
        print(f"\n  liger-kernel:    not installed (correct)")

    # Check transformers < 5.0 (critical)
    import transformers
    from packaging import version
    if version.parse(transformers.__version__) >= version.parse("5.0.0"):
        print(f"\n  CRITICAL: transformers {transformers.__version__} >= 5.0.0 !")
        print(f"           This will break Qwen2.5-VL GRPO training. Downgrade to 4.57.1")
        all_ok = False

    print(f"\n{'All checks passed' if all_ok else 'Some checks FAILED'}")
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    check()
```

---

## 七、关键 API 差异说明（trl 0.17 → 0.25）

上一版 GRPO 代码可能用的是 `trl==0.17.0` 风格的 API（参考 VLM-R1），但我们要用 `trl==0.25.x`，两者有以下重要差异：

| 特性 | trl 0.17.x (旧) | trl 0.25.x (新/我们用) |
|------|----------------|----------------------|
| 配置类 | `GRPOConfig` 参数较少 | `GRPOConfig` 更完整，有 `max_prompt_length`、`num_generations` 等 |
| Reward 函数签名 | `reward_fn(completions, ...)` | `reward_fn(prompts, completions, **kwargs)` |
| VLM 支持 | 需要大量自定义 | 原生支持 image input |
| LoRA 集成 | 通过 `model_init_kwargs` | 直接传 `peft_config` 到 trainer |
| 多 reward 函数 | 列表传入 | 列表传入（相同） |

因此，我们的 `reward_funcs.py` 应遵循 trl 0.25.x 的函数签名，具体参考 2U1 的实现。

---

## 八、结论与行动建议

### 结论

1. **必须使用 2U1/Qwen-VL-Series-Finetune 的依赖版本组合**（Python 3.11, torch 2.8.0, transformers 4.57.1, trl 0.25.1, peft 0.15.2）

2. **原计划的两处错误**:
   - `peft==0.18.0` → **改为 `peft==0.15.2`**（差异半年，API 有破坏性变更）
   - `trl==0.26.0` → **改为 `trl==0.25.1`**（0.26 未在 Qwen2.5-VL 场景下实测）

3. **transformers 严格限制在 4.57.x**，绝对不能升到 5.x

4. **不安装 liger-kernel**，消除 issue #4601 的 Qwen2.5-VL attribute error 风险

5. **必须锁定 `tokenizers==0.22.0`**，防止 Qwen2.5-VL 分词器行为漂移

### 行动步骤（Stage 1 执行顺序）

```
1. conda env remove -n MobiMindGRPO --yes      # 删除旧环境
2. conda create -n grpo python=3.11 -y          # 新建 grpo 环境
3. 执行 6.3 节的完整安装脚本
4. 运行 check_env.py 验证所有版本
5. python -c "from transformers import Qwen2_5_VLForConditionalGeneration"  # 快速冒烟
```

### 代码仓库参考优先级

```
1. 2U1/Qwen-VL-Series-Finetune  ← 环境配置 + GRPOTrainer 集成方式
2. VLM-R1 (om-ai-lab)           ← Reward 函数设计模式（仅参考逻辑，不照搬 API）
3. TRL 官方文档/grpo_vlm.py     ← GRPOConfig 参数含义参考
```

---

## 附录：本次调研参考的关键链接

- [2U1/Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) — 主参考仓库
- [om-ai-lab/VLM-R1](https://github.com/om-ai-lab/VLM-R1) — reward 逻辑参考
- [TRL issue #4601: language_model AttributeError](https://github.com/huggingface/trl/issues/4601) — 核心风险来源
- [TRL GRPOTrainer 文档](https://huggingface.co/docs/trl/main/grpo_trainer) — 官方 API 参考
- [transformers releases](https://github.com/huggingface/transformers/releases) — 版本时间线
- [trl PyPI](https://pypi.org/project/trl/) — 版本发布记录
- [HF Cookbook: VLM GRPO with TRL](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_grpo_trl) — 官方示例教程
