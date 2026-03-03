# WORKSPACE GUIDE



## 一、数据集制作



**标注数据的后处理**`data_tools/annotated`

1. `workspace/data_tools/annotated/labeled_data_checker.py`：检查 `collect/manual/data` 下人工标注轨迹的完整性与一致性，重点核对 `actions.json` 和 `react.json` 的步骤对齐；可选清理 `parse.error`，也可按任务类型批量删除 `react.json` 以便重新标注数据，重新生成react.json
```bash
# 数据完整性检查
python workspace/data_tools/annotated/labeled_data_checker.py --data-path collect/manual/data

# 清理 parse.error
python workspace/data_tools/annotated/labeled_data_checker.py --data-path collect/manual/data --clean-errors

# 按任务类型删除 react.json（示例：livestream）
python workspace/data_tools/annotated/labeled_data_checker.py --data-path collect/manual/data --clean-react livestream
```

2. `workspace/data_tools/annotated/ss_data_generation.py`：把多步任务轨迹拆成单步样本，调用 Gemini 生成单步目标改写与 reasoning 变体使得reasoning描述更加具有通用性，重新生成8个随机不同的task目标描述。输出到 `workspace/data/training_data/ss_data`，用于增强 SFT 的单步决策训练数据。这一步需要调用gemini-2.5pro api进行改写增强，时间很久

```bash
python workspace/data_tools/annotated/ss_data_generation.py \
  --data_path collect/manual/data/淘宝 \
  --livestream_path collect/manual/data/淘宝/livestream \
  --output_path workspace/data/training_data/ss_data/decider \
  --gemini_api_key <API KEY> \
  --model_name gemini-2.5-pro
```

3. `workspace/data_tools/annotated/unexpected_data_generation.py`：通过 ADB 连续截取设备当前画面，构建意外状态图片数据集（`unexpected_data`），用于训练模型处理异常页面或计划外状态。
```bash
python workspace/data_tools/annotated/unexpected_data_generation.py \
  --output_dir workspace/data/training_data/unexpected_data
```



**SFT数据整合**`workspace/data_tools/sft`

1. `workspace/data_tools/sft/construct_sft.py`：将 `collect/manual/data` 的多步人工轨迹、`ss_data` 单步样本和 `unexpected_data` 意外状态图片合并，构建 SFT 训练/验证数据集，输出 `mobimind_decider_*.json`、`mobimind_grounder_*.json` 与 `metadata.json` 到 `workspace/data/training_data/sft_data`。
```bash
python workspace/data_tools/sft/construct_sft.py \
  --data_path collect/manual/data \
  --ss_data_path workspace/data/training_data/ss_data \
  --unexpected_img_path workspace/data/training_data/unexpected_data \
  --out_path workspace/data/training_data/sft_data \
  --factor 0.5 \
  --train_ratio 0.9
```

2. `workspace/data_tools/sft/sft_data_check.py`：对 SFT 数据集进行训练前质量门禁，检查样本字段、动作分布、重复样本、图片可读性与元信息一致性，并输出部分sft数据作为案例样本。
```bash
python workspace/data_tools/sft/sft_data_check.py \
  --sft_data_path workspace/data/training_data/sft_data \
  --no_check_images
```



**GRPO数据集**`workspace/data_tools/grpo`

1. `workspace/data_tools/grpo/construct_grpo.py`：从人工轨迹构建 GRPO 训练/验证数据，完成采样、history 拼接、图片缩放与拷贝，输出到 `workspace/data/training_data/grpo_data`。
```bash
python workspace/data_tools/grpo/construct_grpo.py \
  --data_path collect/manual/data/淘宝 \
  --out_path workspace/data/training_data/grpo_data \
  --factor 0.5 \
  --train_ratio 0.9 \
  --total_samples 1500
```

2. `workspace/data_tools/grpo/grpo_data_check.py`：检查 GRPO 数据格式与样本可用性（字段、动作、图片路径、边界框等），可选自动修复部分无效 click 样本。
```bash
python workspace/data_tools/grpo/grpo_data_check.py \
  --grpo_data_path workspace/data/training_data/grpo_data
  --auto_fix
```

3. `workspace/data_tools/grpo/grpo_reward_test.py`：GRPO 奖励函数测试占位脚本（当前为空文件，未实现可执行测试逻辑）。
```bash
python -m py_compile workspace/data_tools/grpo/grpo_reward_test.py
```



**数据集结果目录**`workspace/data/training_data`

训练数据产物（`ss_data` / `unexpected_data` / `sft_data` / `grpo_data`）



## 二、训练算法代码



### SFT

**SFT 主训练入口**`workspace/training/sft/scripts`

1. `finetune_lora_vision.sh`：SFT 主训练入口（LoRA 微调），读取 `workspace/data/training_data/sft_data` 的训练/验证集并启动 deepspeed 训练。
```bash
bash workspace/training/sft/scripts/finetune_lora_vision.sh
```

**SFT模型处理**`workspace/training/sft/deploy`

1. `workspace/training/sft/deploy/merge_lora.py`：将 SFT 训练得到的 LoRA Adapter 合并回基础模型，导出可独立部署的 merged 模型（读取 `deploy/config.json`）。
```bash
python workspace/training/sft/deploy/merge_lora.py
```



2. `workspace/training/sft/deploy/deploy_lora.py`：LoRA 部署脚本，支持“保存 LoRA 权重”与“vLLM 挂载 LoRA 部署”两种模式（读取 `deploy/config.json`）。

先保存训练结果权重：按照配置文件config.json定义，保存lora训练的输出文件夹`lora_source_path`到`models_dir/lora_save_name`的指定位置

```
python workspace/training/sft/deploy/deploy_lora.py --save
```

保存完权重之后，可以把output目录下的checkpoint删除，这个文件很占空间，因为里面含了基础模型的权重

然后再部署：先在config.json里面配置好基础模型`base_model_path`加lora模型`models_save_dir/lora_save_name`

自动使用vllm命令部署decider模型到8000端口

```
python workspace/training/sft/deploy/deploy_lora.py --deploy
```

**一定要把这里的config文件配置好再训练，这决定着lora文件的保存位置在哪里！！！**



### GRPO

`workspace/training/grpo/scripts`

1. `workspace/training/grpo/scripts/finetune_grpo.sh`：GRPO 训练入口脚本，读取 `workspace/data/training_data/grpo_data`，调用 `src.train.train_grpo` 进行强化训练。
```bash
bash workspace/training/grpo/scripts/finetune_grpo.sh
```

`workspace/training/grpo`

1. `workspace/training/grpo/test_base_model_generation.py`：在 GRPO 前验证基础模型生成能力，使用真实训练样本格式发请求，检查输出 JSON 是否完整、可解析且动作字段合理。
```bash
python workspace/training/grpo/test_base_model_generation.py \
  --api_url http://localhost:8000/v1/chat/completions \
  --data_path workspace/data/training_data/grpo_data/mobimind_decider_grpo_train.json
```

`workspace/training/grpo/archive`

1. `workspace/training/grpo/archive/finetune_grpo.sh`：历史归档版 GRPO 训练脚本，仅用于回看旧实现，不作为当前主训练入口。
```bash
bash workspace/training/grpo/archive/finetune_grpo.sh
```



## 三、benchmark 评估流程



**任务运行结果数据**`workspace/benchmark/runners`

1. `workspace/benchmark/runners/run_task_list.py`：批量执行 `workspace/benchmark/configs/task_list.json` 中任务，按模型名+时间戳输出原始运行轨迹到 `workspace/data/raw_runs`。

```bash
python workspace/benchmark/runners/run_task_list.py \
  --save_raw_data_path workspace/data/raw_runs/testruns \
  --service_ip localhost \
  --decider_port 8000 \
  --grounder_port 8001 \
  --planner_port 8002
```

2. `workspace/benchmark/runners/mobiagent_change_decider_api.py`：单任务/小规模任务调试入口，支持切换 decider API 类型（local / gemini / openai），用于联调 decider-grounder-planner 链路。

```bash
python workspace/benchmark/runners/mobiagent_change_decider_api.py \
  --service_ip localhost \
  --decider_port 8000 \
  --grounder_port 8001 \
  --planner_port 8002 \
  --decider_api_type gemini \
  --decider_model gemini-2.5-pro \
  --decider_api_key <API KEY>
```



**评估脚本**`workspace/benchmark/evaluators`

1. `workspace/benchmark/evaluators/batch_structural_evaluator.py`：结构化批量评估脚本，批量读取 `workspace/data/raw_runs/<model>/<timestamp>`，并将评估结果输出到 `workspace/data/benchmark_results/<model>/<timestamp>`。
```bash
python workspace/benchmark/evaluators/batch_structural_evaluator.py \
  --batch-mode \
  --raw-data-path workspace/data/raw_runs/testruns \
  --eval-result-path workspace/data/benchmark_results/testruns \
  --workers 2
```
