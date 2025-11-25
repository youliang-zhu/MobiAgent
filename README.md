# 📱 MobiAgent/UI-TARS 移动端代理项目运行指南

本项目旨在研究和复现移动端多模态大语言模型（MLLM）在手机上的推理和操作能力，采用 ReAct (Reasoning + Acting) 架构。

## 🚀 一、环境与基本配置

### 1. 设备连接（ADB over Wi-Fi）

项目运行依赖于 ADB 连接到手机设备。

| 步骤 | 命令 | 说明 |
| :--- | :--- | :--- |
| **主机/Windows** | \`usbipd list\` | 查看手机的 \`BUS ID\`。 |
| **主机/Windows** | \`usbipd.exe attach --busid <BUS ID> --wsl\` | 将手机设备绑定到 WSL。 |
| **WSL/Linux** | \`adb devices\` | 确认手机在 WSL 中连接正常。 |
| **WSL/Linux** | \`adb tcpip 5555\` | 设置手机网络端口。 |
| **服务器/CMD** | \`adb connect \$PHONE\_IP:5555\` | 从服务器连接到手机。 |

### 2. 模型服务部署（vLLM）

所有运行流程的基础是部署 Decider, Grounder, Planner 三个模型服务。请在**三个不同的命令行窗口**中运行以下命令，并保持后台响应。

\`\`\`bash
# 窗口 1: Decider 模型 (Port 8000)
CUDA_VISIBLE_DEVICES=0 vllm serve ~/mobiAgent/models/decider \
    --port 8000 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager

# 窗口 2: Grounder 模型 (Port 8001)
CUDA_VISIBLE_DEVICES=1 vllm serve ~/mobiAgent/models/grounder \
    --port 8001 \
    --dtype float16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.63 \
    --enforce-eager

# 窗口 3: Planner 模型 (Port 8002) - 使用张量并行
CUDA_VISIBLE_DEVICES=0,1 vllm serve ~/mobiAgent/models/planner \
    --port 8002 \
    --tensor-parallel-size 2 \
    --max-model-len 10240 \
    --dtype float16 \
    --gpu-memory-utilization 0.35 \
    --enforce-eager
\`\`\`

### 3. 停止模型服务

\`\`\`bash
pkill -f "vllm serve"
\`\`\`

---

## 💻 二、项目主流程运行脚本

### 1. MobiAgent 主程序运行（核心）

执行 \`task.json\` 中定义的任务。

\`\`\`bash
cd ~/mobiAgent/MobiAgent
python -m runner.mobiagent.mobiagent \
    --service_ip localhost \
    --decider_port 8000 \
    --grounder_port 8001 \
    --planner_port 8002
\`\`\`

### 2. UI-TARS Demo 运行

\`\`\`bash
cd ~/mobiAgent/MobiAgent/runner/UI-TARS-agent
python quick_start.py
\`\`\`

### 3. ⭐ 【手动编写】使用外部 API 替换 Decider 模型

这个脚本允许你将 Decider 替换为外部 API 模型（如 GPT-4o-mini 或 Gemini-2.5-Flash）进行测试，以便评估通用模型的性能。

> **文件路径:** \`runner/mobiagent/mobiagent_change_decider_api.py\`

\`\`\`bash
python runner/mobiagent/mobiagent_change_decider_api.py \
    --service_ip localhost \
    --grounder_port 8001 \
    --planner_port 8002 \
    --decider_api_type openai \
    --decider_model "gpt-4o-mini" \
    --decider_api_key "<YOUR_API_KEY>"
\`\`\`

---

## 🛠️ 三、测试与评估脚本（Benchmark）

### 1. ⭐ 【手动编写】 Grounder 模型单独调试

此脚本用于测试 Grounder 的目标识别能力，将 Decider 输出的自然语言描述（\`target\_element\`）转化为屏幕坐标。

> **文件路径:** \`runner/mobiagent/test_grounder\` (Python 模块)

\`\`\`bash
cd ~/mobiAgent/MobiAgent
python -m runner.mobiagent.test_grounder \
    --service_ip localhost \
    --grounder_port 8001
\`\`\`

### 2. ⭐ 【手动编写】 批量任务执行脚本（Benchmark 运行）

此脚本用于批量运行 \`task\_list.json\` 中定义的所有任务，用于生成模型性能基准测试结果。

> **文件路径:** \`runner/mobiagent/run_task_list.py\`

\`\`\`bash
cd ~/mobiAgent/MobiAgent/runner/mobiagent
python run_task_list.py --model \$模型结果文件夹名称\$
\`\`\`

### 3. ⭐ 【手动编写】 结构化测试评估脚本（DAG 框架）

该脚本用于基于 DAG 框架的结构化评估。

> **文件路径:** \`MobiFlow/structural_test_runner.py\`

#### A. 单独测试（少量任务）

测试单个 trace 或某个 type 下的所有 trace。

\`\`\`bash
# 测试单个 trace (例如 type3:1)
cd /home/agent/mobiAgent/MobiAgent/MobiFlow
python structural_test_runner.py task_configs/taobao.json type3:1 \
    --data-base ../run_test_data/mobiagent/taobao
    
# 测试某个 type 下的所有 trace (例如 type3)
python structural_test_runner.py task_configs/taobao.json type3 \
    --data-base ../run_test_data/mobiagent/taobao
\`\`\`

#### B. 批量评估模式

一次性运行所有 App 和所有 Type 的任务，生成 Benchmark 结果。

\`\`\`bash
cd /home/agent/mobiAgent/MobiAgent/MobiFlow
python structural_test_runner.py --batch-mode \
    --timestamp \$时间\$ \
    --model \$模型结果文件夹名称\$ \
    --workers 2
\`\`\`

---

## 📚 四、数据集构建脚本

### 1. 手动采集服务器

启动数据收集服务器，用于人工采集原始截图和 \`action.json\`。

\`\`\`bash
cd ~/mobiAgent/MobiAgent
python -m collect.manual.server
\`\`\`

### 2. 自动采集脚本

调用外部 API 自动生成轨迹数据。

\`\`\`bash
# 示例：使用 Gemini API 进行自动采集
python -m collect.auto.server \
    --model "gemini-2.5-flash" \
    --api_type gemini \
    --max-steps 20
\`\`\`

### 3. 数据标注脚本（生成 React.json）

利用视觉模型生成推理信息 (\`react.json\`)，用于 SFT 训练。

\`\`\`bash
python -m collect.annotate_without_omniparser \
    --data_path collect/manual/data \
    --model qwen-vl-max \
    --api_type compatible \
    --api_key <YOUR_API_KEY>
\`\`\`

### 4. SFT 数据集构建脚本

从 \`actions.json\` 和 \`react.json\` 构建 Decider 和 Grounder 的 SFT 训练集。

\`\`\`bash
python -m collect.construct_sft \
    --data_path ./data/淘宝/点击搜索栏 \
    --ss_data_path ./ss_data_not_exist \
    --out_path ./sft_output \
    --factor 0.5 \
    --train_ratio 0.9
\`\`\`