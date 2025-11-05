# 测试数据组织与结果管理

## 目录结构

```
run_test_data/
├── mobiagent/          # 模型名称
│   ├── taobao/         # 应用名称
│   │   ├── type3/      # 任务类型
│   │   │   ├── 1/      # trace数据
│   │   │   ├── 2/
│   │   │   ├── 7/
│   │   │   └── results/     # 测试结果
│   │   │       ├── 1/
│   │   │       │   ├── summary.txt
│   │   │       │   ├── detailed.json
│   │   │       │   └── run.log
│   │   │       ├── 2/
│   │   │       └── 7/
│   │   └── type4/
│   └── bilibili/
├── gpt/
├── qwen/
└── gemini/
```

## 使用方法

### 1. 测试单个 trace

```bash
cd MobiFlow
python structural_test_runner.py task_configs/taobao.json type3:7 \
    --data-base ../run_test_data/mobiagent/taobao
```

### 2. 测试某个 type 下的所有 trace

```bash
python structural_test_runner.py task_configs/taobao.json type3 \
    --data-base ../run_test_data/mobiagent/taobao
```

## 结果文件说明

每个 trace 的测试结果保存在 `<type>/results/<trace_id>/` 目录下：

- **summary.txt**: 简洁的文本摘要
- **detailed.json**: 完整的JSON格式结果
- **run.log**: 测试执行日志

## 注意事项

- 结果文件会直接覆盖，请手动备份重要数据
- 批量测试时，终端会打印汇总信息，但不生成批次汇总文件
- 每个 trace 的结果独立保存，方便单独查看和管理
