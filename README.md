# MobiAgent Workspace

A reproducible engineering workspace for mobile-agent research, built on top of open-source MobiAgent/UI-TARS pipelines.

This repository is not only a run demo. It packages a full research loop under `workspace/`:

- data curation and quality gating
- SFT/GRPO dataset construction
- model training and deployment scripts
- structural benchmark and model comparison

## Why this repo matters

Most mobile-agent repos stop at “can run”. This repo focuses on “can reproduce, compare, and iterate”.

- End-to-end reproducibility: from raw traces to benchmark reports.
- Experiment efficiency: unified scripts for data, training, and evaluation.
- Research extensibility: easy ablation on Decider API/model and training strategy.

## What is added in `workspace/`

| Module | Key capability | Value |
| --- | --- | --- |
| `workspace/data_tools/annotated` | trace checking, single-step rewrite, unexpected-state data generation | improves data quality and coverage |
| `workspace/data_tools/sft` | multi-source SFT dataset construction + validation | standardizes SFT training inputs |
| `workspace/data_tools/grpo` | GRPO dataset construction + validation | supports RL-style policy optimization |
| `workspace/training/sft` | LoRA SFT training + merge/deploy scripts | shortens model iteration cycle |
| `workspace/training/grpo` | GRPO training scripts and training utilities | enables post-SFT policy refinement |
| `workspace/benchmark/runners` | batch task execution and API-switch runner | reproducible large-scale task rollout |
| `workspace/benchmark/evaluators` | structural/offline evaluator and summary export | comparable benchmark metrics |
| `workspace/data/*` | cached runs, datasets, benchmark artifacts | experiment traceability |

## Repository layout

```text
.
├── collect/                 # data collection pipeline
├── runner/                  # online agent runners
├── MobiFlow/                # structural task rules / evaluation assets
└── workspace/               # reproduction-focused engineering work
    ├── data_tools/
    ├── training/
    ├── benchmark/
    ├── docs/
    └── data/
```

## Reproduction scope

This project is designed for open-source reproduction and controlled comparison of mobile-agent systems:

- reproduce baseline behavior with the original pipeline
- swap Decider backends (`local` / `openai` / `gemini`) for ablation
- train and evaluate upgraded policies with identical task protocols

## Documentation

- Workspace guide: `workspace/docs/WORKSPACE_GUIDE.md`
- Benchmark configs: `workspace/benchmark/configs/`
- Prompt assets: `workspace/prompts/`

## Acknowledgements

- Upstream project: [IPADS-SAI/MobiAgent](https://github.com/IPADS-SAI/MobiAgent)
- This fork focuses on reproducible engineering, training, and benchmark workflows.
