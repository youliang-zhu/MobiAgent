#!/usr/bin/env python3
"""
结构化测试执行器 - 按 type/trace 组织结果目录
将测试结果保存到 run_test_data/<model>/<app>/<type>/results/<trace_id>/ 目录结构中

使用方法：
- 测试单个trace:
  python structural_test_runner.py task_configs/taobao.json type3:7 \\
      --data-base run_test_data/mobiagent/taobao

- 测试某个type下的所有trace:
  python structural_test_runner.py task_configs/taobao.json type3 \\
      --data-base run_test_data/mobiagent/taobao
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, asdict

# 导入原有的 UniversalTestRunner 核心功能
from universal_test_runner import UniversalTestRunner, TestResult, TestSummary
from avdag.logger import configure_logging


class StructuralTestRunner:
    """结构化测试执行器 - 按目录结构组织结果"""
    
    def __init__(self, config_file: str, data_base_dir: str):
        """初始化测试执行器
        
        Args:
            config_file: 任务配置文件路径 (如 task_configs/taobao.json)
            data_base_dir: 数据基础目录 (如 run_test_data/mobiagent/taobao)
        """
        self.config_file = config_file
        self.data_base_dir = os.path.abspath(data_base_dir)
        
        # 加载配置
        with open(config_file, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 临时修改配置中的 data_base_dir 为实际路径
        self.original_data_base = self.config.get('data_base_dir')
        self.config['data_base_dir'] = self.data_base_dir
        
        # 创建基础的 UniversalTestRunner 实例（用于复用测试逻辑）
        self.base_runner = UniversalTestRunner(config_file)
        self.base_runner.config['data_base_dir'] = self.data_base_dir
        self.base_runner.base_dir = os.getcwd()
        
        print(f"=== 结构化测试执行器 ===")
        print(f"任务名称: {self.config['task_name']}")
        print(f"数据目录: {self.data_base_dir}")
    
    def _get_results_dir(self, task_type: str, trace_id: Union[str, int]) -> str:
        """获取指定 trace 的结果目录路径
        
        Returns:
            结果目录的绝对路径，格式: <data_base_dir>/<type>/results/<trace_id>/
        """
        # 处理 trace_id 格式（可能是 "type3/7" 或 7）
        if isinstance(trace_id, str) and '/' in trace_id:
            trace_num = trace_id.split('/')[-1]
        else:
            trace_num = str(trace_id)
        
        # 构建路径: data_base_dir/type3/results/7/
        results_dir = os.path.join(self.data_base_dir, task_type, 'results', trace_num)
        return results_dir
    
    def _save_trace_result(self, result: TestResult, results_dir: str):
        """保存单个 trace 的结果到指定目录
        
        Args:
            result: 测试结果
            results_dir: 结果保存目录
        """
        # 创建目录
        os.makedirs(results_dir, exist_ok=True)
        
        # 保存 summary.txt
        summary_file = os.path.join(results_dir, 'summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"Trace {result.trace_id} 测试结果\n")
            f.write(f"{'='*60}\n")
            f.write(f"任务类型: {result.task_type}\n")
            f.write(f"任务名称: {result.task_name}\n")
            f.write(f"测试结果: {'✓ 成功' if result.success else '✗ 失败'}\n")
            f.write(f"得分: {result.score}\n")
            f.write(f"匹配节点: {', '.join(result.matched_nodes)}\n")
            f.write(f"执行时间: {result.execution_time:.2f}秒\n")
            if result.reason:
                f.write(f"\n详细原因:\n{result.reason}\n")
            if result.error_message:
                f.write(f"\n错误信息:\n{result.error_message}\n")
        
        # 保存 detailed.json
        detailed_file = os.path.join(results_dir, 'detailed.json')
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        
        print(f"✓ 结果已保存: {results_dir}")
    
    def test_single_trace(self, task_type: str, trace_id: Union[str, int]) -> TestResult:
        """测试单个 trace
        
        Args:
            task_type: 任务类型 (如 "type3")
            trace_id: trace ID (如 7 或 "type3/7")
        
        Returns:
            TestResult: 测试结果
        """
        # 确定结果目录
        results_dir = self._get_results_dir(task_type, trace_id)
        
        # 配置日志输出到结果目录
        log_file = os.path.join(results_dir, 'run.log')
        os.makedirs(results_dir, exist_ok=True)
        
        log_config = self.config.get('logging', {})
        configure_logging(
            level=log_config.get('level', 'DEBUG'),
            use_colors=False,  # 文件日志不使用颜色
            show_time=True,
            show_module=True,
            output_file=log_file
        )
        
        print(f"\n{'='*60}")
        print(f"测试 trace {trace_id} [{task_type}]")
        print(f"结果目录: {results_dir}")
        print(f"{'='*60}")
        
        # 调用基础 runner 的测试方法
        result = self.base_runner.test_single_trace(task_type, trace_id)
        
        # 保存结果到结构化目录
        self._save_trace_result(result, results_dir)
        
        return result
    
    def test_all_traces_in_type(self, task_type: str) -> List[TestResult]:
        """测试指定 type 下的所有 trace
        
        Args:
            task_type: 任务类型 (如 "type3")
        
        Returns:
            List[TestResult]: 所有测试结果列表
        """
        if task_type not in self.config['task_types']:
            print(f"错误: 任务类型 '{task_type}' 未在配置中定义")
            return []
        
        task_config = self.config['task_types'][task_type]
        task_name = task_config['name']
        
        # 获取该类型的所有 trace
        traces = self.base_runner._get_traces_for_type(task_type)
        
        if not traces:
            print(f"错误: 类型 '{task_type}' 没有可用的 trace 数据")
            return []
        
        print(f"\n{'='*70}")
        print(f"批量测试: {task_type} - {task_name}")
        print(f"Trace 数量: {len(traces)}")
        print(f"{'='*70}")
        
        results = []
        for i, trace_id in enumerate(traces, 1):
            print(f"\n[{i}/{len(traces)}] 开始测试 trace {trace_id}")
            result = self.test_single_trace(task_type, trace_id)
            results.append(result)
        
        # 打印批量测试汇总（不保存文件）
        self._print_batch_summary(task_type, results)
        
        return results
    
    def _print_batch_summary(self, task_type: str, results: List[TestResult]):
        """在终端打印批量测试汇总"""
        success_count = sum(1 for r in results if r.success)
        total_count = len(results)
        success_rate = (success_count / total_count * 100) if total_count > 0 else 0
        total_time = sum(r.execution_time for r in results)
        
        print(f"\n{'='*70}")
        print(f"批量测试汇总: {task_type}")
        print(f"{'='*70}")
        print(f"总测试数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"失败数: {total_count - success_count}")
        print(f"成功率: {success_rate:.1f}%")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"\n详细结果:")
        print(f"{'-'*70}")
        
        for result in results:
            status = "✓" if result.success else "✗"
            print(f"  Trace {result.trace_id:>3}: {status} | "
                  f"得分: {result.score:>4.0f} | "
                  f"节点: {len(result.matched_nodes):>2} | "
                  f"时间: {result.execution_time:>5.2f}s")
        
        print(f"\n各 trace 的详细结果已分别保存到:")
        print(f"  {self.data_base_dir}/{task_type}/results/<trace_id>/")
        print(f"{'='*70}\n")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='结构化测试执行器 - 按 type/trace 组织结果目录',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个 trace
  python structural_test_runner.py task_configs/taobao.json type3:7 \\
      --data-base run_test_data/mobiagent/taobao
  
  # 测试某个 type 下的所有 trace
  python structural_test_runner.py task_configs/taobao.json type3 \\
      --data-base run_test_data/mobiagent/taobao
        """
    )
    
    parser.add_argument('config_file', help='任务配置文件路径 (如 task_configs/taobao.json)')
    parser.add_argument('target', help='测试目标: type3 或 type3:7')
    parser.add_argument('--data-base', required=True,
                       help='数据基础目录 (如 run_test_data/mobiagent/taobao)')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config_file):
        print(f"错误: 配置文件不存在: {args.config_file}")
        return 1
    
    # 检查数据目录是否存在
    if not os.path.exists(args.data_base):
        print(f"错误: 数据目录不存在: {args.data_base}")
        return 1
    
    # 创建 runner
    runner = StructuralTestRunner(args.config_file, args.data_base)
    
    # 解析测试目标
    target = args.target
    
    if ':' in target:
        # 格式: type3:7
        task_type, trace_id = target.split(':', 1)
        try:
            trace_id = int(trace_id)
        except ValueError:
            pass  # 保持字符串格式
        
        # 测试单个 trace
        result = runner.test_single_trace(task_type, trace_id)
        
        # 返回状态码
        return 0 if result.success else 1
        
    else:
        # 格式: type3 (测试该类型下的所有 trace)
        task_type = target
        results = runner.test_all_traces_in_type(task_type)
        
        # 返回状态码 (全部成功才返回0)
        success_count = sum(1 for r in results if r.success)
        return 0 if success_count == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
