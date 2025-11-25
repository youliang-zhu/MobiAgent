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

- 批量测试整个时间戳文件夹:
  python structural_test_runner.py --batch-mode \\
      --timestamp 20251106_162407 \\
      --model mobiagent
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
from multiprocessing import Pool

# 导入原有的 UniversalTestRunner 核心功能
from universal_test_runner import UniversalTestRunner, TestResult, TestSummary
from avdag.logger import configure_logging


def _test_trace_worker(args):
    """多进程 worker 函数 - 测试单个 trace
    
    必须定义在类外部，因为 multiprocessing 需要可 pickle 的函数
    
    Args:
        args: (config_file, data_base_dir, task_type, trace_id)
    
    Returns:
        TestResult: 测试结果
    """
    config_file, data_base_dir, task_type, trace_id = args
    
    # 只禁用控制台输出，保留文件日志
    # 移除所有 StreamHandler，保留 FileHandler
    import logging
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
    
    # 同时禁用 avdag 日志的控制台输出
    avdag_logger = logging.getLogger('avdag')
    for handler in avdag_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            avdag_logger.removeHandler(handler)
    
    # 创建 runner 并执行测试
    runner = StructuralTestRunner(config_file, data_base_dir)
    return runner.test_single_trace(task_type, trace_id)


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
        
        # 临时禁用全局日志文件（避免在 MobiFlow 目录生成 test-*.log）
        if 'logging' not in self.config:
            self.config['logging'] = {}
        # 设为空字符串表示不输出到文件（设为 None 会导致 format() 报错）
        self.config['logging']['output_file'] = ''
        
        # 保存修改后的配置到临时文件（因为 UniversalTestRunner 会重新读取配置文件）
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
            json.dump(self.config, tmp, ensure_ascii=False, indent=2)
            tmp_config_file = tmp.name
        
        # 创建基础的 UniversalTestRunner 实例（用于复用测试逻辑）
        self.base_runner = UniversalTestRunner(tmp_config_file)
        
        # 删除临时配置文件
        os.unlink(tmp_config_file)
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
        
        # 打印测试结果状态（与universal保持一致）
        status = "✓ 成功" if result.success else "✗ 失败"
        print(f"\n{'='*60}")
        print(f"单个trace测试完成")
        print(f"{'='*60}")
        print(f"trace {result.trace_id}: {status} | score: {result.score} | nodes: {result.matched_nodes}")
        if result.reason:
            print(f"原因: {result.reason}")
        
        # 显示成功率（即使是单个trace）
        success_rate = 100.0 if result.success else 0.0
        print(f"\n成功率: {1 if result.success else 0}/1 ({success_rate:.1f}%)")
        print(f"{'='*60}")
        
        # 保存结果到结构化目录
        self._save_trace_result(result, results_dir)
        
        return result
    
    def test_all_traces_in_type(self, task_type: str, max_workers: int = 2) -> List[TestResult]:
        """测试指定 type 下的所有 trace
        
        Args:
            task_type: 任务类型 (如 "type3")
            max_workers: 并行 worker 数量，默认 2（设为 1 禁用并行）
        
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
        
        # 判断是否使用并行
        if max_workers > 1 and len(traces) > 1:
            # 并行执行
            print(f"使用 {max_workers} 个进程并行测试...")
            
            # 准备任务参数
            tasks = [
                (self.config_file, self.data_base_dir, task_type, trace_id)
                for trace_id in traces
            ]
            
            # 使用进程池并行执行
            with Pool(processes=max_workers) as pool:
                results = pool.map(_test_trace_worker, tasks)
            
            print(f"✓ 并行测试完成")
        else:
            # 串行执行
            if max_workers == 1:
                print(f"串行测试模式（workers=1）...")
            else:
                print(f"串行测试模式（单个 trace）...")
            
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
        
        # 详细结果列表（与 universal 格式一致）
        for result in results:
            status = "✓" if result.success else "✗"
            print(f"trace {result.trace_id}: {status} | score: {result.score} | nodes: {result.matched_nodes}")
        
        # 统计信息
        print(f"\n总测试数: {total_count}")
        print(f"成功数: {success_count}")
        print(f"失败数: {total_count - success_count}")
        print(f"成功率: {success_count}/{total_count} ({success_rate:.1f}%)")
        print(f"总结果测试耗时: {total_time:.2f}秒")
        
        print(f"\n各 trace 的详细结果已分别保存到:")
        print(f"  {self.data_base_dir}/{task_type}/results/<trace_id>/")
        print(f"{'='*70}\n")


class BatchTestRunner:
    """批量测试执行器 - 测试整个时间戳文件夹下的所有app"""
    
    # app 名称到配置文件的映射
    APP_CONFIG_MAP = {
        'taobao': 'task_configs/taobao.json',
        'bilibili': 'task_configs/bilibili.json',
        'cloudmusic': 'task_configs/cloudmusic.json',
        'weixin': 'task_configs/weixin.json',
        'xiaohongshu': 'task_configs/xiaohongshu_auto.json',
    }
    
    def __init__(self, timestamp: str, model: str, base_dir: str = None, max_workers: int = 2):
        """初始化批量测试执行器
        
        Args:
            timestamp: 时间戳 (如 20251106_162407)
            model: 模型名称 (如 mobiagent)
            base_dir: 项目根目录（默认自动检测）
            max_workers: 并行 worker 数量，默认 2
        """
        self.timestamp = timestamp
        self.model = model
        self.max_workers = max_workers
        
        # 自动查找项目根目录（MobiFlow的父目录）
        if base_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # structural_test_runner.py 在 MobiFlow/ 目录下
            # 项目根目录是 MobiFlow 的父目录
            self.base_dir = os.path.dirname(current_dir)
        else:
            self.base_dir = base_dir
        
        # 构建时间戳目录路径
        self.timestamp_dir = os.path.join(
            self.base_dir, 
            'run_test_data', 
            model, 
            timestamp
        )
        
        if not os.path.exists(self.timestamp_dir):
            raise FileNotFoundError(f"时间戳目录不存在: {self.timestamp_dir}")
        
        print(f"\n{'='*80}")
        print(f"批量测试模式")
        print(f"{'='*80}")
        print(f"模型: {model}")
        print(f"时间戳: {timestamp}")
        print(f"测试目录: {self.timestamp_dir}")
        print(f"{'='*80}\n")
    
    def _get_available_apps(self) -> List[str]:
        """扫描时间戳目录，获取所有可用的app"""
        apps = []
        for item in os.listdir(self.timestamp_dir):
            item_path = os.path.join(self.timestamp_dir, item)
            # 只考虑目录，且在配置映射中的app
            if os.path.isdir(item_path) and item in self.APP_CONFIG_MAP:
                apps.append(item)
        return sorted(apps)
    
    def _get_types_in_app(self, app_dir: str) -> List[str]:
        """获取app目录下的所有type"""
        types = []
        for item in os.listdir(app_dir):
            item_path = os.path.join(app_dir, item)
            if os.path.isdir(item_path) and item.startswith('type'):
                types.append(item)
        return sorted(types)
    
    def test_all_apps(self) -> Dict[str, List[TestResult]]:
        """测试所有app，返回所有测试结果"""
        apps = self._get_available_apps()
        
        if not apps:
            print("错误: 未找到任何可测试的app")
            return {}
        
        print(f"发现 {len(apps)} 个app: {', '.join(apps)}\n")
        
        all_results = {}
        
        for i, app in enumerate(apps, 1):
            print(f"\n{'#'*80}")
            print(f"[{i}/{len(apps)}] 开始测试 APP: {app}")
            print(f"{'#'*80}")
            
            app_results = self._test_single_app(app)
            all_results[app] = app_results
        
        # 生成总汇总报告
        self._save_batch_summary(all_results)
        
        return all_results
    
    def _test_single_app(self, app: str) -> List[TestResult]:
        """测试单个app的所有type"""
        config_file = os.path.join(self.base_dir, 'MobiFlow', self.APP_CONFIG_MAP[app])
        
        if not os.path.exists(config_file):
            print(f"警告: 配置文件不存在: {config_file}，跳过 {app}")
            return []
        
        app_dir = os.path.join(self.timestamp_dir, app)
        types = self._get_types_in_app(app_dir)
        
        if not types:
            print(f"警告: {app} 目录下没有找到type子目录，跳过")
            return []
        
        print(f"发现 {len(types)} 个type: {', '.join(types)}")
        
        app_results = []
        
        for type_name in types:
            print(f"\n{'-'*70}")
            print(f"测试 {app}/{type_name}")
            print(f"{'-'*70}")
            
            # 创建 StructuralTestRunner 实例
            data_base = os.path.join(self.timestamp_dir, app)
            runner = StructuralTestRunner(config_file, data_base)
            
            # 测试该type的所有trace
            try:
                results = runner.test_all_traces_in_type(type_name, self.max_workers)
                app_results.extend(results)
            except Exception as e:
                print(f"错误: 测试 {app}/{type_name} 时出错: {e}")
                import traceback
                traceback.print_exc()
        
        return app_results
    
    def _save_batch_summary(self, all_results: Dict[str, List[TestResult]]):
        """保存批量测试的总汇总报告"""
        summary_file = os.path.join(self.timestamp_dir, 'batch_test_summary.txt')
        detailed_file = os.path.join(self.timestamp_dir, 'batch_test_detailed.json')
        
        # 计算总体统计
        total_count = sum(len(results) for results in all_results.values())
        total_success = sum(sum(1 for r in results if r.success) for results in all_results.values())
        total_time = sum(sum(r.execution_time for r in results) for results in all_results.values())
        total_success_rate = (total_success / total_count * 100) if total_count > 0 else 0
        
        # 计算全局得分统计
        global_actual_score = sum(sum(r.score for r in results) for results in all_results.values())
        
        # 计算全局步数统计
        total_action_count = sum(sum(r.action_count for r in results) for results in all_results.values())
        avg_action_count = (total_action_count / total_count) if total_count > 0 else 0
        
        # 生成文本汇总报告
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"批量测试汇总报告\n")
            f.write(f"{'='*80}\n")
            f.write(f"时间戳: {self.timestamp}\n")
            f.write(f"模型: {self.model}\n")
            f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            
            # 每个app的统计
            for app, results in all_results.items():
                if not results:
                    continue
                
                f.write(f"【{app}】\n")
                
                # 按type分组统计
                type_groups = {}
                for result in results:
                    type_name = result.task_type
                    if type_name not in type_groups:
                        type_groups[type_name] = []
                    type_groups[type_name].append(result)
                
                # app级别的得分和时间统计
                app_actual_score = 0
                app_total_time = 0
                
                for type_name in sorted(type_groups.keys()):
                    type_results = type_groups[type_name]
                    type_success = sum(1 for r in type_results if r.success)
                    type_total = len(type_results)
                    type_rate = (type_success / type_total * 100) if type_total > 0 else 0
                    
                    # 计算该type的得分和时间统计
                    type_actual_score = 0
                    type_total_time = 0
                    
                    f.write(f"  {type_name}: {type_success}/{type_total} 成功 ({type_rate:.1f}%)\n")
                    
                    # 列出每个trace的详细得分和时间
                    for result in type_results:
                        trace_num = result.trace_id.split('/')[-1] if '/' in result.trace_id else result.trace_id
                        actual = result.score
                        exec_time = result.execution_time
                        steps = result.action_count
                        type_actual_score += actual
                        type_total_time += exec_time
                        f.write(f"    - trace {trace_num}: 得分 {actual} | 步数 {steps} | 结果测试耗时 {exec_time:.1f}秒\n")
                    
                    # type小计
                    f.write(f"    小计得分: {type_actual_score} | 总结果测试耗时: {type_total_time:.1f}秒\n\n")
                    
                    app_actual_score += type_actual_score
                    app_total_time += type_total_time
                
                # app总计
                app_success = sum(1 for r in results if r.success)
                app_total = len(results)
                app_rate = (app_success / app_total * 100) if app_total > 0 else 0
                
                f.write(f"  【{app} 应用总计】\n")
                f.write(f"  任务成功率: {app_success}/{app_total} ({app_rate:.1f}%)\n")
                f.write(f"  总得分: {app_actual_score}\n")
                f.write(f"  总结果测试耗时: {app_total_time:.1f}秒\n\n")
            
            # 全局总计
            f.write(f"{'='*80}\n")
            f.write(f"【全局统计】\n")
            f.write(f"总任务数: {total_count}\n")
            f.write(f"成功数: {total_success}\n")
            f.write(f"失败数: {total_count - total_success}\n")
            f.write(f"总成功率: {total_success}/{total_count} ({total_success_rate:.1f}%)\n")
            f.write(f"总得分: {global_actual_score}\n")
            f.write(f"总步数: {total_action_count}\n")
            f.write(f"平均步数: {avg_action_count:.1f}\n")
            f.write(f"总结果测试耗时: {total_time:.1f}秒\n")
            f.write(f"{'='*80}\n")
        
        # 生成JSON详细报告
        detailed_data = {
            'timestamp': self.timestamp,
            'model': self.model,
            'test_time': datetime.now().isoformat(),
            'summary': {
                'total_tasks': total_count,
                'success_count': total_success,
                'failure_count': total_count - total_success,
                'success_rate': total_success_rate,
                'total_actual_score': global_actual_score,
                'total_time': total_time
            },
            'apps': {}
        }
        
        for app, results in all_results.items():
            detailed_data['apps'][app] = {
                'results': [asdict(r) for r in results],
                'success_count': sum(1 for r in results if r.success),
                'total_count': len(results)
            }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        # 在终端打印汇总
        print(f"\n{'='*80}")
        print(f"批量测试完成！")
        print(f"{'='*80}")
        print(f"总任务数: {total_count}")
        print(f"成功数: {total_success}")
        print(f"失败数: {total_count - total_success}")
        print(f"总成功率: {total_success}/{total_count} ({total_success_rate:.1f}%)")
        print(f"总得分: {global_actual_score}")
        print(f"总步数: {total_action_count}")
        print(f"平均步数: {avg_action_count:.1f}")
        print(f"总结果测试耗时: {total_time:.1f}秒")
        print(f"\n汇总报告已保存:")
        print(f"  文本: {summary_file}")
        print(f"  JSON: {detailed_file}")
        print(f"{'='*80}\n")


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
  
  # 批量测试整个时间戳文件夹
  python structural_test_runner.py --batch-mode \\
      --timestamp 20251106_162407 \\
      --model mobiagent
        """
    )
    
    # 批量测试模式参数
    parser.add_argument('--batch-mode', action='store_true',
                       help='批量测试模式：测试整个时间戳文件夹下的所有app')
    parser.add_argument('--timestamp', type=str,
                       help='时间戳 (批量模式必需，如 20251106_162407)')
    parser.add_argument('--model', type=str,
                       help='模型名称 (批量模式必需，如 mobiagent)')
    parser.add_argument('--workers', type=int, default=2,
                       help='并行 worker 数量 (默认: 2，设为 1 禁用并行)')
    
    # 单个测试模式参数
    parser.add_argument('config_file', nargs='?', 
                       help='任务配置文件路径 (如 task_configs/taobao.json)')
    parser.add_argument('target', nargs='?',
                       help='测试目标: type3 或 type3:7')
    parser.add_argument('--data-base',
                       help='数据基础目录 (如 run_test_data/mobiagent/taobao)')
    
    args = parser.parse_args()
    
    # 判断是批量模式还是单个测试模式
    if args.batch_mode:
        # 批量测试模式
        if not args.timestamp or not args.model:
            parser.error("批量模式需要 --timestamp 和 --model 参数")
        
        try:
            batch_runner = BatchTestRunner(args.timestamp, args.model, max_workers=args.workers)
            batch_runner.test_all_apps()
            return 0
        except Exception as e:
            print(f"批量测试失败: {e}")
            import traceback
            traceback.print_exc()
            return 1
    
    else:
        # 单个测试模式
        if not args.config_file or not args.target or not args.data_base:
            parser.error("单个测试模式需要 config_file, target 和 --data-base 参数")
        
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
