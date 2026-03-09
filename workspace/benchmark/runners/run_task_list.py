"""
run_task_list.py - 批量运行 task_list.json 中的所有任务并生成结构化的测试数据

用法:
    python run_task_list.py --save_raw_data_path <路径> [--service_ip localhost] [--decider_port 8000] [--grounder_port 8001] [--planner_port 8002] [--resume]

输出结构:
    <save_raw_data_path>/
      error_log.txt
      {app}/
        {type}/
          {global_index}/
            actions.json
            react.json
            *.jpg
"""

import os
import sys
import json
import time
import logging
import argparse
import traceback
import shutil
from datetime import datetime

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# 导入 mobiagent 模块
from runner.mobiagent.mobiagent import AndroidDevice, init, get_app_package_name, task_in_app

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _format_time(seconds):
    """将秒数格式化为 分秒 格式"""
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes > 0:
        return f"{minutes}分{secs:.1f}秒 ({seconds:.1f}秒)"
    else:
        return f"{seconds:.1f}秒"


def _load_json_safely(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_task_completed(task_output_dir):
    """判断单个任务目录是否可视为已完成（用于断点续跑）。"""
    if not os.path.isdir(task_output_dir):
        return False

    actions_path = os.path.join(task_output_dir, "actions.json")
    react_path = os.path.join(task_output_dir, "react.json")
    if not (os.path.exists(actions_path) and os.path.exists(react_path)):
        return False

    actions_data = _load_json_safely(actions_path)
    react_data = _load_json_safely(react_path)
    if actions_data is None or react_data is None:
        return False

    actions = actions_data.get("actions") if isinstance(actions_data, dict) else None
    if not isinstance(actions, list) or len(actions) == 0:
        return False
    if not isinstance(react_data, list) or len(react_data) == 0:
        return False

    action_count = actions_data.get("action_count")
    if isinstance(action_count, int) and action_count != len(actions):
        return False
    if len(actions) != len(react_data):
        return False

    return True


def _save_run_summary(output_root, overall_start_time, overall_end_time, task_execution_records):
    """生成 Agent 运行时间汇总报告"""
    summary_file = os.path.join(output_root, 'agent_run_summary.txt')
    
    total_execution_time = overall_end_time - overall_start_time
    total_tasks = len(task_execution_records)
    
    # 按 app 和 type 分组统计
    app_stats = {}
    for record in task_execution_records:
        app = record['app']
        task_type = record['type']
        
        if app not in app_stats:
            app_stats[app] = {}
        if task_type not in app_stats[app]:
            app_stats[app][task_type] = []
        
        app_stats[app][task_type].append(record)
    
    # 生成报告
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Agent 运行时间汇总报告\n")
        f.write(f"{'='*80}\n")
        f.write(f"输出目录: {output_root}\n")
        f.write(f"开始时间: {datetime.fromtimestamp(overall_start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"结束时间: {datetime.fromtimestamp(overall_end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总运行时间: {_format_time(total_execution_time)}\n")
        f.write(f"{'='*80}\n\n")
        
        # 任务运行时间明细
        f.write("【任务运行时间明细】\n")
        for record in task_execution_records:
            if record['status'] == 'failed':
                status_tag = " [执行失败]"
            elif record['status'] == 'skipped':
                status_tag = " [断点跳过]"
            else:
                status_tag = ""
            f.write(f"Task {record['global_index']} ({record['app']}/{record['type']}): "
                   f"运行时间 {record['execution_time']:.1f}秒{status_tag}\n")
        
        f.write(f"\n{'='*80}\n")
        
        # 按应用分组统计
        f.write("【按应用分组统计】\n")
        for app in sorted(app_stats.keys()):
            f.write(f"{app}:\n")
            
            app_total_time = 0
            app_total_tasks = 0
            
            for task_type in sorted(app_stats[app].keys()):
                type_records = app_stats[app][task_type]
                type_count = len(type_records)
                type_total_time = sum(r['execution_time'] for r in type_records)
                type_avg_time = type_total_time / type_count if type_count > 0 else 0
                
                f.write(f"  - {task_type}: {type_count}个任务, 平均运行时间 {type_avg_time:.1f}秒\n")
                
                app_total_time += type_total_time
                app_total_tasks += type_count
            
            app_avg_time = app_total_time / app_total_tasks if app_total_tasks > 0 else 0
            f.write(f"  小计: {app_total_tasks}个任务, 总运行时间 {app_total_time:.1f}秒, 平均 {app_avg_time:.1f}秒\n\n")
        
        # 全局统计
        f.write(f"{'='*80}\n")
        f.write("【全局统计】\n")
        f.write(f"总任务数: {total_tasks}\n")
        f.write(f"总运行时间: {_format_time(total_execution_time)}\n")
        
        avg_time = total_execution_time / total_tasks if total_tasks > 0 else 0
        f.write(f"平均任务运行时间: {avg_time:.1f}秒\n")
        
        # 失败任务统计
        failed_count = sum(1 for r in task_execution_records if r['status'] == 'failed')
        if failed_count > 0:
            f.write(f"\n执行失败任务数: {failed_count}\n")
        skipped_count = sum(1 for r in task_execution_records if r['status'] == 'skipped')
        if skipped_count > 0:
            f.write(f"断点跳过任务数: {skipped_count}\n")

        f.write(f"{'='*80}\n")
    
    logging.info(f"运行时间汇总报告已保存: {summary_file}")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="批量运行 task_list.json 中的所有任务")
    parser.add_argument("--save_raw_data_path", type=str, required=True, help="原始运行数据保存路径（如 workspace/data/raw_runs/mobiagent/20251106_162407）")
    parser.add_argument("--service_ip", type=str, required=True, help="服务IP地址")
    parser.add_argument("--decider_port", type=int, required=True, help="Decider服务端口")
    parser.add_argument("--grounder_port", type=int, required=True, help="Grounder服务端口")
    parser.add_argument("--planner_port", type=int, required=True, help="Planner服务端口")
    parser.add_argument("--resume", action="store_true", help="断点续跑：检测已有完整任务目录并跳过，缺失或损坏任务会重跑")
    
    args = parser.parse_args()
    
    # 切换到项目根目录，确保能找到 msyh.ttf 等资源文件
    project_root = PROJECT_ROOT
    os.chdir(project_root)
    logging.info(f"切换工作目录到: {os.getcwd()}")

    # 初始化客户端
    logging.info(f"初始化服务连接: {args.service_ip}:{args.decider_port}/{args.grounder_port}/{args.planner_port}")
    init(args.service_ip, args.decider_port, args.grounder_port, args.planner_port)

    # 连接设备
    logging.info("连接Android设备...")
    device = AndroidDevice()

    # 读取 task_list.json（使用绝对路径）
    task_list_path = os.path.join(project_root, "workspace", "benchmark", "configs", "task_list.json")
    if not os.path.exists(task_list_path):
        logging.error(f"task_list.json 未找到: {task_list_path}")
        sys.exit(1)
    
    with open(task_list_path, "r", encoding="utf-8") as f:
        task_list = json.load(f)
    
    # 创建输出目录结构（使用绝对路径）
    output_root = os.path.abspath(args.save_raw_data_path)
    os.makedirs(output_root, exist_ok=True)
    
    error_log_path = os.path.join(output_root, "error_log.txt")
    
    logging.info(f"输出目录: {output_root}")
    logging.info(f"错误日志: {error_log_path}")
    
    # 统计信息
    global_index = 1
    total_tasks = sum(len(tasks) for types in task_list.values() for tasks in types.values())
    
    # 记录每个任务的执行信息
    task_execution_records = []
    overall_start_time = time.time()
    resume_skipped_count = 0
    
    logging.info(f"总任务数: {total_tasks}")
    logging.info("=" * 80)
    
    # 遍历所有任务
    for app_key, types in task_list.items():
        for type_key, tasks in types.items():
            for task_desc in tasks:
                task_output_dir = os.path.join(output_root, app_key, type_key, str(global_index))

                if args.resume and _is_task_completed(task_output_dir):
                    logging.info(f"\n[{global_index}/{total_tasks}] 已完成，断点跳过: {app_key}/{type_key}")
                    task_execution_records.append({
                        'global_index': global_index,
                        'app': app_key,
                        'type': type_key,
                        'task_desc': task_desc,
                        'execution_time': 0.0,
                        'status': 'skipped',
                        'error': None
                    })
                    resume_skipped_count += 1
                    global_index += 1
                    continue

                logging.info(f"\n[{global_index}/{total_tasks}] 开始任务: {app_key}/{type_key}")
                logging.info(f"任务描述: {task_desc}")
                
                # 记录任务开始时间
                task_start_time = time.time()
                task_status = 'success'
                error_msg = None
                
                try:
                    # 调用 planner 推断 app 和 package_name
                    logging.info("调用 Planner 推断应用和包名...")
                    app_name, package_name, optimized_desc, template_name = get_app_package_name(task_desc)
                    logging.info(f"应用名称: {app_name}, 包名: {package_name}, 模板: {template_name}")
                    
                    # 启动 APP
                    logging.info(f"启动应用: {package_name}")
                    device.app_start(package_name)
                    
                    # 断点续跑时：如果目录存在但不完整，先清理后重跑，避免混入脏结果
                    if args.resume and os.path.isdir(task_output_dir):
                        logging.info(f"检测到未完成或损坏目录，清理后重跑: {task_output_dir}")
                        shutil.rmtree(task_output_dir)

                    # 创建任务输出目录
                    os.makedirs(task_output_dir, exist_ok=True)
                    
                    # 执行任务
                    logging.info(f"执行任务，输出到: {task_output_dir}")
                    task_in_app(app_name, task_desc, optimized_desc, device, task_output_dir, bbox_flag=True)
                    
                    logging.info(f"[{global_index}/{total_tasks}] 任务完成")
                    
                except Exception as e:
                    # 记录错误
                    task_status = 'failed'
                    error_msg = str(e)
                    
                    error_log_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Task {global_index} ({app_key}/{type_key}) failed: {error_msg}\n"
                    logging.error(error_log_msg.strip())
                    
                    with open(error_log_path, "a", encoding="utf-8") as f:
                        f.write(error_log_msg)
                        # 可选: 记录详细的 traceback
                        # f.write(traceback.format_exc() + "\n")
                
                finally:
                    # 记录任务执行时间
                    task_end_time = time.time()
                    task_execution_time = task_end_time - task_start_time
                    
                    task_execution_records.append({
                        'global_index': global_index,
                        'app': app_key,
                        'type': type_key,
                        'task_desc': task_desc,
                        'execution_time': task_execution_time,
                        'status': task_status,
                        'error': error_msg
                    })
                    
                    # 任务结束后的清理流程
                    if global_index < total_tasks:  # 不是最后一个任务
                        logging.info("等待 3 秒...")
                        time.sleep(3)
                        
                        logging.info("按 HOME 键清理状态...")
                        device.d.press("home")
                        
                        logging.info("等待 2 秒...")
                        time.sleep(2)
                    
                    global_index += 1
    
    # 记录总体结束时间
    overall_end_time = time.time()
    
    # 生成运行时间汇总报告
    _save_run_summary(output_root, overall_start_time, overall_end_time, task_execution_records)
    
    logging.info("=" * 80)
    logging.info(f"所有任务执行完成！总任务数: {total_tasks}")
    if args.resume:
        logging.info(f"断点跳过任务数: {resume_skipped_count}")
    logging.info(f"结果保存在: {output_root}")
    
    if os.path.exists(error_log_path):
        with open(error_log_path, "r", encoding="utf-8") as f:
            error_count = len(f.readlines())
        if error_count > 0:
            logging.warning(f"有 {error_count} 个任务失败，详见: {error_log_path}")
    else:
        logging.info("所有任务均成功完成！")


if __name__ == "__main__":
    main()
