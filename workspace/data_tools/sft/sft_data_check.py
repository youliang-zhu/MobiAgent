#!/usr/bin/env python3
"""
SFT 数据集质量检查工具

检查由 construct_sft.py 生成的训练数据集的完整性和质量
"""

import os
import json
import argparse
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any, Tuple
import hashlib
from PIL import Image
from tqdm import tqdm
import random


class SFTDataChecker:
    def __init__(self, sft_data_path: str, check_images: bool = True, verbose: bool = False):
        self.sft_data_path = Path(sft_data_path)
        self.check_images = check_images
        self.verbose = verbose
        
        # 文件路径
        self.metadata_path = self.sft_data_path / "metadata.json"
        self.decider_train_path = self.sft_data_path / "mobimind_decider_train.json"
        self.decider_val_path = self.sft_data_path / "mobimind_decider_val.json"
        self.grounder_train_path = self.sft_data_path / "mobimind_grounder_train.json"
        self.grounder_val_path = self.sft_data_path / "mobimind_grounder_val.json"
        
        # 统计数据
        self.stats = {
            'basic': {},
            'integrity': {},
            'quality': {},
            'decider': {},
            'grounder': {},
            'images': {}
        }
        
        # 问题记录
        self.issues = []
        
    def log(self, message: str, level: str = "INFO"):
        """打印日志"""
        prefix = {
            "INFO": "ℹ️ ",
            "SUCCESS": "✅",
            "WARNING": "⚠️ ",
            "ERROR": "❌",
            "SECTION": "\n" + "="*80 + "\n"
        }.get(level, "")
        
        print(f"{prefix} {message}")
        
    def load_json(self, file_path: Path) -> Any:
        """加载 JSON 文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            self.issues.append(f"文件不存在: {file_path}")
            return None
        except json.JSONDecodeError as e:
            self.issues.append(f"JSON 格式错误 {file_path}: {e}")
            return None
            
    def check_basic_stats(self):
        """基础统计检查"""
        self.log("基础统计", "SECTION")
        
        # 加载元数据
        metadata = self.load_json(self.metadata_path)
        if metadata is None:
            self.log("无法加载 metadata.json", "ERROR")
            return
            
        # 加载数据集
        decider_train = self.load_json(self.decider_train_path)
        decider_val = self.load_json(self.decider_val_path)
        grounder_train = self.load_json(self.grounder_train_path)
        grounder_val = self.load_json(self.grounder_val_path)
        
        # 实际样本数
        actual_counts = {
            'decider_train': len(decider_train) if decider_train else 0,
            'decider_val': len(decider_val) if decider_val else 0,
            'grounder_train': len(grounder_train) if grounder_train else 0,
            'grounder_val': len(grounder_val) if grounder_val else 0,
        }
        
        # 计算总数
        total_decider_train = (metadata.get('decider_entries_train', 0) + 
                               metadata.get('decider_no_history_entries_train', 0) + 
                               metadata.get('terminate_entries_train', 0) + 
                               metadata.get('decider_ss_entry_train', 0))
        
        total_decider_val = (metadata.get('decider_entries_val', 0) + 
                            metadata.get('decider_no_history_entries_val', 0) + 
                            metadata.get('terminate_entries_val', 0) + 
                            metadata.get('decider_ss_entry_val', 0))
        
        total_grounder_train = (metadata.get('grounder_entries_train', 0) + 
                               metadata.get('grounder_ss_entry_train', 0))
        
        total_grounder_val = (metadata.get('grounder_entries_val', 0) + 
                             metadata.get('grounder_ss_entry_val', 0))
        
        expected_counts = {
            'decider_train': total_decider_train,
            'decider_val': total_decider_val,
            'grounder_train': total_grounder_train,
            'grounder_val': total_grounder_val,
        }
        
        # 输出统计
        self.log("样本数量统计:")
        self.log(f"  Decider Train: {actual_counts['decider_train']:,} (预期: {expected_counts['decider_train']:,})")
        self.log(f"  Decider Val:   {actual_counts['decider_val']:,} (预期: {expected_counts['decider_val']:,})")
        self.log(f"  Grounder Train: {actual_counts['grounder_train']:,} (预期: {expected_counts['grounder_train']:,})")
        self.log(f"  Grounder Val:   {actual_counts['grounder_val']:,} (预期: {expected_counts['grounder_val']:,})")
        
        # 一致性检查
        all_match = True
        for key in actual_counts:
            if actual_counts[key] != expected_counts[key]:
                self.issues.append(f"{key} 数量不一致: 实际 {actual_counts[key]}, 预期 {expected_counts[key]}")
                all_match = False
                
        if all_match:
            self.log("✓ 所有数据集数量与 metadata.json 一致", "SUCCESS")
        else:
            self.log("✗ 数据集数量与 metadata 不一致", "ERROR")
            
        # 训练/验证集比例
        total_decider = actual_counts['decider_train'] + actual_counts['decider_val']
        total_grounder = actual_counts['grounder_train'] + actual_counts['grounder_val']
        
        if total_decider > 0:
            train_ratio_decider = actual_counts['decider_train'] / total_decider
            self.log(f"  Decider 训练/验证比例: {train_ratio_decider:.2%} / {1-train_ratio_decider:.2%}")
            
        if total_grounder > 0:
            train_ratio_grounder = actual_counts['grounder_train'] / total_grounder
            self.log(f"  Grounder 训练/验证比例: {train_ratio_grounder:.2%} / {1-train_ratio_grounder:.2%}")
        
        # 保存到统计数据
        self.stats['basic'] = {
            'actual': actual_counts,
            'expected': expected_counts,
            'metadata': metadata
        }
        
        return {
            'decider_train': decider_train,
            'decider_val': decider_val,
            'grounder_train': grounder_train,
            'grounder_val': grounder_val
        }
        
    def check_data_integrity(self, datasets: Dict[str, List[Dict]]):
        """数据完整性检查"""
        self.log("数据完整性检查", "SECTION")
        
        required_fields = ['instruction', 'output', 'images', 'input']
        
        for name, data in datasets.items():
            if not data:
                continue
                
            self.log(f"检查 {name}...")
            
            missing_fields = defaultdict(int)
            invalid_samples = 0
            
            for i, sample in enumerate(data):
                # 检查必需字段
                for field in required_fields:
                    if field not in sample:
                        missing_fields[field] += 1
                        if self.verbose:
                            self.log(f"  样本 {i}: 缺少字段 '{field}'", "WARNING")
                            
                # 检查 images 字段
                if 'images' in sample:
                    if not isinstance(sample['images'], list):
                        invalid_samples += 1
                        if self.verbose:
                            self.log(f"  样本 {i}: 'images' 应为列表", "WARNING")
                    elif len(sample['images']) == 0:
                        invalid_samples += 1
                        if self.verbose:
                            self.log(f"  样本 {i}: 'images' 为空", "WARNING")
                            
            # 报告结果
            if missing_fields:
                for field, count in missing_fields.items():
                    self.log(f"  ✗ {count} 个样本缺少字段 '{field}'", "ERROR")
                    self.issues.append(f"{name}: {count} 个样本缺少字段 '{field}'")
            else:
                self.log(f"  ✓ 所有样本包含必需字段", "SUCCESS")
                
            if invalid_samples > 0:
                self.log(f"  ✗ {invalid_samples} 个样本的 'images' 字段无效", "ERROR")
                self.issues.append(f"{name}: {invalid_samples} 个样本的 'images' 字段无效")
                
    def check_data_quality(self, datasets: Dict[str, List[Dict]]):
        """数据质量检查"""
        self.log("数据质量检查", "SECTION")
        
        for name, data in datasets.items():
            if not data:
                continue
                
            self.log(f"分析 {name}...")
            
            # 长度统计
            instruction_lengths = []
            output_lengths = []
            image_counts = []
            
            for sample in data:
                if 'instruction' in sample:
                    instruction_lengths.append(len(sample['instruction']))
                if 'output' in sample:
                    output_lengths.append(len(sample['output']))
                if 'images' in sample and isinstance(sample['images'], list):
                    image_counts.append(len(sample['images']))
                    
            if instruction_lengths:
                self.log(f"  Instruction 长度: min={min(instruction_lengths)}, "
                        f"max={max(instruction_lengths)}, "
                        f"avg={sum(instruction_lengths)/len(instruction_lengths):.1f}")
                        
            if output_lengths:
                self.log(f"  Output 长度: min={min(output_lengths)}, "
                        f"max={max(output_lengths)}, "
                        f"avg={sum(output_lengths)/len(output_lengths):.1f}")
                        
            if image_counts:
                counter = Counter(image_counts)
                self.log(f"  每样本图片数: {dict(counter)}")
                
            # 重复检测
            hashes = set()
            duplicates = 0
            
            for sample in data:
                # 创建样本哈希
                hash_str = json.dumps({
                    'instruction': sample.get('instruction', ''),
                    'output': sample.get('output', ''),
                    'images': sample.get('images', [])
                }, sort_keys=True)
                sample_hash = hashlib.md5(hash_str.encode()).hexdigest()
                
                if sample_hash in hashes:
                    duplicates += 1
                else:
                    hashes.add(sample_hash)
                    
            if duplicates > 0:
                self.log(f"  ⚠️  发现 {duplicates} 个重复样本", "WARNING")
                self.issues.append(f"{name}: 发现 {duplicates} 个重复样本")
            else:
                self.log(f"  ✓ 无重复样本", "SUCCESS")
                
    def check_decider_specific(self, datasets: Dict[str, List[Dict]]):
        """Decider 特定检查"""
        self.log("Decider 特定检查", "SECTION")
        
        decider_data = []
        if 'decider_train' in datasets and datasets['decider_train']:
            decider_data.extend(datasets['decider_train'])
        if 'decider_val' in datasets and datasets['decider_val']:
            decider_data.extend(datasets['decider_val'])
            
        if not decider_data:
            self.log("无 Decider 数据", "WARNING")
            return
            
        action_types = Counter()
        has_history = 0
        no_history = 0
        reasoning_lengths = []
        param_issues = 0
        param_issue_samples = {
            'main': [],      # 完整轨迹数据
            'ss': [],        # 单步数据
            'unexpected': [] # 意外图片数据
        }
        
        for sample in tqdm(decider_data, desc="检查 Decider 数据", disable=not self.verbose):
            # 解析 output
            try:
                output = json.loads(sample['output'])
                action = output.get('action', '')
                action_types[action] += 1
                
                # reasoning 长度
                if 'reasoning' in output:
                    reasoning_lengths.append(len(output['reasoning']))
                    
                # 参数完整性检查
                params = output.get('parameters', {})
                has_issue = False
                issue_type = ""
                
                if action == 'click' and 'target_element' not in params:
                    param_issues += 1
                    has_issue = True
                    issue_type = "click 缺少 target_element"
                elif action == 'input' and 'text' not in params:
                    param_issues += 1
                    has_issue = True
                    issue_type = "input 缺少 text"
                elif action == 'swipe' and 'direction' not in params:
                    param_issues += 1
                    has_issue = True
                    issue_type = "swipe 缺少 direction"
                
                # 记录有问题的样本信息
                if has_issue:
                    image_path = sample.get('images', [''])[0]
                    image_name = os.path.basename(image_path)
                    
                    # 识别数据来源类型和提取标识符
                    data_type = None
                    identifier = None
                    
                    if image_name.startswith('main_'):
                        data_type = 'main'
                        # main_type3_special_8_4.jpg -> type3_special/8
                        parts = image_name[5:].replace('.jpg', '').rsplit('_', 1)
                        identifier = parts[0].replace('_', '/')
                    elif image_name.startswith('ss_'):
                        data_type = 'ss'
                        # ss_decider_step_027_1.jpg -> decider/step_027
                        parts = image_name[3:].replace('.jpg', '').rsplit('_', 1)
                        identifier = parts[0].replace('_', '/', 1)  # 只替换第一个下划线
                    elif image_name.startswith('unexpected_'):
                        data_type = 'unexpected'
                        # unexpected_unexpected_001.jpg -> unexpected_001
                        identifier = image_name.replace('.jpg', '').replace('unexpected_', '', 1)
                    else:
                        data_type = 'unknown'
                        identifier = image_name.replace('.jpg', '')
                    
                    if data_type and data_type in param_issue_samples:
                        param_issue_samples[data_type].append({
                            'identifier': identifier,
                            'issue': issue_type,
                            'action': action,
                            'params': params
                        })
                    
            except json.JSONDecodeError:
                if self.verbose:
                    self.log(f"  无法解析 output JSON", "WARNING")
                continue
                
            # 检查历史记录
            instruction = sample.get('instruction', '')
            if '(No history)' in instruction or 'Your action history is:\n(No history)' in instruction:
                no_history += 1
            else:
                has_history += 1
                
        # 输出统计
        self.log(f"Action 类型分布:")
        for action, count in action_types.most_common():
            percentage = count / len(decider_data) * 100
            self.log(f"  {action}: {count} ({percentage:.1f}%)")
            
        total_history_samples = has_history + no_history
        if total_history_samples > 0:
            self.log(f"历史记录统计:")
            self.log(f"  有历史: {has_history} ({has_history/total_history_samples*100:.1f}%)")
            self.log(f"  无历史: {no_history} ({no_history/total_history_samples*100:.1f}%)")
            
        if reasoning_lengths:
            self.log(f"Reasoning 长度: min={min(reasoning_lengths)}, "
                    f"max={max(reasoning_lengths)}, "
                    f"avg={sum(reasoning_lengths)/len(reasoning_lengths):.1f}")
                    
        if param_issues > 0:
            self.log(f"⚠️  {param_issues} 个样本缺少必需参数", "WARNING")
            self.log(f"  问题样本详情（按数据来源分组）:")
            
            # 按类型分组输出
            type_names = {
                'main': '完整轨迹数据 (main_)',
                'ss': '单步数据 (ss_)',
                'unexpected': '意外图片数据 (unexpected_)'
            }
            
            for data_type in ['main', 'ss', 'unexpected']:
                if param_issue_samples[data_type]:
                    self.log(f"\n  [{type_names[data_type]}]")
                    for issue_sample in param_issue_samples[data_type]:
                        self.log(f"    - {issue_sample['identifier']}: {issue_sample['issue']}", "WARNING")
                        if self.verbose:
                            self.log(f"      action={issue_sample['action']}, params={issue_sample['params']}", "WARNING")
            
            self.issues.append(f"Decider: {param_issues} 个样本缺少必需参数")
        else:
            self.log(f"✓ 所有样本参数完整", "SUCCESS")
            
        # 保存统计
        self.stats['decider'] = {
            'action_types': dict(action_types),
            'has_history': has_history,
            'no_history': no_history,
            'reasoning_lengths': {
                'min': min(reasoning_lengths) if reasoning_lengths else 0,
                'max': max(reasoning_lengths) if reasoning_lengths else 0,
                'avg': sum(reasoning_lengths)/len(reasoning_lengths) if reasoning_lengths else 0
            }
        }
        
    def check_grounder_specific(self, datasets: Dict[str, List[Dict]]):
        """Grounder 特定检查"""
        self.log("Grounder 特定检查", "SECTION")
        
        grounder_data = []
        if 'grounder_train' in datasets and datasets['grounder_train']:
            grounder_data.extend(datasets['grounder_train'])
        if 'grounder_val' in datasets and datasets['grounder_val']:
            grounder_data.extend(datasets['grounder_val'])
            
        if not grounder_data:
            self.log("无 Grounder 数据", "WARNING")
            return
            
        output_types = Counter()
        coord_issues = 0
        bbox_issues = 0
        
        for sample in tqdm(grounder_data, desc="检查 Grounder 数据", disable=not self.verbose):
            try:
                output = json.loads(sample['output'])
                
                if 'coordinates' in output:
                    output_types['coordinates'] += 1
                    coords = output['coordinates']
                    
                    # 验证坐标格式
                    if not isinstance(coords, list) or len(coords) != 2:
                        coord_issues += 1
                    elif not all(isinstance(x, (int, float)) and x >= 0 for x in coords):
                        coord_issues += 1
                        
                elif 'bbox' in output:
                    output_types['bbox'] += 1
                    bbox = output['bbox']
                    
                    # 验证 bbox 格式
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        bbox_issues += 1
                    elif not all(isinstance(x, (int, float)) and x >= 0 for x in bbox):
                        bbox_issues += 1
                    # 检查左上右下顺序
                    elif bbox[0] >= bbox[2] or bbox[1] >= bbox[3]:
                        bbox_issues += 1
                        if self.verbose:
                            self.log(f"  bbox 顺序错误: {bbox}", "WARNING")
                            
            except json.JSONDecodeError:
                if self.verbose:
                    self.log(f"  无法解析 output JSON", "WARNING")
                continue
                
        # 输出统计
        self.log(f"Output 格式分布:")
        for fmt, count in output_types.items():
            percentage = count / len(grounder_data) * 100
            self.log(f"  {fmt}: {count} ({percentage:.1f}%)")
            
        if coord_issues > 0:
            self.log(f"⚠️  {coord_issues} 个 coordinates 格式错误", "WARNING")
            self.issues.append(f"Grounder: {coord_issues} 个 coordinates 格式错误")
            
        if bbox_issues > 0:
            self.log(f"⚠️  {bbox_issues} 个 bbox 格式错误", "WARNING")
            self.issues.append(f"Grounder: {bbox_issues} 个 bbox 格式错误")
            
        if coord_issues == 0 and bbox_issues == 0:
            self.log(f"✓ 所有输出格式正确", "SUCCESS")
            
        # 保存统计
        self.stats['grounder'] = {
            'output_types': dict(output_types)
        }
        
    def check_image_files(self, datasets: Dict[str, List[Dict]]):
        """图片检查"""
        if not self.check_images:
            return
            
        self.log("图片检查", "SECTION")
        
        all_images = set()
        missing_images = []
        image_sizes = Counter()
        
        # 收集所有图片路径
        for name, data in datasets.items():
            if not data:
                continue
            for sample in data:
                if 'images' in sample and isinstance(sample['images'], list):
                    all_images.update(sample['images'])
                    
        self.log(f"总共引用 {len(all_images)} 个不同的图片文件")
        
        # 检查图片
        for img_path in tqdm(all_images, desc="检查图片文件"):
            if not os.path.exists(img_path):
                missing_images.append(img_path)
            else:
                try:
                    with Image.open(img_path) as img:
                        image_sizes[img.size] += 1
                except Exception as e:
                    if self.verbose:
                        self.log(f"  无法读取图片 {img_path}: {e}", "WARNING")
                        
        # 报告结果
        if missing_images:
            self.log(f"❌ {len(missing_images)} 个图片文件不存在", "ERROR")
            self.issues.append(f"图片: {len(missing_images)} 个文件不存在")
            if self.verbose:
                for img in missing_images[:10]:  # 只显示前10个
                    self.log(f"    {img}", "WARNING")
                if len(missing_images) > 10:
                    self.log(f"    ... 还有 {len(missing_images) - 10} 个", "WARNING")
        else:
            self.log(f"✓ 所有图片文件存在", "SUCCESS")
            
        # 图片尺寸统计
        if image_sizes:
            self.log(f"图片尺寸分布:")
            total_images = sum(image_sizes.values())
            
            # 找出主流尺寸
            most_common_size, most_common_count = image_sizes.most_common(1)[0]
            
            for size, count in image_sizes.most_common():
                percentage = count / total_images * 100
                marker = "✓ 主流尺寸" if size == most_common_size else "⚠️  异常尺寸"
                self.log(f"  {size[0]}x{size[1]}: {count} ({percentage:.1f}%) {marker}")
                
            if len(image_sizes) > 1:
                self.log(f"⚠️  发现 {len(image_sizes)} 种不同的图片尺寸", "WARNING")
                self.issues.append(f"图片: 发现 {len(image_sizes)} 种不同的尺寸")
            else:
                self.log(f"✓ 所有图片尺寸一致", "SUCCESS")
                
        self.stats['images'] = {
            'total_unique': len(all_images),
            'missing': len(missing_images),
            'sizes': {f"{w}x{h}": count for (w, h), count in image_sizes.items()}
        }
        
    def generate_samples(self, datasets: Dict[str, List[Dict]]):
        """生成示例数据文件"""
        self.log("生成示例数据", "SECTION")
        
        # 合并 train 和 val 数据
        all_decider = []
        all_grounder = []
        
        if datasets.get('decider_train'):
            all_decider.extend(datasets['decider_train'])
        if datasets.get('decider_val'):
            all_decider.extend(datasets['decider_val'])
        if datasets.get('grounder_train'):
            all_grounder.extend(datasets['grounder_train'])
        if datasets.get('grounder_val'):
            all_grounder.extend(datasets['grounder_val'])
        
        # 按数据来源分类
        def classify_by_source(data_list):
            main_samples = []
            ss_samples = []
            unexpected_samples = []
            
            for sample in data_list:
                if 'images' in sample and sample['images']:
                    image_name = os.path.basename(sample['images'][0])
                    if image_name.startswith('main_'):
                        main_samples.append(sample)
                    elif image_name.startswith('ss_'):
                        ss_samples.append(sample)
                    elif image_name.startswith('unexpected_'):
                        unexpected_samples.append(sample)
            
            return main_samples, ss_samples, unexpected_samples
        
        # 分类 decider 和 grounder 数据
        decider_main, decider_ss, decider_unexpected = classify_by_source(all_decider)
        grounder_main, grounder_ss, grounder_unexpected = classify_by_source(all_grounder)
        
        # 构建样本结构
        samples = {
            'main_trajectory': {
                'decider': random.sample(decider_main, min(5, len(decider_main))) if decider_main else [],
                'grounder': random.sample(grounder_main, min(5, len(grounder_main))) if grounder_main else []
            },
            'single_step': {
                'decider': random.sample(decider_ss, min(5, len(decider_ss))) if decider_ss else [],
                'grounder': random.sample(grounder_ss, min(5, len(grounder_ss))) if grounder_ss else []
            },
            'unexpected': {
                'decider': random.sample(decider_unexpected, min(5, len(decider_unexpected))) if decider_unexpected else [],
                'grounder': random.sample(grounder_unexpected, min(5, len(grounder_unexpected))) if grounder_unexpected else []
            }
        }
        
        # 输出统计
        self.log(f"  完整轨迹: Decider {len(samples['main_trajectory']['decider'])} 条, Grounder {len(samples['main_trajectory']['grounder'])} 条")
        self.log(f"  单步数据: Decider {len(samples['single_step']['decider'])} 条, Grounder {len(samples['single_step']['grounder'])} 条")
        self.log(f"  意外图片: Decider {len(samples['unexpected']['decider'])} 条, Grounder {len(samples['unexpected']['grounder'])} 条")
        
        # 保存到文件
        output_path = Path(__file__).parent / "sft_data_samples.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
            
        self.log(f"✓ 示例数据已保存到: {output_path}", "SUCCESS")
        
    def run(self):
        """运行所有检查"""
        self.log(f"开始检查 SFT 数据集: {self.sft_data_path}", "SECTION")
        
        # 基础统计
        datasets = self.check_basic_stats()
        
        if not datasets:
            self.log("无法加载数据集，检查终止", "ERROR")
            return
            
        # 数据完整性
        self.check_data_integrity(datasets)
        
        # 数据质量
        self.check_data_quality(datasets)
        
        # Decider 检查
        self.check_decider_specific(datasets)
        
        # Grounder 检查
        self.check_grounder_specific(datasets)
        
        # 图片检查
        if self.check_images:
            self.check_image_files(datasets)
            
        # 生成示例
        self.generate_samples(datasets)
        
        # 总结
        self.log("检查完成", "SECTION")
        if self.issues:
            self.log(f"发现 {len(self.issues)} 个问题:", "WARNING")
            for issue in self.issues:
                self.log(f"  - {issue}", "WARNING")
        else:
            self.log("✓ 所有检查通过，数据集质量良好！", "SUCCESS")


def main():
    parser = argparse.ArgumentParser(description="SFT 数据集质量检查工具")
    parser.add_argument(
        "--sft_data_path",
        type=str,
        default="/home/agent/mobiAgent/MobiAgent/workspace/data/training_data/sft_data",
        help="SFT 数据集根目录路径"
    )
    parser.add_argument(
        "--check_images",
        action="store_true",
        default=True,
        help="检查图片文件完整性和尺寸（默认：True）"
    )
    parser.add_argument(
        "--no_check_images",
        action="store_false",
        dest="check_images",
        help="跳过图片检查"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    
    args = parser.parse_args()
    
    checker = SFTDataChecker(
        sft_data_path=args.sft_data_path,
        check_images=args.check_images,
        verbose=args.verbose
    )
    
    checker.run()


if __name__ == "__main__":
    main()
