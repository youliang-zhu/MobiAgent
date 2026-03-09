#!/usr/bin/env python3
"""
GRPO 数据集质量检查工具

推荐检查指令（项目根目录执行）:
python workspace/data_tools/grpo/grpo_data_check.py \
    --grpo_data_path /home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data

若需要自动清理缺少 bounds 的 click 样本:
python workspace/data_tools/grpo/grpo_data_check.py \
    --grpo_data_path /home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data \
    --auto_fix
"""

import os
import json
import argparse
from collections import Counter
from pathlib import Path
from tqdm import tqdm
import random


class GRPODataChecker:
    def __init__(self, grpo_data_path: str, check_images: bool = True, auto_fix: bool = False):
        self.grpo_data_path = Path(grpo_data_path)
        self.should_check_images = check_images
        self.auto_fix = auto_fix
        self.issues = []
        
    def load_json(self, file_path: Path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.issues.append(f"加载失败 {file_path}: {e}")
            return None
    
    def save_json(self, data, file_path: Path):
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
    def check_basic_stats(self):
        print("\n" + "="*60 + "\n基础统计\n" + "="*60)
        
        metadata = self.load_json(self.grpo_data_path / "metadata.json")
        train_data = self.load_json(self.grpo_data_path / "mobimind_decider_grpo_train.json")
        val_data = self.load_json(self.grpo_data_path / "mobimind_decider_grpo_val.json")
        
        if metadata:
            print(f"元数据: 总步骤={metadata.get('total_steps_available')}, 采样={metadata.get('total_sampled')}")
            
        actual = {'train': len(train_data) if train_data else 0, 'val': len(val_data) if val_data else 0}
        expected = {'train': metadata.get('train_count', 0) if metadata else 0, 
                   'val': metadata.get('val_count', 0) if metadata else 0}
        
        print(f"Train: {actual['train']} (预期: {expected['train']})")
        print(f"Val: {actual['val']} (预期: {expected['val']})")
        
        if actual['train'] != expected['train'] or actual['val'] != expected['val']:
            self.issues.append("数量与 metadata 不一致")
        else:
            print("✓ 数量一致")
            
        return train_data, val_data
        
    def check_data_integrity(self, train_data, val_data):
        print("\n" + "="*60 + "\n数据完整性\n" + "="*60)
        
        all_data = (train_data or []) + (val_data or [])
        required = ['instruction', 'images', 'gt_action']
        
        missing = {f: 0 for f in required}
        for sample in all_data:
            for field in required:
                if field not in sample:
                    missing[field] += 1
                    
        for field, count in missing.items():
            if count > 0:
                print(f"❌ {count} 个样本缺少 '{field}'")
                self.issues.append(f"{count} 个样本缺少 '{field}'")
                
        if all(c == 0 for c in missing.values()):
            print("✓ 所有样本字段完整")
            
    def check_action_quality(self, train_data, val_data):
        print("\n" + "="*60 + "\nAction 质量检查\n" + "="*60)
        
        all_data = (train_data or []) + (val_data or [])
        action_types = Counter()
        param_issues = 0
        
        # 记录需要删除的样本
        train_to_remove = []
        val_to_remove = []
        images_to_remove = []
        
        train_len = len(train_data) if train_data else 0
        
        for idx, sample in enumerate(all_data):
            gt = sample.get('gt_action', {})
            action_type = gt.get('type', 'unknown')
            action_types[action_type] += 1
            
            # 参数完整性
            should_remove = False
            if action_type == 'click':
                if 'position_x' not in gt:
                    param_issues += 1
                    img_path = sample.get('images', ['unknown'])[0]
                    print(f"  ⚠️ 样本 {idx}: click 缺少 position_x, 图片: {img_path}")
                    should_remove = True
                elif 'bounds' not in gt or not gt['bounds']:
                    param_issues += 1
                    img_path = sample.get('images', ['unknown'])[0]
                    print(f"  ⚠️ 样本 {idx}: click 缺少 bounds, 图片: {img_path}")
                    should_remove = True
                elif 'target_element' not in gt or not gt['target_element']:
                    param_issues += 1
                    img_path = sample.get('images', ['unknown'])[0]
                    print(f"  ⚠️ 样本 {idx}: click 缺少 target_element, 图片: {img_path}")
                    should_remove = True
            elif action_type == 'input' and 'text' not in gt:
                param_issues += 1
            elif action_type == 'swipe' and 'direction' not in gt:
                param_issues += 1
            
            if should_remove:
                # 记录图片路径
                for img in sample.get('images', []):
                    images_to_remove.append(img)
                # 记录是 train 还是 val
                if idx < train_len:
                    train_to_remove.append(idx)
                else:
                    val_to_remove.append(idx - train_len)
                
        print("Action 分布:")
        for action, count in action_types.most_common():
            print(f"  {action}: {count} ({count/len(all_data)*100:.1f}%)")
            
        if param_issues > 0:
            print(f"⚠️ {param_issues} 个样本缺少必需参数")
            self.issues.append(f"{param_issues} 个样本缺少必需参数")
            
            if self.auto_fix and (train_to_remove or val_to_remove):
                self._fix_invalid_samples(train_data, val_data, train_to_remove, val_to_remove, images_to_remove)
        else:
            print("✓ 所有参数完整")
    
    def _fix_invalid_samples(self, train_data, val_data, train_to_remove, val_to_remove, images_to_remove):
        print(f"\n开始清理无效样本...")
        
        # 删除 train 中的无效样本
        if train_to_remove and train_data:
            train_to_remove_set = set(train_to_remove)
            new_train = [s for i, s in enumerate(train_data) if i not in train_to_remove_set]
            train_path = self.grpo_data_path / "mobimind_decider_grpo_train.json"
            self.save_json(new_train, train_path)
            print(f"  ✓ 从 train 删除 {len(train_to_remove)} 条，剩余 {len(new_train)} 条")
        
        # 删除 val 中的无效样本
        if val_to_remove and val_data:
            val_to_remove_set = set(val_to_remove)
            new_val = [s for i, s in enumerate(val_data) if i not in val_to_remove_set]
            val_path = self.grpo_data_path / "mobimind_decider_grpo_val.json"
            self.save_json(new_val, val_path)
            print(f"  ✓ 从 val 删除 {len(val_to_remove)} 条，剩余 {len(new_val)} 条")
        
        # 删除对应的图片
        deleted_images = 0
        for img_path in images_to_remove:
            if os.path.exists(img_path):
                os.remove(img_path)
                deleted_images += 1
        print(f"  ✓ 删除 {deleted_images} 个图片文件")
        
        # 更新 metadata
        metadata_path = self.grpo_data_path / "metadata.json"
        metadata = self.load_json(metadata_path)
        if metadata:
            old_train = metadata.get('train_count', 0)
            old_val = metadata.get('val_count', 0)
            new_train_count = old_train - len(train_to_remove)
            new_val_count = old_val - len(val_to_remove)
            
            metadata['train_count'] = new_train_count
            metadata['val_count'] = new_val_count
            metadata['total_sampled'] = new_train_count + new_val_count
            metadata['removed_invalid_samples'] = len(train_to_remove) + len(val_to_remove)
            
            self.save_json(metadata, metadata_path)
            print(f"  ✓ 更新 metadata: train={new_train_count}, val={new_val_count}")
        
        print("清理完成！")
            
    def check_images(self, train_data, val_data):
        if not self.should_check_images:
            return
            
        print("\n" + "="*60 + "\n图片检查\n" + "="*60)
        
        all_data = (train_data or []) + (val_data or [])
        
        all_images = set()
        for sample in all_data:
            imgs = sample.get('images', [])
            if isinstance(imgs, list):
                all_images.update(imgs)
                
        print(f"共 {len(all_images)} 个图片")
        
        missing = 0
        sizes = Counter()
        try:
            from PIL import Image
        except ImportError:
            msg = "未安装 Pillow，无法执行图片检查（可用 --no_check_images 跳过）"
            print(f"❌ {msg}")
            self.issues.append(msg)
            return

        for img_path in tqdm(all_images, desc="检查图片"):
            if not os.path.exists(img_path):
                missing += 1
            else:
                try:
                    with Image.open(img_path) as img:
                        sizes[img.size] += 1
                except:
                    pass
                    
        if missing > 0:
            print(f"❌ {missing} 个图片不存在")
            self.issues.append(f"{missing} 个图片不存在")
        else:
            print("✓ 所有图片存在")
            
        if sizes:
            print("尺寸分布:")
            for size, count in sizes.most_common():
                print(f"  {size[0]}x{size[1]}: {count}")

    def generate_samples(self, train_data, val_data):
        """生成少量 GRPO 样本，便于快速人工检查"""
        print("\n" + "="*60 + "\n示例数据导出\n" + "="*60)

        train_data = train_data or []
        val_data = val_data or []
        all_data = train_data + val_data

        if not all_data:
            print("⚠️ 数据为空，跳过示例导出")
            return

        # 先按数据拆分，再按动作类型拆分
        grouped = {
            "train": {"all": train_data, "by_action": {}},
            "val": {"all": val_data, "by_action": {}},
        }

        for split_name, split_data in [("train", train_data), ("val", val_data)]:
            by_action = {}
            for sample in split_data:
                action_type = sample.get("gt_action", {}).get("type", "unknown")
                by_action.setdefault(action_type, []).append(sample)
            grouped[split_name]["by_action"] = by_action

        # 每个 split 抽样 5 条，每个 action 最多抽 3 条
        samples = {
            "overview": {
                "train_count": len(train_data),
                "val_count": len(val_data),
                "total_count": len(all_data),
            },
            "random_samples": {
                "train": random.sample(train_data, min(5, len(train_data))) if train_data else [],
                "val": random.sample(val_data, min(5, len(val_data))) if val_data else [],
            },
            "action_samples": {
                "train": {},
                "val": {},
            },
        }

        for split_name in ["train", "val"]:
            by_action = grouped[split_name]["by_action"]
            for action_type, action_samples in sorted(by_action.items()):
                samples["action_samples"][split_name][action_type] = random.sample(
                    action_samples, min(3, len(action_samples))
                )

        output_path = Path(__file__).parent / "grpo_data_samples.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)

        print(f"✓ 示例数据已保存到: {output_path}")
        print(
            "  Train 随机样本: {}/{} | Val 随机样本: {}/{}".format(
                len(samples["random_samples"]["train"]),
                len(train_data),
                len(samples["random_samples"]["val"]),
                len(val_data),
            )
        )
        print("  Action 分组导出:")
        for split_name in ["train", "val"]:
            action_map = samples["action_samples"][split_name]
            if not action_map:
                print(f"    {split_name}: 无")
                continue
            stats = ", ".join([f"{k}:{len(v)}" for k, v in action_map.items()])
            print(f"    {split_name}: {stats}")
                
    def run(self):
        print(f"检查 GRPO 数据集: {self.grpo_data_path}")
        
        train_data, val_data = self.check_basic_stats()
        self.check_data_integrity(train_data, val_data)
        self.check_action_quality(train_data, val_data)
        self.check_images(train_data, val_data)
        self.generate_samples(train_data, val_data)
        
        print("\n" + "="*60 + "\n检查完成\n" + "="*60)
        if self.issues:
            print(f"发现 {len(self.issues)} 个问题:")
            for issue in self.issues:
                print(f"  - {issue}")
        else:
            print("✓ 所有检查通过")


def main():
    parser = argparse.ArgumentParser(description="GRPO 数据集检查")
    parser.add_argument("--grpo_data_path", type=str,
                        default="/home/agent/mobiAgent/MobiAgent/workspace/data/training_data/grpo_data")
    parser.add_argument("--no_check_images", action="store_true")
    parser.add_argument("--auto_fix", action="store_true", 
                        help="自动删除缺少 bounds 的 click 样本")
    args = parser.parse_args()
    
    checker = GRPODataChecker(args.grpo_data_path, 
                              check_images=not args.no_check_images,
                              auto_fix=args.auto_fix)
    checker.run()


if __name__ == "__main__":
    main()
