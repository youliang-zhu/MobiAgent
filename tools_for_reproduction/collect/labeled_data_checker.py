import os
import json
import argparse

def check_data_integrity(data_dir="../collect/manual/data", clean_errors=False, clean_react=None):
    if not os.path.exists(data_dir):
        print(f"目录不存在: {data_dir}")
        return
    
    total_dirs = 0
    complete_dirs = 0
    missing_react = []
    missing_actions = []
    error_files = []
    cleaned_react_files = []
    
    for root, dirs, files in os.walk(data_dir):
        dirs.sort(key=lambda x: int(x) if x.isdigit() else float('inf'))
        
        has_actions = "actions.json" in files
        has_react = "react.json" in files
        
        if has_actions or has_react:
            total_dirs += 1
            rel_path = os.path.relpath(root, data_dir)
            
            # 检查是否需要删除 react.json
            if clean_react and has_react:
                # 检查路径是否包含指定的任务类型
                if clean_react in root:
                    react_file_path = os.path.join(root, "react.json")
                    os.remove(react_file_path)
                    cleaned_react_files.append(rel_path)
                    has_react = False  # 更新状态
            
            if has_actions and has_react:
                complete_dirs += 1
            else:
                if not has_react:
                    missing_react.append(rel_path)
                if not has_actions:
                    missing_actions.append(rel_path)
            
            # 检查并处理 parse.error 文件
            if "parse.error" in files:
                error_files.append(rel_path)
                if clean_errors:
                    error_file_path = os.path.join(root, "parse.error")
                    os.remove(error_file_path)
    
    print(f"\n{'='*60}")
    print(f"数据完整性检查结果")
    print(f"{'='*60}")
    print(f"总数据目录: {total_dirs}")
    print(f"完整目录: {complete_dirs}")
    print(f"缺失文件目录: {total_dirs - complete_dirs}")
    print(f"{'='*60}\n")
    
    if cleaned_react_files:
        print(f"🗑️  已删除 react.json (任务类型: {clean_react}, 共 {len(cleaned_react_files)} 个):")
        for path in cleaned_react_files:
            print(f"   {path}")
        print()
    
    if missing_actions:
        print(f"❌ 缺失 actions.json ({len(missing_actions)} 个):")
        for path in missing_actions:
            print(f"   {path}")
        print()
    
    if missing_react:
        print(f"⚠️  缺失 react.json ({len(missing_react)} 个):")
        for path in missing_react:
            print(f"   {path}")
        print()
    
    if error_files:
        if clean_errors:
            print(f"🗑️  已删除 parse.error ({len(error_files)} 个):")
        else:
            print(f"⚠️  发现 parse.error ({len(error_files)} 个):")
        for path in error_files:
            print(f"   {path}")
        print()
        if not clean_errors:
            print("提示：使用 --clean-errors 参数可自动删除这些错误文件\n")
    
    if not missing_actions and not missing_react and not cleaned_react_files:
        print("✅ 所有数据目录完整")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check data integrity and optionally clean error files')
    parser.add_argument('--data-path', type=str, default='collect/manual/data', help='Data directory path (default: collect/manual/data)')
    parser.add_argument('--clean-errors', action='store_true', help='Delete parse.error files if found')
    parser.add_argument('--clean-react', type=str, help='Delete react.json files for specific task type (e.g., livestream, type1, type2)')
    
    args = parser.parse_args()
    check_data_integrity(args.data_path, args.clean_errors, args.clean_react)
