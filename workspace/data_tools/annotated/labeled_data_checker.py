import os
import json
import argparse

def check_action_consistency(actions_json_path, react_json_path):
    """检查 actions 和 react 的一致性"""
    errors = []
    suspicious = []
    
    try:
        with open(actions_json_path, 'r', encoding='utf-8') as f:
            actions_data = json.load(f)
        with open(react_json_path, 'r', encoding='utf-8') as f:
            react_data = json.load(f)
    except Exception as e:
        return [f"文件读取错误: {e}"], []
    
    # 检查 react_data 格式
    if not isinstance(react_data, list):
        return [f"react.json格式错误: 应为列表，实际为{type(react_data).__name__}"], []
    
    actions = actions_data.get("actions", [])
    
    # 检查步骤数量
    if len(actions) != len(react_data):
        errors.append(f"步骤数量不匹配: actions有{len(actions)}步, react有{len(react_data)}步")
        return errors, suspicious  # 数量不匹配就不继续检查了
    
    # 逐步检查
    for i, (action, react) in enumerate(zip(actions, react_data), 1):
        # 检查 react 是否为字典
        if not isinstance(react, dict):
            errors.append(f"第{i}步: react数据格式错误，应为字典，实际为{type(react).__name__}")
            continue
        
        # 检查必需字段
        if "reasoning" not in react:
            errors.append(f"第{i}步: react缺失reasoning字段")
        if "function" not in react:
            errors.append(f"第{i}步: react缺失function字段")
            continue
        
        function = react.get("function", {})
        if not isinstance(function, dict):
            errors.append(f"第{i}步: function应为字典，实际为{type(function).__name__}")
            continue
        
        if "name" not in function:
            errors.append(f"第{i}步: function缺失name字段")
            continue
        if "parameters" not in function:
            errors.append(f"第{i}步: function缺失parameters字段")
            continue
        
        action_type = action.get("type")  # actions.json 中是 "type" 字段
        react_type = function.get("name")
        params = function.get("parameters", {})
        reasoning = react.get("reasoning", "")
        
        if not isinstance(params, dict):
            errors.append(f"第{i}步: parameters应为字典，实际为{type(params).__name__}")
            continue
        
        # 检查类型一致性
        if action_type != react_type:
            errors.append(f"第{i}步: 类型不匹配 (actions={action_type}, react={react_type})")
            continue  # 类型不匹配就跳过参数检查
        
        # 检查 click 的 target_element
        if react_type == "click":
            if "target_element" not in params:
                errors.append(f"第{i}步: click操作缺失target_element参数")
            else:
                target_element = params.get("target_element", "")
                # 可疑检查：reasoning含"输入"，click但target_element不含"搜索"
                if "输入" in reasoning and "搜索" not in target_element:
                    suspicious.append(f"第{i}步: click操作可疑 (reasoning提到'输入'但target_element='{target_element}'不含'搜索')")
        
        # 检查 swipe 的 direction
        elif react_type == "swipe":
            direction = params.get("direction")
            if not direction:
                errors.append(f"第{i}步: swipe操作缺失direction参数")
            elif direction.upper() not in ["UP", "DOWN", "LEFT", "RIGHT"]:
                errors.append(f"第{i}步: swipe操作direction无效 (值为'{direction}'，应为UP/DOWN/LEFT/RIGHT)")
        
        # 检查 input 的 text
        elif react_type == "input":
            if "text" not in params:
                errors.append(f"第{i}步: input操作缺失text参数")
        
        # wait 只检查类型一致性（已在上面检查）
        # done 不检查参数
    
    return errors, suspicious

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
    consistency_errors = {}
    suspicious_cases = {}
    
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
                
                # 执行一致性检查
                actions_path = os.path.join(root, "actions.json")
                react_path = os.path.join(root, "react.json")
                errors, suspicious = check_action_consistency(actions_path, react_path)
                if errors:
                    consistency_errors[rel_path] = errors
                if suspicious:
                    suspicious_cases[rel_path] = suspicious
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
    
    # 输出一致性检查结果
    if consistency_errors:
        print(f"{'='*60}")
        print(f"❌ 数据一致性错误汇总 ({len(consistency_errors)} 个目录)")
        print(f"{'='*60}\n")
        for rel_path, errors in sorted(consistency_errors.items()):
            print(f"📁 {rel_path}")
            for error in errors:
                print(f"   ❌ {error}")
            print()
    else:
        print(f"{'='*60}")
        print("✅ 所有数据一致性检查通过")
        print(f"{'='*60}\n")
    
    # 输出可疑案例
    if suspicious_cases:
        print(f"{'='*60}")
        print(f"⚠️  可疑数据汇总 ({len(suspicious_cases)} 个目录)")
        print(f"{'='*60}\n")
        for rel_path, suspicious in sorted(suspicious_cases.items()):
            print(f"📁 {rel_path}")
            for item in suspicious:
                print(f"   ⚠️  {item}")
            print()
    
    if not missing_actions and not missing_react and not cleaned_react_files and not consistency_errors:
        print("✅ 所有检查通过，数据完整且一致")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check data integrity and optionally clean error files')
    parser.add_argument('--data-path', type=str, required=True, help='Data directory path')
    parser.add_argument('--clean-errors', action='store_true', help='Delete parse.error files if found')
    parser.add_argument('--clean-react', type=str, help='Delete react.json files for specific task type (e.g., livestream, type1, type2)')
    
    args = parser.parse_args()
    check_data_integrity(args.data_path, args.clean_errors, args.clean_react)
