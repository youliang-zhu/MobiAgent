import os
import subprocess
import argparse
from datetime import datetime

def get_next_number(output_dir):
    """获取下一个编号"""
    if not os.path.exists(output_dir):
        return 1
    
    existing_files = [f for f in os.listdir(output_dir) if f.startswith('unexpected_') and f.endswith('.jpg')]
    if not existing_files:
        return 1
    
    # 提取所有编号
    numbers = []
    for f in existing_files:
        try:
            num = int(f.replace('unexpected_', '').replace('.jpg', ''))
            numbers.append(num)
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def capture_screenshot(output_dir, device_id=None):
    """通过 ADB 截图并保存"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查 ADB 连接
    try:
        cmd = ["adb"]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.append("devices")
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if "device" not in result.stdout:
            print("错误: 未检测到 ADB 设备连接")
            print("请确保:")
            print("1. 手机已通过 USB 连接")
            print("2. 已开启开发者选项和 USB 调试")
            print("3. 已授权此计算机的 USB 调试请求")
            return False
    except subprocess.CalledProcessError as e:
        print(f"错误: ADB 命令执行失败: {e}")
        return False
    except FileNotFoundError:
        print("错误: 未找到 ADB 工具")
        print("请确保已安装 Android SDK Platform Tools")
        return False
    
    # 获取下一个编号
    next_num = get_next_number(output_dir)
    filename = f"unexpected_{next_num:03d}.jpg"
    local_path = os.path.join(output_dir, filename)
    
    # 远程截图路径
    remote_path = "/sdcard/screenshot_temp.png"
    
    try:
        # 执行截图
        cmd = ["adb"]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["shell", "screencap", "-p", remote_path])
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 拉取到本地
        cmd = ["adb"]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["pull", remote_path, local_path])
        subprocess.run(cmd, check=True, capture_output=True)
        
        # 删除远程临时文件
        cmd = ["adb"]
        if device_id:
            cmd.extend(["-s", device_id])
        cmd.extend(["shell", "rm", remote_path])
        subprocess.run(cmd, check=True, capture_output=True)
        
        print(f"✓ 截图已保存: {os.path.abspath(local_path)}")
        print(f"  当前共有 {next_num} 张意外图片")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"错误: 截图失败: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="截取手机屏幕并保存为意外图片数据")
    parser.add_argument("--output_dir", type=str,
                       default=os.path.expanduser("~/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/unexpected_data"),
                       help="输出目录 (default: ~/mobiAgent/MobiAgent/tools_for_reproduction/generated_data/unexpected_data)")
    parser.add_argument("--device_id", type=str, default=None,
                       help="ADB 设备 ID (可选，如果只连接一个设备可不指定)")
    
    args = parser.parse_args()
    
    capture_screenshot(args.output_dir, args.device_id)
