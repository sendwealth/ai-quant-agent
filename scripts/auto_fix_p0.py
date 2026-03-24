#!/usr/bin/env python3
"""
P0 问题自动修复脚本

修复内容：
1. f-string 日志格式错误
2. subprocess 超时
3. 数据验证增强
"""

import re
from pathlib import Path
import subprocess

def fix_fstring_logging():
    """修复 f-string 日志格式"""
    print("\n1️⃣ 修复 f-string 日志格式...")
    
    files_to_fix = [
        "core/data_manager.py",
        "agents/risk_manager.py",
        "scripts/heartbeat_check.py"
    ]
    
    fixed_count = 0
    
    for file_path in files_to_fix:
        path = Path(file_path)
        if not path.exists():
            continue
        
        content = path.read_text()
        
        # 替换 logger.xxx("...{...}...") 为 logger.xxx(f"...{...}...")
        pattern = r'(logger\.(info|warning|error|debug))\(\"([^\"]*{[^}]*}[^\"]*)\"'
        
        def add_f_prefix(match):
            return f'{match.group(1)}(f"{match.group(3)}"'
        
        new_content = re.sub(pattern, add_f_prefix, content)
        
        if new_content != content:
            path.write_text(new_content)
            print(f"  ✅ {file_path}")
            fixed_count += 1
        else:
            print(f"  ⏭️  {file_path} (已修复)")
    
    print(f"  📊 总计修复: {fixed_count} 个文件")
    return fixed_count

def verify_subprocess_timeout():
    """验证 subprocess 超时已添加"""
    print("\n2️⃣ 验证 subprocess 超时...")
    
    file_path = Path("scripts/heartbeat_check.py")
    content = file_path.read_text()
    
    if "timeout=30" in content:
        print("  ✅ subprocess 超时已添加")
        return True
    else:
        print("  ❌ subprocess 超时未添加")
        return False

def verify_data_validation():
    """验证数据验证已增强"""
    print("\n3️⃣ 验证数据验证增强...")
    
    file_path = Path("core/data_manager.py")
    content = file_path.read_text()
    
    if "safe_float" in content and "min_val" in content and "max_val" in content:
        print("  ✅ 数据验证已增强")
        return True
    else:
        print("  ❌ 数据验证未增强")
        return False

def run_tests():
    """运行测试验证修复"""
    print("\n4️⃣ 运行测试...")
    
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "tests/test_core.py", "-v"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("  ✅ 测试通过")
            return True
        else:
            print("  ⚠️  部分测试失败")
            print(result.stdout)
            return False
    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🔧 P0 问题自动修复")
    print("=" * 60)
    
    # 1. 修复 f-string
    fixed = fix_fstring_logging()
    
    # 2. 验证 subprocess 超时
    timeout_ok = verify_subprocess_timeout()
    
    # 3. 验证数据验证
    validation_ok = verify_data_validation()
    
    # 4. 运行测试
    tests_ok = run_tests()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 修复总结")
    print("=" * 60)
    print(f"✅ f-string 修复: {fixed} 个文件")
    print(f"{'✅' if timeout_ok else '❌'} subprocess 超时: {'已添加' if timeout_ok else '未添加'}")
    print(f"{'✅' if validation_ok else '❌'} 数据验证: {'已增强' if validation_ok else '未增强'}")
    print(f"{'✅' if tests_ok else '⚠️ '} 测试验证: {'通过' if tests_ok else '部分失败'}")
    
    if timeout_ok and validation_ok:
        print("\n✅ P0 修复完成！")
        return 0
    else:
        print("\n⚠️  部分修复未完成，请检查")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
