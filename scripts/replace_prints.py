#!/usr/bin/env python3
"""
替换 print 为 logger 的自动化脚本
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

def replace_prints_in_file(file_path: Path) -> int:
    """替换文件中的 print 为 logger"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 添加 logger 导入（如果还没有）
    if 'from utils.logger import get_logger' not in content:
        # 在文件开头的 import 之后添加
        import_pattern = r'(import sys\n)'
        if re.search(import_pattern, content):
            content = re.sub(
                import_pattern,
                r'\1\nfrom utils.logger import get_logger\nlogger = get_logger(__name__)\n',
                content,
                count=1
            )
    
    # 2. 替换 print 为 logger
    # print("✅ xxx") -> logger.info("xxx")
    # print("⚠️  xxx") -> logger.warning("xxx")
    # print("❌ xxx") -> logger.error("xxx")
    
    # info
    content = re.sub(
        r'print\(f?"✅ (.+?)"\)',
        r'logger.info("\1")',
        content
    )
    content = re.sub(
        r'print\(f?"✅ {(.+?)}"\)',
        r'logger.info(f"{\1}")',
        content
    )
    
    # warning
    content = re.sub(
        r'print\(f?"⚠️  (.+?)"\)',
        r'logger.warning("\1")',
        content
    )
    content = re.sub(
        r'print\(f?"⚠️  {(.+?)}"\)',
        r'logger.warning(f"{\1}")',
        content
    )
    
    # error
    content = re.sub(
        r'print\(f?"❌ (.+?)"\)',
        r'logger.error("\1")',
        content
    )
    content = re.sub(
        r'print\(f?"❌ {(.+?)}"\)',
        r'logger.error(f"{\1}")',
        content
    )
    
    # 普通 print -> logger.info
    content = re.sub(
        r'print\(f?"(.+?)"\)',
        r'logger.info("\1")',
        content
    )
    content = re.sub(
        r'print\(f"(.+?)"\)',
        r'logger.info(f"\1")',
        content
    )
    
    # 写回文件
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return 1
    
    return 0

def main():
    """主函数"""
    print("======================================================================")
    print("🔄 替换 print 为 logger")
    print("======================================================================")
    print()
    
    # 处理 core/ 目录
    print("1️⃣ 处理 core/ 目录...")
    core_dir = PROJECT_ROOT / "core"
    count = 0
    for py_file in core_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if replace_prints_in_file(py_file):
            print(f"   ✅ {py_file.name}")
            count += 1
    print(f"   处理了 {count} 个文件")
    print()
    
    # 处理 agents/ 目录
    print("2️⃣ 处理 agents/ 目录...")
    agents_dir = PROJECT_ROOT / "agents"
    count = 0
    for py_file in agents_dir.glob("*.py"):
        if py_file.name == "__init__.py":
            continue
        if replace_prints_in_file(py_file):
            print(f"   ✅ {py_file.name}")
            count += 1
    print(f"   处理了 {count} 个文件")
    print()
    
    print("======================================================================")
    print("✅ 替换完成")
    print("======================================================================")

if __name__ == "__main__":
    main()
