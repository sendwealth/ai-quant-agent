#!/usr/bin/env python3
"""
移除所有模拟数据脚本

扫描并移除项目中的所有模拟数据：
1. core/data_manager.py - 移除 mock_data 方法
2. agents/growth_analyst.py - 移除 mock_data
3. agents/analysis_agent.py - 移除模拟数据生成
4. 其他文件 - 检查并清理
"""

import re
from pathlib import Path

def remove_mock_from_file(file_path):
    """从文件中移除模拟数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 1. 移除 mock_data 字典定义
    content = re.sub(
        r'mock_data\s*=\s*\{[^}]+\}',
        '# mock_data 已移除 - 使用真实数据',
        content,
        flags=re.MULTILINE | re.DOTALL
    )
    
    # 2. 移除返回 mock_data 的语句
    content = re.sub(
        r'return\s+mock_data\.get\([^)]+\)',
        'raise DataFetchError("无法获取真实数据，请检查网络或股票代码")',
        content
    )
    
    content = re.sub(
        '',
        content,
        flags=re.MULTILINE
    )
    
    # 4. 移除模拟数据降级逻辑
    content = re.sub(
        'raise DataFetchError("数据获取失败")\n',
        content,
        flags=re.DOTALL
    )
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def scan_project():
    """扫描项目中的模拟数据"""
    project_dir = Path('/Users/rowan/clawd/projects/ai-quant-agent')
    
    print("\n" + "=" * 90)
    print("🔍 扫描项目中的模拟数据")
    print("=" * 90)
    
    # 查找包含模拟数据的文件
    files_with_mock = []
    
    for py_file in project_dir.rglob('*.py'):
        # 跳过虚拟环境
        if '.venv' in str(py_file) or 'venv' in str(py_file):
            continue
        
        with open(py_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含模拟数据
        if re.search(r'mock_data|模拟数据|MOCK', content):
            files_with_mock.append(py_file)
    
    print(f"\n找到 {len(files_with_mock)} 个文件包含模拟数据：")
    for i, file_path in enumerate(files_with_mock, 1):
        rel_path = file_path.relative_to(project_dir)
        print(f"  {i}. {rel_path}")
    
    return files_with_mock

def remove_all_mock_data():
    """移除所有模拟数据"""
    files_with_mock = scan_project()
    
    if not files_with_mock:
        print("\n✅ 未发现模拟数据")
        return
    
    print(f"\n" + "=" * 90)
    print("🔧 开始移除模拟数据")
    print("=" * 90)
    
    fixed_count = 0
    
    for file_path in files_with_mock:
        rel_path = file_path.relative_to('/Users/rowan/clawd/projects/ai-quant-agent')
        
        if remove_mock_from_file(file_path):
            print(f"  ✅ 已修复: {rel_path}")
            fixed_count += 1
        else:
            print(f"  ⚠️  需要手动处理: {rel_path}")
    
    print(f"\n" + "=" * 90)
    print(f"✅ 修复完成：{fixed_count}/{len(files_with_mock)} 个文件")
    print("=" * 90)
    
    # 验证
    print(f"\n🔍 验证修复结果...")
    remaining = scan_project()
    
    if remaining:
        print(f"\n⚠️  仍有 {len(remaining)} 个文件包含模拟数据，需要手动处理")
    else:
        print(f"\n✅ 所有模拟数据已移除！")

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("🚀 移除所有模拟数据")
    print("=" * 90)
    print("\n目标：")
    print("  1. ✅ 移除 mock_data 字典")
    print("  2. ✅ 移除模拟数据降级逻辑")
    print("  4. ✅ 强制使用真实数据")
    print("\n" + "=" * 90)
    
    remove_all_mock_data()
    
    print(f"\n" + "=" * 90)
    print("🎯 下一步")
    print("=" * 90)
    print("\n1. 运行测试验证修复")
    print("   pytest tests/")
    print("\n2. 更新文档")
    print("   - 强调100%真实数据")
    print("   - 移除模拟数据相关说明")
    print("\n3. 提交代码")
    print("   git add .")
    print("   git commit -m 'fix: 移除所有模拟数据'")
