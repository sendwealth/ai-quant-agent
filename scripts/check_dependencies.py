#!/usr/bin/env python3
"""检查所有依赖是否在 requirements.txt 中

使用方法:
    python scripts/check_dependencies.py

返回码:
    0: 所有依赖完整
    1: 有缺失的依赖
"""

import os
import re
from pathlib import Path

def get_imports():
    """提取所有 import 语句"""
    imports = set()
    for py_file in Path('.').rglob('*.py'):
        # 跳过虚拟环境和测试覆盖率目录
        if any(skip in str(py_file) for skip in ['.venv', 'htmlcov', '__pycache__', '.git', 'site-packages']):
            continue
        try:
            content = py_file.read_text()
            # 匹配 import xxx 或 from xxx import
            matches = re.findall(r'^(?:import|from)\s+(\w+)', content, re.MULTILINE)
            imports.update(matches)
        except:
            pass
    return imports

def get_requirements():
    """读取 requirements.txt"""
    req_file = Path('requirements.txt')
    if not req_file.exists():
        return set()

    requirements = set()
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            # 提取包名（去掉版本号）
            pkg = re.match(r'^([a-zA-Z0-9_-]+)', line)
            if pkg:
                requirements.add(pkg.group(1).lower().replace('-', '_'))

    return requirements

def main():
    imports = get_imports()
    requirements = get_requirements()

    # 标准库模块（不需要在 requirements.txt 中）
    STDLIB = {
        'os', 'sys', 're', 'json', 'time', 'datetime', 'pathlib',
        'typing', 'collections', 'itertools', 'functools', 'abc',
        'dataclasses', 'enum', 'copy', 'math', 'random', 'logging',
        'warnings', 'contextlib', 'threading', 'multiprocessing',
        'concurrent', 'asyncio', 'subprocess', 'shutil', 'tempfile',
        'hashlib', 'hmac', 'secrets', 'base64', 'binascii',
        'struct', 'codecs', 'io', 'string', 'textwrap',
        'unicodedata', 'locale', 'calendar', 'argparse', 'optparse',
        'getopt', 'configparser', 'traceback', 'warnings', 'unittest',
        'doctest', 'pdb', 'profile', 'cProfile', 'timeit',
        'email', 'smtplib', 'pickle', 'statistics',  # 新增标准库
    }

    # 项目内部模块（不需要在 requirements.txt 中）
    LOCAL_MODULES = {
        'config', 'core', 'utils', 'trading', 'agents', 'data',
        'scripts', 'tests', 'cache', 'logs', 'reports', 'templates',
    }

    # 第三方库包名映射（import名 → pip包名）
    PACKAGE_MAP = {
        'cv2': 'opencv-python',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn',
        'bs4': 'beautifulsoup4',
        'dateutil': 'python-dateutil',
        'yaml': 'pyyaml',
        'dotenv': 'python-dotenv',
    }

    missing = []
    for imp in sorted(imports):
        if imp in STDLIB:
            continue
        if imp in LOCAL_MODULES:
            continue

        # 转换为 pip 包名
        pkg_name = PACKAGE_MAP.get(imp, imp).lower().replace('-', '_')

        if pkg_name not in requirements and imp.lower() not in requirements:
            missing.append(imp)

    if missing:
        print("❌ 缺失的依赖:")
        for m in missing:
            pkg = PACKAGE_MAP.get(m, m)
            print(f"  - {m} (pip install {pkg})")
        print("\n💡 添加到 requirements.txt:")
        for m in missing:
            pkg = PACKAGE_MAP.get(m, m)
            print(f"  {pkg}")
        return 1
    else:
        print("✅ 所有依赖都在 requirements.txt 中")
        return 0

if __name__ == '__main__':
    exit(main())
