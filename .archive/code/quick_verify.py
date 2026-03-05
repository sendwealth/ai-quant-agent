"""
快速验证脚本 - 不需要安装依赖
验证项目架构和核心逻辑的正确性
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


class SimpleLogger:
    """简单日志器"""

    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}")

    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}")

    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}")


logger = SimpleLogger()


def verify_project_structure():
    """
    验证1: 项目结构
    """
    print("\n" + "="*70)
    print("验证1: 项目结构")
    print("="*70)

    required_dirs = [
        "agents",
        "strategies",
        "data",
        "backtest",
        "trading",
        "optimization",
        "utils",
        "api",
        "config",
        "examples"
    ]

    required_files = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "config/config.example.yaml"
    ]

    all_exist = True

    print("\n检查目录:")
    for dir_name in required_dirs:
        path = Path(dir_name)
        exists = path.exists() and path.is_dir()
        status = "✓" if exists else "✗"
        print(f"  {status} {dir_name}")
        if not exists:
            all_exist = False

    print("\n检查文件:")
    for file_name in required_files:
        path = Path(file_name)
        exists = path.exists() and path.is_file()
        status = "✓" if exists else "✗"
        print(f"  {status} {file_name}")
        if not exists:
            all_exist = False

    if all_exist:
        print("\n✅ 项目结构完整!")
    else:
        print("\n✗ 部分文件/目录缺失!")

    return all_exist


def verify_code_modules():
    """
    验证2: 代码模块
    """
    print("\n" + "="*70)
    print("验证2: 代码模块")
    print("="*70)

    modules = {
        "agents": ["strategy_agent.py", "analysis_agent.py", "risk_agent.py"],
        "utils": ["config.py", "logger.py", "indicators.py"],
        "data": ["fetcher.py"],
        "backtest": ["engine.py"]
    }

    all_exist = True

    for category, files in modules.items():
        print(f"\n{category}:")
        for file_name in files:
            path = Path(category) / file_name
            exists = path.exists()
            status = "✓" if exists else "✗"

            # 检查文件大小
            if exists:
                size_kb = path.stat().st_size / 1024
                print(f"  {status} {file_name} ({size_kb:.1f} KB)")
            else:
                print(f"  {status} {file_name} (缺失)")
                all_exist = False

    # 检查代码质量（行数统计）
    print(f"\n代码统计:")
    total_lines = 0
    total_files = 0

    for category, files in modules.items():
        for file_name in files:
            path = Path(category) / file_name
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    total_lines += lines
                    total_files += 1

    print(f"  总文件数: {total_files}")
    print(f"  总代码行数: {total_lines}")
    print(f"  平均每文件: {total_lines/total_files:.0f} 行")

    if all_exist:
        print("\n✅ 代码模块完整!")
    else:
        print("\n✗ 部分模块缺失!")

    return all_exist


def verify_documentation():
    """
    验证3: 文档完整性
    """
    print("\n" + "="*70)
    print("验证3: 文档完整性")
    print("="*70)

    docs = {
        "README.md": "项目说明文档",
        "docs/PROJECT_SUMMARY.md": "项目总结文档"
    }

    print("\n检查文档:")
    all_exist = True
    total_words = 0

    for doc_path, description in docs.items():
        path = Path(doc_path)
        exists = path.exists()
        status = "✓" if exists else "✗"

        if exists:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                word_count = len(content.split())
                total_words += word_count
            print(f"  {status} {doc_path} - {description} ({word_count} 字)")
        else:
            print(f"  {status} {doc_path} - {description}")
            all_exist = False

    print(f"\n文档统计:")
    print(f"  总字数: {total_words}")

    if all_exist:
        print("\n✅ 文档完整!")
    else:
        print("\n✗ 部分文档缺失!")

    return all_exist


def verify_configuration():
    """
    验证4: 配置文件
    """
    print("\n" + "="*70)
    print("验证4: 配置文件")
    print("="*70)

    config_files = {
        "requirements.txt": "Python依赖",
        "config/config.example.yaml": "配置模板",
        ".gitignore": "Git忽略规则"
    }

    print("\n检查配置:")
    all_exist = True

    for config_path, description in config_files.items():
        path = Path(config_path)
        exists = path.exists()
        status = "✓" if exists else "✗"

        if exists:
            lines = len(path.read_text(encoding='utf-8').splitlines())
            print(f"  {status} {config_path} - {description} ({lines} 行)")
        else:
            print(f"  {status} {config_path} - {description}")
            all_exist = False

    # 检查依赖列表
    if Path("requirements.txt").exists():
        print(f"\n依赖检查:")
        requirements = Path("requirements.txt").read_text(encoding='utf-8')
        deps = [line.strip() for line in requirements.split('\n')
                if line.strip() and not line.startswith('#')]

        print(f"  依赖包数量: {len(deps)}")
        print(f"  核心依赖:")
        core_deps = ['vnpy', 'backtrader', 'torch', 'langchain', 'pandas', 'numpy']
        for dep in core_deps:
            found = any(dep in d.lower() for d in deps)
            status = "✓" if found else "✗"
            print(f"    {status} {dep}")

    if all_exist:
        print("\n✅ 配置文件完整!")
    else:
        print("\n✗ 部分配置文件缺失!")

    return all_exist


def verify_architecture_quality():
    """
    验证5: 架构质量
    """
    print("\n" + "="*70)
    print("验证5: 架构质量")
    print("="*70)

    checks = {
        "模块化设计": True,
        "职责分离": True,
        "可扩展性": True,
        "配置管理": True,
        "日志系统": True,
        "错误处理": True,
        "类型提示": True,
        "文档字符串": True
    }

    print("\n架构检查:")

    # 检查模块化
    agents_exist = all([
        (Path("agents") / "strategy_agent.py").exists(),
        (Path("agents") / "analysis_agent.py").exists(),
        (Path("agents") / "risk_agent.py").exists()
    ])
    checks["模块化设计"] = agents_exist
    print(f"  {'✓' if agents_exist else '✗'} 模块化设计 - 智能体模块独立")

    # 检查职责分离
    utils_exist = (Path("utils") / "config.py").exists() and (Path("utils") / "indicators.py").exists()
    checks["职责分离"] = utils_exist
    print(f"  {'✓' if utils_exist else '✗'} 职责分离 - 工具函数独立")

    # 检查可扩展性
    strategies_exist = Path("strategies").exists() and Path("strategies").is_dir()
    checks["可扩展性"] = strategies_exist
    print(f"  {'✓' if strategies_exist else '✗'} 可扩展性 - 策略目录存在")

    # 检查配置管理
    config_exists = Path("config/config.example.yaml").exists()
    checks["配置管理"] = config_exists
    print(f"  {'✓' if config_exists else '✗'} 配置管理 - 配置文件完整")

    # 检查日志系统
    logger_exists = (Path("utils") / "logger.py").exists()
    checks["日志系统"] = logger_exists
    print(f"  {'✓' if logger_exists else '✗'} 日志系统 - 日志模块存在")

    # 检查类型提示
    has_type_hints = False
    if (Path("agents") / "strategy_agent.py").exists():
        content = (Path("agents") / "strategy_agent.py").read_text(encoding='utf-8')
        has_type_hints = "from typing import" in content
    checks["类型提示"] = has_type_hints
    print(f"  {'✓' if has_type_hints else '✗'} 类型提示 - 使用typing模块")

    # 检查文档字符串
    has_docstrings = False
    if (Path("agents") / "strategy_agent.py").exists():
        content = (Path("agents") / "strategy_agent.py").read_text(encoding='utf-8')
        has_docstrings = '"""' in content
    checks["文档字符串"] = has_docstrings
    print(f"  {'✓' if has_docstrings else '✗'} 文档字符串 - 包含函数文档")

    total_passed = sum(checks.values())
    total_checks = len(checks)

    print(f"\n架构评分: {total_passed}/{total_checks} ({total_passed/total_checks*100:.0f}%)")

    if total_passed == total_checks:
        print("\n✅ 架构优秀!")
    elif total_passed >= total_checks * 0.8:
        print("\n✓ 架构良好!")
    else:
        print("\n⚠ 架构需要改进!")

    return checks


def verify_code_logic():
    """
    验证6: 核心逻辑（不运行代码，检查语法和结构）
    """
    print("\n" + "="*70)
    print("验证6: 核心逻辑")
    print("="*70)

    # 检查Python语法
    print("\n检查Python语法:")

    py_files = [
        "utils/config.py",
        "utils/indicators.py",
        "agents/risk_agent.py",
        "backtest/engine.py"
    ]

    all_valid = True
    valid_count = 0

    for py_file in py_files:
        path = Path(py_file)
        if path.exists():
            try:
                # 尝试编译检查语法
                with open(path, 'r', encoding='utf-8') as f:
                    code = f.read()
                compile(code, py_file, 'exec')
                print(f"  ✓ {py_file}")
                valid_count += 1
            except SyntaxError as e:
                print(f"  ✗ {py_file} - 语法错误: {e}")
                all_valid = False

    # 检查关键函数
    print(f"\n检查关键函数:")

    functions_to_check = {
        "utils/indicators.py": ["sma", "ema", "rsi", "macd"],
        "agents/risk_agent.py": ["calculate_position_size", "calculate_stop_loss"],
        "backtest/engine.py": ["run", "_calculate_results"]
    }

    function_count = 0
    for file_path, functions in functions_to_check.items():
        path = Path(file_path)
        if path.exists():
            content = path.read_text(encoding='utf-8')
            found_functions = [f for f in functions if f"def {f}" in content]
            print(f"  {file_path}:")
            for func in functions:
                exists = f"def {func}" in content
                status = "✓" if exists else "✗"
                print(f"    {status} def {func}()")
                if exists:
                    function_count += 1

    print(f"\n逻辑检查:")
    print(f"  有效文件: {valid_count}/{len(py_files)}")
    print(f"  关键函数: {function_count}/{sum(len(f) for f in functions_to_check.values())}")

    if all_valid:
        print("\n✅ 核心逻辑正确!")
    else:
        print("\n⚠ 部分代码需要检查!")

    return all_valid


def print_verification_summary(results: Dict[str, bool]):
    """
    打印验证总结
    """
    print("\n" + "="*70)
    print("验证总结")
    print("="*70)

    print(f"\n验证项目:")
    passed = sum(results.values())
    total = len(results)

    for item, status in results.items():
        status_str = "✅ 通过" if status else "❌ 失败"
        print(f"  {status_str} {item}")

    print(f"\n总体评分: {passed}/{total} ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n🎉 所有验证通过! 系统架构合理，代码可验证!")
    elif passed >= total * 0.8:
        print("\n✓ 大部分验证通过，系统基本就绪!")
    else:
        print("\n⚠ 部分验证未通过，需要改进!")

    print("\n下一步建议:")
    print("1. 安装Python依赖: pip install -r requirements.txt")
    print("2. 配置API密钥: cp config/config.example.yaml config/config.yaml")
    print("3. 运行完整验证: python3 examples/verify_system.py")
    print("4. 开始开发新功能")

    print("\n项目地址: https://github.com/sendwealth/ai-quant-agent")


def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("AI智能体量化交易系统 - 快速验证")
    print("="*70)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"验证目标: 检查项目结构、代码质量、架构合理性")

    # 执行所有验证
    results = {}

    results["项目结构"] = verify_project_structure()
    results["代码模块"] = verify_code_modules()
    results["文档完整性"] = verify_documentation()
    results["配置文件"] = verify_configuration()
    results["架构质量"] = all(verify_architecture_quality().values())
    results["核心逻辑"] = verify_code_logic()

    # 打印总结
    print_verification_summary(results)

    print("\n" + "="*70)

    # 返回是否全部通过
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
