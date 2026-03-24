#!/usr/bin/env python3
"""
快速启动脚本 - AI 对冲基金系统

一键运行：
1. 分析股票
2. 生成报告
3. 实时监控
"""

import subprocess
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_analysts(stocks):
    """运行所有分析师"""
    print("\n" + "=" * 60)
    print("📊 运行分析师...")
    print("=" * 60)

    for stock in stocks:
        print(f"\n分析 {stock}...")
        analysts = [
            "buffett_analyst.py",
            "growth_analyst.py",
            "technical_analyst.py",
            "fundamentals_analyst.py",
            "sentiment_analyst.py",
        ]

        for script in analysts:
            cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && python3 agents/{script} --stock {stock}"
            subprocess.run(cmd, shell=True, capture_output=True)
            print(f"  ✅ {script.replace('_analyst.py', '')}")


def run_risk_manager():
    """运行 Risk Manager"""
    print("\n" + "=" * 60)
    print("🛡️  风险管理...")
    print("=" * 60)

    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && python3 agents/risk_manager.py"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)


def show_dashboard():
    """显示监控仪表盘"""
    print("\n" + "=" * 60)
    print("📊 监控仪表盘")
    print("=" * 60)

    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && python3 scripts/realtime_monitor.py --once"
    subprocess.run(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description="AI 对冲基金快速启动")
    parser.add_argument(
        "--stocks",
        nargs="+",
        default=["300750", "002475"],
        help="股票代码列表（默认：宁德时代、立讯精密）",
    )
    parser.add_argument("--monitor", action="store_true", help="启动实时监控")
    parser.add_argument("--interval", type=int, default=300, help="监控间隔（秒）")

    args = parser.parse_args()

    print("=" * 60)
    print("🦞 AI 对冲基金系统 - 快速启动")
    print("=" * 60)

    # 1. 运行分析师
    run_analysts(args.stocks)

    # 2. 风险管理
    run_risk_manager()

    # 3. 显示仪表盘
    show_dashboard()

    # 4. 可选：实时监控
    if args.monitor:
        print("\n" + "=" * 60)
        print("🔄 启动实时监控...")
        print("=" * 60)

        cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && python3 scripts/realtime_monitor.py --interval {args.interval}"
        subprocess.run(cmd, shell=True)

    print("\n" + "=" * 60)
    print("✅ 完成！")
    print("=" * 60)
    print("\n📁 查看报告:")
    print(f"  - 信号: ls {PROJECT_ROOT}/data/signals/")
    print(f"  - 报告: ls {PROJECT_ROOT}/data/reports/")
    print("\n💡 下一步:")
    print("  - 实时监控: python3 scripts/quickstart.py --monitor")
    print("  - 自定义股票: python3 scripts/quickstart.py --stocks 300750 002475 601318")


if __name__ == "__main__":
    main()
