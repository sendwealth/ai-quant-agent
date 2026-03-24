#!/usr/bin/env python3
"""
端到端测试脚本 - ClawTeam AI 对冲基金

测试流程：
1. 运行所有分析师生成信号
2. Risk Manager 汇总信号
3. 通过 ClawTeam 消息传递报告
4. 更新任务状态
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def run_command(cmd: str, description: str = ""):
    """运行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"命令: {cmd}\n")

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)

    if result.returncode != 0:
        print(f"❌ 错误: {result.stderr}")
        return False

    return True


def main():
    print("=" * 60)
    print("🦞 ClawTeam AI 对冲基金 - 端到端测试")
    print("=" * 60)

    stock_code = "300750"  # 宁德时代

    # 1. 运行所有分析师
    analysts = [
        ("buffett_analyst.py", "Buffett Analyst (价值投资)"),
        ("growth_analyst.py", "Growth Analyst (成长投资)"),
        ("technical_analyst.py", "Technical Analyst (技术分析)"),
        ("fundamentals_analyst.py", "Fundamentals Analyst (基本面)"),
        ("sentiment_analyst.py", "Sentiment Analyst (情绪分析)"),
    ]

    print("\n📊 Phase 1: 运行所有分析师...")
    for script, desc in analysts:
        cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && python3 agents/{script} --stock {stock_code}"
        if not run_command(cmd, desc):
            print(f"⚠️  {desc} 失败，但继续测试...")

    # 2. Risk Manager 汇总
    print("\n📊 Phase 2: Risk Manager 汇总分析...")
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && python3 agents/risk_manager.py"
    if not run_command(cmd, "Risk Manager 汇总"):
        print("⚠️  Risk Manager 汇总失败")

    # 3. ClawTeam 任务管理
    print("\n📊 Phase 3: ClawTeam 任务管理...")

    # 创建任务
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam task create quant-fund '分析{stock_code}' -o buffett-analyst"
    run_command(cmd, "创建任务")

    # 查看任务
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam task list quant-fund"
    run_command(cmd, "查看任务列表")

    # 更新任务
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam task update quant-fund $(clawteam task list quant-fund --json | python3 -c 'import sys, json; print(json.load(sys.stdin)[\"tasks\"][0][\"id\"])') --status completed"
    run_command(cmd, "更新任务状态")

    # 4. ClawTeam 消息传递
    print("\n📊 Phase 4: ClawTeam 消息传递...")

    # 发送消息
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam inbox send quant-fund risk-manager '分析完成：{stock_code} BUY 信号'"
    run_command(cmd, "发送消息给 Risk Manager")

    # 接收消息
    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam inbox receive quant-fund --agent risk-manager"
    run_command(cmd, "Risk Manager 接收消息")

    # 5. 查看最终状态
    print("\n📊 Phase 5: 最终状态...")

    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam team status quant-fund"
    run_command(cmd, "团队状态")

    cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam board show quant-fund"
    run_command(cmd, "团队看板")

    print("\n" + "=" * 60)
    print("✅ 端到端测试完成！")
    print("=" * 60)
    print("\n📁 生成的文件:")
    print(f"  - 信号文件: {PROJECT_ROOT}/data/signals/*.json")
    print(f"  - 风险报告: {PROJECT_ROOT}/data/reports/*.json")
    print("\n💡 下一步:")
    print("  1. 查看风险报告: cat data/reports/risk_report_*.json")
    print("  2. 监控团队: clawteam board attach quant-fund")
    print("  3. Spawn 真实 agents: clawteam spawn tmux openclaw --team quant-fund --agent-name buffett-analyst --task '分析'")


if __name__ == "__main__":
    main()
