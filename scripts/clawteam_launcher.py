#!/usr/bin/env python3
"""
ClawTeam AI 对冲基金启动器

一键启动 7-agent 投资分析团队
"""

import subprocess
import sys
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
DATA_DIR = PROJECT_ROOT / "data"
SIGNALS_DIR = DATA_DIR / "signals"
REPORTS_DIR = DATA_DIR / "reports"


def run_command(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """运行 shell 命令"""
    print(f"🔧 执行: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"❌ 错误: {result.stderr}")
        sys.exit(1)
    return result


def setup_directories():
    """创建必要的目录"""
    print("📁 创建目录...")
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ 目录创建完成")


def check_clawteam():
    """检查 ClawTeam 是否安装"""
    print("🔍 检查 ClawTeam...")
    result = run_command("clawteam --help", check=False)
    if result.returncode != 0:
        print("❌ ClawTeam 未安装")
        print("💡 安装命令: pip install clawteam")
        sys.exit(1)
    print("✅ ClawTeam 已安装")


def spawn_team(team_name: str, description: str):
    """创建团队"""
    print(f"\n🦞 创建团队: {team_name}")
    cmd = f'clawteam team spawn-team {team_name} -d "{description}" -n portfolio-manager'
    result = run_command(cmd)
    print(result.stdout)
    print(f"✅ 团队 {team_name} 创建完成")


def spawn_agents(team_name: str):
    """Spawn 所有分析师 agents"""
    agents = [
        ("buffett-analyst", "价值投资分析"),
        ("growth-analyst", "成长投资分析"),
        ("technical-analyst", "技术分析"),
        ("fundamentals-analyst", "基本面分析"),
        ("sentiment-analyst", "情绪分析"),
        ("risk-manager", "风险管理"),
    ]

    print(f"\n🤖 Spawn agents...")
    for agent_name, task in agents:
        print(f"  Spawn {agent_name}...")
        cmd = f'clawteam spawn tmux openclaw --team {team_name} --agent-name {agent_name} --task "{task}"'
        result = run_command(cmd, check=False)
        if result.returncode == 0:
            print(f"  ✅ {agent_name} spawned")
        else:
            print(f"  ⚠️  {agent_name} spawn 失败: {result.stderr}")


def show_team_status(team_name: str):
    """显示团队状态"""
    print(f"\n📊 团队状态:")
    cmd = f"clawteam team status {team_name}"
    result = run_command(cmd, check=False)
    print(result.stdout)


def attach_board(team_name: str):
    """附加到团队看板"""
    print(f"\n👀 监控团队:")
    print(f"  运行: clawteam board attach {team_name}")
    print(f"  或: clawteam board serve --port 8080")


def main():
    """主函数"""
    print("=" * 60)
    print("🦞 ClawTeam AI 对冲基金启动器")
    print("=" * 60)

    # 1. 检查环境
    check_clawteam()

    # 2. 创建目录
    setup_directories()

    # 3. 创建团队
    team_name = "quant-fund"
    description = "AI量化对冲基金 - 7-agent协作系统"
    spawn_team(team_name, description)

    # 4. Spawn agents
    spawn_agents(team_name)

    # 5. 显示状态
    show_team_status(team_name)

    # 6. 提示监控
    attach_board(team_name)

    print("\n" + "=" * 60)
    print("✅ 启动完成！")
    print("=" * 60)
    print("\n📚 下一步:")
    print("1. 监控团队: clawteam board attach quant-fund")
    print("2. 查看任务: clawteam task list quant-fund")
    print("3. 发送消息: clawteam inbox send quant-fund portfolio-manager '开始分析'")
    print("\n💡 提示:")
    print("- 每个 agent 有独立的 tmux 窗口和 git worktree")
    print("- 使用 'clawteam board attach' 可以同时查看所有 agent")
    print("- 消息通过 inbox 系统传递")


if __name__ == "__main__":
    main()
