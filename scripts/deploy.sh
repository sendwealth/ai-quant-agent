#!/bin/bash
# AI 量化对冲基金系统 - 生产环境部署脚本

set -e  # 遇到错误立即退出

echo "======================================================================"
echo "🦞 AI 量化对冲基金系统 - 生产环境部署"
echo "======================================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}项目目录: $PROJECT_ROOT${NC}"

# 1. 检查环境
echo ""
echo "======================================================================"
echo "1️⃣  检查环境依赖"
echo "======================================================================"

# Python 版本
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python 版本: $PYTHON_VERSION${NC}"

# 虚拟环境
if [ ! -d ".venv" ]; then
    echo -e "${YELLOW}⚠ 虚拟环境不存在，正在创建...${NC}"
    uv venv --python 3.11
fi

# 激活虚拟环境
source .venv/bin/activate
echo -e "${GREEN}✓ 虚拟环境已激活${NC}"

# 2. 安装依赖
echo ""
echo "======================================================================"
echo "2️⃣  安装依赖"
echo "======================================================================"

# 安装 Python 包
echo "安装 Python 依赖..."
uv pip install clawteam akshare redis

# 检查 tmux（可选）
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}⚠ tmux 未安装${NC}"
    echo "  安装命令:"
    echo "    macOS: brew install tmux"
    echo "    Ubuntu: sudo apt install tmux"
    echo ""
    echo "  或者使用 subprocess backend:"
    echo "    clawteam spawn subprocess python --team ..."
else
    echo -e "${GREEN}✓ tmux 已安装: $(tmux -V)${NC}"
fi

# 检查 Redis（可选）
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis 已运行${NC}"
    else
        echo -e "${YELLOW}⚠ Redis 未运行，启动命令: redis-server${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Redis 未安装（可选）${NC}"
fi

# 3. 创建必要目录
echo ""
echo "======================================================================"
echo "3️⃣  创建目录结构"
echo "======================================================================"

mkdir -p data/signals
mkdir -p data/reports
mkdir -p data/cache
mkdir -p logs/monitor
mkdir -p logs/clawteam

echo -e "${GREEN}✓ 目录创建完成${NC}"

# 4. 配置环境变量
echo ""
echo "======================================================================"
echo "4️⃣  配置环境变量"
echo "======================================================================"

if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# 数据源配置
AKSHARE_ENABLED=true
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_ENABLED=false

# ClawTeam 配置
CLAWTEAM_DATA_DIR=~/.clawteam
CLAWTEAM_TRANSPORT=file
CLAWTEAM_DEFAULT_BACKEND=tmux
CLAWTEAM_SKIP_PERMISSIONS=true

# 监控配置
MONITOR_INTERVAL=300
MONITOR_ENABLED=true

# 告警配置（可选）
ALERT_EMAIL=
ALERT_WEBHOOK=

# 日志配置
LOG_LEVEL=INFO
LOG_DIR=logs
EOF
    echo -e "${GREEN}✓ .env 文件已创建${NC}"
else
    echo -e "${GREEN}✓ .env 文件已存在${NC}"
fi

# 5. 测试系统
echo ""
echo "======================================================================"
echo "5️⃣  测试系统"
echo "======================================================================"

echo "测试数据管理器..."
python3 core/data_manager.py --stock 300750 --no-cache > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 数据管理器正常${NC}"
else
    echo -e "${RED}✗ 数据管理器测试失败${NC}"
fi

echo "测试分析师..."
python3 agents/buffett_analyst.py --stock 300750 > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 分析师正常${NC}"
else
    echo -e "${RED}✗ 分析师测试失败${NC}"
fi

echo "测试 Risk Manager..."
python3 agents/risk_manager.py > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Risk Manager 正常${NC}"
else
    echo -e "${RED}✗ Risk Manager 测试失败${NC}"
fi

# 6. 配置定时任务
echo ""
echo "======================================================================"
echo "6️⃣  配置定时任务（可选）"
echo "======================================================================"

echo "Cron 任务示例:"
echo ""
echo "# 每30分钟运行分析"
echo "*/30 * * * * cd $PROJECT_ROOT && source .venv/bin/activate && python3 scripts/quickstart.py --stocks 300750 002475"
echo ""
echo "# 每小时运行监控检查"
echo "0 * * * * cd $PROJECT_ROOT && source .venv/bin/activate && python3 scripts/realtime_monitor.py --once"
echo ""
echo "# 每天早上8点发送日报"
echo "0 8 * * * cd $PROJECT_ROOT && source .venv/bin/activate && python3 scripts/daily_report.py"
echo ""

read -p "是否配置定时任务？(y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "请手动编辑 crontab:"
    echo "  crontab -e"
fi

# 7. 配置 Systemd 服务（可选）
echo ""
echo "======================================================================"
echo "7️⃣  配置 Systemd 服务（可选）"
echo "======================================================================"

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Linux 系统，可以配置 systemd 服务"
    echo ""
    read -p "是否创建 systemd 服务文件？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cat > deploy/ai-quant-monitor.service << EOF
[Unit]
Description=AI Quant Agent Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_ROOT
Environment="PATH=$PROJECT_ROOT/.venv/bin"
ExecStart=$PROJECT_ROOT/.venv/bin/python3 scripts/realtime_monitor.py --interval 300
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
        echo -e "${GREEN}✓ systemd 服务文件已创建: deploy/ai-quant-monitor.service${NC}"
        echo ""
        echo "安装命令:"
        echo "  sudo cp deploy/ai-quant-monitor.service /etc/systemd/system/"
        echo "  sudo systemctl daemon-reload"
        echo "  sudo systemctl enable ai-quant-monitor"
        echo "  sudo systemctl start ai-quant-monitor"
    fi
else
    echo "非 Linux 系统，跳过 systemd 配置"
fi

# 8. 生成部署摘要
echo ""
echo "======================================================================"
echo "8️⃣  部署摘要"
echo "======================================================================"

cat << EOF
✅ 部署完成！

项目目录: $PROJECT_ROOT
Python 版本: $PYTHON_VERSION
虚拟环境: .venv

已安装组件:
  ✓ clawteam
  ✓ akshare
  ✓ redis (Python 客户端)

可选组件:
  $(command -v tmux &> /dev/null && echo "✓ tmux" || echo "✗ tmux (未安装)")
  $(command -v redis-cli &> /dev/null && echo "✓ Redis" || echo "✗ Redis (未安装)")

目录结构:
  ✓ data/signals/
  ✓ data/reports/
  ✓ data/cache/
  ✓ logs/monitor/
  ✓ logs/clawteam/

配置文件:
  ✓ .env

下一步:
  1. 编辑 .env 配置文件
  2. 运行测试: python3 scripts/quickstart.py --stocks 300750 002475
  3. 启动监控: python3 scripts/realtime_monitor.py --interval 300
  4. 配置定时任务（可选）: crontab -e
  5. 配置 systemd 服务（可选，Linux）

快速命令:
  # 运行分析
  python3 scripts/quickstart.py --stocks 300750 002475 601318 600276

  # 启动监控
  python3 scripts/realtime_monitor.py --interval 300

  # 查看团队状态
  clawteam team status quant-fund

  # 查看任务列表
  clawteam task list quant-fund

文档:
  - 完整报告: docs/CLAWTEAM_FINAL_REPORT.md
  - 快速参考: docs/CLAWTEAM_QUICKSTART.md
  - 部署指南: docs/PHASE4-6_IMPLEMENTATION_GUIDE.md
  - 进化路线: docs/EVOLUTION_ROADMAP.md

EOF

echo "======================================================================"
echo "🎉 部署完成！"
echo "======================================================================"
