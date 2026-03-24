#!/bin/bash
# 手动配置指南 - 定时任务和监控

echo "======================================================================"
echo "⏰ AI 对冲基金系统 - 定时任务配置指南"
echo "======================================================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT=~/clawd/projects/ai-quant-agent

echo ""
echo "1️⃣  安装 tmux"
echo "======================================================================"
echo ""
echo "macOS:"
echo "  brew install tmux"
echo ""
echo "Ubuntu/Debian:"
echo "  sudo apt update && sudo apt install tmux"
echo ""
echo "验证安装:"
echo "  tmux -V"
echo ""

echo "2️⃣  安装 Redis（可选）"
echo "======================================================================"
echo ""
echo "macOS:"
echo "  brew install redis"
echo "  brew services start redis"
echo ""
echo "Ubuntu/Debian:"
echo "  sudo apt install redis-server"
echo "  sudo systemctl start redis"
echo ""
echo "验证安装:"
echo "  redis-cli ping"
echo ""

echo "3️⃣  配置定时任务"
echo "======================================================================"
echo ""
echo -e "${YELLOW}需要手动执行:${NC}"
echo ""
echo "  # 编辑 crontab"
echo "  crontab -e"
echo ""
echo "  # 添加以下任务:"
echo ""
cat << 'EOF'
# ========================================
# AI量化对冲基金 - 定时任务
# ========================================

# 每30分钟运行分析
*/30 * * * * cd ~/clawd/projects/ai-quant-agent && source .venv/bin/activate && python3 scripts/quickstart.py --stocks 300750 002475 601318 600276 >> logs/analysis_$(date +\%Y\%m\%d).log 2>&1

# 每小时运行监控检查
0 * * * * cd ~/clawd/projects/ai-quant-agent && source .venv/bin/activate && python3 scripts/realtime_monitor.py --once >> logs/monitor_$(date +\%Y\%m\%d).log 2>&1

# 每天早上8点发送日报
0 8 * * * cd ~/clawd/projects/ai-quant-agent && source .venv/bin/activate && python3 scripts/daily_report.py >> logs/daily_report_$(date +\%Y\%m\%d).log 2>&1

# 每周一早上9点发送周报
0 9 * * 1 cd ~/clawd/projects/ai-quant-agent && source .venv/bin/activate && python3 scripts/weekly_report.py >> logs/weekly_report_$(date +\%Y\%m\%d).log 2>&1

# 每天凌晨2点清理旧日志（保留7天）
0 2 * * * find ~/clawd/projects/ai-quant-agent/logs -name "*.log" -mtime +7 -delete
EOF

echo ""
echo "4️⃣  验证定时任务"
echo "======================================================================"
echo ""
echo "  # 查看当前定时任务"
echo "  crontab -l"
echo ""
echo "  # 查看定时任务日志"
echo "  tail -f ~/clawd/projects/ai-quant-agent/logs/*.log"
echo ""

echo "5️⃣  测试命令"
echo "======================================================================"
echo ""
echo -e "${GREEN}推荐先手动测试每个命令:${NC}"
echo ""
echo "  # 测试分析"
echo "  cd $PROJECT_ROOT && source .venv/bin/activate"
echo "  python3 scripts/quickstart.py --stocks 300750 002475"
echo ""
echo "  # 测试监控"
echo "  python3 scripts/realtime_monitor.py --once"
echo ""
echo "  # 测试日报（需要实现）"
echo "  python3 scripts/daily_report.py"
echo ""

echo "6️⃣  监控和日志"
echo "======================================================================"
echo ""
echo "日志位置:"
echo "  $PROJECT_ROOT/logs/"
echo ""
echo "监控日志:"
echo "  tail -f $PROJECT_ROOT/logs/monitor_*.log"
echo ""
echo "分析日志:"
echo "  tail -f $PROJECT_ROOT/logs/analysis_*.log"
echo ""

echo "7️⃣  Systemd 服务（Linux 可选）"
echo "======================================================================"
echo ""
echo "如果要在 Linux 上配置 systemd 服务:"
echo ""
echo "  1. 复制服务文件:"
echo "     sudo cp $PROJECT_ROOT/deploy/ai-quant-monitor.service /etc/systemd/system/"
echo ""
echo "  2. 重新加载 systemd:"
echo "     sudo systemctl daemon-reload"
echo ""
echo "  3. 启用并启动服务:"
echo "     sudo systemctl enable ai-quant-monitor"
echo "     sudo systemctl start ai-quant-monitor"
echo ""
echo "  4. 查看状态:"
echo "     sudo systemctl status ai-quant-monitor"
echo ""
echo "  5. 查看日志:"
echo "     journalctl -u ai-quant-monitor -f"
echo ""

echo "======================================================================"
echo "✅ 配置完成检查清单"
echo "======================================================================"
echo ""
echo "必需:"
echo "  [ ] tmux 已安装 (tmux -V)"
echo "  [ ] 定时任务已配置 (crontab -l)"
echo ""
echo "可选:"
echo "  [ ] Redis 已安装并运行 (redis-cli ping)"
echo "  [ ] systemd 服务已配置（Linux）"
echo ""
echo "测试:"
echo "  [ ] 分析脚本测试通过"
echo "  [ ] 监控脚本测试通过"
echo "  [ ] 日志正常记录"
echo ""
echo "======================================================================"

# 检查 tmux 是否安装
if command -v tmux &> /dev/null; then
    echo -e "${GREEN}✓ tmux 已安装: $(tmux -V)${NC}"
else
    echo -e "${YELLOW}✗ tmux 未安装${NC}"
fi

# 检查 Redis 是否运行
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis 已运行${NC}"
    else
        echo -e "${YELLOW}✗ Redis 未运行${NC}"
    fi
else
    echo -e "${YELLOW}✗ Redis 未安装（可选）${NC}"
fi

# 检查 crontab
if crontab -l 2>/dev/null | grep -q "ai-quant-agent"; then
    echo -e "${GREEN}✓ 定时任务已配置${NC}"
else
    echo -e "${YELLOW}✗ 定时任务未配置${NC}"
fi

echo ""
echo "📚 更多信息请查看:"
echo "  - docs/PRODUCTION_READY_REPORT.md"
echo "  - docs/PHASE4-6_IMPLEMENTATION_GUIDE.md"
