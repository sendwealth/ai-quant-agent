#!/bin/bash
# 量化监控定时任务配置脚本
# Quantitative Monitoring Cron Setup

set -e

echo "======================================"
echo "📊 量化监控定时任务配置"
echo "======================================"
echo ""

# 项目目录
PROJECT_DIR="$HOME/clawd/ai-quant-agent"
LOG_DIR="$PROJECT_DIR/logs"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安装"
    exit 1
fi

echo "✅ Python3: $(python3 --version)"
echo "✅ 项目目录: $PROJECT_DIR"
echo ""

# 备份当前crontab
echo "📦 备份当前crontab..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null || true

# 添加新任务
echo ""
echo "⏰ 配置定时任务..."
echo ""

# 读取当前crontab
CURRENT_CRON=$(crontab -l 2>/dev/null || true)

# 检查是否已存在
if echo "$CURRENT_CRON" | grep -q "daily_monitor.py"; then
    echo "⚠️  每日监控任务已存在，跳过"
else
    # 添加盘前分析任务 (每个交易日 8:50)
    echo "➕ 添加盘前分析任务 (交易日 8:50)"
    (echo "$CURRENT_CRON"; echo "# 量化策略盘前分析 - 每个交易日 8:50 运行"; echo "50 8 * * 1-5 cd $PROJECT_DIR && /usr/bin/python3 examples/daily_monitor.py >> logs/pre_market_\$(date +\\%Y\\%m\\%d).log 2>&1") | crontab -
fi

# 再次读取更新后的crontab
CURRENT_CRON=$(crontab -l 2>/dev/null || true)

if echo "$CURRENT_CRON" | grep -q "weekly_analysis.py"; then
    echo "⚠️  每周分析任务已存在，跳过"
else
    # 添加每周分析任务 (每周日 10:00)
    echo "➕ 添加每周分析任务 (周日 10:00)"
    (echo "$CURRENT_CRON"; echo "# 量化策略每周分析 - 每周日 10:00 运行"; echo "0 10 * * 0 cd $PROJECT_DIR && /usr/bin/python3 examples/weekly_analysis.py >> logs/weekly_\$(date +\\%Y\\%m\\%d).log 2>&1") | crontab -
fi

echo ""
echo "✅ 定时任务配置完成!"
echo ""
echo "📋 当前定时任务:"
echo "----------------------------------------"
crontab -l | grep -v "^#" | grep -v "^$"
echo "----------------------------------------"
echo ""
echo "📁 日志位置: $LOG_DIR"
echo ""
echo "🔧 管理命令:"
echo "   查看任务: crontab -l"
echo "   编辑任务: crontab -e"
echo "   删除任务: crontab -r"
echo "   查看日志: tail -f $LOG_DIR/pre_market_*.log"
echo ""
echo "======================================"
