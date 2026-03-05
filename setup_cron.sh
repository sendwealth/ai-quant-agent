#!/bin/bash

# 模拟盘自动设置脚本

echo "🚀 模拟盘自动设置向导"
echo "=================================================="
echo ""

# 检查是否在项目目录
if [ ! -f "examples/auto_trading_bot.py" ]; then
    echo "❌ 请在项目根目录运行此脚本"
    echo "cd /Users/rowan/clawd/ai-quant-agent"
    echo "bash setup_cron.sh"
    exit 1
fi

echo "📍 当前目录: $(pwd)"
echo ""

# 1. 检查脚本权限
echo "1️⃣ 检查脚本权限..."
if [ ! -x "daily_run.sh" ]; then
    echo "  设置执行权限..."
    chmod +x daily_run.sh
    chmod +x examples/*.py
    echo "  ✅ 权限设置完成"
else
    echo "  ✅ 权限正常"
fi
echo ""

# 2. 检查日志目录
echo "2️⃣ 检查日志目录..."
if [ ! -d "logs" ]; then
    echo "  创建日志目录..."
    mkdir -p logs
    echo "  ✅ 日志目录已创建"
else
    echo "  ✅ 日志目录存在"
fi
echo ""

# 3. 备份现有cron
echo "3️⃣ 备份现有cron任务..."
crontab -l > /tmp/crontab_backup_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null
echo "  ✅ 备份完成: /tmp/crontab_backup_*.txt"
echo ""

# 4. 添加cron任务
echo "4️⃣ 设置每日自动运行..."
echo ""

# 获取项目路径
PROJECT_DIR=$(pwd)

# 检查是否已经有相同的cron任务
if crontab -l 2>/dev/null | grep -q "daily_run.sh"; then
    echo "  ⚠️  检测到已有自动运行任务"
    echo "  如需更新，请先删除旧任务："
    echo "  crontab -e"
    echo ""
else
    echo "  添加每日自动运行任务..."
    echo "  时间: 每周一到周五 15:30"
    echo "  脚本: $PROJECT_DIR/daily_run.sh"
    echo ""

    # 添加cron任务
    (crontab -l 2>/dev/null; echo "30 15 * * 1-5 cd $PROJECT_DIR && ./daily_run.sh >> logs/auto_trading.log 2>&1") | crontab -

    echo "  ✅ 每日自动运行已设置"
fi
echo ""

# 5. 询问是否设置每周数据更新
echo "5️⃣ 是否设置每周自动更新数据？"
echo "  (建议设置，每周日18:00更新)"
echo ""
read -p "  设置吗？(y/n): " setup_weekly

if [ "$setup_weekly" = "y" ] || [ "$setup_weekly" = "Y" ]; then
    echo ""
    echo "  添加每周数据更新任务..."
    (crontab -l 2>/dev/null; echo "0 18 * * 0 cd $PROJECT_DIR && python3 examples/fetch_tushare_auto.py >> logs/data_update.log 2>&1") | crontab -
    echo "  ✅ 每周数据更新已设置"
fi
echo ""

# 6. 显示当前cron任务
echo "6️⃣ 当前cron任务列表:"
echo "=================================================="
crontab -l 2>/dev/null | grep -v "^#" | grep -v "^$"
echo "=================================================="
echo ""

# 7. 测试运行
echo "7️⃣ 是否测试运行一次？"
read -p "  测试吗？(y/n): " test_run

if [ "$test_run" = "y" ] || [ "$test_run" = "Y" ]; then
    echo ""
    echo "  运行自动交易机器人..."
    echo "  =================================================="
    python3 examples/auto_trading_bot.py
    echo ""
    echo "  ✅ 测试完成"
fi
echo ""

# 8. 完成
echo "🎉 设置完成！"
echo "=================================================="
echo ""
echo "📊 系统状态:"
echo "  - 每日运行: 15:30 (周一到周五)"
echo "  - 日志位置: $PROJECT_DIR/logs/auto_trading.log"
echo "  - 持仓文件: $PROJECT_DIR/data/auto_portfolio.json"
echo ""
echo "💡 常用命令:"
echo "  - 查看状态: python3 examples/status_monitor.py"
echo "  - 查看日志: tail -f logs/auto_trading.log"
echo "  - 手动运行: python3 examples/auto_trading_bot.py"
echo "  - 查看cron: crontab -l"
echo "  - 编辑cron: crontab -e"
echo ""
echo "📅 时间表:"
echo "  - 明天 15:30: 首次自动运行"
echo "  - 每天 15:30: 自动检测信号"
echo "  - 1-2周内: 等待买入信号"
echo "  - 3个月后: 评估结果"
echo ""
echo "✅ 所有设置已完成！系统将自动运行！"
echo ""
