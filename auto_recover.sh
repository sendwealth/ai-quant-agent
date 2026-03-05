#!/bin/bash

# 自动恢复脚本 - 修复常见问题

echo "🔧 自动恢复脚本"
echo "=================================================="
echo ""

# 1. 修复文件权限
echo "1️⃣ 检查并修复文件权限..."
chmod +x daily_run.sh 2>/dev/null
chmod +x examples/*.py 2>/dev/null
chmod +x *.sh 2>/dev/null
echo "  ✅ 权限已修复"
echo ""

# 2. 创建必要的目录
echo "2️⃣ 检查并创建必要目录..."
mkdir -p logs
mkdir -p backups
mkdir -p data
echo "  ✅ 目录已创建"
echo ""

# 3. 检查cron任务
echo "3️⃣ 检查cron任务..."
if ! crontab -l 2>/dev/null | grep -q "daily_run.sh"; then
    echo "  ⚠️  Cron任务缺失，正在添加..."
    PROJECT_DIR=$(pwd)
    (crontab -l 2>/dev/null; echo "30 15 * * 1-5 cd $PROJECT_DIR && ./daily_run.sh >> logs/auto_trading.log 2>&1") | crontab -
    echo "  ✅ Cron任务已添加"
else
    echo "  ✅ Cron任务正常"
fi
echo ""

# 4. 清理旧日志
echo "4️⃣ 清理旧日志（保留30天）..."
find logs/ -name "*.log" -mtime +30 -delete 2>/dev/null
echo "  ✅ 旧日志已清理"
echo ""

# 5. 清理旧备份
echo "5️⃣ 清理旧备份（保留30天）..."
find backups/ -name "*.json" -mtime +30 -delete 2>/dev/null
echo "  ✅ 旧备份已清理"
echo ""

# 6. 检查数据文件
echo "6️⃣ 检查数据文件..."
DATA_COUNT=$(ls data/real_*.csv 2>/dev/null | wc -l)
if [ "$DATA_COUNT" -lt 4 ]; then
    echo "  ⚠️  数据文件不足 ($DATA_COUNT/4)"
    echo "  运行数据更新: python3 examples/fetch_tushare_auto.py"
else
    echo "  ✅ 数据文件完整 ($DATA_COUNT个)"
fi
echo ""

# 7. 运行健康检查
echo "7️⃣ 运行健康检查..."
python3 examples/health_check.py
HEALTH_EXIT=$?
echo ""

if [ $HEALTH_EXIT -eq 0 ]; then
    echo "✅ 系统恢复完成！所有检查通过！"
else
    echo "⚠️  系统恢复完成，但部分检查未通过"
    echo "请查看上面的详细信息"
fi

echo ""
echo "💡 后续操作:"
echo "  - 查看状态: python3 examples/status_monitor.py"
echo "  - 手动运行: python3 examples/auto_trading_bot.py"
echo "  - 查看日志: tail -f logs/auto_trading.log"
