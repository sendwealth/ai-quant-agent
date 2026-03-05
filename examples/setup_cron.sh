#!/bin/bash
# 设置自动定时任务
# Setup Auto Trading Cron Job

echo "设置升级版自动交易系统定时任务..."
echo ""

# 添加cron任务（每天15:30运行）
CRON_JOB="30 15 * * 1-5 cd /Users/rowan/clawd/ai-quant-agent && /usr/bin/python3 examples/upgraded_auto_trading_bot.py >> logs/upgraded_trading.log 2>&1"

# 检查是否已存在
if crontab -l | grep -q "upgraded_auto_trading_bot.py"; then
    echo "⚠️  定时任务已存在"
else
    # 添加任务
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
    echo "✅ 定时任务已添加"
fi

echo ""
echo "当前定时任务:"
crontab -l | grep "upgraded_auto_trading_bot.py"

echo ""
echo "✅ 设置完成！"
echo "系统将在每天15:30（周一到周五）自动运行"
