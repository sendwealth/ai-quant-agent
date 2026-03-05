#!/bin/bash

# 自动交易 - 每日运行脚本

echo "🤖 自动交易机器人 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "=================================================="

# 进入项目目录
cd /Users/rowan/clawd/ai-quant-agent

# 更新数据（可选，建议每周更新一次）
# python3 examples/fetch_tushare_auto.py

# 运行自动交易
python3 examples/auto_trading_bot.py

echo ""
echo "✅ 运行完成！"
echo "查看持仓: cat data/auto_portfolio.json"
echo "查看日志: tail -20 logs/auto_trading.log"
