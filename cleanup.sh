#!/bin/bash

# 项目清理和维护脚本

echo "🧹 开始清理项目..."

# 1. 清理Python缓存
echo "清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "  ✅ Python缓存已清理"

# 2. 清理临时文件
echo "清理临时文件..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.bak" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null
echo "  ✅ 临时文件已清理"

# 3. 确保必要目录存在
echo "检查目录..."
mkdir -p logs
mkdir -p data
mkdir -p docs/archive
echo "  ✅ 目录检查完成"

# 4. 显示项目结构
echo ""
echo "📁 项目结构:"
echo "  ├── README.md          # 项目概览"
echo "  ├── START_HERE.md      # 快速开始"
echo "  ├── PROJECT_SUMMARY.md # 项目总结"
echo "  ├── docs/              # 文档"
echo "  ├── examples/          # 代码"
echo "  ├── data/              # 数据"
echo "  └── logs/              # 日志"

# 5. 显示核心文件
echo ""
echo "📄 核心文档:"
echo "  - docs/FINAL_CONCLUSION.md          # 最终结论"
echo "  - docs/AUTO_TRADING_GUIDE.md        # 自动化交易指南"
echo "  - docs/QUICK_REFERENCE_CARD.md      # 快速参考卡"
echo "  - docs/COMPREHENSIVE_BACKTEST_REPORT.md  # 验证报告"

echo ""
echo "💻 核心代码:"
echo "  - examples/auto_trading_bot.py      # 自动交易机器人"
echo "  - examples/smart_screener_v2.py     # 智能选股"
echo "  - examples/comprehensive_backtest.py # 充分回测"

# 6. 显示使用方法
echo ""
echo "🚀 快速开始:"
echo "  python3 examples/auto_trading_bot.py"

echo ""
echo "✅ 清理完成！"
