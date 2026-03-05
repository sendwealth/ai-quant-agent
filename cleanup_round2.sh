#!/bin/bash

# 第二轮清理 - 只保留核心文件

echo "🧹 第二轮清理：只保留核心文件..."
echo ""

# 创建核心文件列表
cat > /tmp/core_files.txt << 'EOF'
auto_trading_bot.py
smart_screener_v2.py
comprehensive_backtest.py
weight_optimizer.py
param_optimizer.py
portfolio_backtest.py
fetch_tushare_auto.py
trading_monitor.py
improved_strategy_v5.py
EOF

# 归档非核心文件
echo "📦 归档非核心代码..."
for file in examples/*.py; do
    filename=$(basename "$file")
    if ! grep -q "$filename" /tmp/core_files.txt; then
        mv "$file" .archive/code/ 2>/dev/null
    fi
done

echo "  ✅ 非核心代码已归档"

# 显示结果
echo ""
echo "✅ 清理完成！"
echo ""
echo "📊 最终统计:"
echo "  - 核心代码: $(ls examples/*.py 2>/dev/null | wc -l) 个文件"
echo "  - 归档代码: $(ls .archive/code/*.py 2>/dev/null | wc -l) 个文件"
echo ""
echo "📁 核心文件列表:"
ls examples/*.py 2>/dev/null | xargs -n1 basename
echo ""
echo "💡 提示:"
echo "  - 只保留了最核心的9个文件"
echo "  - 其他文件已归档到 .archive/code/"
echo "  - 项目更加清晰易维护"
