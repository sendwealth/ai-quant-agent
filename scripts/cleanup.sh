#!/bin/bash
# 项目清理脚本 - 删除冗余文件

echo "🧹 开始清理项目冗余文件..."
echo ""

# 1. 删除旧的报告文件（保留最新5个）
echo "1️⃣ 清理旧报告文件..."
cd data/reports
# 保留最新的complete_backtest和dynamic_selection
ls -t complete_backtest_*.json 2>/dev/null | tail -n +2 | xargs rm -f
ls -t dynamic_selection_*.json 2>/dev/null | tail -n +6 | xargs rm -f
ls -t full_backtest_*.json 2>/dev/null | xargs rm -f
ls -t comprehensive_backtest_*.json 2>/dev/null | xargs rm -f
echo "  ✅ 保留最新报告"

# 2. 删除旧的信号文件
echo ""
echo "2️⃣ 清理旧信号文件..."
cd ../signals
# 保留最新的（今天生成的）
find . -name "*.json" -mtime +1 -delete 2>/dev/null
echo "  ✅ 保留最新信号"

# 3. 删除.archive目录
cd ../..
if [ -d ".archive" ]; then
    echo ""
    echo "3️⃣ 删除归档目录..."
    rm -rf .archive
    echo "  ✅ .archive已删除"
fi

# 4. 删除Python缓存
echo ""
echo "4️⃣ 清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "  ✅ Python缓存已清理"

# 5. 删除临时文件
echo ""
echo "5️⃣ 清理临时文件..."
rm -f *.tmp *.log 2>/dev/null
rm -f .DS_Store 2>/dev/null
echo "  ✅ 临时文件已删除"

echo ""
echo "✅ 清理完成！"

# 统计
echo ""
echo "📊 清理后统计:"
echo "  报告文件: $(ls data/reports/*.json 2>/dev/null | wc -l | tr -d ' ')"
echo "  信号文件: $(ls data/signals/*.json 2>/dev/null | wc -l | tr -d ' ')"
