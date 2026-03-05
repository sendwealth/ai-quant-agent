#!/bin/bash

# 仓库清理脚本 - 删除不必要的文件

echo "🧹 开始清理仓库..."
echo ""

# 创建备份目录
mkdir -p .archive/code
mkdir -p .archive/docs

# === 1. 归档中间版本代码 ===
echo "📦 归档中间版本代码..."

# V1-V3策略文件（保留V4和V5用于对比）
mv examples/adaptive_strategy.py .archive/code/ 2>/dev/null
mv examples/advanced_strategy_v2.py .archive/code/ 2>/dev/null
mv examples/aggressive_strategy_v3.py .archive/code/ 2>/dev/null
mv examples/bull_bear_v1.py .archive/code/ 2>/dev/null
mv examples/dynamic_strategy_v9.py .archive/code/ 2>/dev/null
mv examples/enhanced_strategy.py .archive/code/ 2>/dev/null
mv examples/final_strategy.py .archive/code/ 2>/dev/null
mv examples/final_strategy_v8.py .archive/code/ 2>/dev/null
mv examples/hybrid_strategy_v10.py .archive/code/ 2>/dev/null
mv examples/dynamic_risk_v16.py .archive/code/ 2>/dev/null

# 测试文件
mv examples/*_test*.py .archive/code/ 2>/dev/null
mv examples/test_*.py .archive/code/ 2>/dev/null
mv examples/stress_test.py .archive/code/ 2>/dev/null

# 验证文件（保留comprehensive_backtest.py）
mv examples/complete_validation.py .archive/code/ 2>/dev/null
mv examples/complete_verification.py .archive/code/ 2>/dev/null
mv examples/deep_validation.py .archive/code/ 2>/dev/null
mv examples/deep_validation_fixed.py .archive/code/ 2>/dev/null
mv examples/final_validation.py .archive/code/ 2>/dev/null
mv examples/full_validation.py .archive/code/ 2>/dev/null
mv examples/verify_profitability.py .archive/code/ 2>/dev/null

# 其他中间文件
mv examples/auto_run.py .archive/code/ 2>/dev/null
mv examples/better_fetch.py .archive/code/ 2>/dev/null
mv examples/complete_system.py .archive/code/ 2>/dev/null
mv examples/data_generator.py .archive/code/ 2>/dev/null
mv examples/fetch_and_validate.py .archive/code/ 2>/dev/null
mv examples/fetch_tushare.py .archive/code/ 2>/dev/null
mv examples/final_report.py .archive/code/ 2>/dev/null
mv examples/full_optimizer.py .archive/code/ 2>/dev/null
mv examples/full_system_test.py .archive/code/ 2>/dev/null
mv examples/multi_backtest_v2.py .archive/code/ 2>/dev/null
mv examples/paper_trading_test_10cycles.py .archive/code/ 2>/dev/null
mv examples/trend_strength_v12.py .archive/code/ 2>/dev/null

echo "  ✅ 中间版本代码已归档"

# === 2. 归档旧文档 ===
echo "📦 归档旧文档..."

# docs/archive/ 中的文件
mv docs/archive/* .archive/docs/ 2>/dev/null
rmdir docs/archive 2>/dev/null

echo "  ✅ 旧文档已归档"

# === 3. 清理Python缓存 ===
echo "🧽 清理Python缓存..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type f -name "*.pyo" -delete 2>/dev/null
echo "  ✅ Python缓存已清理"

# === 4. 清理临时文件 ===
echo "🧽 清理临时文件..."
find . -type f -name "*.tmp" -delete 2>/dev/null
find . -type f -name "*.bak" -delete 2>/dev/null
find . -type f -name "*~" -delete 2>/dev/null
echo "  ✅ 临时文件已清理"

# === 5. 显示结果 ===
echo ""
echo "✅ 清理完成！"
echo ""
echo "📊 清理统计:"
echo "  - 归档代码: $(ls .archive/code/*.py 2>/dev/null | wc -l) 个文件"
echo "  - 归档文档: $(ls .archive/docs/*.md 2>/dev/null | wc -l) 个文件"
echo "  - 剩余代码: $(ls examples/*.py 2>/dev/null | wc -l) 个文件"
echo "  - 剩余文档: $(ls docs/*.md 2>/dev/null | wc -l) 个文件"
echo ""
echo "📁 核心文件:"
echo "  - examples/auto_trading_bot.py"
echo "  - examples/smart_screener_v2.py"
echo "  - examples/comprehensive_backtest.py"
echo "  - examples/weight_optimizer.py"
echo "  - examples/param_optimizer.py"
echo "  - examples/portfolio_backtest.py"
echo "  - examples/fetch_tushare_auto.py"
echo ""
echo "💡 提示:"
echo "  - 归档文件保存在 .archive/ 目录"
echo "  - 如需恢复，从 .archive/ 目录移回即可"
echo "  - 如需彻底删除，运行: rm -rf .archive/"
