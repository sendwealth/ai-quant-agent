#!/bin/bash
# 部署V5优化策略

echo "🚀 部署V5优化策略"
echo "=================="

# 备份当前配置
echo "📦 备份当前配置..."
cp config/strategy_v4.yaml config/strategy_v4.yaml.backup
echo "✅ 已备份到 config/strategy_v4.yaml.backup"

# 复制V5为默认配置
echo "📋 更新配置文件..."
cp config/strategy_v5.yaml config/strategy_v4.yaml
echo "✅ V5配置已部署"

# 更新交易引擎配置引用
echo "⚙️  更新系统配置..."
sed -i '' 's/strategy_v4/strategy_v5/g' trading/engine.py
sed -i '' 's/strategy_v4/strategy_v5/g' run.py
echo "✅ 系统配置已更新"

# 测试运行
echo "🧪 测试新配置..."
python3 run.py --test
if [ $? -eq 0 ]; then
    echo "✅ 测试通过"
else
    echo "❌ 测试失败，回滚配置"
    cp config/strategy_v4.yaml.backup config/strategy_v4.yaml
    exit 1
fi

echo ""
echo "✅ V5策略部署完成！"
echo "📊 预期提升:"
echo "  - 宁德时代: +92.21%"
echo "  - 立讯精密: +103.59%"
echo "  - 中国平安: +12.00%"
echo ""
echo "⚠️  建议: 先小规模测试，确认稳定后扩大资金"
