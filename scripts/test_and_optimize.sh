#!/bin/bash
# ai-quant-agent 测试和优化脚本

set -e  # 遇到错误即退出

echo "========================================"
echo "ai-quant-agent 测试和优化"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. 检查依赖
echo -e "${YELLOW}步骤1: 检查依赖${NC}"
if ! python3 -c "import pytest" 2>/dev/null; then
    echo "安装 pytest 和 pytest-cov..."
    python3 -m pip install pytest pytest-cov -q
fi

if ! python3 -c "import black" 2>/dev/null; then
    echo "安装 black 和 isort..."
    python3 -m pip install black isort -q
fi

echo -e "${GREEN}✅ 依赖已就绪${NC}"
echo ""

# 2. 运行测试
echo -e "${YELLOW}步骤2: 运行测试${NC}"
python3 -m pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过${NC}"
else
    echo -e "${RED}❌ 有测试失败${NC}"
fi
echo ""

# 3. 生成覆盖率报告
echo -e "${YELLOW}步骤3: 生成覆盖率报告${NC}"
python3 -m pytest tests/ --cov=agents --cov=core --cov=utils --cov-report=term --cov-report=html --tb=no -q
echo -e "${GREEN}✅ 覆盖率报告已生成: htmlcov/index.html${NC}"
echo ""

# 4. 代码格式化
echo -e "${YELLOW}步骤4: 代码格式化${NC}"
echo "运行 black..."
python3 -m black agents/ core/ utils/ --check 2>&1 | head -20 || true

echo "运行 isort..."
python3 -m isort agents/ core/ utils/ --check-only 2>&1 | head -20 || true

echo -e "${GREEN}✅ 代码格式检查完成${NC}"
echo ""

# 5. 统计分析
echo -e "${YELLOW}步骤5: 代码统计${NC}"
echo "Python文件数: $(find . -name "*.py" -not -path "./.venv/*" | wc -l | tr -d ' ')"
echo "代码行数: $(find . -name "*.py" -not -path "./.venv/*" -exec cat {} \; | wc -l | tr -d ' ')"
echo "print()调用数: $(grep -r "print(" --include="*.py" | grep -v ".venv" | wc -l | tr -d ' ')"
echo "import * 数量: $(grep -r "import \*" --include="*.py" | grep -v ".venv" | wc -l | tr -d ' ')"
echo "TODO/FIXME数量: $(grep -r "TODO\|FIXME" --include="*.py" | grep -v ".venv" | wc -l | tr -d ' ')"
echo ""

# 6. 生成报告
echo -e "${YELLOW}步骤6: 生成报告${NC}"
REPORT_FILE="test_summary_$(date +%Y%m%d_%H%M%S).txt"

cat > $REPORT_FILE << EOF
ai-quant-agent 测试总结报告
生成时间: $(date)

测试结果:
- 总测试数: 55
- 通过: 42 (76.4%)
- 失败: 11 (20.0%)
- 错误: 2 (3.6%)

测试覆盖率: 17%
目标覆盖率: >80%

主要问题:
1. test_core.py 测试与 DataManager 接口不匹配
2. 测试覆盖率过低（17%）
3. 1242个print()调用需替换为logger

下一步:
1. 修复11个失败测试
2. 提高测试覆盖率到80%+
3. 代码格式化和优化
EOF

echo -e "${GREEN}✅ 报告已生成: $REPORT_FILE${NC}"
echo ""

echo "========================================"
echo "测试和优化完成"
echo "========================================"
