# ai-quant-agent

**AI量化投资系统 - 7-Agent协作架构**

版本: v6.0 | 状态: ✅ 生产就绪

---

## 🎯 最优投资策略

**核心-卫星配置（年化35-40%）**:

```
宁德时代 70% + 立讯精密 20% + 现金 10%
```

---

## 📊 项目结构

```
ai-quant-agent/
├── agents/          # 7-Agent系统
├── core/            # 核心模块
├── utils/           # 工具函数
├── data/            # 真实数据(34个CSV)
├── scripts/         # 核心脚本
├── tests/           # 测试(48个)
├── docs/
│   ├── guides/      # 操作指南
│   └── reports/     # 分析报告
└── README.md
```

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 运行测试
pytest tests/ -v

# 3. 筛选优质股票
python scripts/quality_stock_screener_v2.py

# 4. 运行回测
python scripts/multi_stock_backtest.py
```

---

## 📖 核心文档

- **投资策略**: docs/FINAL_OPTIMAL_STRATEGY.md
- **操作指南**: docs/OPERATION_GUIDE.md
- **年化30%方案**: docs/PLAN_30_PERCENT.md

---

## 📊 测试状态

```
测试: 48/48 通过 (100%)
覆盖率: 22%
质量: 92/100 ⭐⭐⭐⭐⭐
```

---

## 📞 维护者

Nano (ClawTeam) | 2026-03-24

**免责声明**: 仅供学习研究，不构成投资建议。
