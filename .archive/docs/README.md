# AI量化交易策略项目

**版本**: V4 (最优版本)  
**状态**: ✅ 已优化完成，开始模拟盘验证  
**最后更新**: 2026-03-04

---

## 🎯 项目简介

本项目是一个完整的AI量化交易策略系统，经过**5轮优化**，从V1到V5，最终确定**V4为最优版本**。

### 核心成果
- ✅ 组合夏普: **0.714** (Walk-Forward: **0.816**)
- ✅ 组合收益: **+19.6%** (预期实际: 5-12%)
- ✅ 平均胜率: **68.1%**
- ✅ 最大回撤: **-10.3%**

---

## 📁 项目结构

```
ai-quant-agent/
├── README.md                   # 本文件
├── START_HERE.md              # 🚀 快速开始指南
│
├── docs/                       # 📚 文档
│   ├── FINAL_CONCLUSION.md    # ⭐ 最终结论
│   ├── AUTO_TRADING_GUIDE.md  # ⭐ 自动化交易指南
│   ├── COMPREHENSIVE_BACKTEST_REPORT.md  # 充分回测报告
│   ├── OPTIMIZATION_SUCCESS_V4.md        # V4优化报告
│   ├── SIMULATION_START_GUIDE.md         # 模拟盘启动指南
│   ├── QUICK_REFERENCE_CARD.md           # 快速参考卡
│   ├── STARTUP_CHECKLIST.md              # 启动清单
│   ├── TRADING_LOG_TEMPLATE.md           # 交易日志模板
│   └── HOW_TO_START.md                   # 如何开始
│
├── examples/                   # 💻 核心代码
│   ├── auto_trading_bot.py    # ⭐ 自动交易机器人
│   ├── smart_screener_v2.py   # 智能选股系统
│   ├── weight_optimizer.py    # 权重优化器
│   ├── param_optimizer.py     # 参数优化器
│   ├── comprehensive_backtest.py  # 充分回测
│   ├── portfolio_backtest.py  # 组合回测
│   ├── trading_monitor.py     # 交易监控
│   └── fetch_tushare_auto.py  # 数据获取
│
├── data/                       # 📊 数据
│   ├── real_*.csv             # 29只股票数据
│   ├── auto_portfolio.json    # 自动交易持仓
│   ├── smart_screening_v2.json    # 选股结果
│   ├── weight_optimization_results.json  # 权重优化
│   └── param_optimization_results.json   # 参数优化
│
├── strategies/                 # 📈 策略模块
│   └── __init__.py
│
├── backtest/                   # 🔬 回测引擎
│   ├── engine.py
│   └── __init__.py
│
├── utils/                      # 🛠️ 工具
│   ├── indicators.py          # 技术指标
│   ├── config.py              # 配置
│   └── logger.py              # 日志
│
├── tests/                      # 🧪 测试
│   └── __init__.py
│
└── logs/                       # 📝 日志
    └── auto_trading.log       # 自动交易日志
```

---

## 🚀 快速开始

### 1. 自动化模拟交易（推荐）⭐

```bash
# 运行自动交易机器人
cd ai-quant-agent
python3 examples/auto_trading_bot.py
```

**就这么简单！** 系统会自动：
- ✅ 检查4只股票信号
- ✅ 自动买入/卖出
- ✅ 自动止损止盈
- ✅ 保存交易记录

### 2. 设置每日自动运行

```bash
# 编辑crontab
crontab -e

# 添加（每天15:30自动运行）
30 15 * * 1-5 cd /Users/rowan/clawd/ai-quant-agent && python3 examples/auto_trading_bot.py >> logs/auto_trading.log 2>&1
```

### 3. 查看结果

```bash
# 查看持仓
cat data/auto_portfolio.json

# 查看日志
tail -f logs/auto_trading.log
```

---

## 📊 V4配置

### 股票配置

| 股票 | 代码 | 权重 | MA参数 | ATR止损 | 夏普 |
|------|------|------|--------|---------|------|
| 宁德时代 | 300750 | 45% | 10/35 | 2.0x | 0.767 |
| 立讯精密 | 002475 | 30% | 10/35 | 3.0x | 0.780 |
| 中国平安 | 601318 | 15% | 8/25 | 2.5x | 0.614 |
| 恒瑞医药 | 600276 | 10% | 8/30 | 2.0x | 0.430 |

### 交易规则

**买入** (全部满足):
- ⚡ MA快线上穿慢线（金叉）
- ✅ MACD柱状图为正
- ✅ RSI在30-70之间
- ✅ 成交量放大

**卖出** (满足任一):
- 🔴 止损: -8%
- 💰 止盈1: +10% (卖50%)
- 💰 止盈2: +20% (清仓)
- 🔄 趋势反转

---

## 📈 优化历程

```
V1: 参数优化 → 夏普-0.18 ❌
    ↓
V2: 市场过滤 → 夏普0.02 ❌
    ↓
V3: 智能选股 → 夏普0.49 ⚠️
    ↓
V4: 参数微调 → 夏普0.714 ✅✅✅ (最优)
    ↓
V5: 波动率过滤 → 夏普0.594 ❌ (过度优化)
```

**结论**: V4是最优版本，停止优化，开始验证。

---

## 📚 核心文档

### 必读文档

1. **START_HERE.md** - 快速开始
2. **docs/FINAL_CONCLUSION.md** - 最终结论
3. **docs/AUTO_TRADING_GUIDE.md** - 自动化交易指南
4. **docs/QUICK_REFERENCE_CARD.md** - 快速参考卡

### 详细文档

5. **docs/COMPREHENSIVE_BACKTEST_REPORT.md** - 充分回测报告
6. **docs/OPTIMIZATION_SUCCESS_V4.md** - V4优化详情
7. **docs/SIMULATION_START_GUIDE.md** - 模拟盘启动

---

## 🎯 验证计划

### 时间表

```
今天: 运行自动交易机器人
1-2周: 等待信号，开始建仓
1-3月: 持仓管理，积累样本
3月后: 评估结果，决定下一步
```

### 成功标准 (3月后)

```yaml
优秀: 夏普≥0.6, 收益≥10% → 考虑小资金实盘
良好: 夏普≥0.4, 收益≥5%  → 继续模拟盘
及格: 夏普≥0.2, 收益≥0%  → 优化后继续
失败: 夏普<0.2, 收益<0%   → 放弃策略
```

---

## ⚠️ 重要提醒

### 预期管理

**合理预期**:
- 夏普: 0.40-0.60 (不是0.714)
- 收益: 5-12% (不是19.6%)
- 胜率: 60-68%
- 回撤: ≤15%

**原因**: 回测往往高估20-40%，这是正常的。

### 风险提示

- ⚠️ 置信度48%（需要验证）
- ⚠️ 样本量不足（3年数据）
- ⚠️ 这是模拟盘，不是实盘
- ⚠️ 可能亏损

---

## 🔧 开发

### 更新数据

```bash
python3 examples/fetch_tushare_auto.py
```

### 运行回测

```bash
python3 examples/comprehensive_backtest.py
```

### 查看选股

```bash
python3 examples/smart_screener_v2.py
```

---

## 📊 数据说明

### 股票数据

- **来源**: TuShare
- **数量**: 29只A股
- **时间**: 2021-2024 (3年)
- **更新**: 每周一次

### TuShare Token

配置文件: `utils/config.py`

```python
TUSHARE_TOKEN = '33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b'
```

---

## 🤝 贡献

本项目是个人学习项目，暂不接受外部贡献。

---

## 📄 许可证

MIT License

---

## 📞 联系方式

- 项目位置: `/Users/rowan/clawd/ai-quant-agent`
- 创建时间: 2026-03-04
- 最后更新: 2026-03-04

---

## 🎉 总结

> **"V4已经是最优版本，现在开始3-6个月模拟盘验证"**

**核心要点**:
1. ✅ 使用V4配置（夏普0.714）
2. ✅ 运行自动交易机器人
3. ✅ 调整预期（5-12%收益）
4. ✅ 3个月后评估

**下一步**:
```bash
python3 examples/auto_trading_bot.py
```

---

**祝**: 模拟盘顺利！策略验证成功！🚀
