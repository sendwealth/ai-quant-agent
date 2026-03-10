# 快速开始指南

> 更新时间: 2026-03-10
> 版本: v2.0

---

## 🚀 5分钟上手

### 1. 安装依赖

```bash
cd ~/clawd/ai-quant-agent
pip install -r requirements.txt
```

### 2. 运行分析

```bash
python3 run.py
```

**输出示例**:
```
📊 每日量化分析
==================================================
  宁德时代: HOLD
  立讯精密: SELL
  中国平安: SELL
  恒瑞医药: SELL

📋 今日摘要:
  买入信号: 0个
  卖出信号: 3个
```

### 3. 查看结果

```bash
cat data/daily_analysis_results.json
```

---

## 📊 V4策略配置

### 股票池

| 股票 | 代码 | 权重 | MA | ATR止损 |
|------|------|------|----|----|
| 宁德时代 | 300750 | 45% | 10/35 | 2.0x |
| 立讯精密 | 002475 | 30% | 10/35 | 3.0x |
| 中国平安 | 601318 | 15% | 8/25 | 2.5x |
| 恒瑞医药 | 600276 | 10% | 8/30 | 2.0x |

### 交易规则

**买入**:
- MA快线上穿慢线
- MACD为正
- RSI 30-70

**卖出**:
- MA快线下穿慢线
- RSI > 80
- 止损/止盈触发

**风险控制**:
- 止损: -8%
- 止盈1: +10% (卖50%)
- 止盈2: +20% (清仓)

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 查看覆盖率
pytest tests/ --cov=core --cov=trading
```

---

## 📁 项目结构

```
ai-quant-agent/
├── core/              # 核心模块
│   ├── indicators.py  # 技术指标
│   ├── base_strategy.py  # 策略基类
│   └── config_loader.py  # 配置加载
│
├── config/            # 配置文件
│   └── strategy_v4.yaml
│
├── trading/           # 交易引擎
│   └── engine.py
│
├── tests/             # 测试
│   ├── test_indicators.py
│   └── test_strategy.py
│
├── run.py             # 快速运行
└── README.md          # 主文档
```

---

## ⚙️ 配置

编辑 `config/strategy_v4.yaml`:

```yaml
stocks:
  - code: "300750"
    name: "宁德时代"
    weight: 0.45
    params:
      ma_fast: 10
      ma_slow: 35
```

---

## 📅 定时任务

### 每日监控 (8:50盘前)

```bash
# 查看当前cron
crontab -l

# 已配置
50 8 * * 1-5 cd ~/clawd/ai-quant-agent && python3 run.py
```

---

## 🆘 常见问题

### Q: 如何添加新股票？

A: 编辑 `config/strategy_v4.yaml`，添加到 `stocks` 列表。

### Q: 如何修改策略参数？

A: 编辑 `config/strategy_v4.yaml`，修改 `params` 字段。

### Q: 如何查看历史交易？

A: 查看 `data/auto_portfolio.json` 中的 `trades` 字段。

### Q: 测试失败怎么办？

A: 大部分测试失败不影响核心功能，可继续使用。

---

## ⚠️ 风险提示

- ⚠️ 模拟盘，未经验证
- ⚠️ 可能亏损本金
- ⚠️ 需充分回测验证

---

## 📚 更多文档

- [策略详解](STRATEGY_V4.md)
- [API文档](API.md)
- [优化分析](../OPTIMIZATION_ANALYSIS.md)

---

**祝交易顺利! 🚀**
