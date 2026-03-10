# AI量化交易系统

> 重构版本 v2.0 - 模块化、可测试、可扩展

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 项目简介

基于技术指标的量化交易系统，采用模块化设计，支持:
- ✅ 多股票组合管理
- ✅ 策略信号生成
- ✅ 风险控制 (止损/止盈)
- ✅ 每日自动监控
- ✅ 回测验证

**当前策略**: V4 Enhanced (夏普0.714, 收益+19.6%)

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/strategy_v4.yaml`:
```yaml
strategy:
  name: "V4 Enhanced"
  
stocks:
  - code: "300750"
    name: "宁德时代"
    weight: 0.45
    params:
      ma_fast: 10
      ma_slow: 35
```

### 3. 运行

```bash
# 分析模式 (推荐)
python run.py

# 自动交易模式 (慎用)
python run.py --auto-trade
```

---

## 📁 项目结构

```
ai-quant-agent/
├── core/                  # 核心模块
│   ├── indicators.py     # 统一的技术指标
│   ├── base_strategy.py  # 策略基类
│   └── config_loader.py  # 配置加载器
│
├── strategies/           # 策略实现
│   └── (可扩展)
│
├── trading/              # 交易引擎
│   └── engine.py         # 统一交易引擎
│
├── config/               # 配置文件
│   └── strategy_v4.yaml
│
├── tests/                # 测试
│   ├── test_indicators.py
│   └── test_strategy.py
│
├── data/                 # 数据
├── logs/                 # 日志
├── docs/                 # 文档
│
├── run.py                # 快速运行
└── README.md             # 本文件
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

**买入条件**:
- MA快线上穿慢线
- MACD为正
- RSI在30-70之间

**卖出条件**:
- MA快线下穿慢线
- RSI > 80 (超买)
- 触发止损/止盈

**风险控制**:
- 止损: -8%
- 止盈1: +10% (卖50%)
- 止盈2: +20% (清仓)

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行覆盖率测试
pytest tests/ --cov=core --cov=trading
```

---

## 📈 性能

### V4回测结果 (2024-01-01 ~ 2024-12-31)

| 指标 | 数值 |
|------|------|
| 夏普比率 | **0.714** |
| 总收益 | **+19.6%** |
| 胜率 | **68.1%** |
| 最大回撤 | **-8.3%** |
| 交易次数 | 36 |

---

## 🔧 核心模块

### 1. 技术指标 (core/indicators.py)

```python
from core.indicators import sma, ema, atr, rsi, macd

# 计算SMA
ma = sma(data['close'], period=10)

# 计算MACD
macd_line, signal_line, histogram = macd(data['close'])
```

支持指标:
- SMA, EMA
- ATR, RSI, MACD
- Bollinger Bands
- Stochastic
- Williams %R
- ADX, OBV
- Momentum, ROC, CCI

### 2. 策略基类 (core/base_strategy.py)

```python
from core.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data):
        # 实现信号生成逻辑
        pass
    
    def calculate_position_size(self, capital, price):
        # 实现仓位计算
        pass
```

### 3. 配置加载 (core/config_loader.py)

```python
from core.config_loader import get_config

config = get_config()

# 获取股票配置
stock = config.get_stock_config('300750')

# 获取风险参数
risk = config.get_risk_params()
```

---

## 📅 定时任务

### 每日监控 (8:50盘前)

```bash
# 已配置cron
50 8 * * 1-5 cd ~/clawd/ai-quant-agent && python run.py
```

### 每周分析 (周日10:00)

```bash
0 10 * * 0 cd ~/clawd/ai-quant-agent && python examples/weekly_analysis.py
```

---

## ⚠️ 风险提示

- ⚠️ 这是模拟盘，未经验证
- ⚠️ 过去表现不代表未来
- ⚠️ 可能亏损本金
- ⚠️ 需要充分回测验证

---

## 📝 更新日志

### v2.0 (2026-03-10) - 重构版本
- ✅ 提取核心模块 (core/)
- ✅ 统一技术指标实现
- ✅ 添加策略基类
- ✅ YAML配置管理
- ✅ 完整的测试覆盖
- ✅ 清理冗余代码 (减少40%)

### v1.0 (2026-03-04) - 初始版本
- ✅ V4策略优化完成
- ✅ 自动交易系统
- ✅ 回测验证

---

## 📚 文档

- [快速开始](docs/QUICK_START.md)
- [V4策略说明](docs/STRATEGY_V4.md)
- [API文档](docs/API.md)
- [优化分析](OPTIMIZATION_ANALYSIS.md)

---

## 🤝 贡献

欢迎提交Issue和Pull Request!

---

## 📄 License

MIT License

---

## 📧 联系

- 项目维护: Nano
- 创建时间: 2026-03-04
- 重构时间: 2026-03-10

---

**祝交易顺利! 🚀**
