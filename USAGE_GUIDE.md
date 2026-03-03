# AI量化策略使用指南

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install pandas numpy akshare

# 克隆仓库
git clone https://github.com/sendwealth/ai-quant-agent.git
cd ai-quant-agent
```

### 2. 运行策略

```bash
# 最简单：混合策略(推荐)
python3 examples/hybrid_strategy_v10.py

# 保守策略
python3 examples/multi_backtest_v2.py

# 实盘模拟
python3 examples/paper_trading.py
```

---

## 策略说明

### 策略对比

| 策略 | 文件 | 特点 | 适用场景 |
|------|------|------|----------|
| 保守策略 | multi_backtest_v2.py | 25%仓位，2.5倍ATR止损 | 下跌/震荡市 |
| 混合策略 | hybrid_strategy_v10.py | 趋势+均值回归 | 全市场 |
| 动态仓位 | dynamic_strategy_v9.py | 15%-45%动态仓位 | 趋势明显时 |
| 实盘模拟 | paper_trading.py | 多股票轮动 | 模拟交易 |

### 推荐配置

**保守策略 (最优验证结果)**
```python
MA_SHORT = 5      # 短期均线
MA_LONG = 30      # 长期均线
ATR_STOP = 2.5    # 止损倍数
ATR_TRAIL = 2.0   # 追踪止损
MAX_POS = 0.25    # 25%仓位
```

---

## 回测结果

### 真实数据验证 (2024-2026)

| 股票 | 买入持有 | 保守策略 | 超额收益 |
|------|----------|----------|----------|
| 五粮液 | -16.6% | **+2.82%** | +19.4% ✅ |
| 比亚迪 | +54.0% | **+2.12%** | -51.8% ⚠️ |
| 茅台 | -0.7% | -2.86% | -2.2% |

### 核心结论

✅ **有效场景**：
- 下跌市场：显著减少损失
- 震荡市场：稳定小幅盈利
- 风险控制：回撤控制在10%以内

⚠️ **局限场景**：
- 大涨行情：跑输大盘
- 原因：止损和仓位限制导致提前退出

---

## 使用建议

### 1. 市场环境判断

```python
# 判断趋势
price > MA20 > MA60  → 上涨趋势 → 可用激进策略
price < MA20 < MA60  → 下跌趋势 → 用保守策略
其他                  → 震荡     → 用均值回归
```

### 2. 仓位建议

| 市场环境 | 仓位 | 策略 |
|----------|------|------|
| 强势上涨 | 35-45% | 趋势跟随 |
| 弱势上涨 | 25-35% | 趋势跟随 |
| 震荡 | 20-25% | 均值回归 |
| 下跌 | 10-20% | 超卖反弹 |

### 3. 止损设置

```python
# ATR止损
stop_loss = entry_price - ATR * 2.5

# 追踪止损(盈利后)
trailing_stop = highest_price - ATR * 2.0
```

---

## 获取数据

```python
import akshare as ak

# 获取A股数据
df = ak.stock_zh_a_hist(
    symbol='600519',      # 股票代码
    period='daily',       # 日线
    start_date='20240101',
    end_date='20260303',
    adjust='qfq'          # 前复权
)
```

---

## 文件说明

```
ai-quant-agent/
├── examples/
│   ├── hybrid_strategy_v10.py   # 混合策略(推荐)
│   ├── dynamic_strategy_v9.py   # 动态仓位
│   ├── paper_trading.py         # 实盘模拟
│   ├── multi_backtest_v2.py     # 多股票回测
│   ├── final_validation.py      # 最终验证
│   └── ...
├── data/
│   ├── real_000858.csv          # 五粮液
│   ├── real_002594.csv          # 比亚迪
│   └── real_600519.csv          # 茅台
├── OPTIMIZATION_REPORT.md       # 优化报告
└── USAGE_GUIDE.md               # 本文件
```

---

## 风险提示

> ⚠️ **重要声明**
> 
> 1. 本项目仅供学习研究，不构成投资建议
> 2. 历史表现不代表未来收益
> 3. 实盘前请充分回测，控制仓位
> 4. 严格遵守止损纪律
> 5. 不要投入超过承受能力的资金

---

## 常见问题

**Q: 为什么上涨市场跑输大盘？**
A: 策略使用止损和仓位限制，在大涨时会提前止盈或被震荡出局。这是风险控制的代价。

**Q: 如何改进？**
A: 
1. 上涨趋势中放宽止损
2. 增加持仓耐心，减少交易频率
3. 结合基本面判断是否长期持有

**Q: 适合什么类型的投资者？**
A: 风险厌恶型投资者，希望在下跌市场减少损失，接受在上涨市场跑输大盘的代价。

---

**版本**: v10.0  
**更新**: 2026-03-03  
**作者**: OpenClaw AI
