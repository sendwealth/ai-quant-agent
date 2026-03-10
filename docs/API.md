# API文档

> 更新时间: 2026-03-10
> 版本: v2.0

---

## 📦 核心模块

### core.indicators

统一的技术指标计算模块。

#### SMA - 简单移动平均

```python
from core.indicators import sma

# 计算10日均线
ma10 = sma(data['close'], period=10)
```

**参数**:
- `data` (pd.Series): 价格数据
- `period` (int): 周期

**返回**: pd.Series

---

#### EMA - 指数移动平均

```python
from core.indicators import ema

ema10 = ema(data['close'], period=10)
```

---

#### ATR - 平均真实波幅

```python
from core.indicators import atr

atr14 = atr(data['high'], data['low'], data['close'], period=14)
```

---

#### RSI - 相对强弱指标

```python
from core.indicators import rsi

rsi14 = rsi(data['close'], period=14)
```

**返回**: 0-100之间的序列

---

#### MACD

```python
from core.indicators import macd

macd_line, signal_line, histogram = macd(
    data['close'],
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```

**返回**: (macd_line, signal_line, histogram)

---

#### Bollinger Bands

```python
from core.indicators import bollinger_bands

upper, middle, lower = bollinger_bands(
    data['close'],
    period=20,
    std_dev=2.0
)
```

---

#### 交叉检测

```python
from core.indicators import detect_crossover

# 检测MA金叉/死叉
signals = detect_crossover(ma_fast, ma_slow)
# 1: 上穿, -1: 下穿, 0: 无交叉
```

---

### core.base_strategy

策略基类和工厂。

#### BaseStrategy

```python
from core.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, config):
        super().__init__(config)
    
    def generate_signals(self, data):
        # 实现信号生成
        pass
    
    def calculate_position_size(self, capital, price):
        # 实现仓位计算
        pass
```

**必须实现的方法**:
- `generate_signals(data)` - 返回信号序列 (1/0/-1)
- `calculate_position_size(capital, price)` - 返回股数

**可选方法**:
- `validate_data(data)` - 验证数据
- `check_risk(position, price)` - 检查风险

---

#### MAStrategy

V4策略实现。

```python
from core.base_strategy import MAStrategy

config = {
    'name': 'TestMA',
    'weight': 0.25,
    'params': {
        'ma_fast': 10,
        'ma_slow': 30
    },
    'risk': {
        'stop_loss': -0.08,
        'take_profit_1': 0.10,
        'take_profit_2': 0.20
    }
}

strategy = MAStrategy(config)
signals = strategy.generate_signals(data)
```

---

#### StrategyFactory

```python
from core.base_strategy import StrategyFactory

strategy = StrategyFactory.create({
    'type': 'ma',
    'name': 'V4',
    # ...
})
```

---

### core.config_loader

配置加载器。

#### ConfigLoader

```python
from core.config_loader import ConfigLoader

# 加载配置
loader = ConfigLoader('config/strategy_v4.yaml')

# 获取配置
strategy_config = loader.get_strategy_config()
stock_config = loader.get_stock_config('300750')
risk_params = loader.get_risk_params()

# 获取所有股票
stocks = loader.get_all_stocks(enabled_only=True)
```

**主要方法**:
- `get_strategy_config()` - 获取策略配置
- `get_stock_config(code)` - 获取单只股票配置
- `get_all_stocks(enabled_only)` - 获取所有股票
- `get_risk_params()` - 获取风险参数
- `validate()` - 验证配置

---

#### 全局实例

```python
from core.config_loader import get_config

config = get_config()
stock = config.get_stock_config('300750')
```

---

### trading.engine

统一交易引擎。

#### TradingEngine

```python
from trading.engine import TradingEngine

# 创建引擎
engine = TradingEngine()

# 运行分析 (不自动交易)
results = engine.run(auto_trade=False)

# 运行分析 + 自动交易
results = engine.run(auto_trade=True)

# 执行单个信号
engine.execute_signal('300750', 1, 350.0)  # 买入
engine.execute_signal('300750', -1, 360.0)  # 卖出
```

**返回结果**:
```python
{
    'timestamp': '2026-03-10T22:01:33',
    'signals': {
        '300750': {
            'name': '宁德时代',
            'signal': 'HOLD',
            'price': 350.25,
            'change_pct': 0.0
        }
    },
    'alerts': [],
    'summary': {
        'buy_count': 0,
        'sell_count': 3,
        'alert_count': 0,
        'performance': {
            'total_value': 52547.00,
            'total_return': -0.4745
        }
    }
}
```

---

## 🔧 工具函数

### 数据验证

```python
from core.base_strategy import BaseStrategy

strategy = MAStrategy(config)
is_valid = strategy.validate_data(data)
```

### 风险检查

```python
position = {
    'entry_price': 100,
    'shares': 100
}

should_close, reason = strategy.check_risk(position, 91.0)
# should_close: True (触发止损)
# reason: "触发止损 (亏损-9.00%)"
```

### 绩效计算

```python
performance = engine._calculate_performance()
# {
#     'total_value': 52547.00,
#     'total_return': -0.4745,
#     'cash_ratio': 1.0
# }
```

---

## 📝 配置格式

### strategy_v4.yaml

```yaml
strategy:
  name: "V4 Enhanced"
  version: "4.1"
  type: "ma"

stocks:
  - code: "300750"
    name: "宁德时代"
    weight: 0.45
    params:
      ma_fast: 10
      ma_slow: 35
      atr_stop: 2.0
    enabled: true

risk:
  stop_loss: -0.08
  take_profit_1: 0.10
  take_profit_2: 0.20
  max_position: 0.15

capital:
  initial: 100000
  reserve_ratio: 0.10
```

---

## 🧪 测试

### 运行测试

```bash
# 所有测试
pytest tests/ -v

# 指标测试
pytest tests/test_indicators.py -v

# 策略测试
pytest tests/test_strategy.py -v

# 覆盖率
pytest tests/ --cov=core --cov=trading --cov-report=html
```

### 测试示例

```python
import pytest
from core.indicators import sma, rsi
from core.base_strategy import MAStrategy

def test_sma():
    data = pd.Series([1, 2, 3, 4, 5])
    result = sma(data, 3)
    assert result.iloc[4] == 4.0

def test_strategy_signals():
    strategy = MAStrategy(config)
    signals = strategy.generate_signals(data)
    assert len(signals) == len(data)
```

---

## 📊 数据格式

### 股票数据

```python
df = pd.DataFrame({
    'datetime': [...],
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})
```

### 持仓数据

```json
{
  "update_time": "2026-03-10T22:01:33",
  "initial_capital": 100000,
  "cash": 52547.00,
  "positions": {
    "300750": {
      "code": "300750",
      "name": "宁德时代",
      "shares": 100,
      "entry_price": 350.0,
      "entry_time": "2026-03-01T10:00:00"
    }
  },
  "trades": [...]
}
```

---

## 🔗 快速链接

- [快速开始](QUICK_START.md)
- [策略详解](STRATEGY_V4.md)
- [优化分析](../OPTIMIZATION_ANALYSIS.md)

---

**最后更新**: 2026-03-10  
**维护人**: Nano
