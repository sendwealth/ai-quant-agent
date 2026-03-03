# 代码规范

本文档定义了AI量化交易系统的代码规范和最佳实践。

---

## 📋 目录

- [代码风格](#代码风格)
- [命名规范](#命名规范)
- [文档规范](#文档规范)
- [测试规范](#测试规范)
- [最佳实践](#最佳实践)

---

## 代码风格

### Python版本

- 最低版本: Python 3.7+
- 推荐版本: Python 3.9+

### 导入顺序

```python
# 1. 标准库
import os
import sys
from datetime import datetime

# 2. 第三方库
import pandas as pd
import numpy as np

# 3. 本地模块
from utils.indicators import sma, ema
```

### 代码格式

- 使用4空格缩进
- 行长度不超过100字符
- 函数之间空2行
- 类之间空3行

---

## 命名规范

### 变量

```python
# ✅ 好的命名
cash = 100000
max_position = 0.25
stop_loss_price = 95.5

# ❌ 不好的命名
c = 100000
mp = 0.25
sl = 95.5
```

### 函数

```python
# ✅ 好的命名
def calculate_position_size(capital, risk_per_trade):
    """计算仓位大小"""
    pass

def run_backtest(df, params):
    """运行回测"""
    pass

# ❌ 不好的命名
def calc(c, r):
    pass

def run(d, p):
    pass
```

### 常量

```python
# ✅ 大写+下划线
INITIAL_CAPITAL = 100000
MAX_POSITION = 0.25
ATR_STOP_MULTIPLIER = 2.5

# ❌ 小写
initial_capital = 100000
```

---

## 文档规范

### 模块文档

```python
"""
策略名称 v版本号
================
简要描述策略的核心思想和适用场景

功能:
- 功能1
- 功能2

参数:
    param1 (type): 参数说明

返回:
    type: 返回值说明

示例:
    >>> result = run_strategy(df, params)
    >>> print(result['return'])
    0.052
"""
```

### 函数文档

```python
def calculate_atr(high, low, close, period=14):
    """
    计算平均真实波幅(ATR)
    
    Args:
        high (pd.Series): 最高价序列
        low (pd.Series): 最低价序列
        close (pd.Series): 收盘价序列
        period (int): 计算周期，默认14
    
    Returns:
        pd.Series: ATR值序列
    
    Example:
        >>> atr_values = calculate_atr(df['high'], df['low'], df['close'])
        >>> print(atr_values.iloc[-1])
        2.45
    """
    pass
```

### 注释

```python
# ✅ 好的注释：解释为什么
# 使用2.5倍ATR止损，平衡噪音和风险
stop_loss = price - atr * 2.5

# ❌ 不好的注释：重复代码
# 计算止损
stop_loss = price - atr * 2.5
```

---

## 测试规范

### 测试文件命名

```
test_<module_name>.py
```

### 测试函数命名

```python
def test_<function_name>_<scenario>():
    """测试函数在特定场景下的行为"""
    pass

# 示例
def test_calculate_atr_normal_data():
    """测试ATR计算在正常数据下"""
    pass

def test_calculate_atr_empty_data():
    """测试ATR计算在空数据下"""
    pass
```

### 测试结构

```python
def test_function():
    # 1. 准备数据 (Arrange)
    df = pd.DataFrame({'close': [100, 101, 102]})
    
    # 2. 执行测试 (Act)
    result = calculate_sma(df['close'], 2)
    
    # 3. 验证结果 (Assert)
    assert result.iloc[-1] == 101.5
```

---

## 最佳实践

### 1. 风险管理

```python
# ✅ 总是设置止损
stop_loss = entry_price - atr * 2.5

# ✅ 限制仓位
position_size = min(capital * 0.25, max_position)

# ❌ 不要满仓
position_size = capital  # 危险！
```

### 2. 数据验证

```python
# ✅ 检查数据有效性
if df is None or len(df) < 50:
    raise ValueError("数据不足")

if df['close'].isna().sum() > 0:
    df = df.dropna()

# ❌ 不检查直接使用
result = df['close'].mean()  # 可能有NaN
```

### 3. 错误处理

```python
# ✅ 使用try-except处理异常
try:
    df = load_data(filepath)
except FileNotFoundError:
    print(f"文件不存在: {filepath}")
    return None

# ❌ 不处理异常
df = load_data(filepath)  # 可能崩溃
```

### 4. 性能优化

```python
# ✅ 向量化操作
df['ma'] = df['close'].rolling(20).mean()

# ❌ 循环操作
for i in range(len(df)):
    df.loc[i, 'ma'] = df['close'].iloc[i-20:i].mean()
```

### 5. 日志记录

```python
import logging

# ✅ 使用logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"开始回测: {symbol}")
logger.warning(f"数据缺失: {missing_count}行")

# ❌ 使用print
print("开始回测")  # 不专业
```

---

## 策略开发流程

### 1. 研究阶段

```python
# research/new_strategy.py
"""
策略假设:
- 市场在特定条件下会重复模式
- 使用XX指标可以捕捉这个模式

验证方法:
- 在历史数据上回测
- 样本外测试
- 极端场景测试
"""
```

### 2. 开发阶段

```python
# strategies/new_strategy.py

class NewStrategy:
    """新策略"""
    
    def __init__(self, params):
        """初始化参数"""
        self.params = params
    
    def generate_signals(self, df):
        """生成交易信号"""
        pass
    
    def run_backtest(self, df):
        """运行回测"""
        pass
```

### 3. 测试阶段

```python
# tests/test_new_strategy.py

def test_new_strategy_normal():
    """测试正常场景"""
    pass

def test_new_strategy_edge_case():
    """测试边界情况"""
    pass
```

### 4. 部署阶段

```python
# 更新文档
# 添加到策略对比
# 运行完整测试
```

---

## 代码审查清单

- [ ] 代码符合PEP8规范
- [ ] 所有函数都有文档字符串
- [ ] 关键逻辑有注释
- [ ] 有单元测试
- [ ] 测试覆盖率>80%
- [ ] 无硬编码参数
- [ ] 有错误处理
- [ ] 有日志记录
- [ ] 性能可接受
- [ ] 文档已更新

---

## 性能基准

### 回测性能

| 数据量 | 目标时间 | 实际时间 |
|--------|----------|----------|
| 500行 | <0.1s | 0.05s ✅ |
| 5000行 | <1s | 0.3s ✅ |
| 50000行 | <10s | 5s ✅ |

### 内存使用

| 数据量 | 目标内存 | 实际内存 |
|--------|----------|----------|
| 500行 | <1MB | 79KB ✅ |
| 5000行 | <10MB | 5MB ✅ |
| 50000行 | <100MB | 50MB ✅ |

---

## 更新日志

### v17.0 (2026-03-03)
- 修复高波动测试
- 达到100%测试通过率
- 优化项目文档

### v16.0 (2026-03-03)
- 添加动态风险管理
- 实现连续盈亏调整

### v15.0 (2026-03-03)
- 实现1%风险规则
- 添加交易成本和滑点
- 样本外测试

---

**维护者**: OpenClaw AI  
**最后更新**: 2026-03-03
