# 🎯 下一步行动计划

**日期**: 2026-03-04  
**目标**: 改进策略，达到实盘条件  

---

## 📋 当前状态

### ✅ 已完成

- [x] 策略优化（参数网格搜索）
- [x] 单股票验证成功（比亚迪夏普0.869）
- [x] 多股票验证（4只A股）
- [x] 问题诊断（熊市环境导致3只未达标）

### ⚠️ 待解决

- [ ] 整体夏普-0.157（目标≥0.5）
- [ ] 整体胜率53.1%（目标≥60%）
- [ ] 整体收益-1.25%（目标≥5%）
- [ ] 样本量小（目标≥10只）

---

## 🚀 优先级1: 市场环境过滤 (最关键)

**为什么重要**: 
- 3只失败案例都是熊市环境
- 过滤掉熊市可以大幅提升指标
- 实现简单，效果显著

**实现步骤**:

### Step 1: 获取大盘数据 (1小时)

```python
# 文件: examples/add_market_filter.py

import akshare as ak

def get_market_index():
    """获取上证指数作为大盘指标"""
    # 上证指数
    df = ak.stock_zh_index_daily(symbol="sh000001")
    
    # 计算MA20和MA60
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()
    
    # 判断趋势
    df['trend'] = df['ma20'] > df['ma60']  # True=上涨, False=下跌
    
    return df
```

**任务**:
- [ ] 获取上证指数数据
- [ ] 保存为`data/market_index.csv`
- [ ] 验证数据质量

### Step 2: 实现市场过滤函数 (2小时)

```python
def should_trade(date, stock_data, market_data):
    """
    判断是否应该交易
    
    参数:
    - date: 交易日期
    - stock_data: 个股数据
    - market_data: 大盘数据
    
    返回:
    - True: 允许交易
    - False: 不允许交易
    """
    # 1. 大盘趋势必须向上
    market_trend = market_data.loc[date, 'trend']
    if not market_trend:
        return False  # 大盘下跌，不交易
    
    # 2. 个股趋势必须向上
    stock_ma10 = stock_data['ma10']
    stock_ma30 = stock_data['ma30']
    if stock_ma10 < stock_ma30:
        return False  # 个股下跌，不交易
    
    # 3. 大盘不能在高波动期
    market_volatility = market_data.loc[date, 'volatility']
    if market_volatility > 0.03:  # 日波动率>3%
        return False  # 市场恐慌，不交易
    
    return True
```

**任务**:
- [ ] 实现市场趋势判断
- [ ] 添加波动率过滤
- [ ] 测试过滤逻辑

### Step 3: 集成到策略 (1小时)

```python
def backtest_with_filter(df, market_data, params):
    """带市场过滤的回测"""
    
    for i in range(60, len(df)):
        date = df.index[i]
        
        # 原有买入信号
        if ma_fast > ma_slow and shares == 0:
            
            # 新增：市场过滤
            if not should_trade(date, df.iloc[i], market_data):
                continue  # 市场环境不好，跳过
            
            # 执行买入
            ...
```

**任务**:
- [ ] 修改回测函数
- [ ] 添加市场数据参数
- [ ] 测试集成

### Step 4: 验证效果 (2小时)

**任务**:
- [ ] 在4只股票上测试
- [ ] 对比过滤前后表现
- [ ] 统计提升幅度

**预期结果**:
```
过滤前:
- 平均夏普: -0.157
- 平均胜率: 53.1%
- 平均收益: -1.25%

过滤后:
- 平均夏普: 0.6+ ✅
- 平均胜率: 65%+ ✅
- 平均收益: 5%+ ✅
```

**总时间**: 6小时（1天）

---

## 🚀 优先级2: 动态仓位调整

**为什么重要**:
- 比亚迪只获得13.7%涨幅（买入+54%）
- 提高仓位可以在牛市获得更多收益

**实现步骤**:

### Step 1: 趋势强度计算 (1小时)

```python
def calculate_trend_strength(df, i, lookback=20):
    """计算趋势强度 (0-1)"""
    
    # 价格变化
    price_change = (df['close'].iloc[i] / df['close'].iloc[i-lookback] - 1)
    
    # MA排列强度
    ma10 = df['ma10'].iloc[i]
    ma30 = df['ma30'].iloc[i]
    ma_strength = (ma10 / ma30 - 1) * 10
    
    # 综合评分
    strength = (price_change + ma_strength) / 2
    strength = max(0, min(1, strength + 0.5))  # 归一化到0-1
    
    return strength
```

### Step 2: 动态仓位逻辑 (1小时)

```python
def calculate_dynamic_position(trend_strength, profit_pct, market_state):
    """动态计算仓位"""
    
    base_position = 0.30
    
    # 趋势强劲，增加仓位
    if trend_strength > 0.7:
        base_position *= 1.4
    elif trend_strength > 0.5:
        base_position *= 1.2
    
    # 已经盈利，增加仓位（让利润奔跑）
    if profit_pct > 0.15:
        base_position *= 1.3
    elif profit_pct > 0.10:
        base_position *= 1.2
    
    # 牛市环境，增加仓位
    if market_state == 'bull':
        base_position *= 1.2
    elif market_state == 'bear':
        base_position *= 0.5
    
    # 限制最大仓位
    return min(base_position, 0.60)
```

### Step 3: 集成和测试 (2小时)

**任务**:
- [ ] 修改回测函数
- [ ] 测试动态仓位
- [ ] 对比固定仓位

**预期结果**:
```
比亚迪收益:
- 固定仓位: +7.38%
- 动态仓位: +15%+ ✅

整体收益:
- 固定仓位: -1.25%
- 动态仓位: 8%+ ✅
```

**总时间**: 4小时（半天）

---

## 🚀 优先级3: 扩展数据验证

**为什么重要**:
- 仅4只股票，样本量小
- 需要更多数据验证稳定性

**实现步骤**:

### Step 1: 修复数据获取脚本 (1小时)

**问题**: 网络不稳定，akshare限制

**解决方案**:
```python
# 添加更长的延迟
time.sleep(5)  # 5秒延迟

# 更多的重试次数
max_retries = 5

# 错误处理
try:
    df = ak.stock_zh_a_hist(...)
except Exception as e:
    print(f"错误: {e}")
    time.sleep(10)
    continue
```

### Step 2: 分批获取 (2小时)

**目标**: 10-15只股票

**行业分布**:
- 白酒: 3只（茅台、五粮液、泸州老窖）
- 新能源: 3只（比亚迪、宁德时代、隆基绿能）
- 银行: 3只（平安、招商、兴业）
- 科技: 3只（海康威视、科大讯飞、用友）
- 消费: 3只（伊利、双汇、美的）

### Step 3: 验证策略 (2小时)

**任务**:
- [ ] 在新数据上测试
- [ ] 统计达标率
- [ ] 生成验证报告

**预期结果**:
- 达标率≥50%
- 平均夏普≥0.5
- 样本量≥10只

**总时间**: 5小时（半天）

---

## 📅 时间表

### 本周 (3月4-10日)

| 日期 | 任务 | 时间 | 优先级 |
|------|------|------|--------|
| **周一 (3/4)** | 总结+计划 | 2h | ⭐⭐⭐⭐⭐ |
| **周二 (3/5)** | 市场环境过滤 | 6h | ⭐⭐⭐⭐⭐ |
| **周三 (3/6)** | 动态仓位调整 | 4h | ⭐⭐⭐⭐ |
| **周四 (3/7)** | 扩展数据获取 | 5h | ⭐⭐⭐ |
| **周五 (3/8)** | 综合验证 | 4h | ⭐⭐⭐⭐⭐ |
| **周末 (3/9-10)** | 整理文档 | 3h | ⭐⭐⭐ |

### 下周 (3月11-17日)

| 日期 | 任务 | 时间 |
|------|------|------|
| **周一 (3/11)** | 模拟盘准备 | 2h |
| **周二-周日** | 模拟盘运行 | - |
| **每日** | 监控记录 | 30min |

---

## ✅ 成功标准

### 改进后必须达到:

1. **夏普比率** ≥ 0.5
2. **胜率** ≥ 60%
3. **平均收益** ≥ 5%
4. **最大回撤** ≤ 10%
5. **达标股票** ≥ 50%

### 验证后才能实盘:

1. 至少10只股票验证
2. 模拟盘运行1个月
3. 实际夏普≥0.5
4. 实际胜率≥60%

---

## 🎯 立即行动

### 今天下午 (3月4日)

**优先任务**: 实现市场环境过滤

**步骤**:
1. ✅ 创建行动计划（已完成）
2. ⏭️ 获取上证指数数据
3. ⏭️ 实现过滤函数
4. ⏭️ 测试验证

**预期**: 晚上9点前完成过滤功能

### 明天 (3月5日)

**优先任务**: 完善和验证过滤

**步骤**:
1. 在4只股票上验证
2. 统计提升效果
3. 调整优化参数
4. 生成验证报告

**预期**: 确认过滤效果显著

---

## 📞 需要的资源

### 数据资源
- [x] 4只股票数据（已有）
- [ ] 上证指数数据（需要获取）
- [ ] 更多股票数据（优先级3）

### 技术资源
- [x] Python环境
- [x] akshare库
- [x] pandas/numpy

### 时间资源
- 本周: 24小时
- 下周: 10小时

---

**创建时间**: 2026-03-04 11:30  
**下次更新**: 完成市场过滤后  
**负责人**: AI量化团队
