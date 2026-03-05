# 📋 模拟账户开通与交易计划

**日期**: 2026-03-04  
**目标**: 验证优化策略，积累实战经验  

---

## 🎯 交易策略

### 推荐配置

**股票池**:
1. **宁德时代 (300750)** - 60%仓位
   - 夏普: 0.75 ✅
   - 收益: +21.3% ✅
   - 胜率: 58.6%
   
2. **中国平安 (601318)** - 40%仓位
   - 夏普: 0.50 ✅
   - 收益: +9.8% ✅
   - 胜率: 61.5% ✅

**策略参数**:
```python
MA周期: 10/30
ATR止损: 2.5倍
初始仓位: 30%
动态仓位: 是
MACD确认: 是
RSI过滤: 是
分批止盈: 10%卖50%, 20%清仓
```

**预期表现**:
- 预期收益: 5-15%/年
- 最大回撤: <10%
- 夏普比率: 0.5-0.7

---

## 💰 模拟账户选择

### 方案A: 同花顺模拟炒股 (推荐⭐⭐⭐⭐⭐)

**优点**:
- ✅ 免费
- ✅ 界面友好
- ✅ 实时行情
- ✅ 支持所有A股

**开通步骤**:
1. 下载"同花顺"APP
2. 注册账号
3. 点击"模拟炒股"
4. 初始资金: 50万元（可调）

**网址**: http://www.10jqka.com.cn/

### 方案B: 东方财富模拟炒股

**优点**:
- ✅ 免费
- ✅ 数据准确
- ✅ 社区活跃

**开通步骤**:
1. 访问: http://guba.eastmoney.com/
2. 注册东方财富账号
3. 进入"模拟炒股"

### 方案C: 雪球组合

**优点**:
- ✅ 专业
- ✅ 社区强大
- ✅ 可公开分享

**开通步骤**:
1. 下载"雪球"APP
2. 注册账号
3. 创建组合

### 方案D: TradingView (国际版)

**优点**:
- ✅ 图表强大
- ✅ 策略回测
- ✅ 国际化

**缺点**:
- ⚠️ A股数据可能不全

---

## 📝 交易计划

### 初始配置

```yaml
模拟资金: 100,000元
股票池: 宁德时代(60%), 中国平安(40%)
单只最大仓位: 60%
总仓位上限: 70%
止损线: -8%
```

### 交易规则

#### 买入条件

```python
def should_buy(stock_data):
    # 1. MA趋势向上
    if ma_10 < ma_30:
        return False
    
    # 2. MACD确认
    if macd_histogram < 0:
        return False
    
    # 3. RSI适中
    if rsi > 70 or rsi < 30:
        return False
    
    # 4. 成交量放大
    if volume < avg_volume_20:
        return False
    
    return True
```

#### 卖出条件

```python
def should_sell(position, current_price):
    # 1. 止损
    if current_price <= position.stop_loss:
        return "止损"
    
    # 2. 止盈1 (盈利10%)
    if position.profit_pct >= 0.10 and not position.partial_sold:
        return "止盈50%"
    
    # 3. 止盈2 (盈利20%)
    if position.profit_pct >= 0.20:
        return "清仓"
    
    # 4. 趋势反转
    if ma_10 < ma_30 and macd_histogram < 0:
        return "趋势反转"
    
    return None
```

### 仓位管理

```python
def calculate_position(market_state, volatility):
    base = 0.30
    
    # 牛市加仓
    if market_state == 'bull':
        base *= 1.2  # 36%
    
    # 熊市减仓
    elif market_state == 'bear':
        base *= 0.5  # 15%
    
    # 高波动减仓
    if volatility > 0.03:
        base *= 0.7
    
    return min(base, 0.60)  # 最大60%
```

---

## 📊 监控工具

### 交易日志模板

```markdown
## 日期: YYYY-MM-DD

### 持仓情况
| 股票 | 数量 | 成本 | 现价 | 盈亏 | 盈亏% |
|------|------|------|------|------|-------|
| 宁德时代 | 200 | 180.00 | 185.00 | +1000 | +2.8% |
| 中国平安 | 100 | 45.00 | 46.00 | +100 | +2.2% |

### 今日操作
- [ ] 买入宁德时代 200股 @ 180.00
- [ ] 卖出中国平安 50股 @ 46.00 (止盈)

### 市场分析
- 大盘趋势: [上涨/下跌/震荡]
- MA20: XXXX, MA60: XXXX
- 趋势: [多头/空头]

### 交易信号
- 宁德时代: MA金叉, MACD向上, RSI 55
- 中国平安: MA多头, MACD走平, RSI 60

### 明日计划
- 宁德时代: 持有，止损175.00
- 中国平安: 若跌破44.00则减仓
```

### Excel记录表

创建Excel表格，包含以下列：

```
日期 | 股票代码 | 操作 | 数量 | 价格 | 手续费 | 
总市值 | 现金 | 总资产 | 日收益 | 累计收益 | 备注
```

### Python监控脚本

```python
# 文件: monitor.py
import pandas as pd
from datetime import datetime

class PortfolioTracker:
    def __init__(self):
        self.trades = []
        self.positions = {}
        self.cash = 100000
        self.initial_capital = 100000
    
    def buy(self, code, name, shares, price, date):
        """买入"""
        cost = shares * price
        if cost > self.cash:
            print(f"❌ 资金不足")
            return
        
        self.cash -= cost
        
        if code not in self.positions:
            self.positions[code] = {
                'name': name,
                'shares': 0,
                'avg_cost': 0
            }
        
        # 更新持仓
        pos = self.positions[code]
        total_shares = pos['shares'] + shares
        pos['avg_cost'] = (pos['avg_cost'] * pos['shares'] + cost) / total_shares
        pos['shares'] = total_shares
        
        # 记录交易
        self.trades.append({
            'date': date,
            'code': code,
            'name': name,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'amount': cost
        })
        
        print(f"✅ 买入 {name} {shares}股 @ {price:.2f}")
    
    def sell(self, code, shares, price, date, reason=''):
        """卖出"""
        if code not in self.positions or self.positions[code]['shares'] < shares:
            print(f"❌ 持仓不足")
            return
        
        amount = shares * price
        self.cash += amount
        
        pos = self.positions[code]
        profit = (price - pos['avg_cost']) * shares
        
        pos['shares'] -= shares
        if pos['shares'] == 0:
            del self.positions[code]
        
        self.trades.append({
            'date': date,
            'code': code,
            'name': pos['name'],
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'amount': amount,
            'profit': profit,
            'reason': reason
        })
        
        print(f"✅ 卖出 {pos['name']} {shares}股 @ {price:.2f}, 盈亏: {profit:+.2f} ({reason})")
    
    def summary(self, current_prices):
        """汇总"""
        total_value = self.cash
        
        print("\n" + "="*70)
        print("持仓汇总")
        print("="*70)
        
        for code, pos in self.positions.items():
            current_price = current_prices.get(code, pos['avg_cost'])
            market_value = pos['shares'] * current_price
            profit = (current_price - pos['avg_cost']) * pos['shares']
            profit_pct = (current_price / pos['avg_cost'] - 1) * 100
            
            total_value += market_value
            
            print(f"\n{pos['name']} ({code})")
            print(f"  持仓: {pos['shares']}股")
            print(f"  成本: {pos['avg_cost']:.2f}")
            print(f"  现价: {current_price:.2f}")
            print(f"  市值: {market_value:.2f}")
            print(f"  盈亏: {profit:+.2f} ({profit_pct:+.2f}%)")
        
        total_return = (total_value / self.initial_capital - 1) * 100
        
        print(f"\n现金: {self.cash:.2f}")
        print(f"总资产: {total_value:.2f}")
        print(f"总收益: {total_return:+.2f}%")
        print("="*70)
        
        return {
            'cash': self.cash,
            'total_value': total_value,
            'total_return': total_return
        }

# 使用示例
tracker = PortfolioTracker()

# 买入
tracker.buy('300750', '宁德时代', 200, 180.00, '2026-03-05')
tracker.buy('601318', '中国平安', 100, 45.00, '2026-03-05')

# 查看持仓
current_prices = {'300750': 185.00, '601318': 46.00}
tracker.summary(current_prices)

# 卖出
tracker.sell('601318', 50, 46.00, '2026-03-10', '止盈50%')
```

---

## 📅 执行计划

### 第1周: 准备阶段

**Day 1 (今天)**:
- [x] 确定交易策略
- [ ] 开通模拟账户
- [ ] 配置初始资金10万

**Day 2**:
- [ ] 设置股票池
- [ ] 熟悉交易界面
- [ ] 准备监控工具

**Day 3-7**:
- [ ] 观察市场
- [ ] 等待买入信号
- [ ] 记录每日行情

### 第2-4周: 建仓阶段

**目标**:
- 宁德时代: 建仓60% (6万)
- 中国平安: 建仓40% (4万)

**策略**:
- 分批建仓（每次30%）
- 等待MA金叉信号
- 严格执行止损

### 第2-3月: 持仓管理

**日常任务**:
- 每日收盘后检查信号
- 更新交易日志
- 调整止损止盈位

**每周任务**:
- 复盘本周交易
- 分析盈亏原因
- 优化参数

### 第4周: 评估阶段

**评估指标**:
- 总收益率
- 夏普比率
- 最大回撤
- 胜率
- 交易次数

---

## ✅ 检查清单

### 开通前

- [ ] 选择模拟平台（推荐同花顺）
- [ ] 下载APP
- [ ] 注册账号
- [ ] 开通模拟炒股
- [ ] 设置初始资金10万

### 交易前

- [ ] 添加宁德时代、中国平安到自选
- [ ] 设置MA指标(10/30)
- [ ] 设置MACD指标
- [ ] 设置RSI指标
- [ ] 准备Excel记录表

### 每日检查

- [ ] 收盘后检查MA信号
- [ ] 更新交易日志
- [ ] 计算当日盈亏
- [ ] 记录市场状态

---

## 🎯 成功标准

### 3个月后评估

**优秀** (继续实盘):
- 收益 ≥ 10%
- 夏普 ≥ 0.5
- 最大回撤 < 8%

**良好** (继续观察):
- 收益 5-10%
- 夏普 0.3-0.5
- 最大回撤 < 12%

**一般** (需要优化):
- 收益 0-5%
- 夏普 0-0.3
- 最大回撤 < 15%

**失败** (重新设计):
- 收益 < 0%
- 夏普 < 0
- 最大回撤 > 15%

---

## 📞 技术支持

### 遇到问题

1. **平台问题**: 查看平台帮助中心
2. **策略问题**: 参考本项目的`FINAL_REPORT_20260304.md`
3. **数据问题**: 使用TuShare获取实时数据

### 学习资源

- 《海龟交易法则》
- 《通向财务自由之路》
- 雪球网: https://xueqiu.com
- 集思录: https://www.jisilu.cn

---

## 🚀 开始行动

### 立即执行

1. **下载同花顺APP** (5分钟)
2. **注册并开通模拟炒股** (10分钟)
3. **添加宁德时代、中国平安到自选** (5分钟)
4. **设置技术指标** (10分钟)
5. **创建Excel记录表** (10分钟)

**总计**: 40分钟准备完成！

---

**创建时间**: 2026-03-04 15:35  
**执行时间**: 3个月  
**下次评估**: 2026-06-04  
**负责人**: 你自己 💪
