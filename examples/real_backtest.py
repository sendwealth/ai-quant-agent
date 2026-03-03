"""
真实数据回测验证
"""

import pandas as pd
import numpy as np
from datetime import datetime

# 读取真实数据
df = pd.read_csv('data/real_600519.csv')
df = df.rename(columns={
    '日期': 'datetime', '开盘': 'open', '最高': 'high',
    '最低': 'low', '收盘': 'close', '成交量': 'volume'
})
df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
df['datetime'] = pd.to_datetime(df['datetime'])

print(f"\n{'='*60}")
print(f"📊 真实数据回测 - 600519 茅台")
print(f"{'='*60}")
print(f"数据: {len(df)} 条")
print(f"时间: {df['datetime'].iloc[0]} - {df['datetime'].iloc[-1]}")
print(f"价格: ¥{df['close'].iloc[0]:.2f} -> ¥{df['close'].iloc[-1]:.2f}")
print(f"涨跌: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:+.2f}%")

# 技术指标
def sma(data, period): return data.rolling(window=period).mean()
def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

# 策略参数
MA_SHORT, MA_LONG = 5, 20
ATR_STOP, ATR_TRAIL = 2.0, 1.5
MAX_POS = 0.3

# 计算指标
df['ma_s'] = sma(df['close'], MA_SHORT)
df['ma_l'] = sma(df['close'], MA_LONG)
df['atr'] = atr(df['high'], df['low'], df['close'], 14)

# 回测
cash = 100000
position = 0
entry_price = 0
stop_loss = 0
take_profit = 0
trailing_stop = 0
highest = 0

trades = []
equity = [cash]

for i in range(30, len(df)):
    price = df['close'].iloc[i]
    ma_s = df['ma_s'].iloc[i]
    ma_l = df['ma_l'].iloc[i]
    atr_val = df['atr'].iloc[i]
    
    # 更新追踪止损
    if position > 0 and price > highest:
        highest = price
        trailing_stop = max(trailing_stop, highest - atr_val * ATR_TRAIL)
    
    # 止损
    if position > 0 and (price <= stop_loss or price <= trailing_stop):
        pnl = (price - entry_price) / entry_price
        trades.append({'type': 'stop', 'price': price, 'pnl': pnl})
        cash = position * price
        position = 0
        continue
    
    # 止盈
    if position > 0 and price >= take_profit:
        pnl = (price - entry_price) / entry_price
        trades.append({'type': 'profit', 'price': price, 'pnl': pnl})
        cash = position * price
        position = 0
        continue
    
    # 买入信号
    if ma_s > ma_l and position == 0:
        shares = int(cash * MAX_POS / price)
        if shares > 0:
            position = shares
            cash -= shares * price
            entry_price = price
            stop_loss = price - atr_val * ATR_STOP
            take_profit = price + atr_val * ATR_STOP * 2
            trailing_stop = stop_loss
            highest = price
            trades.append({'type': 'buy', 'price': price, 'shares': shares})
    
    # 卖出信号
    elif ma_s < ma_l and position > 0:
        pnl = (price - entry_price) / entry_price
        trades.append({'type': 'signal', 'price': price, 'pnl': pnl})
        cash = position * price
        position = 0
    
    equity.append(cash + position * price)

# 最终平仓
if position > 0:
    cash = position * df['close'].iloc[-1]

final_equity = cash
total_return = (final_equity - 100000) / 100000

# 计算夏普
eq = pd.Series(equity)
rets = eq.pct_change().dropna()
sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0

# 最大回撤
peak = eq.expanding().max()
dd = (eq - peak) / peak
max_dd = dd.min()

# 胜率
sells = [t for t in trades if 'pnl' in t]
wins = [t for t in sells if t['pnl'] > 0]
win_rate = len(wins) / len(sells) if sells else 0

print(f"\n{'='*60}")
print("📊 回测结果")
print("="*60)
print(f"最终权益: ¥{final_equity:,.2f}")
print(f"总收益: {total_return*100:+.2f}%")
print(f"夏普比率: {sharpe:.2f}")
print(f"最大回撤: {max_dd*100:.2f}%")
print(f"交易次数: {len([t for t in trades if t['type']=='buy'])}")
print(f"胜率: {win_rate*100:.1f}%")

# 评级
if sharpe > 1.5 and max_dd > -0.10:
    grade = "A 🏆"
elif sharpe > 1.0 and max_dd > -0.15:
    grade = "B ✅"
elif sharpe > 0.5:
    grade = "C ⚠️"
else:
    grade = "D ❌"

print(f"\n策略评级: {grade}")
print("="*60)

# 交易详情
print("\n交易记录:")
for t in trades[:10]:
    if t['type'] == 'buy':
        print(f"  买入: {t['shares']}股 @ ¥{t['price']:.2f}")
    else:
        print(f"  {t['type']}: @ ¥{t['price']:.2f} | PnL: {t['pnl']*100:+.2f}%")

if len(trades) > 10:
    print(f"  ... 共 {len(trades)} 笔交易")
