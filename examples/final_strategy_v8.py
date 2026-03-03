"""
最终优化策略 v8.0
=================
核心原则:
1. 只在上涨趋势中交易
2. 下跌/震荡时空仓
3. 严格风控
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def sma(data, period): return data.rolling(window=period).mean()
def ema(data, period): return data.ewm(span=period, adjust=False).mean()
def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def is_uptrend(df, i):
    """判断是否为上涨趋势"""
    close = df['close'].iloc[:i+1]
    if len(close) < 60:
        return False
    
    price = close.iloc[-1]
    ma_20 = sma(close, 20).iloc[-1]
    ma_60 = sma(close, 60).iloc[-1]
    
    # 价格在均线上方 + 均线多头排列
    return price > ma_20 > ma_60

def final_backtest(df):
    """最终策略"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], 10)
    df['ma_l'] = sma(df['close'], 30)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    
    trades = []
    equity = [cash]
    in_uptrend = False
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 检查趋势
        was_uptrend = in_uptrend
        in_uptrend = is_uptrend(df, i)
        
        # 更新止损
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * 2.5
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 趋势转坏，立即卖出
        if shares > 0 and not in_uptrend:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'trend_exit', 'pnl': pnl})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 买入：进入上涨趋势 + 金叉
        if in_uptrend and ma_s > ma_l and shares == 0:
            new_shares = int(cash * 0.3 / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * 3
                highest = price
                trades.append({'type': 'buy', 'price': price})
        
        # 卖出：死叉
        elif ma_s < ma_l and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl})
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    final = cash
    ret = (final - 100000) / 100000
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    sells = [t for t in trades if t['type'] != 'buy']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    return {
        'final': final, 'return': ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate
    }

print("="*70)
print("📊 最终优化策略 v8.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略: 只在上涨趋势中交易，下跌/震荡时空仓")

data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
}

all_results = []

for stock_name, filepath in data_files.items():
    if not Path(filepath).exists():
        continue
    
    df = load_data(filepath)
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    result = final_backtest(df)
    result['stock'] = stock_name
    result['buy_hold'] = buy_hold
    all_results.append(result)
    
    excess = result['return'] * 100 - buy_hold
    status = "✅" if result['return'] > 0 else "❌"
    
    print(f"\n{stock_name} (买入持有: {buy_hold:+.1f}%):")
    print(f"  {status} 策略收益: {result['return']*100:+.2f}%")
    print(f"  超额收益: {excess:+.1f}%")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  交易次数: {result['trades']}")
    print(f"  胜率: {result['win_rate']*100:.0f}%")

# 汇总
print("\n" + "="*70)
print("📊 汇总")
print("="*70)

avg_ret = np.mean([r['return'] for r in all_results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in all_results])
avg_dd = np.mean([r['max_dd'] for r in all_results]) * 100
avg_bh = np.mean([r['buy_hold'] for r in all_results])
win_count = sum(1 for r in all_results if r['return'] > 0)

print(f"盈利股票: {win_count}/{len(all_results)}")
print(f"\n策略平均收益: {avg_ret:+.2f}%")
print(f"买入持有收益: {avg_bh:+.1f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")

if avg_sharpe > 1.0:
    grade = "A 🏆"
elif avg_sharpe > 0.5:
    grade = "B ✅"
elif avg_sharpe > 0:
    grade = "C ⚠️"
else:
    grade = "D ❌"

print(f"\n策略评级: {grade}")
print("="*70)
