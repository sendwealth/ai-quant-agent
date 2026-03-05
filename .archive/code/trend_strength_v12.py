"""
趋势强度策略 v12.0
==================
核心思路：
- 强趋势：大仓位持有，放宽止损
- 弱趋势：小仓位，严格止损
- 无趋势：空仓
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def sma(data, period): return data.rolling(window=period).mean()
def ema(data, period): return data.ewm(span=period, adjust=False).mean()
def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))
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

def calc_trend_strength(df, i):
    """计算趋势强度 0-100"""
    close = df['close'].iloc[:i+1]
    if len(close) < 60:
        return 0
    
    price = close.iloc[-1]
    ma_10 = sma(close, 10).iloc[-1]
    ma_20 = sma(close, 20).iloc[-1]
    ma_60 = sma(close, 60).iloc[-1]
    
    score = 0
    
    # 价格位置
    if price > ma_10: score += 15
    if price > ma_20: score += 15
    if price > ma_60: score += 10
    
    # 均线排列
    if ma_10 > ma_20: score += 15
    if ma_20 > ma_60: score += 15
    
    # 动量
    ret_10 = (price / close.iloc[-10] - 1) if len(close) > 10 else 0
    ret_20 = (price / close.iloc[-20] - 1) if len(close) > 20 else 0
    
    if ret_10 > 0.03: score += 10
    if ret_20 > 0.05: score += 10
    if ret_10 > 0.05: score += 5
    if ret_20 > 0.10: score += 5
    
    return min(score, 100)

def trend_strength_backtest(df):
    """趋势强度策略"""
    df = df.copy()
    df['ma_10'] = sma(df['close'], 10)
    df['ma_20'] = sma(df['close'], 20)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    
    trades = []
    equity = [cash]
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_10 = float(df['ma_10'].iloc[i])
        ma_20 = float(df['ma_20'].iloc[i])
        
        if pd.isna(ma_10) or pd.isna(ma_20) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 趋势强度
        strength = calc_trend_strength(df, i)
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            # 强趋势用更宽的追踪止损
            trail_mult = 3.0 if strength > 70 else 2.5 if strength > 50 else 2.0
            new_stop = highest - atr_val * trail_mult
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl, 'strength': strength})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 买入：趋势强度>40 + 金叉
        if strength > 40 and ma_10 > ma_20 and shares == 0:
            # 强度决定仓位
            if strength > 80:
                pos = 0.50  # 超强趋势
            elif strength > 60:
                pos = 0.40  # 强趋势
            elif strength > 50:
                pos = 0.30  # 中等趋势
            else:
                pos = 0.20  # 弱趋势
            
            new_shares = int(cash * pos / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                # 强度决定止损
                stop_mult = 3.5 if strength > 70 else 3.0 if strength > 50 else 2.5
                stop_loss = price - atr_val * stop_mult
                highest = price
                trades.append({'type': 'buy', 'strength': strength, 'pos': pos})
        
        # 卖出：死叉 或 趋势强度<30
        elif (ma_10 < ma_20 or strength < 30) and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl, 'strength': strength})
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
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
    
    sells = [t for t in trades if 'pnl' in t]
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    return {
        'final': final, 'return': ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate
    }

print("="*70)
print("📊 趋势强度策略 v12.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略:")
print("  强度>80: 仓位50%, 止损3.5x")
print("  强度>60: 仓位40%, 止损3.0x")
print("  强度>50: 仓位30%, 止损3.0x")
print("  强度>40: 仓位20%, 止损2.5x")
print("  强度<30: 空仓/卖出")

data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
    '茅台': 'data/real_600519.csv',
}

results = []
for name, filepath in data_files.items():
    if not Path(filepath).exists():
        continue
    
    df = load_data(filepath)
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    r = trend_strength_backtest(df)
    r['stock'] = name
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    excess = r['return'] * 100 - bh
    
    print(f"\n{name} (买入持有: {bh:+.1f}%):")
    print(f"  {status} 策略: {r['return']*100:+.2f}% | 超额: {excess:+.1f}%")
    print(f"  夏普: {r['sharpe']:.2f} | 回撤: {r['max_dd']*100:.1f}% | 交易: {r['trades']}次 | 胜率: {r['win_rate']*100:.0f}%")

# 汇总
print("\n" + "="*70)
print("📈 汇总")
print("="*70)

avg_ret = np.mean([r['return'] for r in results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['max_dd'] for r in results]) * 100
win_count = sum(1 for r in results if r['return'] > 0)

print(f"盈利股票: {win_count}/{len(results)}")
print(f"平均收益: {avg_ret:+.2f}%")
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

print(f"\n评级: {grade}")
print("="*70)
