"""
综合最优策略 v13.0 (最终版)
==========================
结合所有验证结果：
- 保守基础参数
- 波动率自适应
- 趋势跟随
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

def optimal_backtest(df):
    """综合最优策略"""
    df = df.copy()
    df['ma_5'] = sma(df['close'], 5)
    df['ma_10'] = sma(df['close'], 10)
    df['ma_30'] = sma(df['close'], 30)
    df['ma_60'] = sma(df['close'], 60)
    df['ema_10'] = ema(df['close'], 10)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    
    trades = []
    equity = [cash]
    cooldown = 0
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_5 = float(df['ma_5'].iloc[i])
        ma_10 = float(df['ma_10'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        ma_60 = float(df['ma_60'].iloc[i])
        ema_10 = float(df['ema_10'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        
        if pd.isna(ma_5) or pd.isna(ma_30) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 计算波动率
        close = df['close'].iloc[:i+1]
        vol = close.pct_change().tail(20).std() * np.sqrt(252)
        
        # 判断趋势
        uptrend = price > ma_30 > ma_60
        downtrend = price < ma_30 < ma_60
        
        # 根据波动率和趋势决定参数
        if uptrend and vol < 0.35:
            # 低波动上涨：激进
            pos = 0.40
            atr_stop = 3.0
            atr_trail = 2.5
        elif uptrend:
            # 高波动上涨：适中
            pos = 0.30
            atr_stop = 2.5
            atr_trail = 2.0
        elif downtrend:
            # 下跌：保守
            pos = 0.15
            atr_stop = 2.0
            atr_trail = 1.5
        else:
            # 震荡：均值回归
            pos = 0.25
            atr_stop = 2.5
            atr_trail = 2.0
        
        if cooldown > 0:
            cooldown -= 1
            equity.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * atr_trail
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl})
            cash += shares * price
            shares = 0
            cooldown = 3
            equity.append(cash)
            continue
        
        # 买入信号
        should_buy = False
        
        if uptrend:
            # 上涨趋势：均线金叉
            should_buy = ma_5 > ma_30 and ema_10 > ma_30 and rsi_val < 70
        elif downtrend:
            # 下跌趋势：严重超卖
            should_buy = rsi_val < 25
        else:
            # 震荡：超卖
            should_buy = rsi_val < 30
        
        if should_buy and shares == 0:
            new_shares = int(cash * pos / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * atr_stop
                highest = price
                trades.append({'type': 'buy', 'mode': 'uptrend' if uptrend else 'reversion'})
        
        # 卖出信号
        should_sell = False
        
        if uptrend:
            # 上涨趋势：死叉
            should_sell = ma_5 < ma_30
        elif downtrend:
            # 下跌趋势：反弹5%
            should_sell = price > entry_price * 1.05
        else:
            # 震荡：超买
            should_sell = rsi_val > 65 or (shares > 0 and price > entry_price * 1.03)
        
        if should_sell and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl})
            cash += shares * price
            shares = 0
            cooldown = 2
        
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
print("📊 综合最优策略 v13.0 (最终版)")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略组合:")
print("  低波动上涨: 40%仓位, 止损3.0x")
print("  高波动上涨: 30%仓位, 止损2.5x")
print("  下跌市场: 15%仓位, 止损2.0x, 超卖反弹")
print("  震荡市场: 25%仓位, 止损2.5x, 均值回归")

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
    
    r = optimal_backtest(df)
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
print("📈 最终汇总")
print("="*70)

avg_ret = np.mean([r['return'] for r in results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['max_dd'] for r in results]) * 100
avg_bh = np.mean([r['bh'] for r in results])
win_count = sum(1 for r in results if r['return'] > 0)

print(f"盈利股票: {win_count}/{len(results)}")
print(f"\n策略平均收益: {avg_ret:+.2f}%")
print(f"买入持有平均: {avg_bh:+.1f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")

if avg_sharpe > 1.0:
    grade = "A 🏆 优秀"
elif avg_sharpe > 0.5:
    grade = "B ✅ 良好"
elif avg_sharpe > 0:
    grade = "C ⚠️ 一般"
else:
    grade = "D ❌ 需优化"

print(f"\n最终评级: {grade}")
print("="*70)
