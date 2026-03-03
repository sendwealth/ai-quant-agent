"""
波动率自适应策略 v11.0
=====================
根据市场波动率自动调整：
- 高波动：收紧止损，减少仓位
- 低波动：放宽止损，增加仓位
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

def get_volatility_regime(df, i):
    """判断波动率状态"""
    close = df['close'].iloc[:i+1]
    if len(close) < 60:
        return 'normal'
    
    # 计算年化波动率
    rets = close.pct_change().tail(20)
    vol = rets.std() * np.sqrt(252)
    
    if vol > 0.35:  # 高波动
        return 'high'
    elif vol < 0.15:  # 低波动
        return 'low'
    else:
        return 'normal'

def get_params_for_regime(regime):
    """根据波动率返回参数"""
    params = {
        'high':   {'atr_stop': 2.0, 'atr_trail': 1.5, 'max_pos': 0.15, 'name': '高波动'},
        'normal': {'atr_stop': 2.5, 'atr_trail': 2.0, 'max_pos': 0.25, 'name': '正常'},
        'low':    {'atr_stop': 3.0, 'atr_trail': 2.5, 'max_pos': 0.35, 'name': '低波动'},
    }
    return params.get(regime, params['normal'])

def volatility_adaptive_backtest(df):
    """波动率自适应回测"""
    df = df.copy()
    df['ma_10'] = sma(df['close'], 10)
    df['ma_30'] = sma(df['close'], 30)
    df['ema_10'] = ema(df['close'], 10)
    df['ema_30'] = ema(df['close'], 30)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    current_params = get_params_for_regime('normal')
    
    trades = []
    equity = [cash]
    regime_history = []
    cooldown = 0
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_10 = float(df['ma_10'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        ema_10 = float(df['ema_10'].iloc[i])
        ema_30 = float(df['ema_30'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        
        if pd.isna(ma_10) or pd.isna(ma_30) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 波动率判断
        regime = get_volatility_regime(df, i)
        regime_history.append(regime)
        
        if cooldown > 0:
            cooldown -= 1
            equity.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * current_params['atr_trail']
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl, 'regime': regime})
            cash += shares * price
            shares = 0
            cooldown = 3
            equity.append(cash)
            continue
        
        # 买入信号
        signal_buy = ma_10 > ma_30 and ema_10 > ema_30 and rsi_val < 70
        
        if signal_buy and shares == 0:
            current_params = get_params_for_regime(regime)
            new_shares = int(cash * current_params['max_pos'] / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * current_params['atr_stop']
                highest = price
                trades.append({'type': 'buy', 'regime': regime, 'params': current_params['name']})
        
        # 卖出信号
        elif ma_10 < ma_30 and ema_10 < ema_30 and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl, 'regime': regime})
            cash += shares * price
            shares = 0
            cooldown = 3
        
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
    
    # 波动率分布
    regime_dist = {}
    for r in regime_history:
        regime_dist[r] = regime_dist.get(r, 0) + 1
    
    return {
        'final': final, 'return': ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate, 'regime_dist': regime_dist
    }

print("="*70)
print("📊 波动率自适应策略 v11.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略特点:")
print("  - 高波动(>35%): 止损2.0x, 仓位15%")
print("  - 正常波动: 止损2.5x, 仓位25%")
print("  - 低波动(<15%): 止损3.0x, 仓位35%")

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
    
    r = volatility_adaptive_backtest(df)
    r['stock'] = name
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    excess = r['return'] * 100 - bh
    
    print(f"\n{name} (买入持有: {bh:+.1f}%):")
    print(f"  {status} 策略: {r['return']*100:+.2f}% | 超额: {excess:+.1f}%")
    print(f"  夏普: {r['sharpe']:.2f} | 回撤: {r['max_dd']*100:.1f}%")
    print(f"  交易: {r['trades']}次 | 胜率: {r['win_rate']*100:.0f}%")
    print(f"  波动率分布: {r['regime_dist']}")

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
    print("评级: A 🏆")
elif avg_sharpe > 0.5:
    print("评级: B ✅")
elif avg_sharpe > 0:
    print("评级: C ⚠️")
else:
    print("评级: D ❌")
print("="*70)
