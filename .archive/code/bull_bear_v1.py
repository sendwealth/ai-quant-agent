"""
牛熊分离策略 v1.0
================
核心：不同市场环境使用不同参数
- 牛市：激进（大仓位，宽止损）
- 熊市：保守（小仓位，紧止损）
- 震荡：均值回归
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
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def detect_market_regime(df, i):
    """检测市场状态：bull/bear/sideways"""
    close = df['close'].iloc[:i+1]
    if len(close) < 60:
        return 'sideways'
    
    price = close.iloc[-1]
    ma_20 = sma(close, 20).iloc[-1]
    ma_60 = sma(close, 60).iloc[-1]
    
    # 20日收益率
    ret_20 = (price / close.iloc[-20] - 1) if len(close) > 20 else 0
    
    # 牛市：价格在均线上方，持续上涨
    if price > ma_20 > ma_60 and ret_20 > 0.05:
        return 'bull'
    # 熊市：价格在均线下方，持续下跌
    elif price < ma_20 < ma_60 and ret_20 < -0.05:
        return 'bear'
    else:
        return 'sideways'

def get_regime_params(regime):
    """根据市场状态返回参数"""
    params = {
        'bull': {
            'ma_short': 10, 'ma_long': 30,
            'atr_stop': 3.5, 'atr_trail': 3.0,
            'max_pos': 0.40,
            'name': '牛市激进'
        },
        'bear': {
            'ma_short': 5, 'ma_long': 20,
            'atr_stop': 1.5, 'atr_trail': 1.0,
            'max_pos': 0.10,
            'name': '熊市保守'
        },
        'sideways': {
            'ma_short': 10, 'ma_long': 30,
            'atr_stop': 2.5, 'atr_trail': 2.0,
            'max_pos': 0.15,
            'name': '震荡均值回归'
        }
    }
    return params.get(regime, params['sideways'])

def bull_bear_backtest(df):
    """牛熊分离回测"""
    df = df.copy()
    
    # 计算基础指标
    df['ma_10'] = sma(df['close'], 10)
    df['ma_20'] = sma(df['close'], 20)
    df['ma_30'] = sma(df['close'], 30)
    df['ma_60'] = sma(df['close'], 60)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    current_regime = 'sideways'
    current_params = get_regime_params('sideways')
    
    trades = []
    equity = [cash]
    regime_stats = {'bull': 0, 'bear': 0, 'sideways': 0}
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        
        if pd.isna(df['ma_10'].iloc[i]) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 检测市场状态
        prev_regime = current_regime
        current_regime = detect_market_regime(df, i)
        regime_stats[current_regime] += 1
        
        # 市场状态变化时更新参数
        if current_regime != prev_regime:
            current_params = get_regime_params(current_regime)
            
            # 如果转熊且持仓，考虑减仓
            if current_regime == 'bear' and shares > 0:
                pnl = (price - entry_price) / entry_price
                trades.append({'type': 'regime_change', 'from': prev_regime, 'to': current_regime, 'pnl': pnl})
                cash += shares * price
                shares = 0
        
        # 追踪止损
        if shares > 0 and high > highest:
            highest = high
            new_stop = highest - atr_val * current_params['atr_trail']
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and low <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'regime': current_regime, 'pnl': pnl})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 根据市场状态采用不同策略
        if current_regime == 'bull':
            # 牛市：趋势跟随
            should_buy = (
                shares == 0 and
                df['ma_10'].iloc[i] > df['ma_30'].iloc[i] and
                rsi_val < 70
            )
        elif current_regime == 'bear':
            # 熊市：严重超卖才买
            should_buy = (
                shares == 0 and
                rsi_val < 25
            )
        else:
            # 震荡：均值回归
            should_buy = (
                shares == 0 and
                rsi_val < 30
            )
        
        if should_buy:
            new_shares = int(cash * current_params['max_pos'] / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * current_params['atr_stop']
                highest = high
                trades.append({'type': 'buy', 'regime': current_regime, 'params': current_params['name']})
        
        # 卖出信号
        should_sell = False
        if current_regime == 'bull' and df['ma_10'].iloc[i] < df['ma_30'].iloc[i]:
            should_sell = True
        elif current_regime == 'bear' and rsi_val > 50:
            should_sell = True
        elif current_regime == 'sideways' and rsi_val > 65:
            should_sell = True
        
        if should_sell and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'regime': current_regime, 'pnl': pnl})
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    # 计算
    final = cash
    ret = (final - 100000) / 100000
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    sells = [t for t in trades if 'pnl' in t]
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    total = sum(regime_stats.values())
    regime_pct = {k: v/total*100 for k, v in regime_stats.items()}
    
    return {
        'return': ret, 'sharpe': sharpe, 'max_dd': max_dd,
        'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate, 'regime_pct': regime_pct
    }

print("="*70)
print("🐂 牛熊分离策略 v1.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略参数:")
print("  牛市: 仓位40%, 止损3.5x ATR, 趋势跟随")
print("  熊市: 仓位10%, 止损1.5x ATR, 超卖反弹")
print("  震荡: 仓位15%, 止损2.5x ATR, 均值回归")

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
    r = bull_bear_backtest(df)
    r['stock'] = name
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    
    print(f"\n{name} (买入持有: {bh:+.1f}%):")
    print(f"  {status} 策略: {r['return']*100:+.2f}% | 夏普: {r['sharpe']:.2f}")
    print(f"  回撤: {r['max_dd']*100:.1f}% | 交易: {r['trades']}次 | 胜率: {r['win_rate']*100:.0f}%")
    print(f"  市场分布: 牛{r['regime_pct']['bull']:.0f}% 熊{r['regime_pct']['bear']:.0f}% 震荡{r['regime_pct']['sideways']:.0f}%")

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
    print("\n评级: A 🏆")
elif avg_sharpe > 0.5:
    print("\n评级: B ✅")
else:
    print("\n评级: C ⚠️")

print("="*70)
