"""
简洁高效策略 v7.0
=================
核心原则:
1. 减少交易频率 - 只在明确信号时交易
2. 趋势跟随 - 只做顺势交易
3. 严格止损 - ATR动态止损
4. 适度仓位 - 固定25%仓位
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

def simple_backtest(df, ma_short=10, ma_long=30, atr_mult=3.0, max_pos=0.25):
    """简洁策略"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], ma_short)
    df['ma_l'] = sma(df['close'], ma_long)
    df['ema_s'] = ema(df['close'], ma_short)
    df['ema_l'] = ema(df['close'], ma_long)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    
    trades = []
    equity = [cash]
    prev_signal = 0
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        ema_s = float(df['ema_s'].iloc[i])
        ema_l = float(df['ema_l'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 更新最高价和止损
        if shares > 0:
            if price > highest:
                highest = price
                # 追踪止损：从最高价回撤2个ATR
                new_stop = highest - atr_val * 2
                if new_stop > stop_loss:
                    stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl, 'price': price})
            cash += shares * price
            shares = 0
            equity.append(cash)
            prev_signal = -1
            continue
        
        # 信号：MA和EMA同时金叉
        signal = 1 if (ma_s > ma_l and ema_s > ema_l) else -1 if (ma_s < ma_l and ema_s < ema_l) else 0
        
        # 买入：信号从非正变正
        if signal == 1 and prev_signal != 1 and shares == 0:
            new_shares = int(cash * max_pos / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * atr_mult
                highest = price
                trades.append({'type': 'buy', 'price': price, 'shares': shares})
                prev_signal = 1
        
        # 卖出：信号从非负变负
        elif signal == -1 and prev_signal != -1 and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl, 'price': price})
            cash += shares * price
            shares = 0
            prev_signal = -1
        
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
print("📊 简洁高效策略 v7.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 参数组合测试
params_list = [
    {'name': 'MA10/30', 'ma_short': 10, 'ma_long': 30, 'atr_mult': 3.0, 'max_pos': 0.25},
    {'name': 'MA5/20', 'ma_short': 5, 'ma_long': 20, 'atr_mult': 2.5, 'max_pos': 0.30},
    {'name': 'MA10/50', 'ma_short': 10, 'ma_long': 50, 'atr_mult': 3.5, 'max_pos': 0.20},
]

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
    
    print(f"\n{'='*70}")
    print(f"{stock_name} (买入持有: {buy_hold:+.1f}%)")
    print("="*70)
    
    for p in params_list:
        result = simple_backtest(df, **{k: v for k, v in p.items() if k != 'name'})
        result['stock'] = stock_name
        result['param'] = p['name']
        result['buy_hold'] = buy_hold
        all_results.append(result)
        
        excess = result['return'] * 100 - buy_hold
        status = "✅" if result['return'] > 0 else "❌"
        beat = "📈" if excess > 0 else "📉"
        
        print(f"{status} {p['name']}: {result['return']*100:+.2f}% | 超额{excess:+.1f}% {beat} | "
              f"夏普{result['sharpe']:.2f} | 回撤{result['max_dd']*100:.1f}% | "
              f"交易{result['trades']}次 | 胜率{result['win_rate']*100:.0f}%")

# 找最佳
print("\n" + "="*70)
print("📊 最佳组合")
print("="*70)

best = max(all_results, key=lambda x: x['sharpe'])
print(f"股票: {best['stock']}")
print(f"参数: {best['param']}")
print(f"收益: {best['return']*100:+.2f}%")
print(f"夏普: {best['sharpe']:.2f}")
print(f"回撤: {best['max_dd']*100:.1f}%")
print(f"胜率: {best['win_rate']*100:.0f}%")

# 汇总
avg_ret = np.mean([r['return'] for r in all_results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in all_results])
win_count = sum(1 for r in all_results if r['return'] > 0)

print(f"\n汇总: 盈利{win_count}/{len(all_results)}, 平均收益{avg_ret:+.2f}%, 平均夏普{avg_sharpe:.2f}")
