"""
多股票真实数据回测验证
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 指标函数
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
    """加载数据"""
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def backtest(df, name, params):
    """回测"""
    MA_S, MA_L = params['ma_short'], params['ma_long']
    ATR_STOP, ATR_TRAIL = params['atr_stop'], params['atr_trail']
    MAX_POS = params['max_pos']
    
    # 指标
    df['ma_s'] = sma(df['close'], MA_S)
    df['ma_l'] = sma(df['close'], MA_L)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    # 交易
    cash = 100000
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    trailing_stop = 0
    highest = 0
    
    trades = []
    equity = [cash]
    cooldown = 0
    
    for i in range(50, len(df)):
        price = df['close'].iloc[i]
        atr_val = df['atr'].iloc[i]
        ma_s = df['ma_s'].iloc[i]
        ma_l = df['ma_l'].iloc[i]
        rsi_val = df['rsi'].iloc[i]
        
        if cooldown > 0:
            cooldown -= 1
            equity.append(cash + position * price)
            continue
        
        # 更新追踪止损
        if position > 0 and price > highest:
            highest = price
            new_trail = highest - atr_val * ATR_TRAIL
            trailing_stop = max(trailing_stop, new_trail)
        
        # 止损
        if position > 0 and (price <= stop_loss or price <= trailing_stop):
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl})
            cash = position * price
            position = 0
            cooldown = 3
            equity.append(cash)
            continue
        
        # 止盈
        if position > 0 and price >= take_profit:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'profit', 'pnl': pnl})
            cash = position * price
            position = 0
            cooldown = 2
            equity.append(cash)
            continue
        
        # 买入: 均线金叉 + RSI不超买
        if ma_s > ma_l and rsi_val < 70 and position == 0:
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
        
        # 卖出: 均线死叉
        elif ma_s < ma_l and position > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl})
            cash = position * price
            position = 0
            cooldown = 3
        
        equity.append(cash + position * price)
    
    # 平仓
    if position > 0:
        cash = position * df['close'].iloc[-1]
    
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
    
    return {
        'name': name,
        'final': final,
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate
    }

# 参数组合
param_sets = [
    {'name': '保守', 'ma_short': 5, 'ma_long': 30, 'atr_stop': 2.5, 'atr_trail': 2.0, 'max_pos': 0.25},
    {'name': '平衡', 'ma_short': 5, 'ma_long': 20, 'atr_stop': 2.0, 'atr_trail': 1.5, 'max_pos': 0.3},
    {'name': '激进', 'ma_short': 3, 'ma_long': 15, 'atr_stop': 1.5, 'atr_trail': 1.2, 'max_pos': 0.4},
]

print("\n" + "="*70)
print("📊 真实数据多策略回测")
print("="*70)

# 加载数据
data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
}

all_results = []

for stock_name, filepath in data_files.items():
    if not Path(filepath).exists():
        continue
    
    df = load_data(filepath)
    print(f"\n{stock_name}: {len(df)}条, ¥{df['close'].iloc[0]:.2f} -> ¥{df['close'].iloc[-1]:.2f}")
    
    for params in param_sets:
        result = backtest(df, stock_name, params)
        result['strategy'] = params['name']
        all_results.append(result)
        
        status = "✅" if result['return'] > 0 else "❌"
        print(f"  {status} {params['name']}: 收益{result['return']*100:+.2f}%, 夏普{result['sharpe']:.2f}, 胜率{result['win_rate']*100:.0f}%")

# 汇总
print("\n" + "="*70)
print("📈 汇总统计")
print("="*70)

avg_return = np.mean([r['return'] for r in all_results])
avg_sharpe = np.mean([r['sharpe'] for r in all_results])
win_count = sum(1 for r in all_results if r['return'] > 0)

print(f"测试组合: {len(all_results)}个")
print(f"盈利组合: {win_count}/{len(all_results)} ({win_count/len(all_results)*100:.0f}%)")
print(f"平均收益: {avg_return*100:+.2f}%")
print(f"平均夏普: {avg_sharpe:.2f}")

# 最佳组合
best = max(all_results, key=lambda x: x['sharpe'])
print(f"\n最佳组合: {best['name']} - {best['strategy']}")
print(f"  收益: {best['return']*100:+.2f}%")
print(f"  夏普: {best['sharpe']:.2f}")
print(f"  回撤: {best['max_dd']*100:.2f}%")
print(f"  胜率: {best['win_rate']*100:.0f}%")
