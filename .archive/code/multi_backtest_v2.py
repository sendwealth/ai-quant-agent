"""
修复版多股票回测
"""
import pandas as pd
import numpy as np
from pathlib import Path

def sma(data, period): return data.rolling(window=period).mean()
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

def backtest(df, name, params):
    """修复版回测"""
    MA_S, MA_L = params['ma_short'], params['ma_long']
    ATR_STOP, ATR_TRAIL = params['atr_stop'], params['atr_trail']
    MAX_POS = params['max_pos']
    
    df = df.copy()
    df['ma_s'] = sma(df['close'], MA_S)
    df['ma_l'] = sma(df['close'], MA_L)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    initial_cash = 100000.0
    cash = initial_cash
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trailing_stop = 0.0
    highest = 0.0
    
    trades = []
    equity_curve = []
    cooldown = 0
    
    for i in range(50, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity_curve.append(cash + shares * price)
            continue
        
        if cooldown > 0:
            cooldown -= 1
            equity_curve.append(cash + shares * price)
            continue
        
        # 更新追踪止损
        if shares > 0 and price > highest:
            highest = price
            new_trail = highest - atr_val * ATR_TRAIL
            if new_trail > trailing_stop:
                trailing_stop = new_trail
        
        # 止损
        if shares > 0 and (price <= stop_loss or price <= trailing_stop):
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'stop', 'pnl': pnl, 'price': price})
            cash += shares * price
            shares = 0
            cooldown = 3
            equity_curve.append(cash)
            continue
        
        # 止盈
        if shares > 0 and price >= take_profit:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'profit', 'pnl': pnl, 'price': price})
            cash += shares * price
            shares = 0
            cooldown = 2
            equity_curve.append(cash)
            continue
        
        # 买入
        if ma_s > ma_l and rsi_val < 70 and shares == 0:
            position_value = cash * MAX_POS
            new_shares = int(position_value / price)
            if new_shares > 0:
                cost = new_shares * price
                if cost <= cash:
                    shares = new_shares
                    cash -= cost
                    entry_price = price
                    stop_loss = price - atr_val * ATR_STOP
                    take_profit = price + atr_val * ATR_STOP * 2
                    trailing_stop = stop_loss
                    highest = price
                    trades.append({'action': 'buy', 'price': price, 'shares': shares})
        
        # 卖出
        elif ma_s < ma_l and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'signal', 'pnl': pnl, 'price': price})
            cash += shares * price
            shares = 0
            cooldown = 3
        
        equity_curve.append(cash + shares * price)
    
    # 最终平仓
    if shares > 0:
        final_price = float(df['close'].iloc[-1])
        cash += shares * final_price
        shares = 0
    
    # 计算指标
    final_equity = cash
    total_return = (final_equity - initial_cash) / initial_cash
    
    if len(equity_curve) > 1:
        eq = pd.Series(equity_curve)
        rets = eq.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        
        peak = eq.expanding().max()
        dd = (eq - peak) / peak
        max_dd = dd.min()
    else:
        sharpe = 0
        max_dd = 0
    
    sells = [t for t in trades if t['action'] == 'sell']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    return {
        'name': name,
        'strategy': params['name'],
        'final': final_equity,
        'return': total_return,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['action'] == 'buy']),
        'win_rate': win_rate,
        'trades_detail': trades
    }

# 参数
param_sets = [
    {'name': '保守', 'ma_short': 5, 'ma_long': 30, 'atr_stop': 2.5, 'atr_trail': 2.0, 'max_pos': 0.25},
    {'name': '平衡', 'ma_short': 5, 'ma_long': 20, 'atr_stop': 2.0, 'atr_trail': 1.5, 'max_pos': 0.3},
    {'name': '激进', 'ma_short': 3, 'ma_long': 15, 'atr_stop': 1.5, 'atr_trail': 1.2, 'max_pos': 0.4},
]

print("\n" + "="*70)
print("📊 真实数据多策略回测 (修复版)")
print("="*70)

data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
}

all_results = []

for stock_name, filepath in data_files.items():
    if not Path(filepath).exists():
        continue
    
    df = load_data(filepath)
    price_change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"\n{stock_name}: {len(df)}条, 涨跌{price_change:+.1f}%")
    
    for params in param_sets:
        result = backtest(df, stock_name, params)
        all_results.append(result)
        
        status = "✅" if result['return'] > 0 else "❌"
        print(f"  {status} {params['name']}: 收益{result['return']*100:+.2f}%, 夏普{result['sharpe']:.2f}, 交易{result['trades']}次, 胜率{result['win_rate']*100:.0f}%")

# 汇总
print("\n" + "="*70)
print("📈 汇总")
print("="*70)

avg_return = np.mean([r['return'] for r in all_results])
avg_sharpe = np.mean([r['sharpe'] for r in all_results])
win_count = sum(1 for r in all_results if r['return'] > 0)

print(f"盈利组合: {win_count}/{len(all_results)}")
print(f"平均收益: {avg_return*100:+.2f}%")
print(f"平均夏普: {avg_sharpe:.2f}")

if all_results:
    best = max(all_results, key=lambda x: x['sharpe'])
    print(f"\n最佳: {best['name']} - {best['strategy']}")
    print(f"  收益: {best['return']*100:+.2f}%")
    print(f"  夏普: {best['sharpe']:.2f}")
    print(f"  回撤: {best['max_dd']*100:.1f}%")
