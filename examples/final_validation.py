"""
最终完整验证 - 3只股票
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def sma(data, period): return data.rolling(window=period).mean()
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

def backtest(df, params):
    """保守策略回测"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], params['ma_short'])
    df['ma_l'] = sma(df['close'], params['ma_long'])
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    highest = 0.0
    
    trades = []
    equity = [cash]
    cooldown = 0
    
    for i in range(50, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        if cooldown > 0:
            cooldown -= 1
            equity.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * params['atr_trail']
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
        
        # 买入
        if ma_s > ma_l and shares == 0:
            new_shares = int(cash * params['max_pos'] / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * params['atr_stop']
                highest = price
                trades.append({'type': 'buy', 'price': price})
        
        # 卖出
        elif ma_s < ma_l and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl})
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
    
    sells = [t for t in trades if t['type'] != 'buy']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    return {
        'final': final, 'return': ret, 'sharpe': sharpe,
        'max_dd': max_dd, 'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate
    }

# 最优参数
PARAMS = {'ma_short': 5, 'ma_long': 30, 'atr_stop': 2.5, 'atr_trail': 2.0, 'max_pos': 0.25}

print("="*70)
print("📊 最终验证 - 3只股票")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    
    r = backtest(df, PARAMS)
    r['stock'] = name
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    excess = r['return'] * 100 - bh
    
    print(f"\n{name} ({len(df)}天, 买入持有: {bh:+.1f}%):")
    print(f"  {status} 策略: {r['return']*100:+.2f}% | 超额: {excess:+.1f}% | "
          f"夏普: {r['sharpe']:.2f} | 回撤: {r['max_dd']*100:.1f}% | "
          f"交易: {r['trades']}次 | 胜率: {r['win_rate']*100:.0f}%")

# 汇总
print("\n" + "="*70)
print("📈 汇总统计")
print("="*70)

avg_ret = np.mean([r['return'] for r in results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['max_dd'] for r in results]) * 100
avg_bh = np.mean([r['bh'] for r in results])
win_count = sum(1 for r in results if r['return'] > 0)

print(f"测试股票: {len(results)}只")
print(f"盈利股票: {win_count}/{len(results)}")
print(f"\n策略平均收益: {avg_ret:+.2f}%")
print(f"买入持有平均: {avg_bh:+.1f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")

# 评级
if avg_sharpe > 1.0 and avg_dd > -10:
    grade = "A 🏆 优秀"
elif avg_sharpe > 0.5 and avg_dd > -15:
    grade = "B ✅ 良好"
elif avg_sharpe > 0:
    grade = "C ⚠️ 一般"
else:
    grade = "D ❌ 需优化"

print(f"\n最终评级: {grade}")
print("="*70)
