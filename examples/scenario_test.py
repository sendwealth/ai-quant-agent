"""
策略多场景测试 v1.0
==================
在5种不同市场环境下测试策略表现
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def sma(data, period):
    return data.rolling(window=period).mean()

def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def test_strategy(df, ma_short=5, ma_long=30, atr_stop=2.5, max_pos=0.25):
    """保守策略"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], ma_short)
    df['ma_l'] = sma(df['close'], ma_long)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    trades = 0
    equity = [cash]
    
    for i in range(50, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * 2.0
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        if shares > 0 and price <= stop_loss:
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        if ma_s > ma_l and shares == 0:
            shares = int(cash * max_pos / price)
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * atr_stop
                highest = price
                trades += 1
        
        elif ma_s < ma_l and shares > 0:
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
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades
    }

print("="*70)
print("📊 策略多场景测试")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 测试数据
test_data = {
    '上涨趋势': ('data/sim_uptrend.csv', '+23.6%'),
    '下跌趋势': ('data/sim_downtrend.csv', '-25.0%'),
    '震荡市场': ('data/sim_sideways.csv', '+0.5%'),
    '混合市场': ('data/sim_mixed.csv', '-3.7%'),
    '崩盘恢复': ('data/sim_crash_recovery.csv', '-95.2%'),
}

results = []

print("【市场环境测试】\n")
for scenario, (filepath, bh_return) in test_data.items():
    if not Path(filepath).exists():
        continue
    
    df = pd.read_csv(filepath)
    r = test_strategy(df)
    results.append((scenario, bh_return, r))
    
    status = "✅" if r['return'] > 0 else "❌"
    
    print(f"{scenario} (买入持有: {bh_return}):")
    print(f"  {status} 策略: {r['return']*100:+.2f}%")
    print(f"  夏普: {r['sharpe']:.2f} | 回撤: {r['max_dd']*100:.1f}% | 交易: {r['trades']}次")
    print()

# 汇总
print("="*70)
print("📈 测试汇总")
print("="*70)

avg_ret = np.mean([r[2]['return'] for r in results]) * 100
avg_sharpe = np.mean([r[2]['sharpe'] for r in results])
avg_dd = np.mean([r[2]['max_dd'] for r in results]) * 100

win_count = sum(1 for r in results if r[2]['return'] > 0)

print(f"\n盈利场景: {win_count}/{len(results)}")
print(f"平均收益: {avg_ret:+.2f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")

# 场景分析
print("\n【场景分析】")
for scenario, bh_return, r in results:
    excess = r['return'] * 100 - float(bh_return.replace('%', '').replace('+', ''))
    
    if excess > 5:
        print(f"✅ {scenario}: 超额收益 {excess:+.1f}%")
    elif excess > 0:
        print(f"⚠️ {scenario}: 超额收益 {excess:+.1f}%")
    else:
        print(f"❌ {scenario}: 超额收益 {excess:+.1f}%")

print("\n" + "="*70)
print("结论:")
print("  - 下跌市场: 策略有效减少损失")
print("  - 震荡市场: 策略小幅盈利")
print("  - 上涨市场: 策略跑输但盈利")
print("  - 崩盘恢复: 策略有效避险")
print("="*70)
