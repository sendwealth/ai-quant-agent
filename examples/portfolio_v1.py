"""
多策略组合系统 v1.0
==================
核心：资金分散到多个策略
- 50% 保守策略
- 30% 激进策略  
- 20% 现金储备
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

def conservative_strategy(df, capital):
    """保守策略：25%仓位，2.5x ATR止损"""
    df = df.copy()
    df['ma_5'] = sma(df['close'], 5)
    df['ma_30'] = sma(df['close'], 30)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = capital
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    equity = [cash]
    
    for i in range(50, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_5 = float(df['ma_5'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        
        if pd.isna(ma_5) or pd.isna(ma_30):
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
        
        if ma_5 > ma_30 and shares == 0:
            new_shares = int(cash * 0.25 / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * 2.5
                highest = price
        
        elif ma_5 < ma_30 and shares > 0:
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    return cash, pd.Series(equity)

def aggressive_strategy(df, capital):
    """激进策略：40%仓位，3.5x ATR止损"""
    df = df.copy()
    df['ma_10'] = sma(df['close'], 10)
    df['ma_30'] = sma(df['close'], 30)
    df['ma_60'] = sma(df['close'], 60)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    cash = capital
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    equity = [cash]
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_10 = float(df['ma_10'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        ma_60 = float(df['ma_60'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        
        if pd.isna(ma_10) or pd.isna(ma_60):
            equity.append(cash + shares * price)
            continue
        
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * 3.0
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        if shares > 0 and price <= stop_loss:
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 激进买入：多头排列 + 不超买
        should_buy = (
            shares == 0 and
            ma_10 > ma_30 > ma_60 and
            rsi_val < 70
        )
        
        if should_buy:
            new_shares = int(cash * 0.40 / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * 3.5
                highest = price
        
        elif ma_10 < ma_30 and shares > 0:
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    return cash, pd.Series(equity)

def portfolio_backtest(df, total_capital=100000):
    """组合回测"""
    # 资金分配
    conservative_capital = total_capital * 0.50  # 50%
    aggressive_capital = total_capital * 0.30   # 30%
    cash_reserve = total_capital * 0.20         # 20%
    
    # 运行两个策略
    conservative_final, conservative_eq = conservative_strategy(df, conservative_capital)
    aggressive_final, aggressive_eq = aggressive_strategy(df, aggressive_capital)
    
    # 合并权益曲线（对齐长度）
    min_len = min(len(conservative_eq), len(aggressive_eq))
    conservative_eq = conservative_eq.iloc[:min_len]
    aggressive_eq = aggressive_eq.iloc[:min_len]
    
    # 组合权益 = 保守 + 激进 + 现金储备
    portfolio_eq = conservative_eq.values + aggressive_eq.values + cash_reserve
    
    # 最终资金
    final_capital = conservative_final + aggressive_final + cash_reserve
    
    # 计算指标
    ret = (final_capital - total_capital) / total_capital
    
    eq = pd.Series(portfolio_eq)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    # 单策略收益
    conservative_ret = (conservative_final - conservative_capital) / conservative_capital
    aggressive_ret = (aggressive_final - aggressive_capital) / aggressive_capital
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'final': final_capital,
        'conservative_ret': conservative_ret,
        'aggressive_ret': aggressive_ret,
        'cash_reserve': cash_reserve
    }

print("="*70)
print("📊 多策略组合系统 v1.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n资金分配:")
print("  50% (5万) → 保守策略 (25%仓位, 2.5x止损)")
print("  30% (3万) → 激进策略 (40%仓位, 3.5x止损)")
print("  20% (2万) → 现金储备")

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
    r = portfolio_backtest(df, 100000)
    r['stock'] = name
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    
    print(f"\n{name} (买入持有: {bh:+.1f}%):")
    print(f"  {status} 组合: {r['return']*100:+.2f}% | 夏普: {r['sharpe']:.2f}")
    print(f"  回撤: {r['max_dd']*100:.1f}%")
    print(f"  分策略: 保守{r['conservative_ret']*100:+.2f}% + 激进{r['aggressive_ret']*100:+.2f}% + 现金")

# 汇总
print("\n" + "="*70)
print("📈 汇总")
print("="*70)

avg_ret = np.mean([r['return'] for r in results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['max_dd'] for r in results]) * 100
avg_cons = np.mean([r['conservative_ret'] for r in results]) * 100
avg_agg = np.mean([r['aggressive_ret'] for r in results]) * 100
win_count = sum(1 for r in results if r['return'] > 0)

print(f"盈利股票: {win_count}/{len(results)}")
print(f"平均收益: {avg_ret:+.2f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")
print(f"\n分策略平均:")
print(f"  保守: {avg_cons:+.2f}%")
print(f"  激进: {avg_agg:+.2f}%")
print(f"  现金: 0% (储备)")

print("\n💡 组合效果:")
print(f"  • 分散风险：单一策略失败不影响全局")
print(f"  • 平滑收益：激进+保守对冲波动")
print(f"  • 保留子弹：20%现金应对机会")

if avg_sharpe > 1.0:
    print("\n评级: A 🏆")
elif avg_sharpe > 0.5:
    print("\n评级: B ✅")
else:
    print("\n评级: C ⚠️")

print("="*70)
