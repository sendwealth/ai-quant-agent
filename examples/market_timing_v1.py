"""
市场择时策略 v1.0
================
核心：只在有利市场环境交易
- 大盘趋势向上才交易
- 个股必须强势
- 下跌时100%空仓
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def sma(data, period):
    return data.rolling(window=period).mean()

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def load_data(filepath):
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def market_timing_backtest(df):
    """市场择时回测"""
    df = df.copy()
    
    # 计算指标
    df['ma_10'] = sma(df['close'], 10)
    df['ma_30'] = sma(df['close'], 30)
    df['ma_60'] = sma(df['close'], 60)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    trades = []
    equity = [cash]
    in_market_days = 0
    total_days = 0
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_10 = float(df['ma_10'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        ma_60 = float(df['ma_60'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        volume = float(df['volume'].iloc[i])
        vol_ma = float(df['vol_ma'].iloc[i])
        
        total_days += 1
        
        if pd.isna(ma_10) or pd.isna(ma_60) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and high > highest:
            highest = high
            new_stop = highest - atr_val * 2.5
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and low <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 市场环境判断
        market_ok = (
            price > ma_60 and  # 长期趋势向上
            ma_10 > ma_30 and  # 短期趋势向上
            rsi_val < 70       # 不超买
        )
        
        # 如果市场环境恶化，清仓
        if not market_ok and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'market_exit', 'pnl': pnl})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 买入信号（仅在市场环境好时）
        should_buy = (
            shares == 0 and
            market_ok and
            ma_10 > ma_30 > ma_60 and  # 多头排列
            volume > vol_ma * 1.2      # 成交量放大
        )
        
        if should_buy:
            new_shares = int(cash * 0.35 / price)  # 35%仓位
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * 3.0
                highest = high
                trades.append({'type': 'buy', 'price': price})
        
        # 卖出信号
        elif ma_10 < ma_30 and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl})
            cash += shares * price
            shares = 0
        
        if shares > 0:
            in_market_days += 1
        
        equity.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    # 计算指标
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
    
    market_time_ratio = in_market_days / total_days if total_days > 0 else 0
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate,
        'market_time': market_time_ratio
    }

print("="*70)
print("🎯 市场择时策略 v1.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n策略核心:")
print("  ✓ 只在趋势向上时交易 (价格>MA60)")
print("  ✓ 多头排列确认 (MA10>MA30>MA60)")
print("  ✓ 成交量放大 (>20%)")
print("  ✓ 市场恶化立即空仓")

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
    r = market_timing_backtest(df)
    r['stock'] = name
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    
    print(f"\n{name} (买入持有: {bh:+.1f}%):")
    print(f"  {status} 策略: {r['return']*100:+.2f}% | 夏普: {r['sharpe']:.2f}")
    print(f"  回撤: {r['max_dd']*100:.1f}% | 交易: {r['trades']}次 | 胜率: {r['win_rate']*100:.0f}%")
    print(f"  市场时间: {r['market_time']*100:.1f}% (空仓{(1-r['market_time'])*100:.1f}%)")

# 汇总
print("\n" + "="*70)
print("📈 汇总")
print("="*70)

avg_ret = np.mean([r['return'] for r in results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['max_dd'] for r in results]) * 100
avg_market_time = np.mean([r['market_time'] for r in results]) * 100
win_count = sum(1 for r in results if r['return'] > 0)

print(f"盈利股票: {win_count}/{len(results)}")
print(f"平均收益: {avg_ret:+.2f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")
print(f"平均市场时间: {avg_market_time:.1f}%")

print("\n💡 关键指标:")
print(f"  • 风险控制: 空仓时间 {(100-avg_market_time):.1f}%")
print(f"  • 收益效率: 每市场时间收益 {avg_ret/avg_market_time:.2f}%")

if avg_sharpe > 1.0:
    print("\n评级: A 🏆 优秀")
elif avg_sharpe > 0.5:
    print("\n评级: B ✅ 良好")
elif avg_sharpe > 0:
    print("\n评级: C ⚠️ 一般")
else:
    print("\n评级: D ❌ 需优化")

print("="*70)
