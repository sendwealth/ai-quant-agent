"""
动态风险管理策略 v16.0
======================
基于市场波动率动态调整：
1. 波动率仓位管理
2. 连续亏损后降仓
3. 盈利后逐步加仓
4. 多时间框架确认
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============ 配置 ============
INITIAL_CAPITAL = 100000
BASE_POSITION = 0.25  # 基础仓位25%

# ============ 工具函数 ============
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

def calculate_volatility(close, period=20):
    """计算年化波动率"""
    return close.pct_change().rolling(period).std() * np.sqrt(252)

def adjust_position_by_volatility(volatility):
    """根据波动率调整仓位"""
    if volatility > 0.40:  # 高波动
        return 0.10
    elif volatility > 0.30:  # 中高波动
        return 0.15
    elif volatility > 0.20:  # 正常波动
        return 0.25
    else:  # 低波动
        return 0.35

def adjust_position_by_streak(consecutive_losses, consecutive_wins):
    """根据连续盈亏调整仓位"""
    if consecutive_losses >= 3:
        return 0.05  # 连续3次亏损，降到5%
    elif consecutive_losses >= 2:
        return 0.10  # 连续2次亏损，降到10%
    elif consecutive_wins >= 3:
        return 0.40  # 连续3次盈利，加到40%
    elif consecutive_wins >= 2:
        return 0.30  # 连续2次盈利，加到30%
    else:
        return BASE_POSITION

# ============ 策略引擎 ============
def dynamic_risk_backtest(df):
    """动态风险管理回测"""
    df = df.copy()
    
    # 计算指标
    df['ma_10'] = sma(df['close'], 10)
    df['ma_30'] = sma(df['close'], 30)
    df['ma_60'] = sma(df['close'], 60)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['volatility'] = calculate_volatility(df['close'])
    
    cash = INITIAL_CAPITAL
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    trades = []
    equity = [cash]
    consecutive_losses = 0
    consecutive_wins = 0
    last_trade_pnl = 0
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_10 = float(df['ma_10'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        ma_60 = float(df['ma_60'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        volatility = float(df['volatility'].iloc[i])
        
        if pd.isna(ma_10) or pd.isna(ma_30) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            # 波动率自适应追踪止损
            trail_mult = 3.0 if volatility < 0.25 else 2.5 if volatility < 0.35 else 2.0
            new_stop = highest - atr_val * trail_mult
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'pnl': pnl, 'vol': volatility})
            cash += shares * price
            
            # 更新连续盈亏
            if pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
            else:
                consecutive_wins += 1
                consecutive_losses = 0
            last_trade_pnl = pnl
            
            shares = 0
            equity.append(cash)
            continue
        
        # 买入信号（多时间框架确认）
        should_buy = (
            shares == 0 and
            ma_10 > ma_30 and  # 短期趋势向上
            ma_30 > ma_60 and  # 长期趋势向上
            rsi_val < 70  # 不超买
        )
        
        if should_buy:
            # 综合仓位调整
            vol_position = adjust_position_by_volatility(volatility)
            streak_position = adjust_position_by_streak(consecutive_losses, consecutive_wins)
            final_position = min(vol_position, streak_position)
            
            new_shares = int(cash * final_position / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                # 波动率自适应止损
                stop_mult = 3.5 if volatility < 0.25 else 3.0 if volatility < 0.35 else 2.5
                stop_loss = price - atr_val * stop_mult
                highest = price
                trades.append({'type': 'buy', 'position': final_position, 'vol': volatility})
        
        # 卖出信号
        elif ma_10 < ma_30 and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'pnl': pnl, 'vol': volatility})
            cash += shares * price
            
            # 更新连续盈亏
            if pnl < 0:
                consecutive_losses += 1
                consecutive_wins = 0
            else:
                consecutive_wins += 1
                consecutive_losses = 0
            last_trade_pnl = pnl
            
            shares = 0
        
        equity.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    # 计算指标
    final = cash
    ret = (final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    sells = [t for t in trades if 'pnl' in t]
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    # Profit Factor
    profits = sum(t['pnl'] for t in sells if t['pnl'] > 0)
    losses = abs(sum(t['pnl'] for t in sells if t['pnl'] < 0))
    profit_factor = profits / losses if losses > 0 else 0
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['type'] == 'buy']),
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }

print("="*70)
print("📊 动态风险管理策略 v16.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n核心特性:")
print("  ✓ 波动率自适应仓位(10%-35%)")
print("  ✓ 连续亏损降仓(最低5%)")
print("  ✓ 连续盈利加仓(最高40%)")
print("  ✓ 多时间框架确认(MA10>MA30>MA60)")

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
    r = dynamic_risk_backtest(df)
    r['stock'] = name
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    r['bh'] = bh
    results.append(r)
    
    status = "✅" if r['return'] > 0 else "❌"
    
    print(f"\n{name} (买入持有: {bh:+.1f}%):")
    print(f"  {status} 策略: {r['return']*100:+.2f}% | 夏普: {r['sharpe']:.2f}")
    print(f"  回撤: {r['max_dd']*100:.1f}% | 交易: {r['trades']}次")
    print(f"  胜率: {r['win_rate']*100:.0f}% | Profit Factor: {r['profit_factor']:.2f}")

# 汇总
print("\n" + "="*70)
print("📈 汇总")
print("="*70)

avg_ret = np.mean([r['return'] for r in results]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results])
avg_dd = np.mean([r['max_dd'] for r in results]) * 100
avg_pf = np.mean([r['profit_factor'] for r in results])
win_count = sum(1 for r in results if r['return'] > 0)

print(f"盈利股票: {win_count}/{len(results)}")
print(f"平均收益: {avg_ret:+.2f}%")
print(f"平均夏普: {avg_sharpe:.2f}")
print(f"平均回撤: {avg_dd:.1f}%")
print(f"平均Profit Factor: {avg_pf:.2f}")

if avg_sharpe > 1.0 and avg_pf > 2.0:
    grade = "A 🏆"
elif avg_sharpe > 0.5 and avg_pf > 1.5:
    grade = "B ✅"
elif avg_sharpe > 0:
    grade = "C ⚠️"
else:
    grade = "D ❌"

print(f"\n评级: {grade}")
print("="*70)
