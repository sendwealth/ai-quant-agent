"""
自动化运行系统 v1.0
==================
功能：
- 定时获取数据
- 自动运行策略
- 生成报告
- 发送通知
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time

print("="*70)
print("🤖 自动化运行系统 v1.0")
print("="*70)
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============ 配置 ============
CONFIG = {
    'initial_capital': 100000,
    'max_position': 0.25,
    'stop_loss_atr': 2.5,
    'trail_stop_atr': 2.0,
    'ma_short': 5,
    'ma_long': 30,
}

# ============ 工具函数 ============
def sma(data, period):
    return data.rolling(window=period).mean()

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

# ============ 策略引擎 ============
def run_strategy(df, config):
    """运行保守策略"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], config['ma_short'])
    df['ma_l'] = sma(df['close'], config['ma_long'])
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = config['initial_capital']
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    trades = []
    equity = [cash]
    
    for i in range(50, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * config['trail_stop_atr']
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        # 止损
        if shares > 0 and price <= stop_loss:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'stop', 'pnl': pnl})
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        # 买入
        if ma_s > ma_l and shares == 0:
            new_shares = int(cash * config['max_position'] / price)
            if new_shares > 0:
                shares = new_shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * config['stop_loss_atr']
                highest = price
                trades.append({'action': 'buy', 'price': price})
        
        # 卖出
        elif ma_s < ma_l and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'signal', 'pnl': pnl})
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    # 计算指标
    final = cash
    ret = (final - config['initial_capital']) / config['initial_capital']
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    return {
        'final': final,
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['action'] == 'buy'])
    }

# ============ 主运行流程 ============
print("【步骤1】加载数据")
data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
    '茅台': 'data/real_600519.csv',
}

all_data = {}
for name, path in data_files.items():
    df = load_data(path)
    if df is not None:
        all_data[name] = df
        print(f"  ✓ {name}: {len(df)}天数据")

print(f"\n【步骤2】运行策略")
results = {}
for name, df in all_data.items():
    r = run_strategy(df, CONFIG)
    results[name] = r
    bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    status = "✅" if r['return'] > 0 else "❌"
    print(f"  {status} {name}: {r['return']*100:+.2f}% (买入持有: {bh:+.1f}%)")

print(f"\n【步骤3】生成报告")
avg_ret = np.mean([r['return'] for r in results.values()]) * 100
avg_sharpe = np.mean([r['sharpe'] for r in results.values()])
avg_dd = np.mean([r['max_dd'] for r in results.values()]) * 100

print(f"  平均收益: {avg_ret:+.2f}%")
print(f"  平均夏普: {avg_sharpe:.2f}")
print(f"  平均回撤: {avg_dd:.1f}%")

# 保存报告
report = {
    'timestamp': datetime.now().isoformat(),
    'config': CONFIG,
    'results': {k: {
        'return': v['return'],
        'sharpe': v['sharpe'],
        'max_dd': v['max_dd']
    } for k, v in results.items()},
    'summary': {
        'avg_return': avg_ret,
        'avg_sharpe': avg_sharpe,
        'avg_dd': avg_dd
    }
}

with open('auto_run_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\n【步骤4】保存报告")
print(f"  ✓ auto_run_report.json")

print(f"\n【步骤5】推荐操作")
if avg_ret > 0:
    print(f"  ✅ 策略整体盈利，建议继续运行")
else:
    print(f"  ⚠️ 策略整体亏损，建议优化参数")

if avg_dd < -10:
    print(f"  ⚠️ 回撤较大({avg_dd:.1f}%)，建议降低仓位")
else:
    print(f"  ✅ 风险控制良好")

print("\n" + "="*70)
print("🤖 自动化运行完成")
print("="*70)
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
