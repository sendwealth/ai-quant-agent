"""
完整策略验证报告生成器
基于已有真实数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

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

def backtest(df, params):
    """回测引擎"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], params['ma_short'])
    df['ma_l'] = sma(df['close'], params['ma_long'])
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trailing_stop = 0.0
    highest = 0.0
    
    trades = []
    equity_curve = [cash]
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
        
        # 追踪止损
        if shares > 0 and price > highest:
            highest = price
            trailing_stop = max(trailing_stop, highest - atr_val * params['atr_trail'])
        
        # 止损
        if shares > 0 and (price <= stop_loss or price <= trailing_stop):
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'stop', 'pnl': pnl})
            cash += shares * price
            shares = 0
            cooldown = 3
            equity_curve.append(cash)
            continue
        
        # 止盈
        if shares > 0 and price >= take_profit:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'profit', 'pnl': pnl})
            cash += shares * price
            shares = 0
            cooldown = 2
            equity_curve.append(cash)
            continue
        
        # 买入
        if ma_s > ma_l and rsi_val < 70 and shares == 0:
            new_shares = int(cash * params['max_pos'] / price)
            if new_shares > 0:
                cost = new_shares * price
                if cost <= cash:
                    shares = new_shares
                    cash -= cost
                    entry_price = price
                    stop_loss = price - atr_val * params['atr_stop']
                    take_profit = price + atr_val * params['atr_stop'] * 2
                    trailing_stop = stop_loss
                    highest = price
                    trades.append({'action': 'buy', 'price': price, 'shares': shares})
        
        # 卖出
        elif ma_s < ma_l and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'signal', 'pnl': pnl})
            cash += shares * price
            shares = 0
            cooldown = 3
        
        equity_curve.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    # 计算
    final = cash
    ret = (final - 100000) / 100000
    
    eq = pd.Series(equity_curve)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    sells = [t for t in trades if t['action'] == 'sell']
    wins = [t for t in sells if t.get('pnl', 0) > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    return {
        'final': final,
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['action'] == 'buy']),
        'win_rate': win_rate
    }

# 最优参数
BEST_PARAMS = {
    'ma_short': 5,
    'ma_long': 30,
    'atr_stop': 2.5,
    'atr_trail': 2.0,
    'max_pos': 0.25
}

print("="*70)
print("📊 AI量化策略完整验证报告")
print("="*70)
print(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n策略参数:")
print(f"  短期均线: {BEST_PARAMS['ma_short']}")
print(f"  长期均线: {BEST_PARAMS['ma_long']}")
print(f"  ATR止损: {BEST_PARAMS['atr_stop']}x")
print(f"  追踪止损: {BEST_PARAMS['atr_trail']}x")
print(f"  最大仓位: {BEST_PARAMS['max_pos']*100:.0f}%")

# 加载所有数据
data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
}

all_results = []

print("\n" + "="*70)
print("📈 回测结果")
print("="*70)

for stock_name, filepath in data_files.items():
    if not Path(filepath).exists():
        continue
    
    df = load_data(filepath)
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    
    result = backtest(df, BEST_PARAMS)
    result['stock'] = stock_name
    result['buy_hold'] = buy_hold
    all_results.append(result)
    
    # 超额收益
    excess = result['return'] * 100 - buy_hold
    
    status = "✅" if result['return'] > 0 else "❌"
    beat = "📈" if excess > 0 else "📉"
    
    print(f"\n{stock_name}:")
    print(f"  买入持有: {buy_hold:+.1f}%")
    print(f"  策略收益: {result['return']*100:+.2f}%")
    print(f"  超额收益: {excess:+.1f}% {beat}")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  交易次数: {result['trades']}")
    print(f"  胜率: {result['win_rate']*100:.0f}%")

# 汇总
if all_results:
    print("\n" + "="*70)
    print("📊 汇总统计")
    print("="*70)
    
    avg_return = np.mean([r['return'] for r in all_results]) * 100
    avg_sharpe = np.mean([r['sharpe'] for r in all_results])
    avg_dd = np.mean([r['max_dd'] for r in all_results]) * 100
    win_count = sum(1 for r in all_results if r['return'] > 0)
    
    # 对比买入持有
    avg_bh = np.mean([r['buy_hold'] for r in all_results])
    
    print(f"测试股票: {len(all_results)}只")
    print(f"盈利股票: {win_count}/{len(all_results)}")
    print(f"\n策略平均收益: {avg_return:+.2f}%")
    print(f"买入持有收益: {avg_bh:+.1f}%")
    print(f"超额收益: {avg_return - avg_bh:+.1f}%")
    print(f"\n平均夏普: {avg_sharpe:.2f}")
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
    
    print(f"\n策略评级: {grade}")

# 保存报告
report = {
    'timestamp': datetime.now().isoformat(),
    'params': BEST_PARAMS,
    'results': all_results,
    'summary': {
        'avg_return': avg_return if all_results else 0,
        'avg_sharpe': avg_sharpe if all_results else 0,
        'win_count': win_count if all_results else 0,
        'total_stocks': len(all_results)
    }
}

with open('strategy_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2, default=str)

print("\n报告已保存: strategy_report.json")
print("="*70)
