"""
自适应策略 v6.0
===============
根据市场环境动态调整:
- 上涨市场: 放宽止损，加大仓位
- 下跌市场: 收紧止损，减少仓位
- 震荡市场: 均值回归策略
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

def detect_market(df, i):
    """识别市场环境"""
    close = df['close'].iloc[:i+1]
    
    if len(close) < 60:
        return 'unknown'
    
    ma_20 = sma(close, 20).iloc[-1]
    ma_60 = sma(close, 60).iloc[-1]
    price = close.iloc[-1]
    
    # 20日收益率
    ret_20 = (price / close.iloc[-20] - 1) if len(close) > 20 else 0
    
    # 波动率
    vol = close.pct_change().tail(20).std() * np.sqrt(252)
    
    # 判断
    if price > ma_20 > ma_60 and ret_20 > 0.05:
        return 'strong_bull'  # 强势上涨
    elif price > ma_20 and ret_20 > 0:
        return 'weak_bull'    # 弱势上涨
    elif price < ma_20 < ma_60 and ret_20 < -0.05:
        return 'strong_bear'  # 强势下跌
    elif price < ma_20 and ret_20 < 0:
        return 'weak_bear'    # 弱势下跌
    else:
        return 'sideways'     # 震荡

def get_params(market):
    """根据市场环境返回参数"""
    params = {
        'strong_bull': {'atr_stop': 3.0, 'atr_trail': 2.0, 'max_pos': 0.4, 'use_trail': True},
        'weak_bull': {'atr_stop': 2.5, 'atr_trail': 1.5, 'max_pos': 0.3, 'use_trail': True},
        'sideways': {'atr_stop': 2.0, 'atr_trail': 1.5, 'max_pos': 0.2, 'use_trail': True},
        'weak_bear': {'atr_stop': 2.0, 'atr_trail': 1.2, 'max_pos': 0.15, 'use_trail': True},
        'strong_bear': {'atr_stop': 1.5, 'atr_trail': 1.0, 'max_pos': 0.1, 'use_trail': True},
        'unknown': {'atr_stop': 2.5, 'atr_trail': 1.5, 'max_pos': 0.2, 'use_trail': True},
    }
    return params.get(market, params['unknown'])

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def adaptive_backtest(df):
    """自适应回测"""
    df = df.copy()
    df['ma_s'] = sma(df['close'], 5)
    df['ma_l'] = sma(df['close'], 20)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    
    cash = 100000.0
    shares = 0
    entry_price = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    trailing_stop = 0.0
    highest = 0.0
    current_params = get_params('unknown')
    
    trades = []
    equity_curve = [cash]
    cooldown = 0
    market_history = []
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_s = float(df['ma_s'].iloc[i])
        ma_l = float(df['ma_l'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity_curve.append(cash + shares * price)
            continue
        
        # 检测市场环境
        market = detect_market(df, i)
        market_history.append(market)
        
        if cooldown > 0:
            cooldown -= 1
            equity_curve.append(cash + shares * price)
            continue
        
        # 追踪止损
        if shares > 0 and current_params['use_trail'] and price > highest:
            highest = price
            trailing_stop = max(trailing_stop, highest - atr_val * current_params['atr_trail'])
        
        # 止损
        if shares > 0 and (price <= stop_loss or price <= trailing_stop):
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'stop', 'pnl': pnl, 'market': market})
            cash += shares * price
            shares = 0
            cooldown = 3
            equity_curve.append(cash)
            continue
        
        # 止盈
        if shares > 0 and price >= take_profit:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'profit', 'pnl': pnl, 'market': market})
            cash += shares * price
            shares = 0
            cooldown = 2
            equity_curve.append(cash)
            continue
        
        # 买入信号 - 根据市场环境调整
        should_buy = False
        
        if market in ['strong_bull', 'weak_bull']:
            # 上涨市场：均线金叉即可
            should_buy = ma_s > ma_l and rsi_val < 75
        elif market == 'sideways':
            # 震荡市场：超卖反弹
            should_buy = rsi_val < 35 and price < df['close'].rolling(20).mean().iloc[i]
        elif market in ['weak_bear', 'strong_bear']:
            # 下跌市场：严重超卖才买
            should_buy = rsi_val < 25
        
        if should_buy and shares == 0:
            current_params = get_params(market)
            new_shares = int(cash * current_params['max_pos'] / price)
            if new_shares > 0:
                cost = new_shares * price
                if cost <= cash:
                    shares = new_shares
                    cash -= cost
                    entry_price = price
                    stop_loss = price - atr_val * current_params['atr_stop']
                    take_profit = price + atr_val * current_params['atr_stop'] * 2
                    trailing_stop = stop_loss
                    highest = price
                    trades.append({'action': 'buy', 'price': price, 'shares': shares, 'market': market})
        
        # 卖出信号
        should_sell = False
        if market in ['strong_bull'] and ma_s < ma_l:
            should_sell = True
        elif market in ['weak_bull', 'sideways'] and (ma_s < ma_l or rsi_val > 70):
            should_sell = True
        elif market in ['weak_bear', 'strong_bear'] and rsi_val > 50:
            should_sell = True
        
        if should_sell and shares > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'action': 'sell', 'type': 'signal', 'pnl': pnl, 'market': market})
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
    
    # 市场分布
    market_dist = {}
    for m in market_history:
        market_dist[m] = market_dist.get(m, 0) + 1
    
    return {
        'final': final,
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': len([t for t in trades if t['action'] == 'buy']),
        'win_rate': win_rate,
        'market_dist': market_dist,
        'trades_detail': trades
    }

print("="*70)
print("📊 自适应策略 v6.0 验证")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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
    
    result = adaptive_backtest(df)
    result['stock'] = stock_name
    result['buy_hold'] = buy_hold
    all_results.append(result)
    
    excess = result['return'] * 100 - buy_hold
    status = "✅" if result['return'] > 0 else "❌"
    beat = "📈" if excess > 0 else "📉"
    
    print(f"\n{stock_name} (买入持有: {buy_hold:+.1f}%):")
    print(f"  策略收益: {result['return']*100:+.2f}%")
    print(f"  超额收益: {excess:+.1f}% {beat}")
    print(f"  夏普比率: {result['sharpe']:.2f}")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    print(f"  交易次数: {result['trades']}")
    print(f"  胜率: {result['win_rate']*100:.0f}%")
    print(f"  市场分布: {result['market_dist']}")

# 汇总
if all_results:
    print("\n" + "="*70)
    print("📊 汇总")
    print("="*70)
    
    avg_return = np.mean([r['return'] for r in all_results]) * 100
    avg_sharpe = np.mean([r['sharpe'] for r in all_results])
    avg_dd = np.mean([r['max_dd'] for r in all_results]) * 100
    avg_bh = np.mean([r['buy_hold'] for r in all_results])
    win_count = sum(1 for r in all_results if r['return'] > 0)
    
    print(f"盈利股票: {win_count}/{len(all_results)}")
    print(f"\n策略平均收益: {avg_return:+.2f}%")
    print(f"买入持有收益: {avg_bh:+.1f}%")
    print(f"超额收益: {avg_return - avg_bh:+.1f}%")
    print(f"\n平均夏普: {avg_sharpe:.2f}")
    print(f"平均回撤: {avg_dd:.1f}%")
    
    if avg_sharpe > 1.0 and avg_dd > -10:
        grade = "A 🏆"
    elif avg_sharpe > 0.5 and avg_dd > -15:
        grade = "B ✅"
    elif avg_sharpe > 0:
        grade = "C ⚠️"
    else:
        grade = "D ❌"
    
    print(f"\n策略评级: {grade}")
    print("="*70)
