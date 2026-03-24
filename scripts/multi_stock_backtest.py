#!/usr/bin/env python3
"""
多股票回测 - 使用真实数据
测试多只股票的回测表现
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 技术指标
def sma(data, period): return data.rolling(window=period).mean()
def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def backtest_stock(stock_code, stock_name):
    """回测单只股票"""
    csv_file = f"data/real_{stock_code}.csv"
    
    if not Path(csv_file).exists():
        return None
    
    # 读取数据
    df = pd.read_csv(csv_file)
    df = df.rename(columns={
        '日期': 'datetime', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume'
    })
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 策略参数
    MA_SHORT, MA_LONG = 5, 20
    ATR_STOP, ATR_TRAIL = 2.0, 1.5
    MAX_POS = 0.3
    
    # 计算指标
    df['ma_s'] = sma(df['close'], MA_SHORT)
    df['ma_l'] = sma(df['close'], MA_LONG)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    # 回测
    cash = 100000
    position = 0
    entry_price = 0
    stop_loss = 0
    trailing_stop = 0
    highest = 0
    
    trades = []
    equity = [cash]
    
    for i in range(30, len(df)):
        price = df['close'].iloc[i]
        ma_s = df['ma_s'].iloc[i]
        ma_l = df['ma_l'].iloc[i]
        atr_val = df['atr'].iloc[i]
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            equity.append(cash + position * price)
            continue
        
        # 更新追踪止损
        if position > 0 and price > highest:
            highest = price
            trailing_stop = max(trailing_stop, highest - atr_val * ATR_TRAIL)
        
        # 止损
        if position > 0 and (price <= stop_loss or price <= trailing_stop):
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'stop', 'price': price, 'pnl': pnl})
            cash = position * price
            position = 0
            continue
        
        # 买入信号
        if ma_s > ma_l and position == 0:
            shares = int(cash * MAX_POS / price)
            if shares > 0:
                position = shares
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * ATR_STOP
                trailing_stop = stop_loss
                highest = price
                trades.append({'type': 'buy', 'price': price, 'shares': shares})
        
        # 卖出信号
        elif ma_s < ma_l and position > 0:
            pnl = (price - entry_price) / entry_price
            trades.append({'type': 'signal', 'price': price, 'pnl': pnl})
            cash = position * price
            position = 0
        
        equity.append(cash + position * price)
    
    # 最终平仓
    if position > 0:
        cash = position * df['close'].iloc[-1]
    
    final_equity = cash
    total_return = (final_equity - 100000) / 100000
    
    # 计算夏普
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    # 最大回撤
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    # 胜率
    sells = [t for t in trades if 'pnl' in t]
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    # 基准收益
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[0] - 1)
    
    return {
        'code': stock_code,
        'name': stock_name,
        'data_points': len(df),
        'start_date': df['datetime'].iloc[0].strftime('%Y-%m-%d'),
        'end_date': df['datetime'].iloc[-1].strftime('%Y-%m-%d'),
        'start_price': float(df['close'].iloc[0]),
        'end_price': float(df['close'].iloc[-1]),
        'buy_hold_return': float(buy_hold),
        'final_equity': float(final_equity),
        'total_return': float(total_return),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
        'trades': len([t for t in trades if t['type']=='buy']),
        'win_rate': float(win_rate),
    }

if __name__ == "__main__":
    # 测试股票列表
    stocks = [
        ('300750', '宁德时代'),
        ('002475', '立讯精密'),
        ('601318', '中国平安'),
        ('600519', '贵州茅台'),
        ('000001', '平安银行'),
        ('000858', '五粮液'),
    ]
    
    print("\n" + "=" * 80)
    print("📊 多股票回测结果（简单均线策略）")
    print("=" * 80)
    print(f"{'代码':<8} {'名称':<8} {'数据':<6} {'买入持有':<10} {'策略收益':<10} {'夏普':<6} {'最大回撤':<10} {'评级':<6}")
    print("-" * 80)
    
    results = []
    for code, name in stocks:
        result = backtest_stock(code, name)
        if result:
            results.append(result)
            
            # 评级
            if result['sharpe'] > 1.5 and result['max_drawdown'] > -0.10:
                grade = "A 🏆"
            elif result['sharpe'] > 1.0 and result['max_drawdown'] > -0.15:
                grade = "B ✅"
            elif result['sharpe'] > 0.5:
                grade = "C ⚠️"
            else:
                grade = "D ❌"
            
            print(f"{code:<8} {name:<8} {result['data_points']:<6} "
                  f"{result['buy_hold_return']*100:>+8.2f}% "
                  f"{result['total_return']*100:>+8.2f}% "
                  f"{result['sharpe']:>5.2f} "
                  f"{result['max_drawdown']*100:>8.2f}% "
                  f"{grade:<6}")
    
    print("=" * 80)
    
    # 统计
    if results:
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_dd = np.mean([r['max_drawdown'] for r in results])
        avg_bh = np.mean([r['buy_hold_return'] for r in results])
        
        print(f"\n📈 平均表现:")
        print(f"  买入持有: {avg_bh*100:+.2f}%")
        print(f"  策略收益: {avg_return*100:+.2f}%")
        print(f"  夏普比率: {avg_sharpe:.2f}")
        print(f"  最大回撤: {avg_dd*100:.2f}%")
        
        # 保存结果
        output_file = Path("data/reports/backtest_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'strategy': 'Simple MA Crossover',
                'params': {'ma_short': 5, 'ma_long': 20, 'atr_stop': 2.0, 'atr_trail': 1.5},
                'results': results,
                'summary': {
                    'avg_buy_hold': float(avg_bh),
                    'avg_strategy_return': float(avg_return),
                    'avg_sharpe': float(avg_sharpe),
                    'avg_max_drawdown': float(avg_dd),
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 结果已保存: {output_file}")
