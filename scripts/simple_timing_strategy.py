#!/usr/bin/env python3
"""
简单择时策略 - 无杠杆年化30%+

核心思路：
1. 买入持有为主
2. 只在大趋势转变时调整
3. 年度再平衡
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def calculate_simple_timing_strategy(df, stock_code, start_idx=200):
    """
    简单择时策略：
    1. 买入持有
    2. 年度再平衡（每年检查一次）
    3. 大趋势转变时退出（MA50 < MA200）
    """
    df = df.copy()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    
    INITIAL_CAPITAL = 10000
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    
    signals = []
    equity = []
    
    last_check_year = None
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        ma_50 = df['ma_50'].iloc[i]
        ma_200 = df['ma_200'].iloc[i]
        current_date = df['datetime'].iloc[i]
        
        if pd.isna(ma_200):
            equity.append({
                'date': current_date,
                'equity': cash + position * price
            })
            continue
        
        # 持仓管理
        if position > 0:
            # 大趋势转变：MA50 < MA200
            if ma_50 < ma_200:
                pnl_pct = (price - entry_price) / entry_price * 100
                signals.append({
                    'date': current_date,
                    'price': price,
                    'type': 'SELL',
                    'reason': f'大趋势转变（{pnl_pct:+.1f}%）'
                })
                cash = position * price
                position = 0
        
        # 买入信号：每年检查一次
        current_year = current_date.year
        if current_year != last_check_year and position == 0 and cash > 0:
            # 上升趋势：MA50 > MA200
            if ma_50 > ma_200:
                shares = int(cash * 0.95 / price)
                if shares > 0:
                    position = shares
                    cash -= shares * price
                    entry_price = price
                    
                    signals.append({
                        'date': current_date,
                        'price': price,
                        'type': 'BUY',
                        'reason': f'年度建仓（MA50>MA200）'
                    })
                    
                    last_check_year = current_year
        
        equity.append({
            'date': current_date,
            'equity': cash + position * price
        })
    
    # 最终平仓
    if position > 0:
        final_price = df['close'].iloc[-1]
        pnl_pct = (final_price - entry_price) / entry_price * 100
        
        signals.append({
            'date': df['datetime'].iloc[-1],
            'price': final_price,
            'type': 'SELL',
            'reason': f'最终平仓（{pnl_pct:+.1f}%）'
        })
        
        cash = position * final_price
    
    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # 年化收益
    days = len(df) - start_idx
    annual_return = (1 + total_return) ** (252 / days) - 1
    
    # 夏普比率
    eq = pd.Series([e['equity'] for e in equity])
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    # 最大回撤
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    return {
        'signals': signals,
        'equity': equity,
        'stats': {
            'code': stock_code,
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'trades': len([s for s in signals if s['type'] == 'BUY']),
            'final_equity': float(final_equity),
        }
    }

def calculate_buy_hold(df, start_idx=200):
    """买入持有"""
    start_price = df['close'].iloc[start_idx]
    end_price = df['close'].iloc[-1]
    
    total_return = (end_price / start_price - 1)
    days = len(df) - start_idx
    annual_return = (1 + total_return) ** (252 / days) - 1
    
    return {
        'total_return': float(total_return),
        'annual_return': float(annual_return)
    }

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("💡 简单择时策略 - 无杠杆")
    print("=" * 90)
    print("\n核心思路：")
    print("  1. ✅ 买入持有为主")
    print("  2. ✅ 年度检查一次")
    print("  3. ✅ 大趋势转变时退出（MA50 < MA200）")
    print("\n目标：避免大跌，保住收益")
    print("\n" + "=" * 90)
    
    stocks = [
        ('300750', '宁德时代'),
        ('002475', '立讯精密'),
        ('601318', '中国平安'),
    ]
    
    results = []
    
    for code, name in stocks:
        csv_file = f"data/real_{code}.csv"
        if not Path(csv_file).exists():
            continue
        
        df = pd.read_csv(csv_file)
        df = df.rename(columns={
            '日期': 'datetime', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close', '成交量': 'volume'
        })
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 简单择时
        result = calculate_simple_timing_strategy(df, code)
        stats = result['stats']
        
        # 买入持有
        bh = calculate_buy_hold(df)
        
        print(f"\n📊 {name} ({code})")
        print("-" * 90)
        print(f"{'策略':<20} {'总收益':<12} {'年化收益':<12} {'夏普':<8} {'最大回撤':<12} {'交易次数':<10}")
        print("-" * 90)
        print(f"{'买入持有':<20} {bh['total_return']*100:>+10.2f}% {bh['annual_return']*100:>+10.2f}% {'N/A':>6} {'N/A':>10} {'1':<10}")
        print(f"{'简单择时':<20} {stats['total_return']*100:>+10.2f}% {stats['annual_return']*100:>+10.2f}% {stats['sharpe']:>6.2f} {stats['max_drawdown']*100:>10.2f}% {stats['trades']:<10}")
        print("-" * 90)
        
        # 交易详情
        print(f"\n💡 交易信号：")
        for signal in result['signals']:
            date_str = signal['date'].strftime('%Y-%m-%d')
            signal_type = '🟢 买入' if signal['type'] == 'BUY' else '🔴 卖出'
            print(f"  {date_str}  {signal_type}  ¥{signal['price']:.2f}  {signal['reason']}")
        
        results.append({
            'code': code,
            'name': name,
            'buy_hold': bh,
            'simple_timing': stats
        })
    
    print("\n" + "=" * 90)
    print("📈 总结")
    print("=" * 90)
    
    if results:
        avg_bh = np.mean([r['buy_hold']['annual_return'] for r in results])
        avg_timing = np.mean([r['simple_timing']['annual_return'] for r in results])
        avg_sharpe = np.mean([r['simple_timing']['sharpe'] for r in results])
        avg_dd = np.mean([r['simple_timing']['max_drawdown'] for r in results])
        
        print(f"\n平均表现：")
        print(f"  买入持有年化: {avg_bh*100:+.2f}%")
        print(f"  简单择时年化: {avg_timing*100:+.2f}%")
        print(f"  平均夏普比率: {avg_sharpe:.2f}")
        print(f"  平均最大回撤: {avg_dd*100:.2f}%")
        
        improvement = (avg_timing - avg_bh) * 100
        print(f"\n🎯 相对买入持有: {improvement:+.2f}%")
        
        if avg_timing >= 0.30:
            print(f"✅ 达成年化30%目标！")
        elif avg_timing > avg_bh:
            print(f"✅ 优于买入持有，年化{avg_timing*100:.2f}%")
        else:
            print(f"⚠️  不如买入持有")
        
        print(f"\n💡 结论：")
        print(f"  - 买入持有: 年化{avg_bh*100:.2f}%")
        print(f"  - 简单择时: 年化{avg_timing*100:.2f}%")
        print(f"  - 建议: {'买入持有' if avg_bh > avg_timing else '简单择时'}")
