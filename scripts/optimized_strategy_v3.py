#!/usr/bin/env python3
"""
优化策略 v3.0 - 无杠杆年化30%+

改进点：
1. 优化7-Agent决策逻辑
2. 添加择时过滤（避免熊市）
3. 改进止损策略（动态止损）
4. 组合多维度确认
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def calculate_optimized_strategy(df, stock_code, start_idx=200):
    """
    优化策略：
    1. 多维度确认（价值+技术+趋势）
    2. 动态止损（15%→10%逐步收紧）
    3. 趋势过滤（MA50>MA150）
    4. 减少交易频率（每月检查）
    """
    df = df.copy()
    
    # 计算指标
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_150'] = df['close'].rolling(150).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['atr'] = calculate_atr(df, 14)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    
    INITIAL_CAPITAL = 10000
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    highest_price = 0
    days_held = 0
    
    signals = []
    equity = []
    trades_log = []
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        ma_20 = df['ma_20'].iloc[i]
        ma_50 = df['ma_50'].iloc[i]
        ma_150 = df['ma_150'].iloc[i]
        rsi = df['rsi'].iloc[i]
        atr = df['atr'].iloc[i]
        volatility = df['volatility'].iloc[i]
        
        if pd.isna(ma_150) or pd.isna(rsi):
            equity.append({
                'date': df['datetime'].iloc[i],
                'equity': cash + position * price
            })
            continue
        
        # 持仓管理
        if position > 0:
            days_held += 1
            
            # 更新最高价
            if price > highest_price:
                highest_price = price
            
            # 动态止损（基于持有时间）
            if days_held < 30:
                # 前30天：15%止损
                stop_loss_pct = 0.15
            elif days_held < 90:
                # 30-90天：12%止损
                stop_loss_pct = 0.12
            else:
                # 90天后：10%止损
                stop_loss_pct = 0.10
            
            # 移动止盈（从最高点回撤）
            trailing_stop = highest_price * 0.90  # 回撤10%
            
            # 执行止损
            if price < entry_price * (1 - stop_loss_pct):
                pnl_pct = (price - entry_price) / entry_price * 100
                signals.append({
                    'date': df['datetime'].iloc[i],
                    'price': price,
                    'type': 'SELL',
                    'reason': f'止损（{pnl_pct:+.1f}%，持有{days_held}天）'
                })
                cash = position * price
                position = 0
                days_held = 0
                continue
            
            # 移动止盈
            if price < trailing_stop and price > entry_price * 1.10:
                pnl_pct = (price - entry_price) / entry_price * 100
                signals.append({
                    'date': df['datetime'].iloc[i],
                    'price': price,
                    'type': 'SELL',
                    'reason': f'移动止盈（{pnl_pct:+.1f}%，回撤10%）'
                })
                cash = position * price
                position = 0
                days_held = 0
                continue
            
            # 趋势反转卖出
            if ma_50 < ma_150 and price < ma_20:
                pnl_pct = (price - entry_price) / entry_price * 100
                signals.append({
                    'date': df['datetime'].iloc[i],
                    'price': price,
                    'type': 'SELL',
                    'reason': f'趋势反转（{pnl_pct:+.1f}%）'
                })
                cash = position * price
                position = 0
                days_held = 0
        
        # 买入信号：每月检查一次（降低频率）
        if (i - start_idx) % 20 == 0 and position == 0 and cash > 0:
            buy_score = 0
            reasons = []
            
            # 1. 趋势确认（MA50 > MA150）
            if ma_50 > ma_150:
                buy_score += 2
                reasons.append('上升趋势')
            
            # 2. 价格位置（价格 < MA50 * 1.1，不要追太高）
            if price < ma_50 * 1.1:
                buy_score += 1
                reasons.append('价格合理')
            
            # 3. RSI不过热（RSI < 70）
            if rsi < 70:
                buy_score += 1
                reasons.append(f'RSI正常({rsi:.0f})')
            
            # 4. 波动率适中（波动率 < 3%）
            if not pd.isna(volatility) and volatility < 0.03:
                buy_score += 1
                reasons.append('波动适中')
            
            # 需要3分以上才买入
            if buy_score >= 3:
                shares = int(cash * 0.95 / price)
                if shares > 0:
                    position = shares
                    cash -= shares * price
                    entry_price = price
                    highest_price = price
                    days_held = 0
                    
                    signals.append({
                        'date': df['datetime'].iloc[i],
                        'price': price,
                        'type': 'BUY',
                        'reason': f'多维确认（{buy_score}/5分）：{", ".join(reasons)}'
                    })
                    
                    trades_log.append({
                        'date': df['datetime'].iloc[i],
                        'action': 'BUY',
                        'price': price,
                        'score': buy_score,
                        'reasons': reasons
                    })
        
        equity.append({
            'date': df['datetime'].iloc[i],
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
            'reason': f'最终平仓（{pnl_pct:+.1f}%，持有{days_held}天）'
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
    
    # 胜率
    sells = [s for s in signals if s['type'] == 'SELL']
    wins = [s for s in sells if '+' in s['reason']]
    win_rate = len(wins) / len(sells) if sells else 0
    
    return {
        'signals': signals,
        'equity': equity,
        'trades_log': trades_log,
        'stats': {
            'code': stock_code,
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'sharpe': float(sharpe),
            'max_drawdown': float(max_dd),
            'trades': len([s for s in signals if s['type'] == 'BUY']),
            'win_rate': float(win_rate),
            'final_equity': float(final_equity),
        }
    }

def calculate_rsi(prices, period=14):
    """计算RSI"""
    deltas = prices.diff()
    gain = deltas.where(deltas > 0, 0)
    loss = -deltas.where(deltas < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_atr(df, period=14):
    """计算ATR"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr

if __name__ == "__main__":
    print("\n" + "=" * 90)
    print("🚀 优化策略 v3.0 - 无杠杆年化30%+")
    print("=" * 90)
    print("\n改进点：")
    print("  1. ✅ 多维度确认（趋势+价格+RSI+波动率）")
    print("  2. ✅ 动态止损（15%→12%→10%逐步收紧）")
    print("  3. ✅ 移动止盈（回撤10%止盈）")
    print("  4. ✅ 趋势过滤（MA50 > MA150）")
    print("  5. ✅ 降低频率（每月检查一次）")
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
        
        # 优化策略
        result = calculate_optimized_strategy(df, code)
        stats = result['stats']
        
        # 买入持有基准
        start_idx = 200
        bh_return = (df['close'].iloc[-1] / df['close'].iloc[start_idx] - 1)
        days = len(df) - start_idx
        bh_annual = (1 + bh_return) ** (252 / days) - 1
        
        print(f"\n📊 {name} ({code})")
        print("-" * 90)
        print(f"{'策略':<20} {'总收益':<12} {'年化收益':<12} {'夏普':<8} {'最大回撤':<12} {'交易次数':<10}")
        print("-" * 90)
        print(f"{'买入持有':<20} {bh_return*100:>+10.2f}% {bh_annual*100:>+10.2f}% {'N/A':>6} {'N/A':>10} {'1':<10}")
        print(f"{'优化策略v3.0':<20} {stats['total_return']*100:>+10.2f}% {stats['annual_return']*100:>+10.2f}% {stats['sharpe']:>6.2f} {stats['max_drawdown']*100:>10.2f}% {stats['trades']:<10}")
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
            'buy_hold': {
                'total_return': float(bh_return),
                'annual_return': float(bh_annual)
            },
            'optimized': stats,
            'signals': result['signals']
        })
    
    print("\n" + "=" * 90)
    print("📈 总结")
    print("=" * 90)
    
    if results:
        avg_bh = np.mean([r['buy_hold']['annual_return'] for r in results])
        avg_opt = np.mean([r['optimized']['annual_return'] for r in results])
        avg_sharpe = np.mean([r['optimized']['sharpe'] for r in results])
        avg_dd = np.mean([r['optimized']['max_drawdown'] for r in results])
        
        print(f"\n平均表现：")
        print(f"  买入持有年化: {avg_bh*100:+.2f}%")
        print(f"  优化策略年化: {avg_opt*100:+.2f}%")
        print(f"  平均夏普比率: {avg_sharpe:.2f}")
        print(f"  平均最大回撤: {avg_dd*100:.2f}%")
        
        improvement = (avg_opt - avg_bh) * 100
        print(f"\n🎯 相对买入持有: {improvement:+.2f}%")
        
        if avg_opt >= 0.30:
            print(f"✅ 达成年化30%目标！")
        else:
            print(f"⚠️  未达标，年化{avg_opt*100:.2f}%（目标30%）")
        
        # 保存结果
        output_file = Path("data/reports/optimized_strategy_v3.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'strategy': 'Optimized v3.0 - No Leverage',
                'improvements': [
                    'Multi-dimensional confirmation',
                    'Dynamic stop loss (15%→12%→10%)',
                    'Trailing stop (10% drawdown)',
                    'Trend filter (MA50 > MA150)',
                    'Monthly check frequency'
                ],
                'results': results,
                'summary': {
                    'avg_buy_hold': float(avg_bh),
                    'avg_optimized': float(avg_opt),
                    'avg_sharpe': float(avg_sharpe),
                    'avg_max_drawdown': float(avg_dd),
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 结果已保存: {output_file}")
