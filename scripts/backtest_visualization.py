#!/usr/bin/env python3
"""
回测可视化 - 生成交易信号图表
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def calculate_buy_hold_with_signals(df, start_idx=200):
    """买入持有策略 + 信号"""
    signals = []
    
    # 买入信号
    signals.append({
        'date': df['datetime'].iloc[start_idx],
        'price': df['close'].iloc[start_idx],
        'type': 'BUY',
        'reason': '初始建仓'
    })
    
    # 计算权益曲线
    equity = []
    entry_price = df['close'].iloc[start_idx]
    shares = 10000 / entry_price  # 假设投入1万
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        equity.append({
            'date': df['datetime'].iloc[i],
            'equity': shares * price
        })
    
    # 卖出信号（最后）
    signals.append({
        'date': df['datetime'].iloc[-1],
        'price': df['close'].iloc[-1],
        'type': 'SELL',
        'reason': f'平仓（收益{(df["close"].iloc[-1]/entry_price-1)*100:+.1f}%）'
    })
    
    return signals, equity

def calculate_ma_strategy_with_signals(df, start_idx=200):
    """均线策略 + 信号"""
    df = df.copy()
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_20'] = df['close'].rolling(20).mean()
    
    INITIAL_CAPITAL = 10000
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    signals = []
    equity = []
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        ma_5 = df['ma_5'].iloc[i]
        ma_20 = df['ma_20'].iloc[i]
        
        if pd.isna(ma_5) or pd.isna(ma_20):
            equity.append({
                'date': df['datetime'].iloc[i],
                'equity': cash + position * price
            })
            continue
        
        # 买入信号
        if ma_5 > ma_20 and position == 0 and cash > 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                position = shares
                cash -= shares * price
                entry_price = price
                signals.append({
                    'date': df['datetime'].iloc[i],
                    'price': price,
                    'type': 'BUY',
                    'reason': f'MA5上穿MA20'
                })
        
        # 卖出信号
        elif ma_5 < ma_20 and position > 0:
            pnl = (price - entry_price) / entry_price * 100
            signals.append({
                'date': df['datetime'].iloc[i],
                'price': price,
                'type': 'SELL',
                'reason': f'MA5下穿MA20（{pnl:+.1f}%）'
            })
            cash = position * price
            position = 0
        
        equity.append({
            'date': df['datetime'].iloc[i],
            'equity': cash + position * price
        })
    
    # 最终平仓
    if position > 0:
        pnl = (df['close'].iloc[-1] - entry_price) / entry_price * 100
        signals.append({
            'date': df['datetime'].iloc[-1],
            'price': df['close'].iloc[-1],
            'type': 'SELL',
            'reason': f'最终平仓（{pnl:+.1f}%）'
        })
    
    return signals, equity

def calculate_7agent_strategy_with_signals(df, stock_code, start_idx=200):
    """7-Agent策略 + 信号（简化版）"""
    df = df.copy()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    df['rsi'] = calculate_rsi(df['close'], 14)
    
    INITIAL_CAPITAL = 10000
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    signals = []
    equity = []
    
    for i in range(start_idx, len(df)):
        price = df['close'].iloc[i]
        ma_50 = df['ma_50'].iloc[i]
        ma_200 = df['ma_200'].iloc[i]
        rsi = df['rsi'].iloc[i]
        
        if pd.isna(ma_50) or pd.isna(ma_200) or pd.isna(rsi):
            equity.append({
                'date': df['datetime'].iloc[i],
                'equity': cash + position * price
            })
            continue
        
        # 每20天决策一次
        if (i - start_idx) % 20 == 0:
            # 7-Agent决策（简化）
            buy_score = 0
            
            # Buffett: 价值分析
            if price < ma_200 * 1.1:
                buy_score += 1
            
            # Technical: 技术分析
            if ma_50 > ma_200:
                buy_score += 1
            
            # Growth: 成长分析
            if i > start_idx + 60:
                recent_return = df['close'].iloc[i] / df['close'].iloc[i-60] - 1
                if recent_return > 0.1:
                    buy_score += 1
            
            # 买入
            if buy_score >= 2 and position == 0 and cash > 0:
                shares = int(cash * 0.70 / price)
                if shares > 0:
                    position = shares
                    cash -= shares * price
                    entry_price = price
                    signals.append({
                        'date': df['datetime'].iloc[i],
                        'price': price,
                        'type': 'BUY',
                        'reason': f'7-Agent共识（{buy_score}/3）'
                    })
            
            # 卖出
            elif buy_score < 1 and position > 0:
                pnl = (price - entry_price) / entry_price * 100
                signals.append({
                    'date': df['datetime'].iloc[i],
                    'price': price,
                    'type': 'SELL',
                    'reason': f'7-Agent风险（{pnl:+.1f}%）'
                })
                cash = position * price
                position = 0
        
        # 止损
        if position > 0 and price < entry_price * 0.90:
            pnl = (price - entry_price) / entry_price * 100
            signals.append({
                'date': df['datetime'].iloc[i],
                'price': price,
                'type': 'SELL',
                'reason': f'止损（{pnl:+.1f}%）'
            })
            cash = position * price
            position = 0
        
        equity.append({
            'date': df['datetime'].iloc[i],
            'equity': cash + position * price
        })
    
    # 最终平仓
    if position > 0:
        pnl = (df['close'].iloc[-1] - entry_price) / entry_price * 100
        signals.append({
            'date': df['datetime'].iloc[-1],
            'price': df['close'].iloc[-1],
            'type': 'SELL',
            'reason': f'最终平仓（{pnl:+.1f}%）'
        })
    
    return signals, equity

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

def plot_backtest_results(df, strategies_data, stock_code, stock_name, start_idx=200):
    """绘制回测结果"""
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle(f'{stock_name} ({stock_code}) 回测分析', fontsize=16, fontweight='bold')
    
    # 准备数据
    plot_df = df.iloc[start_idx:].copy()
    
    # 子图1: 价格和交易信号
    ax1 = axes[0]
    ax1.plot(plot_df['datetime'], plot_df['close'], label='股价', linewidth=2, color='black')
    
    # 绘制各策略的交易信号
    colors = {'买入持有': 'blue', '简单均线': 'red', '7-Agent': 'green'}
    
    for strategy_name, data in strategies_data.items():
        signals = data['signals']
        color = colors.get(strategy_name, 'gray')
        
        # 买入信号
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        if buy_signals:
            buy_dates = [s['date'] for s in buy_signals]
            buy_prices = [s['price'] for s in buy_signals]
            ax1.scatter(buy_dates, buy_prices, marker='^', s=100, color=color, 
                       label=f'{strategy_name} 买入', alpha=0.7, zorder=5)
        
        # 卖出信号
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        if sell_signals:
            sell_dates = [s['date'] for s in sell_signals]
            sell_prices = [s['price'] for s in sell_signals]
            ax1.scatter(sell_dates, sell_prices, marker='v', s=100, color=color,
                       label=f'{strategy_name} 卖出', alpha=0.7, zorder=5)
    
    ax1.set_ylabel('股价 (¥)', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('股价走势与交易信号', fontsize=13)
    
    # 子图2: 权益曲线
    ax2 = axes[1]
    for strategy_name, data in strategies_data.items():
        equity = data['equity']
        equity_df = pd.DataFrame(equity)
        color = colors.get(strategy_name, 'gray')
        ax2.plot(equity_df['date'], equity_df['equity'], 
                label=strategy_name, linewidth=2, color=color)
    
    ax2.axhline(y=10000, color='gray', linestyle='--', label='初始资金', alpha=0.5)
    ax2.set_ylabel('权益 (¥)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_title('权益曲线对比', fontsize=13)
    
    # 子图3: 收益率对比
    ax3 = axes[2]
    for strategy_name, data in strategies_data.items():
        equity = data['equity']
        equity_df = pd.DataFrame(equity)
        returns = (equity_df['equity'] / 10000 - 1) * 100
        color = colors.get(strategy_name, 'gray')
        ax3.plot(equity_df['date'], returns, 
                label=strategy_name, linewidth=2, color=color)
    
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_ylabel('收益率 (%)', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_title('收益率对比', fontsize=13)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = Path(f'data/reports/backtest_chart_{stock_code}.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 图表已保存: {output_path}")
    
    return output_path

def print_signals_table(strategies_data):
    """打印交易信号表格"""
    print("\n" + "=" * 90)
    print("📊 交易信号详情")
    print("=" * 90)
    
    for strategy_name, data in strategies_data.items():
        signals = data['signals']
        print(f"\n【{strategy_name}】")
        print("-" * 90)
        print(f"{'日期':<12} {'类型':<6} {'价格':<10} {'原因':<40}")
        print("-" * 90)
        
        for signal in signals:
            date_str = signal['date'].strftime('%Y-%m-%d')
            signal_type = '🟢 买入' if signal['type'] == 'BUY' else '🔴 卖出'
            price = f"¥{signal['price']:.2f}"
            reason = signal['reason']
            print(f"{date_str:<12} {signal_type:<6} {price:<10} {reason:<40}")
        
        # 计算统计
        buys = [s for s in signals if s['type'] == 'BUY']
        sells = [s for s in signals if s['type'] == 'SELL']
        print("-" * 90)
        print(f"总交易次数: {len(buys)} 买入, {len(sells)} 卖出")

if __name__ == "__main__":
    # 测试宁德时代
    stock_code = '300750'
    stock_name = '宁德时代'
    
    csv_file = f"data/real_{stock_code}.csv"
    if not Path(csv_file).exists():
        print(f"❌ 数据文件不存在: {csv_file}")
        exit(1)
    
    # 读取数据
    df = pd.read_csv(csv_file)
    df = df.rename(columns={
        '日期': 'datetime', '开盘': 'open', '最高': 'high',
        '最低': 'low', '收盘': 'close', '成交量': 'volume'
    })
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print(f"\n📊 {stock_name} ({stock_code}) 回测分析")
    print("=" * 90)
    print(f"数据期间: {df['datetime'].iloc[0].strftime('%Y-%m-%d')} 至 {df['datetime'].iloc[-1].strftime('%Y-%m-%d')}")
    print(f"数据天数: {len(df)} 天")
    
    # 计算各策略
    start_idx = 200
    
    print("\n🔄 计算各策略...")
    
    # 1. 买入持有
    bh_signals, bh_equity = calculate_buy_hold_with_signals(df, start_idx)
    print(f"  ✅ 买入持有: {len(bh_signals)} 个信号")
    
    # 2. 简单均线
    ma_signals, ma_equity = calculate_ma_strategy_with_signals(df, start_idx)
    print(f"  ✅ 简单均线: {len(ma_signals)} 个信号")
    
    # 3. 7-Agent
    agent_signals, agent_equity = calculate_7agent_strategy_with_signals(df, stock_code, start_idx)
    print(f"  ✅ 7-Agent: {len(agent_signals)} 个信号")
    
    # 汇总数据
    strategies_data = {
        '买入持有': {'signals': bh_signals, 'equity': bh_equity},
        '简单均线': {'signals': ma_signals, 'equity': ma_equity},
        '7-Agent': {'signals': agent_signals, 'equity': agent_equity},
    }
    
    # 计算收益
    print("\n📈 收益对比:")
    print("-" * 90)
    for strategy_name, data in strategies_data.items():
        final_equity = data['equity'][-1]['equity']
        total_return = (final_equity / 10000 - 1) * 100
        print(f"  {strategy_name:<12} 最终权益: ¥{final_equity:,.2f}  总收益: {total_return:+.2f}%")
    
    # 打印交易信号
    print_signals_table(strategies_data)
    
    # 生成图表
    print("\n🎨 生成可视化图表...")
    chart_path = plot_backtest_results(df, strategies_data, stock_code, stock_name, start_idx)
    
    print("\n" + "=" * 90)
    print("✅ 回测分析完成！")
    print("=" * 90)
    print(f"\n📁 生成文件:")
    print(f"  - 图表: {chart_path}")
    print(f"\n💡 提示:")
    print(f"  - 图表已保存，可直接查看")
    print(f"  - 红色三角 = 买入信号")
    print(f"  - 红色倒三角 = 卖出信号")
    print(f"  - 权益曲线显示资金变化")
