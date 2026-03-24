#!/usr/bin/env python3
"""
7-Agent策略回测 - 使用真实数据和7-Agent决策
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

def simulate_7agent_decision(stock_code, stock_name, price_data):
    """
    模拟7-Agent决策
    基于真实数据和简单的决策逻辑
    """
    # 1. Buffett Analyst - 价值分析
    def buffett_analysis(price_series):
        # 简化：基于价格相对位置
        current_price = price_series.iloc[-1]
        ma_50 = price_series.rolling(50).mean().iloc[-1]
        ma_200 = price_series.rolling(200).mean().iloc[-1]
        
        # 价格低于长期均线 = 价值投资机会
        if current_price < ma_200 * 0.9:
            return "BUY", 0.80, "价格低于200日均线10%"
        elif current_price < ma_50:
            return "BUY", 0.70, "价格低于50日均线"
        else:
            return "HOLD", 0.60, "价格合理"
    
    # 2. Technical Analyst - 技术分析
    def technical_analysis(price_series):
        returns = price_series.pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1]
        
        # 低波动率 = 稳定
        if volatility < 0.02:
            return "BUY", 0.75, "波动率低，走势稳定"
        elif volatility > 0.05:
            return "SELL", 0.65, "波动率高，风险大"
        else:
            return "HOLD", 0.60, "波动正常"
    
    # 3. Growth Analyst - 成长分析
    def growth_analysis(price_series):
        # 简化：基于近期涨跌幅
        recent_return = (price_series.iloc[-1] / price_series.iloc[-60] - 1)
        
        if recent_return > 0.3:
            return "BUY", 0.85, "近期涨幅30%+，强势"
        elif recent_return > 0.1:
            return "BUY", 0.70, "近期涨幅10%+，良好"
        elif recent_return < -0.2:
            return "SELL", 0.60, "近期下跌20%+，弱势"
        else:
            return "HOLD", 0.55, "近期走势平稳"
    
    # 4. Fundamentals Analyst - 基本面（简化）
    def fundamentals_analysis(stock_code):
        # 简化：基于股票代码判断（实际应获取财务数据）
        quality_stocks = ['300750', '002475', '600519', '000858']
        
        if stock_code in quality_stocks:
            return "BUY", 0.80, "优质蓝筹股"
        else:
            return "HOLD", 0.60, "一般股票"
    
    # 5. Sentiment Analyst - 情绪（简化）
    def sentiment_analysis(price_series):
        # 简化：基于成交量变化
        volume_trend = 1  # 假设成交量正常
        return "HOLD", 0.60, "情绪中性"
    
    # 6. Risk Manager - 风险控制
    def risk_check(price_series, decisions):
        # 检查风险敞口
        buy_count = sum(1 for d in decisions if d[0] == "BUY")
        sell_count = sum(1 for d in decisions if d[0] == "SELL")
        
        if buy_count >= 3:
            return "BUY", 0.75, f"多维度确认（{buy_count}/5）"
        elif sell_count >= 3:
            return "SELL", 0.70, f"多维度风险（{sell_count}/5）"
        else:
            return "HOLD", 0.60, "信号不明确"
    
    # 执行所有分析
    decisions = [
        buffett_analysis(price_data['close']),
        technical_analysis(price_data['close']),
        growth_analysis(price_data['close']),
        fundamentals_analysis(stock_code),
        sentiment_analysis(price_data['close']),
    ]
    
    # Risk Manager
    final_decision = risk_check(price_data['close'], decisions)
    
    return final_decision

def backtest_7agent_strategy(stock_code, stock_name):
    """使用7-Agent策略回测"""
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
    
    # 回测参数
    INITIAL_CAPITAL = 100000
    MAX_POSITION = 0.30  # 30%仓位
    STOP_LOSS = 0.05     # 5%止损
    TAKE_PROFIT = 0.15   # 15%止盈
    
    # 回测
    cash = INITIAL_CAPITAL
    position = 0
    entry_price = 0
    trades = []
    equity = [cash]
    
    for i in range(200, len(df)):  # 从第200天开始（需要足够数据）
        current_data = df.iloc[:i+1]
        price = df['close'].iloc[i]
        
        # 每5天决策一次（降低交易频率）
        if i % 5 == 0:
            decision, confidence, reason = simulate_7agent_decision(
                stock_code, stock_name, current_data
            )
            
            # 持仓管理
            if position > 0:
                # 止损
                if price <= entry_price * (1 - STOP_LOSS):
                    pnl = (price - entry_price) / entry_price
                    trades.append({'type': 'stop_loss', 'price': price, 'pnl': pnl})
                    cash = position * price * 0.999  # 扣除手续费
                    position = 0
                    continue
                
                # 止盈
                if price >= entry_price * (1 + TAKE_PROFIT):
                    pnl = (price - entry_price) / entry_price
                    trades.append({'type': 'take_profit', 'price': price, 'pnl': pnl})
                    cash = position * price * 0.999
                    position = 0
                    continue
                
                # 卖出信号
                if decision == "SELL" and confidence > 0.65:
                    pnl = (price - entry_price) / entry_price
                    trades.append({'type': 'signal', 'price': price, 'pnl': pnl})
                    cash = position * price * 0.999
                    position = 0
            
            # 买入信号
            elif decision == "BUY" and confidence > 0.70 and cash > 0:
                shares = int(cash * MAX_POSITION / price)
                if shares > 0:
                    position = shares
                    cash -= shares * price * 1.001  # 加上手续费
                    entry_price = price
                    trades.append({'type': 'buy', 'price': price, 'shares': shares})
        
        equity.append(cash + position * price)
    
    # 最终平仓
    if position > 0:
        cash = position * df['close'].iloc[-1] * 0.999
    
    final_equity = cash
    total_return = (final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    # 计算指标
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    sells = [t for t in trades if 'pnl' in t]
    wins = [t for t in sells if t['pnl'] > 0]
    win_rate = len(wins) / len(sells) if sells else 0
    
    # 基准
    buy_hold = (df['close'].iloc[-1] / df['close'].iloc[200] - 1)
    
    return {
        'code': stock_code,
        'name': stock_name,
        'data_points': len(df),
        'start_date': df['datetime'].iloc[200].strftime('%Y-%m-%d'),
        'end_date': df['datetime'].iloc[-1].strftime('%Y-%m-%d'),
        'start_price': float(df['close'].iloc[200]),
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
    stocks = [
        ('300750', '宁德时代'),
        ('002475', '立讯精密'),
        ('601318', '中国平安'),
        ('600519', '贵州茅台'),
        ('000001', '平安银行'),
        ('000858', '五粮液'),
    ]
    
    print("\n" + "=" * 80)
    print("📊 7-Agent策略回测结果")
    print("=" * 80)
    print(f"{'代码':<8} {'名称':<8} {'买入持有':<10} {'7-Agent收益':<12} {'夏普':<6} {'最大回撤':<10} {'评级':<6}")
    print("-" * 80)
    
    results = []
    for code, name in stocks:
        result = backtest_7agent_strategy(code, name)
        if result:
            results.append(result)
            
            # 评级
            if result['sharpe'] > 2.0 and result['max_drawdown'] > -0.10:
                grade = "A 🏆"
            elif result['sharpe'] > 1.5 and result['max_drawdown'] > -0.15:
                grade = "B ✅"
            elif result['sharpe'] > 1.0:
                grade = "C ⚠️"
            else:
                grade = "D ❌"
            
            print(f"{code:<8} {name:<8} {result['buy_hold_return']*100:>+8.2f}% "
                  f"{result['total_return']*100:>+10.2f}% "
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
        print(f"  7-Agent策略: {avg_return*100:+.2f}%")
        print(f"  夏普比率: {avg_sharpe:.2f}")
        print(f"  最大回撤: {avg_dd*100:.2f}%")
        
        # 对比
        improvement = (avg_return - avg_bh) * 100
        print(f"\n🎯 策略改进:")
        print(f"  相对买入持有: {improvement:+.2f}%")
        
        # 保存
        output_file = Path("data/reports/backtest_7agent_results.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'strategy': '7-Agent Multi-Dimensional Decision',
                'params': {
                    'max_position': 0.30,
                    'stop_loss': 0.05,
                    'take_profit': 0.15,
                    'decision_frequency': '5 days'
                },
                'results': results,
                'summary': {
                    'avg_buy_hold': float(avg_bh),
                    'avg_strategy_return': float(avg_return),
                    'avg_sharpe': float(avg_sharpe),
                    'avg_max_drawdown': float(avg_dd),
                }
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 结果已保存: {output_file}")
