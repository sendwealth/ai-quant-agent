"""
组合策略验证
============
目标：验证4只股票组合的整体表现
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 导入基础函数
from smart_screener_v2 import sma, ema, atr, rsi, macd, backtest_strategy

def backtest_portfolio(stock_codes, weights, initial_capital=100000):
    """组合回测"""
    
    # 加载数据
    stock_data = {}
    for code in stock_codes:
        filepath = Path(f'data/real_{code}.csv')
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        stock_data[code] = df
    
    if len(stock_data) != len(stock_codes):
        print(f"⚠️ 只能加载 {len(stock_data)}/{len(stock_codes)} 只股票")
        return None
    
    # 策略参数
    params = {
        'ma_fast': 10, 'ma_slow': 30,
        'atr_stop_mult': 2.5, 'atr_trail_mult': 2.0,
        'use_dynamic_position': True,
        'use_macd': True, 'use_rsi': True,
        'take_profit_1': 0.10, 'take_profit_2': 0.20,
        'partial_exit_1': 0.5, 'partial_exit_2': 0.5,
        'base_position': 0.30
    }
    
    # 单只股票回测
    results = {}
    for code, df in stock_data.items():
        result = backtest_strategy(df, params)
        results[code] = result
    
    # 组合回测（等权重）
    portfolio_return = sum([results[code]['return'] * weights[i] for i, code in enumerate(stock_codes)])
    
    # 计算组合夏普（简化版）
    portfolio_sharpe = sum([results[code]['sharpe'] * weights[i] for i, code in enumerate(stock_codes)])
    
    # 计算组合最大回撤（简化版）
    portfolio_max_dd = min([results[code]['max_dd'] for code in stock_codes])
    
    # 平均胜率
    avg_win_rate = np.mean([results[code]['win_rate'] for code in stock_codes])
    
    # 总交易次数
    total_trades = sum([results[code]['trades'] for code in stock_codes])
    
    return {
        'return': portfolio_return,
        'sharpe': portfolio_sharpe,
        'max_dd': portfolio_max_dd,
        'win_rate': avg_win_rate,
        'total_trades': total_trades,
        'stocks': results
    }

def main():
    """主函数"""
    print("="*70)
    print("组合策略验证")
    print("="*70)
    
    # 推荐股票
    recommended = [
        ('600276', '恒瑞医药'),
        ('002475', '立讯精密'),
        ('300750', '宁德时代'),
        ('601318', '中国平安')
    ]
    
    print("\n【组合配置】")
    print(f"\n股票池:")
    for i, (code, name) in enumerate(recommended, 1):
        print(f"  {i}. {name} ({code}): 25%仓位")
    
    # 等权重配置
    codes = [code for code, _ in recommended]
    weights = [0.25, 0.25, 0.25, 0.25]
    
    print("\n开始回测...")
    
    # 组合回测
    result = backtest_portfolio(codes, weights)
    
    if not result:
        print("❌ 回测失败")
        return
    
    # 显示结果
    print("\n" + "="*70)
    print("组合回测结果")
    print("="*70)
    
    print(f"\n{'股票':<10} {'收益':<10} {'夏普':<8} {'胜率':<8} {'交易次数':<10}")
    print("-" * 70)
    
    for code in codes:
        r = result['stocks'][code]
        name = [n for c, n in recommended if c == code][0]
        print(f"{name:<10} {r['return']*100:+8.1f}%  {r['sharpe']:6.2f}  {r['win_rate']*100:6.1f}%  {r['trades']:6d}次")
    
    print("-" * 70)
    print(f"{'组合整体':<10} {result['return']*100:+8.1f}%  {result['sharpe']:6.2f}  {result['win_rate']*100:6.1f}%  {result['total_trades']:6d}次")
    
    # 达标检查
    print("\n" + "="*70)
    print("达标检查")
    print("="*70)
    
    checks = [
        ('夏普≥0.3', result['sharpe'] >= 0.3, result['sharpe']),
        ('夏普≥0.5', result['sharpe'] >= 0.5, result['sharpe']),
        ('胜率≥60%', result['win_rate'] >= 0.60, result['win_rate']*100),
        ('收益≥5%', result['return'] >= 0.05, result['return']*100),
        ('最大回撤≤15%', result['max_dd'] >= -0.15, result['max_dd']*100)
    ]
    
    for criterion, passed, value in checks:
        status = "✅" if passed else "❌"
        print(f"  {status} {criterion}: {value:.2f}")
    
    # 综合评价
    passed_count = sum([1 for _, p, _ in checks if p])
    
    print("\n" + "="*70)
    print("综合评价")
    print("="*70)
    
    if passed_count >= 4:
        print("\n✅ 组合表现优秀！")
        print("\n推荐行动:")
        print("  1. 可以开始模拟盘测试")
        print("  2. 资金: 100,000元")
        print("  3. 配置: 4只股票各25%")
        print("  4. 测试期: 3个月")
        print("  5. 如果稳定达标，考虑小资金实盘")
    elif passed_count >= 3:
        print("\n⚠️ 组合表现良好，但仍需改进")
        print("\n建议:")
        print("  1. 先模拟盘测试")
        print("  2. 仔细观察每只股票表现")
        print("  3. 可能需要调整权重或替换股票")
    else:
        print("\n❌ 组合未达标，需要继续优化")
        print("\n建议:")
        print("  1. 分析失败原因")
        print("  2. 调整策略参数")
        print("  3. 更换股票")
    
    # 保存结果
    output = {
        'test_time': datetime.now().isoformat(),
        'portfolio': {
            'return': float(result['return']),
            'sharpe': float(result['sharpe']),
            'max_dd': float(result['max_dd']),
            'win_rate': float(result['win_rate']),
            'total_trades': int(result['total_trades'])
        },
        'stocks': {code: {
            'return': float(r['return']),
            'sharpe': float(r['sharpe']),
            'max_dd': float(r['max_dd']),
            'win_rate': float(r['win_rate']),
            'trades': int(r['trades'])
        } for code, r in result['stocks'].items()},
        'passed_checks': passed_count,
        'total_checks': len(checks)
    }
    
    with open('data/portfolio_backtest_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: data/portfolio_backtest_results.json")
    
    # 生成交易计划
    print("\n" + "="*70)
    print("交易计划")
    print("="*70)
    
    print(f"\n【资金配置】")
    print(f"  总资金: 100,000元")
    print(f"  单只股票: 25,000元 (25%)")
    print(f"  初始仓位: 30% (7,500元)")
    print(f"  最大仓位: 50% (12,500元)")
    
    print(f"\n【股票列表】")
    for i, (code, name) in enumerate(recommended, 1):
        r = result['stocks'][code]
        print(f"  {i}. {name} ({code})")
        print(f"     - 建议买入价: 参考MA10/30金叉")
        print(f"     - 止损位: 成本价-8%")
        print(f"     - 止盈1: +10%卖50%")
        print(f"     - 止盈2: +20%清仓")
        print(f"     - 历史夏普: {r['sharpe']:.2f}")
        print(f"     - 历史收益: {r['return']*100:+.1f}%")
    
    print(f"\n【预期表现】")
    print(f"  组合夏普: {result['sharpe']:.2f}")
    print(f"  组合收益: {result['return']*100:+.1f}%")
    print(f"  平均胜率: {result['win_rate']*100:.1f}%")
    print(f"  最大回撤: {result['max_dd']*100:.1f}%")
    
    print(f"\n【执行步骤】")
    print(f"  1. 开通模拟账户 (同花顺/东方财富)")
    print(f"  2. 添加4只股票到自选")
    print(f"  3. 设置MA10/30、MACD、RSI指标")
    print(f"  4. 等待买入信号 (MA金叉)")
    print(f"  5. 分批建仓 (每次30%)")
    print(f"  6. 严格执行止损止盈")
    print(f"  7. 每日记录和监控")
    print(f"  8. 3个月后评估")

if __name__ == '__main__':
    main()
