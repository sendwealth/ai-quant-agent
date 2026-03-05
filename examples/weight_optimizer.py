"""
组合权重优化器
==============
目标：找到最优权重配置，提升组合夏普至0.5+
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
from itertools import product

# 导入基础函数
from smart_screener_v2 import backtest_strategy

def optimize_weights(stock_codes, initial_capital=100000, step=0.05):
    """优化组合权重"""
    
    print("\n" + "="*70)
    print("组合权重优化")
    print("="*70)
    
    # 加载数据
    stock_data = {}
    stock_names = {
        '600276': '恒瑞医药',
        '002475': '立讯精密',
        '300750': '宁德时代',
        '601318': '中国平安'
    }
    
    for code in stock_codes:
        filepath = Path(f'data/real_{code}.csv')
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        stock_data[code] = df
    
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
    print("\n单只股票回测...")
    results = {}
    for code, df in stock_data.items():
        result = backtest_strategy(df, params)
        results[code] = result
        print(f"  {stock_names[code]}: 夏普{result['sharpe']:.2f}, 收益{result['return']*100:+.1f}%")
    
    # 权重优化（简化版：网格搜索）
    print(f"\n开始权重优化 (步长{step*100:.0f}%)...")
    
    best_sharpe = 0
    best_weights = None
    best_return = 0
    
    # 生成权重组合
    weight_range = np.arange(0.10, 0.50, step)  # 10%-45%
    
    combinations = 0
    for w1 in weight_range:
        for w2 in weight_range:
            for w3 in weight_range:
                w4 = 1.0 - w1 - w2 - w3
                if 0.10 <= w4 <= 0.50:  # 确保所有权重在10%-50%
                    weights = [w1, w2, w3, w4]
                    
                    # 计算组合指标
                    portfolio_return = sum([results[code]['return'] * weights[i] 
                                          for i, code in enumerate(stock_codes)])
                    portfolio_sharpe = sum([results[code]['sharpe'] * weights[i] 
                                          for i, code in enumerate(stock_codes)])
                    
                    if portfolio_sharpe > best_sharpe:
                        best_sharpe = portfolio_sharpe
                        best_weights = weights
                        best_return = portfolio_return
                    
                    combinations += 1
    
    print(f"  测试组合数: {combinations}")
    
    # 显示最优配置
    print("\n" + "="*70)
    print("最优权重配置")
    print("="*70)
    
    print(f"\n{'股票':<10} {'权重':<10} {'夏普':<8} {'收益':<10} {'贡献度'}")
    print("-" * 70)
    
    for i, code in enumerate(stock_codes):
        name = stock_names[code]
        weight = best_weights[i]
        sharpe = results[code]['sharpe']
        ret = results[code]['return']
        contribution = sharpe * weight
        
        print(f"{name:<10} {weight*100:>6.1f}%   {sharpe:6.2f}  {ret*100:>+7.1f}%  {contribution:.3f}")
    
    print("-" * 70)
    print(f"{'组合整体':<10} {'100.0%':<10} {best_sharpe:6.2f}  {best_return*100:>+7.1f}%")
    
    # 对比等权重
    equal_weights = [0.25, 0.25, 0.25, 0.25]
    equal_sharpe = sum([results[code]['sharpe'] * 0.25 for code in stock_codes])
    equal_return = sum([results[code]['return'] * 0.25 for code in stock_codes])
    
    print("\n" + "="*70)
    print("对比分析")
    print("="*70)
    
    print(f"\n等权重配置 (各25%):")
    print(f"  夏普: {equal_sharpe:.3f}")
    print(f"  收益: {equal_return*100:+.2f}%")
    
    print(f"\n优化权重配置:")
    print(f"  夏普: {best_sharpe:.3f}")
    print(f"  收益: {best_return*100:+.2f}%")
    
    improvement = best_sharpe - equal_sharpe
    print(f"\n提升: {improvement:+.3f} ({improvement/equal_sharpe*100:+.1f}%)")
    
    # 达标检查
    print("\n" + "="*70)
    print("达标检查")
    print("="*70)
    
    if best_sharpe >= 0.5:
        print(f"\n✅ 夏普达到0.5！")
        print(f"   优化后夏普: {best_sharpe:.3f}")
    else:
        gap = 0.5 - best_sharpe
        print(f"\n⚠️ 距离0.5还差 {gap:.3f}")
        print(f"   优化后夏普: {best_sharpe:.3f}")
    
    # 保存结果
    output = {
        'optimize_time': datetime.now().isoformat(),
        'optimal_weights': {
            stock_names[code]: float(w) 
            for code, w in zip(stock_codes, best_weights)
        },
        'portfolio_sharpe': float(best_sharpe),
        'portfolio_return': float(best_return),
        'equal_weight_sharpe': float(equal_sharpe),
        'improvement': float(improvement),
        'stock_results': {
            stock_names[code]: {
                'weight': float(best_weights[i]),
                'sharpe': float(results[code]['sharpe']),
                'return': float(results[code]['return']),
                'win_rate': float(results[code]['win_rate'])
            } for i, code in enumerate(stock_codes)
        }
    }
    
    with open('data/weight_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: data/weight_optimization_results.json")
    
    # 生成配置建议
    print("\n" + "="*70)
    print("配置建议")
    print("="*70)
    
    print(f"\n总资金: 100,000元")
    print(f"\n股票配置:")
    for i, code in enumerate(stock_codes):
        name = stock_names[code]
        weight = best_weights[i]
        amount = 100000 * weight
        print(f"  {name}: {amount:.0f}元 ({weight*100:.1f}%)")
    
    return best_weights, best_sharpe, best_return

if __name__ == '__main__':
    stock_codes = ['600276', '002475', '300750', '601318']
    optimize_weights(stock_codes)
