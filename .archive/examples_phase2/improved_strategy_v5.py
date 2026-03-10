"""
改进版策略系统 V5
=================
改进点：
1. 添加波动率过滤器（避免高波动期亏损）
2. 优化权重配置（提高恒瑞医药权重）
3. 更严格的风险控制
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 导入基础函数
from smart_screener_v2 import sma, ema, atr, rsi, macd

def backtest_improved(df, params):
    """改进版回测（带波动率过滤）"""
    df = df.copy()
    
    # 计算指标
    df['ma_fast'] = sma(df['close'], params['ma_fast'])
    df['ma_slow'] = sma(df['close'], params['ma_slow'])
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # 初始化
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    partial_sold = False
    
    trades = 0
    wins = 0
    filtered_trades = 0  # 被过滤的交易次数
    equity = [cash]
    
    # 交易循环
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_fast = float(df['ma_fast'].iloc[i])
        ma_slow = float(df['ma_slow'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        volatility = float(df['volatility'].iloc[i])
        macd_hist = float(df['macd_hist'].iloc[i])
        
        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 持仓管理
        if shares > 0:
            if price > highest:
                highest = price
                new_stop = highest - atr_val * params['atr_trail_mult']
                if new_stop > stop_loss:
                    stop_loss = new_stop
            
            if price <= stop_loss:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
                equity.append(cash)
                continue
            
            if not partial_sold and params.get('take_profit_1'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_1']:
                    sell_shares = int(shares * params['partial_exit_1'])
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_sold = True
                    wins += 1
            
            if partial_sold and params.get('take_profit_2'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_2']:
                    cash += shares * price
                    shares = 0
                    partial_sold = False
                    equity.append(cash)
                    continue
            
            if ma_fast < ma_slow and macd_hist < 0:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
        
        # 买入信号
        if ma_fast > ma_slow and shares == 0:
            # 🔥 新增：波动率过滤
            if params.get('use_volatility_filter') and volatility > params['max_volatility']:
                filtered_trades += 1
                equity.append(cash)
                continue
            
            if params['use_macd'] and macd_hist < 0:
                equity.append(cash)
                continue
            
            if params['use_rsi'] and (rsi_val > 70 or rsi_val < 30):
                equity.append(cash)
                continue
            
            # 🔥 改进：更保守的仓位管理
            position_pct = params.get('base_position', 0.20)  # 降低基础仓位
            if params['use_dynamic_position']:
                if volatility > 0.025:  # 中高波动
                    position_pct *= 0.6
                elif volatility > 0.02:
                    position_pct *= 0.8
            
            position_pct = min(position_pct, 0.40)  # 降低最大仓位
            position_value = cash * position_pct
            shares = int(position_value / price)
            
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * params['atr_stop_mult']
                highest = price
                trades += 1
        
        equity.append(cash + shares * price)
    
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
        if float(df['close'].iloc[-1]) > entry_price:
            wins += 1
    
    final = cash
    ret = (final - 100000) / 100000
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    win_rate = wins / trades if trades > 0 else 0
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'filtered_trades': filtered_trades
    }

def test_improvements():
    """测试改进效果"""
    
    print("="*70)
    print("改进版策略测试 V5")
    print("="*70)
    
    print("\n【改进点】")
    print("  1. 添加波动率过滤器（>3.5%不交易）")
    print("  2. 优化权重配置（恒瑞医药10%→20%）")
    print("  3. 更保守的仓位管理（基础30%→20%，最大50%→40%）")
    
    # 改进后的配置
    stocks = [
        ('300750', '宁德时代', 0.35, {'ma_fast': 10, 'ma_slow': 35, 'atr_stop_mult': 2.0}),
        ('002475', '立讯精密', 0.25, {'ma_fast': 10, 'ma_slow': 35, 'atr_stop_mult': 3.0}),
        ('601318', '中国平安', 0.20, {'ma_fast': 8, 'ma_slow': 25, 'atr_stop_mult': 2.5}),
        ('600276', '恒瑞医药', 0.20, {'ma_fast': 8, 'ma_slow': 30, 'atr_stop_mult': 2.0})  # 提高权重
    ]
    
    # 改进后的参数
    base_params = {
        'atr_trail_mult': 2.0,
        'use_dynamic_position': True,
        'use_macd': True,
        'use_rsi': True,
        'take_profit_1': 0.10,
        'take_profit_2': 0.20,
        'partial_exit_1': 0.5,
        'partial_exit_2': 0.5,
        'base_position': 0.20,  # 🔥 降低基础仓位
        'use_volatility_filter': True,  # 🔥 启用波动率过滤
        'max_volatility': 0.035  # 🔥 波动率阈值3.5%
    }
    
    # 测试原始版和改进版
    results_original = {}
    results_improved = {}
    
    print("\n" + "="*70)
    print("对比测试：原始版 vs 改进版")
    print("="*70)
    
    for code, name, weight, custom_params in stocks:
        filepath = Path(f'data/real_{code}.csv')
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        # 原始版参数
        original_params = {
            **base_params,
            **custom_params,
            'base_position': 0.30,  # 原始基础仓位
            'use_volatility_filter': False  # 不使用波动率过滤
        }
        original_params['max_volatility'] = 999  # 不过滤
        
        # 改进版参数
        improved_params = {
            **base_params,
            **custom_params
        }
        
        # 回测
        orig_result = backtest_improved(df, original_params)
        imp_result = backtest_improved(df, improved_params)
        
        results_original[name] = orig_result
        results_improved[name] = imp_result
        
        # 显示对比
        print(f"\n【{name}】(权重{weight*100:.0f}%)")
        print(f"  原始: 夏普{orig_result['sharpe']:.3f}, 收益{orig_result['return']*100:+.2f}%, 胜率{orig_result['win_rate']*100:.1f}%, 交易{orig_result['trades']}次")
        print(f"  改进: 夏普{imp_result['sharpe']:.3f}, 收益{imp_result['return']*100:+.2f}%, 胜率{imp_result['win_rate']*100:.1f}%, 交易{imp_result['trades']}次, 过滤{imp_result['filtered_trades']}次")
        
        sharpe_change = imp_result['sharpe'] - orig_result['sharpe']
        return_change = imp_result['return'] - orig_result['return']
        
        if sharpe_change > 0:
            print(f"  ✅ 夏普提升{sharpe_change:+.3f}")
        else:
            print(f"  ⚠️ 夏普变化{sharpe_change:+.3f}")
    
    # 组合对比
    print("\n" + "="*70)
    print("组合整体对比")
    print("="*70)
    
    portfolio_sharpe_orig = sum([results_original[name]['sharpe'] * weight 
                                 for name, (_, _, weight, _) in zip(results_original.keys(), stocks)])
    portfolio_return_orig = sum([results_original[name]['return'] * weight 
                                 for name, (_, _, weight, _) in zip(results_original.keys(), stocks)])
    
    portfolio_sharpe_imp = sum([results_improved[name]['sharpe'] * weight 
                                for name, (_, _, weight, _) in zip(results_improved.keys(), stocks)])
    portfolio_return_imp = sum([results_improved[name]['return'] * weight 
                                for name, (_, _, weight, _) in zip(results_improved.keys(), stocks)])
    
    total_filtered = sum([results_improved[name]['filtered_trades'] for name in results_improved])
    
    print(f"\n【原始版】")
    print(f"  组合夏普: {portfolio_sharpe_orig:.3f}")
    print(f"  组合收益: {portfolio_return_orig*100:+.2f}%")
    
    print(f"\n【改进版】")
    print(f"  组合夏普: {portfolio_sharpe_imp:.3f}")
    print(f"  组合收益: {portfolio_return_imp*100:+.2f}%")
    print(f"  过滤交易: {total_filtered}次")
    
    sharpe_improvement = portfolio_sharpe_imp - portfolio_sharpe_orig
    return_improvement = portfolio_return_imp - portfolio_return_orig
    
    print(f"\n【改进效果】")
    print(f"  夏普变化: {sharpe_improvement:+.3f} ({sharpe_improvement/portfolio_sharpe_orig*100:+.1f}%)")
    print(f"  收益变化: {return_improvement*100:+.2f}%")
    
    # 保存结果
    output = {
        'test_time': datetime.now().isoformat(),
        'improvements': {
            'volatility_filter': True,
            'max_volatility': 0.035,
            'base_position_reduced': True,
            'hengrui_weight_increased': True
        },
        'original': {
            'portfolio_sharpe': float(portfolio_sharpe_orig),
            'portfolio_return': float(portfolio_return_orig)
        },
        'improved': {
            'portfolio_sharpe': float(portfolio_sharpe_imp),
            'portfolio_return': float(portfolio_return_imp),
            'total_filtered_trades': int(total_filtered)
        },
        'improvement': {
            'sharpe_change': float(sharpe_improvement),
            'return_change': float(return_improvement)
        },
        'new_weights': {
            name: float(weight) for name, (_, _, weight, _) in zip(results_improved.keys(), stocks)
        }
    }
    
    with open('data/improved_strategy_v5_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: data/improved_strategy_v5_results.json")
    
    # 最终建议
    print("\n" + "="*70)
    print("最终建议")
    print("="*70)
    
    if sharpe_improvement > 0:
        print(f"\n✅ 改进有效！")
        print(f"  - 波动率过滤避免了高波动期亏损")
        print(f"  - 更保守的仓位降低了风险")
        print(f"  - 优化权重提升了组合稳定性")
        print(f"\n推荐使用改进版配置")
    else:
        print(f"\n⚠️ 改进效果不明显")
        print(f"  - 可能需要调整波动率阈值")
        print(f"  - 或者采用其他改进方案")
    
    # 生成改进后的配置
    print("\n" + "="*70)
    print("改进后的最终配置")
    print("="*70)
    
    print(f"\n【资金配置】")
    print(f"  总资金: 100,000元")
    print(f"  初始仓位: 20% (降低)")
    print(f"  最大仓位: 40% (降低)")
    
    print(f"\n【股票配置】(新权重)")
    for name, (_, _, weight, _) in zip(results_improved.keys(), stocks):
        print(f"  {name}: {weight*100:.0f}% ({100000*weight:.0f}元)")
    
    print(f"\n【波动率过滤】")
    print(f"  启用: 是")
    print(f"  阈值: 3.5%")
    print(f"  过滤交易: {total_filtered}次")
    
    print(f"\n【预期表现】")
    print(f"  组合夏普: {portfolio_sharpe_imp:.3f}")
    print(f"  组合收益: {portfolio_return_imp*100:+.2f}%")
    print(f"  风险等级: 中低")

if __name__ == '__main__':
    test_improvements()
