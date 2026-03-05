"""
参数微调优化器
=============
目标：为每只股票找到最优参数，进一步提升夏普
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 导入基础函数
from smart_screener_v2 import sma, ema, atr, rsi, macd

def backtest_with_params(df, params):
    """使用指定参数回测"""
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
            if params['use_macd'] and macd_hist < 0:
                equity.append(cash)
                continue
            
            if params['use_rsi'] and (rsi_val > 70 or rsi_val < 30):
                equity.append(cash)
                continue
            
            position_pct = params.get('base_position', 0.30)
            if params['use_dynamic_position']:
                if volatility > 0.03:
                    position_pct *= 0.7
                elif volatility > 0.02:
                    position_pct *= 0.85
            
            position_pct = min(position_pct, 0.50)
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
        'win_rate': win_rate
    }

def optimize_params_for_stock(code, name, df):
    """为单只股票优化参数"""
    
    print(f"\n优化 {name} 参数...")
    
    # 参数范围
    ma_fast_range = [8, 10, 12]
    ma_slow_range = [25, 30, 35]
    atr_stop_range = [2.0, 2.5, 3.0]
    
    best_sharpe = -999
    best_params = None
    best_result = None
    
    total = len(ma_fast_range) * len(ma_slow_range) * len(atr_stop_range)
    tested = 0
    
    for ma_fast in ma_fast_range:
        for ma_slow in ma_slow_range:
            for atr_stop in atr_stop_range:
                if ma_fast >= ma_slow:
                    continue
                
                params = {
                    'ma_fast': ma_fast,
                    'ma_slow': ma_slow,
                    'atr_stop_mult': atr_stop,
                    'atr_trail_mult': 2.0,
                    'use_dynamic_position': True,
                    'use_macd': True,
                    'use_rsi': True,
                    'take_profit_1': 0.10,
                    'take_profit_2': 0.20,
                    'partial_exit_1': 0.5,
                    'partial_exit_2': 0.5,
                    'base_position': 0.30
                }
                
                result = backtest_with_params(df, params)
                tested += 1
                
                if result['sharpe'] > best_sharpe:
                    best_sharpe = result['sharpe']
                    best_params = params
                    best_result = result
    
    print(f"  测试{tested}组参数")
    print(f"  最优夏普: {best_sharpe:.3f}")
    print(f"  最优参数: MA{best_params['ma_fast']}/{best_params['ma_slow']}, ATR{best_params['atr_stop_mult']}x")
    
    return best_params, best_result

def main():
    """主函数"""
    print("="*70)
    print("参数微调优化器")
    print("="*70)
    
    # 股票配置
    stocks = [
        ('300750', '宁德时代', 0.45),  # 权重从权重优化结果
        ('002475', '立讯精密', 0.30),
        ('601318', '中国平安', 0.15),
        ('600276', '恒瑞医药', 0.10)
    ]
    
    # 原始参数
    default_params = {
        'ma_fast': 10, 'ma_slow': 30,
        'atr_stop_mult': 2.5, 'atr_trail_mult': 2.0,
        'use_dynamic_position': True,
        'use_macd': True, 'use_rsi': True,
        'take_profit_1': 0.10, 'take_profit_2': 0.20,
        'partial_exit_1': 0.5, 'partial_exit_2': 0.5,
        'base_position': 0.30
    }
    
    # 优化每只股票
    optimized_params = {}
    optimized_results = {}
    
    for code, name, weight in stocks:
        filepath = Path(f'data/real_{code}.csv')
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        # 优化参数
        params, result = optimize_params_for_stock(code, name, df)
        
        optimized_params[code] = params
        optimized_results[code] = result
    
    # 计算优化后的组合指标
    print("\n" + "="*70)
    print("优化结果")
    print("="*70)
    
    print(f"\n{'股票':<10} {'权重':<8} {'原夏普':<8} {'优化夏普':<8} {'提升':<8} {'参数'}")
    print("-" * 70)
    
    portfolio_sharpe = 0
    
    for code, name, weight in stocks:
        if code not in optimized_results:
            continue
        
        # 原始夏普
        filepath = Path(f'data/real_{code}.csv')
        df = pd.read_csv(filepath)
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        default_result = backtest_with_params(df, default_params)
        default_sharpe = default_result['sharpe']
        
        # 优化后夏普
        opt_result = optimized_results[code]
        opt_sharpe = opt_result['sharpe']
        improvement = opt_sharpe - default_sharpe
        
        params = optimized_params[code]
        param_str = f"MA{params['ma_fast']}/{params['ma_slow']},ATR{params['atr_stop_mult']}"
        
        print(f"{name:<10} {weight*100:>5.0f}%  {default_sharpe:6.3f}  {opt_sharpe:6.3f}  {improvement:+6.3f}  {param_str}")
        
        portfolio_sharpe += opt_sharpe * weight
    
    print("-" * 70)
    print(f"{'组合整体':<10} {'100%':<8} {'0.544':<8} {portfolio_sharpe:6.3f}")
    
    # 保存结果
    output = {
        'optimize_time': datetime.now().isoformat(),
        'portfolio_sharpe': float(portfolio_sharpe),
        'optimized_params': {
            code: {
                'name': name,
                'weight': float(weight),
                'params': optimized_params[code],
                'sharpe': float(optimized_results[code]['sharpe']),
                'return': float(optimized_results[code]['return']),
                'win_rate': float(optimized_results[code]['win_rate'])
            } for code, name, weight in stocks if code in optimized_params
        }
    }
    
    with open('data/param_optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: data/param_optimization_results.json")
    
    # 生成最终配置
    print("\n" + "="*70)
    print("最终配置")
    print("="*70)
    
    print(f"\n总资金: 100,000元")
    print(f"组合夏普: {portfolio_sharpe:.3f}")
    
    print(f"\n股票配置:")
    for code, name, weight in stocks:
        if code not in optimized_params:
            continue
        
        params = optimized_params[code]
        result = optimized_results[code]
        amount = 100000 * weight
        
        print(f"\n  {name} ({code}):")
        print(f"    资金: {amount:.0f}元 ({weight*100:.0f}%)")
        print(f"    参数: MA{params['ma_fast']}/{params['ma_slow']}, ATR止损{params['atr_stop_mult']}x")
        print(f"    预期夏普: {result['sharpe']:.3f}")
        print(f"    预期收益: {result['return']*100:+.1f}%")
        print(f"    预期胜率: {result['win_rate']*100:.1f}%")

if __name__ == '__main__':
    main()
