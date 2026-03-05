"""
简化版优化脚本 - 使用现有数据
==============================
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 导入策略函数
def sma(data, period):
    return data.rolling(window=period).mean()

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_market_state(df, lookback=50):
    if len(df) < lookback:
        return 'unknown'
    
    recent = df['close'].iloc[-lookback:]
    returns = recent.pct_change().dropna()
    
    ma_short = sma(df['close'], 20).iloc[-1]
    ma_long = sma(df['close'], 50).iloc[-1]
    
    volatility = returns.std()
    trend = (recent.iloc[-1] / recent.iloc[0] - 1)
    
    if trend > 0.1 and ma_short > ma_long:
        return 'bull'
    elif trend < -0.1 and ma_short < ma_long:
        return 'bear'
    else:
        return 'range'

def calculate_position_size(volatility, market_state):
    base_position = 0.30
    
    if volatility > 0.03:
        vol_adj = 0.7
    elif volatility > 0.02:
        vol_adj = 0.85
    else:
        vol_adj = 1.0
    
    if market_state == 'bull':
        market_adj = 1.2
    elif market_state == 'bear':
        market_adj = 0.5
    else:
        market_adj = 0.8
    
    position = base_position * vol_adj * market_adj
    return min(position, 0.50)

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def advanced_backtest(df, params):
    """高级策略回测"""
    df = df.copy()
    
    # 计算指标
    df['ma_fast'] = sma(df['close'], params['ma_fast'])
    df['ma_slow'] = sma(df['close'], params['ma_slow'])
    df['ema_20'] = ema(df['close'], 20)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    partial_sold = False
    
    trades = 0
    wins = 0
    equity = [cash]
    position_sizes = []
    
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
        
        market_state = detect_market_state(df.iloc[:i+1])
        
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
        
        if ma_fast > ma_slow and shares == 0:
            if params['use_macd'] and macd_hist < 0:
                equity.append(cash)
                continue
            
            if params['use_rsi'] and (rsi_val > 70 or rsi_val < 30):
                equity.append(cash)
                continue
            
            if params['use_dynamic_position']:
                position_pct = calculate_position_size(volatility, market_state)
            else:
                position_pct = 0.30
            
            position_sizes.append(position_pct)
            
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
    avg_position = np.mean(position_sizes) if position_sizes else 0.30
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'avg_position': avg_position
    }

def grid_search(df, param_ranges):
    """网格搜索"""
    print("\n" + "="*70)
    print("参数网格搜索优化")
    print("="*70)
    
    results = []
    best_sharpe = -999
    best_params = None
    best_result = None
    
    count = 0
    total = (len(param_ranges['ma_fast']) * 
             len(param_ranges['ma_slow']) * 
             len(param_ranges['atr_stop']) *
             len(param_ranges['position']))
    
    print(f"\n总测试数: {total}")
    print("开始优化...\n")
    
    for ma_fast in param_ranges['ma_fast']:
        for ma_slow in param_ranges['ma_slow']:
            if ma_fast >= ma_slow:
                continue
            
            for atr_stop in param_ranges['atr_stop']:
                for position in param_ranges['position']:
                    count += 1
                    
                    params = {
                        'ma_fast': ma_fast,
                        'ma_slow': ma_slow,
                        'atr_stop_mult': atr_stop,
                        'atr_trail_mult': atr_stop * 0.8,
                        'use_dynamic_position': True,
                        'use_macd': True,
                        'use_rsi': True,
                        'take_profit_1': 0.10,
                        'take_profit_2': 0.20,
                        'partial_exit_1': 0.5,
                        'partial_exit_2': 0.5,
                    }
                    
                    try:
                        r = advanced_backtest(df, params)
                        
                        results.append({
                            'ma_fast': ma_fast,
                            'ma_slow': ma_slow,
                            'atr_stop': atr_stop,
                            'position': position,
                            'sharpe': r['sharpe'],
                            'return': r['return'],
                            'win_rate': r['win_rate'],
                            'max_dd': r['max_dd'],
                            'trades': r['trades']
                        })
                        
                        if r['sharpe'] > best_sharpe:
                            best_sharpe = r['sharpe']
                            best_params = params.copy()
                            best_result = r.copy()
                        
                        if count % 20 == 0:
                            print(f"  进度: {count}/{total} | 最佳夏普: {best_sharpe:.3f}")
                    
                    except Exception as e:
                        pass
    
    print(f"\n✅ 完成! 测试了{len(results)}组参数")
    
    if best_params:
        print(f"\n【最佳参数】")
        print(f"  MA周期: {best_params['ma_fast']}/{best_params['ma_slow']}")
        print(f"  ATR止损: {best_params['atr_stop_mult']:.1f}x")
        print(f"  基础仓位: 动态调整")
        print(f"\n【最佳结果】")
        print(f"  夏普比率: {best_result['sharpe']:.3f}")
        print(f"  总收益: {best_result['return']*100:+.2f}%")
        print(f"  胜率: {best_result['win_rate']*100:.1f}%")
        print(f"  最大回撤: {best_result['max_dd']*100:.1f}%")
        print(f"  交易次数: {best_result['trades']}")
        
        # 保存结果
        output = {
            'timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'best_params': best_params,
            'best_result': {
                'sharpe': float(best_result['sharpe']),
                'return': float(best_result['return']),
                'win_rate': float(best_result['win_rate']),
                'max_dd': float(best_result['max_dd']),
                'trades': int(best_result['trades'])
            },
            'top_10': sorted(results, key=lambda x: x['sharpe'], reverse=True)[:10]
        }
        
        with open('data/optimization_results.json', 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存: data/optimization_results.json")
        
        return best_params, best_result
    else:
        print("\n❌ 没有找到有效结果")
        return None, None

def main():
    """主流程"""
    print("="*70)
    print("简化版优化流程")
    print("="*70)
    
    data_dir = Path('data')
    
    # 查找可用数据
    real_files = list(data_dir.glob('real_*.csv'))
    sim_files = list(data_dir.glob('sim_*.csv'))
    
    all_results = []
    
    # 测试真实数据
    if real_files:
        print(f"\n【找到{len(real_files)}只真实股票】")
        for filepath in real_files:
            name = filepath.stem
            print(f"\n测试 {name}...")
            
            df = pd.read_csv(filepath)
            if '日期' in df.columns:
                df = df.rename(columns={
                    '日期': 'datetime', '开盘': 'open', '最高': 'high',
                    '最低': 'low', '收盘': 'close', '成交量': 'volume'
                })
            
            # 参数范围
            param_ranges = {
                'ma_fast': [5, 8, 10, 12],
                'ma_slow': [20, 25, 30, 35],
                'atr_stop': [2.5, 3.0, 3.5],
                'position': [0.25, 0.30, 0.35]
            }
            
            best_params, best_result = grid_search(df, param_ranges)
            
            if best_params:
                all_results.append({
                    'stock': name,
                    'best_params': best_params,
                    'best_result': best_result
                })
    
    # 测试模拟数据
    print(f"\n【测试{len(sim_files)}只模拟股票】")
    for filepath in sim_files[:3]:  # 只测试前3只
        name = filepath.stem
        print(f"\n测试 {name}...")
        
        df = pd.read_csv(filepath)
        if '日期' in df.columns:
            df = df.rename(columns={
                '日期': 'datetime', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            })
        
        param_ranges = {
            'ma_fast': [5, 8, 10, 12],
            'ma_slow': [20, 25, 30, 35],
            'atr_stop': [2.5, 3.0, 3.5],
            'position': [0.25, 0.30, 0.35]
        }
        
        best_params, best_result = grid_search(df, param_ranges)
        
        if best_params:
            all_results.append({
                'stock': name,
                'best_params': best_params,
                'best_result': best_result
            })
    
    # 汇总结果
    if all_results:
        print("\n" + "="*70)
        print("优化汇总")
        print("="*70)
        
        avg_sharpe = np.mean([r['best_result']['sharpe'] for r in all_results])
        avg_return = np.mean([r['best_result']['return'] for r in all_results])
        avg_win_rate = np.mean([r['best_result']['win_rate'] for r in all_results])
        
        print(f"\n平均夏普: {avg_sharpe:.3f}")
        print(f"平均收益: {avg_return*100:+.2f}%")
        print(f"平均胜率: {avg_win_rate*100:.1f}%")
        
        # 实盘条件检查
        print("\n【实盘条件】")
        print(f"  {'✅' if avg_sharpe >= 0.5 else '❌'} 夏普 ≥ 0.5: {avg_sharpe:.3f}")
        print(f"  {'✅' if avg_win_rate >= 0.6 else '❌'} 胜率 ≥ 60%: {avg_win_rate*100:.1f}%")
        print(f"  {'✅' if avg_return >= 0.05 else '❌'} 收益 ≥ 5%: {avg_return*100:+.2f}%")

if __name__ == '__main__':
    main()
