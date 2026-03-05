"""
多股票验证 - 使用29只真实A股数据
===================================
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

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

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
    return min(base_position * vol_adj * market_adj, 0.50)

def advanced_backtest(df, params):
    """高级策略回测"""
    df = df.copy()
    
    # 计算指标
    df['ma_fast'] = sma(df['close'], params['ma_fast'])
    df['ma_slow'] = sma(df['close'], params['ma_slow'])
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
            
            position_pct = calculate_position_size(volatility, market_state)
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
    
    return {'return': ret, 'sharpe': sharpe, 'max_dd': max_dd, 'trades': trades, 'win_rate': win_rate}

def main():
    """主函数"""
    print("="*70)
    print("多股票验证 - 29只A股")
    print("="*70)
    
    # 优化参数
    params = {
        'ma_fast': 10, 'ma_slow': 30,
        'atr_stop_mult': 2.5, 'atr_trail_mult': 2.0,
        'use_dynamic_position': True,
        'use_macd': True, 'use_rsi': True,
        'take_profit_1': 0.10, 'take_profit_2': 0.20,
        'partial_exit_1': 0.5, 'partial_exit_2': 0.5
    }
    
    print(f"\n【策略参数】")
    print(f"  MA: {params['ma_fast']}/{params['ma_slow']}, ATR: {params['atr_stop_mult']}x")
    print(f"  动态仓位 + MACD确认 + 分批止盈")
    
    # 股票名称映射
    names = {
        '600519': '茅台', '000858': '五粮液', '000568': '泸州老窖', '000596': '古井贡酒',
        '002304': '洋河股份', '002594': '比亚迪', '300750': '宁德时代', '601012': '隆基绿能',
        '002129': 'TCL中环', '600438': '通威股份', '601318': '中国平安', '601398': '工商银行',
        '600036': '招商银行', '601166': '兴业银行', '000001': '平安银行', '600276': '恒瑞医药',
        '000538': '云南白药', '300760': '迈瑞医疗', '002007': '华兰生物', '000661': '长春高新',
        '002415': '海康威视', '002230': '科大讯飞', '600588': '用友网络', '000725': '京东方A',
        '002475': '立讯精密', '000333': '美的集团', '000651': '格力电器', '600887': '伊利股份',
        '000895': '双汇发展'
    }
    
    # 查找数据文件
    data_dir = Path('data')
    real_files = list(data_dir.glob('real_*.csv'))
    
    print(f"\n【找到{len(real_files)}只股票】\n")
    
    results = []
    
    for filepath in real_files:
        stock_code = filepath.stem.replace('real_', '')
        stock_name = names.get(stock_code, stock_code)
        
        # 加载数据
        df = pd.read_csv(filepath)
        
        # 标准化列名
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        if 'close' not in df.columns or len(df) < 60:
            continue
        
        # 计算买入持有收益
        bh_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # 运行策略
        try:
            r = advanced_backtest(df, params)
            
            result = {
                'code': stock_code,
                'name': stock_name,
                'strategy_return': r['return'] * 100,
                'bh_return': bh_return,
                'excess_return': r['return'] * 100 - bh_return,
                'sharpe': r['sharpe'],
                'win_rate': r['win_rate'] * 100,
                'max_dd': r['max_dd'] * 100,
                'trades': r['trades']
            }
            
            results.append(result)
            
            # 显示结果
            status = "✅" if r['sharpe'] >= 0.5 and r['win_rate'] >= 0.6 else "⚠️"
            print(f"  {status} {stock_name:<8} 策略{r['return']*100:>+6.1f}% | 买入{bh_return:>+6.1f}% | 夏普{r['sharpe']:>6.2f} | 胜率{r['win_rate']*100:>5.1f}%")
            
        except Exception as e:
            print(f"  ❌ {stock_name}: {str(e)[:50]}")
    
    # 汇总统计
    if results:
        print("\n" + "="*70)
        print("验证结果汇总")
        print("="*70)
        
        df_results = pd.DataFrame(results)
        
        print(f"\n【整体表现】")
        print(f"  平均收益: {df_results['strategy_return'].mean():+.2f}%")
        print(f"  中位收益: {df_results['strategy_return'].median():+.2f}%")
        print(f"  平均夏普: {df_results['sharpe'].mean():.3f}")
        print(f"  中位夏普: {df_results['sharpe'].median():.3f}")
        print(f"  平均胜率: {df_results['win_rate'].mean():.1f}%")
        print(f"  平均回撤: {df_results['max_dd'].mean():.1f}%")
        
        print(f"\n【对比买入持有】")
        print(f"  策略平均: {df_results['strategy_return'].mean():+.2f}%")
        print(f"  买入持有: {df_results['bh_return'].mean():+.2f}%")
        print(f"  超额收益: {df_results['excess_return'].mean():+.2f}%")
        
        # 实盘条件检查
        print(f"\n【实盘条件检查】")
        avg_sharpe = df_results['sharpe'].mean()
        avg_return = df_results['strategy_return'].mean()
        avg_win_rate = df_results['win_rate'].mean()
        
        print(f"  {'✅' if avg_sharpe >= 0.5 else '❌'} 夏普≥0.5: {avg_sharpe:.3f}")
        print(f"  {'✅' if avg_win_rate >= 60 else '❌'} 胜率≥60%: {avg_win_rate:.1f}%")
        print(f"  {'✅' if avg_return >= 5 else '❌'} 收益≥5%: {avg_return:+.2f}%")
        
        # 达标股票统计
        qualified = df_results[
            (df_results['sharpe'] >= 0.5) & 
            (df_results['win_rate'] >= 60)
        ]
        
        passed = len(qualified)
        total = len(results)
        
        print(f"\n【达标股票】{passed}/{total}只 ({passed/total*100:.1f}%)")
        
        if passed > 0:
            print("\n推荐实盘股票 (Top 10):")
            for _, row in qualified.sort_values('sharpe', ascending=False).head(10).iterrows():
                print(f"  ✅ {row['name']:<8} 夏普{row['sharpe']:.2f} | 收益{row['strategy_return']:+.1f}% | 胜率{row['win_rate']:.0f}%")
        
        # 保存结果
        validation_report = {
            'test_time': datetime.now().isoformat(),
            'total_stocks': total,
            'qualified_stocks': passed,
            'avg_sharpe': float(avg_sharpe),
            'avg_return': float(avg_return),
            'avg_win_rate': float(avg_win_rate),
            'avg_max_dd': float(df_results['max_dd'].mean()),
            'excess_return': float(df_results['excess_return'].mean()),
            'results': results
        }
        
        output_file = data_dir / f'validation_29stocks_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n验证结果已保存: {output_file}")
        
        # 最终建议
        print("\n" + "="*70)
        print("最终建议")
        print("="*70)
        
        if passed >= total * 0.3 and avg_sharpe >= 0.3:
            print("\n✅ 策略在部分股票上验证成功，建议：")
            print(f"  1. 优先在{passed}只达标股票上小规模实盘")
            print(f"  2. 资金: 1-2万元")
            print(f"  3. 预期收益: 5-10%")
        elif passed > 0:
            print("\n⚠️ 少数股票达标，建议：")
            print("  1. 继续优化参数")
            print("  2. 添加市场环境过滤")
            print("  3. 只在达标股票上测试")
        else:
            print("\n❌ 未达标，需要：")
            print("  1. 重新优化参数")
            print("  2. 改进策略逻辑")
            print("  3. 添加市场过滤")

if __name__ == '__main__':
    main()
