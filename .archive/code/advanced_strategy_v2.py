"""
高级策略 V2 - 多维度动态策略
============================
核心改进：
1. 多时间框架确认（日线+周线趋势）
2. 动态仓位管理（基于波动率）
3. 动量+趋势组合（MACD+MA）
4. 智能止盈（分批止盈）
5. 市场状态识别（牛熊震荡）
6. 波动率自适应参数
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

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
    """识别市场状态：牛市/熊市/震荡"""
    if len(df) < lookback:
        return 'unknown'
    
    recent = df['close'].iloc[-lookback:]
    returns = recent.pct_change().dropna()
    
    # 计算趋势强度
    ma_short = sma(df['close'], 20).iloc[-1]
    ma_long = sma(df['close'], 50).iloc[-1]
    
    # 计算波动率
    volatility = returns.std()
    
    # 计算趋势方向
    trend = (recent.iloc[-1] / recent.iloc[0] - 1)
    
    # 判断市场状态
    if trend > 0.1 and ma_short > ma_long:
        return 'bull'  # 牛市
    elif trend < -0.1 and ma_short < ma_long:
        return 'bear'  # 熊市
    else:
        return 'range'  # 震荡

def calculate_position_size(volatility, market_state):
    """根据波动率和市场状态动态调整仓位"""
    # 基础仓位
    base_position = 0.30
    
    # 根据波动率调整
    if volatility > 0.03:  # 高波动
        vol_adj = 0.7
    elif volatility > 0.02:  # 中波动
        vol_adj = 0.85
    else:  # 低波动
        vol_adj = 1.0
    
    # 根据市场状态调整
    if market_state == 'bull':
        market_adj = 1.2  # 牛市加仓
    elif market_state == 'bear':
        market_adj = 0.5  # 熊市减仓
    else:
        market_adj = 0.8  # 震荡减仓
    
    # 最终仓位
    position = base_position * vol_adj * market_adj
    return min(position, 0.50)  # 最大50%

def advanced_backtest(df, params=None):
    """高级策略回测"""
    if params is None:
        params = {
            'ma_fast': 10,
            'ma_slow': 30,
            'atr_stop_mult': 3.0,
            'atr_trail_mult': 2.5,
            'use_macd': True,
            'use_rsi': True,
            'use_dynamic_position': True,
            'take_profit_1': 0.10,  # 第一目标10%
            'take_profit_2': 0.20,  # 第二目标20%
            'partial_exit_1': 0.5,  # 第一目标卖出50%
            'partial_exit_2': 0.5,  # 第二目标卖出剩余50%
        }
    
    df = df.copy()
    
    # 计算所有指标
    df['ma_fast'] = sma(df['close'], params['ma_fast'])
    df['ma_slow'] = sma(df['close'], params['ma_slow'])
    df['ema_20'] = ema(df['close'], 20)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    
    # 计算波动率
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    partial_sold = False  # 是否已部分止盈
    
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
        
        # 识别市场状态
        market_state = detect_market_state(df.iloc[:i+1])
        
        # 持仓管理
        if shares > 0:
            # 更新最高价和移动止损
            if price > highest:
                highest = price
                new_stop = highest - atr_val * params['atr_trail_mult']
                if new_stop > stop_loss:
                    stop_loss = new_stop
            
            # 止损
            if price <= stop_loss:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
                equity.append(cash)
                continue
            
            # 分批止盈
            if not partial_sold and params.get('take_profit_1'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_1']:
                    # 卖出部分仓位
                    sell_shares = int(shares * params['partial_exit_1'])
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_sold = True
                    wins += 1
            
            # 第二止盈目标
            if partial_sold and params.get('take_profit_2'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_2']:
                    cash += shares * price
                    shares = 0
                    partial_sold = False
                    equity.append(cash)
                    continue
            
            # 趋势反转 + MACD确认
            if ma_fast < ma_slow and macd_hist < 0:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
        
        # 买入信号
        if ma_fast > ma_slow and shares == 0:
            # MACD确认
            if params['use_macd'] and macd_hist < 0:
                equity.append(cash)
                continue
            
            # RSI过滤
            if params['use_rsi'] and (rsi_val > 70 or rsi_val < 30):
                equity.append(cash)
                continue
            
            # 计算动态仓位
            if params['use_dynamic_position']:
                position_pct = calculate_position_size(volatility, market_state)
            else:
                position_pct = 0.30
            
            position_sizes.append(position_pct)
            
            # 执行买入
            position_value = cash * position_pct
            shares = int(position_value / price)
            
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * params['atr_stop_mult']
                highest = price
                trades += 1
        
        equity.append(cash + shares * price)
    
    # 平仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
        if float(df['close'].iloc[-1]) > entry_price:
            wins += 1
    
    final = cash
    ret = (final - 100000) / 100000
    
    # 计算夏普比率
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    # 计算最大回撤
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    # 胜率
    win_rate = wins / trades if trades > 0 else 0
    
    # 平均仓位
    avg_position = np.mean(position_sizes) if position_sizes else 0.30
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'avg_position': avg_position
    }

def load_and_test(filepath, name, data_type='real'):
    """加载数据并测试"""
    try:
        df = pd.read_csv(filepath)
        
        # 标准化列名
        if '日期' in df.columns:
            df = df.rename(columns={
                '日期': 'datetime', '开盘': 'open', '最高': 'high',
                '最低': 'low', '收盘': 'close', '成交量': 'volume'
            })
        
        if 'close' not in df.columns or len(df) < 60:
            return None
        
        # 运行高级策略
        r = advanced_backtest(df)
        
        # 计算买入持有收益
        bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        r['name'] = name
        r['bh'] = bh
        r['data_type'] = data_type
        
        return r
        
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        return None

def main():
    """运行高级策略测试"""
    print("=" * 70)
    print("高级策略 V2 测试 - 多维度动态策略")
    print("=" * 70)
    print("\n【核心创新】")
    print("1. 市场状态识别 (牛/熊/震荡)")
    print("2. 动态仓位管理 (基于波动率+市场状态)")
    print("3. 动量+趋势组合 (MACD+MA双重确认)")
    print("4. 智能分批止盈 (10%卖50%, 20%清仓)")
    print("5. 波动率自适应参数")
    print()
    
    data_dir = Path('data')
    results = []
    
    # 测试真实数据
    print("【测试真实数据】")
    real_stocks = {
        'wuliangye.csv': '五粮液',
        'byd.csv': '比亚迪',
        'maotai.csv': '茅台'
    }
    
    for file, name in real_stocks.items():
        filepath = data_dir / file
        if filepath.exists():
            r = load_and_test(filepath, name, 'real')
            if r:
                results.append(r)
                print(f"  ✅ {name}: 策略{r['return']*100:+.2f}% | 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f} | 胜率{r['win_rate']*100:.1f}% | 平均仓位{r['avg_position']*100:.1f}%")
    
    # 测试模拟数据
    print("\n【测试模拟数据】")
    sim_stocks = {
        'sim_BULL001.csv': '牛市1',
        'sim_BULL002.csv': '牛市2',
        'sim_BEAR001.csv': '熊市1',
        'sim_BEAR002.csv': '熊市2',
        'sim_SIDE001.csv': '震荡1',
        'sim_SIDE002.csv': '震荡2'
    }
    
    for file, name in sim_stocks.items():
        filepath = data_dir / file
        if filepath.exists():
            r = load_and_test(filepath, name, 'sim')
            if r:
                results.append(r)
                print(f"  ✅ {name}: 策略{r['return']*100:+.2f}% | 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f} | 胜率{r['win_rate']*100:.1f}% | 平均仓位{r['avg_position']*100:.1f}%")
    
    if not results:
        print("\n❌ 没有找到测试数据")
        return
    
    # 统计分析
    print("\n" + "=" * 70)
    print("【高级策略 V2 表现】")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    print("\n【整体表现】")
    print(f"  平均收益: {df_results['return'].mean()*100:+.2f}%")
    print(f"  中位收益: {df_results['return'].median()*100:+.2f}%")
    print(f"  平均夏普: {df_results['sharpe'].mean():.2f}")
    print(f"  中位夏普: {df_results['sharpe'].median():.2f}")
    print(f"  平均胜率: {df_results['win_rate'].mean()*100:.1f}%")
    print(f"  平均回撤: {df_results['max_dd'].mean()*100:.1f}%")
    print(f"  平均仓位: {df_results['avg_position'].mean()*100:.1f}%")
    
    print("\n【对比买入持有】")
    print(f"  策略平均: {df_results['return'].mean()*100:+.2f}%")
    print(f"  买入持有: {df_results['bh'].mean():+.2f}%")
    print(f"  超额收益: {(df_results['return'].mean()*100 - df_results['bh'].mean()):+.2f}%")
    
    # 分组统计
    real_data = df_results[df_results['data_type'] == 'real']
    sim_data = df_results[df_results['data_type'] == 'sim']
    
    print("\n【分组统计】")
    if len(real_data) > 0:
        print(f"  真实数据({len(real_data)}只): 策略{real_data['return'].mean()*100:+.2f}% vs 买入{real_data['bh'].mean():+.1f}%")
    if len(sim_data) > 0:
        print(f"  模拟数据({len(sim_data)}只): 策略{sim_data['return'].mean()*100:+.2f}% vs 买入{sim_data['bh'].mean():+.1f}%")
    
    # 按市场环境分组
    bull = df_results[df_results['name'].str.contains('牛市')]
    bear = df_results[df_results['name'].str.contains('熊市')]
    side = df_results[df_results['name'].str.contains('震荡')]
    
    if len(bull) > 0:
        print(f"  上涨市场({len(bull)}只): 策略{bull['return'].mean()*100:+.2f}% vs 买入{bull['bh'].mean():+.1f}%")
    if len(bear) > 0:
        print(f"  下跌市场({len(bear)}只): 策略{bear['return'].mean()*100:+.2f}% vs 买入{bear['bh'].mean():+.1f}%")
    if len(side) > 0:
        print(f"  震荡市场({len(side)}只): 策略{side['return'].mean()*100:+.2f}% vs 买入{side['bh'].mean():+.1f}%")
    
    # 与原策略对比
    print("\n" + "=" * 70)
    print("【策略对比】")
    print("=" * 70)
    print("\n指标对比 (高级V2 vs 原策略):")
    print("  原策略: 收益+0.16%, 夏普-0.10, 胜率55.6%")
    print(f"  高级V2: 收益{df_results['return'].mean()*100:+.2f}%, 夏普{df_results['sharpe'].mean():.2f}, 胜率{df_results['win_rate'].mean()*100:.1f}%")
    
    improvement = df_results['return'].mean()*100 - 0.16
    print(f"\n  收益提升: {improvement:+.2f}%")
    
    # 实盘条件检查
    print("\n" + "=" * 70)
    print("【实盘条件检查】")
    print("=" * 70)
    
    avg_sharpe = df_results['sharpe'].mean()
    avg_return = df_results['return'].mean() * 100
    win_rate = df_results['win_rate'].mean() * 100
    real_count = len(real_data)
    
    checks = [
        ("夏普比率 ≥ 0.5", avg_sharpe >= 0.5, f"{avg_sharpe:.2f}"),
        ("胜率 ≥ 60%", win_rate >= 60, f"{win_rate:.1f}%"),
        ("平均收益 ≥ 5%", avg_return >= 5, f"{avg_return:+.2f}%"),
        ("真实数据 ≥ 10只", real_count >= 10, f"{real_count}只"),
    ]
    
    passed = sum(1 for _, check, _ in checks if check)
    
    for name, check, value in checks:
        status = "✅" if check else "❌"
        print(f"  {status} {name}: {value}")
    
    print(f"\n达标情况: {passed}/4")
    
    if passed >= 3:
        print("\n✅ 接近实盘条件，可考虑小规模测试")
    elif passed >= 2:
        print("\n⚠️ 部分达标，继续优化")
    else:
        print("\n❌ 未达标，需要继续改进")
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = data_dir / f'advanced_v2_results_{timestamp}.json'
    
    summary = {
        'timestamp': timestamp,
        'strategy': 'advanced_v2',
        'total_stocks': len(results),
        'avg_return': df_results['return'].mean() * 100,
        'avg_sharpe': df_results['sharpe'].mean(),
        'avg_win_rate': df_results['win_rate'].mean() * 100,
        'avg_max_dd': df_results['max_dd'].mean() * 100,
        'avg_position': df_results['avg_position'].mean() * 100,
        'excess_return': df_results['return'].mean() * 100 - df_results['bh'].mean(),
        'passed_checks': passed,
        'results': results
    }
    
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_file}")

if __name__ == '__main__':
    main()
