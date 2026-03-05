"""
进取策略 V3 - 趋势跟踪加强版
============================
目标：在保持风险控制的同时，大幅提高收益

核心策略：
1. 趋势强度确认（只做强趋势）
2. 金字塔加仓（盈利加仓）
3. 更大的初始仓位（40%）
4. 持有赢家更久（宽止损）
5. 快速止损输家
6. 市场环境自适应
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

def adx(high, low, close, period=14):
    """ADX - 平均趋向指数，衡量趋势强度"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()
    
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_val)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_val)
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_val = dx.rolling(window=period).mean()
    
    return adx_val, plus_di, minus_di

def detect_trend_strength(df, i):
    """检测趋势强度"""
    if i < 50:
        return 'weak', 0
    
    close = df['close'].iloc[i-50:i+1]
    
    # 计算价格变化
    change = (close.iloc[-1] / close.iloc[0] - 1)
    
    # 计算MA斜率
    ma20 = sma(df['close'], 20).iloc[i]
    ma50 = sma(df['close'], 50).iloc[i]
    
    # 计算波动率
    vol = close.pct_change().std()
    
    # 趋势强度评分
    score = 0
    
    # 价格趋势
    if change > 0.15:
        score += 3
    elif change > 0.08:
        score += 2
    elif change > 0.03:
        score += 1
    
    # MA排列
    if ma20 > ma50 * 1.02:
        score += 2
    elif ma20 > ma50:
        score += 1
    
    # 波动率适中
    if 0.01 < vol < 0.03:
        score += 1
    
    if score >= 5:
        return 'strong', score
    elif score >= 3:
        return 'moderate', score
    else:
        return 'weak', score

def aggressive_backtest(df, params=None):
    """进取策略回测"""
    if params is None:
        params = {
            'initial_position': 0.40,  # 初始仓位40%
            'pyramid_position': 0.20,  # 加仓仓位20%
            'atr_stop_initial': 3.5,   # 初始止损3.5倍ATR
            'atr_stop_trail': 3.0,     # 移动止损3.0倍ATR
            'pyramid_threshold': 0.08, # 盈利8%加仓
            'take_profit': 0.25,       # 25%止盈
            'min_trend_strength': 3,   # 最小趋势强度
        }
    
    df = df.copy()
    
    # 计算指标
    df['ema_10'] = ema(df['close'], 10)
    df['ema_30'] = ema(df['close'], 30)
    df['ema_50'] = ema(df['close'], 50)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['adx'], df['plus_di'], df['minus_di'] = adx(df['high'], df['low'], df['close'])
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    pyramided = False  # 是否已加仓
    
    trades = 0
    wins = 0
    equity = [cash]
    trend_scores = []
    
    for i in range(60, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ema_10 = float(df['ema_10'].iloc[i])
        ema_30 = float(df['ema_30'].iloc[i])
        ema_50 = float(df['ema_50'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        adx_val = float(df['adx'].iloc[i])
        
        if pd.isna(ema_10) or pd.isna(ema_30) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        # 持仓管理
        if shares > 0:
            # 更新最高价和移动止损
            if price > highest:
                highest = price
                new_stop = highest - atr_val * params['atr_stop_trail']
                if new_stop > stop_loss:
                    stop_loss = new_stop
            
            # 止损
            if price <= stop_loss:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                pyramided = False
                equity.append(cash)
                continue
            
            # 金字塔加仓
            if not pyramided and shares > 0:
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['pyramid_threshold']:
                    # 加仓
                    add_cash = cash * params['pyramid_position']
                    add_shares = int(add_cash / price)
                    if add_shares > 0:
                        cash -= add_shares * price
                        shares += add_shares
                        pyramided = True
                        # 更新平均成本
                        entry_price = (entry_price * (shares - add_shares) + price * add_shares) / shares
                        # 更新止损
                        stop_loss = entry_price - atr_val * params['atr_stop_trail']
            
            # 止盈
            if params.get('take_profit'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit']:
                    cash += shares * price
                    wins += 1
                    shares = 0
                    pyramided = False
                    equity.append(cash)
                    continue
            
            # 趋势反转卖出
            if ema_10 < ema_30:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                pyramided = False
        
        # 买入信号
        if ema_10 > ema_30 and ema_30 > ema_50 and shares == 0:
            # 趋势强度确认
            trend_type, trend_score = detect_trend_strength(df, i)
            trend_scores.append(trend_score)
            
            if trend_score < params['min_trend_strength']:
                equity.append(cash)
                continue
            
            # ADX确认趋势
            if not pd.isna(adx_val) and adx_val < 20:
                equity.append(cash)
                continue
            
            # RSI过滤
            if rsi_val > 75:  # 避免极端超买
                equity.append(cash)
                continue
            
            # 执行买入
            position_value = cash * params['initial_position']
            shares = int(position_value / price)
            
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * params['atr_stop_initial']
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
    
    # 平均趋势强度
    avg_trend = np.mean(trend_scores) if trend_scores else 0
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'avg_trend': avg_trend
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
        
        # 运进取策略
        r = aggressive_backtest(df)
        
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
    """运进取策略测试"""
    print("=" * 70)
    print("进取策略 V3 测试 - 趋势跟踪加强版")
    print("=" * 70)
    print("\n【核心特点】")
    print("1. 大仓位入场 (40%)")
    print("2. 金字塔加仓 (盈利8%加仓20%)")
    print("3. 趋势强度确认 (只做强趋势)")
    print("4. 宽止损 (3.5x ATR)")
    print("5. 持有赢家 (25%止盈)")
    print("6. ADX趋势确认")
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
                print(f"  ✅ {name}: 策略{r['return']*100:+.2f}% | 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f} | 胜率{r['win_rate']*100:.1f}% | 趋势{r['avg_trend']:.1f}")
    
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
                print(f"  ✅ {name}: 策略{r['return']*100:+.2f}% | 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f} | 胜率{r['win_rate']*100:.1f}% | 趋势{r['avg_trend']:.1f}")
    
    if not results:
        print("\n❌ 没有找到测试数据")
        return
    
    # 统计分析
    print("\n" + "=" * 70)
    print("【进取策略 V3 表现】")
    print("=" * 70)
    
    df_results = pd.DataFrame(results)
    
    print("\n【整体表现】")
    print(f"  平均收益: {df_results['return'].mean()*100:+.2f}%")
    print(f"  中位收益: {df_results['return'].median()*100:+.2f}%")
    print(f"  平均夏普: {df_results['sharpe'].mean():.2f}")
    print(f"  中位夏普: {df_results['sharpe'].median():.2f}")
    print(f"  平均胜率: {df_results['win_rate'].mean()*100:.1f}%")
    print(f"  平均回撤: {df_results['max_dd'].mean()*100:.1f}%")
    print(f"  平均趋势强度: {df_results['avg_trend'].mean():.1f}")
    
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
    
    # 策略对比
    print("\n" + "=" * 70)
    print("【策略进化对比】")
    print("=" * 70)
    print("\n三个版本对比:")
    print("  原策略:  收益+0.16%, 夏普-0.10, 胜率55.6%")
    print("  高级V2:  收益+0.99%, 夏普+0.08, 胜率57.5%")
    print(f"  进取V3:  收益{df_results['return'].mean()*100:+.2f}%, 夏普{df_results['sharpe'].mean():.2f}, 胜率{df_results['win_rate'].mean()*100:.1f}%")
    
    improvement = df_results['return'].mean()*100 - 0.16
    print(f"\n  总收益提升: {improvement:+.2f}%")
    
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
    output_file = data_dir / f'aggressive_v3_results_{timestamp}.json'
    
    summary = {
        'timestamp': timestamp,
        'strategy': 'aggressive_v3',
        'total_stocks': len(results),
        'avg_return': df_results['return'].mean() * 100,
        'avg_sharpe': df_results['sharpe'].mean(),
        'avg_win_rate': df_results['win_rate'].mean() * 100,
        'avg_max_dd': df_results['max_dd'].mean() * 100,
        'avg_trend': df_results['avg_trend'].mean(),
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
