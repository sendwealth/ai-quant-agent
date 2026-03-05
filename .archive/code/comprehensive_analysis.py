"""
完整数据综合分析报告
===================
整合3只真实 + 6只模拟数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def sma(data, period): return data.rolling(window=period).mean()
def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def conservative_backtest(df):
    """保守策略回测"""
    df = df.copy()
    df['ma_5'] = sma(df['close'], 5)
    df['ma_30'] = sma(df['close'], 30)
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    
    trades = 0
    equity = [cash]
    
    for i in range(50, len(df)):
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_5 = float(df['ma_5'].iloc[i])
        ma_30 = float(df['ma_30'].iloc[i])
        
        if pd.isna(ma_5) or pd.isna(ma_30):
            equity.append(cash + shares * price)
            continue
        
        if shares > 0 and price > highest:
            highest = price
            new_stop = highest - atr_val * 2.0
            if new_stop > stop_loss:
                stop_loss = new_stop
        
        if shares > 0 and price <= stop_loss:
            cash += shares * price
            shares = 0
            equity.append(cash)
            continue
        
        if ma_5 > ma_30 and shares == 0:
            shares = int(cash * 0.25 / price)
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * 2.5
                highest = price
                trades += 1
        
        elif ma_5 < ma_30 and shares > 0:
            cash += shares * price
            shares = 0
        
        equity.append(cash + shares * price)
    
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
    
    final = cash
    ret = (final - 100000) / 100000
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    return {'return': ret, 'sharpe': sharpe, 'max_dd': max_dd, 'trades': trades}

def load_and_test(filepath, name, data_type='real'):
    """加载数据并测试"""
    try:
        df = pd.read_csv(filepath)
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                                    '最低': 'low', '收盘': 'close', '成交量': 'volume'})
        
        if 'close' not in df.columns or len(df) < 50:
            return None
        
        r = conservative_backtest(df)
        bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        r['name'] = name
        r['bh'] = bh
        r['data_type'] = data_type
        r['data_len'] = len(df)
        return r
    except:
        return None

print("="*70)
print("📊 完整数据综合分析报告")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============ 测试所有数据 ============
all_results = []

# 真实数据
print("【1. 真实数据回测】")
print("-" * 70)
real_stocks = {
    '000858': '五粮液',
    '002594': '比亚迪',
    '600519': '茅台',
}

for code, name in real_stocks.items():
    filepath = f'data/real_{code}.csv'
    if Path(filepath).exists():
        r = load_and_test(filepath, name, 'real')
        if r:
            all_results.append(r)
            status = "✅" if r['return'] > 0 else "❌"
            print(f"{status} {name}: 策略{r['return']*100:+.2f}% vs 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f}")

# 模拟数据
print("\n【2. 模拟数据回测】")
print("-" * 70)
sim_stocks = {
    'sim_BULL001': '牛市股票1',
    'sim_BULL002': '牛市股票2',
    'sim_BEAR001': '熊市股票1',
    'sim_BEAR002': '熊市股票2',
    'sim_SIDE001': '震荡股票1',
    'sim_SIDE002': '震荡股票2',
}

for filepath, name in sim_stocks.items():
    full_path = f'data/{filepath}.csv'
    if Path(full_path).exists():
        r = load_and_test(full_path, name, 'sim')
        if r:
            all_results.append(r)
            status = "✅" if r['return'] > 0 else "❌"
            print(f"{status} {name}: 策略{r['return']*100:+.2f}% vs 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f}")

# ============ 统计分析 ============
print("\n" + "="*70)
print("📊 综合统计分析")
print("="*70)

if not all_results:
    print("❌ 没有有效数据")
else:
    returns = [r['return'] for r in all_results]
    sharpes = [r['sharpe'] for r in all_results]
    dds = [r['max_dd'] for r in all_results]
    bhs = [r['bh']/100 for r in all_results]
    
    win_count = sum(1 for r in returns if r > 0)
    loss_count = len(returns) - win_count
    
    avg_return = np.mean(returns) * 100
    avg_sharpe = np.mean(sharpes)
    avg_dd = np.mean(dds) * 100
    avg_bh = np.mean(bhs) * 100
    
    median_return = np.median(returns) * 100
    median_sharpe = np.median(sharpes)
    std_return = np.std(returns) * 100
    
    print(f"\n【样本统计】")
    print(f"总测试股票: {len(all_results)}")
    print(f"  真实数据: {len([r for r in all_results if r['data_type']=='real'])} 只")
    print(f"  模拟数据: {len([r for r in all_results if r['data_type']=='sim'])} 只")
    print(f"平均数据长度: {np.mean([r['data_len'] for r in all_results]):.0f}天")
    
    print(f"\n【胜率统计】")
    print(f"盈利: {win_count}/{len(all_results)} ({win_count/len(all_results)*100:.1f}%)")
    print(f"亏损: {loss_count}/{len(all_results)} ({loss_count/len(all_results)*100:.1f}%)")
    
    print(f"\n【收益统计】")
    print(f"平均收益: {avg_return:+.2f}%")
    print(f"中位收益: {median_return:+.2f}%")
    print(f"收益标准差: {std_return:.2f}%")
    print(f"最佳单股: {max(returns)*100:+.2f}%")
    print(f"最差单股: {min(returns)*100:+.2f}%")
    
    print(f"\n【风险统计】")
    print(f"平均夏普: {avg_sharpe:.2f}")
    print(f"中位夏普: {median_sharpe:.2f}")
    print(f"平均回撤: {avg_dd:.1f}%")
    
    print(f"\n【对比买入持有】")
    print(f"策略平均: {avg_return:+.2f}%")
    print(f"买入持有: {avg_bh:+.2f}%")
    print(f"超额收益: {avg_return - avg_bh:+.2f}%")
    
    # 分市场环境统计
    print(f"\n【分市场环境统计】")
    bull = [r for r in all_results if r['bh'] > 20]
    bear = [r for r in all_results if r['bh'] < -20]
    side = [r for r in all_results if -20 <= r['bh'] <= 20]
    
    if bull:
        bull_ret = np.mean([r['return'] for r in bull]) * 100
        bull_bh = np.mean([r['bh'] for r in bull])
        print(f"牛市环境({len(bull)}只): 策略{bull_ret:+.2f}% vs 买入{bull_bh:+.1f}%")
    
    if bear:
        bear_ret = np.mean([r['return'] for r in bear]) * 100
        bear_bh = np.mean([r['bh'] for r in bear])
        print(f"熊市环境({len(bear)}只): 策略{bear_ret:+.2f}% vs 买入{bear_bh:+.1f}%")
    
    if side:
        side_ret = np.mean([r['return'] for r in side]) * 100
        side_bh = np.mean([r['bh'] for r in side])
        print(f"震荡环境({len(side)}只): 策略{side_ret:+.2f}% vs 买入{side_bh:+.1f}%")
    
    # 分数据类型统计
    print(f"\n【分数据类型统计】")
    real = [r for r in all_results if r['data_type'] == 'real']
    sim = [r for r in all_results if r['data_type'] == 'sim']
    
    if real:
        real_ret = np.mean([r['return'] for r in real]) * 100
        real_win = sum(1 for r in real if r['return'] > 0)
        print(f"真实数据({len(real)}只): 平均{real_ret:+.2f}%, 胜率{real_win}/{len(real)}")
    
    if sim:
        sim_ret = np.mean([r['return'] for r in sim]) * 100
        sim_win = sum(1 for r in sim if r['return'] > 0)
        print(f"模拟数据({len(sim)}只): 平均{sim_ret:+.2f}%, 胜率{sim_win}/{len(sim)}")
    
    # 最终评级
    print(f"\n" + "="*70)
    print("【最终评级与建议】")
    print("="*70)
    
    score = 0
    reasons = []
    
    if win_count/len(all_results) >= 0.6:
        score += 2
        reasons.append("✅ 胜率≥60%")
    elif win_count/len(all_results) >= 0.5:
        score += 1
        reasons.append("⚠️ 胜率50-60%")
    else:
        reasons.append("❌ 胜率<50%")
    
    if avg_sharpe >= 0.5:
        score += 2
        reasons.append("✅ 夏普≥0.5")
    elif avg_sharpe >= 0:
        score += 1
        reasons.append("⚠️ 夏普0-0.5")
    else:
        reasons.append("❌ 夏普<0")
    
    if avg_dd >= -10:
        score += 1
        reasons.append("✅ 回撤≤10%")
    else:
        reasons.append("⚠️ 回撤>10%")
    
    for reason in reasons:
        print(f"  {reason}")
    
    print(f"\n综合得分: {score}/5")
    
    if score >= 4:
        grade = "A 🏆"
        advice = "✅ 可以考虑实盘"
        action = "建议用1-2万小资金测试1个月"
    elif score >= 3:
        grade = "B ✅"
        advice = "⚠️ 谨慎实盘"
        action = "建议先模拟盘测试1-2个月"
    elif score >= 2:
        grade = "C ⚠️"
        advice = "❌ 暂不建议实盘"
        action = "需要优化策略参数"
    else:
        grade = "D ❌"
        advice = "❌ 不推荐实盘"
        action = "策略需要重大改进"
    
    print(f"\n评级: {grade}")
    print(f"建议: {advice}")
    print(f"行动: {action}")
    
    # 详细建议
    print(f"\n【详细建议】")
    
    if score >= 3:
        print("\n✅ 如果决定实盘，请注意：")
        print("  1. 资金管理")
        print(f"     • 初始资金：1-2万（占总资金10-20%）")
        print(f"     • 单只股票：≤25%仓位")
        print(f"     • 总持仓：≤60%")
        print(f"     • 现金储备：≥40%")
        print("\n  2. 风险控制")
        print(f"     • 单日止损：-2%")
        print(f"     • 单周止损：-5%")
        print(f"     • 单月止损：-8%")
        print(f"     • 连亏3次：停止交易1天")
        print("\n  3. 执行纪律")
        print("     • 严格按策略信号交易")
        print("     • 不追涨杀跌")
        print("     • 记录每笔交易")
        print("     • 每周复盘总结")
        print("\n  4. 预期管理")
        print(f"     • 合理预期：年化8-15%")
        print(f"     • 可接受回撤：≤10%")
        print(f"     • 最小测试周期：3个月")
    else:
        print("\n⚠️ 暂不建议实盘，原因：")
        if win_count/len(all_results) < 0.5:
            print(f"  • 胜率不足50% ({win_count/len(all_results)*100:.1f}%)")
        if avg_sharpe < 0.5:
            print(f"  • 夏普比率偏低 ({avg_sharpe:.2f})")
        if avg_return < 0:
            print(f"  • 平均收益为负 ({avg_return:+.2f}%)")
        
        print("\n建议改进：")
        print("  1. 优化策略参数")
        print("  2. 增加选股条件")
        print("  3. 改进市场择时")
        print("  4. 获取更多历史数据")
        print("  5. 延长测试周期")

print("\n" + "="*70)
print("📊 分析完成")
print("="*70)
print(f"\n数据来源: {len([r for r in all_results if r['data_type']=='real'])}只真实 + {len([r for r in all_results if r['data_type']=='sim'])}只模拟")
print(f"测试策略: 保守策略 (MA5/30, ATR2.5x, 仓位25%)")
print(f"完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
