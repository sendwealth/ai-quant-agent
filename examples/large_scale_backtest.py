"""
大规模数据回测验证
==================
获取更多股票数据进行全面验证
"""
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime
import time

print("="*70)
print("🔍 大规模数据回测验证")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============ 1. 尝试获取真实数据 ============
print("【步骤1】获取更多真实股票数据...")

stocks_to_try = [
    ('000333', '美的集团'),
    ('002475', '立讯精密'),
    ('300750', '宁德时代'),
    ('603259', '药明康德'),
    ('600309', '万华化学'),
    ('002415', '海康威视'),
    ('000001', '平安银行'),
    ('600036', '招商银行'),
]

real_data = {}
success_count = 0

for code, name in stocks_to_try:
    try:
        print(f"  尝试获取 {name}({code})...", end=' ')
        df = ak.stock_zh_a_hist(symbol=code, period='daily', 
                                start_date='20240101', end_date='20260303', adjust='qfq')
        if len(df) > 100:
            df.to_csv(f'data/real_{code}.csv', index=False)
            change = (df['收盘'].iloc[-1]/df['收盘'].iloc[0]-1)*100
            print(f"✓ {len(df)}条, {change:+.1f}%")
            real_data[code] = {'name': name, 'len': len(df), 'change': change}
            success_count += 1
        time.sleep(2)
    except Exception as e:
        print(f"✗ 失败")
        time.sleep(1)

print(f"\n成功获取: {success_count}/{len(stocks_to_try)} 只股票")

# ============ 2. 生成模拟数据补充 ============
print("\n【步骤2】生成补充模拟数据...")

def generate_market_data(days=500, trend='mixed'):
    """生成不同市场环境的数据"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    if trend == 'bull':
        drift = 0.0008
    elif trend == 'bear':
        drift = -0.0006
    else:
        drift = 0.0002
    
    returns = np.random.randn(days) * 0.02 + drift
    price = 100
    prices = [price]
    for r in returns[1:]:
        price *= (1 + r)
        prices.append(price)
    
    close = np.array(prices)
    return pd.DataFrame({
        'datetime': dates,
        'open': close * (1 + (np.random.rand(days) - 0.5) * 0.01),
        'high': close * (1 + np.random.rand(days) * 0.02),
        'low': close * (1 - np.random.rand(days) * 0.02),
        'close': close,
        'volume': np.random.randint(1000000, 10000000, days)
    })

sim_stocks = {
    'BULL001': ('牛市股票1', generate_market_data(500, 'bull')),
    'BULL002': ('牛市股票2', generate_market_data(500, 'bull')),
    'BEAR001': ('熊市股票1', generate_market_data(500, 'bear')),
    'BEAR002': ('熊市股票2', generate_market_data(500, 'bear')),
    'SIDE001': ('震荡股票1', generate_market_data(500, 'mixed')),
    'SIDE002': ('震荡股票2', generate_market_data(500, 'mixed')),
}

for code, (name, df) in sim_stocks.items():
    filepath = f'data/sim_{code}.csv'
    df.to_csv(filepath, index=False)
    change = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    print(f"  ✓ {name}: {len(df)}天, {change:+.1f}%")

print(f"\n生成模拟数据: {len(sim_stocks)} 只股票")

# ============ 3. 统计总数据量 ============
print("\n【步骤3】数据统计")
print("-" * 70)

total_stocks = len(real_data) + len(sim_stocks)
print(f"总股票数: {total_stocks}")
print(f"  真实数据: {len(real_data)} 只")
print(f"  模拟数据: {len(sim_stocks)} 只")

# ============ 4. 策略回测函数 ============
print("\n【步骤4】开始大规模回测...")
print("-" * 70)

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

def load_and_test(filepath, name):
    """加载数据并测试"""
    try:
        df = pd.read_csv(filepath)
        # 标准化列名
        if '日期' in df.columns:
            df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                                    '最低': 'low', '收盘': 'close', '成交量': 'volume'})
        
        if 'close' not in df.columns or len(df) < 50:
            return None
        
        r = conservative_backtest(df)
        bh = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        r['name'] = name
        r['bh'] = bh
        r['data_len'] = len(df)
        return r
    except:
        return None

# 测试所有数据
all_results = []

# 测试真实数据
print("\n真实数据回测:")
for code, info in real_data.items():
    filepath = f'data/real_{code}.csv'
    r = load_and_test(filepath, info['name'])
    if r:
        all_results.append(r)
        status = "✅" if r['return'] > 0 else "❌"
        print(f"  {status} {r['name']}: 策略{r['return']*100:+.2f}% vs 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f}")

# 测试模拟数据
print("\n模拟数据回测:")
for code, (name, df) in sim_stocks.items():
    filepath = f'data/sim_{code}.csv'
    r = load_and_test(filepath, name)
    if r:
        all_results.append(r)
        status = "✅" if r['return'] > 0 else "❌"
        print(f"  {status} {r['name']}: 策略{r['return']*100:+.2f}% vs 买入{r['bh']:+.1f}% | 夏普{r['sharpe']:.2f}")

# ============ 5. 统计分析 ============
print("\n" + "="*70)
print("📊 大规模回测统计结果")
print("="*70)

if not all_results:
    print("❌ 没有有效的回测结果")
else:
    returns = [r['return'] for r in all_results]
    sharpes = [r['sharpe'] for r in all_results]
    dds = [r['max_dd'] for r in all_results]
    bhs = [r['bh']/100 for r in all_results]
    
    # 基本统计
    win_count = sum(1 for r in returns if r > 0)
    loss_count = len(returns) - win_count
    
    avg_return = np.mean(returns) * 100
    avg_sharpe = np.mean(sharpes)
    avg_dd = np.mean(dds) * 100
    avg_bh = np.mean(bhs) * 100
    
    # 中位数
    median_return = np.median(returns) * 100
    median_sharpe = np.median(sharpes)
    
    # 标准差
    std_return = np.std(returns) * 100
    
    print(f"\n【样本统计】")
    print(f"测试股票数: {len(all_results)}")
    print(f"数据期间: {min([r['data_len'] for r in all_results])}-{max([r['data_len'] for r in all_results])}天")
    
    print(f"\n【胜率统计】")
    print(f"盈利股票: {win_count}/{len(all_results)} ({win_count/len(all_results)*100:.1f}%)")
    print(f"亏损股票: {loss_count}/{len(all_results)} ({loss_count/len(all_results)*100:.1f}%)")
    
    print(f"\n【收益统计】")
    print(f"平均收益: {avg_return:+.2f}%")
    print(f"中位收益: {median_return:+.2f}%")
    print(f"收益标准差: {std_return:.2f}%")
    print(f"最佳: {max(returns)*100:+.2f}%")
    print(f"最差: {min(returns)*100:+.2f}%")
    
    print(f"\n【风险统计】")
    print(f"平均夏普: {avg_sharpe:.2f}")
    print(f"中位夏普: {median_sharpe:.2f}")
    print(f"平均回撤: {avg_dd:.1f}%")
    
    print(f"\n【对比买入持有】")
    print(f"策略平均: {avg_return:+.2f}%")
    print(f"买入持有: {avg_bh:+.2f}%")
    print(f"超额收益: {avg_return - avg_bh:+.2f}%")
    
    # 分组统计
    print(f"\n【分组统计】")
    positive_bh = [r for r in all_results if r['bh'] > 0]
    negative_bh = [r for r in all_results if r['bh'] <= 0]
    
    if positive_bh:
        pos_ret = np.mean([r['return'] for r in positive_bh]) * 100
        pos_bh = np.mean([r['bh'] for r in positive_bh])
        print(f"上涨市场({len(positive_bh)}只): 策略{pos_ret:+.2f}% vs 买入{pos_bh:+.1f}%")
    
    if negative_bh:
        neg_ret = np.mean([r['return'] for r in negative_bh]) * 100
        neg_bh = np.mean([r['bh'] for r in negative_bh])
        print(f"下跌市场({len(negative_bh)}只): 策略{neg_ret:+.2f}% vs 买入{neg_bh:+.1f}%")
    
    # 评级
    print(f"\n【最终评级】")
    if avg_sharpe > 1.0 and win_count/len(all_results) > 0.6:
        grade = "A 🏆 优秀 - 可考虑实盘"
    elif avg_sharpe > 0.5 and win_count/len(all_results) > 0.5:
        grade = "B ✅ 良好 - 建议小资金测试"
    elif avg_sharpe > 0:
        grade = "C ⚠️ 一般 - 需要优化"
    else:
        grade = "D ❌ 不推荐 - 风险较高"
    
    print(f"评级: {grade}")
    
    print(f"\n【实盘建议】")
    if avg_sharpe > 0.5 and win_count/len(all_results) > 0.5:
        print("✅ 可以考虑实盘，但建议：")
        print("  1. 先用1-2万小资金测试1个月")
        print("  2. 严格止损纪律")
        print("  3. 分散投资3-5只股票")
        print("  4. 保留20%以上现金")
    else:
        print("⚠️ 暂不建议实盘，建议：")
        print("  1. 继续优化策略参数")
        print("  2. 获取更多历史数据")
        print("  3. 延长测试周期")
        print("  4. 模拟盘验证")

print("="*70)
print(f"\n完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
