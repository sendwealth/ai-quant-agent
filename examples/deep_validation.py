"""
深度验证脚本
1. 获取更长时间数据 (500+ 交易日)
2. 进一步优化策略参数 (遗传算法)
3. 充分模拟验证 (3个月连续测试)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
import random
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


print("\n" + "="*70)
print("AI智能体量化交易系统 - 深度验证")
print("="*70)
print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = {}


# ============================================
# 任务1: 获取更长时间数据
# ============================================
print("\n" + "="*70)
print("任务1: 获取更长时间数据 (500+ 交易日)")
print("="*70)

from data.astock_fetcher import AStockDataFetcher

fetcher = AStockDataFetcher()
stock_code = '600519'

# 获取2年历史数据（约500个交易日）
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

print(f"\n获取数据: {stock_code}")
print(f"时间范围: {start_date} -> {end_date} (2年)")
print(f"目标: 500+ 交易日")

df = fetcher.fetch_stock_daily(stock_code, start_date, end_date, source='akshare')

if df is not None and len(df) >= 500:
    print(f"\n✓ 数据获取成功!")
    print(f"  记录数: {len(df)} 条")
    print(f"  时间跨度: {(df['datetime'].iloc[-1] - df['datetime'].iloc[0]).days} 天")
    print(f"  首日: {str(df['datetime'].iloc[0])[:10]}")
    print(f"  末日: {str(df['datetime'].iloc[-1])[:10]}")

    # 数据质量检查
    print(f"\n数据质量检查:")
    print(f"  价格范围: ¥{df['close'].min():.2f} - ¥{df['close'].max():.2f}")
    print(f"  价格变动: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:+.2f}%")
    print(f"  成交量: {df['volume'].min():,.0f} - {df['volume'].max():,.0f}")
    print(f"  缺失值: {df.isnull().sum().sum()}")

    results['get_long_data'] = True

else:
    print(f"\n⚠️  数据获取失败或不足")
    print(f"  使用模拟数据...")

    # 生成3年模拟数据（约750个交易日）
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    np.random.seed(42)

    # 模拟价格走势（随机游走）
    price = 1500.0
    prices = []
    for _ in range(len(dates)):
        change = np.random.normal(0, 0.02)  # 2% 日波动
        price = price * (1 + change)
        prices.append(max(price, 100))  # 最低价格

    df = pd.DataFrame({
        'datetime': dates,
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': [int(np.random.uniform(1000000, 5000000)) for _ in prices],
    })

    print(f"  模拟数据: {len(df)} 条")
    results['get_long_data'] = False


# ============================================
# 任务2: 策略参数优化 (遗传算法)
# ============================================
print("\n" + "="*70)
print("任务2: 策略参数优化 (遗传算法)")
print("="*70)

from utils.indicators import sma, rsi

print(f"\n优化目标: 均线交叉策略")
print(f"优化参数: 短期均线周期, 长期均线周期, RSI超买阈值, RSI超卖阈值")

# 计算指标
df['close_price'] = df['close'].copy()

# 定义目标函数
def objective(params, data):
    """
    目标函数：最大化夏普比率

    Args:
        params: [ma_short, ma_long, rsi_overbought, rsi_oversold]
        data: 股票数据

    Returns:
        夏普比率 (负值用于最小化)
    """
    try:
        ma_short, ma_long, rsi_overbought, rsi_oversold = params

        # 计算指标
        data_copy = data.copy()
        data_copy['sma_short'] = sma(data_copy['close_price'], ma_short)
        data_copy['sma_long'] = sma(data_copy['close_price'], ma_long)
        data_copy['rsi'] = rsi(data_copy['close_price'], 14)

        # 生成信号
        data_copy['signal'] = 0

        # 确保指标计算完成（去除NaN）
        data_copy['sma_short'] = data_copy['sma_short'].fillna(method='ffill')
        data_copy['sma_long'] = data_copy['sma_long'].fillna(method='ffill')
        data_copy['rsi'] = data_copy['rsi'].fillna(50)  # RSI默认50

        # 生成信号
        buy_condition = (data_copy['sma_short'] > data_copy['sma_long']) & \
                        (data_copy['rsi'] < rsi_overbought)
        sell_condition = (data_copy['sma_short'] < data_copy['sma_long']) | \
                         (data_copy['rsi'] > rsi_oversold)

        data_copy.loc[buy_condition, 'signal'] = 1
        data_copy.loc[sell_condition, 'signal'] = -1

        # 回测
        capital = 100000
        position = 0
        equity_curve = []

        for i in range(1, len(data_copy)):
            price = data_copy['close'].iloc[i]
            signal = data_copy['signal'].iloc[i]
            prev_signal = data_copy['signal'].iloc[i-1]

            # 买入信号：从0或-1变为1
            if signal == 1 and position == 0:
                position = capital / price
            # 卖出信号：从1变为-1或0
            elif signal <= 0 and position > 0:
                capital = position * price
                position = 0

            equity = position * price if position > 0 else capital
            equity_curve.append(equity)

        if len(equity_curve) < 50:
            return -999  # 数据不足

        # 计算夏普比率
        equity_values = pd.Series(equity_curve)
        daily_returns = equity_values.pct_change().dropna()

        if daily_returns.std() == 0:
            return -999

        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        # 返回负的夏普比率（用于最小化）
        return -sharpe_ratio

    except Exception as e:
        return -999


# 遗传算法优化
def genetic_algorithm(data, pop_size=30, generations=50, mutation_rate=0.1):
    """
    遗传算法参数优化

    Args:
        data: 股票数据
        pop_size: 种群大小
        generations: 迭代次数
        mutation_rate: 变异率

    Returns:
        最优参数
    """
    print(f"\n遗传算法配置:")
    print(f"  种群大小: {pop_size}")
    print(f"  迭代次数: {generations}")
    print(f"  变异率: {mutation_rate}")

    # 初始化种群
    population = []
    for _ in range(pop_size):
        params = [
            random.randint(5, 20),   # ma_short: 5-20
            random.randint(20, 60),  # ma_long: 20-60
            random.randint(70, 90),  # rsi_overbought: 70-90
            random.randint(10, 30),  # rsi_oversold: 10-30
        ]
        population.append(params)

    best_params = None
    best_score = float('inf')

    # 进化
    for gen in range(generations):
        # 评估适应度
        scores = []
        for params in population:
            score = objective(params, data)
            scores.append(score)

            # 更新最优
            if score < best_score:
                best_score = score
                best_params = params

        # 选择（轮盘赌）
        min_score = min(scores)
        max_score = max(scores)
        if max_score - min_score > 0:
            weights = [(score - min_score) / (max_score - min_score) for score in scores]
        else:
            weights = [1.0] * len(scores)

        # 归一化
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(scores)] * len(scores)

        # 选择
        selected = []
        for _ in range(pop_size):
            idx = np.random.choice(range(pop_size), p=weights)
            selected.append(population[idx])

        # 交叉
        offspring = []
        for i in range(0, pop_size, 2):
            parent1 = selected[i]
            parent2 = selected[i + 1] if i + 1 < pop_size else selected[0]

            # 单点交叉
            crossover_point = random.randint(1, 3)
            child1 = parent1[:crossover_point] + parent2[crossover_point:]
            child2 = parent2[:crossover_point] + parent1[crossover_point:]

            offspring.append(child1)
            offspring.append(child2)

        # 变异
        for child in offspring:
            if random.random() < mutation_rate:
                # 随机选择一个基因进行变异
                gene_idx = random.randint(0, 3)
                if gene_idx == 0:
                    child[gene_idx] = random.randint(5, 20)
                elif gene_idx == 1:
                    child[gene_idx] = random.randint(20, 60)
                elif gene_idx == 2:
                    child[gene_idx] = random.randint(70, 90)
                else:
                    child[gene_idx] = random.randint(10, 30)

        population = offspring

        # 进度
        if gen % 10 == 0:
            print(f"  第{gen}代: 最优夏普比率 = {-best_score:.2f}")

    print(f"\n✓ 遗传算法优化完成!")
    print(f"  最优夏普比率: {-best_score:.2f}")

    return best_params


# 运行优化
print(f"\n开始优化...")

best_params = genetic_algorithm(
    df,
    pop_size=30,
    generations=50,
    mutation_rate=0.1
)

if best_params:
    ma_short, ma_long, rsi_overbought, rsi_oversold = best_params

    print(f"\n最优参数:")
    print(f"  短期均线周期: {ma_short}")
    print(f"  长期均线周期: {ma_long}")
    print(f"  RSI超买阈值: {rsi_overbought}")
    print(f"  RSI超卖阈值: {rsi_oversold}")

    # 用最优参数回测
    df['sma_short'] = sma(df['close_price'], ma_short)
    df['sma_long'] = sma(df['close_price'], ma_long)
    df['rsi'] = rsi(df['close_price'], 14)

    # 确保指标计算完成
    df['sma_short'] = df['sma_short'].fillna(method='ffill')
    df['sma_long'] = df['sma_long'].fillna(method='ffill')
    df['rsi'] = df['rsi'].fillna(50)

    df['signal'] = 0
    buy_condition = (df['sma_short'] > df['sma_long']) & (df['rsi'] < rsi_overbought)
    sell_condition = (df['sma_short'] < df['sma_long']) | (df['rsi'] > rsi_oversold)

    df.loc[buy_condition, 'signal'] = 1
    df.loc[sell_condition, 'signal'] = -1

    # 打印信号统计
    print(f"\n信号统计:")
    print(f"  买入信号: {(df['signal'] == 1).sum()}")
    print(f"  卖出信号: {(df['signal'] == -1).sum()}")
    print(f"  持有信号: {(df['signal'] == 0).sum()}")

    # 回测
    capital = 100000
    position = 0
    equity_curve = []
    trades = []

    for i in range(1, len(df)):
        price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        prev_signal = df['signal'].iloc[i-1]

        # 买入信号：从0或-1变为1
        if signal == 1 and position == 0:
            position = capital / price
            trades.append({
                'date': str(df['datetime'].iloc[i])[:10],
                'action': '买入',
                'price': price,
                'shares': position
            })
        # 卖出信号：从1变为-1或0
        elif signal <= 0 and position > 0:
            capital = position * price
            trades.append({
                'date': str(df['datetime'].iloc[i])[:10],
                'action': '卖出',
                'price': price,
                'shares': position
            })
            position = 0

        equity = position * price if position > 0 else capital
        equity_curve.append(equity)

    total_return = (equity_curve[-1] - 100000) / 100000
    annual_return = (1 + total_return) ** (365 / len(df)) - 1

    equity_values = pd.Series(equity_curve)
    daily_returns = equity_values.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    cummax = equity_values.cummax()
    drawdown = (equity_values - cummax) / cummax
    max_drawdown = drawdown.min()

    print(f"\n优化后策略回测结果:")
    print(f"  总收益: {total_return*100:+.2f}%")
    print(f"  年化收益: {annual_return*100:+.2f}%")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown*100:.2f}%")
    print(f"  交易次数: {len(trades)}")

    results['optimize_params'] = True

else:
    print(f"\n❌ 参数优化失败")
    results['optimize_params'] = False


# ============================================
# 任务3: 充分模拟验证 (3个月连续测试)
# ============================================
print("\n" + "="*70)
print("任务3: 充分模拟验证 (3个月连续测试)")
print("="*70)

if len(df) < 500:
    print(f"\n❌ 数据不足，无法进行3个月验证")
    results['continuous_test'] = False
else:
    print(f"\n数据充足，进行3个月连续验证...")
    print(f"  总交易日: {len(df)}")
    print(f"  验证周期: 3个月 (约60个交易日)")
    print(f"  测试次数: {len(df) - 60} 次")

    # 滚动窗口验证
    window_size = 60  # 约3个月
    test_results = []

    print(f"\n执行滚动窗口验证...")

    for start_idx in range(len(df) - window_size):
        end_idx = start_idx + window_size

        # 获取窗口数据
        window_data = df.iloc[start_idx:end_idx].copy()

        # 计算指标
        window_data['sma_short'] = sma(window_data['close_price'], ma_short)
        window_data['sma_long'] = sma(window_data['close_price'], ma_long)
        window_data['rsi'] = rsi(window_data['close_price'], 14)

        # 确保指标计算完成
        window_data['sma_short'] = window_data['sma_short'].fillna(method='ffill')
        window_data['sma_long'] = window_data['sma_long'].fillna(method='ffill')
        window_data['rsi'] = window_data['rsi'].fillna(50)

        # 生成信号
        window_data['signal'] = 0
        buy_condition = (window_data['sma_short'] > window_data['sma_long']) & \
                        (window_data['rsi'] < rsi_overbought)
        sell_condition = (window_data['sma_short'] < window_data['sma_long']) | \
                         (window_data['rsi'] > rsi_oversold)

        window_data.loc[buy_condition, 'signal'] = 1
        window_data.loc[sell_condition, 'signal'] = -1

        # 回测
        capital = 100000
        position = 0
        equity_curve = []

        for i in range(1, len(window_data)):
            price = window_data['close'].iloc[i]
            signal = window_data['signal'].iloc[i]
            prev_signal = window_data['signal'].iloc[i-1]

            # 买入信号：从0或-1变为1
            if signal == 1 and position == 0:
                position = capital / price
            # 卖出信号：从1变为-1或0
            elif signal <= 0 and position > 0:
                capital = position * price
                position = 0

            equity = position * price if position > 0 else capital
            equity_curve.append(equity)

        total_return = (equity_curve[-1] - 100000) / 100000
        annual_return = (1 + total_return) ** (365 / len(window_data)) - 1

        equity_values = pd.Series(equity_curve)
        daily_returns = equity_values.pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        cummax = equity_values.cummax()
        drawdown = (equity_values - cummax) / cummax
        max_drawdown = drawdown.min()

        test_results.append({
            'start_date': str(window_data['datetime'].iloc[0])[:10],
            'end_date': str(window_data['datetime'].iloc[-1])[:10],
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        })

    # 统计结果
    test_df = pd.DataFrame(test_results)

    print(f"\n{'='*70}")
    print("3个月连续验证结果")
    print(f"{'='*70}")

    print(f"\n测试统计:")
    print(f"  测试次数: {len(test_df)}")
    print(f"  总收益: 均值={test_df['total_return'].mean()*100:+.2f}%, 标准差={test_df['total_return'].std()*100:.2f}%")
    print(f"  年化收益: 均值={test_df['annual_return'].mean()*100:+.2f}%, 标准差={test_df['annual_return'].std()*100:.2f}%")
    print(f"  夏普比率: 均值={test_df['sharpe_ratio'].mean():.2f}, 标准差={test_df['sharpe_ratio'].std():.2f}")
    print(f"  最大回撤: 均值={test_df['max_drawdown'].mean()*100:.2f}%, 标准差={test_df['max_drawdown'].std()*100:.2f}%")

    # 稳定性分析
    print(f"\n稳定性分析:")
    positive_returns = (test_df['total_return'] > 0).sum()
    print(f"  正收益窗口: {positive_returns}/{len(test_df)} ({positive_returns/len(test_df)*100:.1f}%)")

    high_sharpe = (test_df['sharpe_ratio'] > 1.0).sum()
    print(f"  夏普比率>1.0: {high_sharpe}/{len(test_df)} ({high_sharpe/len(test_df)*100:.1f}%)")

    controlled_dd = (test_df['max_drawdown'] > -0.15).sum()
    print(f"  回撤<15%: {controlled_dd}/{len(test_df)} ({controlled_dd/len(test_df)*100:.1f}%)")

    # 最佳和最差表现
    print(f"\n最佳表现:")
    best_row = test_df.loc[test_df['total_return'].idxmax()]
    print(f"  时间: {best_row['start_date']} -> {best_row['end_date']}")
    print(f"  总收益: {best_row['total_return']*100:+.2f}%")
    print(f"  夏普比率: {best_row['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {best_row['max_drawdown']*100:.2f}%")

    print(f"\n最差表现:")
    worst_row = test_df.loc[test_df['total_return'].idxmin()]
    print(f"  时间: {worst_row['start_date']} -> {worst_row['end_date']}")
    print(f"  总收益: {worst_row['total_return']*100:+.2f}%")
    print(f"  夏普比率: {worst_row['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {worst_row['max_drawdown']*100:.2f}%")

    # 策略评级
    print(f"\n策略稳定性评级:")

    grade = 'C'
    if positive_returns / len(test_df) > 0.7 and high_sharpe / len(test_df) > 0.6:
        grade = 'A'
    elif positive_returns / len(test_df) > 0.6 and high_sharpe / len(test_df) > 0.5:
        grade = 'B'

    if grade == 'A':
        print(f"  🏆 等级: A (优秀)")
        print(f"     策略在各种市场环境下表现稳定，可以考虑实盘")
    elif grade == 'B':
        print(f"  ✅ 等级: B (良好)")
        print(f"     策略表现较好，但仍有改进空间")
    else:
        print(f"  ⚠️  等级: C (一般)")
        print(f"     策略稳定性不足，建议继续优化或调整参数")

    results['continuous_test'] = True


# ============================================
# 总结
# ============================================
print("\n" + "="*70)
print("深度验证总结")
print("="*70)

print(f"\n任务完成情况:")
tasks = {
    'get_long_data': '1. 获取更长时间数据',
    'optimize_params': '2. 策略参数优化',
    'continuous_test': '3. 充分模拟验证'
}

for key, task_name in tasks.items():
    status = "✅ 完成" if results.get(key) else "❌ 未完成"
    print(f"  {status} {task_name}")

completed = sum(results.values())
total = len(results)

print(f"\n总体进度: {completed}/{total} ({completed/total*100:.0f}%)")

if completed == total:
    print(f"\n🎉 所有深度验证任务完成!")

    print(f"\n关键发现:")
    if results.get('get_long_data'):
        print(f"  ✓ 获取了{len(df)}天的历史数据")
    if results.get('optimize_params'):
        print(f"  ✓ 找到最优参数: MA({ma_short}/{ma_long}), RSI({rsi_overbought}/{rsi_oversold})")
        print(f"  ✓ 优化后夏普比率: {sharpe_ratio:.2f}")
    if results.get('continuous_test'):
        positive_rate = (test_df['total_return'] > 0).sum() / len(test_df) * 100
        print(f"  ✓ {positive_rate:.1f}%的3个月窗口实现正收益")

    print(f"\n下一步建议:")
    if grade == 'A':
        print(f"  ✅ 策略表现优秀，可以进入模拟盘实时验证")
        print(f"  ✅ 准备实盘测试计划（小资金1-2%）")
    elif grade == 'B':
        print(f"  ⚠️  继续优化策略参数")
        print(f"  ⚠️  尝试不同的技术指标组合")
        print(f"  ⚠️  延长验证时间到6个月")
    else:
        print(f"  ❌ 策略稳定性不足")
        print(f"  ❌ 重新考虑策略逻辑")
        print(f"  ❌ 尝试完全不同的策略类型")

else:
    print(f"\n⚠️  部分任务未完成")

print("\n" + "="*70)

sys.exit(0 if all(results.values()) else 1)
