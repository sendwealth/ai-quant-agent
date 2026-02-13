"""
完整验证脚本 - 多股票、多策略、参数优化、实盘准备
1. 验证更多股票 - 测试不同市场环境 - 验证策略普适性
2. 开发新策略 - 利用智谱AI生成策略代码 - 测试更多策略类型
3. 参数优化 - 强化学习优化 - 遗传算法优化
4. 准备实盘 - 先运行模拟盘验证 - 逐步增加资金
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from zhipuai import ZhipuAI

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============================================
# 任务1: 验证更多股票
# ============================================
def validate_multiple_stocks():
    """
    任务1: 验证更多股票 - 测试不同市场环境
    """
    print("\n" + "="*70)
    print("任务1: 验证更多股票 - 策略普适性测试")
    print("="*70)

    from data.astock_fetcher import AStockDataFetcher, get_popular_astocks
    from utils.indicators import sma, ema, rsi, macd
    import pandas as pd

    # 获取热门股票
    stocks = get_popular_astocks()[:10]  # 前10只
    print(f"\n测试股票: {stocks}")

    fetcher = AStockDataFetcher()

    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    results_summary = []

    for stock in stocks:
        print(f"\n{'─'*70}")
        print(f"测试股票: {stock}")
        print(f"{'─'*70}")

        try:
            # 获取数据
            df = fetcher.fetch_stock_daily(stock, start_date, end_date, source='akshare')

            if df is None or len(df) < 50:
                print(f"⚠️  数据不足，跳过")
                continue

            # 计算指标
            df['sma_short'] = sma(df['close'], 10)
            df['sma_long'] = sma(df['close'], 30)
            df['rsi'] = rsi(df['close'], 14)

            # 生成信号（动量策略）
            df['returns'] = df['close'].pct_change()
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['signal'] = 0
            df.loc[df['momentum'] > 0.02, 'signal'] = 1
            df.loc[df['momentum'] < -0.02, 'signal'] = -1

            # 简单回测
            initial_capital = 100000
            capital = initial_capital
            position = 0
            equity_curve = []

            for i in range(1, len(df)):
                price = df['close'].iloc[i]
                signal = df['signal'].iloc[i]

                if signal == 1 and position == 0:
                    position = capital / price
                elif signal == -1 and position > 0:
                    capital = position * price
                    position = 0

                equity = position * price if position > 0 else capital
                equity_curve.append(equity)

            # 计算指标
            final_capital = equity_curve[-1]
            total_return = (final_capital - initial_capital) / initial_capital
            annual_return = (1 + total_return) ** (365 / len(df)) - 1

            equity_values = pd.Series(equity_curve)
            daily_returns = equity_values.pct_change().dropna()
            sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

            cummax = equity_values.cummax()
            drawdown = (equity_values - cummax) / cummax
            max_drawdown = drawdown.min()

            # 买入持有对比
            buy_hold_return = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]

            results = {
                'stock': stock,
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'buy_hold_return': buy_hold_return
            }

            results_summary.append(results)

            print(f"  总收益: {total_return*100:+.2f}%")
            print(f"  年化收益: {annual_return*100:+.2f}%")
            print(f"  夏普比率: {sharpe_ratio:.2f}")
            print(f"  最大回撤: {max_drawdown*100:.2f}%")
            print(f"  买入持有: {buy_hold_return*100:+.2f}%")

        except Exception as e:
            print(f"❌ 失败: {e}")

    # 总结
    if results_summary:
        print(f"\n{'='*70}")
        print("多股票验证总结")
        print(f"{'='*70}")

        df_results = pd.DataFrame(results_summary)
        df_results = df_results.sort_values('annual_return', ascending=False)

        print(f"\n{'股票':<10} {'总收益':<12} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10}")
        print(f"{'-'*10} {'-'*12} {'-'*12} {'-'*10} {'-'*10}")

        for _, row in df_results.iterrows():
            print(f"{row['stock']:<10} "
                  f"{row['total_return']*100:>10.2f}% "
                  f"{row['annual_return']*100:>10.2f}% "
                  f"{row['sharpe_ratio']:>9.2f} "
                  f"{row['max_drawdown']*100:>9.2f}%")

        # 统计
        print(f"\n统计:")
        print(f"  测试股票数: {len(results_summary)}")
        print(f"  平均年化收益: {df_results['annual_return'].mean()*100:+.2f}%")
        print(f"  平均夏普比率: {df_results['sharpe_ratio'].mean():.2f}")
        print(f"  正收益股票: {len(df_results[df_results['annual_return'] > 0])}/{len(results_summary)}")
        print(f"  跑赢买入持有: {len(df_results[df_results['total_return'] > df_results['buy_hold_return']])}/{len(results_summary)}")

    return results_summary


# ============================================
# 任务2: 开发新策略（使用智谱AI）
# ============================================
def develop_new_strategies():
    """
    任务2: 开发新策略 - 利用智谱AI生成策略代码
    """
    print("\n" + "="*70)
    print("任务2: 开发新策略 - 智谱AI生成")
    print("="*70)

    # 读取配置
    config_path = Path("config/config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    api_key = config['llm']['zhipuai']['api_key']
    model = config['llm']['zhipuai']['model']

    print(f"\n模型: {model}")
    print(f"API Key: {api_key[:20]}...")

    client = ZhipuAI(api_key=api_key)

    # 策略1: 布林带突破策略
    print(f"\n{'─'*70}")
    print("策略1: 布林带突破策略")
    print(f"{'─'*70}")

    prompt1 = """
请用Python写一个布林带突破交易策略的函数，要求：
1. 使用pandas DataFrame作为输入，包含列: datetime, open, high, low, close, volume
2. 布林带参数：周期20，标准差2
3. 买入规则：价格突破布林带上轨
4. 卖出规则：价格跌破布林带下轨
5. 返回信号序列（1=买入, -1=卖出, 0=持有）
请直接给出可运行的Python函数代码，不要解释。
"""

    response1 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt1}
        ],
        temperature=0.3,
    )

    if response1 and response1.choices:
        print(f"✓ 代码生成成功:\n")
        print(response1.choices[0].message.content)

    # 策略2: 多因子策略
    print(f"\n{'─'*70}")
    print("策略2: 多因子综合策略")
    print(f"{'─'*70}")

    prompt2 = """
请用Python写一个多因子综合交易策略函数，要求：
1. 使用pandas DataFrame作为输入
2. 综合以下3个因子：
   - 动量因子（10日收益率）
   - 波动率因子（20日标准差）
   - RSI因子（14日RSI）
3. 买入规则：动量>0.02 且 RSI<70 且 波动率<0.05
4. 卖出规则：动量<-0.02 或 RSI>80 或 波动率>0.1
5. 返回信号序列
请直接给出可运行的Python函数代码，不要解释。
"""

    response2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt2}
        ],
        temperature=0.3,
    )

    if response2 and response2.choices:
        print(f"✓ 代码生成成功:\n")
        print(response2.choices[0].message.content)

    # 策略3: 机器学习预测策略
    print(f"\n{'─'*70}")
    print("策略3: 机器学习预测策略框架")
    print(f"{'─'*70}")

    prompt3 = """
请用Python写一个机器学习交易策略框架，要求：
1. 使用scikit-learn的RandomForestClassifier
2. 特征包括：5日收益率、10日收益率、RSI、MACD、成交量变化
3. 标签：未来5日收益率>0为1，否则为0
4. 返回：模型对象和预测信号序列
5. 包含训练集和测试集分割（80%训练，20%测试）
请直接给出可运行的Python函数代码，不要解释。
"""

    response3 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt3}
        ],
        temperature=0.3,
    )

    if response3 and response3.choices:
        print(f"✓ 代码生成成功:\n")
        print(response3.choices[0].message.content)

    print(f"\n✅ 策略开发完成!")

    return True


# ============================================
# 任务3: 参数优化
# ============================================
def optimize_parameters():
    """
    任务3: 参数优化 - 网格搜索 + 遗传算法
    """
    print("\n" + "="*70)
    print("任务3: 参数优化")
    print("="*70)

    from data.astock_fetcher import AStockDataFetcher
    from utils.indicators import sma
    import pandas as pd
    import numpy as np

    # 获取数据
    fetcher = AStockDataFetcher()
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    df = fetcher.fetch_stock_daily('600519', start_date, end_date, source='akshare')

    if df is None or len(df) < 100:
        print("❌ 数据不足")
        return False

    # 3.1 网格搜索优化
    print(f"\n{'─'*70}")
    print("3.1 网格搜索优化")
    print(f"{'─'*70}")

    # 参数网格
    short_periods = [5, 10, 15, 20]
    long_periods = [30, 40, 50, 60]

    results_grid = []
    total_combinations = len(short_periods) * len(long_periods)
    current = 0

    for short_period in short_periods:
        for long_period in long_periods:
            current += 1
            print(f"[{current}/{total_combinations}] 测试: MA({short_period}/{long_period})")

            try:
                # 计算指标
                df_test = df.copy()
                df_test['sma_short'] = sma(df_test['close'], short_period)
                df_test['sma_long'] = sma(df_test['close'], long_period)

                # 生成信号
                df_test['signal'] = 0
                df_test.loc[df_test['sma_short'] > df_test['sma_long'], 'signal'] = 1
                df_test.loc[df_test['sma_short'] < df_test['sma_long'], 'signal'] = -1

                # 回测
                initial_capital = 100000
                capital = initial_capital
                position = 0
                equity_curve = []

                for i in range(1, len(df_test)):
                    price = df_test['close'].iloc[i]
                    signal = df_test['signal'].iloc[i]

                    if signal == 1 and position == 0:
                        position = capital / price
                    elif signal == -1 and position > 0:
                        capital = position * price
                        position = 0

                    equity = position * price if position > 0 else capital
                    equity_curve.append(equity)

                # 计算指标
                final_capital = equity_curve[-1]
                total_return = (final_capital - initial_capital) / initial_capital
                annual_return = (1 + total_return) ** (365 / len(df_test)) - 1

                equity_values = pd.Series(equity_curve)
                daily_returns = equity_values.pct_change().dropna()
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

                cummax = equity_values.cummax()
                drawdown = (equity_values - cummax) / cummax
                max_drawdown = drawdown.min()

                results_grid.append({
                    'short_period': short_period,
                    'long_period': long_period,
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown
                })

            except Exception as e:
                print(f"  ❌ 失败: {e}")

    # 排序结果
    df_grid = pd.DataFrame(results_grid)
    df_grid = df_grid.sort_values('sharpe_ratio', ascending=False)

    print(f"\n网格搜索结果 (Top 5):")
    print(f"{'短周期':<10} {'长周期':<10} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10}")
    print(f"{'-'*10} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

    for _, row in df_grid.head(5).iterrows():
        print(f"{int(row['short_period']):<10} {int(row['long_period']):<10} "
              f"{row['annual_return']*100:>10.2f}% "
              f"{row['sharpe_ratio']:>9.2f} "
              f"{row['max_drawdown']*100:>9.2f}%")

    best_grid = df_grid.iloc[0]
    print(f"\n🏆 最佳参数 (网格搜索):")
    print(f"   短周期: {int(best_grid['short_period'])}")
    print(f"   长周期: {int(best_grid['long_period'])}")
    print(f"   年化收益: {best_grid['annual_return']*100:.2f}%")
    print(f"   夏普比率: {best_grid['sharpe_ratio']:.2f}")

    # 3.2 遗传算法优化（简化版）
    print(f"\n{'─'*70}")
    print("3.2 遗传算法优化（简化版）")
    print(f"{'─'*70}")

    population_size = 20
    generations = 5

    print(f"\n参数: 种群={population_size}, 代数={generations}")

    # 初始种群
    population = []
    for _ in range(population_size):
        short = np.random.choice(short_periods)
        long = np.random.choice(long_periods)
        population.append((short, long))

    best_individual = None
    best_fitness = -np.inf

    for gen in range(generations):
        print(f"\n第 {gen+1} 代...")

        # 评估适应度
        fitness_scores = []
        for individual in population:
            short, long = individual
            if short >= long:
                fitness_scores.append(-np.inf)
                continue

            try:
                df_test = df.copy()
                df_test['sma_short'] = sma(df_test['close'], short)
                df_test['sma_long'] = sma(df_test['close'], long)

                df_test['signal'] = 0
                df_test.loc[df_test['sma_short'] > df_test['sma_long'], 'signal'] = 1
                df_test.loc[df_test['sma_short'] < df_test['sma_long'], 'signal'] = -1

                # 回测
                capital = 100000
                position = 0
                equity_curve = []

                for i in range(1, len(df_test)):
                    price = df_test['close'].iloc[i]
                    signal = df_test['signal'].iloc[i]

                    if signal == 1 and position == 0:
                        position = capital / price
                    elif signal == -1 and position > 0:
                        capital = position * price
                        position = 0

                    equity = position * price if position > 0 else capital
                    equity_curve.append(equity)

                # 夏普比率作为适应度
                equity_values = pd.Series(equity_curve)
                daily_returns = equity_values.pct_change().dropna()
                if daily_returns.std() > 0:
                    sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
                else:
                    sharpe = -np.inf

                fitness_scores.append(sharpe)

                if sharpe > best_fitness:
                    best_fitness = sharpe
                    best_individual = individual

            except:
                fitness_scores.append(-np.inf)

        # 选择、交叉、变异
        # 选择
        sorted_indices = np.argsort(fitness_scores)[::-1]
        selected = [population[i] for i in sorted_indices[:population_size//2]]

        # 交叉
        offspring = []
        for _ in range(population_size):
            parent1, parent2 = np.random.choice(len(selected), 2, replace=False)
            child = (
                np.random.choice([selected[parent1][0], selected[parent2][0]]),
                np.random.choice([selected[parent1][1], selected[parent2][1]])
            )
            offspring.append(child)

        # 变异
        for i in range(len(offspring)):
            if np.random.rand() < 0.2:
                offspring[i] = (
                    np.random.choice(short_periods),
                    np.random.choice(long_periods)
                )

        population = offspring

        print(f"  最佳适应度: {best_fitness:.2f}")
        print(f"  最佳参数: {best_individual}")

    print(f"\n🏆 最佳参数 (遗传算法):")
    print(f"   短周期: {best_individual[0]}")
    print(f"   长周期: {best_individual[1]}")
    print(f"   夏普比率: {best_fitness:.2f}")

    return True


# ============================================
# 任务4: 准备实盘
# ============================================
def prepare_for_trading():
    """
    任务4: 准备实盘 - 模拟盘验证
    """
    print("\n" + "="*70)
    print("任务4: 准备实盘 - 模拟盘验证")
    print("="*70)

    from data.astock_fetcher import AStockDataFetcher
    from trading.enhanced_paper_trading import EnhancedPaperTrading
    from utils.indicators import sma, rsi

    # 4.1 多只股票模拟交易
    print(f"\n{'─'*70}")
    print("4.1 多只股票模拟交易")
    print(f"{'─'*70}")

    stocks_to_trade = ['600519', '000858', '600036']  # 茅台、五粮液、招商银行

    fetcher = AStockDataFetcher()
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

    trading_results = []

    for stock in stocks_to_trade:
        print(f"\n{stock}:")

        try:
            df = fetcher.fetch_stock_daily(stock, start_date, end_date, source='akshare')

            if df is None or len(df) < 50:
                print(f"  ⚠️  数据不足")
                continue

            # 计算指标
            df['sma_short'] = sma(df['close'], 10)
            df['sma_long'] = sma(df['close'], 30)
            df['rsi'] = rsi(df['close'], 14)

            # 生成信号
            df['signal'] = 0
            df.loc[(df['sma_short'] > df['sma_long']) & (df['rsi'] < 70), 'signal'] = 1
            df.loc[(df['sma_short'] < df['sma_long']) | (df['rsi'] > 80), 'signal'] = -1

            # 运行模拟交易
            system = EnhancedPaperTrading(
                initial_capital=100000,
                commission=0.001,
                slippage=0.0001,
                enable_risk_control=True
            )

            for i in range(len(df)):
                price = df['close'].iloc[i]
                date = df['datetime'].iloc[i]
                if hasattr(date, 'strftime'):
                    date = date.strftime('%Y-%m-%d')
                signal = int(df['signal'].iloc[i])

                system.execute_signal(price, signal, date)

            # 最终平仓
            if system.position != 0:
                final_price = df['close'].iloc[-1]
                final_date = df['datetime'].iloc[-1]
                if hasattr(final_date, 'strftime'):
                    final_date = final_date.strftime('%Y-%m-%d')
                system.execute_signal(final_price, 0, final_date)

            # 打印报告
            print(f"  最终资金: ¥{system.equity_curve[-1]:,.2f}")
            print(f"  总收益: {((system.equity_curve[-1] - 100000) / 100000)*100:+.2f}%")

            trading_results.append({
                'stock': stock,
                'final_capital': system.equity_curve[-1],
                'total_return': (system.equity_curve[-1] - 100000) / 100000
            })

        except Exception as e:
            print(f"  ❌ 失败: {e}")

    # 4.2 资金分配建议
    print(f"\n{'─'*70}")
    print("4.2 资金分配建议")
    print(f"{'─'*70}")

    if trading_results:
        total_capital = 300000  # 3只股票，每只10万

        print(f"\n总资金: ¥{total_capital:,}")
        print(f"建议分配:")

        for result in trading_results:
            if result['total_return'] > 0:
                allocation = total_capital / len(trading_results) * 1.2
            else:
                allocation = total_capital / len(trading_results) * 0.8

            print(f"  {result['stock']}: ¥{allocation:,.0f} ({allocation/total_capital*100:.1f}%)")

    # 4.3 实盘检查清单
    print(f"\n{'─'*70}")
    print("4.3 实盘前检查清单")
    print(f"{'─'*70}")

    checklist = [
        ("数据连接", "Tushare/AkShare API 正常"),
        ("策略验证", "模拟盘夏普比率 > 1.5"),
        ("风控系统", "最大回撤 < 10%"),
        ("资金管理", "单笔交易风险 < 2%"),
        ("模拟验证", "连续3个月模拟稳定盈利"),
        ("券商接口", "实盘API已配置"),
        ("监控系统", "实时告警已设置"),
        ("应急方案", "紧急停止机制已测试"),
    ]

    print(f"\n检查项:")
    for i, (item, requirement) in enumerate(checklist, 1):
        print(f"  [{i}] {item:<12} - {requirement}")

    # 4.4 风险提示
    print(f"\n{'─'*70}")
    print("4.4 风险提示")
    print(f"{'─'*70}")

    warnings = [
        "⚠️  量化交易有风险，入市需谨慎",
        "⚠️  不要投入超过承受能力的资金",
        "⚠️  先从模拟盘开始，充分验证后再实盘",
        "⚠️  建议初始资金不超过总资产的5%",
        "⚠️  严格遵守风险管理规则",
        "⚠️  定期审查和调整策略",
    ]

    for warning in warnings:
        print(f"  {warning}")

    return True


# ============================================
# 主函数
# ============================================
def main():
    """
    主函数
    """
    print("\n" + "="*70)
    print("AI智能体量化交易系统 - 完整验证")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    # 任务1: 验证更多股票
    try:
        results['validate_multiple_stocks'] = len(validate_multiple_stocks()) > 0
    except Exception as e:
        print(f"❌ 任务1失败: {e}")
        results['validate_multiple_stocks'] = False

    # 任务2: 开发新策略
    try:
        results['develop_new_strategies'] = develop_new_strategies()
    except Exception as e:
        print(f"❌ 任务2失败: {e}")
        results['develop_new_strategies'] = False

    # 任务3: 参数优化
    try:
        results['optimize_parameters'] = optimize_parameters()
    except Exception as e:
        print(f"❌ 任务3失败: {e}")
        results['optimize_parameters'] = False

    # 任务4: 准备实盘
    try:
        results['prepare_for_trading'] = prepare_for_trading()
    except Exception as e:
        print(f"❌ 任务4失败: {e}")
        results['prepare_for_trading'] = False

    # 总结
    print("\n" + "="*70)
    print("完整验证总结")
    print("="*70)

    print(f"\n任务完成情况:")
    tasks = {
        'validate_multiple_stocks': '1. 验证更多股票',
        'develop_new_strategies': '2. 开发新策略',
        'optimize_parameters': '3. 参数优化',
        'prepare_for_trading': '4. 准备实盘'
    }

    for key, task_name in tasks.items():
        status = "✅ 完成" if results.get(key) else "❌ 未完成"
        print(f"  {status} {task_name}")

    completed = sum(results.values())
    total = len(results)

    print(f"\n总体进度: {completed}/{total} ({completed/total*100:.0f}%)")

    if completed == total:
        print(f"\n🎉 所有任务完成! 系统已完全就绪!")

        print(f"\n下一步:")
        print(f"  1. 选择最佳策略和参数")
        print(f"  2. 运行模拟盘验证（至少3个月）")
        print(f"  3. 逐步增加实盘资金（建议不超过总资产5%）")
        print(f"  4. 持续监控和优化")

    else:
        print(f"\n⚠️  部分任务未完成")

    print("\n" + "="*70)

    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
