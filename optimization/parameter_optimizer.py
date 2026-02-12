"""
参数优化模块
使用网格搜索和随机搜索优化策略参数
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Tuple
from datetime import datetime


class ParameterOptimizer:
    """参数优化器"""

    def __init__(self,
                 backtest_func: Callable,
                 metric: str = 'annual_return',
                 n_jobs: int = 1):
        """
        初始化参数优化器

        Args:
            backtest_func: 回测函数，接收(data, signals)返回metrics
            metric: 优化目标指标 ('annual_return', 'sharpe_ratio', 'total_return')
            n_jobs: 并行任务数
        """
        self.backtest_func = backtest_func
        self.metric = metric
        self.n_jobs = n_jobs

    def grid_search(self,
                     df: pd.DataFrame,
                     strategy_func: Callable,
                     param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        网格搜索

        Args:
            df: 历史数据
            strategy_func: 策略函数
            param_grid: 参数网格 {param_name: [values]}

        Returns:
            优化结果DataFrame
        """
        print(f"\n{'='*70}")
        print(f"网格搜索优化")
        print(f"{'='*70}")
        print(f"优化目标: {self.metric}")
        print(f"参数网格: {list(param_grid.keys())}")
        print(f"总组合数: {np.prod([len(v) for v in param_grid.values()])}")

        # 生成所有参数组合
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        results = []

        for i, combination in enumerate(product(*param_values), 1):
            params = dict(zip(param_names, combination))

            try:
                # 生成信号
                signals = strategy_func(df, **params)

                # 回测
                metrics = self.backtest_func(df, signals)

                results.append({
                    **params,
                    'total_return': metrics['total_return'],
                    'annual_return': metrics['annual_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'volatility': metrics['volatility']
                })

                print(f"[{i}/{np.prod([len(v) for v in param_values.values()])}] "
                      f"{params} -> 收益: {metrics['annual_return']*100:.2f}%, "
                      f"夏普: {metrics['sharpe_ratio']:.2f}")

            except Exception as e:
                print(f"[{i}] 参数 {params} 回测失败: {e}")
                continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 排序
        results_df = results_df.sort_values(by=self.metric, ascending=False)

        print(f"\n✓ 网格搜索完成: 测试了 {len(results)} 组参数")
        print(f"✓ 最佳参数: {results_df.iloc[0][param_names].to_dict()}")
        print(f"✓ 最佳{self.metric}: {results_df.iloc[0][self.metric]*100:.2f}%")

        return results_df

    def random_search(self,
                      df: pd.DataFrame,
                      strategy_func: Callable,
                      param_ranges: Dict[str, Tuple],
                      n_iterations: int = 50) -> pd.DataFrame:
        """
        随机搜索

        Args:
            df: 历史数据
            strategy_func: 策略函数
            param_ranges: 参数范围 {param_name: (min, max)}
            n_iterations: 迭代次数

        Returns:
            优化结果DataFrame
        """
        print(f"\n{'='*70}")
        print(f"随机搜索优化")
        print(f"{'='*70}")
        print(f"优化目标: {self.metric}")
        print(f"参数范围: {list(param_ranges.keys())}")
        print(f"迭代次数: {n_iterations}")

        param_names = list(param_ranges.keys())
        results = []

        for i in range(1, n_iterations + 1):
            # 随机生成参数
            params = {}
            for param_name, (min_val, max_val) in param_ranges.items():
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[param_name] = np.random.uniform(min_val, max_val)

            try:
                # 生成信号
                signals = strategy_func(df, **params)

                # 回测
                metrics = self.backtest_func(df, signals)

                results.append({
                    **params,
                    'total_return': metrics['total_return'],
                    'annual_return': metrics['annual_return'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'max_drawdown': metrics['max_drawdown'],
                    'volatility': metrics['volatility']
                })

                print(f"[{i}/{n_iterations}] "
                      f"{params} -> 收益: {metrics['annual_return']*100:.2f}%, "
                      f"夏普: {metrics['sharpe_ratio']:.2f}")

            except Exception as e:
                print(f"[{i}] 参数 {params} 回测失败: {e}")
                continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        # 排序
        results_df = results_df.sort_values(by=self.metric, ascending=False)

        print(f"\n✓ 随机搜索完成: 测试了 {len(results)} 组参数")
        print(f"✓ 最佳参数: {results_df.iloc[0][param_names].to_dict()}")
        print(f"✓ 最佳{self.metric}: {results_df.iloc[0][self.metric]*100:.2f}%")

        return results_df

    def optimize_sma_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化均线策略

        Args:
            df: 历史数据

        Returns:
            优化结果
        """
        from examples.verify_profitability import simple_sma_strategy

        # 参数网格
        param_grid = {
            'short': [5, 10, 15, 20, 25, 30],
            'long': [40, 50, 60, 80, 100]
        }

        print(f"\n{'#'*70}")
        print(f"# 优化均线交叉策略")
        print(f"{'#'*70}")

        return self.grid_search(df, simple_sma_strategy, param_grid)

    def optimize_rsi_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化RSI策略

        Args:
            df: 历史数据

        Returns:
            优化结果
        """
        from examples.verify_profitability import simple_rsi_strategy

        # 参数网格
        param_grid = {
            'period': [10, 12, 14, 16, 20],
            'oversold': [20, 25, 30, 35],
            'overbought': [65, 70, 75, 80]
        }

        print(f"\n{'#'*70}")
        print(f"# 优化RSI策略")
        print(f"{'#'*70}")

        return self.grid_search(df, simple_rsi_strategy, param_grid)

    def optimize_macd_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化MACD策略

        Args:
            df: 历史数据

        Returns:
            优化结果
        """
        from examples.verify_profitability import simple_macd_strategy

        # 参数网格
        param_grid = {
            'fast': [8, 10, 12, 14],
            'slow': [20, 24, 26, 30],
            'signal': [7, 8, 9, 10]
        }

        print(f"\n{'#'*70}")
        print(f"# 优化MACD策略")
        print(f"{'#'*70}")

        return self.grid_search(df, simple_macd_strategy, param_grid)

    def optimize_momentum_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        优化动量策略

        Args:
            df: 历史数据

        Returns:
            优化结果
        """
        from examples.verify_profitability import momentum_strategy

        # 参数网格
        param_grid = {
            'period': [5, 8, 10, 12, 15],
            'threshold': [0.02, 0.03, 0.04, 0.05]
        }

        print(f"\n{'#'*70}")
        print(f"# 优化动量策略")
        print(f"{'#'*70}")

        return self.grid_search(df, momentum_strategy, param_grid)

    def print_optimization_report(self, results_df: pd.DataFrame, top_n: int = 10):
        """
        打印优化报告

        Args:
            results_df: 优化结果
            top_n: 显示前n个
        """
        print(f"\n{'='*70}")
        print(f"参数优化报告")
        print(f"{'='*70}")

        param_cols = [col for col in results_df.columns
                      if col not in ['total_return', 'annual_return', 'sharpe_ratio',
                                    'max_drawdown', 'volatility']]

        print(f"\n最佳 {top_n} 组参数:")
        print(f"{'排名':<6} ", end='')
        for col in param_cols:
            print(f"{col:<12} ", end='')
        print(f"{'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10}")
        print(f"{'-'*70}")

        for i in range(min(top_n, len(results_df))):
            row = results_df.iloc[i]
            print(f"{i+1:<6} ", end='')
            for col in param_cols:
                print(f"{row[col]:<12} ", end='')
            print(f"{row['annual_return']*100:>10.2f}% "
                  f"{row['sharpe_ratio']:>9.2f} "
                  f"{row['max_drawdown']*100:>9.2f}%")

        # 统计
        print(f"\n优化统计:")
        print(f"  测试参数组合: {len(results_df)}")
        print(f"  平均年化收益: {results_df['annual_return'].mean()*100:.2f}%")
        print(f"  最佳年化收益: {results_df['annual_return'].max()*100:.2f}%")
        print(f"  收益率标准差: {results_df['annual_return'].std()*100:.2f}%")


def simple_backtest(df: pd.DataFrame, signals: pd.Series) -> dict:
    """
    简单回测函数

    Args:
        df: 历史数据
        signals: 信号

    Returns:
        性能指标
    """
    initial_capital = 100000
    commission = 0.001

    cash = initial_capital
    position = 0.0
    equity_curve = []

    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = signals.iloc[i] if i < len(signals) else 0

        if signal == 1 and position <= 0 and cash > 0:
            # 买入
            quantity = (cash * (1 - commission)) / price
            cash -= quantity * price
            position += quantity
        elif signal == -1 and position >= 0:
            # 卖出
            if position > 0:
                cash += position * price * (1 - commission)
                position = 0
        elif signal == 0 and position != 0:
            # 平仓
            if position > 0:
                cash += position * price * (1 - commission)
                position = 0

        equity = cash + position * price
        equity_curve.append(equity)

    # 计算指标
    equity_series = pd.Series(equity_curve)
    total_return = (equity_curve[-1] - initial_capital) / initial_capital

    days = len(equity_curve)
    years = days / 252
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    daily_returns = equity_series.pct_change().dropna()
    volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
    sharpe_ratio = annual_return / volatility if volatility > 0 else 0

    peak = equity_series.expanding().max()
    drawdowns = (equity_series - peak) / peak
    max_drawdown = drawdowns.min()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'volatility': volatility
    }


if __name__ == "__main__":
    # 测试参数优化
    print("="*70)
    print("参数优化测试")
    print("="*70)

    # 生成测试数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500)
    base_price = 400

    returns = np.random.normal(0.001, 0.02, 500)
    prices = base_price * (1 + np.cumsum(returns))

    df = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.randn(500) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(500)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(500)) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 500)
    })

    # 创建优化器
    optimizer = ParameterOptimizer(
        backtest_func=simple_backtest,
        metric='annual_return'
    )

    # 优化动量策略
    results_df = optimizer.optimize_momentum_strategy(df)
    optimizer.print_optimization_report(results_df, top_n=10)
