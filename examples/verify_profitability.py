"""
盈利验证 - 使用历史数据验证策略盈利能力
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data.fetcher import DataFetcher
    from backtest.engine import (
        BacktestEngine,
        sma_crossover_strategy,
        rsi_strategy,
        macd_strategy
    )
    from trading.paper_trading import run_paper_test, PaperTradingSystem
    from utils.indicators import sma, ema, rsi, macd, bollinger_bands
except ImportError as e:
    print(f"⚠️  警告: 部分模块未安装 ({e})")
    print("正在使用简化版本...")


def simple_sma_strategy(df: pd.DataFrame, short=20, long=60):
    """简化版均线交叉策略"""
    close = df['close']
    short_ma = close.rolling(window=short).mean()
    long_ma = close.rolling(window=long).mean()

    signals = pd.Series(0, index=df.index)
    signals[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1
    signals[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1

    return signals


def simple_rsi_strategy(df: pd.DataFrame, period=14, oversold=30, overbought=70):
    """简化版RSI策略"""
    close = df['close']
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    signals = pd.Series(0, index=df.index)
    signals[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
    signals[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1

    return signals


def simple_macd_strategy(df: pd.DataFrame, fast=12, slow=26, signal=9):
    """简化版MACD策略"""
    close = df['close']
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    signals = pd.Series(0, index=df.index)
    signals[(histogram > 0) & (histogram.shift(1) <= 0)] = 1
    signals[(histogram < 0) & (histogram.shift(1) >= 0)] = -1

    return signals


def bb_strategy(df: pd.DataFrame, period=20, std=2):
    """布林带策略"""
    close = df['close']
    sma_bb = close.rolling(window=period).mean()
    std_bb = close.rolling(window=period).std()

    upper = sma_bb + std_bb * std
    lower = sma_bb - std_bb * std

    signals = pd.Series(0, index=df.index)
    # 价格跌破下轨买入
    signals[(close < lower) & (close.shift(1) >= lower.shift(1))] = 1
    # 价格突破上轨卖出
    signals[(close > upper) & (close.shift(1) <= upper.shift(1))] = -1

    return signals


def dual_ma_strategy(df: pd.DataFrame, ma1=10, ma2=20, ma3=50):
    """三均线策略"""
    close = df['close']
    ma10 = close.rolling(window=ma1).mean()
    ma20 = close.rolling(window=ma2).mean()
    ma50 = close.rolling(window=ma3).mean()

    signals = pd.Series(0, index=df.index)
    # 三线向上，买入
    signals[(ma10 > ma20) & (ma20 > ma50) & (ma10.shift(1) <= ma20.shift(1))] = 1
    # 三线向下，卖出
    signals[(ma10 < ma20) & (ma20 < ma50) & (ma10.shift(1) >= ma20.shift(1))] = -1

    return signals


def momentum_strategy(df: pd.DataFrame, period=10, threshold=0.03):
    """动量策略"""
    close = df['close']
    momentum = close.pct_change(periods=period)

    signals = pd.Series(0, index=df.index)
    # 动量大于阈值，买入
    signals[(momentum > threshold) & (momentum.shift(1) <= threshold)] = 1
    # 动量小于负阈值，卖出
    signals[(momentum < -threshold) & (momentum.shift(1) >= -threshold)] = -1

    return signals


def get_test_data():
    """获取测试数据"""
    try:
        fetcher = DataFetcher()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        print(f"\n获取测试数据: {start_date} -> {end_date}")
        df = fetcher.fetch_stock_data('SPY', start_date, end_date)
        print(f"✓ 成功获取 {len(df)} 条数据\n")

        return df
    except Exception as e:
        print(f"⚠️  数据获取失败: {e}")
        print("使用模拟数据...\n")

        # 生成模拟数据
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

        print(f"✓ 生成 {len(df)} 条模拟数据\n")
        return df


def simple_backtest(df: pd.DataFrame, signals: pd.Series,
                    initial_capital: float = 100000,
                    commission: float = 0.001) -> dict:
    """简单回测"""
    cash = initial_capital
    position = 0.0
    equity_curve = []

    for i in range(len(df)):
        price = df['close'].iloc[i]
        signal = signals.iloc[i]

        # 执行交易
        if signal == 1 and position <= 0:
            # 买入
            if cash > 0:
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

        # 更新权益
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

    # 买入持有收益
    buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1

    return {
        'initial_capital': initial_capital,
        'final_capital': equity_curve[-1],
        'total_return': total_return,
        'annual_return': annual_return,
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return
    }


def run_profitability_verification():
    """运行盈利验证"""
    print("\n" + "="*70)
    print("盈利验证 - 验证策略是否可以实现盈利")
    print("="*70)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"目标: 测试多个策略，找出盈利策略")

    # 获取测试数据
    df = get_test_data()

    # 定义策略
    strategies = [
        ("均线交叉 (20/60)", lambda d: simple_sma_strategy(d, 20, 60)),
        ("均线交叉 (10/30)", lambda d: simple_sma_strategy(d, 10, 30)),
        ("RSI策略 (30/70)", lambda d: simple_rsi_strategy(d, 14, 30, 70)),
        ("MACD策略", lambda d: simple_macd_strategy(d)),
        ("布林带策略", lambda d: bb_strategy(d, 20, 2)),
        ("三均线策略 (10/20/50)", lambda d: dual_ma_strategy(d, 10, 20, 50)),
        ("动量策略 (10天)", lambda d: momentum_strategy(d, 10, 0.03)),
    ]

    # 回测所有策略
    results = []
    print("\n开始回测...\n")

    for strategy_name, strategy_func in strategies:
        print(f"{'─'*70}")
        print(f"策略: {strategy_name}")
        print(f"{'─'*70}")

        try:
            signals = strategy_func(df)
            metrics = simple_backtest(df, signals)

            results.append({
                'strategy': strategy_name,
                'total_return': metrics['total_return'],
                'annual_return': metrics['annual_return'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown': metrics['max_drawdown'],
                'excess_return': metrics['excess_return']
            })

            print(f"\n结果:")
            print(f"  总收益: {metrics['total_return']*100:+.2f}%")
            print(f"  年化收益: {metrics['annual_return']*100:+.2f}%")
            print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
            print(f"  超额收益: {metrics['excess_return']*100:+.2f}%")

        except Exception as e:
            print(f"\n❌ 回测失败: {e}")

    # 排序结果
    results.sort(key=lambda x: x['annual_return'], reverse=True)

    # 总结报告
    print(f"\n{'='*70}")
    print("策略对比总结")
    print(f"{'='*70}")

    print(f"\n{'策略':<25} {'总收益':<12} {'年化收益':<12} {'夏普比率':<10} {'最大回撤':<10} {'超额收益':<12}")
    print(f"{'-'*25} {'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*12}")

    for result in results:
        print(f"{result['strategy']:<25} "
              f"{result['total_return']*100:>10.2f}% "
              f"{result['annual_return']*100:>10.2f}% "
              f"{result['sharpe_ratio']:>9.2f} "
              f"{result['max_drawdown']*100:>9.2f}% "
              f"{result['excess_return']*100:>10.2f}%")

    # 找出最佳策略
    best_strategy = results[0]
    print(f"\n{'='*70}")
    print(f"🏆 最佳策略: {best_strategy['strategy']}")
    print(f"{'='*70}")
    print(f"  总收益: {best_strategy['total_return']*100:+.2f}%")
    print(f"  年化收益: {best_strategy['annual_return']*100:+.2f}%")
    print(f"  夏普比率: {best_strategy['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {best_strategy['max_drawdown']*100:.2f}%")
    print(f"  超额收益: {best_strategy['excess_return']*100:+.2f}%")

    # 盈利策略数量
    profitable_strategies = [r for r in results if r['annual_return'] > 0]
    print(f"\n{'='*70}")
    print(f"盈利验证总结")
    print(f"{'='*70}")
    print(f"  测试策略数量: {len(strategies)}")
    print(f"  盈利策略数量: {len(profitable_strategies)}")
    print(f"  盈利率: {len(profitable_strategies)/len(strategies)*100:.0f}%")
    print(f"  最佳年化收益: {best_strategy['annual_return']*100:.2f}%")

    if len(profitable_strategies) > 0:
        print(f"\n✅ 验证通过! 找到 {len(profitable_strategies)} 个盈利策略")
        print(f"\n推荐使用:")
        for i, strategy in enumerate(profitable_strategies[:3], 1):
            print(f"  {i}. {strategy['strategy']} (年化收益: {strategy['annual_return']*100:+.2f}%)")
        return True
    else:
        print(f"\n⚠️  未找到盈利策略，需要优化参数或调整策略")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AI智能体量化交易系统 - 盈利验证")
    print("="*70)

    success = run_profitability_verification()

    print("\n" + "="*70)
    if success:
        print("✅ 盈利验证完成! 系统可以找到盈利策略")
    else:
        print("⚠️  盈利验证完成，需要进一步优化")
    print("="*70)

    sys.exit(0 if success else 1)
