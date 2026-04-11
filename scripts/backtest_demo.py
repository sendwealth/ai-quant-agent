"""完整回测演示 — 使用模拟数据"""

import numpy as np
import pandas as pd
import logging

from quant_agent.backtest.engine import BacktestEngine, CommissionModel, SlippageModel
from quant_agent.strategy.indicators import rsi, macd, ema, bollinger_bands
from quant_agent.observability.metrics import MetricsCollector

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_market_data(
    stock_code: str,
    days: int = 500,
    start_price: float = 50.0,
    trend: float = 0.0003,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """生成模拟行情数据"""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2024-01-01", periods=days, freq="B")

    # 几何布朗运动
    returns = rng.normal(trend, volatility, days)
    prices = start_price * np.exp(np.cumsum(returns))

    # 添加均值回归和周期
    cycle = 3 * np.sin(np.linspace(0, 8 * np.pi, days))
    prices *= (1 + cycle / 100)

    high = prices * (1 + rng.uniform(0.005, 0.025, days))
    low = prices * (1 - rng.uniform(0.005, 0.025, days))
    volume = rng.uniform(500000, 3000000, days).astype(int)

    df = pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "open": prices * (1 + rng.uniform(-0.005, 0.005, days)),
        "high": high,
        "low": low,
        "close": prices,
        "volume": volume,
    })
    return df


def generate_signals(df: pd.DataFrame, strategy: str = "dual_ma") -> pd.Series:
    """生成交易信号"""
    close = df["close"]

    if strategy == "dual_ma":
        # 双均线策略
        ema_fast = ema(close, 20)
        ema_slow = ema(close, 50)
        signals = pd.Series(0, index=range(len(close)))
        signals[ema_fast > ema_slow] = 1
        signals[ema_fast < ema_slow] = -1
        return signals

    elif strategy == "rsi_reversal":
        # RSI 反转策略
        rsi_val = rsi(close, 14)
        signals = pd.Series(0, index=range(len(close)))
        signals[rsi_val < 30] = 1    # 超卖买入
        signals[rsi_val > 70] = -1   # 超买卖出
        return signals

    elif strategy == "bollinger":
        # 布林带策略
        upper, middle, lower = bollinger_bands(close, 20, 2.0)
        signals = pd.Series(0, index=range(len(close)))
        signals[close < lower] = 1   # 触下轨买入
        signals[close > upper] = -1  # 触上轨卖出
        return signals

    else:
        # 买入持有
        return pd.Series(1, index=range(len(close)))


def run_backtest(
    stock_code: str = "300750",
    strategy: str = "dual_ma",
    days: int = 500,
    initial_capital: float = 100000.0,
):
    """运行完整回测"""
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=CommissionModel(commission_rate=0.0003, stamp_tax_rate=0.001),
        slippage=SlippageModel(basis_points=1.0),
    )

    # 生成数据
    logger.info(f"📊 生成 {stock_code} 模拟数据 ({days}天, 策略: {strategy})")
    df = generate_market_data(stock_code, days)
    signals = generate_signals(df, strategy)

    # 运行回测
    result = engine.run(df, signals)

    # 输出结果
    print("\n" + "=" * 60)
    print(f"🦞 AI Quant Agent v3.0 — 回测报告")
    print("=" * 60)
    print(f"  股票: {stock_code}")
    print(f"  策略: {strategy}")
    print(f"  周期: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]} ({days}交易日)")
    print(f"  初始资金: {initial_capital:,.0f}")
    print("-" * 60)
    print(f"  📈 总收益: {result.total_return:.2%}")
    print(f"  📈 年化收益: {result.annual_return:.2%}")
    print(f"  📉 最大回撤: {result.max_drawdown:.2%}")
    print(f"  ⏱️  最大回撤持续: {result.max_drawdown_duration} 天")
    print(f"  📊 Sharpe Ratio: {result.sharpe_ratio:.3f}")
    print(f"  📊 Sortino Ratio: {result.sortino_ratio:.3f}")
    print(f"  📊 Calmar Ratio: {result.calmar_ratio:.3f}")
    print("-" * 60)
    print(f"  🔄 总交易次数: {result.total_trades}")
    print(f"  ✅ 盈利次数: {result.win_trades}")
    print(f"  ❌ 亏损次数: {result.lose_trades}")
    print(f"  🎯 胜率: {result.win_rate:.2%}")
    print(f"  💰 平均盈利: {result.avg_win:,.0f}")
    print(f"  💸 平均亏损: {result.avg_loss:,.0f}")
    print(f"  📊 盈亏比: {result.profit_factor:.2f}")
    print(f"  🔻 最大连亏: {result.max_consecutive_losses} 次")
    print("-" * 60)

    final_equity = result.equity_curve[-1] if result.equity_curve else initial_capital
    print(f"  💵 最终权益: {final_equity:,.0f}")
    print(f"  💵 净利润: {final_equity - initial_capital:,.0f}")
    print("=" * 60)

    return result


if __name__ == "__main__":
    strategies = {
        "双均线 (EMA20/50)": "dual_ma",
        "RSI 反转": "rsi_reversal",
        "布林带": "bollinger",
        "买入持有": "buy_hold",
    }

    results = {}
    for name, strategy in strategies.items():
        print(f"\n{'━' * 60}")
        print(f"  策略: {name}")
        print(f"{'━' * 60}")
        result = run_backtest(strategy=strategy)
        results[name] = result

    # 策略对比
    print("\n\n" + "=" * 60)
    print("📋 策略对比")
    print("=" * 60)
    print(f"{'策略':<16} {'总收益':>8} {'年化':>8} {'最大回撤':>8} {'Sharpe':>7} {'胜率':>6}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<16} {r.total_return:>7.2%} {r.annual_return:>7.2%} {r.max_drawdown:>7.2%} "
              f"{r.sharpe_ratio:>7.3f} {r.win_rate:>5.1%}")

    # 最优策略
    best = max(results.items(), key=lambda x: x[1].sharpe_ratio)
    print(f"\n🏆 最优策略 (Sharpe): {best[0]}")
    print("=" * 60)
