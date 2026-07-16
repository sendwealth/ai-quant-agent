"""Backtest engine -- day-by-day trade simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.gate import DataTrustError, evaluate_trust
from ..portfolio import CommissionModel, Portfolio, Trade, round_shares
from ..strategy.strategy import Strategy, StrategyContext
from .walk_forward import validate_point_in_time

logger = logging.getLogger(__name__)


@dataclass
class SlippageModel:
    """Slippage model."""

    basis_points: float = 1.0  # basis points of slippage

    def apply(self, price: float, direction: str) -> float:
        slip = price * self.basis_points / 10000
        return price + slip if direction == "buy" else price - slip


@dataclass
class BacktestResult:
    """Backtest result."""

    # Basic metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0

    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    win_trades: int = 0
    lose_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0

    # Benchmark
    benchmark_return: float = 0.0
    alpha: float = 0.0
    beta: float = 0.0

    # Raw data
    equity_curve: list[float] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)

    # 可信度审计（推荐 #4）：记录回测所依赖的数据前提，便于复盘与监管
    adjust: str = "qfq"  # 复权方式：qfq（前复权）/ hfq（后复权）/ none
    point_in_time_issues: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Total return: {self.total_return:.2%} | Annualized: {self.annual_return:.2%} | "
            f"Max drawdown: {self.max_drawdown:.2%} | Sharpe: {self.sharpe_ratio:.2f} | "
            f"Win rate: {self.win_rate:.2%} | Profit factor: {self.profit_factor:.2f} | "
            f"Trades: {self.total_trades}"
        )


class BacktestEngine:
    """Backtest engine."""

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: CommissionModel | None = None,
        slippage: SlippageModel | None = None,
        strategy: Strategy | None = None,
        enforce_t_plus_one: bool = False,
        adjust: str = "qfq",
    ):
        self.initial_capital = initial_capital
        self.commission = commission or CommissionModel()
        self.slippage = slippage or SlippageModel()
        self.strategy = strategy
        # P1.4: T+1 规则 — 当日买入的股票次日才能卖出（默认关闭以兼容旧回测）
        self.enforce_t_plus_one = enforce_t_plus_one
        # 复权方式（推荐 #4）：记录回测所依赖的价格基准，便于可信度审计
        self.adjust = adjust

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.Series | None = None,
        benchmark: pd.Series | None = None,
        strategy: Strategy | None = None,
        enforce_t_plus_one: bool | None = None,
        provenance: list | None = None,
        adjust: str | None = None,
        research_mode: bool = False,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            price_data: OHLCV data (columns: date, close, ...)
            signals: Signal series (1=buy, -1=sell, 0=hold), aligned to price_data.
                     Ignored when a *strategy* is provided.
            benchmark: Benchmark return series (optional)
            strategy: Optional :class:`Strategy` that generates signals per day.
                      When given, the backtest reuses the SAME strategy abstraction
                      as the live pipeline, decoupling signal logic from the
                      trade simulator.  (e.g. ``CrossOverStrategy`` for a momentum
                      backtest, or a custom strategy.)
            provenance: 可选数据谱系列表（``DataProvenance``）。命中合成样例/低
                可信度时，回测被硬门禁拦截（推荐 #2），避免受限数据污染回测绩效。
                缺谱系时默认 fail closed（决策用途），仅显式 ``research_mode``
                可豁免。
            research_mode: 显式研究模式。仅当 ``provenance`` 为空时生效：开启后
                允许回测（仅供研究），关闭（默认）则缺谱系即拦截。

        Returns:
            BacktestResult

        Raises:
            DataTrustError: provenance 命中硬门禁，或缺谱系且非研究模式时。
        """
        # 数据可信门禁（推荐 #2）：合成/低可信度数据禁止污染回测绩效；
        # 缺谱系默认 fail closed（决策用途），仅显式研究模式可豁免。
        try:
            evaluate_trust(provenance, "backtest", research_mode=research_mode).require()
        except DataTrustError as e:
            raise DataTrustError(f"回测被数据可信门禁拦截: {e}") from e

        # 复权方式与 point-in-time 校验（推荐 #4）：记录回测前提，作为可信度审计
        adjust_used = adjust or self.adjust
        pit_issues = validate_point_in_time(price_data, signals)
        if pit_issues:
            for issue in pit_issues:
                logger.warning("point-in-time 校验告警: %s", issue)

        active_strategy = strategy or self.strategy
        if price_data.empty:
            return BacktestResult(adjust=adjust_used, point_in_time_issues=pit_issues)
        if active_strategy is None and signals is None:
            logger.warning("Backtest requires either `signals` or a `strategy`")
            return BacktestResult(adjust=adjust_used, point_in_time_issues=pit_issues)

        # P1.4: T+1 强制开关（run 级覆盖实例级）
        t_plus_one = self.enforce_t_plus_one if enforce_t_plus_one is None else enforce_t_plus_one

        # Normalize column names
        if "trade_date" in price_data.columns and "date" not in price_data.columns:
            price_data = price_data.rename(columns={"trade_date": "date"})

        portfolio = Portfolio(
            cash=self.initial_capital,
            commission=self.commission,
            auto_round=True,
        )
        stock_code = "STOCK"

        # Align signals and prices
        signal_len = len(signals) if signals is not None else len(price_data)
        min_len = min(len(price_data), signal_len)
        closes = price_data["close"].iloc[:min_len]
        signal_values = (
            signals.iloc[:min_len]
            if (signals is not None and isinstance(signals, pd.Series))
            else None
        )

        prev_signal = 0
        prev_price = None
        has_position = False
        last_buy_index = -2  # 最近买入的 bar 下标（用于 T+1）

        for i in range(min_len):
            price = float(closes.iloc[i])
            portfolio.update_price(stock_code, price)

            # 决定当日信号：优先用注入的 Strategy，否则用 signals 序列
            if active_strategy is not None:
                ctx = StrategyContext(price=price, prev_price=prev_price, has_position=has_position)
                s = active_strategy.generate_signal(ctx)
                sig = 1 if s.signal == "BUY" else (-1 if s.signal == "SELL" else 0)
            else:
                sv = signal_values.iloc[i] if signal_values is not None else 0
                sig = int(sv) if not pd.isna(sv) else 0

            if sig == 1 and prev_signal != 1:
                shares = int(portfolio.cash / price)
                shares = round_shares(shares)  # A-share board lot: multiples of 100
                if shares > 0:
                    exec_price = self.slippage.apply(price, "buy")
                    portfolio.buy(stock_code, exec_price, shares)
                    last_buy_index = i  # 记录买入日（T+1 用）

            elif sig == -1 and prev_signal != -1 and stock_code in portfolio.positions:
                # P1.4: T+1 — 当日买入的股票次日才可卖
                if t_plus_one and i <= last_buy_index:
                    pass  # 跳过：T 日买入不可 T 日卖出
                else:
                    exec_price = self.slippage.apply(price, "sell")
                    pos = portfolio.positions[stock_code]
                    portfolio.sell(stock_code, exec_price, pos.shares)

            prev_signal = sig
            prev_price = price
            has_position = stock_code in portfolio.positions
            portfolio.record_equity()

        # Liquidate remaining positions
        if stock_code in portfolio.positions:
            last_price = float(closes.iloc[-1])
            pos = portfolio.positions[stock_code]
            portfolio.sell(stock_code, last_price, pos.shares)
            portfolio.record_equity()

        result = self._calculate_metrics(portfolio, benchmark)
        result.adjust = adjust_used
        result.point_in_time_issues = pit_issues
        return result

    def _calculate_metrics(
        self, portfolio: Portfolio, benchmark: pd.Series | None = None
    ) -> BacktestResult:
        """Calculate performance metrics."""
        result = BacktestResult(
            equity_curve=portfolio.equity_curve,
            trades=portfolio.trades,
        )

        if not portfolio.equity_curve:
            return result

        equity = pd.Series(portfolio.equity_curve)
        n_days = len(equity)

        # Returns
        result.total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        trading_years = n_days / 252
        result.annual_return = (1 + result.total_return) ** (1 / max(trading_years, 0.01)) - 1

        # Daily returns
        daily_returns = equity.pct_change().dropna()
        if daily_returns.empty:
            return result

        # Volatility and Sharpe
        vol = daily_returns.std() * np.sqrt(252)
        result.sharpe_ratio = result.annual_return / vol if vol > 0 else 0.0

        # Max drawdown
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        result.max_drawdown = drawdown.min()

        # Max drawdown duration
        is_dd = drawdown < 0
        dd_groups = (~is_dd).cumsum()
        dd_durations = is_dd.groupby(dd_groups).sum()
        result.max_drawdown_duration = int(dd_durations.max()) if not dd_durations.empty else 0

        # Sortino (downside deviation only)
        downside = daily_returns[daily_returns < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
        result.sortino_ratio = result.annual_return / downside_vol

        # Calmar
        result.calmar_ratio = (
            result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0.0
        )

        # Trade statistics
        closed = portfolio.closed_trades
        result.total_trades = len(closed)
        if closed:
            wins = [t for t in closed if t.pnl > 0]
            losses = [t for t in closed if t.pnl <= 0]
            result.win_trades = len(wins)
            result.lose_trades = len(losses)
            result.win_rate = len(wins) / len(closed) if closed else 0.0
            result.avg_win = float(np.mean([t.pnl for t in wins])) if wins else 0.0
            result.avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0
            result.profit_factor = (
                sum(t.pnl for t in wins) / abs(sum(t.pnl for t in losses)) if losses else 0.0
            )

            # Max consecutive losses
            max_consec = 0
            current_consec = 0
            for t in closed:
                if t.pnl <= 0:
                    current_consec += 1
                    max_consec = max(max_consec, current_consec)
                else:
                    current_consec = 0
            result.max_consecutive_losses = max_consec

        # Benchmark comparison
        if benchmark is not None and len(benchmark) >= n_days:
            bench_returns = benchmark.iloc[:n_days]
            result.benchmark_return = (bench_returns.iloc[-1] / bench_returns.iloc[0]) - 1

            # Alpha: annualized excess return over benchmark
            bench_annual = (1 + result.benchmark_return) ** (252 / max(n_days, 1)) - 1
            result.alpha = result.annual_return - bench_annual

            bench_daily = bench_returns.pct_change().dropna()
            min_len = min(len(daily_returns), len(bench_daily))
            if min_len > 1:
                cov = np.cov(daily_returns.iloc[:min_len], bench_daily.iloc[:min_len])
                result.beta = cov[0][1] / cov[1][1] if cov[1][1] != 0 else 1.0

        return result
