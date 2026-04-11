"""Backtest engine -- day-by-day trade simulation."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..portfolio import CommissionModel, Portfolio, Trade, round_shares

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
        commission: Optional[CommissionModel] = None,
        slippage: Optional[SlippageModel] = None,
    ):
        self.initial_capital = initial_capital
        self.commission = commission or CommissionModel()
        self.slippage = slippage or SlippageModel()

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """Run backtest.

        Args:
            price_data: OHLCV data (columns: date, close, ...)
            signals: Signal series (1=buy, -1=sell, 0=hold), aligned to price_data
            benchmark: Benchmark return series (optional)

        Returns:
            BacktestResult
        """
        if price_data.empty:
            return BacktestResult()

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
        min_len = min(len(price_data), len(signals))
        dates = price_data["date"].iloc[:min_len]
        closes = price_data["close"].iloc[:min_len]
        signal_values = signals.iloc[:min_len] if isinstance(signals, pd.Series) else signals

        prev_signal = 0

        for i in range(min_len):
            date = str(dates.iloc[i])
            price = float(closes.iloc[i])
            sig = int(signal_values.iloc[i]) if not pd.isna(signal_values.iloc[i]) else 0

            portfolio.update_price(stock_code, price)

            if sig == 1 and prev_signal != 1:
                shares = int(portfolio.cash / price)
                shares = round_shares(shares)  # A-share board lot: multiples of 100
                if shares > 0:
                    exec_price = self.slippage.apply(price, "buy")
                    portfolio.buy(stock_code, exec_price, shares)

            elif sig == -1 and prev_signal != -1 and stock_code in portfolio.positions:
                exec_price = self.slippage.apply(price, "sell")
                pos = portfolio.positions[stock_code]
                portfolio.sell(stock_code, exec_price, pos.shares)

            prev_signal = sig
            portfolio.record_equity()

        # Liquidate remaining positions
        if stock_code in portfolio.positions:
            last_price = float(closes.iloc[-1])
            pos = portfolio.positions[stock_code]
            portfolio.sell(stock_code, last_price, pos.shares)
            portfolio.record_equity()

        return self._calculate_metrics(portfolio, benchmark)

    def _calculate_metrics(
        self, portfolio: Portfolio, benchmark: Optional[pd.Series] = None
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
        result.calmar_ratio = result.annual_return / abs(result.max_drawdown) if result.max_drawdown != 0 else 0.0

        # Trade statistics
        closed = portfolio.closed_trades
        result.total_trades = len(closed)
        if closed:
            wins = [t for t in closed if t.pnl > 0]
            losses = [t for t in closed if t.pnl <= 0]
            result.win_trades = len(wins)
            result.lose_trades = len(losses)
            result.win_rate = len(wins) / len(closed) if closed else 0.0
            result.avg_win = np.mean([t.pnl for t in wins]) if wins else 0.0
            result.avg_loss = np.mean([t.pnl for t in losses]) if losses else 0.0
            result.profit_factor = abs(result.avg_win / result.avg_loss) if result.avg_loss != 0 else 0.0

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
