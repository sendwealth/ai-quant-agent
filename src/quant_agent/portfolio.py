"""Portfolio -- shared portfolio state for backtest and paper-trading.

Consolidates the position-management and commission logic that previously
lived in ``backtest.engine.Portfolio`` and ``agents.execution`` into a
single source of truth.

A-share trading rules:
- Commission: 0.03 % (wan san), minimum 5 CNY
- Stamp tax:  0.1 % (qian yi), sell only
- Share rounding: multiples of 100 (A-share board lot)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """A single stock position with stop-loss / take-profit levels."""

    stock_code: str
    shares: int
    avg_price: float
    current_price: float = 0.0
    entry_date: str = ""
    stop_loss: float = 0.0
    take_profit: float = 0.0

    @property
    def pnl(self) -> float:
        if self.avg_price <= 0 or self.shares <= 0:
            return 0.0
        return (self.current_price - self.avg_price) * self.shares

    @property
    def pnl_pct(self) -> float:
        if self.avg_price <= 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price

    def update_price(self, price: float) -> None:
        self.current_price = price

    def should_stop_loss(self) -> bool:
        return self.stop_loss > 0 and self.current_price <= self.stop_loss

    def should_take_profit(self) -> bool:
        return self.take_profit > 0 and self.current_price >= self.take_profit


@dataclass
class Trade:
    """Record of a single trade execution."""

    stock_code: str
    direction: str           # "buy" / "sell"
    entry_date: str = ""
    exit_date: Optional[str] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    shares: int = 0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    status: str = "open"     # "open" / "closed"

    @property
    def is_closed(self) -> bool:
        return self.status == "closed"


# ---------------------------------------------------------------------------
# Commission model
# ---------------------------------------------------------------------------

@dataclass
class CommissionModel:
    """A-share commission and stamp-tax calculator.

    - Commission rate: 0.03 % (wan san), minimum *min_commission* CNY
    - Stamp tax: 0.1 % (qian yi), **sell only**
    """

    commission_rate: float = 0.0003
    stamp_tax_rate: float = 0.001
    min_commission: float = 5.0

    def calc(self, price: float, shares: int, direction: str) -> float:
        amount = price * shares
        commission = max(amount * self.commission_rate, self.min_commission)
        tax = amount * self.stamp_tax_rate if direction == "sell" else 0.0
        return commission + tax


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def round_shares(shares: int) -> int:
    """Round down to the nearest board lot (100 shares) for A-shares."""
    return (shares // 100) * 100


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@dataclass
class Portfolio:
    """Portfolio state -- shared by the backtest engine and ExecutionAgent.

    Parameters
    ----------
    cash:
        Starting cash balance.
    commission:
        A ``CommissionModel`` instance used for fee calculation.
        If *None*, a default A-share model is created.
    auto_round:
        When *True* (the default for paper-trading), buy share counts are
        automatically rounded down to multiples of 100.  Set to *False*
        in backtest mode where the caller controls share rounding.
    """

    cash: float = 100000.0
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    commission: CommissionModel = field(default_factory=CommissionModel)
    auto_round: bool = False

    # -- properties ----------------------------------------------------------

    @property
    def position_value(self) -> float:
        return sum(p.shares * p.current_price for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.position_value

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.is_closed]

    # -- position helpers ----------------------------------------------------

    def get_position(self, stock_code: str) -> Optional[Position]:
        return self.positions.get(stock_code)

    def update_price(self, stock_code: str, price: float) -> None:
        pos = self.positions.get(stock_code)
        if pos is not None:
            pos.update_price(price)

    # -- core buy / sell -----------------------------------------------------

    def buy(
        self,
        stock_code: str,
        price: float,
        shares: int,
        commission: float | None = None,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Trade:
        """Open or add to a position.

        If *commission* is *None*, it is calculated automatically using the
        portfolio's ``CommissionModel``.

        When *auto_round* is *True*, *shares* is rounded down to a multiple
        of 100.  If cash is insufficient the share count is reduced
        accordingly.
        """
        if commission is None:
            commission = self.commission.calc(price, shares, "buy")

        cost = price * shares + commission
        if cost > self.cash:
            shares = int((self.cash - commission) / price)
            if self.auto_round:
                shares = round_shares(shares)
            commission = self.commission.calc(price, shares, "buy")
            cost = price * shares + commission
            if shares <= 0:
                logger.warning("Insufficient funds for %s", stock_code)
                return Trade(
                    stock_code=stock_code,
                    direction="buy",
                    entry_price=price,
                    shares=0,
                )

        self.cash -= cost

        existing = self.positions.get(stock_code)
        if existing is not None:
            total_shares = existing.shares + shares
            existing.avg_price = (
                (existing.avg_price * existing.shares + price * shares)
                / total_shares
            )
            existing.shares = total_shares
            existing.current_price = price
        else:
            self.positions[stock_code] = Position(
                stock_code=stock_code,
                shares=shares,
                avg_price=price,
                current_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )

        trade = Trade(
            stock_code=stock_code,
            direction="buy",
            entry_price=price,
            shares=shares,
            commission=commission,
        )
        self.trades.append(trade)
        return trade

    def sell(
        self,
        stock_code: str,
        price: float,
        shares: int | None = None,
        commission: float | None = None,
    ) -> Optional[Trade]:
        """Close (part of) a position.

        If *shares* is *None*, the entire position is sold.
        If *commission* is *None*, it is calculated automatically.
        """
        pos = self.positions.get(stock_code)
        if pos is None:
            return None

        sell_shares = shares if shares is not None else pos.shares
        sell_shares = min(sell_shares, pos.shares)

        if commission is None:
            commission = self.commission.calc(price, sell_shares, "sell")

        revenue = price * sell_shares - commission
        self.cash += revenue

        avg_price = pos.avg_price
        pnl = (price - avg_price) * sell_shares - commission
        pnl_pct = (price - avg_price) / avg_price if avg_price > 0 else 0.0

        pos.shares -= sell_shares
        if pos.shares <= 0:
            del self.positions[stock_code]

        trade = Trade(
            stock_code=stock_code,
            direction="sell",
            entry_price=avg_price,
            exit_price=price,
            shares=sell_shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            status="closed",
        )
        self.trades.append(trade)
        return trade

    # -- equity recording ----------------------------------------------------

    def record_equity(self) -> None:
        self.equity_curve.append(self.total_equity)
