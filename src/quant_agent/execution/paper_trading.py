"""Persistent paper-trading service that survives process restarts.

Serialises the full ``Portfolio`` state (cash, positions, trades,
equity curve) to a JSON file using atomic writes (write-to-temp +
os.replace) so that the state file is never corrupted by a partial
write.

Usage::

    from quant_agent.execution.paper_trading import PaperTradingService

    svc = PaperTradingService(data_dir="data", initial_capital=100_000)
    svc.buy("300750", price=100.0, amount=200, stop_loss=90.0, take_profit=120.0)
    # state is auto-saved after every buy / sell

    # Later, in a fresh process:
    svc2 = PaperTradingService(data_dir="data", initial_capital=100_000)
    assert svc2.portfolio.positions["300750"].shares == 200
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from ..portfolio import CommissionModel, Portfolio, Position, Trade

logger = logging.getLogger(__name__)

# Version embedded in the state file so that future schema changes
# can be handled gracefully.
_STATE_VERSION = 1


class PaperTradingService:
    """Persistent paper-trading service that survives process restarts.

    Parameters
    ----------
    data_dir:
        Root data directory.  State is stored under
        ``<data_dir>/paper_trading/portfolio_state.json``.
    initial_capital:
        Starting cash when no previous state exists.
    commission:
        Optional ``CommissionModel``; defaults to the A-share model.
    """

    def __init__(
        self,
        data_dir: str,
        initial_capital: float = 100_000.0,
        commission: Optional[CommissionModel] = None,
    ) -> None:
        self._state_dir = Path(data_dir) / "paper_trading"
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._state_file = self._state_dir / "portfolio_state.json"
        self._initial_capital = initial_capital
        self._default_commission = commission or CommissionModel()
        self._portfolio = self._load_or_create(initial_capital)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def buy(
        self,
        code: str,
        price: float,
        amount: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Trade:
        """Execute a simulated buy and persist state."""
        result = self._portfolio.buy(
            code, price, amount,
            stop_loss=stop_loss, take_profit=take_profit,
        )
        self.save_state()
        return result

    def sell(
        self,
        code: str,
        price: float,
        amount: Optional[int] = None,
    ) -> Optional[Trade]:
        """Execute a simulated sell and persist state."""
        result = self._portfolio.sell(code, price, amount)
        if result is not None:
            self.save_state()
        return result

    def update_price(self, code: str, price: float) -> None:
        """Update the current price for a held position.

        Does *not* persist state -- price updates are typically very
        frequent (every tick).  Callers should invoke ``save_state()``
        periodically.
        """
        self._portfolio.update_price(code, price)

    def save_state(self) -> None:
        """Serialize current Portfolio state to JSON atomically."""
        data = self._serialize(self._portfolio)
        self._atomic_write_json(data, self._state_file)
        logger.debug("Paper-trading state saved to %s", self._state_file)

    def get_state_summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for display / logging."""
        pf = self._portfolio
        positions = []
        for code, pos in pf.positions.items():
            positions.append({
                "code": code,
                "shares": pos.shares,
                "avg_price": round(pos.avg_price, 4),
                "current_price": round(pos.current_price, 4),
                "pnl": round(pos.pnl, 2),
                "pnl_pct": round(pos.pnl_pct, 4),
                "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            })
        return {
            "cash": round(pf.cash, 2),
            "positions": positions,
            "total_equity": round(pf.total_equity, 2),
            "position_value": round(pf.position_value, 2),
            "unrealized_pnl": round(
                sum(p.pnl for p in pf.positions.values()), 2
            ),
            "trade_count": len(pf.trades),
            "closed_trades": len(pf.closed_trades),
        }

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _serialize(self, pf: Portfolio) -> dict[str, Any]:
        """Convert a Portfolio (with positions, trades, equity) to a dict."""
        return {
            "version": _STATE_VERSION,
            "cash": pf.cash,
            "commission": {
                "commission_rate": pf.commission.commission_rate,
                "stamp_tax_rate": pf.commission.stamp_tax_rate,
                "min_commission": pf.commission.min_commission,
            },
            "positions": [
                {
                    "stock_code": p.stock_code,
                    "shares": p.shares,
                    "avg_price": p.avg_price,
                    "current_price": p.current_price,
                    "entry_date": p.entry_date,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                }
                for p in pf.positions.values()
            ],
            "trades": [
                {
                    "stock_code": t.stock_code,
                    "direction": t.direction,
                    "entry_date": t.entry_date,
                    "exit_date": t.exit_date,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "shares": t.shares,
                    "pnl": t.pnl,
                    "pnl_pct": t.pnl_pct,
                    "commission": t.commission,
                    "status": t.status,
                }
                for t in pf.trades
            ],
            "equity_curve": list(pf.equity_curve),
        }

    def _deserialize(self, data: dict[str, Any]) -> Portfolio:
        """Reconstruct a Portfolio from a persisted dict."""
        # Commission model
        cm_data = data.get("commission", {})
        commission = CommissionModel(
            commission_rate=cm_data.get("commission_rate", 0.0003),
            stamp_tax_rate=cm_data.get("stamp_tax_rate", 0.001),
            min_commission=cm_data.get("min_commission", 5.0),
        )

        positions: dict[str, Position] = {}
        for pd_item in data.get("positions", []):
            pos = Position(
                stock_code=pd_item["stock_code"],
                shares=pd_item["shares"],
                avg_price=pd_item["avg_price"],
                current_price=pd_item.get("current_price", 0.0),
                entry_date=pd_item.get("entry_date", ""),
                stop_loss=pd_item.get("stop_loss", 0.0),
                take_profit=pd_item.get("take_profit", 0.0),
            )
            positions[pos.stock_code] = pos

        trades: list[Trade] = []
        for td in data.get("trades", []):
            trades.append(Trade(
                stock_code=td["stock_code"],
                direction=td["direction"],
                entry_date=td.get("entry_date", ""),
                exit_date=td.get("exit_date"),
                entry_price=td.get("entry_price", 0.0),
                exit_price=td.get("exit_price", 0.0),
                shares=td.get("shares", 0),
                pnl=td.get("pnl", 0.0),
                pnl_pct=td.get("pnl_pct", 0.0),
                commission=td.get("commission", 0.0),
                status=td.get("status", "open"),
            ))

        equity_curve = data.get("equity_curve", [])

        return Portfolio(
            cash=data["cash"],
            positions=positions,
            trades=trades,
            equity_curve=equity_curve,
            commission=commission,
        )

    # ------------------------------------------------------------------
    # Load / create
    # ------------------------------------------------------------------

    def _load_or_create(self, initial_capital: float) -> Portfolio:
        """Load existing state from JSON or create a fresh Portfolio."""
        if self._state_file.exists():
            try:
                return self._load_state()
            except Exception:
                logger.warning(
                    "Failed to load paper-trading state from %s; "
                    "creating fresh portfolio",
                    self._state_file,
                    exc_info=True,
                )
        return Portfolio(
            cash=initial_capital,
            commission=self._default_commission,
        )

    def _load_state(self) -> Portfolio:
        """Deserialize JSON state into Portfolio with Positions."""
        text = self._state_file.read_text(encoding="utf-8")
        data = json.loads(text)
        version = data.get("version", 0)
        if version != _STATE_VERSION:
            logger.warning(
                "State file version %d != expected %d; attempting migration",
                version, _STATE_VERSION,
            )
        return self._deserialize(data)

    # ------------------------------------------------------------------
    # Atomic write (mirrors store.py pattern)
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_write_json(data: dict[str, Any], target: Path) -> None:
        """Write *data* as JSON to *target* atomically.

        Writes to a temporary file in the same directory first, then
        uses ``os.replace()`` which is atomic on the same filesystem.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            suffix=".json.tmp",
            prefix=target.stem + "_",
            dir=str(target.parent),
        )
        try:
            os.close(fd)
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp_path, str(target))
        except BaseException:
            # Clean up the temporary file on any failure.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
