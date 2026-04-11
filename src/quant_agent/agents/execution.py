"""执行 Agent — 模拟交易执行"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from ..audit import AuditLogger
from ..portfolio import CommissionModel, Portfolio, Position
from .base import BaseAgent, AgentResult

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)


@dataclass
class Order:
    """订单"""
    stock_code: str
    direction: str           # "buy" / "sell"
    price: float
    shares: int
    order_type: str = "market"  # "market" / "limit"
    status: str = "pending"     # "pending" / "filled" / "cancelled" / "rejected"
    filled_price: float = 0.0
    filled_shares: int = 0
    commission: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    filled_at: Optional[str] = None
    error: Optional[str] = None


class ExecutionAgent(BaseAgent):
    """执行 Agent — 模拟交易执行器

    Delegates portfolio management (positions, cash, commission) to the
    shared ``Portfolio`` class from ``quant_agent.portfolio``.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float | None = None,
        stamp_tax_rate: float | None = None,
        min_commission: float = 5.0,
        settings: Optional["Settings"] = None,
        audit_logger: Optional[AuditLogger] = None,
        **kwargs,
    ):
        super().__init__(name="execution", **kwargs)

        cr = commission_rate if commission_rate is not None else (
            settings.commission_rate if settings else 0.0003
        )
        sr = stamp_tax_rate if stamp_tax_rate is not None else (
            settings.stamp_tax_rate if settings else 0.001
        )

        self.initial_capital = initial_capital
        self._default_stop_loss = settings.default_stop_loss if settings else -0.08
        self._default_take_profit = settings.default_take_profit_2 if settings else 0.20
        self.audit_logger = audit_logger

        self._portfolio = Portfolio(
            cash=initial_capital,
            commission=CommissionModel(
                commission_rate=cr,
                stamp_tax_rate=sr,
                min_commission=min_commission,
            ),
            auto_round=True,
        )
        self.orders: list[Order] = []
        self.trade_log: list[dict] = []
        self.equity_history: list[dict] = []

    # -- delegate read-only access to portfolio positions --------------------

    @property
    def cash(self) -> float:
        return self._portfolio.cash

    @cash.setter
    def cash(self, value: float) -> None:
        self._portfolio.cash = value

    @property
    def positions(self) -> dict[str, Position]:
        return self._portfolio.positions

    @property
    def position_value(self) -> float:
        return self._portfolio.position_value

    @property
    def total_equity(self) -> float:
        return self._portfolio.total_equity

    @property
    def total_return(self) -> float:
        return (self.total_equity / self.initial_capital) - 1

    # -- BaseAgent interface -------------------------------------------------

    def analyze(self, stock_code: str) -> AgentResult:
        """返回当前组合状态"""
        return AgentResult(
            agent_name=self.name,
            stock_code=stock_code,
            signal="HOLD",
            confidence=0.0,
            reasoning=f"现金: {self.cash:,.0f} | 持仓: {len(self.positions)} | 总权益: {self.total_equity:,.0f}",
            metrics={
                "cash": self.cash,
                "positions": len(self.positions),
                "total_equity": self.total_equity,
                "total_return": self.total_return,
                "pending_orders": sum(1 for o in self.orders if o.status == "pending"),
            },
        )

    # -- execution interface -------------------------------------------------

    def execute_signal(self, stock_code: str, signal: str, position_pct: float = 0.0,
                       current_price: float = 0.0, stop_loss_pct: float | None = None,
                       take_profit_pct: float | None = None,
                       agent_results: list[AgentResult] | None = None) -> Optional[Order]:
        """执行交易信号"""
        if stop_loss_pct is None:
            stop_loss_pct = self._default_stop_loss
        if take_profit_pct is None:
            take_profit_pct = self._default_take_profit

        if current_price <= 0:
            logger.warning(f"无效价格 {stock_code}: {current_price}")
            return None

        # Build the audit record *before* execution so that every decision
        # (including rejections) is captured regardless of outcome.
        final_decision: dict[str, Any] = {
            "action": signal,
            "quantity": 0,
            "price": current_price,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "position_pct": position_pct,
        }

        order: Optional[Order] = None

        if signal == "BUY" and position_pct > 0:
            order = self._buy(stock_code, current_price, position_pct, stop_loss_pct, take_profit_pct)
        elif signal == "SELL":
            order = self._sell(stock_code, current_price)
        else:
            logger.info(f"HOLD {stock_code}")

        if order is not None:
            final_decision["quantity"] = order.filled_shares or order.shares
            final_decision["executed_price"] = order.filled_price or order.price
            final_decision["order_status"] = order.status
            if order.error:
                final_decision["error"] = order.error

        self._audit(stock_code, signal, agent_results, final_decision)
        return order

    def _audit(
        self,
        stock_code: str,
        signal: str,
        agent_results: list[AgentResult] | None,
        final_decision: dict[str, Any],
    ) -> None:
        """Write an audit record if an audit logger is configured."""
        if self.audit_logger is None:
            return

        votes: list[dict[str, Any]] = []
        if agent_results:
            for r in agent_results:
                votes.append({
                    "agent_name": r.agent_name,
                    "signal": r.signal,
                    "confidence": r.confidence,
                    "reasoning": r.reasoning,
                })

        self.audit_logger.log_trade_decision(
            stock_code=stock_code,
            signal=signal,
            agent_results=votes,
            final_decision=final_decision,
        )

    def check_stop_conditions(self, stock_code: str, current_price: float) -> Optional[Order]:
        """检查止损止盈"""
        pos = self._portfolio.get_position(stock_code)
        if pos is None:
            return None
        pos.update_price(current_price)

        if pos.should_stop_loss():
            logger.warning(f"触发止损: {stock_code} @ {current_price}")
            return self._sell(stock_code, current_price, reason="STOP_LOSS")
        if pos.should_take_profit():
            logger.info(f"触发止盈: {stock_code} @ {current_price}")
            return self._sell(stock_code, current_price, reason="TAKE_PROFIT")
        return None

    def update_prices(self, prices: dict[str, float]):
        """批量更新持仓价格"""
        for code, price in prices.items():
            self._portfolio.update_price(code, price)

    def record_equity(self, timestamp: str = ""):
        """记录权益快照"""
        self.equity_history.append({
            "timestamp": timestamp or datetime.now().isoformat(),
            "cash": self.cash,
            "position_value": self.position_value,
            "total_equity": self.total_equity,
        })

    def get_summary(self) -> dict:
        """组合摘要"""
        positions_detail = {}
        for code, pos in self.positions.items():
            positions_detail[code] = {
                "shares": pos.shares, "avg_price": pos.avg_price,
                "current_price": pos.current_price, "pnl": pos.pnl,
                "pnl_pct": pos.pnl_pct, "stop_loss": pos.stop_loss,
                "take_profit": pos.take_profit,
            }

        filled_orders = [o for o in self.orders if o.status == "filled"]
        buy_orders = [o for o in filled_orders if o.direction == "buy"]
        sell_orders = [o for o in filled_orders if o.direction == "sell"]

        return {
            "total_equity": round(self.total_equity, 2),
            "cash": round(self.cash, 2),
            "position_value": round(self.position_value, 2),
            "total_return": round(self.total_return, 4),
            "positions": positions_detail,
            "total_orders": len(filled_orders),
            "buy_orders": len(buy_orders),
            "sell_orders": len(sell_orders),
        }

    # -- internal helpers ----------------------------------------------------

    def _buy(self, stock_code: str, price: float, position_pct: float,
             stop_loss_pct: float, take_profit_pct: float) -> Order:
        amount = self.total_equity * position_pct
        shares = int(amount / price)
        shares = (shares // 100) * 100  # A-share board lot rounding

        # If the target position is too small for even 1 lot, try using all cash
        if shares <= 0:
            shares = int(self.cash / price)
            shares = (shares // 100) * 100
        if shares <= 0:
            order = Order(stock_code=stock_code, direction="buy", price=price, shares=0,
                         status="rejected", error="资金不足或股数<100")
            self.orders.append(order)
            return order

        # Delegate to Portfolio.buy() which handles commission via CommissionModel
        # and auto-reduces shares if cash is insufficient.
        trade = self._portfolio.buy(
            stock_code, price, shares,
            stop_loss=price * (1 + stop_loss_pct),
            take_profit=price * (1 + take_profit_pct),
        )

        actual_shares = trade.shares
        if actual_shares <= 0:
            order = Order(stock_code=stock_code, direction="buy", price=price, shares=0,
                         status="rejected", error="资金不足")
            self.orders.append(order)
            return order

        commission = trade.commission

        order = Order(stock_code=stock_code, direction="buy", price=price,
                     shares=actual_shares, status="filled", filled_price=price,
                     filled_shares=actual_shares, commission=commission)
        self.orders.append(order)
        self._log_trade(order)
        self._log_action("order_filled", stock_code=stock_code, signal="BUY",
                         direction="buy", shares=actual_shares, price=price)
        return order

    def _sell(self, stock_code: str, price: float, reason: str = "SIGNAL") -> Optional[Order]:
        pos = self._portfolio.get_position(stock_code)
        if pos is None:
            return None

        shares = pos.shares
        commission = self._portfolio.commission.calc(price, shares, "sell")

        self._portfolio.sell(stock_code, price, shares, commission=commission)

        order = Order(stock_code=stock_code, direction="sell", price=price,
                     shares=shares, status="filled", filled_price=price,
                     filled_shares=shares, commission=commission)
        self.orders.append(order)
        self._log_trade(order, reason=reason)
        self._log_action("order_filled", stock_code=stock_code, signal="SELL",
                         direction="sell", shares=shares, price=price,
                         reason=reason, pnl=pos.pnl)
        return order

    def _log_trade(self, order: Order, reason: str = ""):
        self.trade_log.append({
            "timestamp": datetime.now().isoformat(),
            "stock_code": order.stock_code,
            "direction": order.direction,
            "price": order.filled_price or order.price,
            "shares": order.filled_shares or order.shares,
            "commission": order.commission,
            "status": order.status,
            "reason": reason,
        })
