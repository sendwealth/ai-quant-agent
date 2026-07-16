"""Persistent paper-trading service — SQLite 后端 + 订单状态机 (P2)。

相对旧版的变化（向后兼容 ``portfolio`` / ``buy`` / ``sell`` / ``save_state``/
``get_state_summary`` 接口）：

- 持久化由 JSON 快照改为 SQLite（WAL），原子写入、线程安全（P2.3）。
- 每次下单生成统一订单 ID，经显式状态机校验（PENDING→FILLED/OrderStatus.REJECTED），
  非法请求以 ``OrderStatus.REJECTED`` 终态 + 拒绝原因记录，而非静默失败（P2.2）。
- 幂等键 (idempotency_key) 防止重复下单（P2.2）。
- 可选强制 T+1（当日买入不可卖）、单只仓位上限、单日亏损熔断（P2.4）。
- 每笔订单流转写入结构化审计流（P2.5）。

用法::

    from quant_agent.execution.paper_trading import PaperTradingService

    svc = PaperTradingService(data_dir="data", initial_capital=100_000,
                              enforce_t_plus_one=True, enforce_position_limit=True)
    svc.buy("300750", price=100.0, amount=200, stop_loss=90.0, take_profit=120.0)
    # 重启后状态不丢
    svc2 = PaperTradingService(data_dir="data", initial_capital=100_000)
    assert svc2.portfolio.positions["300750"].shares == 200
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from ..audit import AuditLogger
from ..portfolio import CommissionModel, Portfolio, Trade
from .orders import (
    Order,
    OrderSide,
    OrderStatus,
    RejectionReason,
)
from .store import SqliteStateStore

logger = logging.getLogger(__name__)


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


class PaperTradingService:
    """SQLite 持久化的模拟交易服务（进程重启不丢状态）。

    Parameters
    ----------
    data_dir:
        根数据目录。状态存于 ``<data_dir>/paper_trading/paper_trading.db``。
    initial_capital:
        无历史状态时新建组合的起始现金。
    commission:
        可选 ``CommissionModel``；默认 A 股模型。
    enforce_t_plus_one:
        为 True 时，当日买入的持仓当日不可卖出（A 股 T+1）。默认 False
        以兼容既有确定性测试。
    max_position_pct:
        单只持仓占组合净值上限（仅 ``enforce_position_limit=True`` 时生效）。
    enforce_position_limit:
        为 True 时校验单只仓位上限与整百股数；默认 False。
    enforce_daily_loss:
        为 True 时启用单日亏损熔断；默认 False。
    max_daily_loss_pct:
        单日亏损熔断阈值（占 initial_capital 的比例，应为负数）。
    audit_logger:
        可选审计器，每笔订单流转写入结构化审计流。
    """

    def __init__(
        self,
        data_dir: str,
        initial_capital: float = 100_000.0,
        commission: CommissionModel | None = None,
        *,
        enforce_t_plus_one: bool = False,
        max_position_pct: float = 0.20,
        enforce_position_limit: bool = False,
        enforce_daily_loss: bool = False,
        max_daily_loss_pct: float = -0.03,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._store = self._open_store(data_dir)
        self._initial_capital = initial_capital
        self._default_commission = commission or CommissionModel()
        self.enforce_t_plus_one = enforce_t_plus_one
        self.max_position_pct = max_position_pct
        self.enforce_position_limit = enforce_position_limit
        self.enforce_daily_loss = enforce_daily_loss
        self.max_daily_loss_pct = max_daily_loss_pct
        self._audit = audit_logger

        loaded = self._safe_load(initial_capital)
        self._portfolio: Portfolio = (
            loaded[0]
            if loaded is not None
            else Portfolio(cash=initial_capital, commission=self._default_commission)
        )
        self._orders: list[Order] = loaded[1] if loaded is not None else []

        # 幂等键集合（从已加载订单重建）
        self._idempotency_keys = {o.idempotency_key for o in self._orders if o.idempotency_key}
        # 单日亏损累计（按 trading_day 维度；会话级，重启不持久化）
        self._day_loss: dict[str, float] = {}

    def _open_store(self, data_dir: str) -> SqliteStateStore:
        """打开 SQLite 存储；文件损坏时删库重建，不崩溃。"""
        try:
            return SqliteStateStore(data_dir)
        except sqlite3.Error:
            logger.warning("状态数据库打开失败（可能损坏），删除后重建", exc_info=True)
            self._remove_db(data_dir)
            return SqliteStateStore(data_dir)

    @staticmethod
    def _remove_db(data_dir: str) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = Path(data_dir) / "paper_trading" / f"paper_trading.db{suffix}"
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    def _safe_load(self, initial_capital: float):
        """加载状态；数据库损坏时删库重建为全新组合，不崩溃。"""
        try:
            return self._store.load(initial_capital)
        except sqlite3.Error:
            logger.warning("状态加载失败（可能损坏），重建为全新组合", exc_info=True)
            self._remove_db(str(self._store.db_path.parent.parent))
            self._store = SqliteStateStore(str(self._store.db_path.parent.parent))
            return None

    # ------------------------------------------------------------------
    # 公共只读访问
    # ------------------------------------------------------------------

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    @property
    def orders(self) -> list[Order]:
        return self._orders

    def get_orders(self, status: OrderStatus | None = None) -> list[Order]:
        if status is None:
            return list(self._orders)
        return [o for o in self._orders if o.status is status]

    def get_order(self, order_id: str) -> Order | None:
        for o in self._orders:
            if o.order_id == order_id:
                return o
        return None

    def pending_orders(self) -> list[Order]:
        return [o for o in self._orders if o.status is OrderStatus.PENDING]

    # ------------------------------------------------------------------
    # 下单（公共 API）
    # ------------------------------------------------------------------

    def buy(
        self,
        code: str,
        price: float,
        amount: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        *,
        trading_day: str | None = None,
        idempotency_key: str | None = None,
    ) -> Trade | None:
        """模拟买入；返回成交 Trade，被拒返回 None。"""
        _order, trade = self._execute_buy(
            code, price, amount, stop_loss, take_profit, trading_day, idempotency_key
        )
        return trade

    def sell(
        self,
        code: str,
        price: float,
        amount: int | None = None,
        *,
        trading_day: str | None = None,
        idempotency_key: str | None = None,
    ) -> Trade | None:
        """模拟卖出；返回成交 Trade，被拒/无持仓返回 None。"""
        _order, trade = self._execute_sell(code, price, amount, trading_day, idempotency_key)
        return trade

    def submit_order(
        self,
        code: str,
        side: OrderSide | str,
        quantity: int,
        price: float,
        *,
        idempotency_key: str | None = None,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        trading_day: str | None = None,
    ) -> Order:
        """统一的下单入口，返回携带状态/拒绝原因的 Order（含幂等键）。"""
        side_enum = OrderSide(side)
        if side_enum is OrderSide.BUY:
            order, _ = self._execute_buy(
                code, price, quantity, stop_loss, take_profit, trading_day, idempotency_key
            )
        else:
            order, _ = self._execute_sell(code, price, quantity, trading_day, idempotency_key)
        return order

    # ------------------------------------------------------------------
    # 内部执行
    # ------------------------------------------------------------------

    def _execute_buy(
        self,
        code: str,
        price: float,
        amount: int,
        stop_loss: float,
        take_profit: float,
        trading_day: str | None,
        idempotency_key: str | None,
    ) -> tuple[Order, Trade | None]:
        trading_day = trading_day or _today()
        order = Order.create(
            code,
            OrderSide.BUY,
            amount,
            price,
            idempotency_key=idempotency_key,
            trading_day=trading_day,
        )
        self._orders.append(order)

        # 幂等检查
        if idempotency_key is not None:
            if idempotency_key in self._idempotency_keys:
                order.transition(
                    OrderStatus.REJECTED,
                    reason=RejectionReason.DUPLICATE_IDEMPOTENCY,
                    detail="重复幂等键，忽略重复下单",
                )
                self._audit_order(order)
                self.save_state()
                return order, None
            self._idempotency_keys.add(idempotency_key)

        reject = self._check_buy_rules(code, price, amount, trading_day)
        if reject is not None:
            reason, detail = reject
            order.transition(OrderStatus.REJECTED, reason=reason, detail=detail)
            self._audit_order(order)
            self.save_state()
            return order, None

        trade = self._portfolio.buy(
            code, price, amount, stop_loss=stop_loss, take_profit=take_profit
        )
        if trade.shares <= 0:
            order.transition(
                OrderStatus.REJECTED,
                reason=RejectionReason.INSUFFICIENT_CASH,
                detail="资金不足或股数为 0",
            )
            self._audit_order(order)
            self.save_state()
            return order, None

        pos = self._portfolio.positions.get(code)
        if pos is not None:
            pos.entry_date = trading_day
        order.transition(OrderStatus.FILLED, filled_price=price, commission=trade.commission)
        self._audit_order(order)
        self.save_state()
        return order, trade

    def _execute_sell(
        self,
        code: str,
        price: float,
        amount: int | None,
        trading_day: str | None,
        idempotency_key: str | None,
    ) -> tuple[Order, Trade | None]:
        trading_day = trading_day or _today()
        pos = self._portfolio.positions.get(code)
        qty = amount if amount is not None else (pos.shares if pos else 0)
        order = Order.create(
            code,
            OrderSide.SELL,
            qty,
            price,
            idempotency_key=idempotency_key,
            trading_day=trading_day,
        )
        self._orders.append(order)

        if idempotency_key is not None:
            if idempotency_key in self._idempotency_keys:
                order.transition(
                    OrderStatus.REJECTED,
                    reason=RejectionReason.DUPLICATE_IDEMPOTENCY,
                    detail="重复幂等键，忽略重复下单",
                )
                self._audit_order(order)
                self.save_state()
                return order, None
            self._idempotency_keys.add(idempotency_key)

        reject = self._check_sell_rules(code, trading_day)
        if reject is not None:
            reason, detail = reject
            order.transition(OrderStatus.REJECTED, reason=reason, detail=detail)
            self._audit_order(order)
            self.save_state()
            return order, None

        trade = self._portfolio.sell(code, price, amount)
        if trade is None:
            order.transition(
                OrderStatus.REJECTED, reason=RejectionReason.NO_POSITION, detail="无持仓"
            )
            self._audit_order(order)
            self.save_state()
            return order, None

        if self.enforce_daily_loss:
            self._day_loss[trading_day] = self._day_loss.get(trading_day, 0.0) + trade.pnl

        order.transition(OrderStatus.FILLED, filled_price=price, commission=trade.commission)
        self._audit_order(order)
        self.save_state()
        return order, trade

    # ------------------------------------------------------------------
    # 规则校验（P2.4）
    # ------------------------------------------------------------------

    def _check_buy_rules(
        self, code: str, price: float, amount: int, trading_day: str
    ) -> tuple[RejectionReason, str] | None:
        if amount <= 0:
            return RejectionReason.INVALID_QUANTITY, f"数量必须为正: {amount}"
        if self.enforce_position_limit:
            if amount % 100 != 0:
                return RejectionReason.INVALID_QUANTITY, f"数量须为 100 的整数倍: {amount}"
            total_equity = self._portfolio.total_equity
            existing = self._portfolio.positions.get(code)
            existing_value = (existing.shares * price) if existing else 0.0
            projected = existing_value + amount * price
            if total_equity > 0 and projected / total_equity > self.max_position_pct:
                pct = projected / total_equity
                return (
                    RejectionReason.POSITION_LIMIT,
                    f"单只仓位 {pct:.2%} 超过上限 {self.max_position_pct:.2%}",
                )
        if self.enforce_daily_loss:
            loss = self._day_loss.get(trading_day, 0.0)
            if loss < self.max_daily_loss_pct * self._initial_capital:
                return (
                    RejectionReason.DAILY_LOSS_LIMIT,
                    f"当日亏损 {loss:,.0f} 已达熔断阈值 {self.max_daily_loss_pct:.2%}",
                )
        return None

    def _check_sell_rules(self, code: str, trading_day: str) -> tuple[RejectionReason, str] | None:
        pos = self._portfolio.positions.get(code)
        if pos is None:
            return RejectionReason.NO_POSITION, f"无持仓: {code}"
        if self.enforce_t_plus_one and pos.entry_date and pos.entry_date == trading_day:
            return (
                RejectionReason.T_PLUS_ONE,
                f"T+1: {code} 于 {pos.entry_date} 买入，当日不可卖",
            )
        return None

    # ------------------------------------------------------------------
    # 价格更新 / 持久化 / 摘要
    # ------------------------------------------------------------------

    def update_price(self, code: str, price: float) -> None:
        """更新持仓现价（不落盘；频繁调用请周期 save_state）。"""
        self._portfolio.update_price(code, price)

    def save_state(self) -> None:
        """原子持久化 Portfolio 快照 + 订单列表。"""
        self._store.save(self._portfolio, self._orders)

    def get_state_summary(self) -> dict[str, Any]:
        pf = self._portfolio
        positions = []
        for code, pos in pf.positions.items():
            positions.append(
                {
                    "code": code,
                    "shares": pos.shares,
                    "avg_price": round(pos.avg_price, 4),
                    "current_price": round(pos.current_price, 4),
                    "pnl": round(pos.pnl, 2),
                    "pnl_pct": round(pos.pnl_pct, 4),
                    "stop_loss": pos.stop_loss,
                    "take_profit": pos.take_profit,
                    "entry_date": pos.entry_date,
                }
            )
        return {
            "cash": round(pf.cash, 2),
            "positions": positions,
            "total_equity": round(pf.total_equity, 2),
            "position_value": round(pf.position_value, 2),
            "unrealized_pnl": round(sum(p.pnl for p in pf.positions.values()), 2),
            "trade_count": len(pf.trades),
            "closed_trades": len(pf.closed_trades),
            "order_count": len(self._orders),
            "rejected_orders": sum(1 for o in self._orders if o.is_rejected()),
        }

    # ------------------------------------------------------------------
    # 审计（P2.5）
    # ------------------------------------------------------------------

    def _audit_order(self, order: Order) -> None:
        if self._audit is not None:
            self._audit.log_order_event(order=order, event=order.status.value)
