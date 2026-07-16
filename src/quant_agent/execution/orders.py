"""模拟订单模型 — 状态机 + 幂等键 + 拒绝原因 (P2.2)。

设计一个显式状态机：所有成交前先做风控 / 规则校验，非法请求以
``REJECTED`` 终态记录拒绝原因，而非静默失败或抛异常。幂等键
(``idempotency_key``) 用于防止重复下单。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .ids import generate_order_id


class OrderSide(str, Enum):
    """订单方向。"""

    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """订单状态机节点。"""

    PENDING = "PENDING"  # 已提交，等待成交（模拟即时成交）
    FILLED = "FILLED"  # 已成交
    REJECTED = "REJECTED"  # 被风控 / 规则拒绝（附 rejection_reason）
    CANCELLED = "CANCELLED"  # 已撤销


class RejectionReason(str, Enum):
    """拒绝原因（结构化，便于告警 / 统计）。"""

    INSUFFICIENT_CASH = "INSUFFICIENT_CASH"  # 资金不足
    POSITION_LIMIT = "POSITION_LIMIT"  # 单只仓位超限
    T_PLUS_ONE = "T_PLUS_ONE"  # T 日买入当日不可卖
    NO_POSITION = "NO_POSITION"  # 卖出无持仓
    INVALID_QUANTITY = "INVALID_QUANTITY"  # 数量非法（<=0）
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"  # 单日亏损熔断
    DUPLICATE_IDEMPOTENCY = "DUPLICATE_IDEMPOTENCY"  # 幂等键重复
    UNKNOWN = "UNKNOWN"


# 允许的状态转移：仅 PENDING 可转移到终态
_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING: {OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED},
    OrderStatus.FILLED: set(),
    OrderStatus.REJECTED: set(),
    OrderStatus.CANCELLED: set(),
}


class InvalidOrderTransition(Exception):
    """非法状态转移（代码缺陷，非业务拒绝）。"""


@dataclass
class Order:
    """模拟订单 — 完整的状态机记录。"""

    order_id: str
    idempotency_key: str | None
    stock_code: str
    side: OrderSide
    quantity: int
    price: float
    status: OrderStatus = OrderStatus.PENDING
    rejection_reason: RejectionReason | None = None
    rejection_detail: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    filled_at: str | None = None
    filled_price: float | None = None
    commission: float = 0.0
    trading_day: str | None = None

    # -- 状态机 ----------------------------------------------------------

    def can_transition(self, new: OrderStatus) -> bool:
        return new in _TRANSITIONS.get(self.status, set())

    def transition(
        self,
        new: OrderStatus,
        *,
        reason: RejectionReason | None = None,
        detail: str | None = None,
        filled_price: float | None = None,
        commission: float = 0.0,
    ) -> None:
        """转移到新状态，校验合法性。"""
        if new == self.status:
            return
        if not self.can_transition(new):
            raise InvalidOrderTransition(
                f"订单 {self.order_id}: {self.status.value} -> {new.value} 不允许"
            )
        self.status = new
        if new is OrderStatus.REJECTED:
            self.rejection_reason = reason or RejectionReason.UNKNOWN
            self.rejection_detail = detail
        elif new is OrderStatus.FILLED:
            self.filled_at = datetime.now().isoformat()
            self.filled_price = filled_price if filled_price is not None else self.price
            self.commission = commission

    def is_terminal(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        )

    def is_filled(self) -> bool:
        return self.status is OrderStatus.FILLED

    def is_rejected(self) -> bool:
        return self.status is OrderStatus.REJECTED

    # -- 序列化 ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "idempotency_key": self.idempotency_key,
            "stock_code": self.stock_code,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "status": self.status.value,
            "rejection_reason": self.rejection_reason.value if self.rejection_reason else None,
            "rejection_detail": self.rejection_detail,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "filled_price": self.filled_price,
            "commission": self.commission,
            "trading_day": self.trading_day,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Order:
        return cls(
            order_id=d["order_id"],
            idempotency_key=d.get("idempotency_key"),
            stock_code=d["stock_code"],
            side=OrderSide(d["side"]),
            quantity=int(d["quantity"]),
            price=float(d["price"]),
            status=OrderStatus(d.get("status", "PENDING")),
            rejection_reason=RejectionReason(d["rejection_reason"])
            if d.get("rejection_reason")
            else None,
            rejection_detail=d.get("rejection_detail"),
            created_at=d.get("created_at", ""),
            filled_at=d.get("filled_at"),
            filled_price=d.get("filled_price"),
            commission=float(d.get("commission", 0.0)),
            trading_day=d.get("trading_day"),
        )

    @classmethod
    def create(
        cls,
        stock_code: str,
        side: OrderSide | str,
        quantity: int,
        price: float,
        *,
        idempotency_key: str | None = None,
        trading_day: str | None = None,
    ) -> Order:
        """工厂方法：自动生成 order_id。"""
        return cls(
            order_id=generate_order_id(),
            idempotency_key=idempotency_key,
            stock_code=stock_code,
            side=OrderSide(side),
            quantity=quantity,
            price=price,
            trading_day=trading_day,
        )
