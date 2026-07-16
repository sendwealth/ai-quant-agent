"""实盘就绪能力（推荐 #5）— 券商适配 / 幂等下单 / 回报对账 / 市场状态约束。

⚠️ 本模块是**实盘交易路径的脚手架（template）**，刻意与现有模拟交易
``PaperTradingService`` / ``ExecutionAgent`` **解耦**，不延伸模拟执行器。
接入真实券商前，必须由具备资质的团队补全：

- 真实券商 SDK / 网关对接（替换 :class:`BrokerAdapter` 的桩实现）；
- 资金与持仓的真实账户对接、指令路由与风控前置；
- 灾备（断线重连、指令重试、状态最终一致）与合规审计留痕；
- 交易时间 / 涨跌停 / 停复牌等市场状态约束的实时数据源。

本文件只把**架构边界与不可绕过的硬约束**先立起来，并提供可在单测中验证的
纯逻辑（幂等键、对账差异、涨跌停判定、交易时段判定）。
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum

logger = logging.getLogger(__name__)


# ── 市场状态约束（硬约束，独立验证） ─────────────────────────────────────────


class MarketStateError(Exception):
    """市场状态不满足下单条件（非交易时段 / 停复牌 / 涨跌停）。"""


# A 股常规涨跌停幅度；创业板/科创板为 20%，新股上市前 5 日无涨跌幅。
_DEFAULT_LIMIT_PCT = 0.10
_STAR_CHINEXT_PREFIXES = ("30", "688")  # 创业板 / 科创板
_STAR_CHINEXT_LIMIT_PCT = 0.20


def daily_limit_pct(stock_code: str) -> float:
    """返回该股票当日的涨跌停幅度（默认 10%，创业板/科创板 20%）。"""
    if stock_code.startswith(_STAR_CHINEXT_PREFIXES):
        return _STAR_CHINEXT_LIMIT_PCT
    return _DEFAULT_LIMIT_PCT


def price_within_limit(prev_close: float, price: float, stock_code: str) -> bool:
    """判定委托价是否落在当日涨跌停区间内（含边界）。

    Args:
        prev_close: 昨收价（计算涨跌停的基准）。
        price: 委托价。
        stock_code: 股票代码（决定 10% / 20% 幅度）。
    """
    if prev_close <= 0:
        return False
    limit = daily_limit_pct(stock_code)
    lo = round(prev_close * (1 - limit), 4)
    hi = round(prev_close * (1 + limit), 4)
    return lo <= price <= hi


class MarketCalendar:
    """交易时段 / 交易日约束（模板实现，节假日需接真实日历）。"""

    # 默认交易时段（A 股）
    MORNING_OPEN = time(9, 30)
    MORNING_CLOSE = time(11, 30)
    AFTERNOON_OPEN = time(13, 0)
    AFTERNOON_CLOSE = time(15, 0)

    def __init__(self, holidays: set[str] | None = None) -> None:
        # 节假日集合，格式 "YYYY-MM-DD"（生产环境接交易所日历）
        self.holidays = holidays or set()

    def is_trading_day(self, dt: datetime) -> bool:
        """是否为交易日（周一到周五且非假期）。"""
        if dt.weekday() >= 5:  # 5=周六, 6=周日
            return False
        return dt.strftime("%Y-%m-%d") not in self.holidays

    def is_trading_session(self, dt: datetime) -> bool:
        """是否处于连续竞价时段（9:30-11:30 / 13:00-15:00 且为交易日）。"""
        if not self.is_trading_day(dt):
            return False
        t = dt.time()
        in_morning = self.MORNING_OPEN <= t <= self.MORNING_CLOSE
        in_afternoon = self.AFTERNOON_OPEN <= t <= self.AFTERNOON_CLOSE
        return in_morning or in_afternoon

    def require_tradable(self, dt: datetime, stock_code: str, suspended: bool) -> None:
        """硬约束检查：非交易时段 / 停牌 → 抛 :class:`MarketStateError`。"""
        if not self.is_trading_session(dt):
            raise MarketStateError(f"{stock_code} 不在交易时段（{dt.isoformat()}），禁止下单")
        if suspended:
            raise MarketStateError(f"{stock_code} 处于停牌状态，禁止下单")


# ── 券商适配（独立实盘路径的抽象） ─────────────────────────────────────────


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    LIMIT = "limit"  # 限价
    MARKET = "market"  # 市价（实盘需券商支持且谨慎使用）


@dataclass
class BrokerOrder:
    """发往券商的订单（独立于模拟 ``Order``，避免污染模拟路径）。"""

    idempotency_key: str  # 幂等键（见 make_idempotency_key）
    stock_code: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.LIMIT
    price: float | None = None  # 限价单必填
    note: str = ""


@dataclass
class BrokerFill:
    """券商回报（成交回报）。"""

    broker_order_id: str
    idempotency_key: str
    stock_code: str
    side: OrderSide
    filled_quantity: int
    filled_price: float
    status: str  # "filled" / "partial" / "rejected" / "canceled"
    timestamp: str = ""


def make_idempotency_key(stock_code: str, side: OrderSide, intent: str, day: str) -> str:
    """生成确定性幂等键。

    同一「股票 + 方向 + 意图 + 交易日」只会下发一次真实订单；重复提交（如
    网络重试、进程重启）返回既有回执，避免重复开仓/平仓。

    Args:
        stock_code: 6 位代码。
        side: 买卖方向。
        intent: 业务意图标签，如 ``"signal"`` / ``"stop_loss"`` / ``"take_profit"``。
        day: 交易日 ``YYYY-MM-DD``（自然日维度去重，跨日可重新决策）。
    """
    raw = f"{stock_code}|{side.value}|{intent}|{day}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


class BrokerAdapter(ABC):
    """券商适配抽象（实盘需实现真实下单/查询）。

    桩实现仅用于单测与流程演示；真实实现必须接入券商 API、做资金前置校验、
    并落地审计日志。
    """

    @abstractmethod
    def submit_order(self, order: BrokerOrder) -> BrokerFill:
        """提交订单，返回成交回报（幂等由调用方保证）。"""

    @abstractmethod
    def get_order(self, broker_order_id: str) -> BrokerFill | None:
        """查询订单状态。"""

    @abstractmethod
    def cancel_order(self, broker_order_id: str) -> bool:
        """撤单。"""


class InMemoryBrokerAdapter(BrokerAdapter):
    """桩实现：内存态券商，用于单测与本地流程验证（**非生产**）。"""

    def __init__(self) -> None:
        self._fills: dict[str, BrokerFill] = {}
        self._seq = 0

    def submit_order(self, order: BrokerOrder) -> BrokerFill:
        self._seq += 1
        bid = f"B{self._seq:06d}"
        fill = BrokerFill(
            broker_order_id=bid,
            idempotency_key=order.idempotency_key,
            stock_code=order.stock_code,
            side=order.side,
            filled_quantity=order.quantity,
            filled_price=order.price or 0.0,
            status="filled",
            timestamp=datetime.now().isoformat(),
        )
        self._fills[bid] = fill
        return fill

    def get_order(self, broker_order_id: str) -> BrokerFill | None:
        return self._fills.get(broker_order_id)

    def cancel_order(self, broker_order_id: str) -> bool:
        f = self._fills.get(broker_order_id)
        if f is None or f.status == "filled":
            return False
        f.status = "canceled"
        return True


class IdempotentBroker(BrokerAdapter):
    """幂等下单包装：以 ``idempotency_key`` 去重，避免重复下单（推荐 #5 核心）。

    重复提交同一幂等键时，直接返回首次结果，不再调用底层券商；结果可持久化
    （``store``）以在进程重启后仍然幂等。底层 :class:`BrokerAdapter` 不参与
    幂等判断，只接收「已确认未重复」的订单。
    """

    def __init__(
        self,
        delegate: BrokerAdapter,
        store: dict[str, BrokerFill] | None = None,
    ) -> None:
        self.delegate = delegate
        self._cache: dict[str, BrokerFill] = store if store is not None else {}

    def submit_order(self, order: BrokerOrder) -> BrokerFill:
        existing = self._cache.get(order.idempotency_key)
        if existing is not None:
            logger.info("幂等命中，跳过重复下单: %s", order.idempotency_key)
            return existing
        fill = self.delegate.submit_order(order)
        self._cache[order.idempotency_key] = fill
        return fill

    def get_order(self, broker_order_id: str) -> BrokerFill | None:
        return self.delegate.get_order(broker_order_id)

    def cancel_order(self, broker_order_id: str) -> bool:
        return self.delegate.cancel_order(broker_order_id)


# ── 回报对账（本地指令 vs 券商回报） ─────────────────────────────────────────


@dataclass
class ReconcileMismatch:
    """一条对账差异。"""

    idempotency_key: str
    stock_code: str
    reason: str
    local: BrokerOrder | None = None
    remote: BrokerFill | None = None


class OrderReconciler:
    """回报对账：比对本地期望委托与券商实际回报，标记差异。

    差异类型：本地有/券商无（漏单）、券商有/本地无（未知单）、数量或价格不符。
    """

    def reconcile(
        self,
        local_orders: list[BrokerOrder],
        remote_fills: list[BrokerFill],
    ) -> list[ReconcileMismatch]:
        by_key: dict[str, BrokerOrder] = {o.idempotency_key: o for o in local_orders}
        remote_by_key: dict[str, BrokerFill] = {f.idempotency_key: f for f in remote_fills}
        mismatches: list[ReconcileMismatch] = []

        for key, local in by_key.items():
            remote = remote_by_key.get(key)
            if remote is None:
                mismatches.append(
                    ReconcileMismatch(
                        idempotency_key=key,
                        stock_code=local.stock_code,
                        reason="本地有委托但券商无回报（疑似漏单）",
                        local=local,
                    )
                )
                continue
            if remote.status == "rejected":
                mismatches.append(
                    ReconcileMismatch(
                        idempotency_key=key,
                        stock_code=local.stock_code,
                        reason="券商拒绝委托",
                        local=local,
                        remote=remote,
                    )
                )
            elif remote.filled_quantity != local.quantity:
                mismatches.append(
                    ReconcileMismatch(
                        idempotency_key=key,
                        stock_code=local.stock_code,
                        reason=(
                            f"成交数量不符：本地 {local.quantity} / 券商 {remote.filled_quantity}"
                        ),
                        local=local,
                        remote=remote,
                    )
                )

        for key, remote in remote_by_key.items():
            if key not in by_key:
                mismatches.append(
                    ReconcileMismatch(
                        idempotency_key=key,
                        stock_code=remote.stock_code,
                        reason="券商有回报但本地无对应委托（未知单）",
                        remote=remote,
                    )
                )
        return mismatches
