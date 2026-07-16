"""推荐 #5 实盘就绪 scaffold 测试：幂等下单、回报对账、市场状态/涨跌停约束。

注意：本模块是模板，不接入真实券商；测试仅验证纯逻辑与硬约束。
"""

from __future__ import annotations

from datetime import datetime

import pytest

from quant_agent.execution.broker import (
    BrokerFill,
    BrokerOrder,
    IdempotentBroker,
    InMemoryBrokerAdapter,
    MarketCalendar,
    MarketStateError,
    OrderReconciler,
    OrderSide,
    OrderType,
    make_idempotency_key,
    price_within_limit,
)


def _order(stock="600519", side=OrderSide.BUY, qty=100, price=100.0, day="2026-07-16"):
    return BrokerOrder(
        idempotency_key=make_idempotency_key(stock, side, "signal", day),
        stock_code=stock,
        side=side,
        quantity=qty,
        order_type=OrderType.LIMIT,
        price=price,
    )


class TestIdempotencyKey:
    def test_deterministic(self):
        a = make_idempotency_key("600519", OrderSide.BUY, "signal", "2026-07-16")
        b = make_idempotency_key("600519", OrderSide.BUY, "signal", "2026-07-16")
        assert a == b

    def test_differs_by_dimension(self):
        base = ("600519", OrderSide.BUY, "signal", "2026-07-16")
        assert make_idempotency_key(*base) != make_idempotency_key(
            "600519", OrderSide.SELL, "signal", "2026-07-16"
        )
        assert make_idempotency_key(*base) != make_idempotency_key(
            "600519", OrderSide.BUY, "stop_loss", "2026-07-16"
        )
        # 跨日允许重新决策（不同幂等键）
        assert make_idempotency_key(*base) != make_idempotency_key(
            "600519", OrderSide.BUY, "signal", "2026-07-17"
        )


class TestIdempotentBroker:
    def test_duplicate_submit_not_forwarded(self):
        inner = InMemoryBrokerAdapter()
        broker = IdempotentBroker(inner)
        o = _order()
        f1 = broker.submit_order(o)
        f2 = broker.submit_order(o)  # 幂等键相同
        assert f1.broker_order_id == f2.broker_order_id
        # 底层只收到一次
        assert len(inner._fills) == 1

    def test_persisted_store_survives_restart(self):
        store: dict[str, BrokerFill] = {}
        inner = InMemoryBrokerAdapter()
        o = _order()
        IdempotentBroker(inner, store=store).submit_order(o)
        # 模拟进程重启：新的 broker 实例，共享持久化 store
        broker2 = IdempotentBroker(InMemoryBrokerAdapter(), store=store)
        f = broker2.submit_order(o)
        assert f is not None
        assert f.status == "filled"


class TestReconciliation:
    def test_no_mismatch(self):
        o = _order()
        fill = BrokerFill(
            broker_order_id="B1",
            idempotency_key=o.idempotency_key,
            stock_code=o.stock_code,
            side=o.side,
            filled_quantity=o.quantity,
            filled_price=o.price or 0.0,
            status="filled",
        )
        assert OrderReconciler().reconcile([o], [fill]) == []

    def test_missing_remote(self):
        o = _order()
        mism = OrderReconciler().reconcile([o], [])
        assert len(mism) == 1
        assert "漏单" in mism[0].reason

    def test_quantity_mismatch(self):
        o = _order(qty=100)
        fill = BrokerFill(
            broker_order_id="B1",
            idempotency_key=o.idempotency_key,
            stock_code=o.stock_code,
            side=o.side,
            filled_quantity=80,
            filled_price=o.price or 0.0,
            status="partial",
        )
        mism = OrderReconciler().reconcile([o], [fill])
        assert len(mism) == 1
        assert "数量不符" in mism[0].reason

    def test_rejected(self):
        o = _order()
        fill = BrokerFill(
            broker_order_id="B1",
            idempotency_key=o.idempotency_key,
            stock_code=o.stock_code,
            side=o.side,
            filled_quantity=0,
            filled_price=0.0,
            status="rejected",
        )
        mism = OrderReconciler().reconcile([o], [fill])
        assert any("拒绝" in m.reason for m in mism)

    def test_unknown_remote(self):
        o = _order()
        other = _order(stock="000001")
        mism = OrderReconciler().reconcile(
            [o],
            [
                BrokerFill(
                    broker_order_id="BX",
                    idempotency_key=other.idempotency_key,
                    stock_code=other.stock_code,
                    side=other.side,
                    filled_quantity=other.quantity,
                    filled_price=other.price or 0.0,
                    status="filled",
                )
            ],
        )
        assert any("未知单" in m.reason for m in mism)


class TestMarketConstraints:
    def test_price_within_limit_main(self):
        # 主板 10%：昨收 100，委托 109 在范围内，110 为涨停价（可委托，边界包含），110.1 超出
        assert price_within_limit(100.0, 109.0, "600519") is True
        assert price_within_limit(100.0, 110.0, "600519") is True
        assert price_within_limit(100.0, 110.1, "600519") is False
        assert price_within_limit(100.0, 90.0, "600519") is True
        assert price_within_limit(100.0, 89.9, "600519") is False

    def test_price_within_limit_star(self):
        # 创业板 20%
        assert price_within_limit(100.0, 119.0, "300750") is True
        assert price_within_limit(100.0, 121.0, "300750") is False

    def test_trading_session(self):
        cal = MarketCalendar()
        # 周中交易时段内
        assert cal.is_trading_session(datetime(2026, 7, 15, 10, 0)) is True
        # 午休
        assert cal.is_trading_session(datetime(2026, 7, 15, 12, 0)) is False
        # 周末
        assert cal.is_trading_session(datetime(2026, 7, 18, 10, 0)) is False
        # 非交易日（假期）
        holiday_cal = MarketCalendar(holidays={"2026-07-15"})
        assert holiday_cal.is_trading_session(datetime(2026, 7, 15, 10, 0)) is False

    def test_require_tradable_raises(self):
        cal = MarketCalendar()
        with pytest.raises(MarketStateError):
            cal.require_tradable(datetime(2026, 7, 18, 10, 0), "600519", suspended=False)
        with pytest.raises(MarketStateError):
            cal.require_tradable(datetime(2026, 7, 15, 10, 0), "600519", suspended=True)
        # 正常不应抛
        cal.require_tradable(datetime(2026, 7, 15, 10, 0), "600519", suspended=False)
