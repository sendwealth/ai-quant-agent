"""P2 测试 — 统一 ID / 订单状态机 / SQLite 存储 / 模拟交易规则与审计。

覆盖：
- P2.1 统一 ID 生成
- P2.2 订单状态机 / 拒绝原因 / 幂等
- P2.3 SQLite 存储（快照往返）
- P2.4 T+1 / 单只仓位上限 / 单日亏损熔断
- P2.5 结构化审计流
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quant_agent.audit import AuditLogger
from quant_agent.execution.ids import IdGenerator, generate_order_id, generate_trade_id
from quant_agent.execution.orders import (
    InvalidOrderTransition,
    Order,
    OrderSide,
    OrderStatus,
    RejectionReason,
)
from quant_agent.execution.paper_trading import PaperTradingService
from quant_agent.execution.store import SqliteStateStore
from quant_agent.portfolio import CommissionModel, Portfolio

# ---------------------------------------------------------------------------
# P2.1 统一 ID
# ---------------------------------------------------------------------------


class TestIdGenerator:
    def test_generate_order_and_trade_ids_unique(self):
        ids = {generate_order_id() for _ in range(100)}
        assert len(ids) == 100

    def test_prefix_present(self):
        assert generate_order_id().startswith("ord_")
        assert generate_trade_id().startswith("trd_")

    def test_deterministic_generator(self):
        g = IdGenerator("t", deterministic=True)
        a, b, c = g.next_id(), g.next_id(), g.next_id()
        assert a != b != c
        assert a.split("_")[-1] == "000001"
        assert c.split("_")[-1] == "000003"


# ---------------------------------------------------------------------------
# P2.2 订单状态机
# ---------------------------------------------------------------------------


class TestOrderStateMachine:
    def test_create_assigns_id(self):
        o = Order.create("300750", OrderSide.BUY, 100, 10.0, idempotency_key="k")
        assert o.order_id.startswith("ord_")
        assert o.idempotency_key == "k"
        assert o.status is OrderStatus.PENDING

    def test_pending_to_filled(self):
        o = Order.create("300750", OrderSide.BUY, 100, 10.0)
        o.transition(OrderStatus.FILLED, filled_price=10.0, commission=0.3)
        assert o.is_filled()
        assert o.filled_price == 10.0
        assert o.commission == 0.3
        assert o.filled_at is not None

    def test_pending_to_rejected_records_reason(self):
        o = Order.create("300750", OrderSide.BUY, 100, 10.0)
        o.transition(OrderStatus.REJECTED, reason=RejectionReason.INSUFFICIENT_CASH, detail="no cash")
        assert o.is_rejected()
        assert o.rejection_reason is RejectionReason.INSUFFICIENT_CASH
        assert o.rejection_detail == "no cash"

    def test_filled_is_terminal(self):
        o = Order.create("300750", OrderSide.BUY, 100, 10.0)
        o.transition(OrderStatus.FILLED)
        with pytest.raises(InvalidOrderTransition):
            o.transition(OrderStatus.REJECTED)

    def test_invalid_transition_raises(self):
        o = Order.create("300750", OrderSide.BUY, 100, 10.0)
        o.transition(OrderStatus.FILLED)  # FILLED 是终态
        with pytest.raises(InvalidOrderTransition):
            o.transition(OrderStatus.REJECTED)  # 终态不可再转移

    def test_pending_to_cancelled_allowed(self):
        o = Order.create("300750", OrderSide.BUY, 100, 10.0)
        o.transition(OrderStatus.CANCELLED)
        assert o.status is OrderStatus.CANCELLED

    def test_dict_roundtrip(self):
        o = Order.create("300750", OrderSide.SELL, 100, 10.0, idempotency_key="k1")
        o.transition(OrderStatus.FILLED, filled_price=10.5)
        d = o.to_dict()
        o2 = Order.from_dict(d)
        assert o2.order_id == o.order_id
        assert o2.side is OrderSide.SELL
        assert o2.status is OrderStatus.FILLED
        assert o2.filled_price == 10.5
        assert o2.idempotency_key == "k1"


# ---------------------------------------------------------------------------
# P2.3 SQLite 存储
# ---------------------------------------------------------------------------


class TestSqliteStore:
    def test_has_state_false_initially(self, tmp_path: Path):
        store = SqliteStateStore(str(tmp_path))
        assert store.has_state() is False
        assert store.load(100_000) is None

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        store = SqliteStateStore(str(tmp_path))
        pf = Portfolio(cash=100_000, commission=CommissionModel())
        pf.buy("300750", 100.0, 200, stop_loss=90.0, take_profit=120.0)
        pf.record_equity()
        order = Order.create("300750", OrderSide.BUY, 200, 100.0, idempotency_key="x")
        order.transition(OrderStatus.FILLED, filled_price=100.0)

        store.save(pf, [order])
        assert store.has_state()

        pf2, orders2 = store.load(100_000)
        exp_commission = pf2.commission.calc(100.0, 200, "buy")
        assert pf2.cash == pytest.approx(100_000 - 200 * 100 - exp_commission)
        pos = pf2.get_position("300750")
        assert pos is not None
        assert pos.shares == 200
        assert pos.stop_loss == pytest.approx(90.0)
        assert len(pf2.equity_curve) == 1
        assert len(orders2) == 1
        assert orders2[0].order_id == order.order_id
        assert orders2[0].status is OrderStatus.FILLED

    def test_reload_preserves_position_entry_date(self, tmp_path: Path):
        store = SqliteStateStore(str(tmp_path))
        pf = Portfolio(cash=100_000)
        pf.buy("300750", 100.0, 200)
        pf.positions["300750"].entry_date = "2024-01-02"
        store.save(pf, [])
        pf2, _ = store.load(100_000)
        assert pf2.get_position("300750").entry_date == "2024-01-02"


# ---------------------------------------------------------------------------
# P2.4 模拟交易规则（T+1 / 仓位 / 熔断）
# ---------------------------------------------------------------------------


class TestEnforcement:
    def test_default_no_enforcement_buy_sell_fill(self, tmp_path: Path):
        svc = PaperTradingService(str(tmp_path), initial_capital=100_000)
        trade = svc.buy("300750", 100.0, 200)
        assert trade is not None and trade.shares == 200
        trade2 = svc.sell("300750", 110.0, 200)
        assert trade2 is not None

    def test_t_plus_one_rejects_same_day_sell(self, tmp_path: Path):
        svc = PaperTradingService(
            str(tmp_path), initial_capital=100_000, enforce_t_plus_one=True
        )
        svc.buy("300750", 100.0, 200, trading_day="2024-01-02")
        order = svc.submit_order("300750", "SELL", 200, 110.0, trading_day="2024-01-02")
        assert order.is_rejected()
        assert order.rejection_reason is RejectionReason.T_PLUS_ONE

    def test_t_plus_one_allows_next_day_sell(self, tmp_path: Path):
        svc = PaperTradingService(
            str(tmp_path), initial_capital=100_000, enforce_t_plus_one=True
        )
        svc.buy("300750", 100.0, 200, trading_day="2024-01-02")
        order = svc.submit_order("300750", "SELL", 200, 110.0, trading_day="2024-01-03")
        assert order.is_filled()

    def test_position_limit_rejects_oversized_buy(self, tmp_path: Path):
        svc = PaperTradingService(
            str(tmp_path),
            initial_capital=100_000,
            enforce_position_limit=True,
            max_position_pct=0.20,
        )
        # 300 shares * 100 = 30k = 30% of 100k > 20%
        order = svc.submit_order("300750", "BUY", 300, 100.0, trading_day="2024-01-02")
        assert order.is_rejected()
        assert order.rejection_reason is RejectionReason.POSITION_LIMIT

    def test_position_limit_rejects_non_board_lot(self, tmp_path: Path):
        svc = PaperTradingService(
            str(tmp_path),
            initial_capital=100_000,
            enforce_position_limit=True,
            max_position_pct=0.20,
        )
        order = svc.submit_order("300750", "BUY", 150, 100.0, trading_day="2024-01-02")
        assert order.is_rejected()
        assert order.rejection_reason is RejectionReason.INVALID_QUANTITY

    def test_position_limit_allows_within_limit(self, tmp_path: Path):
        svc = PaperTradingService(
            str(tmp_path),
            initial_capital=100_000,
            enforce_position_limit=True,
            max_position_pct=0.20,
        )
        order = svc.submit_order("300750", "BUY", 100, 100.0, trading_day="2024-01-02")
        assert order.is_filled()

    def test_daily_loss_circuit_breaker(self, tmp_path: Path):
        svc = PaperTradingService(
            str(tmp_path),
            initial_capital=100_000,
            enforce_daily_loss=True,
            max_daily_loss_pct=-0.03,
        )
        svc.buy("300750", 100.0, 200, trading_day="2024-01-02")
        # 卖出亏损 30/股 * 200 = 6000  (> 3000 阈值)
        svc.sell("300750", 70.0, 200, trading_day="2024-01-02")
        order = svc.submit_order("600519", "BUY", 100, 100.0, trading_day="2024-01-02")
        assert order.is_rejected()
        assert order.rejection_reason is RejectionReason.DAILY_LOSS_LIMIT

    def test_sell_without_position_rejected(self, tmp_path: Path):
        svc = PaperTradingService(str(tmp_path), initial_capital=100_000)
        order = svc.submit_order("300750", "SELL", 100, 100.0)
        assert order.is_rejected()
        assert order.rejection_reason is RejectionReason.NO_POSITION
        # 旧 API 仍返回 None，保持兼容
        assert svc.sell("300750", 100.0, 100) is None


# ---------------------------------------------------------------------------
# P2.2 幂等
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_duplicate_idempotency_key_rejected(self, tmp_path: Path):
        svc = PaperTradingService(str(tmp_path), initial_capital=100_000)
        o1 = svc.submit_order(
            "300750", "BUY", 100, 100.0, idempotency_key="dup1", trading_day="2024-01-02"
        )
        o2 = svc.submit_order(
            "300750", "BUY", 100, 100.0, idempotency_key="dup1", trading_day="2024-01-02"
        )
        assert o1.is_filled()
        assert o2.is_rejected()
        assert o2.rejection_reason is RejectionReason.DUPLICATE_IDEMPOTENCY

    def test_idempotency_survives_restart(self, tmp_path: Path):
        svc = PaperTradingService(str(tmp_path), initial_capital=100_000)
        svc.submit_order(
            "300750", "BUY", 100, 100.0, idempotency_key="k", trading_day="2024-01-02"
        )
        svc2 = PaperTradingService(str(tmp_path), initial_capital=100_000)
        o2 = svc2.submit_order(
            "300750", "BUY", 100, 100.0, idempotency_key="k", trading_day="2024-01-02"
        )
        assert o2.is_rejected()
        assert o2.rejection_reason is RejectionReason.DUPLICATE_IDEMPOTENCY


# ---------------------------------------------------------------------------
# P2.5 结构化审计流
# ---------------------------------------------------------------------------


class TestAuditStream:
    def test_rejected_order_written_to_audit(self, tmp_path: Path):
        log_dir = tmp_path / "audit"
        al = AuditLogger(str(log_dir))
        svc = PaperTradingService(
            str(tmp_path),
            initial_capital=100_000,
            enforce_position_limit=True,
            max_position_pct=0.20,
            audit_logger=al,
        )
        svc.submit_order("300750", "BUY", 300, 100.0, trading_day="2024-01-02")

        files = list(log_dir.glob("audit_*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        import json

        rec = json.loads(lines[0])
        assert rec["event_type"] == "order_event"
        assert rec["stock_code"] == "300750"
        assert rec["status"] == "REJECTED"
        assert rec["rejection_reason"] == "POSITION_LIMIT"

    def test_filled_order_written_to_audit(self, tmp_path: Path):
        log_dir = tmp_path / "audit"
        al = AuditLogger(str(log_dir))
        svc = PaperTradingService(str(tmp_path), initial_capital=100_000, audit_logger=al)
        svc.submit_order("300750", "BUY", 100, 100.0, trading_day="2024-01-02")
        files = list(log_dir.glob("audit_*.jsonl"))
        rec = json.loads(files[0].read_text().strip())
        assert rec["status"] == "FILLED"
