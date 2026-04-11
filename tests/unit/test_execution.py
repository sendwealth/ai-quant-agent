"""执行层 + 可观测性单元测试"""

import json
from pathlib import Path

import pytest

from quant_agent.agents.execution import ExecutionAgent, Order, Position
from quant_agent.agents.base import AgentResult
from quant_agent.audit import AuditLogger
from quant_agent.observability.metrics import MetricsCollector, HealthChecker


class TestOrder:
    def test_create_order(self):
        o = Order(stock_code="300750", direction="buy", price=100.0, shares=100)
        assert o.status == "pending"
        assert o.shares == 100


class TestPosition:
    def test_update_price(self):
        p = Position(stock_code="300750", shares=100, avg_price=10.0)
        p.update_price(12.0)
        assert p.pnl == 200.0
        assert abs(p.pnl_pct - 0.2) < 0.001

    def test_stop_loss(self):
        p = Position(stock_code="300750", shares=100, avg_price=10.0, stop_loss=9.0)
        p.update_price(8.5)
        assert p.should_stop_loss()

    def test_take_profit(self):
        p = Position(stock_code="300750", shares=100, avg_price=10.0, take_profit=12.0)
        p.update_price(12.5)
        assert p.should_take_profit()


class TestExecutionAgent:
    def test_buy(self):
        agent = ExecutionAgent(initial_capital=100000)
        order = agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        assert order is not None
        assert order.status == "filled"
        assert order.shares > 0
        assert agent.cash < 100000
        assert "300750" in agent.positions

    def test_sell(self):
        agent = ExecutionAgent(initial_capital=100000)
        agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        order = agent.execute_signal("300750", "SELL", current_price=110.0)
        assert order is not None
        assert order.status == "filled"
        assert "300750" not in agent.positions
        assert agent.cash > 100000  # 盈利

    def test_stop_loss_trigger(self):
        agent = ExecutionAgent(initial_capital=100000)
        agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        order = agent.check_stop_conditions("300750", 90.0)  # 低于止损
        assert order is not None
        assert "300750" not in agent.positions

    def test_take_profit_trigger(self):
        agent = ExecutionAgent(initial_capital=100000)
        agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        order = agent.check_stop_conditions("300750", 125.0)  # 高于止盈
        assert order is not None

    def test_insufficient_funds(self):
        agent = ExecutionAgent(initial_capital=100)
        order = agent.execute_signal("300750", "BUY", position_pct=1.0, current_price=1000.0)
        assert order.status == "rejected"

    def test_hold(self):
        agent = ExecutionAgent(initial_capital=100000)
        order = agent.execute_signal("300750", "HOLD", current_price=100.0)
        assert order is None

    def test_total_return(self):
        agent = ExecutionAgent(initial_capital=100000)
        agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        assert agent.total_return < 0  # 手续费后
        agent.update_prices({"300750": 110.0})
        assert agent.total_return > 0  # 涨了

    def test_get_summary(self):
        agent = ExecutionAgent(initial_capital=100000)
        agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        s = agent.get_summary()
        assert "total_equity" in s
        assert s["positions"]["300750"]["shares"] > 0

    def test_record_equity(self):
        agent = ExecutionAgent(initial_capital=100000)
        agent.record_equity("2025-01-01T00:00:00")
        assert len(agent.equity_history) == 1
        assert agent.equity_history[0]["total_equity"] == 100000

    def test_analyze_status(self):
        agent = ExecutionAgent(initial_capital=100000)
        result = agent.analyze("300750")
        assert result.metrics["cash"] == 100000


class TestMetricsCollector:
    def test_counter(self):
        m = MetricsCollector()
        m.counter("orders", 1)
        m.counter("orders", 1)
        assert m.get("orders") == 2

    def test_gauge(self):
        m = MetricsCollector()
        m.gauge("equity", 100000.0)
        assert m.get("equity") == 100000.0

    def test_timer(self):
        import time
        m = MetricsCollector()
        with m.timer("analysis"):
            time.sleep(0.01)
        duration = m.get("analysis.duration_ms")
        assert duration is not None
        assert duration >= 10

    def test_query(self):
        m = MetricsCollector()
        m.gauge("cpu", 50.0)
        m.gauge("cpu", 60.0)
        results = m.query("cpu")
        assert len(results) == 2

    def test_counter_with_tags(self):
        m = MetricsCollector()
        m.counter("requests", 1, {"agent": "technical"})
        assert m.get("requests", {"agent": "technical"}) == 1


class TestHealthChecker:
    def test_healthy(self):
        h = HealthChecker()
        h.register("db", lambda: True)
        h.register("api", lambda: True)
        status = h.check()
        assert status.healthy

    def test_unhealthy(self):
        h = HealthChecker()
        h.register("db", lambda: False)
        status = h.check()
        assert not status.healthy
        assert "db: FAILED" in status.errors

    def test_error_handling(self):
        h = HealthChecker()
        h.register("db", lambda: (_ for _ in ()).throw(RuntimeError("connection failed")))
        status = h.check()
        assert not status.healthy
        assert any("ERROR" in e for e in status.errors)

    def test_check_all(self):
        h = HealthChecker()
        h.register("ok", lambda: True)
        result = h.check_all()
        assert "healthy" in result
        assert "timestamp" in result


class TestAuditLogger:
    """Tests for the append-only audit trail."""

    def test_creates_directory(self, tmp_path: Path):
        log_dir = tmp_path / "audit"
        al = AuditLogger(log_dir=str(log_dir))
        assert log_dir.is_dir()

    def test_writes_jsonl_entry(self, tmp_path: Path):
        log_dir = tmp_path / "audit"
        al = AuditLogger(log_dir=str(log_dir))
        al.log_trade_decision(
            stock_code="300750",
            signal="BUY",
            agent_results=[{"agent_name": "fundamental", "signal": "BUY", "confidence": 0.8, "reasoning": "strong"}],
            final_decision={"action": "BUY", "quantity": 200, "price": 100.0, "stop_loss": 92.0, "take_profit": 120.0},
        )
        files = list(log_dir.glob("audit_*.jsonl"))
        assert len(files) == 1
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["stock_code"] == "300750"
        assert entry["signal"] == "BUY"
        assert len(entry["agent_votes"]) == 1
        assert entry["final_decision"]["quantity"] == 200

    def test_append_mode(self, tmp_path: Path):
        log_dir = tmp_path / "audit"
        al = AuditLogger(log_dir=str(log_dir))
        for i in range(5):
            al.log_trade_decision(
                stock_code=f"30075{i}",
                signal="BUY",
                agent_results=[],
                final_decision={"action": "BUY", "quantity": 100},
            )
        files = list(log_dir.glob("audit_*.jsonl"))
        lines = files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 5

    def test_entry_has_timestamp(self, tmp_path: Path):
        log_dir = tmp_path / "audit"
        al = AuditLogger(log_dir=str(log_dir))
        al.log_trade_decision("300750", "HOLD", [], {"action": "HOLD"})
        entry = json.loads(list(log_dir.glob("audit_*.jsonl"))[0].read_text().strip())
        assert "timestamp" in entry
        # ISO format must contain 'T'
        assert "T" in entry["timestamp"]


class TestExecutionAgentAudit:
    """Tests that ExecutionAgent correctly delegates to the audit logger."""

    def test_buy_with_audit(self, tmp_path: Path):
        al = AuditLogger(log_dir=str(tmp_path / "audit"))
        agent = ExecutionAgent(initial_capital=100000, audit_logger=al)
        results = [
            AgentResult("fundamental", "300750", signal="BUY", confidence=0.8, reasoning="good"),
            AgentResult("risk", "300750", signal="BUY", confidence=0.7, reasoning="consensus"),
        ]
        order = agent.execute_signal(
            "300750", "BUY", position_pct=0.20, current_price=100.0,
            agent_results=results,
        )
        assert order is not None
        assert order.status == "filled"

        # Verify audit log
        lines = list((tmp_path / "audit").glob("audit_*.jsonl"))[0].read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["stock_code"] == "300750"
        assert entry["signal"] == "BUY"
        assert len(entry["agent_votes"]) == 2
        assert entry["agent_votes"][0]["agent_name"] == "fundamental"
        assert entry["final_decision"]["order_status"] == "filled"
        assert entry["final_decision"]["quantity"] > 0

    def test_hold_with_audit(self, tmp_path: Path):
        al = AuditLogger(log_dir=str(tmp_path / "audit"))
        agent = ExecutionAgent(initial_capital=100000, audit_logger=al)
        results = [AgentResult("risk", "300750", signal="HOLD", confidence=0.5, reasoning="mixed")]
        order = agent.execute_signal(
            "300750", "HOLD", current_price=100.0,
            agent_results=results,
        )
        assert order is None
        lines = list((tmp_path / "audit").glob("audit_*.jsonl"))[0].read_text().strip().splitlines()
        entry = json.loads(lines[0])
        assert entry["signal"] == "HOLD"
        # HOLD decisions have no order, so order_status is absent
        assert "order_status" not in entry["final_decision"]
        assert entry["final_decision"]["action"] == "HOLD"

    def test_no_audit_without_logger(self, tmp_path: Path):
        agent = ExecutionAgent(initial_capital=100000)
        # Should not raise even though there is no audit logger
        order = agent.execute_signal("300750", "BUY", position_pct=0.20, current_price=100.0)
        assert order is not None
