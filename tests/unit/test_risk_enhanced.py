"""风控体系增强测试 — T+1、日亏损熔断、组合级限制"""

import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.agents.risk import RiskAgent, T1Tracker, DailyPnLTracker


# ── T+1 Tracker ─────────────────────────────────────────────────────────────

class TestT1Tracker:
    def test_can_sell_after_t1(self):
        """Bought on D-1, selling on D should be allowed."""
        tracker = T1Tracker()
        tracker.record_buy("300750", "2025-01-01")
        assert tracker.can_sell("300750", "2025-01-02") is True

    def test_cannot_sell_same_day(self):
        """Bought and selling on the same day should be blocked."""
        tracker = T1Tracker()
        tracker.record_buy("300750", "2025-01-01")
        assert tracker.can_sell("300750", "2025-01-01") is False

    def test_unknown_stock_allowed(self):
        """Stock not in tracker (pre-existing position) should be allowed."""
        tracker = T1Tracker()
        assert tracker.can_sell("600519", "2025-01-01") is True

    def test_clear_removes_tracking(self):
        tracker = T1Tracker()
        tracker.record_buy("300750", "2025-01-01")
        tracker.clear("300750")
        assert tracker.can_sell("300750", "2025-01-01") is True


# ── Daily P&L Circuit Breaker ───────────────────────────────────────────────

class TestDailyPnLTracker:
    def test_no_circuit_on_small_loss(self):
        tracker = DailyPnLTracker(max_daily_loss_pct=-0.03)
        tracker.update_date("2025-01-01", 100000)
        triggered, pnl = tracker.check_circuit_breaker(98500)
        assert triggered is False
        assert abs(pnl - (-0.015)) < 0.001

    def test_circuit_on_large_loss(self):
        tracker = DailyPnLTracker(max_daily_loss_pct=-0.03)
        tracker.update_date("2025-01-01", 100000)
        triggered, pnl = tracker.check_circuit_breaker(96000)
        assert triggered is True
        assert pnl == -0.04

    def test_new_day_resets(self):
        tracker = DailyPnLTracker(max_daily_loss_pct=-0.03)
        tracker.update_date("2025-01-01", 100000)
        tracker.check_circuit_breaker(96000)  # big loss
        # New day resets
        tracker.update_date("2025-01-02", 96000)
        triggered, _ = tracker.check_circuit_breaker(95500)
        assert triggered is False  # -500/96000 = -0.52%, under 3%


# ── RiskAgent Portfolio-Level Controls ──────────────────────────────────────

class TestRiskAgentPortfolio:
    def _buy_signal(self, price=100.0):
        return AgentResult("a1", "300750", signal="BUY", confidence=0.9,
                           success=True, metrics={"current_price": price})

    def test_portfolio_heat_reduces_position(self):
        """Position should be reduced when portfolio is already heavily invested."""
        agent = RiskAgent(max_position=0.20, max_portfolio_risk=0.50)
        results = [self._buy_signal()]

        # Already at 45% invested — only 5% headroom
        current_positions = {"600519": 45000.0}
        result = agent.analyze(
            "300750", results,
            current_positions=current_positions,
            current_equity=100000,
            current_date="2025-01-02",
        )
        assert result.signal == "BUY"
        assert result.metrics["position"] <= 0.05  # capped by headroom

    def test_portfolio_full_blocks_buy(self):
        """Position should be zero when portfolio is at max risk."""
        agent = RiskAgent(max_position=0.20, max_portfolio_risk=0.50)
        results = [self._buy_signal()]

        # Already at 55% invested — exceeds 50% limit
        current_positions = {"600519": 55000.0}
        result = agent.analyze(
            "300750", results,
            current_positions=current_positions,
            current_equity=100000,
            current_date="2025-01-02",
        )
        assert result.signal == "BUY"
        assert result.metrics["position"] == 0.0
        assert any("组合仓位已达上限" in w for w in result.metrics.get("position_warnings", []))

    def test_t1_blocks_sell(self):
        """SELL on same day as BUY should be downgraded to HOLD."""
        agent = RiskAgent()
        # Record a buy today
        agent.t1_tracker.record_buy("300750", "2025-01-01")

        sell_results = [
            AgentResult("a1", "300750", signal="SELL", confidence=0.8, success=True),
            AgentResult("a2", "300750", signal="SELL", confidence=0.8, success=True),
        ]
        result = agent.analyze(
            "300750", sell_results,
            current_date="2025-01-01",
        )
        # SELL should be downgraded to HOLD due to T+1
        assert result.signal == "HOLD"
        assert any("T+1" in w for w in result.metrics.get("position_warnings", []))

    def test_daily_circuit_breaker_blocks_buy(self):
        """BUY should be blocked when daily loss exceeds threshold."""
        agent = RiskAgent(max_daily_loss_pct=-0.03)
        agent.daily_tracker.update_date("2025-01-01", 100000)

        results = [self._buy_signal()]
        result = agent.analyze(
            "300750", results,
            current_equity=96000,  # 4% daily loss
            current_date="2025-01-01",
        )
        # BUY downgraded to HOLD due to circuit breaker
        assert result.signal == "HOLD"
        assert result.metrics["position"] == 0.0

    def test_no_portfolio_context_works(self):
        """Without portfolio context, should still work (backward compat)."""
        agent = RiskAgent()
        results = [self._buy_signal()]
        result = agent.analyze("300750", results)
        assert result.signal == "BUY"
        assert result.metrics["position"] > 0
