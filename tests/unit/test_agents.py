"""Agent 框架单元测试"""

import pytest

from quant_agent.events.bus import EventBus, Event, EventType
from quant_agent.agents.base import BaseAgent, AgentResult
from quant_agent.agents.technical import TechnicalAgent
from quant_agent.agents.fundamental import FundamentalAgent
from quant_agent.agents.risk import RiskAgent


class TestEventBus:
    def test_publish_subscribe(self):
        bus = EventBus()
        received = []
        bus.subscribe(EventType.ANALYSIS_COMPLETED, lambda e: received.append(e))
        bus.publish_simple(EventType.ANALYSIS_COMPLETED, {"stock": "300750"})
        assert len(received) == 1
        assert received[0].payload["stock"] == "300750"

    def test_unsubscribe(self):
        bus = EventBus()
        handler = lambda e: None
        bus.subscribe(EventType.ANALYSIS_STARTED, handler)
        bus.unsubscribe(EventType.ANALYSIS_STARTED, handler)
        count = bus.publish_simple(EventType.ANALYSIS_STARTED)
        assert count == 0

    def test_history(self):
        bus = EventBus()
        bus.publish_simple(EventType.SYSTEM_HEARTBEAT)
        bus.publish_simple(EventType.SYSTEM_HEARTBEAT)
        assert len(bus.history) == 2

    def test_handler_error(self):
        bus = EventBus()
        def bad_handler(e):
            raise ValueError("test")
        bus.subscribe(EventType.SYSTEM_ERROR, bad_handler)
        count = bus.publish_simple(EventType.SYSTEM_ERROR)  # 不应抛出
        assert count == 0


class TestAgentResult:
    def test_to_dict(self):
        r = AgentResult(agent_name="test", stock_code="300750", signal="BUY", confidence=0.8)
        d = r.to_dict()
        assert d["signal"] == "BUY"
        assert d["confidence"] == 0.8

    def test_failed_result(self):
        r = AgentResult(agent_name="test", stock_code="300750", success=False, error="NO_DATA")
        assert not r.success
        assert r.error == "NO_DATA"


class TestFundamentalAgent:
    def test_no_data_service(self):
        agent = FundamentalAgent()
        result = agent.analyze("300750")
        assert not result.success
        assert result.signal == "HOLD"

    def test_with_mock_data(self):
        from quant_agent.data.sources.base import FinancialSnapshot
        agent = FundamentalAgent()
        # 直接注入 snapshot 模拟
        snapshot = FinancialSnapshot("300750", {
            "roe": 0.18, "pe_ttm": 25.0, "pb": 5.6,
            "gross_margin": 0.35, "net_margin": 0.12,
            "debt_ratio": 0.45, "current_ratio": 1.8,
            "revenue_growth": 0.25, "profit_growth": 0.30,
            "price": 400.0, "report_date": "2025-12-31",
        })
        scores = agent._calc_scores(snapshot)
        assert scores["profitability"] >= 5
        assert scores["health"] >= 5

    def test_signal_generation(self):
        from quant_agent.data.sources.base import FinancialSnapshot
        agent = FundamentalAgent()
        # 优秀基本面
        s = FinancialSnapshot("300750", {
            "roe": 0.20, "pe_ttm": 18.0, "pb": 3.0,
            "gross_margin": 0.40, "net_margin": 0.15,
            "debt_ratio": 0.35, "current_ratio": 2.0,
            "revenue_growth": 0.35, "profit_growth": 0.40,
        })
        scores = agent._calc_scores(s)
        signal, conf, reason = agent._generate_signal(s, scores)
        assert signal in ("BUY", "HOLD")

        # 差基本面
        s_bad = FinancialSnapshot("300750", {
            "roe": 0.05, "pe_ttm": 80.0, "pb": 8.0,
            "gross_margin": 0.15, "net_margin": 0.02,
            "debt_ratio": 0.85, "current_ratio": 0.6,
            "revenue_growth": -0.10, "profit_growth": -0.20,
        })
        scores_bad = agent._calc_scores(s_bad)
        signal_bad, _, _ = agent._generate_signal(s_bad, scores_bad)
        assert signal_bad in ("HOLD", "SELL")


class TestRiskAgent:
    def test_consensus_buy(self):
        agent = RiskAgent()
        results = [
            AgentResult("a1", "300750", signal="BUY", confidence=0.8, success=True),
            AgentResult("a2", "300750", signal="BUY", confidence=0.7, success=True),
            AgentResult("a3", "300750", signal="BUY", confidence=0.6, success=True),
        ]
        result = agent.analyze("300750", results)
        assert result.signal == "BUY"
        assert result.metrics["buy_count"] == 3

    def test_consensus_hold(self):
        agent = RiskAgent()
        results = [
            AgentResult("a1", "300750", signal="BUY", confidence=0.6, success=True),
            AgentResult("a2", "300750", signal="HOLD", confidence=0.5, success=True),
            AgentResult("a3", "300750", signal="SELL", confidence=0.5, success=True),
        ]
        result = agent.analyze("300750", results)
        assert result.signal == "HOLD"

    def test_position_calculation(self):
        agent = RiskAgent(max_position=0.20)
        results = [
            AgentResult("a1", "300750", signal="BUY", confidence=0.9, success=True, metrics={"current_price": 100}),
        ]
        result = agent.analyze("300750", results)
        assert result.metrics["position"] > 0
        assert result.metrics["position"] <= 0.20

    def test_no_results(self):
        agent = RiskAgent()
        result = agent.analyze("300750", [])
        assert not result.success
