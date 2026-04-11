"""End-to-end integration tests for the full Orchestrator analysis pipeline.

Exercises the complete pipeline:
  DataService -> FundamentalAgent -> TechnicalAgent -> RiskAgent -> ExecutionAgent

All external dependencies (data sources, file I/O) are mocked so tests run
without network access or real API tokens.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock, patch
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.data.sources.base import FinancialSnapshot
from quant_agent.orchestrator import AnalysisReport, Orchestrator


# ---------------------------------------------------------------------------
# Helpers -- realistic mock data factories
# ---------------------------------------------------------------------------


def make_uptrend_price_df(n_bars: int = 120, base_price: float = 200.0) -> pd.DataFrame:
    """Generate a realistic uptrend price DataFrame with proper OHLCV columns.

    Uses a smooth upward drift with small noise so that technical indicators
    (EMA crossover, MACD golden cross) will produce BUY signals.
    """
    np.random.seed(42)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)
    drift = np.linspace(0, base_price * 0.30, n_bars)  # +30% over the window
    noise = np.cumsum(np.random.randn(n_bars) * 1.5)
    close = base_price + drift + noise

    high = close + np.abs(np.random.randn(n_bars)) * 3
    low = close - np.abs(np.random.randn(n_bars)) * 3
    open_ = close - np.random.randn(n_bars) * 0.5
    volume = np.random.uniform(500_000, 2_000_000, n_bars)

    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_downtrend_price_df(n_bars: int = 120, base_price: float = 200.0) -> pd.DataFrame:
    """Generate a realistic downtrend price DataFrame.

    Downward drift so that EMA-20 < EMA-50 and MACD is a death cross,
    yielding SELL signals from the technical agent.
    """
    np.random.seed(99)
    dates = pd.bdate_range("2025-01-01", periods=n_bars)
    drift = np.linspace(0, -base_price * 0.30, n_bars)
    noise = np.cumsum(np.random.randn(n_bars) * 1.5)
    close = base_price + drift + noise

    # Ensure all prices are positive
    close = np.maximum(close, 10.0)

    high = close + np.abs(np.random.randn(n_bars)) * 2
    low = close - np.abs(np.random.randn(n_bars)) * 2
    open_ = close - np.random.randn(n_bars) * 0.5
    volume = np.random.uniform(500_000, 2_000_000, n_bars)

    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def make_strong_financial_snapshot(stock_code: str = "300750") -> FinancialSnapshot:
    """Strong fundamentals: high ROE, low PE, low debt, high growth."""
    return FinancialSnapshot(stock_code, {
        "roe": 0.22,
        "pe_ttm": 14.0,
        "pb": 2.8,
        "gross_margin": 0.42,
        "net_margin": 0.18,
        "debt_ratio": 0.30,
        "current_ratio": 2.5,
        "revenue_growth": 0.35,
        "profit_growth": 0.40,
        "price": 220.0,
        "report_date": "2025-12-31",
    })


def make_weak_financial_snapshot(stock_code: str = "300750") -> FinancialSnapshot:
    """Weak fundamentals: low ROE, high PE, high debt, negative growth."""
    return FinancialSnapshot(stock_code, {
        "roe": 0.03,
        "pe_ttm": 90.0,
        "pb": 9.0,
        "gross_margin": 0.12,
        "net_margin": 0.01,
        "debt_ratio": 0.88,
        "current_ratio": 0.5,
        "revenue_growth": -0.25,
        "profit_growth": -0.40,
        "price": 100.0,
        "report_date": "2025-12-31",
    })


def make_mock_settings(tmp_path):
    """Build a mock Settings object with sensible defaults for testing."""
    settings = MagicMock()
    settings.parquet_dir = str(tmp_path / "parquet")
    settings.data_dir = str(tmp_path / "data")
    settings.tushare_token = None
    settings.akshare_timeout = 5
    settings.fetch_max_workers = 1
    settings.max_position_pct = 0.20
    settings.max_portfolio_risk = 0.80
    settings.max_daily_loss_pct = -0.03
    settings.default_stop_loss = -0.08
    settings.default_take_profit_1 = 0.10
    settings.default_take_profit_2 = 0.20
    settings.commission_rate = 0.0003
    settings.stamp_tax_rate = 0.001
    settings.debug = True
    # LLM — no API keys in tests → LLMClient will be None
    settings.openai_api_key = None
    settings.zhipu_api_key = None
    settings.openai_model = "gpt-4o"
    settings.zhipu_model = "glm-4"
    settings.openai_base_url = "https://api.openai.com/v1"
    settings.llm_timeout = 5
    settings.llm_max_retries = 0
    settings.email_enabled = False
    settings.email_smtp_server = ""
    settings.email_smtp_port = 465
    settings.email_sender = ""
    settings.email_password = ""
    settings.email_recipients = ""
    return settings


def build_orchestrator_with_mocks(
    mock_settings,
    price_data: Optional[pd.DataFrame],
    financial_snapshot: Optional[FinancialSnapshot],
) -> Orchestrator:
    """Build an Orchestrator whose DataService is fully mocked.

    Mocks ``DataService.get_price_data`` and
    ``DataService.get_financial_snapshot`` so the pipeline runs against
    controlled data without hitting real APIs.

    Also patches ``AuditLogger`` to avoid filesystem side-effects.
    """
    with patch("quant_agent.orchestrator.DataService") as MockDS, \
         patch("quant_agent.orchestrator.AuditLogger") as MockAudit:
        # Build a mock DataService instance
        mock_ds_instance = MagicMock()
        mock_ds_instance.get_price_data.return_value = price_data
        mock_ds_instance.get_financial_snapshot.return_value = financial_snapshot

        # Make the constructor return our mock instance
        MockDS.return_value = mock_ds_instance

        # Mock AuditLogger so it doesn't write to disk
        mock_audit = MagicMock()
        MockAudit.return_value = mock_audit

        orch = Orchestrator(settings=mock_settings)

    # After construction, ensure the data_service mocks stick
    orch.data.get_price_data.return_value = price_data
    orch.data.get_financial_snapshot.return_value = financial_snapshot

    return orch


# ===========================================================================
# Test 1 -- Full bullish pipeline
# ===========================================================================


class TestBullishPipeline:
    """Uptrend prices + strong fundamentals should produce BUY signals."""

    def test_fundamental_buy_signal(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        assert report.fundamental_result is not None
        assert report.fundamental_result.success is True
        assert report.fundamental_result.signal == "BUY"

    def test_technical_buy_signal(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        assert report.technical_result is not None
        assert report.technical_result.success is True
        assert report.technical_result.signal == "BUY"

    def test_risk_buy_consensus(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        assert report.risk_result is not None
        assert report.risk_result.signal == "BUY"
        assert report.risk_result.confidence > 0.0

    def test_report_fully_populated(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        # All four result slots filled
        assert report.fundamental_result is not None
        assert report.technical_result is not None
        assert report.risk_result is not None
        # execution_result may be None (no order) but summary must exist
        assert isinstance(report.summary, dict)

        # to_dict produces a valid dict with all expected keys
        d = report.to_dict()
        assert d["stock_code"] == "300750"
        assert d["signal"] == "BUY"
        assert d["confidence"] > 0.0
        assert d["fundamental"] is not None
        assert d["technical"] is not None
        assert d["risk"] is not None
        assert "total_equity" in d["summary"]

    def test_position_opened_on_buy(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        # A BUY with position > 0 and a valid current_price should open a
        # position in the execution agent's portfolio.
        assert report.risk_result.metrics["position"] > 0
        summary = report.summary
        # The stock should appear in positions detail
        assert "300750" in summary["positions"]


# ===========================================================================
# Test 2 -- Full bearish pipeline
# ===========================================================================


class TestBearishPipeline:
    """Downtrend prices + weak fundamentals should produce SELL or HOLD."""

    def test_risk_signal_is_sell_or_hold(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_weak_financial_snapshot(),
        )
        report = orch.analyze("300750")

        assert report.risk_result is not None
        assert report.risk_result.signal in ("SELL", "HOLD")

    def test_fundamental_sell_signal(self, tmp_path):
        """Very weak fundamentals should generate a SELL signal."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_weak_financial_snapshot(),
        )
        report = orch.analyze("300750")

        assert report.fundamental_result is not None
        assert report.fundamental_result.signal in ("SELL", "HOLD")

    def test_technical_sell_signal(self, tmp_path):
        """Downtrend should generate a SELL signal from technical agent."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_weak_financial_snapshot(),
        )
        report = orch.analyze("300750")

        assert report.technical_result is not None
        assert report.technical_result.signal in ("SELL", "HOLD")

    def test_no_buy_position_in_bearish(self, tmp_path):
        """Bearish pipeline should not open a new BUY position."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_weak_financial_snapshot(),
        )
        report = orch.analyze("300750")

        if report.risk_result.signal != "BUY":
            # Position should remain 0
            assert report.risk_result.metrics["position"] == 0.0


# ===========================================================================
# Test 3 -- Data failure graceful degradation
# ===========================================================================


class TestDataFailureGracefulDegradation:
    """When DataService returns None, agents should fail gracefully."""

    def test_no_price_data_agents_return_failed(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=None,
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        # Technical agent should fail (no price data)
        assert report.technical_result is not None
        assert report.technical_result.success is False

    def test_no_financial_data_agent_returns_failed(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=None,
        )
        report = orch.analyze("300750")

        # Fundamental agent should fail (no financial data)
        assert report.fundamental_result is not None
        assert report.fundamental_result.success is False

    def test_no_data_at_all(self, tmp_path):
        """Both data sources returning None -- entire pipeline should still
        produce a report without crashing."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=None,
            financial_snapshot=None,
        )
        report = orch.analyze("300750")

        assert isinstance(report, AnalysisReport)
        assert report.fundamental_result.success is False
        assert report.technical_result.success is False
        # Risk agent sees only failures -> HOLD
        assert report.risk_result.signal == "HOLD"

    def test_report_to_dict_with_failures(self, tmp_path):
        """to_dict should work even when agents fail."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=None,
            financial_snapshot=None,
        )
        report = orch.analyze("300750")
        d = report.to_dict()

        assert d["stock_code"] == "300750"
        assert d["fundamental"]["success"] is False
        assert d["technical"]["success"] is False
        assert isinstance(d["summary"], dict)


# ===========================================================================
# Test 4 -- Mixed signals (fundamental BUY, technical SELL)
# ===========================================================================


class TestMixedSignals:
    """Strong financial data but downtrend price -- conflicting signals."""

    def test_risk_weighs_conflicting_signals(self, tmp_path):
        """RiskAgent should produce a reasoned consensus when signals disagree."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        # Fundamental should say BUY, technical should say SELL or HOLD
        assert report.fundamental_result.signal == "BUY"
        assert report.technical_result.signal in ("SELL", "HOLD")

        # Risk agent must produce a valid signal (no crash)
        assert report.risk_result is not None
        assert report.risk_result.signal in ("BUY", "SELL", "HOLD")

    def test_mixed_signals_buy_sell_count(self, tmp_path):
        """The risk metrics should correctly count buy/sell votes."""
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        metrics = report.risk_result.metrics
        assert "buy_count" in metrics
        assert "sell_count" in metrics
        assert "hold_count" in metrics
        # At least 1 BUY (fundamental)
        assert metrics["buy_count"] >= 1

    def test_risk_reasoning_mentions_consensus(self, tmp_path):
        """RiskAgent reasoning should describe the consensus.

        The reasoning is in Chinese and contains the consensus signal,
        buy/sell/hold counts, and average confidence.
        """
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        reasoning = report.risk_result.reasoning
        # Reasoning uses Chinese: "共识: <SIGNAL> (买N/卖N/观N)"
        # Contains signal name and vote breakdown
        assert "HOLD" in reasoning or "BUY" in reasoning or "SELL" in reasoning
        assert "confidence" in reasoning.lower() or "信心度" in reasoning


# ===========================================================================
# Test 5 -- Multi-stock independent analysis
# ===========================================================================


class TestMultiStockAnalysis:
    """Analyzing multiple stocks should produce independent results."""

    def test_three_stocks_independent(self, tmp_path):
        stocks = ["300750", "601318", "000001"]
        results: dict[str, AnalysisReport] = {}

        for code in stocks:
            orch = build_orchestrator_with_mocks(
                make_mock_settings(tmp_path),
                price_data=make_uptrend_price_df(),
                financial_snapshot=make_strong_financial_snapshot(),
            )
            results[code] = orch.analyze(code)

        # Each stock gets its own report
        assert len(results) == 3
        for code, report in results.items():
            assert report.stock_code == code
            assert report.fundamental_result is not None
            assert report.technical_result is not None
            assert report.risk_result is not None

    def test_independent_portfolios(self, tmp_path):
        """Each Orchestrator has its own execution portfolio, so analyzing
        different stocks should not cross-contaminate positions."""
        orch_a = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report_a = orch_a.analyze("300750")

        orch_b = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report_b = orch_b.analyze("601318")

        # Orchestrator A should only have 300750 in positions
        pos_a = report_a.summary.get("positions", {})
        pos_b = report_b.summary.get("positions", {})

        # If a BUY was executed, check position isolation
        if "300750" in pos_a:
            assert "601318" not in pos_a
        if "601318" in pos_b:
            assert "300750" not in pos_b

    def test_mixed_bullish_bearish(self, tmp_path):
        """Run bullish and bearish analyses -- signals should differ."""
        orch_bull = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report_bull = orch_bull.analyze("300750")

        orch_bear = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_downtrend_price_df(),
            financial_snapshot=make_weak_financial_snapshot(),
        )
        report_bear = orch_bear.analyze("601318")

        assert report_bull.risk_result.signal == "BUY"
        assert report_bear.risk_result.signal in ("SELL", "HOLD")
        # Confidence on bullish should be higher
        assert report_bull.risk_result.confidence >= report_bear.risk_result.confidence


# ===========================================================================
# Test 6 -- AnalysisReport data structure
# ===========================================================================


class TestAnalysisReportStructure:
    """Verify AnalysisReport fields and serialization work correctly."""

    def test_to_dict_has_all_keys(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")
        d = report.to_dict()

        expected_keys = {
            "stock_code", "timestamp", "signal", "confidence",
            "position_pct", "fundamental", "technical", "risk",
            "execution", "summary",
        }
        assert expected_keys.issubset(d.keys())

    def test_timestamp_is_iso_format(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")

        # Should be parseable as an ISO timestamp
        from datetime import datetime
        datetime.fromisoformat(report.timestamp)

    def test_stock_code_preserved(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")
        assert report.stock_code == "300750"
        assert report.to_dict()["stock_code"] == "300750"

    def test_fundamental_metrics_include_key_ratios(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")
        metrics = report.fundamental_result.metrics

        for key in ("roe", "pe_ttm", "pb", "debt_ratio", "gross_margin", "net_margin"):
            assert key in metrics, f"Missing fundamental metric: {key}"

    def test_technical_metrics_include_indicators(self, tmp_path):
        orch = build_orchestrator_with_mocks(
            make_mock_settings(tmp_path),
            price_data=make_uptrend_price_df(),
            financial_snapshot=make_strong_financial_snapshot(),
        )
        report = orch.analyze("300750")
        metrics = report.technical_result.metrics

        for key in ("rsi", "macd", "macd_status", "ema_trend", "atr", "adx", "current_price"):
            assert key in metrics, f"Missing technical metric: {key}"
