"""Tests for LLMReportGenerator — comprehensive analysis report generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.llm.client import LLMClient
from quant_agent.llm.report import LLMReportGenerator


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_agent_result(
    agent_name: str = "test",
    signal: str = "BUY",
    confidence: float = 0.8,
    reasoning: str = "Test reasoning",
    metrics: Optional[dict] = None,
    success: bool = True,
) -> AgentResult:
    return AgentResult(
        agent_name=agent_name,
        stock_code="300750",
        signal=signal,
        confidence=confidence,
        reasoning=reasoning,
        metrics=metrics or {},
        success=success,
    )


def _make_report(**overrides):
    """Create a minimal AnalysisReport-like object.

    The orchestrator module's AnalysisReport is a dataclass. We replicate its
    fields here to avoid importing the full orchestrator (which may have heavy
    dependencies) and to keep tests self-contained.
    """
    defaults = {
        "stock_code": "300750",
        "timestamp": "2026-04-11T10:00:00",
        "fundamental_result": _make_agent_result(
            agent_name="fundamental",
            signal="BUY",
            confidence=0.85,
            reasoning="Strong ROE and revenue growth.",
            metrics={"roe": 0.18, "revenue_growth": 0.25},
        ),
        "technical_result": _make_agent_result(
            agent_name="technical",
            signal="BUY",
            confidence=0.72,
            reasoning="MACD golden cross, RSI at 55.",
            metrics={"macd_signal": "golden_cross", "rsi": 55},
        ),
        "sentiment_result": _make_agent_result(
            agent_name="sentiment",
            signal="BUY",
            confidence=0.65,
            reasoning="Positive news coverage.",
            metrics={"sentiment_score": 0.45, "key_factors": ["EV policy"]},
        ),
        "risk_result": _make_agent_result(
            agent_name="risk",
            signal="HOLD",
            confidence=0.6,
            reasoning="Moderate volatility.",
            metrics={"position": 0.15},
        ),
        "execution_result": None,
        "llm_analysis": None,
        "risk_interpretation": None,
        "summary": {"total_equity": 100000, "total_return": 0.05},
    }
    defaults.update(overrides)

    # Build a simple namespace with .get() support for summary
    report = MagicMock()
    for k, v in defaults.items():
        setattr(report, k, v)
    # summary needs .get() dict behavior
    report.summary = defaults["summary"]
    return report


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLLMReportGeneratorGenerate:
    """LLMReportGenerator.generate — full report generation."""

    def test_generate_full_report(self):
        """generate() returns the LLM response text for a complete report."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = (
            "# Investment Analysis Report\n\n"
            "## Summary\nCATL shows strong fundamentals..."
        )

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report()
        result = gen.generate(report)

        assert "Investment Analysis Report" in result
        mock_llm.invoke.assert_called_once()

    def test_generate_passes_system_and_user_prompts(self):
        """generate() calls llm.invoke with REPORT_SYSTEM and a formatted user prompt."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report text"

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report()
        gen.generate(report)

        call_args = mock_llm.invoke.call_args
        system_arg = call_args[0][0]
        user_arg = call_args[0][1]

        # System prompt should mention investment analyst role
        assert "投资分析师" in system_arg or "analyst" in system_arg.lower()

        # User prompt should contain stock code and agent results
        assert "300750" in user_arg
        assert "BUY" in user_arg

    def test_generate_with_missing_sentiment(self):
        """generate() works when sentiment_result is None."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report without sentiment."

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report(sentiment_result=None)
        result = gen.generate(report)

        assert result == "Report without sentiment."
        user_arg = mock_llm.invoke.call_args[0][1]
        # Sentiment section should be absent
        assert "情感分析" not in user_arg

    def test_generate_with_failed_sentiment(self):
        """generate() omits sentiment when sentiment_result.success is False."""
        failed_sentiment = _make_agent_result(
            agent_name="sentiment",
            signal="HOLD",
            success=False,
        )
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report text"

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report(sentiment_result=failed_sentiment)
        gen.generate(report)

        user_arg = mock_llm.invoke.call_args[0][1]
        # Failed sentiment should not produce sentiment_section
        assert "情感分析" not in user_arg

    def test_generate_with_all_none_agents(self):
        """generate() handles gracefully when all agent results are None."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Minimal report."

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report(
            fundamental_result=None,
            technical_result=None,
            sentiment_result=None,
            risk_result=None,
        )
        result = gen.generate(report)

        assert result == "Minimal report."
        user_arg = mock_llm.invoke.call_args[0][1]
        # All signals should show N/A
        assert "N/A" in user_arg

    def test_generate_prompt_contains_fundamental_data(self):
        """User prompt includes fundamental signal and metrics."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report"

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report()
        gen.generate(report)

        user_arg = mock_llm.invoke.call_args[0][1]
        assert "BUY" in user_arg  # fundamental signal
        assert "Strong ROE" in user_arg  # fundamental reasoning
        assert "roe" in user_arg  # fundamental metrics JSON

    def test_generate_prompt_contains_risk_position(self):
        """User prompt includes risk position as percentage."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report"

        gen = LLMReportGenerator(llm=mock_llm)
        risk_result = _make_agent_result(
            agent_name="risk",
            metrics={"position": 0.15},
        )
        report = _make_report(risk_result=risk_result)
        gen.generate(report)

        user_arg = mock_llm.invoke.call_args[0][1]
        assert "15.0%" in user_arg  # position 0.15 * 100

    def test_generate_with_empty_risk_result(self):
        """Risk position defaults to 0 when risk_result is None."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report"

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report(risk_result=None)
        gen.generate(report)

        user_arg = mock_llm.invoke.call_args[0][1]
        # When risk_result is None, position is 0*100 = 0, formatted as "0%"
        assert "0%" in user_arg

    def test_generate_prompt_contains_summary(self):
        """User prompt includes portfolio summary numbers."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report"

        gen = LLMReportGenerator(llm=mock_llm)
        report = _make_report(summary={"total_equity": 250000, "total_return": 0.12})
        gen.generate(report)

        user_arg = mock_llm.invoke.call_args[0][1]
        assert "250,000.00" in user_arg
        assert "12.00%" in user_arg

    def test_generate_with_successful_sentiment_includes_section(self):
        """When sentiment result is successful, the prompt includes sentiment section."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.invoke.return_value = "Report"

        gen = LLMReportGenerator(llm=mock_llm)
        sentiment = _make_agent_result(
            agent_name="sentiment",
            signal="BUY",
            confidence=0.65,
            reasoning="Positive EV policy sentiment.",
            metrics={"sentiment_score": 0.45},
        )
        report = _make_report(sentiment_result=sentiment)
        gen.generate(report)

        user_arg = mock_llm.invoke.call_args[0][1]
        assert "情感分析" in user_arg
        assert "Positive EV policy" in user_arg
