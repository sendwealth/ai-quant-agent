"""Tests for SentimentAgent -- LLM-based market news sentiment analysis."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.agents.sentiment import SentimentAgent
from quant_agent.llm.client import LLMClient, LLMError


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _mock_news(count: int = 3, positive: bool = True) -> list[dict]:
    """Create mock news items."""
    if positive:
        items = [
            {"title": f"Positive news {i}", "content": f"Good development {i}", "date": "2026-04-10"}
            for i in range(1, count + 1)
        ]
    else:
        items = [
            {"title": f"Negative news {i}", "content": f"Bad development {i}", "date": "2026-04-10"}
            for i in range(1, count + 1)
        ]
    return items


def _make_llm_response(
    signal: str = "BUY",
    confidence: float = 0.75,
    sentiment_score: float = 0.6,
    reasoning: str = "Strong positive momentum.",
    key_factors: list[str] | None = None,
) -> str:
    """Build a JSON string that the LLM would return for sentiment analysis."""
    return json.dumps({
        "signal": signal,
        "confidence": confidence,
        "sentiment_score": sentiment_score,
        "reasoning": reasoning,
        "key_factors": key_factors or ["factor_a", "factor_b"],
    })


def _make_llm_response_in_fence(**kwargs) -> str:
    """LLM response wrapped in markdown code fences (common LLM behavior)."""
    inner = _make_llm_response(**kwargs)
    return f"```json\n{inner}\n```"


def _create_agent(
    news_data: list[dict] | None = None,
    llm_response: str | None = None,
    llm_side_effect: Exception | None = None,
    data_service: MagicMock | None = None,
) -> SentimentAgent:
    """Create a SentimentAgent with all external deps mocked.

    Args:
        news_data: News items to return from data service.
        llm_response: Text the mock LLM will return from invoke().
        llm_side_effect: If set, llm.invoke() raises this.
        data_service: Pre-built data service mock.
    """
    # Set up data service mock
    if data_service is None:
        data_service = MagicMock()
        if news_data is not None:
            data_service.get_news.return_value = news_data
            data_service.akshare = data_service
        else:
            data_service.get_news.return_value = []
            data_service.akshare = data_service

    # Set up LLM client mock
    llm_client = MagicMock(spec=LLMClient)
    # Force structured_output to fail so tests use the invoke+parse fallback
    llm_client.structured_output.side_effect = LLMError("structured output not configured in test")
    if llm_side_effect:
        llm_client.invoke.side_effect = llm_side_effect
    elif llm_response is not None:
        llm_client.invoke.return_value = llm_response
    else:
        llm_client.invoke.return_value = _make_llm_response()

    agent = SentimentAgent(data_service=data_service, llm_client=llm_client)
    return agent


# ── Analyze tests ────────────────────────────────────────────────────────────


class TestSentimentAgentAnalyze:
    """SentimentAgent.analyze — main analysis entry point."""

    def test_analyze_returns_buy_with_positive_news(self):
        """Positive news yields a BUY signal with high confidence."""
        agent = _create_agent(
            news_data=_mock_news(5, positive=True),
            llm_response=_make_llm_response(
                signal="BUY", confidence=0.8, sentiment_score=0.7,
            ),
        )

        result = agent.analyze("300750")

        assert isinstance(result, AgentResult)
        assert result.signal == "BUY"
        assert result.confidence == 0.8
        assert result.success is True
        assert result.agent_name == "sentiment"
        assert result.stock_code == "300750"
        assert result.metrics["news_count"] == 5
        assert result.metrics["sentiment_score"] == 0.7

    def test_analyze_returns_sell_with_negative_news(self):
        """Negative news yields a SELL signal."""
        agent = _create_agent(
            news_data=_mock_news(3, positive=False),
            llm_response=_make_llm_response(
                signal="SELL", confidence=0.6, sentiment_score=-0.5,
            ),
        )

        result = agent.analyze("600519")

        assert result.signal == "SELL"
        assert result.confidence == 0.6
        assert result.metrics["sentiment_score"] == -0.5

    def test_analyze_with_empty_news_returns_hold_low_confidence(self):
        """No news data results in HOLD with low confidence."""
        agent = _create_agent(
            news_data=[],
            llm_response=_make_llm_response(
                signal="HOLD", confidence=0.3, sentiment_score=0.0,
                reasoning="No sufficient news data",
            ),
        )

        result = agent.analyze("300750")

        assert result.signal == "HOLD"
        assert result.confidence == 0.3
        assert result.metrics["news_count"] == 0

    def test_analyze_with_no_llm_client_returns_failed_result(self):
        """When LLM client raises, a failed AgentResult is returned."""
        agent = _create_agent(
            llm_side_effect=LLMError("API key missing"),
        )

        result = agent.analyze("300750")

        assert result.success is False
        assert result.signal == "HOLD"
        assert result.confidence == 0.0
        assert result.error is not None
        assert "API key missing" in result.error

    def test_analyze_with_llm_returning_invalid_json_returns_failed(self):
        """LLM returning non-JSON text yields a failed result."""
        agent = _create_agent(
            llm_response="This is not JSON at all, just plain text.",
        )

        result = agent.analyze("300750")

        assert result.success is False
        assert "JSON" in result.error or "Expecting" in result.error

    def test_analyze_with_llm_json_in_code_fences(self):
        """LLM response wrapped in markdown code fences is parsed correctly."""
        agent = _create_agent(
            news_data=_mock_news(2),
            llm_response=_make_llm_response_in_fence(
                signal="BUY", confidence=0.7, sentiment_score=0.5,
            ),
        )

        result = agent.analyze("300750")

        assert result.success is True
        assert result.signal == "BUY"
        assert result.confidence == 0.7

    def test_analyze_invalid_signal_defaulted_to_hold(self):
        """LLM returning an unrecognized signal defaults to HOLD."""
        response = json.dumps({
            "signal": "STRONG_BUY",  # Not in BUY/SELL/HOLD
            "confidence": 0.9,
            "sentiment_score": 0.8,
            "reasoning": "Very bullish",
            "key_factors": ["a"],
        })
        agent = _create_agent(llm_response=response)

        result = agent.analyze("300750")

        assert result.signal == "HOLD"  # Defaulted

    def test_analyze_confidence_clamped_above_1(self):
        """Confidence > 1.0 is clamped to 1.0."""
        response = json.dumps({
            "signal": "BUY",
            "confidence": 1.5,  # Out of range
            "sentiment_score": 0.5,
            "reasoning": "Test",
            "key_factors": [],
        })
        agent = _create_agent(llm_response=response)

        result = agent.analyze("300750")

        assert result.confidence == 1.0

    def test_analyze_confidence_clamped_below_0(self):
        """Confidence < 0.0 is clamped to 0.0."""
        response = json.dumps({
            "signal": "SELL",
            "confidence": -0.3,
            "sentiment_score": -0.5,
            "reasoning": "Test",
            "key_factors": [],
        })
        agent = _create_agent(llm_response=response)

        result = agent.analyze("300750")

        assert result.confidence == 0.0

    def test_analyze_scores_include_sentiment(self):
        """Result scores dict includes the sentiment score."""
        agent = _create_agent(
            llm_response=_make_llm_response(sentiment_score=0.42),
        )

        result = agent.analyze("300750")

        assert "sentiment" in result.scores
        assert result.scores["sentiment"] == 0.42

    def test_analyze_metrics_include_key_factors(self):
        """Result metrics include the key_factors list."""
        agent = _create_agent(
            llm_response=_make_llm_response(key_factors=["EV policy", "Q4 earnings"]),
        )

        result = agent.analyze("300750")

        assert "key_factors" in result.metrics
        assert "EV policy" in result.metrics["key_factors"]


# ── _fetch_news tests ───────────────────────────────────────────────────────


class TestSentimentAgentFetchNews:
    """SentimentAgent._fetch_news — news retrieval from data service."""

    def test_fetch_news_returns_list(self):
        """_fetch_news returns a list of dicts from the data service."""
        ds = MagicMock()
        ds.get_news.return_value = [
            {"title": "News 1", "content": "Content 1"},
        ]
        ds.akshare = ds

        agent = SentimentAgent(data_service=ds, llm_client=MagicMock(spec=LLMClient))
        items = agent._fetch_news("300750")

        assert len(items) == 1
        assert items[0]["title"] == "News 1"

    def test_fetch_news_handles_none_data_service(self):
        """_fetch_news returns empty list when data_service is None."""
        agent = SentimentAgent(data_service=None, llm_client=MagicMock(spec=LLMClient))
        items = agent._fetch_news("300750")

        assert items == []

    def test_fetch_news_handles_no_get_news_method(self):
        """_fetch_news returns empty list when data source has no get_news."""
        ds = MagicMock(spec=[])  # No methods at all
        # Explicitly remove get_news
        del ds.get_news

        agent = SentimentAgent(data_service=ds, llm_client=MagicMock(spec=LLMClient))
        items = agent._fetch_news("300750")

        assert items == []

    def test_fetch_news_handles_none_return(self):
        """_fetch_news returns empty list when get_news returns None."""
        ds = MagicMock()
        ds.get_news.return_value = None
        ds.akshare = ds

        agent = SentimentAgent(data_service=ds, llm_client=MagicMock(spec=LLMClient))
        items = agent._fetch_news("300750")

        assert items == []

    def test_fetch_news_handles_exception(self):
        """_fetch_news returns empty list when get_news raises."""
        ds = MagicMock()
        ds.get_news.side_effect = ConnectionError("Network error")
        ds.akshare = ds

        agent = SentimentAgent(data_service=ds, llm_client=MagicMock(spec=LLMClient))
        items = agent._fetch_news("300750")

        assert items == []

    def test_fetch_news_converts_dataframe(self):
        """_fetch_news converts a DataFrame response to a list of dicts."""
        import pandas as pd

        df = pd.DataFrame([
            {"title": "News A", "content": "Content A"},
            {"title": "News B", "content": "Content B"},
        ])

        ds = MagicMock()
        ds.get_news.return_value = df
        ds.akshare = ds

        agent = SentimentAgent(data_service=ds, llm_client=MagicMock(spec=LLMClient))
        items = agent._fetch_news("300750")

        assert isinstance(items, list)
        assert len(items) == 2
        assert items[0]["title"] == "News A"


# ── _format_news tests ──────────────────────────────────────────────────────


class TestSentimentAgentFormatNews:
    """SentimentAgent._format_news — formatting news for LLM prompt."""

    def test_format_news_empty_list(self):
        """Empty news list returns the fallback text."""
        result = SentimentAgent._format_news([])
        assert result == "(无新闻数据)"

    def test_format_news_single_item(self):
        """Single news item is formatted with index, date, title, content."""
        items = [
            {"title": "Breaking News", "content": "Details here", "date": "2026-04-10"},
        ]
        result = SentimentAgent._format_news(items)

        assert "1." in result
        assert "[2026-04-10]" in result
        assert "Breaking News" in result
        assert "Details here" in result

    def test_format_news_truncates_long_content(self):
        """Content longer than 200 characters is truncated."""
        items = [
            {"title": "T", "content": "x" * 300},
        ]
        result = SentimentAgent._format_news(items)

        # Should contain truncated content (200 chars) but not the full 300
        assert "x" * 200 in result
        assert "x" * 300 not in result

    def test_format_news_limits_to_20_items(self):
        """Only the first 20 news items are included."""
        items = [{"title": f"News {i}"} for i in range(30)]
        result = SentimentAgent._format_news(items)

        # Should have items 1-20 but not 21-30
        assert "News 0" in result
        assert "News 19" in result
        assert "News 20" not in result

    def test_format_news_handles_chinese_keys(self):
        """Chinese keys (标题, 内容, 时间) are recognized."""
        items = [
            {"标题": "利好消息", "内容": "业绩超预期", "时间": "2026-04-10"},
        ]
        result = SentimentAgent._format_news(items)

        assert "利好消息" in result
        assert "业绩超预期" in result
        assert "2026-04-10" in result

    def test_format_news_handles_missing_fields(self):
        """Items with missing fields don't crash formatting."""
        items = [{"title": "Just a title"}]  # No content or date
        result = SentimentAgent._format_news(items)

        assert "Just a title" in result

    def test_format_news_uses_description_fallback(self):
        """Content field falls back to 'description' key."""
        items = [{"title": "T", "description": "Fallback content"}]
        result = SentimentAgent._format_news(items)

        assert "Fallback content" in result


# ── _parse_response tests ───────────────────────────────────────────────────


class TestSentimentAgentParseResponse:
    """SentimentAgent._parse_response — JSON parsing from LLM text."""

    def test_parse_clean_json(self):
        """Clean JSON is parsed directly."""
        text = '{"signal": "BUY", "confidence": 0.8}'
        result = SentimentAgent._parse_response(text)

        assert result["signal"] == "BUY"
        assert result["confidence"] == 0.8

    def test_parse_json_with_whitespace(self):
        """JSON with leading/trailing whitespace is parsed."""
        text = '  \n  {"signal": "HOLD"}  \n  '
        result = SentimentAgent._parse_response(text)

        assert result["signal"] == "HOLD"

    def test_parse_json_in_code_fences(self):
        """JSON wrapped in markdown code fences is extracted."""
        text = '```json\n{"signal": "SELL", "confidence": 0.6}\n```'
        result = SentimentAgent._parse_response(text)

        assert result["signal"] == "SELL"
        assert result["confidence"] == 0.6

    def test_parse_json_in_plain_code_fences(self):
        """JSON in code fences without language tag is extracted."""
        text = '```\n{"signal": "BUY"}\n```'
        result = SentimentAgent._parse_response(text)

        assert result["signal"] == "BUY"

    def test_parse_invalid_json_raises(self):
        """Invalid JSON raises JSONDecodeError."""
        with pytest.raises(Exception):
            SentimentAgent._parse_response("not json at all")
