"""情感分析 Agent -- LLM 分析市场新闻情感"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from .base import BaseAgent, AgentResult
from ..llm.client import LLMClient, LLMError
from ..llm.prompts import SENTIMENT_SYSTEM, SENTIMENT_USER

logger = logging.getLogger(__name__)


class SentimentResponse(BaseModel):
    """Structured output schema for LLM sentiment analysis."""
    signal: str = Field(description="BUY, SELL, or HOLD")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score 0-1")
    sentiment_score: float = Field(ge=-1.0, le=1.0, description="Sentiment score -1 to 1")
    reasoning: str = Field(default="", description="Analysis reasoning")
    key_factors: list[str] = Field(default_factory=list, description="Key sentiment factors")


class SentimentAgent(BaseAgent):
    """情感分析 Agent -- LLM 分析市场新闻情感"""

    def __init__(self, data_service, llm_client: LLMClient, **kwargs):
        super().__init__(name="sentiment", **kwargs)
        self.data = data_service
        self.llm = llm_client

    def analyze(self, stock_code: str, **kwargs) -> AgentResult:
        """分析市场新闻情感

        Args:
            stock_code: A 股代码 (6 位数字)

        Returns:
            AgentResult with sentiment signal, confidence, reasoning.
            On any failure returns AgentResult(success=False).
        """
        self._logger.info("情感分析: %s", stock_code)
        self._log_action("analysis_started", stock_code=stock_code)

        try:
            # 1. Fetch news from data service
            news_items = self._fetch_news(stock_code)

            # 2. Format news into text for LLM prompt
            news_text = self._format_news(news_items)

            # 3. Call LLM with structured output, fallback to JSON parse
            user_prompt = SENTIMENT_USER.safe_substitute(
                stock_code=stock_code,
                news_text=news_text,
            )

            parsed = self._get_llm_response(SENTIMENT_SYSTEM, user_prompt)

            signal = parsed.get("signal", "HOLD")
            confidence = float(parsed.get("confidence", 0.3))
            sentiment_score = float(parsed.get("sentiment_score", 0.0))
            reasoning = parsed.get("reasoning", "")
            key_factors = parsed.get("key_factors", [])

            # Validate signal
            if signal not in ("BUY", "SELL", "HOLD"):
                signal = "HOLD"

            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))

            result = AgentResult(
                agent_name=self.name,
                stock_code=stock_code,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics={
                    "sentiment_score": round(sentiment_score, 4),
                    "key_factors": key_factors,
                    "news_count": len(news_items),
                },
                scores={
                    "sentiment": round(sentiment_score, 4),
                },
            )
            self._log_action(
                "analysis_completed",
                stock_code=stock_code,
                signal=signal,
                confidence=confidence,
            )
            return result

        except Exception as e:
            self._logger.error("情感分析失败: %s", e)
            self._log_action("analysis_failed", stock_code=stock_code, error=str(e))
            return AgentResult(
                agent_name=self.name,
                stock_code=stock_code,
                signal="HOLD",
                confidence=0.0,
                reasoning=f"情感分析失败: {e}",
                success=False,
                error=str(e),
            )

    # ── Private helpers ──────────────────────────────────────────────────────

    def _get_llm_response(self, system: str, user: str) -> dict[str, Any]:
        """Get structured response from LLM with fallback.

        Primary: use structured_output with Pydantic schema.
        Fallback: invoke raw and parse JSON manually.
        """
        if self.llm is None:
            raise LLMError("LLM client not configured")

        # Try structured output first
        try:
            resp = self.llm.structured_output(system, user, SentimentResponse)
            return {
                "signal": resp.signal,
                "confidence": resp.confidence,
                "sentiment_score": resp.sentiment_score,
                "reasoning": resp.reasoning,
                "key_factors": resp.key_factors,
            }
        except LLMError:
            pass

        # Fallback: raw invoke + JSON parse
        raw_response = self.llm.invoke(system, user)
        return self._parse_response(raw_response)

    def _fetch_news(self, stock_code: str) -> list[dict[str, Any]]:
        """Fetch news from data service (AkShare stock_news_em).

        Returns a list of dicts with 'title' and 'content' keys,
        or an empty list if news is unavailable.
        """
        if self.data is None:
            return []

        # Try data service akshare source directly
        akshare = getattr(self.data, "akshare", None)
        if akshare is None:
            # Fallback: try data_service itself (duck-typing)
            akshare = self.data

        fetch_fn = getattr(akshare, "get_news", None)
        if fetch_fn is None:
            return []

        try:
            result = fetch_fn(stock_code)
            if result is None:
                return []
            # Accept list of dicts or DataFrame
            if isinstance(result, list):
                return result
            # DataFrame case: convert to list of dicts
            if hasattr(result, "to_dict"):
                return result.to_dict("records")
            return []
        except Exception as e:
            self._logger.warning("新闻获取失败: %s", e)
            return []

    @staticmethod
    def _format_news(news_items: list[dict[str, Any]]) -> str:
        """Format news items into readable text for LLM prompt."""
        if not news_items:
            return "(无新闻数据)"

        lines = []
        for i, item in enumerate(news_items[:20], start=1):
            title = item.get("title", item.get("标题", ""))
            content = item.get("content", item.get("内容", item.get("description", "")))
            date = item.get("date", item.get("时间", item.get("publish_time", "")))

            parts = [f"{i}."]
            if date:
                parts.append(f"[{date}]")
            if title:
                parts.append(title)
            if content:
                # Truncate long content to avoid blowing up the prompt
                snippet = content[:200]
                parts.append(f"  {snippet}")
            lines.append(" ".join(parts))

        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str) -> dict[str, Any]:
        """Parse JSON from LLM response text (fallback method).

        Handles cases where the JSON may be wrapped in markdown
        code fences or have extra text around it.
        """
        text = raw.strip()

        # Strip markdown code fences if present
        if text.startswith("```"):
            # Remove opening fence
            first_newline = text.index("\n") if "\n" in text else len(text)
            text = text[first_newline + 1:]
            # Remove closing fence
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        return json.loads(text)
