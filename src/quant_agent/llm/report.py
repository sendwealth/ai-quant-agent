"""LLMReportGenerator — 综合分析报告生成"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

from .client import LLMClient
from .prompts import REPORT_SYSTEM, REPORT_USER

if TYPE_CHECKING:
    from ..orchestrator import AnalysisReport

logger = logging.getLogger(__name__)


class LLMReportGenerator:
    """LLM 综合分析报告生成器"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate(self, report: AnalysisReport) -> str:
        """综合所有 Agent 结果，生成自然语言投资分析报告

        Args:
            report: 完整分析报告 (含各 Agent 结果)

        Returns:
            Markdown 格式的投资分析报告
        """
        # 构建情感分析部分 (可选)
        sentiment_section = ""
        if report.sentiment_result and report.sentiment_result.success:
            sr = report.sentiment_result
            sentiment_section = (
                f"=== 情感分析 ===\n"
                f"信号: {sr.signal} (信心度: {sr.confidence:.0%})\n"
                f"理由: {sr.reasoning}\n"
            )

        # 格式化 user prompt (safe_substitute prevents KeyError on curly braces
        # in interpolated values like reasoning text)
        user_prompt = REPORT_USER.safe_substitute(
            stock_code=report.stock_code,
            timestamp=report.timestamp,
            fundamental_signal=_sig(report.fundamental_result),
            fundamental_confidence=_conf(report.fundamental_result),
            fundamental_reasoning=_reason(report.fundamental_result),
            fundamental_metrics=_metrics_str(report.fundamental_result),
            technical_signal=_sig(report.technical_result),
            technical_confidence=_conf(report.technical_result),
            technical_reasoning=_reason(report.technical_result),
            technical_metrics=_metrics_str(report.technical_result),
            sentiment_section=sentiment_section,
            risk_signal=_sig(report.risk_result),
            risk_confidence=_conf(report.risk_result),
            risk_reasoning=_reason(report.risk_result),
            risk_position=report.risk_result.metrics.get("position", 0) * 100
            if report.risk_result
            else 0,
            total_equity=f"{report.summary.get('total_equity', 0):,.2f}",
            total_return=f"{report.summary.get('total_return', 0):.2%}",
        )

        result = self.llm.invoke(REPORT_SYSTEM, user_prompt)
        logger.info("LLM 分析报告生成完成: %s (%d chars)", report.stock_code, len(result))
        return result


# ── helpers ────────────────────────────────────────────────────────────────


def _sig(r: Optional[AgentResult]) -> str:
    return r.signal if r else "N/A"


def _conf(r: Optional[AgentResult]) -> str:
    return f"{r.confidence:.0%}" if r else "N/A"


def _reason(r: Optional[AgentResult]) -> str:
    return r.reasoning if r else "N/A"


def _metrics_str(r: Optional[AgentResult]) -> str:
    if not r or not r.metrics:
        return "N/A"
    return json.dumps(r.metrics, ensure_ascii=False, default=str)[:500]
