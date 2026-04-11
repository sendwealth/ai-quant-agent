"""Agent 基类 — 统一协议"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

Signal = Literal["BUY", "SELL", "HOLD"]

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Agent 执行结果"""
    agent_name: str
    stock_code: str
    signal: Signal = "HOLD"          # BUY / HOLD / SELL
    confidence: float = 0.0       # 0-1
    reasoning: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent_name,
            "stock_code": self.stock_code,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "metrics": self.metrics,
            "scores": self.scores,
            "timestamp": self.timestamp,
            "success": self.success,
        }


class BaseAgent(ABC):
    """Agent 基类 — 所有 Agent 遵循统一协议"""

    def __init__(
        self,
        name: str,
        data_service: Optional[Any] = None,
        **kwargs,  # accept and ignore extra kwargs for forward compat
    ):
        self.name = name
        self.data_service = data_service
        self._logger = logging.getLogger(f"agent.{name}")

    @abstractmethod
    def analyze(self, stock_code: str, **kwargs) -> AgentResult:
        """执行分析，返回标准化结果"""
        ...

    def _log_action(
        self,
        action: str,
        stock_code: str = "",
        signal: str = "",
        confidence: float | None = None,
        **extra_fields: Any,
    ) -> None:
        """Structured logging for agent actions.

        Replaces fire-and-forget emit_event calls with structured log records
        that carry agent context in the ``extra`` dict for downstream log
        processors (JSON formatters, log aggregators, etc.).
        """
        structured: dict[str, Any] = {
            "agent_name": self.name,
            "action": action,
        }
        if stock_code:
            structured["stock_code"] = stock_code
        if signal:
            structured["signal"] = signal
        if confidence is not None:
            structured["confidence"] = round(confidence, 4)
        structured.update(extra_fields)

        self._logger.info(
            f"[{self.name}] {action}" + (f" {stock_code}" if stock_code else ""),
            extra=structured,
        )

    def __repr__(self) -> str:
        return f"Agent({self.name})"
