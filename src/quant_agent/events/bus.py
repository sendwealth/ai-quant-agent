"""事件系统 — EventBus + Event 定义"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    # 数据事件
    DATA_QUOTE_UPDATED = "data.quote.updated"
    DATA_FINANCIAL_UPDATED = "data.financial.updated"

    # 分析事件
    ANALYSIS_STARTED = "analysis.started"
    ANALYSIS_COMPLETED = "analysis.completed"
    ANALYSIS_FAILED = "analysis.failed"

    # 决策事件
    SIGNAL_GENERATED = "signal.generated"
    RISK_CHECK_PASSED = "risk.check.passed"
    RISK_CHECK_FAILED = "risk.check.failed"

    # 系统事件
    SYSTEM_HEARTBEAT = "system.heartbeat"
    SYSTEM_ERROR = "system.error"


@dataclass
class Event:
    """事件"""
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "source": self.source,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"Event({self.type.value}, src={self.source})"


# 回调类型
Handler = Callable[[Event], Any]


class EventBus:
    """事件总线 — 进程内异步通信

    当前为内存实现，后续可迁移到 Redis Streams。
    """

    def __init__(self):
        self._handlers: dict[str, list[Handler]] = {}
        self._history: list[Event] = []
        self._max_history: int = 1000

    def subscribe(self, event_type: EventType | str, handler: Handler):
        """订阅事件"""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        if key not in self._handlers:
            self._handlers[key] = []
        self._handlers[key].append(handler)
        logger.debug(f"📡 订阅: {key} -> {handler.__name__}")

    def unsubscribe(self, event_type: EventType | str, handler: Handler):
        """取消订阅"""
        key = event_type.value if isinstance(event_type, EventType) else event_type
        if key in self._handlers:
            self._handlers[key] = [h for h in self._handlers[key] if h != handler]

    def publish(self, event: Event) -> int:
        """发布事件，返回处理数量"""
        self._record(event)
        key = event.type.value
        handlers = self._handlers.get(key, [])

        # 通配符匹配
        if key not in self._handlers:
            for pattern, hs in self._handlers.items():
                if pattern.endswith("*") and key.startswith(pattern[:-1]):
                    handlers.extend(hs)

        count = 0
        for handler in handlers:
            try:
                handler(event)
                count += 1
            except Exception as e:
                logger.error(f"事件处理失败 [{key}]: {e}")

        if count > 0:
            logger.debug(f"📤 {event} → {count} handlers")
        return count

    def publish_simple(self, event_type: EventType, payload: dict = None, source: str = "") -> int:
        """便捷发布"""
        return self.publish(Event(
            type=event_type,
            payload=payload or {},
            source=source,
        ))

    def _record(self, event: Event):
        """记录事件历史"""
        self._history.append(event)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    @property
    def history(self) -> list[Event]:
        return list(self._history)

    def clear(self):
        self._handlers.clear()
        self._history.clear()
