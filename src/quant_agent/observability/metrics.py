"""可观测性 — 日志 + 指标 + 健康检查"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class Metric:
    """指标"""
    name: str
    value: float
    tags: dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class HealthStatus:
    """健康状态"""
    healthy: bool = True
    checks: dict[str, bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timestamp: str = ""


class MetricsCollector:
    """指标收集器"""

    def __init__(self, max_metrics: int = 10000):
        self._metrics: deque[Metric] = deque(maxlen=max_metrics)
        self._counters: dict[str, int] = {}
        self._gauges: dict[str, float] = {}

    def counter(self, name: str, value: int = 1, tags: dict = None):
        """计数器"""
        key = self._key(name, tags)
        self._counters[key] = self._counters.get(key, 0) + value
        self._record(name, float(self._counters[key]), tags)

    def gauge(self, name: str, value: float, tags: dict = None):
        """仪表盘"""
        key = self._key(name, tags)
        self._gauges[key] = value
        self._record(name, value, tags)

    def timer(self, name: str, tags: dict = None):
        """计时器上下文"""
        return _Timer(name, self, tags)

    def _record(self, name: str, value: float, tags: dict = None):
        self._metrics.append(Metric(name=name, value=value, tags=tags or {}))

    def get(self, name: str, tags: dict = None) -> Optional[float]:
        key = self._key(name, tags)
        return self._gauges.get(key) or (self._counters.get(key) if key in self._counters else None)

    def query(self, name: str) -> list[Metric]:
        return [m for m in self._metrics if m.name == name]

    @staticmethod
    def _key(name: str, tags: dict = None) -> str:
        if not tags:
            return name
        tag_str = ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
        return f"{name}{{{tag_str}}}"


class _Timer:
    def __init__(self, name: str, collector: MetricsCollector, tags: dict = None):
        self.name = name
        self.collector = collector
        self.tags = tags
        self.start = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.collector.gauge(f"{self.name}.duration_ms", elapsed * 1000, self.tags)


class HealthChecker:
    """健康检查器"""

    def __init__(self):
        self._checks: dict[str, Any] = {}

    def register(self, name: str, check_fn):
        """注册检查函数，check_fn() -> bool"""
        self._checks[name] = check_fn

    def check(self) -> HealthStatus:
        status = HealthStatus(
            timestamp=datetime.now().isoformat()
        )
        for name, fn in self._checks.items():
            try:
                ok = fn()
                status.checks[name] = ok
                if not ok:
                    status.healthy = False
                    status.errors.append(f"{name}: FAILED")
            except Exception as e:
                status.checks[name] = False
                status.healthy = False
                status.errors.append(f"{name}: ERROR ({e})")
        return status

    def check_all(self) -> dict:
        status = self.check()
        return {
            "healthy": status.healthy,
            "checks": status.checks,
            "errors": status.errors,
            "timestamp": status.timestamp,
        }
