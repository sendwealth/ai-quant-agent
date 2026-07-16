"""告警（P3.3）— 基于健康/指标的规则化告警。

``AlertManager`` 接收一份健康报告（见 :mod:`quant_agent.observability.health`），
按一组规则判定是否触发告警，并通过可选回调（邮件 / 日志 / webhook）分发。
规则可自由扩展；内置两条：数据降级告警、组件异常告警。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# 一条告警规则：输入健康报告，返回 Alert 或 None
AlertRule = Callable[["dict[str, Any]"], "Alert | None"]


@dataclass
class Alert:
    """一条告警。"""

    rule: str
    severity: str  # "info" | "warning" | "critical"
    message: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
        }


def data_degradation_rule(report: dict[str, Any]) -> Alert | None:
    """所有实时数据源不可用 → 数据降级告警。"""
    if report.get("degraded"):
        return Alert(
            rule="data_degradation",
            severity="warning",
            message="所有实时数据源不可用，分析回退到样例/缓存，结论仅供参考",
            context={"smoke": report.get("smoke"), "offline_mode": report.get("offline_mode")},
        )
    return None


def component_down_rule(report: dict[str, Any]) -> Alert | None:
    """任一组件异常（数据源失败 / LLM 未配置但要求） → 组件告警。"""
    for c in report.get("components", []):
        if c.get("degraded") or (not c.get("ok") and c.get("name", "").startswith("datasource")):
            return Alert(
                rule="component_down",
                severity="warning",
                message=f"组件异常: {c.get('name')} - {c.get('detail', '')}",
                context={"component": c},
            )
    return None


class AlertManager:
    """规则化告警管理器。

    Args:
        rules: 告警规则列表；默认内置数据降级 + 组件异常。
        notifier: 可选分发回调 ``callable(Alert)``（如邮件通知器）。
    """

    def __init__(
        self,
        rules: list[AlertRule] | None = None,
        notifier: Callable[[Alert], None] | None = None,
    ) -> None:
        self.rules = rules or [data_degradation_rule, component_down_rule]
        self.notifier = notifier

    def check(self, report: dict[str, Any]) -> list[Alert]:
        """对报告逐条应用规则，返回触发告警；若有 notifier 则同步分发。"""
        alerts: list[Alert] = []
        for rule in self.rules:
            try:
                triggered = rule(report)
            except Exception:
                triggered = None
            if triggered is not None:
                alerts.append(triggered)
        for a in alerts:
            if self.notifier is not None:
                try:
                    self.notifier(a)
                except Exception:
                    # 分发失败不影响告警判定
                    pass
        return alerts
