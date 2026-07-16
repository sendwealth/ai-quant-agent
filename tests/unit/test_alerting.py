"""P3.3 告警模块测试。"""

from __future__ import annotations

from quant_agent.observability.alerting import (
    Alert,
    AlertManager,
    component_down_rule,
    data_degradation_rule,
)


class TestAlertRules:
    def test_data_degradation_triggers(self):
        report = {"degraded": True, "offline_mode": True, "smoke": None, "components": []}
        alert = data_degradation_rule(report)
        assert alert is not None
        assert alert.rule == "data_degradation"
        assert alert.severity == "warning"

    def test_no_degradation_no_alert(self):
        report = {"degraded": False, "components": []}
        assert data_degradation_rule(report) is None

    def test_component_down_triggers(self):
        report = {
            "degraded": False,
            "components": [
                {"name": "datasource:akshare", "ok": False, "degraded": True, "detail": "boom"}
            ],
        }
        alert = component_down_rule(report)
        assert alert is not None
        assert alert.rule == "component_down"

    def test_component_ok_no_alert(self):
        report = {
            "degraded": False,
            "components": [{"name": "datasource:akshare", "ok": True, "degraded": False, "detail": ""}],
        }
        assert component_down_rule(report) is None


class TestAlertManager:
    def test_check_returns_triggered(self):
        report = {"degraded": True, "offline_mode": True, "smoke": None, "components": []}
        am = AlertManager()
        alerts = am.check(report)
        assert len(alerts) >= 1
        assert any(a.rule == "data_degradation" for a in alerts)

    def test_notifier_called(self):
        sent: list[Alert] = []
        am = AlertManager(notifier=lambda a: sent.append(a))
        report = {"degraded": True, "offline_mode": True, "smoke": None, "components": []}
        am.check(report)
        assert len(sent) >= 1
        assert isinstance(sent[0], Alert)

    def test_notifier_failure_does_not_propagate(self):
        def bad(a: Alert) -> None:
            raise RuntimeError("notify failed")

        am = AlertManager(notifier=bad)
        report = {"degraded": True, "offline_mode": True, "smoke": None, "components": []}
        # 不应抛出异常
        alerts = am.check(report)
        assert len(alerts) >= 1
