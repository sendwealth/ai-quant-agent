"""P3.3 告警模块测试。"""

from __future__ import annotations

from quant_agent.observability.alerting import (
    Alert,
    AlertManager,
    component_down_rule,
    data_degradation_rule,
    data_health_score_rule,
    smoke_source_failure_rule,
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


class TestSmokeAlertRules:
    """推荐 #3：smoke 结果接入告警（按真实失败源 / 健康分）。"""

    def _smoke_report(self, failed=(), score=100, healthy=1, skip_count=0):
        return {
            "smoke": {"results": []},  # 有 smoke 即表示跑过真实冒烟
            "data_health_score": {
                "overall_score": score,
                "healthy_count": healthy,
                "failed_count": len(failed),
                "skipped_count": skip_count,
                "failed_sources": list(failed),
            },
        }

    def test_source_failure_critical(self):
        report = self._smoke_report(failed=("akshare", "efinance"))
        alert = smoke_source_failure_rule(report)
        assert alert is not None
        assert alert.rule == "smoke_source_failure"
        assert alert.severity == "critical"
        assert "akshare" in alert.message and "efinance" in alert.message

    def test_source_failure_none_without_smoke(self):
        # 未跑过真实冒烟（smoke=None）→ 不触发，避免误报
        assert smoke_source_failure_rule({"smoke": None, "data_health_score": {}}) is None

    def test_source_failure_none_when_healthy(self):
        assert smoke_source_failure_rule(self._smoke_report(failed=())) is None

    def test_health_score_warning_below_threshold(self):
        report = self._smoke_report(score=40, healthy=1)
        alert = data_health_score_rule(report)
        assert alert is not None
        assert alert.rule == "data_health_score"
        assert alert.severity == "warning"

    def test_health_score_none_when_high(self):
        assert data_health_score_rule(self._smoke_report(score=90)) is None

    def test_health_score_none_without_smoke(self):
        assert data_health_score_rule({"smoke": None, "data_health_score": {}}) is None

    def test_health_score_none_when_no_real_sources(self):
        # 仅跳过源（无真实源）→ 不触发，避免离线/未配置误报
        assert data_health_score_rule(self._smoke_report(score=0, healthy=0, skip_count=2)) is None

    def test_alert_manager_registers_new_rules(self):
        am = AlertManager()
        rule_names = {getattr(r, "__name__", getattr(r, "func", r).__name__) for r in am.rules}
        assert "smoke_source_failure_rule" in rule_names
        assert "data_health_score_rule" in rule_names

    def test_alert_manager_fires_smoke_failure(self):
        am = AlertManager()
        report = self._smoke_report(failed=("akshare",))
        alerts = am.check(report)
        assert any(a.rule == "smoke_source_failure" for a in alerts)


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
