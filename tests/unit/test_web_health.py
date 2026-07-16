"""P3.1 / P3.2 / P3.4 Web 健康端点 / 指标 / 输入校验与鉴权 测试。

均离线、不触网：health_core 接受注入的冒烟结果；E2E 另见 test_web_e2e.py。
"""

from __future__ import annotations

import os

from quant_agent.config import Settings
from quant_agent.observability.health import build_health_report
from quant_agent.web import server as web_server
from quant_agent.web.server import (
    _is_authorized,
    _validate_stock_code_or_raise,
    alerts_core,
    health_core,
    metrics_core,
)


class FakeHandler:
    """最小 handler 替身，仅用于鉴权测试。"""

    def __init__(self, auth_header=""):
        self.headers = {"Authorization": auth_header}


class TestHealthReport:
    def test_offline_is_degraded(self):
        settings = Settings(offline_mode=True, app_name="test")
        rep = build_health_report(settings)
        assert rep["degraded"] is True
        assert rep["offline_mode"] is True
        assert rep["status"] == "degraded"

    def test_online_not_degraded_without_smoke(self):
        settings = Settings(offline_mode=False, app_name="test")
        rep = build_health_report(settings)
        assert rep["degraded"] is False

    def test_smoke_degraded_overrides(self):
        settings = Settings(offline_mode=False, app_name="test")
        smoke = {
            "degraded": True,
            "ok": 0,
            "total": 2,
            "failed": 2,
            "skipped": 0,
            "results": [
                {"source": "akshare", "ok": False, "error": "x", "skipped": False, "rows": 0},
            ],
        }
        rep = build_health_report(settings, smoke_results=smoke)
        assert rep["degraded"] is True
        assert rep["smoke"]["failed"] == 2


class TestHealthCore:
    def test_health_core_with_injected_smoke(self):
        smoke = {"degraded": False, "ok": 1, "total": 1, "failed": 0, "skipped": 0, "results": []}
        rep = health_core(smoke_results=smoke)
        assert rep["degraded"] is False
        assert "components" in rep

    def test_health_core_degraded_sets_status(self):
        smoke = {"degraded": True, "ok": 0, "total": 1, "failed": 1, "skipped": 0, "results": []}
        rep = health_core(smoke_results=smoke)
        assert rep["status"] == "degraded"


class TestMetricsCore:
    def test_metrics_core_prometheus_format(self):
        web_server.METRICS.counter("test_requests_total", tags={"path": "/api/health"})
        text = metrics_core()
        assert "# TYPE" in text
        assert "test_requests_total" in text
        # Prometheus 行格式：name{tag="v"} value
        assert 'path="/api/health"' in text


class TestAlertsCore:
    def test_alerts_core_returns_dict(self):
        # 用降级冒烟触发告警
        web_server.ALERT_MANAGER.check(
            build_health_report(Settings(offline_mode=True, app_name="t"), smoke_results=None)
        )
        out = alerts_core()
        assert "alerts" in out
        assert "count" in out
        assert isinstance(out["alerts"], list)


class TestInputValidation:
    def test_valid_code_passes(self):
        _validate_stock_code_or_raise("300750")  # 不应抛异常

    def test_invalid_code_raises(self):
        import pytest

        with pytest.raises(ValueError):
            _validate_stock_code_or_raise("12ab")


class TestAuth:
    def teardown_method(self):
        os.environ.pop("QUANT_WEB_AUTH_TOKEN", None)

    def test_no_token_no_auth_required(self):
        os.environ.pop("QUANT_WEB_AUTH_TOKEN", None)
        assert _is_authorized(FakeHandler()) is True

    def test_valid_token_accepted(self):
        os.environ["QUANT_WEB_AUTH_TOKEN"] = "secret"
        assert _is_authorized(FakeHandler("Bearer secret")) is True

    def test_wrong_token_rejected(self):
        os.environ["QUANT_WEB_AUTH_TOKEN"] = "secret"
        assert _is_authorized(FakeHandler("Bearer wrong")) is False

    def test_missing_token_rejected(self):
        os.environ["QUANT_WEB_AUTH_TOKEN"] = "secret"
        assert _is_authorized(FakeHandler()) is False
