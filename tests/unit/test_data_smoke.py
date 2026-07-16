"""P1.5 数据源冒烟测试 单元测试

通过 mock DataSource 验证单源检查、聚合报告，以及 DataService.smoke_test
在不触发真实网络的情况下正确聚合。
"""

from __future__ import annotations

import pandas as pd

from quant_agent.data.smoke import (
    SourceSmokeResult,
    smoke_report,
    smoke_test_source,
)
from quant_agent.data.sources.base import DataSource


class _FakeSource(DataSource):
    """可控的假数据源，用于测试冒烟逻辑。"""

    def __init__(self, name="fake", available=True, df=None, raise_error=None):
        self._name = name
        self._available = available
        self._df = df
        self._raise = raise_error

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self._available

    def get_price_data(self, stock_code, days=250, adjust="qfq"):
        if self._raise is not None:
            raise self._raise
        return self._df

    def get_realtime_price(self, stock_code):
        return None


def _good_df(n=5):
    return pd.DataFrame(
        {
            "date": [f"2024-01-{i:02d}" for i in range(1, n + 1)],
            "open": [10.0] * n,
            "high": [10.5] * n,
            "low": [9.5] * n,
            "close": [10.0] * n,
            "volume": [1000] * n,
        }
    )


def test_smoke_source_ok():
    src = _FakeSource(name="tushare", available=True, df=_good_df(5))
    res = smoke_test_source(src, stock_code="600519", days=5)
    assert res.ok is True
    assert res.source == "tushare"
    assert res.rows == 5
    assert res.latency_ms >= 0
    assert res.error is None
    assert res.skipped is False


def test_smoke_source_skipped_when_unavailable():
    src = _FakeSource(name="tushare", available=False)
    res = smoke_test_source(src)
    assert res.skipped is True
    assert res.ok is False
    assert "not available" in (res.skip_reason or "")


def test_smoke_source_fail_on_exception():
    src = _FakeSource(name="akshare", available=True, raise_error=TimeoutError("timeout"))
    res = smoke_test_source(src)
    assert res.ok is False
    assert res.skipped is False
    assert "timeout" in (res.error or "")


def test_smoke_source_fail_on_empty():
    src = _FakeSource(name="efinance", available=True, df=pd.DataFrame())
    res = smoke_test_source(src)
    assert res.ok is False
    assert res.rows == 0


def test_smoke_report_aggregation():
    results = [
        SourceSmokeResult(source="tushare", ok=True, rows=5, latency_ms=12.0),
        SourceSmokeResult(
            source="akshare",
            ok=False,
            error="boom",
            skipped=False,
        ),
        SourceSmokeResult(
            source="baostock", ok=False, skipped=True, skip_reason="no creds"
        ),
    ]
    rep = smoke_report(results)
    assert rep["total"] == 3
    assert rep["ok"] == 1
    assert rep["failed"] == 1
    assert rep["skipped"] == 1
    assert rep["all_healthy"] is False  # 有失败
    assert rep["degraded"] is False  # 至少 1 个健康


def test_smoke_report_degraded_when_none_healthy():
    results = [
        SourceSmokeResult(source="akshare", ok=False, error="boom", skipped=False),
        SourceSmokeResult(
            source="baostock", ok=False, skipped=True, skip_reason="no creds"
        ),
    ]
    rep = smoke_report(results)
    assert rep["degraded"] is True
    assert rep["all_healthy"] is False
    assert rep["degradation_note"] is not None


def test_smoke_report_all_healthy():
    results = [
        SourceSmokeResult(source="tushare", ok=True, rows=5),
        SourceSmokeResult(source="efinance", ok=True, rows=5),
    ]
    rep = smoke_report(results)
    assert rep["all_healthy"] is True
    assert rep["degraded"] is False
    assert rep["failed"] == 0


def test_data_service_smoke_test_aggregates(monkeypatch):
    """DataService.smoke_test 在不触网情况下聚合注入的假源。"""
    from quant_agent.config import get_settings
    from quant_agent.data.service import DataService

    svc = DataService(get_settings())
    fake_sources = [
        _FakeSource(name="tushare", available=True, df=_good_df(5)),
        _FakeSource(name="akshare", available=True, raise_error=RuntimeError("nope")),
    ]
    monkeypatch.setattr(svc, "_sources", fake_sources)

    report = svc.smoke_test(stock_code="600519", days=5)
    assert report["total"] == 2
    assert report["ok"] == 1
    assert report["failed"] == 1
    names = {r["source"] for r in report["results"]}
    assert names == {"tushare", "akshare"}
