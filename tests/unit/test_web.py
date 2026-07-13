"""Web UI 集成测试 — 进程内启动 HTTP 服务并用 urllib 请求

避免后台进程对测试运行器的干扰；覆盖 health / analyze / reports / report / chart / screen。
"""

import json
import os
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from quant_agent.web.server import _Handler, run_web

PORT = 8765


@pytest.fixture(scope="module")
def server():
    # 离线模式：确保样例兜底可用，不触网
    os.environ.setdefault("QUANT_OFFLINE_MODE", "true")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), _Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{PORT}"
    srv.shutdown()
    srv.server_close()


def _get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def test_health(server):
    status, body = _get(f"{server}/api/health")
    assert status == 200
    assert body["status"] == "ok"
    assert "app" in body


def test_analyze_offline(server):
    status, body = _post(
        f"{server}/api/analyze",
        {"stock_code": "600519", "days": 120, "offline": True, "chart": True},
    )
    assert status == 200
    assert body["stock_code"] == "600519"
    assert "markdown" in body and len(body["markdown"]) > 0
    assert "report" in body and "signal" in body["report"]
    # 图表应已生成并可访问
    assert body["chart_url"]
    with urllib.request.urlopen(f"{server}{body['chart_url']}", timeout=10) as r:
        assert r.status == 200
        assert r.headers.get("Content-Type") == "image/png"


def test_reports_list_and_show(server):
    # 先确保有一份报告
    _post(
        f"{server}/api/analyze",
        {"stock_code": "600519", "days": 60, "offline": True, "chart": False},
    )
    status, body = _get(f"{server}/api/reports")
    assert status == 200
    assert isinstance(body["reports"], list)
    assert len(body["reports"]) >= 1

    fname = body["reports"][0]["file"]
    status, body2 = _get(f"{server}/api/report?file={fname}")
    assert status == 200
    assert "markdown" in body2
    # 最新报告
    status, body3 = _get(f"{server}/api/report/latest?stock=600519")
    assert status == 200
    assert body3["report"]["stock_code"] == "600519"


def test_report_not_found(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(f"{server}/api/report?file=missing.json", timeout=10)
    assert e.value.code == 404


def test_analyze_missing_code(server):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError):
        _post(f"{server}/api/analyze", {"stock_code": ""})


def test_static_index(server):
    with urllib.request.urlopen(f"{server}/", timeout=10) as r:
        html = r.read().decode("utf-8")
        assert "<title>" in html
    with urllib.request.urlopen(f"{server}/static/app.js", timeout=10) as r:
        assert "application/javascript" in r.headers.get("Content-Type", "")


def test_screen_returns_stock_names():
    """选股结果应附带股票名称（离线默认池通过内置映射解析）。"""
    import pandas as pd

    from quant_agent.screener.engine import ScreeningEngine, STOCK_NAME_MAP

    def _fake_multi_price(codes, days=120, **_):
        out = {}
        for c in codes:
            out[c] = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=30, freq="D"),
                    "open": [10] * 30,
                    "close": [10 + i for i in range(30)],
                    "high": [11] * 30,
                    "low": [9] * 30,
                    "volume": [1_000_000] * 30,
                }
            )
        return out

    class _FakeDS:
        def get_multi_price(self, codes, days=120, **_):
            return _fake_multi_price(codes, days=days)

    eng = ScreeningEngine(data_service=_FakeDS())
    res = eng.screen(stock_codes=["600519", "601318", "000001"], top_n=3)
    names = {s.stock_code: s.name for s in res.top_stocks}
    assert names["600519"] == STOCK_NAME_MAP["600519"] == "贵州茅台"
    assert names["601318"] == STOCK_NAME_MAP["601318"] == "中国平安"
    assert names["000001"] == STOCK_NAME_MAP["000001"] == "平安银行"
    # 透传到 web 层序列化
    from quant_agent.web.server import _scored_stock_to_dict

    d = _scored_stock_to_dict(res.top_stocks[0])
    assert d["name"] == "贵州茅台"


def test_full_market_name_resolution():
    """全市场缓存加载后，即使不在内置池中的代码也能解析名称。"""
    from quant_agent.screener.stock_names import (
        CACHE_STOCK_NAME_MAP,
        get_stock_name,
    )

    # 缓存应来自 data/stock_names.json（全市场级别，数千条）
    assert len(CACHE_STOCK_NAME_MAP) > 1000
    # 这些代码不在内置 DEFAULT_POOL 中，但应在全市场缓存里
    assert get_stock_name("301269") == "华大九天"
    assert get_stock_name("688981") == "中芯国际"
    # 未知代码返回空
    assert get_stock_name("999999") == ""

