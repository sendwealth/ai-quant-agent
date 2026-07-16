"""Web 服务 — 标准库实现，零额外依赖

提供：
- 静态前端 (index.html / app.js / styles.css)
- JSON API:
    GET  /api/health
    POST /api/analyze        {stock_code, days, offline, chart}
    GET  /api/search?q=&limit=       股票代码/名称智能搜索
    GET  /api/screen?top=&full_scan=&fundamentals=&deep=
    GET  /api/reports
    GET  /api/report?file=<name>
    GET  /api/report/latest?stock=<code>
    GET  /api/chart/<filename>   走势图 PNG

设计取舍：为坚持「零配置 / 离线可用」，刻意不引入 FastAPI/uvicorn，
仅用标准库 http.server。在受限网络环境也可直接运行。

为便于测试，所有业务逻辑抽成 `*_core` 纯函数，由 HTTP 处理器负责序列化。
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast
from urllib.parse import parse_qs, urlparse

from ..config import get_settings
from ..data.gate import evaluate_trust
from ..data.validators import validate_stock_code
from ..observability.alerting import AlertManager
from ..observability.health import build_health_report
from ..observability.metrics import MetricsCollector
from ..orchestrator import Orchestrator
from ..reporting import (
    latest_for_stock,
    list_reports,
    load_report,
    plot_price_chart,
    render_markdown,
    save_report,
)
from ..screener.stock_names import search_stocks

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = Path("data/reports")

# 进程级可观测性单例（P3.2 / P3.3）
METRICS = MetricsCollector()
ALERT_MANAGER = AlertManager()

# 进程内编排器缓存（key = offline 标志）
_ORCH_CACHE: dict[bool, Orchestrator] = {}
_ORCH_LOCK = threading.Lock()


def _get_orchestrator(offline: bool) -> Orchestrator:
    """惰性获取编排器，按 offline 标志缓存，避免重复初始化数据源。

    offline 以请求参数为准，实时切换在线/离线（不再依赖进程级环境变量）。
    """
    with _ORCH_LOCK:
        if offline not in _ORCH_CACHE:
            _ORCH_CACHE[offline] = Orchestrator(offline=offline)
        return _ORCH_CACHE[offline]


# ──────────────────────────────────────────────────────────────────────────
# 核心业务逻辑（可单测）
# ──────────────────────────────────────────────────────────────────────────


def health_core(smoke_results: dict | None = None, run_smoke: bool = False) -> dict:
    """聚合健康报告，反映数据源降级（P3.1）。

    默认不触网，仅基于配置推断降级（离线模式即视为降级）。设置
    ``run_smoke=True`` 会跑一次真实数据源冒烟并以其 ``degraded`` 为准。
    报告生成后会顺带触发告警规则检查（数据降级等）。
    """
    settings = get_settings()
    if smoke_results is None and run_smoke:
        try:
            from ..data.service import DataService

            smoke_results = DataService(settings).smoke_test()
        except Exception as e:  # 冒烟失败不阻塞健康端点
            logger.warning("health smoke failed: %s", e)
    report = build_health_report(settings, smoke_results=smoke_results)
    try:
        ALERT_MANAGER.check(report)
    except Exception:
        pass
    return report


def metrics_core() -> str:
    """返回 Prometheus 文本格式指标（P3.2）。"""
    return METRICS.to_prometheus()


def alerts_core() -> dict:
    """运行告警规则检查，返回触发的告警（P3.3）。"""
    settings = get_settings()
    report = build_health_report(settings)
    alerts = ALERT_MANAGER.check(report)
    return {"alerts": [a.to_dict() for a in alerts], "count": len(alerts)}


def _validate_stock_code_or_raise(code: str) -> None:
    """校验股票代码合法性（P3.4 输入校验）。"""
    try:
        validate_stock_code(code)
    except Exception as e:
        raise ValueError(f"非法股票代码: {code}") from e


def _is_authorized(handler: BaseHTTPRequestHandler) -> bool:
    """Bearer Token 鉴权（P3.4）。未配置 token 时不鉴权。"""
    token = os.environ.get("QUANT_WEB_AUTH_TOKEN")
    if not token:
        return True
    auth = handler.headers.get("Authorization", "")
    return auth == f"Bearer {token}"


def analyze_core(params: dict) -> dict:
    """执行单股分析，返回 {stock_code, markdown, report, chart_url}。"""
    stock_code = (params.get("stock_code") or "").strip()
    if not stock_code:
        raise ValueError("请提供 stock_code")
    _validate_stock_code_or_raise(stock_code)
    days = int(params.get("days", 120) or 120)
    offline = bool(params.get("offline", False))
    want_chart = bool(params.get("chart", False))

    orch = _get_orchestrator(offline)
    # 默认只读预览：分析但不下单，避免「点一下分析就建仓」。
    # 需要真正交易时调用 /api/execute（execute=True）。
    result = orch.analyze(stock_code, days=days, execute=False)

    try:
        save_report(result, base_dir=REPORTS_DIR)
    except Exception as e:
        logger.warning("保存历史失败: %s", e)

    chart_url = None
    if want_chart:
        try:
            price = orch.data.get_price_data(stock_code, days=days)
            chart_path = REPORTS_DIR / f"{stock_code}_chart.png"
            plot_price_chart(price, result, chart_path)
            chart_url = f"/api/chart/{stock_code}_chart.png"
        except Exception as e:
            logger.warning("图表生成失败: %s", e)

    return {
        "stock_code": result.stock_code,
        "markdown": render_markdown(result),
        "report": result.to_dict(),
        "chart_url": chart_url,
        "data_warning": bool(evaluate_trust(result.data_lineage, "report").reasons),
    }


def screen_core(qs: dict) -> dict:
    """执行智能选股，返回 {top_stocks, deep_reports}。"""
    top = int((qs.get("top", ["10"])[0]) or 10)
    full_scan = qs.get("full_scan", ["0"])[0] in ("1", "true", "True")
    fundamentals = qs.get("fundamentals", ["0"])[0] in ("1", "true", "True")
    deep = qs.get("deep", ["0"])[0] in ("1", "true", "True")
    offline = qs.get("offline", ["0"])[0] in ("1", "true", "True")

    orch = _get_orchestrator(offline)
    if deep:
        screen_result, reports = orch.screen_and_analyze(
            use_full_market=full_scan,
            top_n=top,
            include_fundamentals=fundamentals,
        )
        name_map = {s.stock_code: getattr(s, "name", None) or "" for s in screen_result.top_stocks}
        deep_reports = [
            {
                "stock_code": r.stock_code,
                "name": name_map.get(r.stock_code, ""),
                "report": r.to_dict(),
            }
            for r in reports
        ]
    else:
        screen_result = orch.screener.screen(
            use_full_market=full_scan,
            top_n=top,
            include_fundamentals=fundamentals,
        )
        deep_reports = []

    return {
        "top_stocks": [_scored_stock_to_dict(s) for s in screen_result.top_stocks],
        "deep_reports": deep_reports,
    }


def execute_core(params: dict) -> dict:
    """显式交易入口：基于分析结果真正下单（execute=True）。

    参数同 analyze_core（stock_code, days, offline）。返回下单后的报告。
    Web 的「分析」按钮默认只预览；需要建仓/平仓时调用此端点。
    """
    stock_code = (params.get("stock_code") or "").strip()
    if not stock_code:
        raise ValueError("请提供 stock_code")
    _validate_stock_code_or_raise(stock_code)
    days = int(params.get("days", 120) or 120)
    offline = bool(params.get("offline", False))

    orch = _get_orchestrator(offline)
    result = orch.analyze(stock_code, days=days, execute=True)

    try:
        save_report(result, base_dir=REPORTS_DIR)
    except Exception as e:
        logger.warning("保存历史失败: %s", e)

    return {
        "stock_code": result.stock_code,
        "executed": result.risk_result.signal if result.risk_result else "HOLD",
        "markdown": render_markdown(result),
        "report": result.to_dict(),
        "data_warning": bool(evaluate_trust(result.data_lineage, "report").reasons),
    }


def search_core(qs: dict) -> dict:
    """按代码/名称模糊搜索股票，返回 {results: [{code, name}]}。"""
    query = (qs.get("q", [""])[0] or "").strip()
    limit = int((qs.get("limit", ["10"])[0]) or 10)
    return {"results": search_stocks(query, limit=limit)}


def reports_core() -> dict:
    return {"reports": list_reports(REPORTS_DIR)}


def report_core(qs: dict, parts: list[str]) -> dict:
    """按文件名加载报告，或最新报告（parts[0]=='latest'）。"""
    if parts and parts[0] == "latest":
        stock = qs.get("stock", [None])[0]
        if not stock:
            raise ValueError("请提供 stock 参数")
        data = latest_for_stock(stock, REPORTS_DIR)
        if data is None:
            raise FileNotFoundError(f"未找到 {stock} 的历史报告")
        return {"report": data, "markdown": render_markdown_from_dict(data)}

    file = qs.get("file", [None])[0]
    if not file:
        raise ValueError("请提供 file 参数")
    data = load_report(file, REPORTS_DIR)
    if data is None:
        raise FileNotFoundError(f"未找到报告: {file}")
    return {"report": data, "markdown": render_markdown_from_dict(data)}


def _scored_stock_to_dict(s) -> dict:
    return {
        "stock_code": s.stock_code,
        "name": getattr(s, "name", None) or "",
        "price": getattr(s, "price", None),
        "total_score": getattr(s, "total_score", None),
        "technical_score": getattr(s, "technical_score", None),
        "momentum_score": getattr(s, "momentum_score", None),
        "liquidity_score": getattr(s, "liquidity_score", None),
        "fundamental_score": getattr(s, "fundamental_score", None),
    }


def render_markdown_from_dict(data: dict) -> str:
    """将历史报告 dict 渲染为 Markdown（轻量）"""
    lines = [
        f"# 历史报告 — {data.get('stock_code', '?')}",
        f"> 时间: {data.get('timestamp', '')}",
        "",
        f"- 信号: {data.get('signal')}",
        f"- 信心度: {data.get('confidence')}",
        f"- 建议仓位: {data.get('position_pct')}",
    ]
    # 数据谱系警示 (P1.2)
    warnings = data.get("lineage_warnings") or []
    if warnings:
        lines.append("")
        lines.append("> ⚠️ **数据谱系警示 (Data Lineage Warnings)**")
        for w in warnings:
            lines.append(f"> - {w}")
    if data.get("llm_analysis"):
        lines.append("\n## LLM 综合报告\n" + str(data.get("llm_analysis")))
    if data.get("risk_interpretation"):
        lines.append("\n## 风险解读\n" + str(data.get("risk_interpretation")))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# HTTP 处理器
# ──────────────────────────────────────────────────────────────────────────


def _sanitize(obj):
    """递归把 NaN/Infinity 等非标准 JSON 值替换为 None，确保浏览器可解析。"""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _json_response(handler, payload, status=200):
    body = json.dumps(_sanitize(payload), ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _error(handler, message: str, status=400):
    _json_response(handler, {"error": message}, status=status)


def _read_json_body(handler) -> dict:
    length = int(handler.headers.get("Content-Length", 0) or 0)
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    try:
        return cast("dict", json.loads(raw.decode("utf-8")))
    except Exception:
        return {}


def _api_dispatch(handler, path: str, qs: dict) -> bool:
    """处理 /api/* 路由；返回 True 表示已处理。"""
    METRICS.counter("http_requests_total", tags={"path": path, "method": handler.command})

    if path == "/api/health":
        _json_response(handler, health_core())
        return True
    if path == "/api/metrics":
        body = metrics_core().encode("utf-8")
        handler.send_response(200)
        handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)
        return True
    if path == "/api/alerts":
        _json_response(handler, alerts_core())
        return True
    if path == "/api/analyze":
        if handler.command != "POST":
            _error(handler, "仅支持 POST", status=405)
            return True
        if not _is_authorized(handler):
            _error(handler, "未授权：请提供 Authorization: Bearer <token>", status=401)
            return True
        _json_response(handler, analyze_core(_read_json_body(handler)))
        return True
    if path == "/api/execute":
        if handler.command != "POST":
            _error(handler, "仅支持 POST", status=405)
            return True
        if not _is_authorized(handler):
            _error(handler, "未授权：请提供 Authorization: Bearer <token>", status=401)
            return True
        _json_response(handler, execute_core(_read_json_body(handler)))
        return True
    if path == "/api/screen":
        _json_response(handler, screen_core(qs))
        return True
    if path == "/api/search":
        _json_response(handler, search_core(qs))
        return True
    if path == "/api/reports":
        _json_response(handler, reports_core())
        return True
    if path.startswith("/api/report"):
        _json_response(handler, report_core(qs, path[len("/api/report") :].strip("/").split("/")))
        return True
    if path.startswith("/api/chart/"):
        _serve_chart(handler, path[len("/api/chart/") :])
        return True
    return False


def _serve_static(handler, rel_path: str) -> None:
    if rel_path in ("", "/"):
        rel_path = "index.html"
    safe = Path(rel_path.lstrip("/"))
    full = (STATIC_DIR / safe).resolve()
    if not str(full).startswith(str(STATIC_DIR.resolve())) or not full.exists():
        handler.send_error(404, "Not Found")
        return
    ctype = {
        ".html": "text/html; charset=utf-8",
        ".js": "application/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".png": "image/png",
        ".ico": "image/x-icon",
    }.get(full.suffix, "application/octet-stream")
    body = full.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _serve_chart(handler, filename: str) -> None:
    full = (REPORTS_DIR / Path(filename).name).resolve()
    if not full.exists() or full.suffix.lower() != ".png":
        handler.send_error(404, "Not Found")
        return
    body = full.read_bytes()
    handler.send_response(200)
    handler.send_header("Content-Type", "image/png")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        logger.info("HTTP " + fmt, *args)

    def _handle(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        try:
            if path.startswith("/api/"):
                if _api_dispatch(self, path, qs):
                    return
                _error(self, f"未知 API: {path}", status=404)
                return
            if path.startswith("/static/"):
                _serve_static(self, path[len("/static/") :])
                return
            _serve_static(self, path)
        except ValueError as e:
            METRICS.counter("http_errors_total", tags={"type": "bad_request"})
            _error(self, str(e), status=400)
        except FileNotFoundError as e:
            METRICS.counter("http_errors_total", tags={"type": "not_found"})
            _error(self, str(e), status=404)
        except Exception as e:
            METRICS.counter("http_errors_total", tags={"type": "server_error"})
            logger.exception("API error")
            _error(self, f"服务器错误: {e}", status=500)

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def run_web(host: str = "127.0.0.1", port: int = 8000, offline: bool = False) -> None:
    """启动 Web 服务（阻塞）

    安全默认：仅监听 loopback（127.0.0.1）。若显式绑定到非本地地址
    （如 0.0.0.0 或公网 IP），会打印安全告警，提醒用户该端口将对外暴露。
    绑定到非 loopback 地址属于「显式选择」，本系统不默认开启。
    """
    if offline:
        os.environ["QUANT_OFFLINE_MODE"] = "true"

    # P4-04: 非 loopback 绑定需显式确认，避免误暴露到公网
    _LOOPBACK = ("127.0.0.1", "localhost", "::1")
    if host not in _LOOPBACK:
        print(
            "\n  ⚠️  安全提醒：Web 服务将监听非本地地址 "
            f"({host}:{port})，该端口可能对外部网络可见。\n"
            "      本系统默认仅用于研究/模拟，请勿在不受信任的网络中暴露。\n"
        )

    # 允许快速重启（TIME_WAIT 端口复用）
    ThreadingHTTPServer.allow_reuse_address = True
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}"
    print("\n  AI Quant Agent Web UI 已启动")
    print(f"  → {url}")
    print("  (Ctrl+C 退出)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭 Web 服务...")
        server.shutdown()


if __name__ == "__main__":
    run_web()
