"""P3.5 Web E2E — 真实启动 http.server 并请求核心端点（离线、不触网）。

注意：本文件使用真实 socket，需用 ``@pytest.mark.enable_socket`` 解除
pytest-socket 的禁用（与 CI 的 --disable-socket 兼容）。
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from quant_agent.web.server import _Handler


def _start_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, port


def _stop(server):
    try:
        server.shutdown()
    finally:
        server.server_close()


@pytest.mark.enable_socket
def test_health_endpoint_e2e():
    server, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/health", timeout=5) as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert "status" in data
            assert data.get("app")
            assert "components" in data
    finally:
        _stop(server)


@pytest.mark.enable_socket
def test_metrics_endpoint_e2e():
    server, port = _start_server()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/metrics", timeout=5) as resp:
            assert resp.status == 200
            ctype = resp.headers.get("Content-Type", "")
            assert "text/plain" in ctype
            body = resp.read().decode()
            assert "http_requests_total" in body
    finally:
        _stop(server)


@pytest.mark.enable_socket
def test_analyze_invalid_code_returns_400():
    server, port = _start_server()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/analyze",
            data=json.dumps({"stock_code": "not_a_code"}).encode(),
            method="POST",
        )
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "expected HTTPError 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        _stop(server)


@pytest.mark.enable_socket
def test_analyze_requires_auth_when_token_set():
    os.environ["QUANT_WEB_AUTH_TOKEN"] = "e2e-secret"
    try:
        server, port = _start_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/analyze",
                data=json.dumps({"stock_code": "300750"}).encode(),
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                assert False, "expected HTTPError 401 without token"
            except urllib.error.HTTPError as e:
                assert e.code == 401
        finally:
            _stop(server)
    finally:
        os.environ.pop("QUANT_WEB_AUTH_TOKEN", None)


@pytest.mark.enable_socket
def test_analyze_authorized_with_token():
    os.environ["QUANT_WEB_AUTH_TOKEN"] = "e2e-secret"
    try:
        server, port = _start_server()
        try:
            req = urllib.request.Request(
                f"http://127.0.0.1:{port}/api/analyze",
                data=json.dumps({"stock_code": "300750"}).encode(),
                method="POST",
                headers={"Authorization": "Bearer e2e-secret"},
            )
            try:
                urllib.request.urlopen(req, timeout=5)
                # 鉴权通过；离线无数据可能进入 400/500，但不能是 401
            except urllib.error.HTTPError as e:
                assert e.code != 401
        finally:
            _stop(server)
    finally:
        os.environ.pop("QUANT_WEB_AUTH_TOKEN", None)
