"""Smoke tests — require live network access to real data sources.

These are intentionally EXCLUDED from the default test run
(see ``pyproject.toml`` addopts: ``-m "not smoke"``) so that unit and
integration suites stay fully offline and reproducible.

Run them explicitly (manual or on a scheduled CI job) with::

    uv run pytest -m smoke

Each test enables sockets via ``@pytest.mark.enable_socket`` so the
global ``--disable-socket`` policy does not block it.
"""

from __future__ import annotations

import pytest


@pytest.mark.smoke
@pytest.mark.enable_socket
def test_akshare_spot_reachable():
    """Verify akshare can reach its live East Money endpoint.

    Smoke-only: hits the real network and is excluded from PR CI.
    """
    import akshare as ak

    df = ak.stock_zh_a_spot_em()
    assert df is not None and not df.empty


@pytest.mark.smoke
@pytest.mark.enable_socket
def test_efinance_history_reachable():
    """Verify efinance can fetch real daily history.

    Smoke-only: hits the real network and is excluded from PR CI.
    """
    from quant_agent.data.sources.efinance import EfinanceSource

    src = EfinanceSource()
    if not src.available:
        pytest.skip("efinance not installed")
    df = src.get_price_data("600519", days=5)
    assert df is not None and not df.empty
