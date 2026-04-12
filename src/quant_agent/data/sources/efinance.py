"""efinance 数据源适配器 — 免费稳定，基于东方财富 API"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from ..rate_limiter import RateLimiter
from .base import DataSource, FinancialSnapshot

logger = logging.getLogger(__name__)

# Only retry transient network/IO errors
_RETRYABLE = (ConnectionError, TimeoutError, OSError)

# Proxy keys to bypass for domestic API calls
_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy")


@contextmanager
def _no_proxy():
    """Temporarily disable system proxy for domestic API calls.

    On macOS, Clash/Surge/V2Ray intercept DNS and route traffic through
    a local proxy (127.0.0.1:1082). This causes ProxyError for domestic
    APIs like 东方财富. We bypass by:
    1. Removing proxy env vars
    2. Patching efinance's requests.Session to ignore system proxy
    3. Patching urllib.getproxies to return empty dict
    """
    saved_env = {}
    for key in _PROXY_KEYS:
        if key in os.environ:
            saved_env[key] = os.environ.pop(key)

    # Patch efinance's internal session to ignore system proxy
    patched_sessions = []
    try:
        import efinance.common.getter as _eg
        if hasattr(_eg, "session") and _eg.session.trust_env:
            _eg.session.trust_env = False
            patched_sessions.append(_eg.session)
    except ImportError:
        pass

    # Patch urllib.getproxies to return empty (bypasses macOS system proxy)
    import urllib.request
    _orig_getproxies = urllib.request.getproxies
    urllib.request.getproxies = lambda: {}

    try:
        yield
    finally:
        os.environ.update(saved_env)
        for s in patched_sessions:
            s.trust_env = True
        urllib.request.getproxies = _orig_getproxies

# Column name mapping: efinance Chinese → English
_PRICE_COL_MAP = {
    "股票代码": "code",
    "股票名称": "name",
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_change",
    "换手率": "turnover",
}


class EfinanceSource(DataSource):
    """efinance 数据源 — 免费，基于东方财富 API，无需 token。

    优势：稳定、不限频（保守 120/min）、支持价格和基础财务数据。
    """

    def __init__(
        self,
        max_retries: int = 3,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.max_retries = max_retries
        self._rate_limiter = rate_limiter or RateLimiter(
            max_calls=120, period=60.0,
        )

    @property
    def name(self) -> str:
        return "efinance"

    @property
    def available(self) -> bool:
        try:
            import efinance  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Helpers ───────────────────────────────────────────────────────

    def _retry_call(self, func, *args, **kwargs):
        """Retry transient errors only, with exponential backoff."""
        self._rate_limiter.block_until_ready()
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with _no_proxy():
                    return func(*args, **kwargs)
            except _RETRYABLE as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(
                        f"efinance {func.__name__} attempt {attempt}: {e}, "
                        f"retry in {wait}s"
                    )
                    time.sleep(wait)
            except Exception as e:
                logger.warning(f"efinance {func.__name__} non-transient error: {e}")
                raise
        raise last_err  # type: ignore

    # ── Public API ────────────────────────────────────────────────────

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """获取历史日线行情。

        Uses ``ef.stock.get_quote_history()``.
        """
        try:
            import efinance as ef

            start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime(
                "%Y%m%d"
            )

            # klt=101: daily, fqt: 1=qfq, 2=hfq, 0=none
            fqt = 1 if adjust == "qfq" else (2 if adjust == "hfq" else 0)

            df = self._retry_call(
                ef.stock.get_quote_history,
                stock_code,
                beg=start_date,
                klt=101,
                fqt=fqt,
            )

            if df is None or df.empty:
                return None

            # Rename columns
            df = df.rename(
                columns={k: v for k, v in _PRICE_COL_MAP.items() if k in df.columns}
            )

            # Keep only standard columns that exist
            standard_cols = ["date", "open", "high", "low", "close", "volume", "amount"]
            available_cols = [c for c in standard_cols if c in df.columns]
            df = df[available_cols].copy()

            # Ensure numeric types
            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values("date").reset_index(drop=True)
            df = df.tail(days).reset_index(drop=True)
            df = df.dropna(subset=["close"])

            if df.empty:
                return None

            logger.info(f"efinance: fetched {stock_code} {len(df)} bars")
            return df

        except Exception as e:
            logger.warning(f"efinance price data failed for {stock_code}: {e}")
            return None

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格 — 从最新 quote history 行获取."""
        try:
            import efinance as ef

            self._rate_limiter.block_until_ready()

            with _no_proxy():
                df = ef.stock.get_quote_history(stock_code, klt=101, fqt=1, lmt=1)
            if df is not None and not df.empty:
                col = "收盘" if "收盘" in df.columns else "close"
                price = float(df.iloc[-1][col])
                if price > 0:
                    return price
        except Exception as e:
            logger.warning(f"efinance realtime price failed for {stock_code}: {e}")
        return None

    def get_financial_snapshot(
        self, stock_code: str
    ) -> Optional[FinancialSnapshot]:
        """获取财务快照 — 从 efinance 公司业绩数据提取.

        Uses ``ef.stock.get_all_company_performance()`` which returns
        quarterly financial metrics.
        """
        try:
            import efinance as ef

            self._rate_limiter.block_until_ready()

            df = ef.stock.get_all_company_performance(stock_code)
            if df is None or df.empty:
                return None

            # Take the latest row
            latest = df.iloc[-1]

            # Map efinance columns to FinancialSnapshot fields
            data: dict = {}

            # Column name mapping (efinance Chinese → our keys)
            col_map = {
                "净资产收益率": "roe",
                "毛利率": "gross_margin",
                "净利率": "net_margin",
                "资产负债率": "debt_ratio",
                "流动比率": "current_ratio",
                "营收同比增长率": "revenue_growth",
                "净利润同比增长率": "profit_growth",
                "市盈率": "pe_ttm",
                "市净率": "pb",
            }

            for cn_name, en_key in col_map.items():
                if cn_name in latest.index:
                    val = latest[cn_name]
                    try:
                        val = float(val)
                        # Convert percentages: efinance may return "22.5" meaning 22.5%
                        if en_key in ("roe", "gross_margin", "net_margin", "debt_ratio",
                                      "revenue_growth", "profit_growth") and abs(val) > 1:
                            val = val / 100.0
                        data[en_key] = val
                    except (ValueError, TypeError):
                        pass

            # Report date
            for date_col in ["报告期", "日期"]:
                if date_col in latest.index:
                    data["report_date"] = str(latest[date_col])
                    break

            # Get current price from price data
            price = self.get_realtime_price(stock_code)
            if price is not None:
                data["price"] = price

            snapshot = FinancialSnapshot(stock_code, data)
            logger.info(
                f"efinance: financial snapshot for {stock_code}, "
                f"{len([v for v in data.values() if v is not None])} fields"
            )
            return snapshot

        except Exception as e:
            logger.warning(f"efinance financial snapshot failed for {stock_code}: {e}")
            return None
