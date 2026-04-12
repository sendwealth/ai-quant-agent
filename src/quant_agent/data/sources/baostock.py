"""BaoStock 数据源适配器 — 免费、不限频、无需token"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import pandas as pd

from ..rate_limiter import RateLimiter
from .base import DataSource

logger = logging.getLogger(__name__)

# 复权类型映射: qfq=前复权(2), hfq=后复权(1), 其他=不复权(3)
_ADJUST_MAP = {"qfq": "2", "hfq": "1"}

# Only retry transient errors
_RETRYABLE = (ConnectionError, TimeoutError, OSError)


def _retry(max_retries: int = 3):
    """Decorator: retry transient network/IO errors with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except _RETRYABLE as e:
                    last_err = e
                    if attempt < max_retries:
                        wait = 2 ** (attempt - 1)
                        logger.warning(
                            f"BaoStock {func.__name__} attempt {attempt} failed: {e}, "
                            f"retry in {wait}s"
                        )
                        time.sleep(wait)
            raise last_err  # type: ignore
        return wrapper
    return decorator


class BaoStockSource(DataSource):
    """BaoStock 数据源 — 免费行情数据，无需 token

    支持上下文管理器，推荐用法::

        with BaoStockSource() as src:
            df = src.get_price_data("300750")

    向后兼容：也可直接实例化使用，首次查询自动登录，对象销毁时自动登出。
    """

    def __init__(self):
        self._logged_in = False
        self._bs = None
        self._owns_login = False  # True when this instance called login()
        self._rate_limiter = RateLimiter(max_calls=100, period=60.0)

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self) -> BaoStockSource:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.logout()

    # ── Properties ────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "baostock"

    @property
    def available(self) -> bool:
        try:
            import baostock  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Login lifecycle ───────────────────────────────────────────────

    def _ensure_logged_in(self) -> None:
        """确保已登录。仅在本实例未登录时执行 login()。

        使用 BaoStock 全局状态：如果外部代码已调用 bs.login()，
        我们检测到 _bs 为 None 就主动探测一次登录状态，避免重复登录。
        """
        if self._logged_in:
            return

        import baostock as bs

        lg = bs.login()
        if lg.error_code != "0":
            raise ConnectionError(f"BaoStock login failed: {lg.error_msg}")
        self._bs = bs
        self._logged_in = True
        self._owns_login = True

    def logout(self) -> None:
        """登出 BaoStock（幂等，重复调用安全）"""
        if self._logged_in and self._bs is not None:
            try:
                self._bs.logout()
            except Exception:
                logger.debug("BaoStock logout error (ignored)", exc_info=True)
            finally:
                self._logged_in = False
                self._owns_login = False
                self._bs = None

    # Backward-compatible alias
    _ensure_login = _ensure_logged_in

    # ── Helpers ───────────────────────────────────────────────────────

    def _to_bs_code(self, stock_code: str) -> str:
        """转换为 BaoStock 代码格式 (sh.xxxxxx / sz.xxxxxx)"""
        if stock_code.startswith("6") or stock_code.startswith("9"):
            return f"sh.{stock_code}"
        return f"sz.{stock_code}"

    # ── Public API ────────────────────────────────────────────────────

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """获取历史日线行情"""
        try:
            self._rate_limiter.block_until_ready()
            return self._get_price_data_impl(stock_code, days, adjust)
        except _RETRYABLE as e:
            # Re-login on connection errors, then retry once
            logger.warning(f"BaoStock {stock_code} connection error, re-logging in: {e}")
            self.logout()
            try:
                self._rate_limiter.block_until_ready()
                return self._get_price_data_impl(stock_code, days, adjust)
            except Exception as e2:
                logger.warning(f"BaoStock {stock_code} failed after re-login: {e2}")
                return None
        except Exception as e:
            logger.warning(f"BaoStock {stock_code} 失败: {e}")
            return None

    @_retry(max_retries=2)
    def _get_price_data_impl(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """Internal price data fetch with retry."""
        self._ensure_logged_in()
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=int(days * 1.5))).strftime(
            "%Y-%m-%d"
        )

        adjust_flag = _ADJUST_MAP.get(adjust, "3")
        bs_code = self._to_bs_code(stock_code)

        rs = self._bs.query_history_k_data_plus(
            bs_code,
            "date,open,high,low,close,volume,amount,turn",
            start_date=start,
            end_date=end,
            frequency="d",
            adjustflag=adjust_flag,
        )
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None

        df = pd.DataFrame(
            rows,
            columns=[
                "date", "open", "high", "low", "close",
                "volume", "amount", "turnover",
            ],
        )
        df = df[df["date"] != ""].reset_index(drop=True)
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        df["amount"] = df["amount"].astype(float) / 1000
        return df

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """BaoStock 不支持实时行情"""
        return None

    def get_price_data_batch(
        self, codes: list[str], days: int = 250, adjust: str = "qfq"
    ) -> dict[str, pd.DataFrame]:
        """批量获取日线（一次登录）"""
        self._ensure_logged_in()
        results = {}
        for code in codes:
            df = self.get_price_data(code, days, adjust)
            if df is not None:
                results[code] = df
        return results

    def get_stock_list(self) -> pd.DataFrame:
        """获取 A 股股票列表"""
        self._ensure_logged_in()
        rs = self._bs.query_stock_basic()
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=["code", "code_name", "ipoDate", "outDate", "type", "status"],
        )
        df = df[df["status"] == "1"].copy()
        df["symbol"] = df["code"].str.extract(r"\.(.*)")
        df["name"] = df["code_name"]
        df = df[~df["name"].str.contains("ST|退")]
        df = df[~df["symbol"].str.startswith("688")]
        df = df[~df["symbol"].str.startswith("8")]
        df = df[~df["symbol"].str.startswith("4")]
        return df[["symbol", "name"]].reset_index(drop=True)
