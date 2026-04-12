"""AkShare 数据源适配器 — 行情数据（免费，主用）"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

import pandas as pd

from ..rate_limiter import RateLimiter
from .base import DataSource, FinancialSnapshot

logger = logging.getLogger(__name__)

# Only retry transient network/IO errors, not data or auth errors.
_RETRYABLE = (ConnectionError, TimeoutError, OSError)

# Proxy keys to bypass for domestic API calls
_PROXY_KEYS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy")


@contextmanager
def _no_proxy():
    """Temporarily disable system proxy for domestic API calls.

    On macOS, Clash/Surge intercept DNS and route traffic through a local
    proxy. This bypasses by removing env vars AND patching getproxies()
    so requests won't use system proxy settings.
    """
    saved = {}
    for key in _PROXY_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)

    # Patch urllib.getproxies to return empty dict (bypasses macOS system proxy)
    import urllib.request
    _orig_getproxies = urllib.request.getproxies
    urllib.request.getproxies = lambda: {}

    try:
        yield
    finally:
        os.environ.update(saved)
        urllib.request.getproxies = _orig_getproxies


class AkshareSource(DataSource):
    """AkShare 数据源 — 免费行情数据，带重试退避"""

    def __init__(
        self,
        timeout: int = 10,
        max_retries: int = 3,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self._rate_limiter = rate_limiter or RateLimiter(
            max_calls=60, period=60.0,
        )

    @property
    def name(self) -> str:
        return "akshare"

    @property
    def available(self) -> bool:
        try:
            import akshare as ak  # noqa: F401
            return True
        except ImportError:
            return False

    def _retry_call(self, func, *args, **kwargs):
        """带退避的重试（含限速）— 仅重试瞬态网络错误"""
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
                    logger.warning(f"AkShare {func.__name__} 第{attempt}次网络错误: {e}, {wait:.0f}s后重试")
                    time.sleep(wait)
            except Exception as e:
                # Non-retryable (auth, data format, rate-limit errors): fail fast
                logger.warning(f"AkShare {func.__name__} 非瞬态错误，不重试: {e}")
                raise
        raise last_err  # type: ignore

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """获取历史行情（AkShare）"""
        try:
            import akshare as ak
            from datetime import datetime, timedelta

            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")

            df = self._retry_call(
                ak.stock_zh_a_hist,
                symbol=stock_code,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust or "qfq",
            )
            if df is None or df.empty:
                return None

            # 标准化列名
            column_map = {
                "日期": "date", "开盘": "open", "收盘": "close",
                "最高": "high", "最低": "low", "成交量": "volume",
                "成交额": "amount", "涨跌幅": "pct_change", "换手率": "turnover",
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            df = df.sort_values("date").reset_index(drop=True)
            df = df.tail(days).reset_index(drop=True)

            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"✅ AkShare: 获取 {stock_code} {len(df)} 天行情")
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"AkShare 行情获取失败: {e}")
            return None

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格（腾讯财经）"""
        import requests

        try:
            self._rate_limiter.block_until_ready()
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://web.sqt.gtimg.cn/q={market}{stock_code}"
            with _no_proxy():
                resp = requests.get(url, timeout=self.timeout)
            if "~" in resp.text:
                parts = resp.text.split("~")
                if len(parts) > 3:
                    price = float(parts[3])
                    if price > 0:
                        return price
        except Exception as e:
            logger.warning(f"AkShare 实时价格获取失败: {e}")
        return None

    def get_financial_indicators(self, stock_code: str) -> Optional[pd.DataFrame]:
        """获取财务指标（AkShare）"""
        try:
            import akshare as ak
            df = self._retry_call(
                ak.stock_financial_analysis_indicator,
                symbol=stock_code,
            )
            if df is not None and not df.empty:
                logger.info(f"✅ AkShare: 获取 {stock_code} 财务指标 ({len(df)}期)")
                return df
        except Exception as e:
            logger.warning(f"AkShare 财务指标获取失败: {e}")
        return None

    def get_news(self, stock_code: str, count: int = 20) -> Optional[pd.DataFrame]:
        """获取个股新闻（东方财富源）

        Args:
            stock_code: A 股代码 (6 位数字)
            count: 获取新闻条数 (默认 20)

        Returns:
            DataFrame with columns: title, content, date, source
            or None on failure
        """
        try:
            import akshare as ak
            df = self._retry_call(
                ak.stock_news_em,
                symbol=stock_code,
            )
            if df is None or df.empty:
                return None

            # Standardize column names from akshare output
            column_map = {
                "新闻标题": "title",
                "新闻内容": "content",
                "发布时间": "date",
                "文章来源": "source",
                "新闻链接": "url",
            }
            df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
            df = df.head(count)

            logger.info(f"✅ AkShare: 获取 {stock_code} 新闻 ({len(df)}条)")
            return df
        except Exception as e:
            logger.warning(f"AkShare 新闻获取失败: {e}")
            return None

    def get_financial_snapshot(
        self, stock_code: str
    ) -> Optional[FinancialSnapshot]:
        """获取财务快照 — 从 AkShare 财务指标接口提取.

        Uses ``stock_financial_analysis_indicator`` to get quarterly metrics,
        maps Chinese column names to ``FinancialSnapshot`` fields.
        """
        try:
            df = self.get_financial_indicators(stock_code)
            if df is None or df.empty:
                return None

            latest = df.iloc[-1]

            # Chinese → English column mapping
            col_map = {
                "净资产收益率": "roe",
                "销售毛利率": "gross_margin",
                "销售净利率": "net_margin",
                "资产负债率": "debt_ratio",
                "流动比率": "current_ratio",
                "营业收入同比增长率": "revenue_growth",
                "净利润同比增长率": "profit_growth",
            }

            data: dict = {}
            for cn_name, en_key in col_map.items():
                if cn_name in latest.index:
                    try:
                        val = float(latest[cn_name])
                        # AkShare returns percentages as decimals (e.g. 0.22 for 22%)
                        # but sometimes as raw numbers. Normalize: if > 1, divide by 100.
                        if en_key in ("roe", "gross_margin", "net_margin", "debt_ratio",
                                      "revenue_growth", "profit_growth") and abs(val) > 1:
                            val = val / 100.0
                        data[en_key] = val
                    except (ValueError, TypeError):
                        pass

            # Report date
            for date_col in ["日期", "报告期"]:
                if date_col in latest.index:
                    data["report_date"] = str(latest[date_col])
                    break

            # Current price
            price = self.get_realtime_price(stock_code)
            if price is not None:
                data["price"] = price

            snapshot = FinancialSnapshot(stock_code, data)
            logger.info(
                f"AkShare: financial snapshot for {stock_code}, "
                f"{len([v for v in data.values() if v is not None])} fields"
            )
            return snapshot

        except Exception as e:
            logger.warning(f"AkShare financial snapshot failed for {stock_code}: {e}")
            return None
