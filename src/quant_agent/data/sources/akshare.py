"""AkShare 数据源适配器 — 行情数据（免费，主用）"""

from __future__ import annotations

import logging
import time
from typing import Optional

import pandas as pd

from ..rate_limiter import RateLimiter
from .base import DataSource

logger = logging.getLogger(__name__)


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
        """带退避的重试（含限速）"""
        self._rate_limiter.block_until_ready()
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(f"AkShare {func.__name__} 第{attempt}次失败: {e}, {wait:.0f}s后重试")
                    time.sleep(wait)
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
