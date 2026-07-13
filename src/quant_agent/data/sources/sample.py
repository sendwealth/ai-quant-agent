"""样例数据源 — 开箱即用的离线兜底

当所有网络数据源（Tushare/efinance/AkShare/BaoStock）均不可用，
或处于离线模式且无本地缓存时，SamplePriceSource 提供兜底数据：

1. 优先读取内置的「真实历史行情样例」parquet（data/samples/price/{code}.parquet）；
2. 若不存在，则基于股票代码生成「确定性合成演示行情」，保证任何代码都能跑通全流程。

合成数据仅用于演示与功能验证，会在日志中明确标注 [DEMO]，不应作为投资依据。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..sources.base import DataSource, FinancialSnapshot
from ..normalizer import normalize_price_data
from ..validators import validate_stock_code

logger = logging.getLogger(__name__)


class SamplePriceSource(DataSource):
    """离线兜底数据源 — 样例优先，合成演示次之"""

    def __init__(self, settings=None, samples_dir: Optional[str] = None):
        self.settings = settings
        if samples_dir is None and settings is not None:
            samples_dir = getattr(settings, "sample_data_dir", "data/samples")
        self.samples_dir = Path(samples_dir)
        self._price_dir = self.samples_dir / "price"

    @property
    def name(self) -> str:
        return "sample"

    @property
    def available(self) -> bool:
        # 样例源始终可用（可生成合成数据兜底）
        return True

    # ── 行情 ──

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        stock_code = validate_stock_code(stock_code)

        # 1. 优先使用内置真实样例
        bundled = self._price_dir / f"{stock_code}.parquet"
        if bundled.exists():
            try:
                df = pd.read_parquet(bundled)
                logger.info("样例源：使用内置真实历史行情 %s (%d 行)", stock_code, len(df))
                return normalize_price_data(df)
            except Exception as e:
                logger.warning("样例源：读取内置样例失败 %s: %s", stock_code, e)

        # 2. 生成确定性合成演示行情
        logger.warning(
            "样例源：为 %s 生成 [DEMO] 合成演示行情（非真实数据，仅供功能验证）",
            stock_code,
        )
        return self._generate_synthetic(stock_code, days)

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        df = self.get_price_data(stock_code, days=5)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    # ── 财务 ──

    def get_financial_snapshot(self, stock_code: str) -> Optional[FinancialSnapshot]:
        stock_code = validate_stock_code(stock_code)

        # 优先读取内置样例财务
        bundled = self.samples_dir / "financial" / f"{stock_code}.parquet"
        if bundled.exists():
            try:
                df = pd.read_parquet(bundled)
                if not df.empty:
                    data = df.iloc[0].to_dict()
                    data.pop("index", None)
                    data.pop("stock_code", None)
                    logger.info("样例源：使用内置样例财务 %s", stock_code)
                    return FinancialSnapshot(stock_code, data)
            except Exception as e:
                logger.warning("样例源：读取内置财务样例失败 %s: %s", stock_code, e)

        # 否则生成合成演示财务
        logger.warning("样例源：为 %s 生成 [DEMO] 合成演示财务", stock_code)
        return FinancialSnapshot(stock_code, self._generate_synthetic_financial(stock_code))

    # ── 合成数据生成 ──

    @staticmethod
    def _seed_for(code: str) -> int:
        """由股票代码派生确定性随机种子"""
        h = 0
        for ch in code:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h

    def _generate_synthetic(self, stock_code: str, days: int) -> pd.DataFrame:
        """生成确定性的合成 OHLCV 行情（几何随机游走）"""
        rng = np.random.default_rng(self._seed_for(stock_code))

        # 起始价格：按代码映射到 10~2000 之间的合理量级
        base = 20.0 + (self._seed_for(stock_code) % 2000) / 10.0
        n = max(int(days), 30)
        daily_returns = rng.normal(0.0005, 0.02, size=n)
        prices = base * np.exp(np.cumsum(daily_returns))
        prices = np.maximum(prices, 1.0)

        # 生成交易日序列（截至最近一个工作日）
        end = pd.Timestamp.now().normalize()
        dates = pd.bdate_range(end=end, periods=n)

        closes = prices
        opens = np.concatenate([[closes[0]], closes[:-1]]) * (
            1 + rng.normal(0, 0.005, size=n)
        )
        highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.01, size=n)))
        lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.01, size=n)))
        volumes = rng.integers(1_000_000, 50_000_000, size=n).astype(float)

        df = pd.DataFrame(
            {
                "date": dates,
                "open": np.round(opens, 2),
                "high": np.round(highs, 2),
                "low": np.round(lows, 2),
                "close": np.round(closes, 2),
                "volume": volumes,
            }
        )
        df.attrs["source"] = "sample-demo"
        return normalize_price_data(df)

    @staticmethod
    def _generate_synthetic_financial(stock_code: str) -> dict[str, Any]:
        """生成确定性合成演示财务指标"""
        rng = np.random.default_rng(SamplePriceSource._seed_for(stock_code) + 7)
        report_date = pd.Timestamp.now().normalize().strftime("%Y-%m-%d")
        return {
            "roe": round(float(rng.uniform(5, 25)), 2),
            "gross_margin": round(float(rng.uniform(20, 60)), 2),
            "net_margin": round(float(rng.uniform(5, 30)), 2),
            "debt_ratio": round(float(rng.uniform(20, 60)), 2),
            "current_ratio": round(float(rng.uniform(1.0, 3.0)), 2),
            "revenue_growth": round(float(rng.uniform(-10, 40)), 2),
            "profit_growth": round(float(rng.uniform(-20, 50)), 2),
            "pe_ttm": round(float(rng.uniform(10, 60)), 2),
            "pb": round(float(rng.uniform(1, 10)), 2),
            "report_date": report_date,
        }
