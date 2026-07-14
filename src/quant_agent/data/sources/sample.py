"""本地样例数据源 — 仅提供内置的「真实历史行情 / 财务样例」

当所有网络数据源（Tushare/efinance/AkShare/BaoStock）均不可用，
或处于离线模式且无本地缓存时，尝试读取内置的**真实**历史样例：

- 行情：data/samples/price/{code}.parquet
- 财务：data/samples/financial/{code}.parquet

若样例不存在，一律返回 ``None`` 表示「没有数据」。
本数据源**绝不生成任何合成 / 模拟 / 演示数据**。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ..sources.base import DataSource, FinancialSnapshot
from ..normalizer import normalize_price_data
from ..validators import validate_stock_code

logger = logging.getLogger(__name__)


class SamplePriceSource(DataSource):
    """本地样例数据源 — 仅读取内置真实样例，无任何合成/模拟数据"""

    def __init__(self, settings=None, samples_dir: Optional[str] = None):
        self.settings = settings
        if samples_dir is None and settings is not None:
            samples_dir = getattr(settings, "sample_data_dir", "data/samples")
        self.samples_dir = Path(samples_dir or "data/samples")
        self._price_dir = self.samples_dir / "price"

    @property
    def name(self) -> str:
        return "sample"

    @property
    def available(self) -> bool:
        # 仅当内置真实样例目录存在时才算可用
        return self._price_dir.exists()

    # ── 行情 ──

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        stock_code = validate_stock_code(stock_code)

        bundled = self._price_dir / f"{stock_code}.parquet"
        if bundled.exists():
            try:
                df = pd.read_parquet(bundled)
                logger.info(
                    "样例源：使用内置真实历史行情 %s (%d 行)", stock_code, len(df)
                )
                return normalize_price_data(df)
            except Exception as e:
                logger.warning("样例源：读取内置样例失败 %s: %s", stock_code, e)
                return None

        logger.info("样例源：无 %s 的内置样例行情，返回无数据", stock_code)
        return None

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        df = self.get_price_data(stock_code, days=5)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    # ── 财务 ──

    def get_financial_snapshot(self, stock_code: str) -> Optional[FinancialSnapshot]:
        stock_code = validate_stock_code(stock_code)

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
                return None

        logger.info("样例源：无 %s 的内置样例财务，返回无数据", stock_code)
        return None
