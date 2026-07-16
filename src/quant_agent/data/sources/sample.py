"""本地样例数据源 — 内置「演示样例」行情 / 财务兜底

当所有网络数据源（Tushare/efinance/AkShare/BaoStock）均不可用，
或处于离线模式且无本地缓存时，尝试读取内置样例：

- 行情：{samples_dir}/price/{code}.parquet
- 财务：{samples_dir}/financial/{code}.parquet

查找顺序：先找当前工作区 ``data/samples``（用户可放置自己的真实样例），
再回退到随包发布的 ``quant_agent/data/samples``（演示样例）。

⚠️ 演示样例为**确定性合成数据**（固定随机种子，见
``scripts/generate_demo_samples.py``），仅用于离线演示与端到端测试，
**不代表任何真实行情或财务表现**。真实分析请配置数据源 token 并用
``quant-agent preload`` 预下载数据。样例缺失时一律返回 ``None``。
"""

from __future__ import annotations

import importlib.resources as importlib_resources
import logging
from pathlib import Path

import pandas as pd

from ..normalizer import normalize_price_data
from ..sources.base import DataSource, FinancialSnapshot
from ..validators import validate_stock_code

logger = logging.getLogger(__name__)


def _resolve_bundled_samples() -> Path | None:
    """定位随包发布的样例目录（pip install 后也能找到）。"""
    try:
        ref = importlib_resources.files("quant_agent") / "data" / "samples"
        path = Path(str(ref))
        if (path / "price").exists():
            return path
    except Exception:
        pass
    return None


class SamplePriceSource(DataSource):
    """本地样例数据源 — 读取内置/随包演示样例，无任何实时或合成抓取"""

    def __init__(self, settings=None, samples_dir: str | None = None):
        self.settings = settings
        if samples_dir is None and settings is not None:
            samples_dir = getattr(settings, "sample_data_dir", "data/samples")
        self.samples_dir = Path(samples_dir or "data/samples")
        self._price_dir = self.samples_dir / "price"
        self._financial_dir = self.samples_dir / "financial"
        # 随包发布的演示样例目录（post-install 兜底）
        self._bundled = _resolve_bundled_samples()

    @property
    def name(self) -> str:
        return "sample"

    @property
    def available(self) -> bool:
        # 工作区样例目录或随包演示样例存在即算可用
        return self._price_dir.exists() or (
            self._bundled is not None and (self._bundled / "price").exists()
        )

    def _price_path(self, stock_code: str) -> Path | None:
        cand = self._price_dir / f"{stock_code}.parquet"
        if cand.exists():
            return cand
        if self._bundled is not None:
            cand = self._bundled / "price" / f"{stock_code}.parquet"
            if cand.exists():
                return cand
        return None

    def _financial_path(self, stock_code: str) -> Path | None:
        cand = self._financial_dir / f"{stock_code}.parquet"
        if cand.exists():
            return cand
        if self._bundled is not None:
            cand = self._bundled / "financial" / f"{stock_code}.parquet"
            if cand.exists():
                return cand
        return None

    # ── 行情 ──

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> pd.DataFrame | None:
        stock_code = validate_stock_code(stock_code)

        path = self._price_path(stock_code)
        if path is not None:
            try:
                df = pd.read_parquet(path)
                logger.info("样例源：使用内置演示样例行情 %s (%d 行)", stock_code, len(df))
                return normalize_price_data(df)
            except Exception as e:
                logger.warning("样例源：读取内置样例失败 %s: %s", stock_code, e)
                return None

        logger.info("样例源：无 %s 的内置样例行情，返回无数据", stock_code)
        return None

    def get_realtime_price(self, stock_code: str) -> float | None:
        df = self.get_price_data(stock_code, days=5)
        if df is not None and not df.empty:
            return float(df["close"].iloc[-1])
        return None

    # ── 财务 ──

    def get_financial_snapshot(self, stock_code: str) -> FinancialSnapshot | None:
        stock_code = validate_stock_code(stock_code)

        path = self._financial_path(stock_code)
        if path is not None:
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    data = df.iloc[0].to_dict()
                    data.pop("index", None)
                    data.pop("stock_code", None)
                    logger.info("样例源：使用内置演示样例财务 %s", stock_code)
                    return FinancialSnapshot(stock_code, data)
            except Exception as e:
                logger.warning("样例源：读取内置财务样例失败 %s: %s", stock_code, e)
                return None

        logger.info("样例源：无 %s 的内置样例财务，返回无数据", stock_code)
        return None
