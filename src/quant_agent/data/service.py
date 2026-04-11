"""数据服务 — 统一入口，多数据源降级 + 缓存"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd

from ..config import Settings, get_settings
from .sources.base import DataSource, FinancialSnapshot
from .sources.tushare import TushareSource
from .sources.akshare import AkshareSource
from .normalizer import normalize_price_data
from .validator import validate_price_data, clean_price_data
from .validators import validate_stock_code
from .store import DataStore

logger = logging.getLogger(__name__)


class DataService:
    """数据服务统一入口

    数据获取优先级：缓存 → 本地存储 → Tushare → AkShare
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.store = DataStore(self.settings.parquet_dir)

        # 初始化数据源
        self._sources: list[DataSource] = []

        # Tushare（财务数据主力）
        try:
            self._tushare = TushareSource(token=self.settings.tushare_token)
            if self._tushare.available:
                self._sources.append(self._tushare)
                logger.info("✅ Tushare 数据源就绪")
        except Exception as e:
            logger.warning(f"⚠️ Tushare 初始化失败: {e}")

        # AkShare（行情数据主力，免费）
        try:
            self._akshare = AkshareSource(timeout=self.settings.akshare_timeout)
            if self._akshare.available:
                self._sources.append(self._akshare)
                logger.info("✅ AkShare 数据源就绪")
        except Exception as e:
            logger.warning(f"⚠️ AkShare 初始化失败: {e}")

    @property
    def tushare(self) -> Optional[TushareSource]:
        return self._tushare if hasattr(self, "_tushare") else None

    @property
    def akshare(self) -> Optional[AkshareSource]:
        return self._akshare if hasattr(self, "_akshare") else None

    # ── 行情数据 ──

    def get_price_data(
        self,
        stock_code: str,
        days: int = 250,
        use_cache: bool = True,
        clean: bool = True,
    ) -> Optional[pd.DataFrame]:
        """获取标准化行情数据

        优先级: 缓存/存储 → 各数据源依次尝试
        """
        stock_code = validate_stock_code(stock_code)
        # 1. 尝试缓存
        if use_cache and self.store.is_fresh(stock_code, max_age_hours=4):
            df = self.store.load_price(stock_code)
            if df is not None and len(df) >= days * 0.8:
                logger.info(f"📦 缓存命中: {stock_code} ({len(df)}行)")
                return normalize_price_data(df)

        # 2. 依次尝试数据源
        for source in self._sources:
            df = source.get_price_data(stock_code, days)
            if df is not None and not df.empty:
                df = normalize_price_data(df)

                # 校验
                report = validate_price_data(df)
                if not report.is_valid:
                    logger.warning(f"数据校验失败 ({source.name}): {report.errors}")
                    continue

                # 清洗
                if clean:
                    df = clean_price_data(df)

                # 持久化
                self.store.save_price(stock_code, df, source=source.name)
                return df

        logger.error(f"❌ 所有数据源失败: {stock_code}")
        return None

    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格"""
        stock_code = validate_stock_code(stock_code)
        for source in self._sources:
            price = source.get_realtime_price(stock_code)
            if price and price > 0:
                return price
        return None

    # ── 财务数据 ──

    def get_financial_snapshot(
        self, stock_code: str, max_age_days: int = 365
    ) -> Optional[FinancialSnapshot]:
        """获取财务快照（真实数据）

        优先使用 Tushare（财务报表数据最全），降级到本地缓存

        Args:
            stock_code: 股票代码
            max_age_days: 缓存财务数据最大允许天数（默认365天）
        """
        stock_code = validate_stock_code(stock_code)
        # 1. Tushare（真实财务报表）
        if self.tushare and self.tushare.available:
            snapshot = self.tushare.get_financial_snapshot(stock_code)
            if snapshot is not None:
                # 持久化
                self.store.save_financial(stock_code, snapshot.to_dict(), source="tushare")
                return snapshot

        # 2. 本地缓存（检查过期）
        cached = self.store.load_financial(stock_code, latest=True)
        if cached is not None and not cached.empty:
            data = cached.iloc[0].to_dict()
            data.pop("index", None)
            report_date = data.get("report_date") or data.get("end_date")
            if report_date is not None:
                try:
                    from datetime import datetime

                    if isinstance(report_date, str):
                        rd = datetime.strptime(report_date[:10], "%Y-%m-%d")
                    else:
                        rd = pd.Timestamp(report_date).to_pydatetime()
                    age_days = (datetime.now() - rd).days
                    if age_days > max_age_days:
                        logger.warning(
                            f"⚠️ 缓存财务数据过期 ({age_days}天 > {max_age_days}天): {stock_code}"
                        )
                        return None
                except (ValueError, TypeError):
                    pass
            logger.info(f"📦 使用缓存财务数据: {stock_code}")
            return FinancialSnapshot(stock_code, data)

        logger.error(f"❌ 无法获取财务数据: {stock_code}")
        return None

    def get_financial_statements(
        self, stock_code: str, statement_type: str, periods: int = 4
    ) -> Optional[pd.DataFrame]:
        """获取原始财务报表"""
        stock_code = validate_stock_code(stock_code)
        if self.tushare and self.tushare.available:
            from .sources.base import StatementType
            st = StatementType(statement_type)
            return self.tushare.get_financial_statements(stock_code, st, periods)
        return None

    # ── 批量操作 ──

    def get_multi_price(
        self,
        stock_codes: list[str],
        days: int = 250,
        max_workers: Optional[int] = None,
    ) -> dict[str, pd.DataFrame]:
        """批量获取行情（支持并发）

        Args:
            stock_codes: 股票代码列表
            days: 获取天数
            max_workers: 并发线程数，None 时使用 settings.fetch_max_workers，
                         1 时退化为顺序执行

        Returns:
            dict[str, pd.DataFrame]: 成功获取的股票行情，失败的静默跳过
        """
        workers = max_workers if max_workers is not None else self.settings.fetch_max_workers

        # Sequential fallback when max_workers == 1
        if workers <= 1:
            results = {}
            for code in stock_codes:
                df = self.get_price_data(code, days)
                if df is not None:
                    results[code] = df
            return results

        # Concurrent execution
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(code: str) -> tuple[str, Optional[pd.DataFrame]]:
            try:
                return code, self.get_price_data(code, days)
            except Exception as exc:
                logger.warning(f"并发获取 {code} 异常: {exc}")
                return code, None

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_fetch_one, code): code for code in stock_codes}
            for future in as_completed(futures):
                code, df = future.result()
                if df is not None:
                    results[code] = df

        return results

    def get_multi_financial(
        self, stock_codes: list[str]
    ) -> dict[str, FinancialSnapshot]:
        """批量获取财务快照"""
        results = {}
        for code in stock_codes:
            snapshot = self.get_financial_snapshot(code)
            if snapshot is not None:
                results[code] = snapshot
        return results
