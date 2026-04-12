"""数据服务 — 统一入口，多数据源降级 + 缓存 + 修复 + 离线"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import pandas as pd

from ..config import Settings, get_settings
from .sources.base import DataSource, FinancialSnapshot
from .sources.tushare import TushareSource
from .sources.akshare import AkshareSource
from .sources.baostock import BaoStockSource
from .normalizer import normalize_price_data
from .validator import validate_price_data, clean_price_data, repair_price_data
from .validators import validate_stock_code
from .store import DataStore

logger = logging.getLogger(__name__)


class DataService:
    """数据服务统一入口

    数据获取优先级：缓存 → 本地存储 → Tushare → efinance → AkShare → BaoStock
    支持：离线模式、数据修复、财务多源合并
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.store = DataStore(self.settings.parquet_dir)

        # 初始化数据源（按优先级顺序）
        self._sources: list[DataSource] = []

        # Tushare（财务数据主力）
        try:
            self._tushare = TushareSource(token=self.settings.tushare_token)
            if self._tushare.available:
                self._sources.append(self._tushare)
                logger.info("Tushare data source ready")
        except Exception as e:
            logger.warning(f"Tushare init failed: {e}")

        # efinance（免费稳定，东方财富 API）
        try:
            from .sources.efinance import EfinanceSource
            self._efinance = EfinanceSource()
            if self._efinance.available:
                self._sources.append(self._efinance)
                logger.info("efinance data source ready")
        except Exception as e:
            logger.warning(f"efinance init failed: {e}")

        # AkShare（行情数据主力，免费）
        try:
            self._akshare = AkshareSource(timeout=self.settings.akshare_timeout)
            if self._akshare.available:
                self._sources.append(self._akshare)
                logger.info("AkShare data source ready")
        except Exception as e:
            logger.warning(f"AkShare init failed: {e}")

        # BaoStock（免费行情，降级备选）
        try:
            self._baostock = BaoStockSource()
            if self._baostock.available:
                self._sources.append(self._baostock)
                logger.info("BaoStock data source ready")
        except Exception as e:
            logger.warning(f"BaoStock init failed: {e}")

    @property
    def tushare(self) -> Optional[TushareSource]:
        return self._tushare if hasattr(self, "_tushare") else None

    @property
    def efinance(self):
        return self._efinance if hasattr(self, "_efinance") else None

    @property
    def akshare(self) -> Optional[AkshareSource]:
        return self._akshare if hasattr(self, "_akshare") else None

    @property
    def baostock(self) -> Optional[BaoStockSource]:
        return self._baostock if hasattr(self, "_baostock") else None

    # ── 行情数据 ──

    def _cache_max_age_hours(self) -> float:
        """Get cache TTL from settings (convert seconds → hours)."""
        return getattr(self.settings, "data_cache_ttl", 14400) / 3600

    def get_price_data(
        self,
        stock_code: str,
        days: int = 250,
        use_cache: bool = True,
        clean: bool = True,
    ) -> Optional[pd.DataFrame]:
        """获取标准化行情数据

        优先级: 缓存/存储 → 各数据源依次尝试（含数据修复）
        """
        stock_code = validate_stock_code(stock_code)

        # 离线模式：只读缓存
        if getattr(self.settings, "offline_mode", False):
            df = self.store.load_price(stock_code)
            if df is not None:
                logger.info(f"Offline mode: using cached data for {stock_code}")
                return normalize_price_data(df)
            logger.warning(f"Offline mode: no cached data for {stock_code}")
            return None

        # 1. 尝试缓存
        max_age = self._cache_max_age_hours()
        if use_cache and self.store.is_fresh(stock_code, max_age_hours=max_age):
            df = self.store.load_price(stock_code)
            if df is not None and len(df) >= days * 0.8:
                logger.info(f"Cache hit: {stock_code} ({len(df)} rows)")
                return normalize_price_data(df)

        # 2. 依次尝试数据源（含数据修复）
        for source in self._sources:
            df = source.get_price_data(stock_code, days)
            if df is not None and not df.empty:
                df = normalize_price_data(df)

                # 校验
                report = validate_price_data(df)
                if not report.is_valid:
                    # 尝试修复再校验
                    logger.warning(
                        f"Validation failed ({source.name}): {report.errors}, "
                        f"attempting repair"
                    )
                    repaired = repair_price_data(df)
                    if repaired is not None and not repaired.empty:
                        report2 = validate_price_data(repaired)
                        if report2.is_valid:
                            logger.info(f"Data repaired for {stock_code} from {source.name}")
                            df = repaired
                        else:
                            logger.warning(f"Repair failed for {stock_code}: {report2.errors}")
                            continue
                    else:
                        continue

                # 清洗
                if clean:
                    df = clean_price_data(df)

                # 持久化
                self.store.save_price(stock_code, df, source=source.name)
                return df

        logger.error(f"All sources failed: {stock_code}")
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
        """获取财务快照 — 多源降级 + 合并。

        降级链: Tushare → efinance → AkShare → 本地缓存。
        多源合并: 如果单个源数据不完整，尝试从多个源填补空字段。

        Args:
            stock_code: 股票代码
            max_age_days: 缓存财务数据最大允许天数（默认365天）
        """
        stock_code = validate_stock_code(stock_code)

        # 离线模式：只读缓存
        if getattr(self.settings, "offline_mode", False):
            return self._load_cached_financial(stock_code, max_age_days)

        # 1. 遍历所有支持 get_financial_snapshot 的数据源
        snapshots: dict[str, FinancialSnapshot] = {}
        for source in self._sources:
            get_fn = getattr(source, "get_financial_snapshot", None)
            if get_fn is None:
                continue
            try:
                snapshot = get_fn(stock_code)
                if snapshot is not None:
                    snapshots[source.name] = snapshot
            except Exception as e:
                logger.warning(f"Financial snapshot failed ({source.name}): {e}")

        # 2. 如果获得完整快照，直接返回
        for name, snap in snapshots.items():
            report = snap.validate()
            if not report.missing_required:
                self.store.save_financial(
                    stock_code, snap.to_dict(), source=name
                )
                return snap

        # 3. 多源合并（填补空字段）
        if snapshots:
            merged_data: dict[str, Any] = {}
            for name, snap in snapshots.items():
                for key in FinancialSnapshot.SCHEMA:
                    val = snap.get(key)
                    if val is not None and key not in merged_data:
                        merged_data[key] = val

            if merged_data:
                merged = FinancialSnapshot(stock_code, merged_data)
                logger.info(
                    f"Merged financial data from {list(snapshots.keys())} "
                    f"for {stock_code}"
                )
                self.store.save_financial(
                    stock_code, merged.to_dict(), source="merged"
                )
                return merged

        # 4. 本地缓存降级
        return self._load_cached_financial(stock_code, max_age_days)

    def _load_cached_financial(
        self, stock_code: str, max_age_days: int = 365
    ) -> Optional[FinancialSnapshot]:
        """Load financial data from local parquet cache."""
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
                            f"Cached financial data expired ({age_days}d > {max_age_days}d): "
                            f"{stock_code}"
                        )
                        return None
                except (ValueError, TypeError):
                    pass
            logger.info(f"Using cached financial data: {stock_code}")
            return FinancialSnapshot(stock_code, data)
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

    # ── 新闻数据 ──

    def get_news(self, stock_code: str, count: int = 20) -> Optional[pd.DataFrame]:
        """获取个股新闻

        Args:
            stock_code: 股票代码
            count: 获取新闻条数

        Returns:
            DataFrame with news items or None
        """
        stock_code = validate_stock_code(stock_code)
        for source in self._sources:
            fetch_fn = getattr(source, "get_news", None)
            if fetch_fn is None:
                continue
            try:
                df = fetch_fn(stock_code, count)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"新闻获取失败 ({source.name}): {e}")
        return None
