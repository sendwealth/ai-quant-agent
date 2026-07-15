"""数据服务 — 统一入口，多数据源降级 + 缓存 + 修复 + 离线"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from .normalizer import normalize_price_data
from .sources.akshare import AkshareSource
from .sources.baostock import BaoStockSource
from .sources.base import DataProvenance, DataSource, FinancialSnapshot
from .sources.sample import SamplePriceSource
from .sources.tushare import TushareSource
from .store import DataStore
from .validator import clean_price_data, repair_price_data, validate_price_data
from .validators import validate_stock_code

logger = logging.getLogger(__name__)


class DataService:
    """数据服务统一入口

    数据获取优先级：缓存 → 本地存储 → Tushare → efinance → AkShare → BaoStock
    支持：离线模式、数据修复、财务多源合并
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.store = DataStore(self.settings.parquet_dir)
        # 数据谱系：stock_code -> 本次分析会话获取的数据来源记录（P3）
        self._lineage: dict[str, list[DataProvenance]] = {}

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

        # 内置样例源：仅读取内置的「真实」历史样例（data/samples），
        # 不参与健康检查的"真实源"判定，且绝不生成合成/模拟数据；
        # 若样例不存在则返回 None（表示没有数据）。
        self._sample = SamplePriceSource(self.settings)
        logger.info("Sample data source ready (real bundled data only; never synthetic)")

    @property
    def tushare(self) -> TushareSource | None:
        return self._tushare if hasattr(self, "_tushare") else None

    @property
    def efinance(self):
        return self._efinance if hasattr(self, "_efinance") else None

    @property
    def akshare(self) -> AkshareSource | None:
        return self._akshare if hasattr(self, "_akshare") else None

    @property
    def baostock(self) -> BaoStockSource | None:
        return self._baostock if hasattr(self, "_baostock") else None

    # ── 行情数据 ──

    def _cache_max_age_hours(self) -> float:
        """Get cache TTL from settings (convert seconds → hours)."""
        return getattr(self.settings, "data_cache_ttl", 14400) / 3600

    # ── 数据谱系 (P3) ────────────────────────────────────────────

    def _record_lineage(
        self,
        stock_code: str,
        source: str,
        data_type: str,
        confidence: str = "high",
    ) -> None:
        """记录一条数据谱系（来源 + 获取时间）。"""
        if not hasattr(self, "_lineage"):
            self._lineage = {}
        prov = DataProvenance(
            source=source,
            identifier=stock_code,
            fetched_at=datetime.now().isoformat(timespec="seconds"),
            data_type=data_type,
            confidence=confidence,
        )
        self._lineage.setdefault(stock_code, []).append(prov)

    def get_lineage(self, stock_code: str) -> list[DataProvenance]:
        """返回某股票本次会话获取的数据谱系记录（用于报告透明展示）。"""
        return list(getattr(self, "_lineage", {}).get(stock_code, []))

    def get_price_data(
        self,
        stock_code: str,
        days: int = 250,
        use_cache: bool = True,
        clean: bool = True,
    ) -> pd.DataFrame | None:
        """获取标准化行情数据

        优先级: 缓存/存储 → 各数据源依次尝试（含数据修复）
        """
        stock_code = validate_stock_code(stock_code)

        # 离线模式：只读缓存，无缓存则尝试内置真实样例（若存在），否则报告无数据
        if getattr(self.settings, "offline_mode", False):
            df = self.store.load_price(stock_code)
            if df is not None:
                logger.info(f"Offline mode: using cached data for {stock_code}")
                self._record_lineage(stock_code, "cache", "price")
                return normalize_price_data(df)
            try:
                demo = self._sample.get_price_data(stock_code, days)
                if demo is not None and not demo.empty:
                    logger.info(f"Offline mode: using sample fallback for {stock_code}")
                    self._record_lineage(stock_code, "sample", "price", confidence="low")
                    return normalize_price_data(demo)
            except Exception as e:
                logger.warning(f"Offline sample fallback failed: {e}")
            logger.warning(f"Offline mode: no data for {stock_code}")
            return None

        # 1. 尝试缓存
        max_age = self._cache_max_age_hours()
        if use_cache and self.store.is_fresh(stock_code, max_age_hours=int(max_age)):
            df = self.store.load_price(stock_code)
            if df is not None and len(df) >= days * 0.8:
                logger.info(f"Cache hit: {stock_code} ({len(df)} rows)")
                self._record_lineage(stock_code, "cache", "price")
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
                        f"Validation failed ({source.name}): {report.errors}, attempting repair"
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
                self._record_lineage(stock_code, source.name, "price")
                return df

        # 3. 内置真实样例兜底（仅当无任何可用真实数据源时，如各源初始化均失败）。
        #    样例源只读内置真实历史数据，绝不合成；样例不写入常规缓存，
        #    避免覆盖后续真实数据获取。在线模式下若真实源已初始化但网络失败，
        #    仍按原行为返回 None（"全源失败→None"），触发无数据上报。
        if not self._sources:
            try:
                bundled = self._sample.get_price_data(stock_code, days)
                if bundled is not None and not bundled.empty:
                    bundled = normalize_price_data(bundled)
                    report = validate_price_data(bundled)
                    if report.is_valid:
                        logger.info(f"Using real bundled sample for {stock_code}")
                        return bundled
            except Exception as e:
                logger.warning(f"Sample fallback failed for {stock_code}: {e}")

        # 4. 最终兜底：在线/离线联网全失败时，返回本地缓存（即使已过期），
        #    保证有本地历史数据时仍可分析，而不是彻底无数据。
        #    （受限网络环境下数据源常不可达，本地缓存即真实历史数据。）
        cached = self.store.load_price(stock_code)
        if cached is not None and not cached.empty:
            logger.warning(
                f"All live sources failed; falling back to local cache "
                f"(may be stale) for {stock_code}"
            )
            return normalize_price_data(cached)

        logger.error(f"All sources failed: {stock_code}")
        return None

    def get_realtime_price(self, stock_code: str) -> float | None:
        """获取实时价格"""
        stock_code = validate_stock_code(stock_code)
        for source in self._sources:
            price = source.get_realtime_price(stock_code)
            if price and price > 0:
                return price
        # 兜底：仅读取内置真实样例（无真实源或离线模式且样例存在时）
        if not self._sources or getattr(self.settings, "offline_mode", False):
            try:
                price = self._sample.get_realtime_price(stock_code)
                if price and price > 0:
                    return price
            except Exception:
                pass
        return None

    # ── 财务数据 ──

    def get_financial_snapshot(
        self, stock_code: str, max_age_days: int = 365
    ) -> FinancialSnapshot | None:
        """获取财务快照 — 多源降级 + 合并。

        降级链: Tushare → efinance → AkShare → 本地缓存。
        多源合并: 如果单个源数据不完整，尝试从多个源填补空字段。

        Args:
            stock_code: 股票代码
            max_age_days: 缓存财务数据最大允许天数（默认365天）
        """
        stock_code = validate_stock_code(stock_code)

        # 离线模式：只读缓存，无缓存则尝试内置真实样例财务（若存在），否则无数据
        if getattr(self.settings, "offline_mode", False):
            cached = self._load_cached_financial(stock_code, max_age_days)
            if cached is not None:
                return cached
            try:
                snap = self._sample.get_financial_snapshot(stock_code)
                if snap is not None:
                    snap.add_provenance(
                        DataProvenance(
                            source="sample",
                            identifier=stock_code,
                            fetched_at=datetime.now().isoformat(timespec="seconds"),
                            data_type="financial",
                            confidence="low",
                        )
                    )
                    self._record_lineage(stock_code, "sample", "financial", confidence="low")
                    return snap
            except Exception as e:
                logger.warning(f"Offline financial fallback failed: {e}")
            return None

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
                self.store.save_financial(stock_code, snap.to_dict(), source=name)
                snap.add_provenance(
                    DataProvenance(
                        source=name,
                        identifier=stock_code,
                        fetched_at=datetime.now().isoformat(timespec="seconds"),
                        data_type="financial",
                        confidence="high",
                    )
                )
                self._record_lineage(stock_code, name, "financial")
                return snap

        # 3. 多源合并（填补空字段）
        if snapshots:
            merged_data: dict[str, Any] = {}
            for snap in snapshots.values():
                for key in FinancialSnapshot.SCHEMA:
                    val = snap.get(key)
                    if val is not None and key not in merged_data:
                        merged_data[key] = val

            if merged_data:
                merged = FinancialSnapshot(stock_code, merged_data)
                for src_name in snapshots:
                    merged.add_provenance(
                        DataProvenance(
                            source=src_name,
                            identifier=stock_code,
                            fetched_at=datetime.now().isoformat(timespec="seconds"),
                            data_type="financial",
                            confidence="partial",
                        )
                    )
                self._record_lineage(stock_code, "merged", "financial", confidence="partial")
                logger.info(f"Merged financial data from {list(snapshots.keys())} for {stock_code}")
                self.store.save_financial(stock_code, merged.to_dict(), source="merged")
                return merged

        # 4. 本地缓存降级（含离线模式的样例兜底，见上方 offline 分支）
        return self._load_cached_financial(stock_code, max_age_days)

    def _load_cached_financial(
        self, stock_code: str, max_age_days: int = 365
    ) -> FinancialSnapshot | None:
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
            snap = FinancialSnapshot(stock_code, data)
            snap.add_provenance(
                DataProvenance(
                    source="cache",
                    identifier=stock_code,
                    fetched_at=datetime.now().isoformat(timespec="seconds"),
                    data_type="financial",
                    confidence="high",
                )
            )
            self._record_lineage(stock_code, "cache", "financial")
            return snap
        return None

    def get_financial_statements(
        self, stock_code: str, statement_type: str, periods: int = 4
    ) -> pd.DataFrame | None:
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
        max_workers: int | None = None,
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
            seq_results: dict[str, pd.DataFrame] = {}
            for code in stock_codes:
                df = self.get_price_data(code, days)
                if df is not None:
                    seq_results[code] = df
            return seq_results

        # Concurrent execution
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None]:
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

    def get_multi_financial(self, stock_codes: list[str]) -> dict[str, FinancialSnapshot]:
        """批量获取财务快照"""
        results = {}
        for code in stock_codes:
            snapshot = self.get_financial_snapshot(code)
            if snapshot is not None:
                results[code] = snapshot
        return results

    # ── 新闻数据 ──

    def get_news(self, stock_code: str, count: int = 20) -> pd.DataFrame | None:
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
