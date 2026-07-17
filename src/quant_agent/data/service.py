"""数据服务 — 统一入口，多数据源降级 + 缓存 + 修复 + 离线"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import pandas as pd

from ..config import Settings, get_settings
from .normalizer import normalize_price_data
from .smoke import smoke_report, smoke_test_source
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

        # WeStock（腾讯自选股，免费，npx CLI，无 token）
        try:
            from .sources.westock import WeStockSource

            self._westock = WeStockSource(enabled=self.settings.westock_enabled)
            if self._westock.available:
                self._sources.append(self._westock)
                logger.info("WeStock data source ready")
        except Exception as e:
            logger.warning(f"WeStock init failed: {e}")

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
    def westock(self):
        return self._westock if hasattr(self, "_westock") else None

    @property
    def akshare(self) -> AkshareSource | None:
        return self._akshare if hasattr(self, "_akshare") else None

    @property
    def baostock(self) -> BaoStockSource | None:
        return self._baostock if hasattr(self, "_baostock") else None

    # ------------------------------------------------------------------
    # P1.5 数据源冒烟测试
    # ------------------------------------------------------------------

    def smoke_test(self, stock_code: str = "600519", days: int = 5) -> dict[str, Any]:
        """对所有已构建数据源跑一次最小请求，返回聚合健康报告。

        见 :mod:`quant_agent.data.smoke`。仅读取、不写缓存，适合定时/
        CI 触发；任何单源异常都不会扩散（由 :func:`smoke_test_source`
        捕获并计入 ``failed``）。
        """
        results = [smoke_test_source(s, stock_code=stock_code, days=days) for s in self._sources]
        return smoke_report(results)

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
        data: Any | None = None,
        degradation_reason: str | None = None,
        adjust_status: str | None = None,
        trading_day: str | None = None,
        missing_fields: list[str] | None = None,
        cache_age_seconds: float | None = None,
    ) -> None:
        """记录一条数据谱系（来源 + 获取时间 + v2 扩展字段）。

        自动为 sample / cache / merged 来源设置降级原因，并可在传入 ``data``
        时计算数据哈希，供报告透明展示与复现校验。
        """
        if not hasattr(self, "_lineage"):
            self._lineage = {}
        if degradation_reason is None and source in ("sample", "cache", "merged"):
            degradation_reason = f"{source}_fallback"
        prov = DataProvenance(
            source=source,
            identifier=stock_code,
            fetched_at=datetime.now().isoformat(timespec="seconds"),
            data_type=data_type,
            confidence=confidence,
            degradation_reason=degradation_reason,
            adjust_status=adjust_status,
            trading_day=trading_day,
            missing_fields=missing_fields,
            cache_age_seconds=cache_age_seconds,
            data_hash=DataProvenance.compute_hash(data) if data is not None else None,
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
                self._record_lineage(stock_code, "cache", "price", data=df)
                return normalize_price_data(df)
            try:
                demo = self._sample.get_price_data(stock_code, days)
                if demo is not None and not demo.empty:
                    logger.info(f"Offline mode: using sample fallback for {stock_code}")
                    self._record_lineage(stock_code, "sample", "price", confidence="low", data=demo)
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
                self._record_lineage(stock_code, "cache", "price", data=df)
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
                            degradation_reason="sample_fallback",
                        )
                    )
                    self._record_lineage(
                        stock_code, "sample", "financial", confidence="low", data=snap.to_dict()
                    )
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
            cache_age_seconds: float | None = None
            if report_date is not None:
                try:
                    if isinstance(report_date, str):
                        rd = datetime.strptime(report_date[:10], "%Y-%m-%d")
                    else:
                        rd = pd.Timestamp(report_date).to_pydatetime()
                    cache_age_seconds = (datetime.now() - rd).total_seconds()
                    age_days = int(cache_age_seconds // 86400)
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
                    degradation_reason="cache_fallback",
                )
            )
            self._record_lineage(
                stock_code, "cache", "financial", data=data, cache_age_seconds=cache_age_seconds
            )
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
        per_call_timeout: float = 30.0,
        time_budget: float | None = 90.0,
    ) -> dict[str, pd.DataFrame]:
        """批量获取行情（支持并发）

        Args:
            stock_codes: 股票代码列表
            days: 获取天数
            max_workers: 并发线程数，None 时使用 settings.fetch_max_workers，
                         1 时退化为顺序执行
            per_call_timeout: 单只取数的超时上限（秒）。任一数据源卡死/
                长时间无响应时，跳过该股票以保证整批一定返回，避免批量
                取数（如智能选股对上百只股票）整体挂起。
            time_budget: 整批取数的时间预算（秒）。超过预算后停止收集、
                直接返回已拿到的结果，避免被限速的免费源（如 westock 60/min）
                或返回空却耗时的坏源拖垮整批，使智能选股能「尽快出结果」。
                None 表示不限制。

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

        # Concurrent execution.
        # 注意：不使用 \`with ThreadPoolExecutor\` 上下文管理器，因为它会在退出时
        # 等待「所有」工作线程结束。若某个数据源请求无超时（如个别免费源在
        # 受限网络下连接挂起），该线程会长时间甚至永久不退出，导致整批批量
        # 取数卡死（智能选股对上百只股票会永久无响应）。改为显式
        # \`shutdown(wait=False)\`：收集到已完成的结果后即返回，残留线程在其
        # 各自源的超时/重试结束后自行退出，不阻塞整批。
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(code: str) -> tuple[str, pd.DataFrame | None]:
            try:
                return code, self.get_price_data(code, days)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"并发获取 {code} 异常: {exc}")
                return code, None

        executor = ThreadPoolExecutor(max_workers=workers)
        start = time.monotonic()
        try:
            futures = {executor.submit(_fetch_one, code): code for code in stock_codes}
            for future in as_completed(futures):
                # 时间预算：超过预算即停止收集，返回已拿到的结果。
                # 关键修复：默认池有 179 只，Tushare 仅覆盖约 50 只，
                # 其余降级到被限速的免费源（westock 60/min）或返回空却
                # 耗时的坏源，整批可能永远跑不完。时间预算确保智能选股
                # 「尽快出结果」而非永久卡死。
                if time_budget is not None and (time.monotonic() - start) > time_budget:
                    logger.warning(
                        "批量取数达到时间预算 %.0fs，停止收集（已获取 %d/%d）",
                        time_budget,
                        len(results),
                        len(stock_codes),
                    )
                    break
                code = futures[future]
                try:
                    _, df = future.result(timeout=per_call_timeout)
                except TimeoutError:
                    # 单只取数超时（某数据源卡死/无响应）：跳过，保证整批返回
                    logger.warning(
                        f"并发获取 {code} 超时（>{per_call_timeout:.0f}s），跳过"
                    )
                    continue
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"并发获取 {code} 异常: {exc}")
                    continue
                if df is not None:
                    results[code] = df
        finally:
            # 不等待未完成线程（避免被卡死/限速的源拖垮整批）
            executor.shutdown(wait=False)

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
