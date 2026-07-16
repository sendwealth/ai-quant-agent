"""数据源抽象接口 — 所有数据源的基类"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

# 数据谱系 schema 版本。报告/存储/回测 manifest 应记录此版本以便迁移。
PROVENANCE_SCHEMA_VERSION = "2.0"


@dataclass
class DataProvenance:
    """数据谱系 — 记录一份数据的来源、获取时间与可信度（可复现审计）。

    扩展后的 versioned schema（v2）新增：交易日、复权状态、缓存年龄、
    字段缺失、降级原因与数据哈希，使每个信号与回测结论都能回答
    「用了什么数据、何时得到、能否重跑」。

    所有新增字段均有默认值，向后兼容 v1（仅含前 5 个字段）的序列化。
    """

    source: str  # 数据源名称：tushare / efinance / akshare / baostock / cache / sample / merged
    identifier: str  # 数据标识：通常为股票代码，或 "CODE:financial" / "CODE:price"
    fetched_at: str  # 获取时间（ISO 8601）
    data_type: str  # price / financial
    confidence: str = "high"  # high / partial / low
    # ── v2 扩展字段 ──
    version: str = PROVENANCE_SCHEMA_VERSION
    trading_day: str | None = None  # 数据对应的交易日 (YYYY-MM-DD)
    adjust_status: str | None = None  # 复权状态：qfq / hfq / raw
    cache_age_seconds: float | None = None  # 缓存年龄（秒）；None 表示非缓存来源
    missing_fields: list[str] | None = None  # 缺失字段（部分财务合并 / 样例缺失时显著）
    degradation_reason: str | None = None  # 降级原因（如 sample_fallback / cache_only / merged）
    data_hash: str | None = None  # 数据指纹 (sha256 前 16 位)，用于复现与篡改检测

    @staticmethod
    def compute_hash(data: Any) -> str:
        """计算数据指纹 (sha256 前 16 位)，用于复现性与篡改检测。

        支持 pandas.DataFrame / dict / 其它可序列化对象；失败则退化为 str(data)。
        """
        try:
            if isinstance(data, pd.DataFrame):
                payload = data.to_json(orient="records", date_format="iso")
            elif isinstance(data, dict):
                payload = json.dumps(data, default=str, sort_keys=True, ensure_ascii=False)
            elif isinstance(data, (list, tuple)):
                payload = json.dumps(list(data), default=str, sort_keys=True, ensure_ascii=False)
            else:
                payload = str(data)
        except Exception:
            payload = str(data)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def is_degraded(self) -> bool:
        """是否处于降级 / 可疑状态（需在报告中显著警示）。"""
        return (
            self.confidence in ("low", "partial")
            or self.source in ("sample", "cache", "merged")
            or bool(self.degradation_reason)
            or bool(self.missing_fields)
        )

    def warning_reasons(self) -> list[str]:
        """返回该条谱系需要警示的原因列表（供报告渲染）。"""
        reasons: list[str] = []
        if self.source == "sample":
            reasons.append("使用内置演示样例（合成数据，非真实行情）")
        elif self.source == "cache":
            reasons.append("使用本地缓存（未命中实时数据源）")
        elif self.source == "merged":
            reasons.append("多源合并补全（部分字段来自不同源）")
        if self.confidence == "low":
            reasons.append("数据可信度为低")
        elif self.confidence == "partial":
            reasons.append("数据可信度为部分")
        if self.missing_fields:
            reasons.append(f"缺失字段: {', '.join(self.missing_fields)}")
        if self.degradation_reason:
            reasons.append(f"降级原因: {self.degradation_reason}")
        if self.cache_age_seconds is not None and self.cache_age_seconds > 86400 * 30:
            reasons.append(f"缓存较旧 ({self.cache_age_seconds / 86400:.0f} 天)")
        return reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "identifier": self.identifier,
            "fetched_at": self.fetched_at,
            "data_type": self.data_type,
            "confidence": self.confidence,
            "trading_day": self.trading_day,
            "adjust_status": self.adjust_status,
            "cache_age_seconds": self.cache_age_seconds,
            "missing_fields": self.missing_fields,
            "degradation_reason": self.degradation_reason,
            "data_hash": self.data_hash,
        }


class StatementType(str, Enum):
    """财务报表类型"""

    INCOME = "income"  # 利润表
    BALANCE = "balance"  # 资产负债表
    CASHFLOW = "cashflow"  # 现金流量表
    INDICATORS = "indicators"  # 财务指标


class FinancialSnapshot:
    """财务快照 — 单只股票的核心财务指标

    Schema 验证:
      - __init__ 对已知键做类型检查（float/int/str/None）
      - 未知键原样保留不报错（支持扩展）
      - validate() 返回完整性报告，可检查哪些必填/可选键缺失
    """

    # ── Schema 定义 ──────────────────────────────────────────────
    # key -> (expected_types, required, description)
    # expected_types: 允许的 Python 类型（不含 None，None 由可选语义隐含）
    # required: True 表示 validate() 时视为必填
    SCHEMA: dict[str, tuple[tuple[type, ...], bool, str]] = {
        # 盈利能力
        "roe": ((float, int), True, "净资产收益率"),
        "gross_margin": ((float, int), True, "毛利率"),
        "net_margin": ((float, int), True, "净利率"),
        # 偿债能力
        "debt_ratio": ((float, int), True, "资产负债率"),
        "current_ratio": ((float, int), False, "流动比率"),
        # 成长性
        "revenue_growth": ((float, int), True, "营收增长率"),
        "profit_growth": ((float, int), True, "净利润增长率"),
        # 估值
        "pe_ttm": ((float, int), False, "市盈率 TTM"),
        "pb": ((float, int), False, "市净率"),
        "ps_ttm": ((float, int), False, "市销率 TTM"),
        # 市值 / 价格
        "total_mv": ((float, int), False, "总市值（万元）"),
        "price": ((float, int), False, "最新价格"),
        # 元数据
        "report_date": ((str,), False, "报告期"),
        # 审计用（Tushare 交叉验证计算值）
        "roe_calc": ((float, int), False, "ROE 计算值（审计用）"),
    }

    def __init__(
        self,
        stock_code: str,
        data: dict[str, Any],
        provenance: list[DataProvenance] | None = None,
    ):
        self.stock_code = stock_code
        self._provenance: list[DataProvenance] = list(provenance or [])
        # ── 类型校验：对 schema 中的已知键检查值类型 ──
        errors: list[str] = []
        for key, (expected_types, _required, _desc) in self.SCHEMA.items():
            if key not in data:
                continue  # 缺失键不报错（validate() 负责报告）
            value = data[key]
            if value is None:
                continue  # None 对所有键合法（表示暂无数据）
            if not isinstance(value, expected_types):
                errors.append(
                    f"  {key}: expected {self._type_names(expected_types)} or None, "
                    f"got {type(value).__name__} ({value!r})"
                )
        if errors:
            raise ValueError(
                f"FinancialSnapshot({stock_code}): schema violation:\n" + "\n".join(errors)
            )
        self._data = data

    @staticmethod
    def _type_names(types: tuple[type, ...]) -> str:
        return "/".join(t.__name__ for t in types)

    # ── 属性访问器 ───────────────────────────────────────────────

    @property
    def roe(self) -> float | None:
        """净资产收益率（真实计算）"""
        return self._data.get("roe")

    @property
    def gross_margin(self) -> float | None:
        """毛利率"""
        return self._data.get("gross_margin")

    @property
    def net_margin(self) -> float | None:
        """净利率"""
        return self._data.get("net_margin")

    @property
    def debt_ratio(self) -> float | None:
        """资产负债率"""
        return self._data.get("debt_ratio")

    @property
    def current_ratio(self) -> float | None:
        """流动比率"""
        return self._data.get("current_ratio")

    @property
    def pe_ttm(self) -> float | None:
        """市盈率 TTM"""
        return self._data.get("pe_ttm")

    @property
    def pb(self) -> float | None:
        """市净率"""
        return self._data.get("pb")

    @property
    def revenue_growth(self) -> float | None:
        """营收增长率"""
        return self._data.get("revenue_growth")

    @property
    def profit_growth(self) -> float | None:
        """净利润增长率"""
        return self._data.get("profit_growth")

    # ── Dict-like 访问 ──────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    @property
    def provenance(self) -> list[DataProvenance]:
        """数据谱系：本快照来自哪些数据源及获取时间。"""
        return self._provenance

    def add_provenance(self, prov: DataProvenance) -> None:
        """追加一条数据谱系记录。"""
        self._provenance.append(prov)

    def to_dict(self) -> dict[str, Any]:
        d = {**self._data, "stock_code": self.stock_code}
        if self._provenance:
            d["provenance"] = [p.to_dict() for p in self._provenance]
        return d

    # ── 验证 ────────────────────────────────────────────────────

    def validate(self) -> _ValidationReport:
        """检查数据完整性：哪些必填/可选键缺失、是否存在未知键。

        Returns:
            _ValidationReport 包含 missing_required / missing_optional / unknown_keys。
        """
        missing_required: list[str] = []
        missing_optional: list[str] = []
        unknown_keys: list[str] = []

        schema_keys = set(self.SCHEMA)
        for key, (_types, required, _desc) in self.SCHEMA.items():
            if key not in self._data or self._data[key] is None:
                if required:
                    missing_required.append(key)
                else:
                    missing_optional.append(key)

        for key in self._data:
            if key not in schema_keys:
                unknown_keys.append(key)

        return _ValidationReport(
            stock_code=self.stock_code,
            missing_required=missing_required,
            missing_optional=missing_optional,
            unknown_keys=unknown_keys,
        )

    def __repr__(self) -> str:
        return f"FinancialSnapshot({self.stock_code}, ROE={self.roe})"


class _ValidationReport:
    """FinancialSnapshot.validate() 的返回值"""

    __slots__ = ("stock_code", "missing_required", "missing_optional", "unknown_keys")

    def __init__(
        self,
        stock_code: str,
        missing_required: list[str],
        missing_optional: list[str],
        unknown_keys: list[str],
    ):
        self.stock_code = stock_code
        self.missing_required = missing_required
        self.missing_optional = missing_optional
        self.unknown_keys = unknown_keys

    @property
    def is_complete(self) -> bool:
        """所有必填键都存在且非 None"""
        return len(self.missing_required) == 0

    def summary(self) -> str:
        lines = [f"FinancialSnapshot({self.stock_code}) validation:"]
        if self.missing_required:
            lines.append(f"  MISSING REQUIRED: {', '.join(self.missing_required)}")
        if self.missing_optional:
            lines.append(f"  missing optional:  {', '.join(self.missing_optional)}")
        if self.unknown_keys:
            lines.append(f"  unknown keys:      {', '.join(self.unknown_keys)}")
        if not self.missing_required and not self.missing_optional and not self.unknown_keys:
            lines.append("  OK - all schema keys present")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class DataSource(ABC):
    """数据源抽象基类"""

    @abstractmethod
    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> pd.DataFrame | None:
        """获取历史行情数据

        Args:
            stock_code: 股票代码（如 300750）
            days: 回溯天数
            adjust: 复权类型 qfq=前复权 hfq=后复权 None=不复权

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        ...

    @abstractmethod
    def get_realtime_price(self, stock_code: str) -> float | None:
        """获取实时价格"""
        ...

    def get_financial_statements(
        self, stock_code: str, statement_type: StatementType, periods: int = 4
    ) -> pd.DataFrame | None:
        """获取财务报表

        Args:
            stock_code: 股票代码
            statement_type: 报表类型
            periods: 最近 N 期

        Returns:
            标准化的财务报表 DataFrame
        """
        # 默认不支持，子类可选实现
        return None

    def get_financial_snapshot(self, stock_code: str) -> FinancialSnapshot | None:
        """获取财务快照（核心指标汇总）

        默认实现：分别获取各报表后计算。子类可覆写以优化。
        """
        return None

    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        ...

    @property
    def available(self) -> bool:
        """数据源是否可用"""
        return True
