"""数据源抽象接口 — 所有数据源的基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

import pandas as pd


class StatementType(str, Enum):
    """财务报表类型"""
    INCOME = "income"           # 利润表
    BALANCE = "balance"         # 资产负债表
    CASHFLOW = "cashflow"       # 现金流量表
    INDICATORS = "indicators"   # 财务指标


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
        "roe":             ((float, int), True,  "净资产收益率"),
        "gross_margin":    ((float, int), True,  "毛利率"),
        "net_margin":      ((float, int), True,  "净利率"),
        # 偿债能力
        "debt_ratio":      ((float, int), True,  "资产负债率"),
        "current_ratio":   ((float, int), False, "流动比率"),
        # 成长性
        "revenue_growth":  ((float, int), True,  "营收增长率"),
        "profit_growth":   ((float, int), True,  "净利润增长率"),
        # 估值
        "pe_ttm":          ((float, int), False, "市盈率 TTM"),
        "pb":              ((float, int), False, "市净率"),
        "ps_ttm":          ((float, int), False, "市销率 TTM"),
        # 市值 / 价格
        "total_mv":        ((float, int), False, "总市值（万元）"),
        "price":           ((float, int), False, "最新价格"),
        # 元数据
        "report_date":     ((str,),       False, "报告期"),
        # 审计用（Tushare 交叉验证计算值）
        "roe_calc":        ((float, int), False, "ROE 计算值（审计用）"),
    }

    def __init__(self, stock_code: str, data: dict[str, Any]):
        self.stock_code = stock_code
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
                f"FinancialSnapshot({stock_code}): schema violation:\n"
                + "\n".join(errors)
            )
        self._data = data

    @staticmethod
    def _type_names(types: tuple[type, ...]) -> str:
        return "/".join(t.__name__ for t in types)

    # ── 属性访问器 ───────────────────────────────────────────────

    @property
    def roe(self) -> Optional[float]:
        """净资产收益率（真实计算）"""
        return self._data.get("roe")

    @property
    def gross_margin(self) -> Optional[float]:
        """毛利率"""
        return self._data.get("gross_margin")

    @property
    def net_margin(self) -> Optional[float]:
        """净利率"""
        return self._data.get("net_margin")

    @property
    def debt_ratio(self) -> Optional[float]:
        """资产负债率"""
        return self._data.get("debt_ratio")

    @property
    def current_ratio(self) -> Optional[float]:
        """流动比率"""
        return self._data.get("current_ratio")

    @property
    def pe_ttm(self) -> Optional[float]:
        """市盈率 TTM"""
        return self._data.get("pe_ttm")

    @property
    def pb(self) -> Optional[float]:
        """市净率"""
        return self._data.get("pb")

    @property
    def revenue_growth(self) -> Optional[float]:
        """营收增长率"""
        return self._data.get("revenue_growth")

    @property
    def profit_growth(self) -> Optional[float]:
        """净利润增长率"""
        return self._data.get("profit_growth")

    # ── Dict-like 访问 ──────────────────────────────────────────

    def __getitem__(self, key: str) -> Any:
        return self._data.get(key)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {**self._data, "stock_code": self.stock_code}

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
    ) -> Optional[pd.DataFrame]:
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
    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格"""
        ...

    def get_financial_statements(
        self, stock_code: str, statement_type: StatementType, periods: int = 4
    ) -> Optional[pd.DataFrame]:
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

    def get_financial_snapshot(self, stock_code: str) -> Optional[FinancialSnapshot]:
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
