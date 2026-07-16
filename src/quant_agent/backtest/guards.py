"""回测假设守卫 (Backtest Assumption Guards) — P1.4

回测结果只有在「假设透明、且输入满足基本市场规则」时才可信。本模块对回测
输入做一组检查，产出 :class:`BacktestAssumptionReport`，覆盖：

- 交易日历：数据中是否混入非交易日（周末）行情
- 停牌：是否存在零成交量行（当日不可交易）
- 涨跌停：当日价格触及 ±10% 时，对应方向无法成交
- 复权 / 分红：复权状态是否被显式声明（影响收益率口径）
- 流动性：成交量过低时滑点/无法成交风险
- 前视偏差 (look-ahead)：信号是否使用了同 bar 收盘价既决策又成交
  （基线引擎默认如此，属已知前视假设，需显式声明）

默认 ``strict=False``：仅记录警告，不中断回测。当 ``strict=True`` 时，
对「硬违规」（非交易日 / 停牌数据参与回测 / 未声明复权）抛出
:class:`BacktestAssumptionViolation`。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# A 股涨跌停阈值（主板 ±10%；创业板/科创板 ±20% 此处统一按 10% 近似）
LIMIT_UP_RATIO = 0.095
LIMIT_DOWN_RATIO = -0.095
# 流动性下限（成交量，手）：低于此视为低流动性警告
LOW_LIQUIDITY_VOLUME = 1_000


class BacktestAssumptionViolation(Exception):
    """回测假设硬违规 — strict 模式下由 :func:`check_assumptions` 抛出。"""


@dataclass
class BacktestAssumptionReport:
    """回测假设检查报告。

    所有检测项均为可序列化字段，便于写入回测清单 / 报告。
    """

    trading_days: int = 0
    non_trading_days: list[str] = field(default_factory=list)
    suspended_dates: list[str] = field(default_factory=list)
    limit_up_dates: list[str] = field(default_factory=list)
    limit_down_dates: list[str] = field(default_factory=list)
    low_liquidity_dates: list[str] = field(default_factory=list)
    adjust_status: str | None = None
    dividend_adjusted: bool = False
    look_ahead_same_bar: bool = True  # 基线引擎默认同 bar 决策+成交（前视假设）
    hard_violations: list[str] = field(default_factory=list)

    def warnings(self) -> list[str]:
        """汇总为可读的警告列表。"""
        w: list[str] = []
        if self.non_trading_days:
            w.append(f"数据含非交易日(周末): {len(self.non_trading_days)} 天，可能污染收益")
        if self.suspended_dates:
            w.append(f"存在停牌日(零成交量): {len(self.suspended_dates)} 天，当日不可交易")
        if self.limit_up_dates:
            w.append(f"触及涨停: {len(self.limit_up_dates)} 天，买入信号当日无法成交")
        if self.limit_down_dates:
            w.append(f"触及跌停: {len(self.limit_down_dates)} 天，卖出信号当日无法成交")
        if self.low_liquidity_dates:
            w.append(f"低流动性日: {len(self.low_liquidity_dates)} 天，滑点/无法成交风险")
        if self.adjust_status is None:
            w.append("未声明复权状态，收益率口径可能失真")
        elif not self.dividend_adjusted:
            w.append(f"复权状态={self.adjust_status}，但未标记分红调整，价格含除权缺口")
        if self.look_ahead_same_bar:
            w.append(
                "信号使用同 bar 收盘价既决策又成交（前视偏差假设）；"
                "如需无前视，请使用独立信号生成时点"
            )
        return w

    def to_dict(self) -> dict[str, Any]:
        return {
            "trading_days": self.trading_days,
            "non_trading_days": self.non_trading_days,
            "suspended_dates": self.suspended_dates,
            "limit_up_dates": self.limit_up_dates,
            "limit_down_dates": self.limit_down_dates,
            "low_liquidity_dates": self.low_liquidity_dates,
            "adjust_status": self.adjust_status,
            "dividend_adjusted": self.dividend_adjusted,
            "look_ahead_same_bar": self.look_ahead_same_bar,
            "hard_violations": self.hard_violations,
            "warnings": self.warnings(),
        }


def _parse_dates(price_data: pd.DataFrame) -> pd.Series:
    """把 date 列解析为 datetime；解析失败返回 NaT。"""
    if "date" not in price_data.columns:
        return pd.Series([pd.NaT] * len(price_data))
    return pd.to_datetime(price_data["date"], errors="coerce")


def check_assumptions(
    price_data: pd.DataFrame,
    signals: pd.Series | None = None,
    adjust: str | None = "qfq",
    look_ahead_same_bar: bool = True,
    strict: bool = False,
) -> BacktestAssumptionReport:
    """检查回测输入的市场假设。

    Args:
        price_data: OHLCV DataFrame（需含 date, close, volume；high/low 可选）
        signals: 信号序列（1/-1/0），可选，仅用于辅助说明
        adjust: 复权状态声明（qfq/hfq/raw/None）
        look_ahead_same_bar: 是否使用同 bar 收盘价决策+成交（前视假设）
        strict: 硬违规时抛出 :class:`BacktestAssumptionViolation`

    Returns:
        BacktestAssumptionReport
    """
    report = BacktestAssumptionReport(adjust_status=adjust, look_ahead_same_bar=look_ahead_same_bar)
    if price_data is None or price_data.empty:
        report.hard_violations.append("空价格数据")
        if strict:
            raise BacktestAssumptionViolation("价格数据为空，无法回测")
        return report

    report.trading_days = len(price_data)
    dates = _parse_dates(price_data)
    date_strs = dates.dt.strftime("%Y-%m-%d").tolist()

    # 1. 交易日历 — 周末
    weekend_mask = dates.dt.dayofweek.isin([5, 6])
    for i in weekend_mask[weekend_mask].index:
        report.non_trading_days.append(str(date_strs[i]))

    # 2. 停牌 — 零成交量
    if "volume" in price_data.columns:
        vol = pd.to_numeric(price_data["volume"], errors="coerce").fillna(0)
        for i in vol[vol <= 0].index:
            report.suspended_dates.append(str(date_strs[i]))
        # 5. 流动性
        for i in vol[(vol > 0) & (vol < LOW_LIQUIDITY_VOLUME)].index:
            report.low_liquidity_dates.append(str(date_strs[i]))

    # 3. 涨跌停（需 high/low 与上一日收盘）
    if {"high", "low", "close"}.issubset(price_data.columns):
        close = pd.to_numeric(price_data["close"], errors="coerce")
        high = pd.to_numeric(price_data["high"], errors="coerce")
        low = pd.to_numeric(price_data["low"], errors="coerce")
        prev_close = close.shift(1)
        for i in range(1, len(price_data)):
            pc = prev_close.iloc[i]
            if pd.isna(pc) or pc <= 0:
                continue
            if (high.iloc[i] - pc) / pc >= LIMIT_UP_RATIO:
                report.limit_up_dates.append(str(date_strs[i]))
            if (low.iloc[i] - pc) / pc <= LIMIT_DOWN_RATIO:
                report.limit_down_dates.append(str(date_strs[i]))

    # 4. 复权 / 分红
    report.dividend_adjusted = adjust in ("qfq", "hfq")

    # 硬违规判定
    if report.non_trading_days:
        report.hard_violations.append(f"含 {len(report.non_trading_days)} 个非交易日")
    if report.suspended_dates:
        report.hard_violations.append(f"含 {len(report.suspended_dates)} 个停牌日")
    if adjust is None:
        report.hard_violations.append("未声明复权状态")

    if strict and report.hard_violations:
        raise BacktestAssumptionViolation("; ".join(report.hard_violations))

    return report
