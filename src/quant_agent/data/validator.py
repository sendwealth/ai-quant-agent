"""数据质量校验 — null检测、停牌标记、异常值过滤"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """数据校验报告"""
    is_valid: bool = True
    total_rows: int = 0
    null_count: int = 0
    null_pct: float = 0.0
    stale_rows: int = 0  # 成交量为0（停牌）
    outlier_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        status = "✅ PASS" if self.is_valid else "❌ FAIL"
        return (
            f"{status} | rows={self.total_rows} nulls={self.null_count}({self.null_pct:.1%}) "
            f"stale={self.stale_rows} outliers={self.outlier_rows}"
        )


def validate_price_data(
    df: pd.DataFrame,
    max_null_pct: float = 0.05,
    max_daily_return: float = 0.22,  # A股涨跌停 20%（注册制）/ 10%
) -> ValidationReport:
    """校验行情数据质量

    Args:
        df: 标准化后的行情 DataFrame
        max_null_pct: 最大允许空值比例
        max_daily_return: 最大单日收益率（超过视为异常）
    """
    report = ValidationReport(total_rows=len(df))

    if df.empty:
        report.is_valid = False
        report.errors.append("DataFrame 为空")
        return report

    # 1. 空值检测
    required = ["close", "volume"]
    null_mask = df[required].isna().any(axis=1)
    report.null_count = null_mask.sum()
    report.null_pct = report.null_count / report.total_rows

    if report.null_pct > max_null_pct:
        report.is_valid = False
        report.errors.append(f"空值比例 {report.null_pct:.1%} 超过阈值 {max_null_pct:.1%}")

    # 2. 停牌检测（成交量=0）
    if "volume" in df.columns:
        stale_mask = df["volume"] == 0
        report.stale_rows = stale_mask.sum()
        if report.stale_rows > report.total_rows * 0.5:
            report.warnings.append(
                f"停牌天数占比 {(report.stale_rows/report.total_rows):.1%}，数据可能不完整"
            )

    # 3. 异常值检测（涨跌幅）
    if "close" in df.columns and len(df) > 1:
        returns = df["close"].pct_change().abs()
        outlier_mask = returns > max_daily_return
        # 排除停牌日（成交量=0）的异常
        if "volume" in df.columns:
            outlier_mask = outlier_mask & (df["volume"] > 0)
        report.outlier_rows = outlier_mask.sum()
        if report.outlier_rows > 0:
            report.warnings.append(f"检测到 {report.outlier_rows} 个异常涨跌幅（>{max_daily_return:.0%}）")

    # 4. 价格合理性
    if "close" in df.columns:
        if (df["close"] <= 0).any():
            report.is_valid = False
            report.errors.append("存在非正收盘价")

    return report


def clean_price_data(
    df: pd.DataFrame,
    remove_stale: bool = True,
    remove_outliers: bool = False,
    max_daily_return: float = 0.22,
) -> pd.DataFrame:
    """清洗行情数据

    Args:
        df: 标准化后的行情 DataFrame
        remove_stale: 是否移除停牌日（成交量=0）
        remove_outliers: 是否移除异常涨跌幅
        max_daily_return: 异常阈值
    """
    result = df.copy()

    if remove_stale and "volume" in result.columns:
        before = len(result)
        result = result[result["volume"] > 0]
        removed = before - len(result)
        if removed > 0:
            logger.info(f"移除 {removed} 个停牌日")

    if remove_outliers and len(result) > 1:
        returns = result["close"].pct_change().abs()
        mask = returns <= max_daily_return
        # 保留第一行（无前一日数据无法计算收益率）
        mask.iloc[0] = True
        before = len(result)
        result = result[mask]
        removed = before - len(result)
        if removed > 0:
            logger.info(f"移除 {removed} 个异常值")

    return result.reset_index(drop=True)


def repair_price_data(
    df: pd.DataFrame,
    max_null_pct: float = 0.05,
) -> Optional[pd.DataFrame]:
    """Attempt to repair data quality issues in price data.

    Repair strategies:
    1. Forward-fill then back-fill NaN values in price columns (OHLC)
    2. Linear interpolation for volume NaN values
    3. Drop rows where close is still NaN after repair
    4. Drop rows with non-positive close prices

    Returns None if the repaired DataFrame is empty.
    """
    if df is None or df.empty:
        return None

    result = df.copy()

    # 1. Forward-fill + back-fill price columns
    price_cols = [c for c in ["open", "high", "low", "close"] if c in result.columns]
    if price_cols:
        result[price_cols] = result[price_cols].ffill().bfill()

    # 2. Interpolate volume (linear)
    if "volume" in result.columns:
        result["volume"] = result["volume"].interpolate(method="linear")
        # Fill remaining NaN with 0 (suspended trading)
        result["volume"] = result["volume"].fillna(0)

    # 3. Drop rows where close is still NaN
    if "close" in result.columns:
        result = result.dropna(subset=["close"])

    # 4. Drop rows with non-positive close prices
    if "close" in result.columns:
        result = result[result["close"] > 0]

    result = result.reset_index(drop=True)

    if result.empty:
        return None

    # Log repair stats
    orig_nulls = df.isna().sum().sum()
    new_nulls = result.isna().sum().sum()
    if orig_nulls > 0:
        logger.info(
            f"Data repair: {orig_nulls} nulls → {new_nulls}, "
            f"rows {len(df)} → {len(result)}"
        )

    return result
