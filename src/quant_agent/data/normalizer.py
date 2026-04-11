"""数据标准化 — 列名统一、复权、异常值处理"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 标准列名映射（中文 → 英文）
COLUMN_MAP = {
    "日期": "date", "开盘": "open", "收盘": "close",
    "最高": "high", "最低": "low", "成交量": "volume",
    "成交额": "amount", "涨跌幅": "pct_change", "涨跌额": "change",
    "换手率": "turnover", "振幅": "amplitude",
    "trade_date": "date", "vol": "volume", "pct_chg": "pct_change",
}

REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名为英文标准格式"""
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    return df


def normalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """标准化数据类型"""
    # 日期列
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 数值列
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def normalize_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """完整的行情数据标准化流程"""
    df = normalize_columns(df)
    df = normalize_dtypes(df)

    # 确保必要列存在
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"行情数据缺少必要列: {missing}. "
            f"现有列: {list(df.columns)}"
        )

    # 按日期排序
    df = df.sort_values("date").reset_index(drop=True)

    return df


def clip_outliers(
    series: pd.Series,
    lower_pct: float = 0.001,
    upper_pct: float = 0.999,
) -> pd.Series:
    """截断异常值（百分位法）"""
    lower = series.quantile(lower_pct)
    upper = series.quantile(upper_pct)
    return series.clip(lower, upper)


