#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的技术指标计算模块
Technical Indicators Module

所有技术指标集中在这里，避免重复实现
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def sma(data: pd.Series, period: int) -> pd.Series:
    """
    简单移动平均 (Simple Moving Average)

    Args:
        data: 价格数据
        period: 周期

    Returns:
        SMA序列
    """
    return data.rolling(window=period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    """
    指数移动平均 (Exponential Moving Average)

    Args:
        data: 价格数据
        period: 周期

    Returns:
        EMA序列
    """
    return data.ewm(span=period, adjust=False).mean()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    平均真实波幅 (Average True Range)

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期 (默认14)

    Returns:
        ATR序列
    """
    # 计算真实波幅
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 计算ATR — Wilders smoothing (consistent with ADX)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    相对强弱指标 (Relative Strength Index) — Wilders smoothing

    Uses exponential smoothing with alpha=1/period (Wilders method), which
    matches all major trading platforms (TradingView,通达信,同花顺).

    Args:
        close: 收盘价
        period: 周期 (默认14)

    Returns:
        RSI序列 (0-100)
    """
    delta = close.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    # Wilders smoothing: EMA with alpha = 1/period
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # Avoid 0/0 NaN when price is constant (avg_loss == 0)
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi_value = 100.0 - (100.0 / (1.0 + rs))
    # When both gain and loss are 0, RSI is undefined — fill with 50 (neutral)
    rsi_value = rsi_value.fillna(50.0)

    return rsi_value


def macd(
    close: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD指标 (Moving Average Convergence Divergence)

    Args:
        close: 收盘价
        fast_period: 快线周期 (默认12)
        slow_period: 慢线周期 (默认26)
        signal_period: 信号线周期 (默认9)

    Returns:
        (macd_line, signal_line, histogram)
    """
    ema_fast = ema(close, fast_period)
    ema_slow = ema(close, slow_period)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    close: pd.Series, period: int = 20, std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带 (Bollinger Bands)

    Args:
        close: 收盘价
        period: 周期 (默认20)
        std_dev: 标准差倍数 (默认2.0)

    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = sma(close, period)
    std = close.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> Tuple[pd.Series, pd.Series]:
    """
    随机指标 (Stochastic Oscillator)

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        k_period: K线周期 (默认14)
        d_period: D线周期 (默认3)

    Returns:
        (k_line, d_line)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    price_range = highest_high - lowest_low
    # Protect against division by zero when price does not move
    k_line = pd.Series(
        np.where(price_range > 0, 100 * (close - lowest_low) / price_range, 50.0),
        index=close.index,
    )
    d_line = k_line.rolling(window=d_period).mean()

    return k_line, d_line


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    威廉指标 (Williams %R)

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期 (默认14)

    Returns:
        Williams %R序列 (-100 to 0)
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    price_range = highest_high - lowest_low
    wr = pd.Series(
        np.where(price_range > 0, -100 * (highest_high - close) / price_range, -50.0),
        index=close.index,
    )

    return wr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    平均趋向指数 (Average Directional Index) — Wilders smoothing

    Uses exponential smoothing with alpha=1/period for +DI, -DI, and ADX,
    matching standard implementations on all major trading platforms.

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期 (默认14)

    Returns:
        ADX序列 (0-100)
    """
    # 计算+DM和-DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    # 计算TR
    tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(
        axis=1
    )

    # Wilders smoothing: EMA with alpha=1/period
    alpha = 1.0 / period
    atr_val = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_plus_dm = plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    smooth_minus_dm = minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    # 计算DI
    plus_di = 100 * (smooth_plus_dm / atr_val)
    minus_di = 100 * (smooth_minus_dm / atr_val)

    # 计算DX
    di_sum = plus_di + minus_di
    dx = pd.Series(
        np.where(di_sum > 0, 100 * abs(plus_di - minus_di) / di_sum, 0.0),
        index=close.index,
    )

    # 计算ADX — Wilders smoothing on DX
    adx_value = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()

    return adx_value


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    能量潮指标 (On-Balance Volume)

    Args:
        close: 收盘价
        volume: 成交量

    Returns:
        OBV序列
    """
    direction = np.sign(close.diff())
    obv_value = (direction * volume).fillna(volume.iloc[0]).cumsum()

    return obv_value


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """
    动量指标 (Momentum)

    Args:
        close: 收盘价
        period: 周期 (默认10)

    Returns:
        动量序列
    """
    return close - close.shift(period)


def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
    """
    变化率指标 (Rate of Change)

    Args:
        close: 收盘价
        period: 周期 (默认10)

    Returns:
        ROC序列 (百分比)
    """
    return ((close - close.shift(period)) / close.shift(period)) * 100


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    顺势指标 (Commodity Channel Index)

    Args:
        high: 最高价
        low: 最低价
        close: 收盘价
        period: 周期 (默认20)

    Returns:
        CCI序列
    """
    tp = (high + low + close) / 3
    sma_tp = sma(tp, period)
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci_value = (tp - sma_tp) / (0.015 * mad)

    return cci_value


# ========== 辅助函数 ==========


def detect_crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """
    检测交叉信号

    Args:
        series1: 序列1
        series2: 序列2

    Returns:
        交叉信号 (1: 上穿, -1: 下穿, 0: 无交叉)
    """
    prev_diff = series1.shift(1) - series2.shift(1)
    curr_diff = series1 - series2

    signals = pd.Series(0, index=series1.index)
    signals[(prev_diff <= 0) & (curr_diff > 0)] = 1  # 上穿
    signals[(prev_diff >= 0) & (curr_diff < 0)] = -1  # 下穿

    return signals


def normalize(data: pd.Series, min_val: float = 0, max_val: float = 1) -> pd.Series:
    """
    归一化数据到指定范围

    Args:
        data: 输入数据
        min_val: 最小值
        max_val: 最大值

    Returns:
        归一化后的数据
    """
    data_min = data.min()
    data_max = data.max()

    if data_max == data_min:
        # Constant series — return midpoint of target range
        return pd.Series((min_val + max_val) / 2, index=data.index)

    normalized = (data - data_min) / (data_max - data_min)
    return normalized * (max_val - min_val) + min_val


# ========== 导出所有指标 ==========
__all__ = [
    # 趋势指标
    "sma",
    "ema",
    "macd",
    "adx",
    "bollinger_bands",
    # 动量指标
    "rsi",
    "stochastic",
    "williams_r",
    "momentum",
    "rate_of_change",
    "cci",
    # 波动率指标
    "atr",
    # 成交量指标
    "obv",
    # 辅助函数
    "detect_crossover",
    "normalize",
]
