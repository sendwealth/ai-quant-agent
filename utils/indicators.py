"""
技术指标计算模块
提供常用的技术分析指标
"""

import numpy as np
import pandas as pd
from typing import Tuple


def sma(data: pd.Series, period: int) -> pd.Series:
    """
    简单移动平均线 (Simple Moving Average)

    Args:
        data: 价格序列
        period: 周期

    Returns:
        SMA序列
    """
    return data.rolling(window=period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    """
    指数移动平均线 (Exponential Moving Average)

    Args:
        data: 价格序列
        period: 周期

    Returns:
        EMA序列
    """
    return data.ewm(span=period, adjust=False).mean()


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    相对强弱指标 (Relative Strength Index)

    Args:
        data: 价格序列
        period: 周期，默认14

    Returns:
        RSI序列
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    MACD指标 (Moving Average Convergence Divergence)

    Args:
        data: 价格序列
        fast: 快线周期，默认12
        slow: 慢线周期，默认26
        signal: 信号线周期，默认9

    Returns:
        (MACD线, 信号线, 柱状图)
    """
    ema_fast = ema(data, fast)
    ema_slow = ema(data, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    布林带 (Bollinger Bands)

    Args:
        data: 价格序列
        period: 周期，默认20
        std_dev: 标准差倍数，默认2

    Returns:
        (上轨, 中轨, 下轨)
    """
    middle_band = sma(data, period)
    std = data.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)

    return upper_band, middle_band, lower_band


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    平均真实波幅 (Average True Range)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期，默认14

    Returns:
        ATR序列
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    平均方向指标 (Average Directional Index)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期，默认14

    Returns:
        ADX序列
    """
    # 计算+DM和-DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    plus_dm[(plus_dm - minus_dm) < 0] = 0
    minus_dm[(minus_dm - plus_dm) < 0] = 0

    # 计算TR
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # 平滑处理
    atr_val = atr(high, low, close, period)
    plus_dm_smooth = plus_dm.rolling(window=period).mean()
    minus_dm_smooth = minus_dm.rolling(window=period).mean()

    # 计算+DI和-DI
    plus_di = 100 * (plus_dm_smooth / atr_val)
    minus_di = 100 * (minus_dm_smooth / atr_val)

    # 计算DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # 计算ADX
    adx_val = dx.rolling(window=period).mean()

    return adx_val


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    随机指标 (Stochastic Oscillator)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        k_period: %K周期，默认14
        d_period: %D周期，默认3

    Returns:
        (%K, %D)
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()

    return k_percent, d_percent


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """
    顺势指标 (Commodity Channel Index)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期，默认20

    Returns:
        CCI序列
    """
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

    cci_val = (tp - sma_tp) / (0.015 * mad)
    return cci_val


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    威廉指标 (Williams %R)

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期，默认14

    Returns:
        Williams %R序列
    """
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()

    williams = -100 * (highest_high - close) / (highest_high - lowest_low)
    return williams
