"""Multi-dimension stock scoring -- technical, momentum, liquidity, fundamental.

Each scorer receives a price DataFrame (and optionally a FinancialSnapshot)
and returns a dict with the dimension score and a breakdown of sub-scores.

All thresholds come from ``_Thresh`` (screener.scoring.* branch).
All indicator calculations reuse ``quant_agent.strategy.indicators``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ..data.sources.base import FinancialSnapshot
from ..strategy.indicators import adx, ema, macd, rsi
from ..thresholds import _Thresh

logger = logging.getLogger(__name__)

# Minimum data length required for each scorer
_MIN_BARS_TECHNICAL = 60
_MIN_BARS_MOMENTUM = 60
_MIN_BARS_LIQUIDITY = 20


# ---------------------------------------------------------------------------
# Tier scoring helper
# ---------------------------------------------------------------------------

def _tier_score(value: float, tiers: list[list]) -> float:
    """Score *value* against a list of [threshold, points] tiers.

    Tiers are checked in order; the first tier whose threshold *value*
    exceeds wins.  Example::

        _tier_score(0.08, [[0.10, 15], [0.05, 10], [0.00, 5]]) → 10
    """
    for threshold, points in tiers:
        if value > threshold:
            return float(points)
    return 0.0


# ---------------------------------------------------------------------------
# Technical scorer (~40 pts max)
# ---------------------------------------------------------------------------

def score_technical(
    df: pd.DataFrame,
    thresh: _Thresh,
) -> dict[str, Any]:
    """Score technical indicators.

    Dimensions: EMA crossover, MACD histogram, RSI zone, ADX trend strength.
    """
    result: dict[str, Any] = {
        "technical": 0.0,
        "breakdown": {
            "ema_cross": 0.0,
            "ema_price_above": 0.0,
            "macd_positive": 0.0,
            "macd_rising": 0.0,
            "rsi_zone": 0.0,
            "adx_trend": 0.0,
        },
    }

    if len(df) < _MIN_BARS_TECHNICAL:
        return result

    close = df["close"]
    high = df["high"] if "high" in df.columns else close
    low = df["low"] if "low" in df.columns else close
    current = float(close.iloc[-1])

    # EMA crossover
    e20 = ema(close, 20)
    e60 = ema(close, 60)
    if len(e60) > 0 and not pd.isna(e60.iloc[-1]):
        if float(e20.iloc[-1]) > float(e60.iloc[-1]):
            pts = float(thresh.get("ema_cross_score", 12))
            result["breakdown"]["ema_cross"] = pts
            if current > float(e20.iloc[-1]):
                pts2 = float(thresh.get("ema_price_above_score", 5))
                result["breakdown"]["ema_price_above"] = pts2

    # MACD histogram
    _, _, hist = macd(close)
    if len(hist) >= 2 and not pd.isna(hist.iloc[-1]):
        hist_last = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2])
        if hist_last > 0:
            pts = float(thresh.get("macd_positive_score", 8))
            result["breakdown"]["macd_positive"] = pts
            if hist_last > hist_prev:
                pts2 = float(thresh.get("macd_rising_score", 5))
                result["breakdown"]["macd_rising"] = pts2

    # RSI zone
    rsi_series = rsi(close, 14)
    if len(rsi_series) > 0 and not pd.isna(rsi_series.iloc[-1]):
        rsi_val = float(rsi_series.iloc[-1])
        neutral_low = float(thresh.get("rsi_neutral_low", 35))
        neutral_high = float(thresh.get("rsi_neutral_high", 65))
        if neutral_low < rsi_val < neutral_high:
            result["breakdown"]["rsi_zone"] = float(thresh.get("rsi_neutral_score", 5))
        elif rsi_val <= neutral_low:
            result["breakdown"]["rsi_zone"] = float(thresh.get("rsi_oversold_score", 3))
        result["breakdown"]["rsi_value"] = round(rsi_val, 2)

    # ADX trend strength
    adx_series = adx(high, low, close, 14)
    if len(adx_series) > 0 and not pd.isna(adx_series.iloc[-1]):
        adx_val = float(adx_series.iloc[-1])
        threshold = float(thresh.get("adx_strong_threshold", 25))
        if adx_val > threshold:
            result["breakdown"]["adx_trend"] = float(thresh.get("adx_strong_score", 5))

    result["technical"] = sum(
        v for k, v in result["breakdown"].items()
        if isinstance(v, (int, float)) and not k.endswith("_value")
    )
    return result


# ---------------------------------------------------------------------------
# Momentum scorer (~35 pts max)
# ---------------------------------------------------------------------------

def score_momentum(
    df: pd.DataFrame,
    thresh: _Thresh,
) -> dict[str, Any]:
    """Score momentum based on returns and volatility."""
    result: dict[str, Any] = {
        "momentum": 0.0,
        "breakdown": {
            "return_20d": 0.0,
            "return_60d": 0.0,
            "volatility": 0.0,
        },
    }

    if len(df) < _MIN_BARS_MOMENTUM:
        return result

    close = df["close"]

    # 20-day return
    r20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0.0
    tiers_20d = thresh.get("return_20d_tiers", [[0.10, 15], [0.05, 10], [0.00, 5]])
    result["breakdown"]["return_20d"] = _tier_score(r20, tiers_20d)
    result["breakdown"]["return_20d_value"] = round(r20, 4)

    # 60-day return
    r60 = float(close.iloc[-1] / close.iloc[-60] - 1) if len(close) >= 60 else 0.0
    tiers_60d = thresh.get("return_60d_tiers", [[0.20, 15], [0.10, 10], [0.00, 5]])
    result["breakdown"]["return_60d"] = _tier_score(r60, tiers_60d)
    result["breakdown"]["return_60d_value"] = round(r60, 4)

    # Annualised volatility
    daily_ret = close.pct_change().dropna()
    if len(daily_ret) > 20:
        vol = float(daily_ret.std() * np.sqrt(252))
        ideal_low = float(thresh.get("volatility_ideal_low", 0.15))
        ideal_high = float(thresh.get("volatility_ideal_high", 0.40))
        if ideal_low < vol < ideal_high:
            result["breakdown"]["volatility"] = float(thresh.get("volatility_ideal_score", 5))
        elif vol <= ideal_low:
            result["breakdown"]["volatility"] = float(thresh.get("volatility_low_score", 2))
        result["breakdown"]["volatility_value"] = round(vol, 4)

    result["momentum"] = sum(
        v for k, v in result["breakdown"].items()
        if isinstance(v, (int, float)) and not k.endswith("_value")
    )
    return result


# ---------------------------------------------------------------------------
# Liquidity scorer (~25 pts max)
# ---------------------------------------------------------------------------

def score_liquidity(
    df: pd.DataFrame,
    thresh: _Thresh,
) -> dict[str, Any]:
    """Score liquidity based on average amount and volume ratio."""
    result: dict[str, Any] = {
        "liquidity": 0.0,
        "breakdown": {
            "avg_amount": 0.0,
            "volume_ratio": 0.0,
        },
    }

    if len(df) < _MIN_BARS_LIQUIDITY or "amount" not in df.columns:
        return result

    tail_20 = df.tail(20)
    avg_amount = float(tail_20["amount"].mean())

    # Amount tiers
    amount_tiers = thresh.get(
        "avg_amount_tiers",
        [[50000, 15], [20000, 10], [10000, 7], [0, 4]],
    )
    result["breakdown"]["avg_amount"] = _tier_score(avg_amount, amount_tiers)
    result["breakdown"]["avg_amount_value"] = round(avg_amount, 0)

    # Volume ratio (5-day vs 20-day)
    if "volume" in df.columns:
        vol_col = df["volume"]
        if len(vol_col) >= 20:
            avg_vol_20 = float(vol_col.tail(20).mean())
            avg_vol_5 = float(vol_col.tail(5).mean())
            if avg_vol_20 > 0:
                vr = avg_vol_5 / avg_vol_20
                ideal_low = float(thresh.get("volume_ratio_ideal_low", 1.2))
                ideal_high = float(thresh.get("volume_ratio_ideal_high", 2.0))
                moderate = float(thresh.get("volume_ratio_moderate", 0.8))
                if ideal_low < vr < ideal_high:
                    result["breakdown"]["volume_ratio"] = float(
                        thresh.get("volume_ratio_ideal_score", 10)
                    )
                elif vr >= ideal_high:
                    result["breakdown"]["volume_ratio"] = float(
                        thresh.get("volume_ratio_high_score", 5)
                    )
                elif vr > moderate:
                    result["breakdown"]["volume_ratio"] = float(
                        thresh.get("volume_ratio_moderate_score", 5)
                    )
                result["breakdown"]["volume_ratio_value"] = round(vr, 3)

    result["liquidity"] = sum(
        v for k, v in result["breakdown"].items()
        if isinstance(v, (int, float)) and not k.endswith("_value")
    )
    return result


# ---------------------------------------------------------------------------
# Fundamental scorer (~10 pts max, optional)
# ---------------------------------------------------------------------------

def score_fundamental(
    snapshot: FinancialSnapshot | None,
    thresh: _Thresh,
) -> dict[str, Any]:
    """Score fundamentals from a FinancialSnapshot.

    Returns score 0 when *snapshot* is ``None`` (no penalty, no reward).
    """
    result: dict[str, Any] = {
        "fundamental": 0.0,
        "breakdown": {
            "roe": 0.0,
            "revenue_growth": 0.0,
            "pe_valuation": 0.0,
            "debt_ratio": 0.0,
        },
    }

    if snapshot is None:
        result["fundamental"] = float(thresh.get("no_data_score", 0))
        return result

    # ROE
    roe = snapshot.get("roe")
    if roe is not None and roe > 0:
        roe_threshold = float(thresh.get("roe_excellent", 0.15))
        if roe >= roe_threshold:
            result["breakdown"]["roe"] = float(thresh.get("roe_score", 3))

    # Revenue growth
    rev_g = snapshot.get("revenue_growth")
    if rev_g is not None and rev_g > 0:
        rev_threshold = float(thresh.get("revenue_growth_good", 0.15))
        if rev_g >= rev_threshold:
            result["breakdown"]["revenue_growth"] = float(thresh.get("revenue_growth_score", 2))

    # PE valuation (lower is better)
    pe = snapshot.get("pe_ttm")
    if pe is not None and pe > 0:
        pe_cheap = float(thresh.get("pe_cheap", 15))
        if pe <= pe_cheap:
            result["breakdown"]["pe_valuation"] = float(thresh.get("pe_cheap_score", 2))

    # Debt ratio (high is bad)
    debt = snapshot.get("debt_ratio")
    if debt is not None:
        debt_high = float(thresh.get("debt_high", 0.7))
        if debt >= debt_high:
            result["breakdown"]["debt_ratio"] = float(thresh.get("debt_high_penalty", -2))

    result["fundamental"] = sum(
        v for v in result["breakdown"].values() if isinstance(v, (int, float))
    )
    return result
