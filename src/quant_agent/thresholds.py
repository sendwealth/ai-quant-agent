"""Threshold configuration loader — loads agent scoring thresholds from YAML.

Falls back to hardcoded defaults when the YAML file is not found, ensuring
backward compatibility out of the box.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_DEFAULT_YAML = Path(__file__).resolve().parent.parent.parent / "config" / "agent_thresholds.yaml"

# ---------------------------------------------------------------------------
# Dot-access helper — wraps a plain dict so thresholds read naturally
# ---------------------------------------------------------------------------


class _Thresh:
    """Lightweight dict wrapper that supports dot-access and nested traversal.

    ``t.rsi.oversold``  instead of ``t["rsi"]["oversold"]``.
    Falls through to the wrapped dict for ``__contains__`` / ``len`` / iteration.
    """

    __slots__ = ("_d",)

    def __init__(self, d: dict[str, Any] | None = None):
        object.__setattr__(self, "_d", d or {})

    # -- dot access ----------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        v = self._d.get(name)
        if isinstance(v, dict):
            return _Thresh(v)
        if v is None:
            # Return a silent _Thresh so chained dot-access on missing keys
            # doesn't crash (caller gets None at the leaf), but log a warning
            # so YAML typos are visible in diagnostics.
            logger.warning("Unknown threshold key accessed: %s", name)
            return _Thresh()
        return v

    def __repr__(self) -> str:
        return f"Thresh({self._d})"

    # -- dict-like helpers ---------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value by key, returning *default* when the key is absent.

        Unlike dot-access (which returns an empty ``_Thresh`` for missing
        keys), this returns the raw value or the caller-supplied default.
        """
        v = self._d.get(key)
        return v if v is not None else default

    def __contains__(self, key: str) -> bool:  # pragma: no cover
        return key in self._d

    def __len__(self) -> int:  # pragma: no cover
        return len(self._d)

    def __iter__(self):  # pragma: no cover
        return iter(self._d)

    def keys(self):  # pragma: no cover
        return self._d.keys()

    def values(self):  # pragma: no cover
        return self._d.values()

    def items(self):  # pragma: no cover
        return self._d.items()

    # -- booly: treat empty / all-None as falsy ------------------------------

    def __bool__(self) -> bool:
        return bool(self._d)


# ---------------------------------------------------------------------------
# Hardcoded defaults (kept in sync with agent_thresholds.yaml)
# ---------------------------------------------------------------------------

_TECHNICAL_DEFAULTS: dict[str, Any] = {
    "rsi": {
        "oversold": 30,
        "overbought": 70,
        "oversold_buy_score": 2.0,
        "near_oversold": 45,
        "near_oversold_buy_score": 0.5,
        "near_overbought": 60,
        "near_overbought_sell_score": 0.5,
        "overbought_sell_score": 2.0,
    },
    "macd": {
        "golden_cross_score": 2.0,
        "death_cross_score": 2.0,
    },
    "ema": {
        "uptrend_score": 1.5,
        "downtrend_score": 1.5,
    },
    "adx": {
        "strong_threshold": 25,
        "multiplier": 1.2,
    },
    "volume": {
        "high_ratio": 1.5,
        "low_ratio": 0.7,
        "confirmation_score": 1.0,
    },
    "signal": {
        "strong_buy_diff": 3.0,
        "strong_buy_confidence": 0.80,
        "strong_sell_diff": -3.0,
        "strong_sell_confidence": 0.75,
        "weak_buy_diff": 1.0,
        "weak_buy_confidence": 0.65,
        "weak_sell_diff": -1.0,
        "weak_sell_confidence": 0.60,
        "hold_confidence": 0.50,
    },
}

_FUNDAMENTAL_DEFAULTS: dict[str, Any] = {
    "profitability": {
        "base": 5,
        "roe_excellent": 0.15,
        "roe_excellent_score": 2,
        "roe_good": 0.10,
        "roe_good_score": 1,
        "net_margin_excellent": 0.15,
        "net_margin_excellent_score": 1,
        "net_margin_poor": 0.05,
        "net_margin_poor_score": -1,
        "min": 1,
        "max": 10,
    },
    "valuation": {
        "base": 5,
        "pe_negative_score": -3,
        "pe_cheap": 15,
        "pe_cheap_score": 3,
        "pe_fair": 25,
        "pe_fair_score": 1,
        "pe_expensive": 35,
        "pe_expensive_score": -1,
        "pe_very_expensive": 50,
        "pe_very_expensive_score": -2,
        "pb_low": 2.0,
        "pb_low_score": 1,
        "pb_high": 6.0,
        "pb_high_score": -1,
        "min": 1,
        "max": 10,
    },
    "health": {
        "base": 5,
        "debt_low": 0.4,
        "debt_low_score": 2,
        "debt_moderate": 0.6,
        "debt_moderate_score": 1,
        "debt_high": 0.8,
        "debt_high_score": -2,
        "current_ratio_strong": 2.0,
        "current_ratio_strong_score": 1,
        "current_ratio_weak": 1.0,
        "current_ratio_weak_score": -2,
        "min": 1,
        "max": 10,
    },
    "growth": {
        "base": 5,
        "revenue_high": 0.30,
        "revenue_high_score": 3,
        "revenue_good": 0.15,
        "revenue_good_score": 1,
        "revenue_negative_score": -2,
        "profit_high": 0.30,
        "profit_high_score": 1,
        "profit_negative_score": -1,
        "min": 1,
        "max": 10,
    },
    "signal": {
        "excellent_score": 7.5,
        "excellent_min_valuation": 6,
        "excellent_confidence_base": 0.60,
        "excellent_confidence_divisor": 50,
        "excellent_confidence_cap": 0.90,
        "good_score": 6.0,
        "good_confidence_base": 0.55,
        "good_confidence_divisor": 50,
        "good_confidence_cap": 0.80,
        "fair_score": 4.5,
        "fair_confidence": 0.55,
        "weak_score": 3.0,
        "weak_confidence": 0.50,
        "poor_sell_confidence": 0.65,
    },
}

_SCREENER_DEFAULTS: dict[str, Any] = {
    "prefilter": {
        "min_avg_amount": 5000,        # 20日均成交额下限(千元)
        "min_price": 5.0,
        "max_price": 300.0,
        "exclude_st": True,
    },
    "scoring": {
        "technical": {
            "ema_cross_score": 12,
            "ema_price_above_score": 5,
            "macd_positive_score": 8,
            "macd_rising_score": 5,
            "rsi_neutral_low": 35,
            "rsi_neutral_high": 65,
            "rsi_neutral_score": 5,
            "rsi_oversold_score": 3,
            "adx_strong_threshold": 25,
            "adx_strong_score": 5,
        },
        "momentum": {
            "return_20d_tiers": [[0.10, 15], [0.05, 10], [0.00, 5]],
            "return_60d_tiers": [[0.20, 15], [0.10, 10], [0.00, 5]],
            "volatility_ideal_low": 0.15,
            "volatility_ideal_high": 0.40,
            "volatility_ideal_score": 5,
            "volatility_low_score": 2,
        },
        "liquidity": {
            "avg_amount_tiers": [[50000, 15], [20000, 10], [10000, 7], [0, 4]],
            "volume_ratio_ideal_low": 1.2,
            "volume_ratio_ideal_high": 2.0,
            "volume_ratio_ideal_score": 10,
            "volume_ratio_high_score": 5,
            "volume_ratio_moderate": 0.8,
            "volume_ratio_moderate_score": 5,
        },
        "fundamental": {
            "roe_excellent": 0.15,
            "roe_score": 3,
            "revenue_growth_good": 0.15,
            "revenue_growth_score": 2,
            "pe_cheap": 15,
            "pe_cheap_score": 2,
            "debt_high": 0.7,
            "debt_high_penalty": -2,
            "no_data_score": 0,
        },
    },
    "output": {
        "default_top_n": 20,
        "min_score_to_pass": 30,
    },
}

_FULL_DEFAULTS: dict[str, Any] = {
    "technical": _TECHNICAL_DEFAULTS,
    "fundamental": _FUNDAMENTAL_DEFAULTS,
    "screener": _SCREENER_DEFAULTS,
}


# ---------------------------------------------------------------------------
# Deep merge helper
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Merge *override* into *base*, returning a new dict."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_thresholds(path: Path | str | None = None) -> _Thresh:
    """Load thresholds from YAML, falling back to defaults on any failure.

    Returns a ``_Thresh`` with ``.technical`` and ``.fundamental`` branches.
    """
    target = Path(path) if path is not None else _DEFAULT_YAML

    if target.exists():
        try:
            import yaml

            with open(target, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            merged = _deep_merge(_FULL_DEFAULTS, data)
            logger.debug("Loaded agent thresholds from %s", target)
            return _Thresh(merged)
        except Exception:
            logger.warning("Failed to load thresholds from %s — using defaults", target, exc_info=True)

    return _Thresh(_FULL_DEFAULTS)


# Module-level singleton — loaded once, safe for concurrent reads.
_thresholds: _Thresh | None = None


def get_thresholds(path: Path | str | None = None) -> _Thresh:
    """Return the module-level threshold singleton (lazy-loaded)."""
    global _thresholds
    if _thresholds is None:
        _thresholds = load_thresholds(path)
    return _thresholds


def reset_thresholds() -> None:
    """Reset the singleton so the next ``get_thresholds()`` call re-reads the file."""
    global _thresholds
    _thresholds = None
