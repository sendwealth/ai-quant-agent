"""Pre-screening filters -- eliminate clearly unsuitable stocks before scoring.

Filters run in two phases:
  1. Name-based (before data fetching): removes ST / 退市 stocks
  2. Data-based (after fetching): removes low-liquidity and extreme-price stocks
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from ..thresholds import _Thresh, get_thresholds

logger = logging.getLogger(__name__)


class PreFilter:
    """Pre-screening filters that run BEFORE expensive data fetching.

    Eliminates clearly unsuitable stocks to avoid wasting API calls.
    """

    def __init__(self, threshold_config: _Thresh | None = None):
        self._thresh = threshold_config or get_thresholds().screener.prefilter
        self._logger = logging.getLogger("screener.prefilter")

    # -- phase 1: name-based filtering (zero network cost) -------------------

    def filter_universe(self, stock_list: pd.DataFrame) -> pd.DataFrame:
        """Apply name-based filters to the stock universe DataFrame.

        Args:
            stock_list: DataFrame with at least ``['symbol']`` column.
                        May also have ``['name']`` for ST filtering.

        Returns:
            Filtered DataFrame (ST/退市 rows removed if exclude_st is True).
        """
        if stock_list.empty:
            return stock_list

        df = stock_list.copy()
        before = len(df)

        if self._thresh.get("exclude_st", True) and "name" in df.columns:
            mask = ~df["name"].apply(self.is_st)
            df = df[mask].reset_index(drop=True)
            removed = before - len(df)
            if removed > 0:
                self._logger.info("Pre-filter removed %d ST/退市 stocks", removed)

        self._logger.debug("Universe after name filter: %d → %d", before, len(df))
        return df

    @staticmethod
    def is_st(name: str) -> bool:
        """Check if stock name contains ST or 退 markers."""
        if not isinstance(name, str):
            return False
        return "ST" in name.upper() or "退" in name

    # -- phase 2: data-based filtering (after price data fetched) ------------

    def filter_by_price_data(
        self,
        price_data: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Remove stocks failing price/liquidity filters.

        Uses the last 20 bars to compute average amount and current price.
        Stocks below *min_avg_amount*, below *min_price*, or above
        *max_price* are removed.

        Args:
            price_data: ``{code: DataFrame}`` from DataService.get_multi_price()

        Returns:
            Filtered subset of *price_data*.
        """
        if not price_data:
            return {}

        min_amount = float(self._thresh.get("min_avg_amount", 5000))
        min_price = float(self._thresh.get("min_price", 5.0))
        max_price = float(self._thresh.get("max_price", 300.0))

        filtered: dict[str, pd.DataFrame] = {}
        for code, df in price_data.items():
            if df is None or df.empty or len(df) < 5:
                continue

            close = df["close"]
            current_price = float(close.iloc[-1])

            if current_price < min_price or current_price > max_price:
                continue

            # Compute 20-day average amount (千万元 → 千元 scale varies by source)
            tail = df.tail(20)
            amount_col = "amount" if "amount" in tail.columns else None
            if amount_col is not None:
                avg_amount = float(tail[amount_col].mean())
                if avg_amount < min_amount:
                    continue

            filtered[code] = df

        self._logger.info(
            "Price/liquidity filter: %d → %d stocks",
            len(price_data), len(filtered),
        )
        return filtered
