"""Screening engine -- the main entry point for stock screening.

Coordinates the full pipeline: universe → pre-filter → fetch → filter → score → rank.

Not an Agent -- operates on multiple stocks, no ``analyze(stock_code)`` interface.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from ..data.sources.base import FinancialSnapshot
from ..thresholds import _Thresh, get_thresholds
from .filters import PreFilter
from .scorers import score_fundamental, score_liquidity, score_momentum, score_technical

logger = logging.getLogger(__name__)

# Default stock pool (沪深300 core ~150 stocks) -- fallback when BaoStock unavailable.
DEFAULT_POOL: list[str] = [
    "000001", "000002", "000063", "000333", "000338", "000425", "000538", "000568",
    "000651", "000661", "000725", "000776", "000858", "000895", "000938", "001979",
    "002007", "002120", "002129", "002142", "002230", "002236", "002241", "002304",
    "002352", "002415", "002459", "002460", "002475", "002493", "002555", "002594",
    "002601", "002602", "002709", "002714", "002812", "002841", "002916", "003816",
    "300003", "300014", "300015", "300033", "300059", "300124", "300142", "300223",
    "300274", "300347", "300394", "300408", "300413", "300418", "300433", "300454",
    "300457", "300496", "300529", "300601", "300628", "300676", "300750", "300760",
    "300782", "300832", "300896", "301269", "600009", "600010", "600016", "600019",
    "600025", "600028", "600029", "600030", "600031", "600036", "600048", "600050",
    "600061", "600085", "600089", "600104", "600111", "600115", "600150", "600153",
    "600160", "600176", "600177", "600196", "600208", "600219", "600230", "600271",
    "600276", "600282", "600299", "600309", "600332", "600346", "600362", "600369",
    "600383", "600390", "600406", "600436", "600438", "600486", "600489", "600498",
    "600519", "600521", "600570", "600588", "600589", "600596", "600600", "600606",
    "600690", "600703", "600745", "600809", "600837", "600845", "600859", "600887",
    "600893", "600900", "600905", "600918", "600919", "600941", "601006", "601012",
    "601066", "601088", "601111", "601127", "601138", "601166", "601211", "601225",
    "601228", "601236", "601288", "601318", "601328", "601336", "601390", "601398",
    "601601", "601628", "601633", "601668", "601669", "601688", "601728", "601766",
    "601788", "601799", "601818", "601857", "601881", "601899", "601901", "601919",
    "601939", "601985", "601989", "603160", "603259", "603288", "603369", "603501",
    "603799", "603833", "603986",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class StockScore:
    """Single stock screening result."""

    stock_code: str
    name: str = ""
    price: float = 0.0
    total_score: float = 0.0
    technical_score: float = 0.0
    momentum_score: float = 0.0
    liquidity_score: float = 0.0
    fundamental_score: float = 0.0
    breakdown: dict[str, Any] = field(default_factory=dict)
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stock_code": self.stock_code,
            "name": self.name,
            "price": self.price,
            "total_score": self.total_score,
            "technical_score": self.technical_score,
            "momentum_score": self.momentum_score,
            "liquidity_score": self.liquidity_score,
            "fundamental_score": self.fundamental_score,
            "passed": self.passed,
        }


@dataclass
class ScreeningResult:
    """Full screening run result."""

    timestamp: str = ""
    universe_size: int = 0
    filtered_size: int = 0
    scored_count: int = 0
    passed_count: int = 0
    top_stocks: list[StockScore] = field(default_factory=list)
    all_scores: list[StockScore] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert top stocks to a sorted DataFrame for display."""
        rows = [s.to_dict() for s in self.top_stocks]
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "universe_size": self.universe_size,
            "filtered_size": self.filtered_size,
            "scored_count": self.scored_count,
            "passed_count": self.passed_count,
            "top_stocks": [s.to_dict() for s in self.top_stocks],
        }


# ---------------------------------------------------------------------------
# ScreeningEngine
# ---------------------------------------------------------------------------

class ScreeningEngine:
    """Multi-dimensional stock screening engine.

    Workflow:
        1. Get stock universe (BaoStock or hardcoded pool)
        2. Pre-filter by name (ST, 退市)
        3. Batch fetch price data
        4. Filter by price/liquidity
        5. Score remaining stocks (technical + momentum + liquidity)
        6. Optionally add fundamental scores
        7. Rank and return top N
    """

    def __init__(
        self,
        data_service: Any,
        threshold_config: _Thresh | None = None,
        settings: Any | None = None,
    ):
        self.data = data_service
        self._thresh = threshold_config or get_thresholds().screener
        self._settings = settings
        self._prefilter = PreFilter(self._thresh.prefilter)
        self._logger = logging.getLogger("screener.engine")

    def get_universe(self, use_full_market: bool = False) -> pd.DataFrame:
        """Get stock universe as a DataFrame with columns ['symbol', 'name'].

        Args:
            use_full_market: If True, fetch from BaoStock. Otherwise use
                             DEFAULT_POOL.
        """
        if use_full_market:
            try:
                baostock = getattr(self.data, "baostock", None)
                if baostock is not None:
                    df = baostock.get_stock_list()
                    if df is not None and not df.empty:
                        self._logger.info("BaoStock universe: %d stocks", len(df))
                        return df
            except Exception:
                self._logger.warning("BaoStock get_stock_list failed, using DEFAULT_POOL")

        # Fallback to hardcoded pool
        return pd.DataFrame({"symbol": DEFAULT_POOL, "name": [""] * len(DEFAULT_POOL)})

    def screen(
        self,
        stock_codes: list[str] | None = None,
        use_full_market: bool = False,
        top_n: int | None = None,
        include_fundamentals: bool = False,
        days: int = 120,
    ) -> ScreeningResult:
        """Run the full screening pipeline.

        Args:
            stock_codes: Explicit list of codes (overrides universe fetch).
            use_full_market: Scan all A-shares via BaoStock.
            top_n: Number of top stocks to return (default from config).
            include_fundamentals: Fetch FinancialSnapshots for scoring.
            days: Price history length to fetch.

        Returns:
            ScreeningResult with ranked stock scores.
        """
        result = ScreeningResult(timestamp=datetime.now().isoformat())

        try:
            # 1. Get universe
            if stock_codes is not None:
                universe_df = pd.DataFrame({
                    "symbol": stock_codes,
                    "name": [""] * len(stock_codes),
                })
            else:
                universe_df = self.get_universe(use_full_market)

            result.universe_size = len(universe_df)

            # 2. Name-based pre-filter
            universe_df = self._prefilter.filter_universe(universe_df)
            codes = universe_df["symbol"].tolist()

            if not codes:
                self._logger.warning("No stocks after pre-filtering")
                return result

            # 3. Batch fetch price data
            self._logger.info("Fetching price data for %d stocks...", len(codes))
            price_data = self.data.get_multi_price(codes, days=days)
            self._logger.info("Fetched price data for %d stocks", len(price_data))

            # 4. Price/liquidity filter
            price_data = self._prefilter.filter_by_price_data(price_data)
            result.filtered_size = len(price_data)

            if not price_data:
                self._logger.warning("No stocks after price/liquidity filter")
                return result

            # 5. Optionally fetch financials
            financials: dict[str, FinancialSnapshot | None] = {}
            if include_fundamentals:
                try:
                    financials = self.data.get_multi_financial(list(price_data.keys()))
                except Exception:
                    self._logger.warning("Financial data fetch failed, skipping fundamentals")

            # 6. Score each stock
            scoring_thresh = self._thresh.scoring
            all_scores: list[StockScore] = []

            for code, df in price_data.items():
                try:
                    score = self._score_stock(code, df, scoring_thresh, financials.get(code))
                    all_scores.append(score)
                except Exception:
                    self._logger.debug("Scoring failed for %s", code, exc_info=True)

            result.scored_count = len(all_scores)

            # 7. Rank
            min_score = float(self._thresh.output.get("min_score_to_pass", 30))
            for s in all_scores:
                s.passed = s.total_score >= min_score
            result.passed_count = sum(1 for s in all_scores if s.passed)

            all_scores.sort(key=lambda s: s.total_score, reverse=True)

            n = top_n or int(self._thresh.output.get("default_top_n", 20))
            result.top_stocks = all_scores[:n]
            result.all_scores = all_scores

        except Exception:
            self._logger.error("Screening pipeline failed", exc_info=True)

        return result

    # -- internal --------------------------------------------------------

    def _score_stock(
        self,
        code: str,
        df: pd.DataFrame,
        scoring_thresh: _Thresh,
        snapshot: FinancialSnapshot | None = None,
    ) -> StockScore:
        """Score a single stock across all dimensions."""
        close = df["close"]
        price = float(close.iloc[-1])

        tech = score_technical(df, scoring_thresh.technical)
        mom = score_momentum(df, scoring_thresh.momentum)
        liq = score_liquidity(df, scoring_thresh.liquidity)
        fun = score_fundamental(snapshot, scoring_thresh.fundamental) if snapshot else {
            "fundamental": 0.0, "breakdown": {},
        }

        total = (
            tech["technical"]
            + mom["momentum"]
            + liq["liquidity"]
            + fun["fundamental"]
        )

        breakdown = {
            **tech.get("breakdown", {}),
            **mom.get("breakdown", {}),
            **liq.get("breakdown", {}),
            **fun.get("breakdown", {}),
        }

        return StockScore(
            stock_code=code,
            price=price,
            total_score=total,
            technical_score=tech["technical"],
            momentum_score=mom["momentum"],
            liquidity_score=liq["liquidity"],
            fundamental_score=fun["fundamental"],
            breakdown=breakdown,
        )
