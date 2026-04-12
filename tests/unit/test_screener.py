"""Tests for the screener module — PreFilter, Scorers, ScreeningEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_agent.screener.engine import (
    DEFAULT_POOL,
    ScreeningEngine,
    ScreeningResult,
    StockScore,
)
from quant_agent.screener.filters import PreFilter
from quant_agent.screener.scorers import (
    score_fundamental,
    score_liquidity,
    score_momentum,
    score_technical,
)
from quant_agent.thresholds import _Thresh, get_thresholds, reset_thresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_thresh():
    """Ensure fresh thresholds for each test."""
    reset_thresholds()
    yield
    reset_thresholds()


def _make_df(
    n: int = 120,
    trend: str = "up",
    base_price: float = 50.0,
    avg_amount: float = 30000.0,
) -> pd.DataFrame:
    """Build a price DataFrame with specified trend."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    if trend == "up":
        close = base_price + np.linspace(0, 20, n) + np.random.RandomState(42).randn(n) * 0.5
    elif trend == "down":
        close = base_price + 20 - np.linspace(0, 20, n) + np.random.RandomState(42).randn(n) * 0.5
    else:
        close = np.full(n, base_price) + np.random.RandomState(42).randn(n) * 0.5

    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "close": close,
        "high": close + 1,
        "low": close - 1,
        "volume": np.ones(n) * 100000,
        "amount": np.ones(n) * avg_amount,
    })


@pytest.fixture
def thresh():
    """Screener scoring thresholds."""
    return get_thresholds().screener.scoring


@pytest.fixture
def uptrend_df():
    return _make_df(120, "up")


@pytest.fixture
def downtrend_df():
    return _make_df(120, "down")


@pytest.fixture
def low_liq_df():
    return _make_df(120, "up", avg_amount=2000)


@pytest.fixture
def mock_data_service():
    """Mock DataService with get_multi_price and get_multi_financial."""
    ds = MagicMock()

    # Default: return uptrend data for all requested codes
    def _multi_price(codes, days=120, **kw):
        return {code: _make_df(days, "up") for code in codes}

    ds.get_multi_price = MagicMock(side_effect=_multi_price)
    ds.get_multi_financial = MagicMock(return_value={})
    ds.baostock = None

    return ds


# ---------------------------------------------------------------------------
# TestPreFilter
# ---------------------------------------------------------------------------


class TestPreFilter:
    def test_is_st_detects_st(self):
        assert PreFilter.is_st("ST某某") is True
        assert PreFilter.is_st("*ST某某") is True

    def test_is_st_normal_name(self):
        assert PreFilter.is_st("宁德时代") is False
        assert PreFilter.is_st("贵州茅台") is False

    def test_is_st_empty(self):
        assert PreFilter.is_st("") is False
        assert PreFilter.is_st(None) is False

    def test_is_st_tui(self):
        assert PreFilter.is_st("退市海润") is True

    def test_filter_universe_removes_st(self):
        df = pd.DataFrame({
            "symbol": ["300750", "600000", "000001"],
            "name": ["宁德时代", "ST平安", "平安银行"],
        })
        pf = PreFilter()
        result = pf.filter_universe(df)
        assert len(result) == 2
        assert "ST平安" not in result["name"].values

    def test_filter_universe_empty(self):
        pf = PreFilter()
        result = pf.filter_universe(pd.DataFrame(columns=["symbol", "name"]))
        assert len(result) == 0

    def test_filter_by_price_data_removes_low_amount(self):
        data = {
            "GOOD": _make_df(120, "up", avg_amount=30000),
            "BAD": _make_df(120, "up", avg_amount=1000),
        }
        pf = PreFilter()
        result = pf.filter_by_price_data(data)
        assert "GOOD" in result
        assert "BAD" not in result

    def test_filter_by_price_data_removes_extreme_price(self):
        data = {
            "CHEAP": _make_df(120, "flat", base_price=3.0),  # flat keeps it at 3.0
            "PRICY": _make_df(120, "flat", base_price=350.0),
            "NORMAL": _make_df(120, "flat", base_price=50.0),
        }
        pf = PreFilter()
        result = pf.filter_by_price_data(data)
        assert "NORMAL" in result
        assert "CHEAP" not in result
        assert "PRICY" not in result

    def test_filter_by_price_data_empty(self):
        pf = PreFilter()
        assert pf.filter_by_price_data({}) == {}

    def test_filter_by_price_data_short_df(self):
        """DataFrames with < 5 rows are skipped."""
        short_df = _make_df(3, "up")
        data = {"SHORT": short_df}
        pf = PreFilter()
        result = pf.filter_by_price_data(data)
        assert "SHORT" not in result


# ---------------------------------------------------------------------------
# TestScorers
# ---------------------------------------------------------------------------


class TestScorers:
    def test_technical_uptrend_high_score(self, uptrend_df, thresh):
        result = score_technical(uptrend_df, thresh.technical)
        assert result["technical"] > 0
        # Uptrend should have EMA cross score at minimum
        assert result["breakdown"]["ema_cross"] > 0

    def test_technical_downtrend_low_score(self, downtrend_df, thresh):
        result = score_technical(downtrend_df, thresh.technical)
        # Downtrend: no EMA cross, possibly no MACD positive
        assert isinstance(result["technical"], float)

    def test_technical_short_data_returns_zero(self, thresh):
        short_df = _make_df(30, "up")
        result = score_technical(short_df, thresh.technical)
        assert result["technical"] == 0.0

    def test_momentum_strong_positive(self, uptrend_df, thresh):
        result = score_momentum(uptrend_df, thresh.momentum)
        # Uptrend should have positive return scores
        assert result["momentum"] > 0
        assert "return_20d_value" in result["breakdown"]
        assert "return_60d_value" in result["breakdown"]

    def test_momentum_negative_returns(self, downtrend_df, thresh):
        result = score_momentum(downtrend_df, thresh.momentum)
        # Downtrend may have 0 return score but volatility score
        assert isinstance(result["momentum"], float)

    def test_momentum_short_data_returns_zero(self, thresh):
        result = score_momentum(_make_df(30, "up"), thresh.momentum)
        assert result["momentum"] == 0.0

    def test_liquidity_high_volume(self, uptrend_df, thresh):
        result = score_liquidity(uptrend_df, thresh.liquidity)
        assert result["liquidity"] > 0
        assert result["breakdown"]["avg_amount"] > 0

    def test_liquidity_low_volume(self, low_liq_df, thresh):
        result = score_liquidity(low_liq_df, thresh.liquidity)
        # Low liquidity gets lower score
        assert result["liquidity"] < 15

    def test_liquidity_no_amount_column(self, thresh):
        df = _make_df(120, "up")
        df = df.drop(columns=["amount"])
        result = score_liquidity(df, thresh.liquidity)
        assert result["liquidity"] == 0.0

    def test_fundamental_excellent(self, thresh):
        snapshot = MagicMock()
        snapshot.get.side_effect = lambda k: {
            "roe": 0.20,
            "revenue_growth": 0.25,
            "pe_ttm": 12.0,
            "debt_ratio": 0.3,
        }.get(k)
        result = score_fundamental(snapshot, thresh.fundamental)
        assert result["fundamental"] > 0
        assert result["breakdown"]["roe"] > 0
        assert result["breakdown"]["pe_valuation"] > 0

    def test_fundamental_no_snapshot(self, thresh):
        result = score_fundamental(None, thresh.fundamental)
        assert result["fundamental"] == 0.0

    def test_fundamental_poor(self, thresh):
        snapshot = MagicMock()
        snapshot.get.side_effect = lambda k: {
            "roe": 0.02,
            "revenue_growth": -0.1,
            "pe_ttm": 80.0,
            "debt_ratio": 0.85,
        }.get(k)
        result = score_fundamental(snapshot, thresh.fundamental)
        # High debt should get penalty
        assert result["breakdown"]["debt_ratio"] < 0

    def test_fundamental_negative_pe(self, thresh):
        """Negative PE should not get cheap valuation score."""
        snapshot = MagicMock()
        snapshot.get.side_effect = lambda k: {"pe_ttm": -5.0}.get(k)
        result = score_fundamental(snapshot, thresh.fundamental)
        assert result["breakdown"]["pe_valuation"] == 0.0

    def test_tier_score_matching(self):
        from quant_agent.screener.scorers import _tier_score
        tiers = [[0.10, 15], [0.05, 10], [0.00, 5]]
        assert _tier_score(0.15, tiers) == 15
        assert _tier_score(0.08, tiers) == 10
        assert _tier_score(0.02, tiers) == 5
        assert _tier_score(-0.05, tiers) == 0

    def test_technical_value_fields_excluded_from_score(self, uptrend_df, thresh):
        """_value metadata fields must NOT contribute to the technical score."""
        result = score_technical(uptrend_df, thresh.technical)
        breakdown = result["breakdown"]
        # Compute score from non-_value fields only
        expected = sum(
            v for k, v in breakdown.items()
            if isinstance(v, (int, float)) and not k.endswith("_value")
        )
        assert result["technical"] == expected
        # If rsi_value exists, it must NOT be in the sum
        if "rsi_value" in breakdown:
            assert result["technical"] != sum(
                v for v in breakdown.values() if isinstance(v, (int, float))
            )

    def test_momentum_value_fields_excluded_from_score(self, uptrend_df, thresh):
        """_value metadata fields must NOT contribute to the momentum score."""
        result = score_momentum(uptrend_df, thresh.momentum)
        breakdown = result["breakdown"]
        expected = sum(
            v for k, v in breakdown.items()
            if isinstance(v, (int, float)) and not k.endswith("_value")
        )
        assert result["momentum"] == expected

    def test_liquidity_value_fields_excluded_from_score(self, uptrend_df, thresh):
        """_value metadata fields must NOT contribute to the liquidity score."""
        result = score_liquidity(uptrend_df, thresh.liquidity)
        breakdown = result["breakdown"]
        expected = sum(
            v for k, v in breakdown.items()
            if isinstance(v, (int, float)) and not k.endswith("_value")
        )
        assert result["liquidity"] == expected

    def test_technical_no_high_low_columns(self, thresh):
        """When high/low columns are missing, close is used as fallback."""
        df = _make_df(120, "up").drop(columns=["high", "low"])
        result = score_technical(df, thresh.technical)
        # Should not crash; ADX may still produce a value
        assert isinstance(result["technical"], float)

    def test_fundamental_all_zeros(self, thresh):
        """Snapshot with all zero values should get score 0."""
        snapshot = MagicMock()
        snapshot.get.side_effect = lambda k: {
            "roe": 0.0,
            "revenue_growth": 0.0,
            "pe_ttm": 0.0,
            "debt_ratio": 0.0,
        }.get(k)
        result = score_fundamental(snapshot, thresh.fundamental)
        assert result["fundamental"] == 0.0

    def test_liquidity_no_volume_column(self, thresh):
        """Missing volume column should still score avg_amount."""
        df = _make_df(120, "up").drop(columns=["volume"])
        result = score_liquidity(df, thresh.liquidity)
        # avg_amount should still be scored
        assert result["breakdown"]["avg_amount"] > 0
        # volume_ratio stays at 0
        assert result["breakdown"]["volume_ratio"] == 0.0

    def test_momentum_volatility_high(self, thresh):
        """High volatility (outside ideal range) should not get ideal score."""
        np.random.seed(42)
        n = 120
        dates = pd.date_range("2025-01-01", periods=n, freq="B")
        # Wild swings → high volatility
        close = 50 + np.cumsum(np.random.randn(n) * 5)
        close = np.maximum(close, 5.0)
        df = pd.DataFrame({
            "date": dates.strftime("%Y%m%d"),
            "close": close,
        })
        result = score_momentum(df, thresh.momentum)
        # Volatility score may be 0 or low
        assert isinstance(result["breakdown"]["volatility"], float)


# ---------------------------------------------------------------------------
# TestScreeningEngine
# ---------------------------------------------------------------------------


class TestScreeningEngine:
    def test_screen_with_explicit_codes(self, mock_data_service):
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750", "002475"])
        assert isinstance(result, ScreeningResult)
        assert result.universe_size == 2

    def test_screen_returns_screening_result(self, mock_data_service):
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750"])
        assert isinstance(result, ScreeningResult)
        assert result.timestamp != ""

    def test_screen_top_n(self, mock_data_service):
        codes = [f"30000{i}" for i in range(10)]
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=codes, top_n=3)
        assert len(result.top_stocks) <= 3

    def test_screen_no_data_returns_empty(self, mock_data_service):
        # Override side_effect to return empty dict
        mock_data_service.get_multi_price = MagicMock(return_value={})
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750"])
        assert result.scored_count == 0
        assert result.top_stocks == []

    def test_screen_scores_sorted_descending(self, mock_data_service):
        # Create different-trending data for different codes
        def _varied_prices(codes, days=120, **kw):
            data = {}
            for i, code in enumerate(codes):
                trend = "up" if i % 2 == 0 else "down"
                data[code] = _make_df(days, trend)
            return data

        mock_data_service.get_multi_price = MagicMock(side_effect=_varied_prices)
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750", "002475", "601318", "600276"])
        scores = [s.total_score for s in result.top_stocks]
        assert scores == sorted(scores, reverse=True)

    def test_screen_min_score_filter(self, mock_data_service):
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750"])
        # Check that passed field is set
        for s in result.all_scores:
            assert isinstance(s.passed, bool)

    def test_to_dataframe(self, mock_data_service):
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750", "002475"])
        df = result.to_dataframe()
        if not df.empty:
            assert "stock_code" in df.columns
            assert "total_score" in df.columns

    def test_include_fundamentals(self, mock_data_service):
        mock_snapshot = MagicMock()
        mock_snapshot.get.return_value = 0.2
        mock_data_service.get_multi_financial.return_value = {"300750": mock_snapshot}

        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=["300750"], include_fundamentals=True)
        mock_data_service.get_multi_financial.assert_called_once()

    def test_full_market_flag(self, mock_data_service):
        mock_bs = MagicMock()
        mock_bs.get_stock_list.return_value = pd.DataFrame({
            "symbol": ["300750", "002475"],
            "name": ["宁德时代", "赣锋锂业"],
        })
        mock_data_service.baostock = mock_bs

        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(use_full_market=True)
        mock_bs.get_stock_list.assert_called_once()

    def test_fallback_to_default_pool(self, mock_data_service):
        mock_data_service.baostock = None
        engine = ScreeningEngine(data_service=mock_data_service)
        universe = engine.get_universe(use_full_market=True)
        # Should fallback to DEFAULT_POOL
        assert len(universe) > 0

    def test_default_pool_not_empty(self):
        assert len(DEFAULT_POOL) > 100

    def test_screen_all_scores_includes_all(self, mock_data_service):
        codes = ["300750", "002475", "601318"]
        engine = ScreeningEngine(data_service=mock_data_service)
        result = engine.screen(stock_codes=codes)
        # all_scores may be fewer than codes due to filters, but should
        # contain at least the filtered entries
        assert result.scored_count <= len(codes)

    def test_stock_score_to_dict(self):
        s = StockScore(stock_code="300750", total_score=85.0)
        d = s.to_dict()
        assert d["stock_code"] == "300750"
        assert d["total_score"] == 85.0

    def test_screening_result_to_dict(self):
        r = ScreeningResult(
            timestamp="2025-01-01",
            universe_size=100,
            top_stocks=[StockScore(stock_code="300750", total_score=80.0, passed=True)],
        )
        d = r.to_dict()
        assert d["universe_size"] == 100
        assert len(d["top_stocks"]) == 1

    def test_screening_result_empty_to_dataframe(self):
        r = ScreeningResult()
        df = r.to_dataframe()
        assert df.empty


# ---------------------------------------------------------------------------
# TestThresholds
# ---------------------------------------------------------------------------


class TestScreenerThresholds:
    def test_screener_defaults_load(self):
        t = get_thresholds().screener
        assert bool(t)  # non-empty

    def test_prefilter_values(self):
        t = get_thresholds().screener.prefilter
        assert t.get("min_avg_amount", 0) == 5000
        assert t.get("min_price", 0) == 5.0
        assert t.get("max_price", 0) == 300.0

    def test_scoring_technical_present(self):
        t = get_thresholds().screener.scoring.technical
        assert t.get("ema_cross_score", 0) == 12
        assert t.get("macd_positive_score", 0) == 8

    def test_scoring_momentum_tiers(self):
        t = get_thresholds().screener.scoring.momentum
        tiers = t.get("return_20d_tiers", [])
        assert isinstance(tiers, list)
        assert len(tiers) == 3

    def test_output_defaults(self):
        t = get_thresholds().screener.output
        assert t.get("default_top_n", 0) == 20
        assert t.get("min_score_to_pass", 0) == 30
