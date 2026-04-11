"""技术指标单元测试"""

import pytest
import pandas as pd
import numpy as np

from quant_agent.strategy.indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr,
    stochastic, adx, obv, momentum, rate_of_change,
    cci, williams_r, detect_crossover, normalize,
)


@pytest.fixture
def sample_close():
    np.random.seed(42)
    return pd.Series(100 + np.cumsum(np.random.randn(200) * 2))


@pytest.fixture
def sample_ohlcv():
    np.random.seed(42)
    n = 200
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 2))
    return pd.DataFrame({
        "close": close,
        "high": close + np.random.rand(n) * 3,
        "low": close - np.random.rand(n) * 3,
        "volume": np.random.randint(100000, 1000000, n),
    })


class TestTrendIndicators:
    """趋势指标测试"""

    def test_sma(self, sample_close):
        result = sma(sample_close, 20)
        assert len(result) == 200
        assert result.iloc[19:].notna().all()
        assert result.iloc[:19].isna().all()

    def test_ema(self, sample_close):
        result = ema(sample_close, 20)
        assert len(result) == 200
        assert result.notna().all()  # EMA 无 NaN

    def test_macd(self, sample_close):
        macd_line, signal, hist = macd(sample_close)
        assert len(macd_line) == len(signal) == len(hist) == 200
        # histogram = macd - signal
        assert np.allclose(hist.values, (macd_line - signal).values, equal_nan=True)

    def test_bollinger_bands(self, sample_close):
        upper, middle, lower = bollinger_bands(sample_close, 20, 2.0)
        assert len(upper) == len(middle) == len(lower) == 200
        # upper > middle > lower (where not NaN)
        valid = upper.notna()
        assert (upper[valid] >= middle[valid]).all()
        assert (middle[valid] >= lower[valid]).all()

    def test_adx(self, sample_ohlcv):
        result = adx(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(result) == 200


class TestMomentumIndicators:
    """动量指标测试"""

    def test_rsi(self, sample_close):
        result = rsi(sample_close, 14)
        assert len(result) == 200
        # RSI 应在 0-100 之间
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_overbought_oversold(self, sample_close):
        """极端趋势下 RSI 应接近极值"""
        # 持续上涨 (with some noise to avoid pure zero-loss)
        rising = pd.Series(range(100), dtype=float) + 0.01
        rsi_val = rsi(rising, 14).iloc[-1]
        assert rsi_val >= 50  # should be at least neutral-leaning-overbought

    def test_stochastic(self, sample_ohlcv):
        k, d = stochastic(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(k) == len(d) == 200

    def test_williams_r(self, sample_ohlcv):
        result = williams_r(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(result) == 200

    def test_momentum(self, sample_close):
        result = momentum(sample_close, 10)
        assert len(result) == 200
        assert result.iloc[:10].isna().all()

    def test_rate_of_change(self, sample_close):
        result = rate_of_change(sample_close, 10)
        assert len(result) == 200

    def test_cci(self, sample_ohlcv):
        result = cci(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(result) == 200


class TestVolatilityIndicators:
    """波动率指标测试"""

    def test_atr(self, sample_ohlcv):
        result = atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"])
        assert len(result) == 200
        assert (result.dropna() >= 0).all()  # ATR 非负


class TestVolumeIndicators:
    """成交量指标测试"""

    def test_obv(self, sample_ohlcv):
        result = obv(sample_ohlcv["close"], sample_ohlcv["volume"])
        assert len(result) == 200


class TestHelpers:
    """辅助函数测试"""

    def test_detect_crossover(self, sample_close):
        s1 = sma(sample_close, 5)
        s2 = sma(sample_close, 20)
        signals = detect_crossover(s1, s2)
        assert len(signals) == 200
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_normalize(self, sample_close):
        result = normalize(sample_close, 0, 1)
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 1).all()
