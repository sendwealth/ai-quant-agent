"""回测引擎单元测试"""

import pytest
import pandas as pd
import numpy as np

from quant_agent.backtest.engine import (
    BacktestEngine, BacktestResult, Portfolio, Trade,
    CommissionModel, SlippageModel,
)


@pytest.fixture
def uptrend_data():
    """上涨趋势数据"""
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100 + np.linspace(0, 30, n) + np.random.RandomState(42).randn(n) * 2
    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "close": close,
        "high": close + 1,
        "low": close - 1,
        "volume": np.ones(n) * 100000,
    })


@pytest.fixture
def downtrend_data():
    """下跌趋势数据"""
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 130 - np.linspace(0, 30, n) + np.random.RandomState(42).randn(n) * 2
    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "close": close,
        "high": close + 1,
        "low": close - 1,
        "volume": np.ones(n) * 100000,
    })


@pytest.fixture
def buy_signals(uptrend_data):
    """MA交叉买入信号：前20天观望，之后持续持有"""
    signals = pd.Series(0, index=range(len(uptrend_data)))
    signals.iloc[20:] = 1  # 从第20天开始买入
    return signals


@pytest.fixture
def sell_signals(downtrend_data):
    """卖出信号：前20天持有，之后卖出"""
    signals = pd.Series(1, index=range(len(downtrend_data)))
    signals.iloc[20:] = -1
    return signals


class TestPortfolio:
    def test_buy(self):
        p = Portfolio(cash=100000)
        t = p.buy("TEST", 10.0, 100, commission=5.0)
        assert t.shares == 100
        assert p.cash < 100000

    def test_sell(self):
        p = Portfolio(cash=100000)
        p.buy("TEST", 10.0, 100, commission=5.0)
        t = p.sell("TEST", 12.0, commission=5.0)
        assert t.pnl > 0
        assert t.status == "closed"
        assert "TEST" not in p.positions

    def test_insufficient_funds(self):
        p = Portfolio(cash=10)
        t = p.buy("TEST", 10.0, 100, commission=5.0)
        assert t.shares == 0  # 资金不足

    def test_equity(self):
        p = Portfolio(cash=100000)
        p.buy("TEST", 10.0, 100, commission=5.0)
        p.update_price("TEST", 12.0)
        assert p.total_equity > 100000


class TestCommissionModel:
    def test_buy_commission(self):
        m = CommissionModel()
        c = m.calc(10.0, 1000, "buy")
        assert c >= 5.0  # 最低佣金

    def test_sell_with_tax(self):
        m = CommissionModel()
        c_buy = m.calc(10.0, 1000, "buy")
        c_sell = m.calc(10.0, 1000, "sell")
        assert c_sell > c_buy  # 卖出有印花税


class TestSlippageModel:
    def test_buy_slip(self):
        m = SlippageModel(basis_points=1.0)
        assert m.apply(100.0, "buy") > 100.0

    def test_sell_slip(self):
        m = SlippageModel(basis_points=1.0)
        assert m.apply(100.0, "sell") < 100.0


class TestBacktestEngine:
    def test_uptrend_buy(self, uptrend_data, buy_signals):
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(uptrend_data, buy_signals)
        assert result.total_return > 0
        assert result.total_trades >= 1

    def test_downtrend_sell(self, downtrend_data, sell_signals):
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(downtrend_data, sell_signals)
        # 卖出后不再持有，避免下跌
        assert isinstance(result, BacktestResult)

    def test_sharpe_positive_uptrend(self, uptrend_data, buy_signals):
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(uptrend_data, buy_signals)
        assert result.sharpe_ratio > 0

    def test_max_drawdown_negative(self, uptrend_data, buy_signals):
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(uptrend_data, buy_signals)
        assert result.max_drawdown <= 0

    def test_empty_data(self):
        engine = BacktestEngine()
        result = engine.run(pd.DataFrame(), pd.Series([]))
        assert result.total_return == 0

    def test_with_benchmark(self, uptrend_data, buy_signals):
        benchmark = uptrend_data["close"]
        engine = BacktestEngine()
        result = engine.run(uptrend_data, buy_signals, benchmark=benchmark)
        assert result.benchmark_return != 0 or result.alpha == 0

    def test_result_summary(self, uptrend_data, buy_signals):
        engine = BacktestEngine()
        result = engine.run(uptrend_data, buy_signals)
        s = result.summary()
        assert "Total return" in s or "总收益" in s
        assert "Sharpe" in s

    def test_hodl_strategy(self, uptrend_data):
        """买入并持有策略"""
        signals = pd.Series(1, index=range(len(uptrend_data)))
        engine = BacktestEngine()
        result = engine.run(uptrend_data, signals)
        assert result.total_return > 0


# ---------------------------------------------------------------------------
# Known-answer (deterministic) tests -- hand-computed expected values
# ---------------------------------------------------------------------------
# All tests below use SlippageModel(basis_points=0) so that execution price
# equals the bar close price exactly.  This makes the backtest fully
# deterministic and allows us to verify every yuan of commission and tax.
# ---------------------------------------------------------------------------


class TestKnownAnswerDeterministic:
    """Deterministic backtest tests with hand-computed expected values.

    Commission rules (A-share, from CommissionModel):
        commission = max(amount * 0.0003, 5.0)
        stamp_tax  = amount * 0.001          (sell only)

    Engine behaviour:
        shares = int(portfolio.cash / price) then round_shares(shares)
        Portfolio.buy() auto_round=True, recalculates commission if shares reduced
        slippage applies to execution price (disabled here)
        remaining positions liquidated at last close price at end
        total_return = equity_curve[-1] / equity_curve[0] - 1
    """

    @staticmethod
    def _make_price_data(prices: list[float]) -> pd.DataFrame:
        """Build a minimal price DataFrame from a list of close prices."""
        n = len(prices)
        return pd.DataFrame({
            "date": [f"202501{d:02d}" for d in range(1, n + 1)],
            "close": [float(p) for p in prices],
        })

    # ---- Test 1: Single buy-sell round trip --------------------------------

    def test_single_round_trip(self):
        """Buy at 100, sell at 110, zero slippage.

        With commission-aware share reduction:
            raw_shares = int(100000 / 100) = 1000
            commission  = max(100000 * 0.0003, 5) = 30.0
            cost > cash -> reduce to 900 shares, commission recalculated = 27.0
            buy amount  = 100 * 900 = 90000
            buy cost    = 90000 + 27.0 = 90027.0
            cash after buy = 100000 - 90027.0 = 9973.0

            sell amount = 110 * 900 = 99000
            sell comm   = max(99000 * 0.0003, 5) = 29.70
            stamp tax   = 99000 * 0.001 = 99.00
            sell fees   = 29.70 + 99.00 = 128.70
            sell revenue = 99000 - 128.70 = 98871.30
            cash after  = 9973.0 + 98871.30 = 108844.30

            total_return = 108844.30 / 100000 - 1 = 0.088443
        """
        prices = [100] * 2 + [100] * 5 + [110] * 3  # 10 bars total
        price_data = self._make_price_data(prices)

        # signals: bar 0 = 0, bar 1 = 1 (buy), bars 2-6 = 1 (hold),
        # bar 7 = -1 (sell), bars 8-9 = 0
        signals = pd.Series([0, 1, 1, 1, 1, 1, 1, -1, 0, 0])

        engine = BacktestEngine(
            initial_capital=100000,
            slippage=SlippageModel(basis_points=0.0),
        )
        result = engine.run(price_data, signals)

        expected_return = 108844.30 / 100000 - 1  # 0.088443

        assert abs(result.total_return - expected_return) < 0.0001
        assert result.total_trades == 1
        assert result.win_trades == 1
        assert result.lose_trades == 0
        assert result.win_rate == 1.0

    # ---- Test 2: Buy and hold, liquidated at end ---------------------------

    def test_buy_and_hold_liquidated_at_end(self):
        """Buy at 100, price rises to 120 by end, liquidated at last close.

        With commission-aware share reduction:
            raw_shares = int(100000 / 100) = 1000
            cost > cash -> reduce to 900 shares, commission recalculated = 27.0
            buy amount  = 100 * 900 = 90000
            buy cost    = 90000 + 27.0 = 90027.0
            cash after buy = 100000 - 90027.0 = 9973.0

            equity_curve[0] = 9973.0 + 900*100 = 99973.0

            Liquidation at bar 9 close = 120:
            sell amount = 120 * 900 = 108000
            sell comm   = max(108000 * 0.0003, 5) = 32.40
            stamp tax   = 108000 * 0.001 = 108.00
            sell fees   = 32.40 + 108.00 = 140.40
            sell revenue = 108000 - 140.40 = 107859.60
            cash after  = 9973.0 + 107859.60 = 117832.60

            total_return = equity_curve[-1] / equity_curve[0] - 1
                         = 117832.60 / 99973.0 - 1 = 0.178606...
        """
        # 10 bars: price rises from 100 to 120
        prices = [100, 102, 104, 106, 108, 110, 112, 114, 116, 120]
        price_data = self._make_price_data(prices)

        # Buy signal on bar 0, hold through bar 9, liquidated at end
        signals = pd.Series([1] * 10)

        engine = BacktestEngine(
            initial_capital=100000,
            slippage=SlippageModel(basis_points=0.0),
        )
        result = engine.run(price_data, signals)

        # equity_curve[0] = 99973.0, equity_curve[-1] = 117832.60
        expected_return = 117832.60 / 99973.0 - 1

        assert abs(result.total_return - expected_return) < 0.0001
        assert result.total_trades == 1

        # Verify equity curve length: 10 bars + 1 liquidation = 11 entries
        assert len(result.equity_curve) == 11
        # First equity = cash (9973.0) + position value (900 * 100)
        assert abs(result.equity_curve[0] - 99973.0) < 0.01

    # ---- Test 3: Multiple trades with a loss then a gain -------------------

    def test_two_round_trips_loss_then_gain(self):
        """Buy at 100, sell at 95 (loss); buy at 95, sell at 105 (gain).

        With board-lot rounding and commission-aware share reduction:

        Trade 1 -- BUY at bar 1, close=100:
            shares = round_shares(int(100000 / 100)) = round_shares(1000) = 1000
            commission = max(1000*100*0.0003, 5) = 30.0
            cost > cash -> reduce to 900, commission recalculated = 27.0
            buy cost    = 90000 + 27.0 = 90027.0
            cash after buy = 100000 - 90027.0 = 9973.0

        Trade 1 -- SELL at bar 4, close=95:
            sell amount = 95 * 900 = 85500
            sell comm   = max(85500 * 0.0003, 5) = 25.65
            stamp tax   = 85500 * 0.001 = 85.50
            sell fees   = 25.65 + 85.50 = 111.15
            sell revenue = 85500 - 111.15 = 85388.85
            cash after  = 9973.0 + 85388.85 = 95361.85

        Trade 2 -- BUY at bar 5, close=95:
            shares = round_shares(int(95361.85 / 95)) = round_shares(int(1003.80)) = 1000
            buy amount = 95 * 1000 = 95000
            buy comm   = max(95000 * 0.0003, 5) = 28.50
            buy cost   = 95000 + 28.50 = 95028.50
            cash after buy = 95361.85 - 95028.50 = 333.35

        Trade 2 -- SELL at bar 8, close=105:
            sell amount = 105 * 1000 = 105000
            sell comm   = max(105000 * 0.0003, 5) = 31.50
            stamp tax   = 105000 * 0.001 = 105.00
            sell fees   = 31.50 + 105.00 = 136.50
            sell revenue = 105000 - 136.50 = 104863.50
            cash after  = 333.35 + 104863.50 = 105196.85

        total_return = 105196.85 / 100000 - 1 = 0.0519685
        """
        prices = [100, 100, 97, 96, 95, 95, 98, 102, 105, 105]
        price_data = self._make_price_data(prices)

        # Bar 0: hold, Bar 1: buy, Bars 2-3: hold, Bar 4: sell
        # Bar 5: buy, Bars 6-7: hold, Bar 8: sell, Bar 9: hold
        signals = pd.Series([0, 1, 1, 1, -1, 1, 1, 1, -1, 0])

        engine = BacktestEngine(
            initial_capital=100000,
            slippage=SlippageModel(basis_points=0.0),
        )
        result = engine.run(price_data, signals)

        expected_return = 105196.85 / 100000 - 1  # 0.0519685

        assert abs(result.total_return - expected_return) < 0.0001
        assert result.total_trades == 2
        assert result.win_trades == 1
        assert result.lose_trades == 1
        assert abs(result.win_rate - 0.5) < 0.01

    # ---- Test 4: Commission minimum kicks in --------------------------------

    def test_commission_minimum_applied(self):
        """Small trade where raw commission < 5 CNY, so minimum applies.

        With commission-aware share reduction, capital=10000, price=10:
            raw_shares = int(10000 / 10) = 1000
            commission = max(10000 * 0.0003, 5) = 5.0 (minimum)
            cost = 10005 > cash -> reduce to 900, commission recalculated = 5.0
            buy amount = 10 * 900 = 9000
            buy cost = 9000 + 5.0 = 9005.0
            cash after buy = 10000 - 9005 = 995.0

            equity_curve[0] = 995.0 + 900*10 = 9995.0

        Liquidation at bar 4, close=12:
            sell amount = 12 * 900 = 10800
            raw commission = 10800 * 0.0003 = 3.24 -> min 5.0 applies
            stamp tax = 10800 * 0.001 = 10.80
            sell fees = 5.0 + 10.80 = 15.80
            sell revenue = 10800 - 15.80 = 10784.20
            cash after = 995.0 + 10784.20 = 11779.20

            total_return = equity_curve[-1] / equity_curve[0] - 1
                         = 11779.20 / 9995.0 - 1 = 0.17843...
        """
        prices = [10, 10, 11, 11, 12]
        price_data = self._make_price_data(prices)

        signals = pd.Series([1, 1, 1, 1, 1])  # buy bar 0, hold, liquidated at end

        engine = BacktestEngine(
            initial_capital=10000,
            slippage=SlippageModel(basis_points=0.0),
        )
        result = engine.run(price_data, signals)

        # equity_curve[0] = 9995.0, equity_curve[-1] = 11779.20
        expected_return = 11779.20 / 9995.0 - 1  # 0.17843...

        assert abs(result.total_return - expected_return) < 0.0001

        # Verify the buy commission was indeed the minimum
        buy_trade = result.trades[0]
        assert buy_trade.direction == "buy"
        assert abs(buy_trade.commission - 5.0) < 0.001

    # ---- Test 5: Share rounding (board lot = 100) --------------------------

    def test_share_truncation(self):
        """Shares must be rounded to board lots (multiples of 100).

        Capital = 100000, price = 149:
            raw_shares = int(100000 / 149) = int(671.14) = 671
            rounded    = round_shares(671) = 600
            buy amount = 149 * 600 = 89400
            buy comm   = max(89400 * 0.0003, 5) = 26.82
            buy cost   = 89400 + 26.82 = 89426.82
            cash after = 100000 - 89426.82 = 10573.18

            equity_curve[0] = 10573.18 + 600 * 149 = 10573.18 + 89400 = 99973.18

        Liquidation at close=160:
            sell amount = 160 * 600 = 96000
            sell comm   = max(96000 * 0.0003, 5) = 28.80
            stamp tax   = 96000 * 0.001 = 96.00
            sell fees   = 28.80 + 96.00 = 124.80
            sell revenue = 96000 - 124.80 = 95875.20
            cash after  = 10573.18 + 95875.20 = 106448.38

            total_return = equity_curve[-1] / equity_curve[0] - 1
                         = 106448.38 / 99973.18 - 1 = 0.06475...
        """
        prices = [149, 149, 155, 158, 160]
        price_data = self._make_price_data(prices)

        signals = pd.Series([1, 1, 1, 1, 1])  # buy and hold, liquidated at end

        engine = BacktestEngine(
            initial_capital=100000,
            slippage=SlippageModel(basis_points=0.0),
        )
        result = engine.run(price_data, signals)

        # equity_curve[0] = 99973.18, equity_curve[-1] = 106448.38
        expected_return = 106448.38 / 99973.18 - 1

        assert abs(result.total_return - expected_return) < 0.0001

        # Verify that exactly 600 shares were bought (board lot rounding)
        buy_trade = result.trades[0]
        assert buy_trade.shares == 600
