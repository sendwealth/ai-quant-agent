"""Board lot rounding test — verify 100-share rounding in backtest"""

import pandas as pd
import numpy as np

from quant_agent.backtest.engine import BacktestEngine, SlippageModel


class TestBoardLotRounding:
    def test_shares_rounded_to_100s(self):
        """Verify backtest buys in multiples of 100 shares."""
        prices = [100] * 10
        price_data = pd.DataFrame({
            "date": [f"2025010{d}" for d in range(10)],
            "close": [float(p) for p in prices],
        })
        signals = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        engine = BacktestEngine(
            initial_capital=100000,
            slippage=SlippageModel(basis_points=0.0),
        )
        result = engine.run(price_data, signals)

        # All trades must have shares in multiples of 100
        for trade in result.trades:
            if trade.direction == "buy":
                assert trade.shares % 100 == 0, (
                    f"Shares {trade.shares} not a multiple of 100"
                )

    def test_no_zero_share_trades(self):
        """Verify no zero-share buy trades occur."""
        prices = [500] * 5
        price_data = pd.DataFrame({
            "date": [f"2025010{d}" for d in range(5)],
            "close": [float(p) for p in prices],
        })
        signals = pd.Series([1, 1, 1, 1, 1])

        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(price_data, signals)

        for trade in result.trades:
            if trade.direction == "buy":
                assert trade.shares > 0
