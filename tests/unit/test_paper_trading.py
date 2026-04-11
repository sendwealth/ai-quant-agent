"""Tests for PaperTradingService -- persistent portfolio state."""

import json
import os
from pathlib import Path

import pytest

from quant_agent.execution.paper_trading import PaperTradingService
from quant_agent.portfolio import CommissionModel, Portfolio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _svc(tmp_path: Path, initial_capital: float = 100_000.0) -> PaperTradingService:
    """Create a PaperTradingService rooted in *tmp_path*."""
    return PaperTradingService(
        data_dir=str(tmp_path),
        initial_capital=initial_capital,
    )


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / "paper_trading" / "portfolio_state.json"


# ---------------------------------------------------------------------------
# Fresh start (no state file)
# ---------------------------------------------------------------------------

class TestFreshStart:
    """When no state file exists, a new Portfolio is created."""

    def test_new_portfolio_has_initial_capital(self, tmp_path: Path):
        svc = _svc(tmp_path, initial_capital=250_000)
        assert svc.portfolio.cash == 250_000

    def test_new_portfolio_has_no_positions(self, tmp_path: Path):
        svc = _svc(tmp_path)
        assert len(svc.portfolio.positions) == 0

    def test_new_portfolio_has_no_trades(self, tmp_path: Path):
        svc = _svc(tmp_path)
        assert len(svc.portfolio.trades) == 0

    def test_state_dir_created(self, tmp_path: Path):
        svc = _svc(tmp_path)
        assert (tmp_path / "paper_trading").is_dir()


# ---------------------------------------------------------------------------
# State persistence (buy -> reload -> verify)
# ---------------------------------------------------------------------------

class TestStatePersistence:
    """State survives creating a new service instance."""

    def test_buy_persists_across_instances(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=200, stop_loss=90.0, take_profit=120.0)

        # Create a fresh instance pointing at the same data dir
        svc2 = _svc(tmp_path)
        pos = svc2.portfolio.get_position("300750")
        assert pos is not None
        assert pos.shares == 200
        assert pos.avg_price == pytest.approx(100.0)
        assert pos.stop_loss == pytest.approx(90.0)
        assert pos.take_profit == pytest.approx(120.0)

    def test_cash_persists(self, tmp_path: Path):
        svc = _svc(tmp_path, initial_capital=100_000)
        svc.buy("300750", price=100.0, amount=100)
        cash_after_buy = svc.portfolio.cash

        svc2 = _svc(tmp_path)
        assert svc2.portfolio.cash == pytest.approx(cash_after_buy)

    def test_trades_persist(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=50.0, amount=100)
        svc.sell("300750", price=55.0, amount=100)

        svc2 = _svc(tmp_path)
        # Two trades: one buy + one sell
        assert len(svc2.portfolio.trades) == 2
        sell_trade = svc2.portfolio.trades[1]
        assert sell_trade.direction == "sell"
        assert sell_trade.status == "closed"

    def test_equity_curve_persists(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.portfolio.record_equity()
        svc.portfolio.record_equity()
        svc.save_state()

        svc2 = _svc(tmp_path)
        assert len(svc2.portfolio.equity_curve) == 2

    def test_commission_model_persists(self, tmp_path: Path):
        custom = CommissionModel(commission_rate=0.0005, stamp_tax_rate=0.002, min_commission=10.0)
        svc = PaperTradingService(
            data_dir=str(tmp_path),
            initial_capital=100_000,
            commission=custom,
        )
        svc.buy("300750", price=100.0, amount=100)

        svc2 = _svc(tmp_path)
        cm = svc2.portfolio.commission
        assert cm.commission_rate == pytest.approx(0.0005)
        assert cm.stamp_tax_rate == pytest.approx(0.002)
        assert cm.min_commission == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

class TestAtomicWrites:
    """State file must never be in a partial / corrupt state."""

    def test_state_file_is_valid_json(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=200)

        raw = _state_file(tmp_path).read_text(encoding="utf-8")
        data = json.loads(raw)  # must not raise
        assert data["cash"] == svc.portfolio.cash
        assert len(data["positions"]) == 1

    def test_no_leftover_temp_files(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)

        tmp_files = list((tmp_path / "paper_trading").glob("*.tmp"))
        assert len(tmp_files) == 0

    def test_state_file_has_version(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.save_state()

        data = json.loads(_state_file(tmp_path).read_text())
        assert "version" in data
        assert data["version"] >= 1


# ---------------------------------------------------------------------------
# Sell and state updates
# ---------------------------------------------------------------------------

class TestSellAndUpdate:
    """Sell operations and price updates modify state correctly."""

    def test_sell_removes_position(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=200)
        assert "300750" in svc.portfolio.positions

        svc.sell("300750", price=110.0, amount=200)
        assert "300750" not in svc.portfolio.positions

        # Persisted
        svc2 = _svc(tmp_path)
        assert "300750" not in svc2.portfolio.positions

    def test_partial_sell(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=300)
        svc.sell("300750", price=110.0, amount=100)

        pos = svc.portfolio.get_position("300750")
        assert pos is not None
        assert pos.shares == 200

    def test_sell_nonexistent_returns_none(self, tmp_path: Path):
        svc = _svc(tmp_path)
        result = svc.sell("000001", price=10.0, amount=100)
        assert result is None

    def test_update_price_does_not_auto_save(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)
        svc.update_price("300750", 120.0)

        # Load fresh -- price update was not persisted
        svc2 = _svc(tmp_path)
        pos = svc2.portfolio.get_position("300750")
        # current_price is whatever was saved during buy (100.0)
        assert pos.current_price == pytest.approx(100.0)

    def test_manual_save_after_price_update(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)
        svc.update_price("300750", 120.0)
        svc.save_state()

        svc2 = _svc(tmp_path)
        assert svc2.portfolio.get_position("300750").current_price == pytest.approx(120.0)

    def test_sell_increases_cash(self, tmp_path: Path):
        svc = _svc(tmp_path, initial_capital=100_000)
        svc.buy("300750", price=100.0, amount=100)
        cash_after_buy = svc.portfolio.cash

        svc.sell("300750", price=110.0, amount=100)
        # Cash after selling at 110 should be higher than after buying
        assert svc.portfolio.cash > cash_after_buy


# ---------------------------------------------------------------------------
# Multiple positions
# ---------------------------------------------------------------------------

class TestMultiplePositions:
    """Service handles multiple concurrent positions."""

    def test_multiple_buys(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)
        svc.buy("600519", price=50.0, amount=200)
        svc.buy("000858", price=30.0, amount=300)

        assert len(svc.portfolio.positions) == 3
        assert svc.portfolio.get_position("300750").shares == 100
        assert svc.portfolio.get_position("600519").shares == 200
        assert svc.portfolio.get_position("000858").shares == 300

    def test_multiple_positions_persist(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)
        svc.buy("600519", price=50.0, amount=200)

        svc2 = _svc(tmp_path)
        assert len(svc2.portfolio.positions) == 2
        assert svc2.portfolio.get_position("600519").shares == 200

    def test_sell_one_of_many(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)
        svc.buy("600519", price=50.0, amount=200)
        svc.sell("300750", price=110.0, amount=100)

        assert len(svc.portfolio.positions) == 1
        assert "600519" in svc.portfolio.positions

        svc2 = _svc(tmp_path)
        assert len(svc2.portfolio.positions) == 1
        assert "600519" in svc2.portfolio.positions

    def test_add_to_existing_position(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=100)
        svc.buy("300750", price=110.0, amount=100)

        pos = svc.portfolio.get_position("300750")
        assert pos.shares == 200
        # Average price should be between 100 and 110
        assert 100.0 < pos.avg_price < 110.0


# ---------------------------------------------------------------------------
# get_state_summary
# ---------------------------------------------------------------------------

class TestGetStateSummary:
    def test_empty_portfolio(self, tmp_path: Path):
        svc = _svc(tmp_path)
        summary = svc.get_state_summary()
        assert summary["cash"] == 100_000
        assert summary["positions"] == []
        assert summary["total_equity"] == 100_000
        assert summary["unrealized_pnl"] == 0.0
        assert summary["trade_count"] == 0

    def test_with_positions(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=200)
        svc.update_price("300750", 110.0)

        summary = svc.get_state_summary()
        assert len(summary["positions"]) == 1
        pos_info = summary["positions"][0]
        assert pos_info["code"] == "300750"
        assert pos_info["shares"] == 200
        assert pos_info["pnl"] == summary["unrealized_pnl"]

    def test_summary_after_sell(self, tmp_path: Path):
        svc = _svc(tmp_path)
        svc.buy("300750", price=100.0, amount=200)
        svc.sell("300750", price=110.0, amount=200)

        summary = svc.get_state_summary()
        assert summary["positions"] == []
        assert summary["closed_trades"] == 1
        assert summary["trade_count"] == 2


# ---------------------------------------------------------------------------
# Corrupted state file fallback
# ---------------------------------------------------------------------------

class TestCorruptedState:
    """A corrupted state file should not crash the service."""

    def test_corrupted_json_creates_fresh_portfolio(self, tmp_path: Path):
        state_dir = tmp_path / "paper_trading"
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / "portfolio_state.json").write_text("NOT VALID JSON{{{")

        svc = _svc(tmp_path)
        assert svc.portfolio.cash == 100_000
        assert len(svc.portfolio.positions) == 0
