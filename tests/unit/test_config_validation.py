"""配置校验测试 — 边界值校验"""

import pytest
from pydantic import ValidationError

from quant_agent.config import Settings


def _make_settings(**kwargs):
    """Create Settings without reading .env file (avoid secrets leaking into tests)."""
    return Settings.model_validate(kwargs)


class TestConfigValidation:
    def test_valid_defaults(self):
        """Default settings should be valid."""
        s = _make_settings()
        assert 0 < s.max_position_pct <= 1
        assert s.default_stop_loss < 0
        assert s.commission_rate >= 0

    def test_invalid_max_position_zero(self):
        with pytest.raises(ValidationError):
            _make_settings(max_position_pct=0)

    def test_invalid_max_position_over_one(self):
        with pytest.raises(ValidationError):
            _make_settings(max_position_pct=1.5)

    def test_invalid_stop_loss_positive(self):
        with pytest.raises(ValidationError):
            _make_settings(default_stop_loss=0.05)

    def test_invalid_stop_loss_zero(self):
        with pytest.raises(ValidationError):
            _make_settings(default_stop_loss=0)

    def test_invalid_commission_negative(self):
        with pytest.raises(ValidationError):
            _make_settings(commission_rate=-0.001)

    def test_invalid_max_portfolio_risk(self):
        with pytest.raises(ValidationError):
            _make_settings(max_portfolio_risk=0)

    def test_valid_extreme_values(self):
        """Valid boundary values should be accepted."""
        s = _make_settings(
            max_position_pct=1.0,
            default_stop_loss=-0.99,
            commission_rate=0.0,
        )
        assert s.max_position_pct == 1.0
        assert s.default_stop_loss == -0.99
        assert s.commission_rate == 0.0


class TestInitialCapitalValidation:
    """Boundary validation for initial_capital."""

    def test_valid_default_capital(self):
        s = _make_settings()
        assert s.initial_capital == 100000.0

    def test_valid_small_capital(self):
        s = _make_settings(initial_capital=1.0)
        assert s.initial_capital == 1.0

    def test_valid_large_capital(self):
        s = _make_settings(initial_capital=1e9)
        assert s.initial_capital == 1e9

    def test_zero_capital_rejected(self):
        with pytest.raises(ValidationError, match="initial_capital"):
            _make_settings(initial_capital=0)

    def test_negative_capital_rejected(self):
        with pytest.raises(ValidationError, match="initial_capital"):
            _make_settings(initial_capital=-100000)

    def test_tiny_positive_capital_accepted(self):
        s = _make_settings(initial_capital=0.01)
        assert s.initial_capital == 0.01
