"""DataService 单元测试"""

import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np


@pytest.fixture
def sample_price_df():
    """生成测试用行情数据"""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    close = 100 + np.cumsum(np.random.randn(100) * 2)
    return pd.DataFrame({
        "date": dates.strftime("%Y%m%d"),
        "open": close - np.random.rand(100),
        "high": close + np.random.rand(100) * 2,
        "low": close - np.random.rand(100) * 2,
        "close": close,
        "volume": np.random.randint(100000, 1000000, 100).astype(float),
    })


@pytest.fixture
def mock_settings(tmp_path):
    settings = MagicMock()
    settings.parquet_dir = str(tmp_path / "parquet")
    settings.tushare_token = None
    settings.akshare_timeout = 5
    settings.debug = True
    return settings


class TestNormalizer:
    """数据标准化测试"""

    def test_normalize_columns_chinese(self, sample_price_df):
        from quant_agent.data.normalizer import normalize_columns
        df = sample_price_df.copy()
        df = normalize_columns(df)
        assert "close" in df.columns

    def test_normalize_price_data(self, sample_price_df):
        from quant_agent.data.normalizer import normalize_price_data
        df = normalize_price_data(sample_price_df)
        assert "date" in df.columns
        assert "close" in df.columns
        assert len(df) == 100


class TestValidator:
    """数据校验测试"""

    def test_valid_data(self, sample_price_df):
        from quant_agent.data.normalizer import normalize_price_data
        from quant_agent.data.validator import validate_price_data
        df = normalize_price_data(sample_price_df)
        report = validate_price_data(df)
        assert report.is_valid

    def test_empty_data(self):
        from quant_agent.data.validator import validate_price_data
        report = validate_price_data(pd.DataFrame())
        assert not report.is_valid

    def test_clean_stale_rows(self):
        from quant_agent.data.validator import clean_price_data
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "close": [100.0, 100.0, 101.0],
            "volume": [1000.0, 0.0, 1000.0],
        })
        cleaned = clean_price_data(df, remove_stale=True)
        assert len(cleaned) == 2

    def test_validation_report_summary(self, sample_price_df):
        from quant_agent.data.normalizer import normalize_price_data
        from quant_agent.data.validator import validate_price_data
        df = normalize_price_data(sample_price_df)
        report = validate_price_data(df)
        assert "PASS" in report.summary()


class TestFinancialSnapshot:
    """FinancialSnapshot 测试"""

    def test_basic_access(self):
        from quant_agent.data.sources.base import FinancialSnapshot
        snap = FinancialSnapshot("300750", {"roe": 0.18, "pe_ttm": 25.0, "pb": 5.6})
        assert snap.roe == 0.18
        assert snap.pe_ttm == 25.0
        assert snap.gross_margin is None

    def test_to_dict(self):
        from quant_agent.data.sources.base import FinancialSnapshot
        snap = FinancialSnapshot("300750", {"roe": 0.18})
        d = snap.to_dict()
        assert d["stock_code"] == "300750"
        assert d["roe"] == 0.18

    def test_repr(self):
        from quant_agent.data.sources.base import FinancialSnapshot
        snap = FinancialSnapshot("300750", {"roe": 0.18})
        assert "300750" in repr(snap)
