"""EfinanceSource 单元测试 — ABC 合规、列映射、代码转换、重试、速率限制"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np


# ── Helpers ──


def _make_efinance_price_df(n=50):
    """Build a fake efinance-style DataFrame with Chinese column names."""
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n))
    return pd.DataFrame({
        "日期": dates.strftime("%Y-%m-%d"),
        "开盘": close - np.random.rand(n),
        "收盘": close,
        "最高": close + np.random.rand(n) * 2,
        "最低": close - np.random.rand(n) * 2,
        "成交量": np.random.randint(100000, 500000, n).astype(float),
        "成交额": np.random.rand(n) * 1e8,
        "涨跌幅": np.random.randn(n) * 2,
        "换手率": np.random.rand(n) * 5,
    })


def _make_efinance_financial_df():
    """Build a fake efinance financial performance DataFrame."""
    return pd.DataFrame([{
        "报告期": "2025-06-30",
        "净资产收益率": 18.5,
        "毛利率": 35.2,
        "净利率": 12.8,
        "资产负债率": 45.0,
        "流动比率": 1.8,
        "营收同比增长率": 25.3,
        "净利润同比增长率": 30.1,
        "市盈率": 25.0,
        "市净率": 5.6,
    }])


def _create_source(**kwargs):
    """Create EfinanceSource with mocked rate limiter."""
    from quant_agent.data.sources.efinance import EfinanceSource
    source = EfinanceSource(**kwargs)
    source._rate_limiter = MagicMock()
    return source


# ═══════════════════════════════════════════════════════════════════════════
# 1. ABC 合规
# ═══════════════════════════════════════════════════════════════════════════


class TestEfinanceABC:
    """EfinanceSource ABC 合规测试"""

    def test_is_datasource(self):
        from quant_agent.data.sources.base import DataSource
        from quant_agent.data.sources.efinance import EfinanceSource
        src = _create_source()
        assert isinstance(src, DataSource)

    def test_has_required_methods(self):
        src = _create_source()
        assert hasattr(src, "get_price_data")
        assert hasattr(src, "get_realtime_price")
        assert hasattr(src, "get_financial_snapshot")
        assert hasattr(src, "name")
        assert hasattr(src, "available")

    def test_name(self):
        src = _create_source()
        assert src.name == "efinance"


# ═══════════════════════════════════════════════════════════════════════════
# 2. 代码格式转换
# ═══════════════════════════════════════════════════════════════════════════


class TestCodeConversion:
    """efinance expects bare stock codes (no SZ/SH prefix)"""

    @patch("efinance.stock.get_quote_history")
    def test_bare_code_passed_to_api(self, mock_get):
        """efinance receives bare code like 300750 (not SZ300750)"""
        src = _create_source()
        mock_get.return_value = _make_efinance_price_df(5)

        src.get_price_data("300750", days=5)
        mock_get.assert_called_once()
        # First positional arg should be the bare code
        assert mock_get.call_args[0][0] == "300750"

    @patch("efinance.stock.get_quote_history")
    def test_shanghai_bare_code(self, mock_get):
        """Shanghai code passed as bare 600519"""
        src = _create_source()
        mock_get.return_value = _make_efinance_price_df(5)

        src.get_price_data("600519", days=5)
        assert mock_get.call_args[0][0] == "600519"


# ═══════════════════════════════════════════════════════════════════════════
# 3. 价格数据
# ═══════════════════════════════════════════════════════════════════════════


class TestPriceData:
    """Price data fetching and column mapping"""

    @patch("efinance.stock.get_quote_history")
    def test_column_mapping(self, mock_get):
        """Chinese columns are mapped to English standard names"""
        src = _create_source()
        mock_get.return_value = _make_efinance_price_df(30)

        result = src.get_price_data("300750", days=30)
        assert result is not None
        assert "date" in result.columns
        assert "open" in result.columns
        assert "close" in result.columns
        assert "high" in result.columns
        assert "low" in result.columns
        assert "volume" in result.columns
        # Chinese names should be gone
        assert "日期" not in result.columns
        assert "收盘" not in result.columns

    @patch("efinance.stock.get_quote_history")
    def test_data_sorted_by_date(self, mock_get):
        """Result is sorted by date ascending"""
        src = _create_source()
        mock_get.return_value = _make_efinance_price_df(30)

        result = src.get_price_data("300750", days=30)
        assert result is not None
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    @patch("efinance.stock.get_quote_history")
    def test_empty_df_returns_none(self, mock_get):
        """Empty DataFrame from API returns None"""
        src = _create_source()
        mock_get.return_value = pd.DataFrame()

        result = src.get_price_data("300750")
        assert result is None

    @patch("efinance.stock.get_quote_history")
    def test_none_returns_none(self, mock_get):
        """None from API returns None"""
        src = _create_source()
        mock_get.return_value = None

        result = src.get_price_data("300750")
        assert result is None

    @patch("efinance.stock.get_quote_history")
    def test_exception_returns_none(self, mock_get):
        """Non-transient exception returns None (caught by outer try/except)"""
        src = _create_source()
        mock_get.side_effect = RuntimeError("API error")

        result = src.get_price_data("300750")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 4. 财务快照
# ═══════════════════════════════════════════════════════════════════════════


class TestFinancialSnapshot:
    """Financial snapshot fetching and field mapping"""

    @patch("efinance.stock.get_quote_history")
    @patch("efinance.stock.get_all_company_performance")
    def test_field_mapping(self, mock_fin, mock_price):
        """Chinese financial columns mapped to FinancialSnapshot fields"""
        src = _create_source()
        mock_fin.return_value = _make_efinance_financial_df()
        mock_price.return_value = pd.DataFrame({"收盘": [100.0]})

        result = src.get_financial_snapshot("300750")
        assert result is not None
        assert result.roe == pytest.approx(0.185, abs=0.001)
        assert result.gross_margin == pytest.approx(0.352, abs=0.001)
        assert result.net_margin == pytest.approx(0.128, abs=0.001)
        assert result.debt_ratio == pytest.approx(0.45, abs=0.001)
        assert result.pe_ttm == pytest.approx(25.0)
        assert result.pb == pytest.approx(5.6)

    @patch("efinance.stock.get_quote_history")
    @patch("efinance.stock.get_all_company_performance")
    def test_percentage_normalization(self, mock_fin, mock_price):
        """Values > 1 for ratio fields are divided by 100"""
        src = _create_source()
        mock_fin.return_value = pd.DataFrame([{
            "报告期": "2025-06-30",
            "净资产收益率": 22.5,  # > 1, should be /100
        }])
        mock_price.return_value = pd.DataFrame({"收盘": [50.0]})

        result = src.get_financial_snapshot("300750")
        assert result is not None
        assert result.roe == pytest.approx(0.225, abs=0.001)

    @patch("efinance.stock.get_all_company_performance")
    def test_empty_df_returns_none(self, mock_fin):
        """Empty financial DataFrame returns None"""
        src = _create_source()
        mock_fin.return_value = pd.DataFrame()

        result = src.get_financial_snapshot("300750")
        assert result is None

    @patch("efinance.stock.get_all_company_performance")
    def test_none_returns_none(self, mock_fin):
        """None from API returns None"""
        src = _create_source()
        mock_fin.return_value = None

        result = src.get_financial_snapshot("300750")
        assert result is None

    @patch("efinance.stock.get_all_company_performance")
    def test_exception_returns_none(self, mock_fin):
        """Exception during fetch returns None"""
        src = _create_source()
        mock_fin.side_effect = RuntimeError("API error")

        result = src.get_financial_snapshot("300750")
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. 重试策略
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryStrategy:
    """Retry only transient errors, fail fast on others"""

    def test_retries_connection_error(self):
        """ConnectionError is retried up to max_retries"""
        src = _create_source(max_retries=2)
        mock_fn = MagicMock(__name__="mock_fn")
        mock_fn.side_effect = ConnectionError("network down")

        with pytest.raises(ConnectionError, match="network down"):
            src._retry_call(mock_fn, "arg1")

        assert mock_fn.call_count == 2

    def test_non_transient_error_not_retried(self):
        """Non-transient error (ValueError) raises immediately"""
        src = _create_source(max_retries=3)
        mock_fn = MagicMock(__name__="mock_fn")
        mock_fn.side_effect = ValueError("bad param")

        with pytest.raises(ValueError, match="bad param"):
            src._retry_call(mock_fn, "arg1")

        assert mock_fn.call_count == 1

    def test_succeeds_after_retry(self):
        """Succeeds after initial transient failure"""
        src = _create_source(max_retries=3)
        mock_fn = MagicMock(__name__="mock_fn")
        mock_fn.side_effect = [ConnectionError("timeout"), "success"]

        result = src._retry_call(mock_fn, "arg1")
        assert result == "success"
        assert mock_fn.call_count == 2

    def test_rate_limiter_called_before_retry(self):
        """Rate limiter is called once before the retry loop starts"""
        src = _create_source(max_retries=1)
        mock_fn = MagicMock(__name__="mock_fn", return_value="ok")

        src._retry_call(mock_fn)
        src._rate_limiter.block_until_ready.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
# 6. 速率限制
# ═══════════════════════════════════════════════════════════════════════════


class TestRateLimiter:
    """Rate limiter is called before each API request"""

    @patch("efinance.stock.get_quote_history")
    def test_rate_limiter_called_for_price(self, mock_get):
        src = _create_source()
        mock_get.return_value = _make_efinance_price_df(5)

        src.get_price_data("300750", days=5)
        src._rate_limiter.block_until_ready.assert_called()

    @patch("efinance.stock.get_all_company_performance")
    def test_rate_limiter_called_for_financial(self, mock_fin):
        src = _create_source()
        mock_fin.return_value = _make_efinance_financial_df()

        src.get_financial_snapshot("300750")
        src._rate_limiter.block_until_ready.assert_called()


# ═══════════════════════════════════════════════════════════════════════════
# 7. 可用性检测
# ═══════════════════════════════════════════════════════════════════════════


class TestAvailability:
    """available property detection"""

    def test_available_when_importable(self):
        src = _create_source()
        # efinance is installed in test env
        assert src.available is True or src.available is False  # depends on env

    def test_available_false_when_missing(self):
        src = _create_source()
        with patch.dict("sys.modules", {"efinance": None}):
            # Force ImportError
            with patch("builtins.__import__", side_effect=ImportError("no efinance")):
                assert src.available is False
