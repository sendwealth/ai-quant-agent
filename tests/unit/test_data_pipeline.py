"""数据管道集成测试 -- 降级链路、normalizer 边界、stock code 验证、持久化"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock
import pandas as pd
import numpy as np


# ── Fixtures ──


@pytest.fixture
def sample_price_df():
    """完整的行情数据（包含所有 REQUIRED_COLUMNS）"""
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
    settings.fetch_max_workers = 1  # 顺序执行（保持与原有测试行为一致）
    settings.debug = True
    return settings


def _make_source(name: str, price_data=None, realtime_price=None, financial_snapshot=None):
    """创建 mock DataSource，使用 PropertyMock 确保 name 属性兼容"""
    source = MagicMock()
    type(source).name = PropertyMock(return_value=name)
    source.available = True
    source.get_price_data.return_value = price_data
    source.get_realtime_price.return_value = realtime_price
    source.get_financial_snapshot.return_value = financial_snapshot
    return source


def _build_service(mock_settings, sources, use_real_store=False):
    """构建 DataService 实例（绕过 __init__ 避免连接真实 API）

    Args:
        mock_settings: 配置 mock
        sources: 数据源列表
        use_real_store: 是否使用真实 DataStore（用于持久化集成测试）
    """
    from quant_agent.data.service import DataService

    ds = DataService.__new__(DataService)
    ds.settings = mock_settings

    if use_real_store:
        from quant_agent.data.store import DataStore
        ds.store = DataStore(mock_settings.parquet_dir)
    else:
        ds.store = MagicMock()
        ds.store.is_fresh.return_value = False

    ds._sources = sources
    return ds


# ═══════════════════════════════════════════════════════════════════════════
# 1. DataService 降级链路
# ═══════════════════════════════════════════════════════════════════════════


class TestDataServiceFallback:
    """数据管道降级链路测试"""

    # -- 行情数据降级 -------------------------------------------------------

    def test_first_source_succeeds(self, mock_settings, sample_price_df):
        """第一个数据源成功时，不调用第二个"""
        src1 = _make_source("source1", price_data=sample_price_df)
        src2 = _make_source("source2")
        ds = _build_service(mock_settings, [src1, src2])

        result = ds.get_price_data("300750", use_cache=False, clean=False)
        assert result is not None
        src1.get_price_data.assert_called_once()
        src2.get_price_data.assert_not_called()

    def test_first_source_fails_second_succeeds(self, mock_settings, sample_price_df):
        """第一个数据源失败时，降级到第二个"""
        src1 = _make_source("source1", price_data=None)
        src2 = _make_source("source2", price_data=sample_price_df)
        ds = _build_service(mock_settings, [src1, src2])

        result = ds.get_price_data("300750", use_cache=False, clean=False)
        assert result is not None
        src1.get_price_data.assert_called_once()
        src2.get_price_data.assert_called_once()

    def test_all_sources_fail(self, mock_settings):
        """所有数据源失败时，返回 None"""
        src1 = _make_source("source1", price_data=None)
        src2 = _make_source("source2", price_data=None)
        ds = _build_service(mock_settings, [src1, src2])

        result = ds.get_price_data("300750", use_cache=False)
        assert result is None

    def test_source_returns_empty_df(self, mock_settings, sample_price_df):
        """空 DataFrame 视为失败，降级到下一个数据源"""
        src1 = _make_source("source1", price_data=pd.DataFrame())
        src2 = _make_source("source2", price_data=sample_price_df)
        ds = _build_service(mock_settings, [src1, src2])

        result = ds.get_price_data("300750", use_cache=False, clean=False)
        assert result is not None
        src1.get_price_data.assert_called_once()
        src2.get_price_data.assert_called_once()

    def test_source_returns_invalid_data(self, mock_settings, sample_price_df):
        """数据源返回校验不通过的数据（高空值比例）时，降级到下一个数据源

        validator 检测到空值比例超过阈值 -> is_valid=False -> service
        继续尝试下一个 source。
        """
        # 构建高空值数据: 20 行中有 15 行 close/volume 为 NaN -> 75% nulls
        n = 20
        bad_df = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y%m%d"),
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [99.0] * n,
            "close": [np.nan] * 15 + [100.0] * 5,  # 75% nulls
            "volume": [np.nan] * 15 + [5000.0] * 5,
        })

        src1 = _make_source("source1", price_data=bad_df)
        src2 = _make_source("source2", price_data=sample_price_df)
        ds = _build_service(mock_settings, [src1, src2])

        result = ds.get_price_data("300750", use_cache=False, clean=False)
        assert result is not None
        src1.get_price_data.assert_called_once()
        src2.get_price_data.assert_called_once()

    def test_cache_hit_skips_sources(self, mock_settings, sample_price_df):
        """缓存命中时不调用数据源"""
        ds = _build_service(mock_settings, [])
        ds.store.is_fresh.return_value = True
        ds.store.load_price.return_value = sample_price_df

        src1 = _make_source("source1")
        ds._sources = [src1]

        # days=80 ensures cached data (100 rows) >= 80*0.8=64
        result = ds.get_price_data("300750", days=80, use_cache=True)
        assert result is not None
        src1.get_price_data.assert_not_called()

    # -- 实时价格降级 -------------------------------------------------------

    def test_realtime_price_fallback(self, mock_settings):
        """实时价格降级测试"""
        src1 = _make_source("source1", realtime_price=None)
        src2 = _make_source("source2", realtime_price=150.0)
        ds = _build_service(mock_settings, [src1, src2])

        result = ds.get_realtime_price("300750")
        assert result == 150.0

    def test_realtime_price_first_succeeds(self, mock_settings):
        """第一个数据源返回有效实时价格时，第二个不被调用"""
        src1 = _make_source("source1", realtime_price=250.50)
        src2 = _make_source("source2", realtime_price=251.00)
        ds = _build_service(mock_settings, [src1, src2])

        price = ds.get_realtime_price("601318")
        assert price == 250.50
        src1.get_realtime_price.assert_called_once_with("601318")
        src2.get_realtime_price.assert_not_called()

    def test_realtime_price_all_fail(self, mock_settings):
        """所有数据源实时价格均失败 -> 返回 None"""
        src1 = _make_source("source1", realtime_price=None)
        src2 = _make_source("source2", realtime_price=None)
        ds = _build_service(mock_settings, [src1, src2])

        price = ds.get_realtime_price("601318")
        assert price is None

    def test_realtime_price_zero_treated_as_failure(self, mock_settings):
        """实时价格为 0 视为无效（service 要求 price > 0），触发降级"""
        src1 = _make_source("source1", realtime_price=0)
        src2 = _make_source("source2", realtime_price=100.0)
        ds = _build_service(mock_settings, [src1, src2])

        price = ds.get_realtime_price("601318")
        assert price == 100.0
        src2.get_realtime_price.assert_called_once()

    # -- 财务快照降级 -------------------------------------------------------

    def test_financial_snapshot_no_sources(self, mock_settings):
        """无可用数据源且无缓存时，返回 None"""
        ds = _build_service(mock_settings, [])
        ds.store.load_financial.return_value = None

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=None):
            result = ds.get_financial_snapshot("300750")
        assert result is None

    def test_financial_snapshot_tushare_succeeds(self, mock_settings):
        """Tushare 返回财务快照时，数据被持久化并返回"""
        from quant_agent.data.sources.base import FinancialSnapshot

        snapshot = FinancialSnapshot("300750", {"roe": 0.18, "pe_ttm": 25.0, "pb": 5.6})

        ds = _build_service(mock_settings, [])
        mock_tushare = MagicMock()
        mock_tushare.available = True
        mock_tushare.get_financial_snapshot.return_value = snapshot

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=mock_tushare):
            result = ds.get_financial_snapshot("300750")

        assert result is not None
        assert result.roe == 0.18
        assert result.pe_ttm == 25.0
        ds.store.save_financial.assert_called_once()

    def test_financial_snapshot_tushare_fails_fallback_to_cache(self, mock_settings):
        """Tushare 不可用时，降级到本地缓存"""
        from quant_agent.data.sources.base import FinancialSnapshot

        cached_df = pd.DataFrame([{
            "roe": 0.15, "pe_ttm": 20.0, "pb": 4.0, "report_date": "2025-06-30",
        }])

        ds = _build_service(mock_settings, [])
        mock_tushare = MagicMock()
        mock_tushare.available = False
        ds.store.load_financial.return_value = cached_df

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=mock_tushare):
            result = ds.get_financial_snapshot("300750")

        assert result is not None
        assert result.roe == 0.15
        assert result.pe_ttm == 20.0
        ds.store.load_financial.assert_called_once_with("300750", latest=True)

    def test_financial_snapshot_all_fail(self, mock_settings):
        """Tushare 返回 None 且缓存也为空 -> 返回 None"""
        ds = _build_service(mock_settings, [])
        mock_tushare = MagicMock()
        mock_tushare.available = True
        mock_tushare.get_financial_snapshot.return_value = None
        ds.store.load_financial.return_value = None

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=mock_tushare):
            result = ds.get_financial_snapshot("300750")

        assert result is None

    def test_financial_snapshot_cache_returns_empty_df(self, mock_settings):
        """缓存返回空 DataFrame -> 视为无数据 -> 返回 None"""
        ds = _build_service(mock_settings, [])
        mock_tushare = MagicMock()
        mock_tushare.available = False
        ds.store.load_financial.return_value = pd.DataFrame()

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=mock_tushare):
            result = ds.get_financial_snapshot("300750")

        assert result is None

    def test_financial_staleness_rejected(self, mock_settings):
        """过期财务数据（超过 max_age_days）被拒绝，返回 None"""
        from datetime import datetime, timedelta

        ds = _build_service(mock_settings, [])

        old_date = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
        ds.store.load_financial.return_value = pd.DataFrame([{
            "roe": 0.18,
            "report_date": old_date,
        }])

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=None):
            result = ds.get_financial_snapshot("300750", max_age_days=365)
        assert result is None

    # -- 批量操作 -----------------------------------------------------------

    def test_get_multi_price_partial_success(self, mock_settings, sample_price_df):
        """批量获取行情：部分成功只返回有效结果"""
        src1 = _make_source("source1", price_data=sample_price_df)
        ds = _build_service(mock_settings, [src1])

        src1.get_price_data.side_effect = lambda code, days=250: (
            sample_price_df.copy() if code == "300750" else None
        )

        results = ds.get_multi_price(["300750", "601318"])
        assert "300750" in results
        assert "601318" not in results

    def test_get_multi_financial_partial_success(self, mock_settings):
        """批量获取财务快照：部分成功只返回有效结果"""
        from quant_agent.data.sources.base import FinancialSnapshot

        ds = _build_service(mock_settings, [])
        snap_300750 = FinancialSnapshot("300750", {"roe": 0.18})
        mock_tushare = MagicMock()
        mock_tushare.available = True
        mock_tushare.get_financial_snapshot.side_effect = lambda code: (
            snap_300750 if code == "300750" else None
        )
        ds.store.load_financial.return_value = None

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=mock_tushare):
            results = ds.get_multi_financial(["300750", "601318"])

        assert "300750" in results
        assert "601318" not in results


# ═══════════════════════════════════════════════════════════════════════════
# 2. Normalizer 边界测试
# ═══════════════════════════════════════════════════════════════════════════


class TestNormalizerEdgeCases:
    """normalizer 边界条件测试"""

    def test_missing_required_columns_raises(self):
        """缺少必要列应抛出 ValueError"""
        from quant_agent.data.normalizer import normalize_price_data
        df = pd.DataFrame({"date": ["2025-01-01"], "close": [100.0]})
        with pytest.raises(ValueError, match="缺少必要列"):
            normalize_price_data(df)

    def test_all_columns_present_no_error(self, sample_price_df):
        """完整数据不应抛异常"""
        from quant_agent.data.normalizer import normalize_price_data
        result = normalize_price_data(sample_price_df)
        assert "date" in result.columns
        assert "close" in result.columns
        assert len(result) == 100

    def test_empty_dataframe_raises(self):
        """空 DataFrame 缺少必要列"""
        from quant_agent.data.normalizer import normalize_price_data
        with pytest.raises(ValueError, match="缺少必要列"):
            normalize_price_data(pd.DataFrame())

    def test_partial_columns_raises(self):
        """只有部分列应抛异常"""
        from quant_agent.data.normalizer import normalize_price_data
        df = pd.DataFrame({
            "date": ["2025-01-01"],
            "close": [100.0],
            "open": [99.0],
        })
        with pytest.raises(ValueError, match="缺少必要列"):
            normalize_price_data(df)

    def test_chinese_column_names_normalized(self):
        """中文列名映射为英文"""
        from quant_agent.data.normalizer import normalize_price_data

        df = pd.DataFrame({
            "日期": ["20250101", "20250102", "20250103"],
            "开盘": [100.0, 101.0, 102.0],
            "收盘": [101.0, 102.0, 103.0],
            "最高": [102.0, 103.0, 104.0],
            "最低": [99.0, 100.0, 101.0],
            "成交量": [10000.0, 11000.0, 12000.0],
        })
        result = normalize_price_data(df)
        assert list(result.columns) == ["date", "open", "close", "high", "low", "volume"]

    def test_tushare_column_names_normalized(self):
        """Tushare 风格列名（trade_date, vol）被正确映射"""
        from quant_agent.data.normalizer import normalize_price_data

        df = pd.DataFrame({
            "trade_date": ["20250101", "20250102"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "vol": [5000.0, 6000.0],
        })
        result = normalize_price_data(df)
        assert "date" in result.columns
        assert "volume" in result.columns
        assert "trade_date" not in result.columns

    def test_single_row_dataframe(self):
        """单行 DataFrame 应正常通过"""
        from quant_agent.data.normalizer import normalize_price_data

        df = pd.DataFrame({
            "date": ["20250101"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [5000.0],
        })
        result = normalize_price_data(df)
        assert len(result) == 1
        assert result["close"].iloc[0] == 103.0

    def test_date_sorting_after_normalization(self):
        """标准化后按日期升序排列"""
        from quant_agent.data.normalizer import normalize_price_data

        df = pd.DataFrame({
            "date": ["20250103", "20250101", "20250102"],
            "open": [102.0, 100.0, 101.0],
            "high": [107.0, 105.0, 106.0],
            "low": [101.0, 99.0, 100.0],
            "close": [105.0, 103.0, 104.0],
            "volume": [7000.0, 5000.0, 6000.0],
        })
        result = normalize_price_data(df)
        dates = result["date"].tolist()
        assert dates == sorted(dates)

    def test_nan_values_preserved(self):
        """NaN 值在数值列中被保留（不丢弃）"""
        from quant_agent.data.normalizer import normalize_price_data

        df = pd.DataFrame({
            "date": ["20250101", "20250102"],
            "open": [100.0, np.nan],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [5000.0, 6000.0],
        })
        result = normalize_price_data(df)
        assert result["open"].isna().sum() == 1

    def test_missing_only_open_raises(self):
        """缺少 open 列（其余都有）仍应报错"""
        from quant_agent.data.normalizer import normalize_price_data

        df = pd.DataFrame({
            "date": ["20250101"],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [5000.0],
        })
        with pytest.raises(ValueError, match="缺少必要列"):
            normalize_price_data(df)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Stock Code 验证
# ═══════════════════════════════════════════════════════════════════════════


class TestStockCodeValidation:
    """stock code 输入验证测试"""

    def test_valid_shanghai(self):
        from quant_agent.data.validators import validate_stock_code
        assert validate_stock_code("601318") == "601318"

    def test_valid_shenzhen_main(self):
        from quant_agent.data.validators import validate_stock_code
        assert validate_stock_code("000001") == "000001"

    def test_valid_chinext(self):
        from quant_agent.data.validators import validate_stock_code
        assert validate_stock_code("300750") == "300750"

    def test_valid_beijing(self):
        from quant_agent.data.validators import validate_stock_code
        assert validate_stock_code("830799") == "830799"

    def test_whitespace_stripped(self):
        from quant_agent.data.validators import validate_stock_code
        assert validate_stock_code("  300750  ") == "300750"

    def test_invalid_prefix(self):
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError, match="prefix"):
            validate_stock_code("123456")

    def test_invalid_prefix_40(self):
        """前缀 '40' 不是有效 A 股前缀"""
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError, match="prefix"):
            validate_stock_code("400001")

    def test_too_short(self):
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError, match="6 digits"):
            validate_stock_code("30075")

    def test_too_long(self):
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError, match="6 digits"):
            validate_stock_code("3007501")

    def test_non_numeric(self):
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError, match="digits"):
            validate_stock_code("30075A")

    def test_none_raises(self):
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError, match="None"):
            validate_stock_code(None)

    def test_empty_string_raises(self):
        from quant_agent.data.validators import validate_stock_code
        with pytest.raises(ValueError):
            validate_stock_code("")

    def test_integer_input_converted(self):
        """整数输入会被转为字符串并验证"""
        from quant_agent.data.validators import validate_stock_code
        assert validate_stock_code(300750) == "300750"


# ═══════════════════════════════════════════════════════════════════════════
# 4. DataStore 持久化集成测试
# ═══════════════════════════════════════════════════════════════════════════


class TestDataStoreIntegration:
    """DataStore 持久化层集成测试"""

    def test_save_and_load_price(self, tmp_path):
        """行情数据保存 -> 加载 循环"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [5000.0, 6000.0],
        })
        store.save_price("300750", df, source="test")
        loaded = store.load_price("300750")

        assert loaded is not None
        assert len(loaded) == 2
        pd.testing.assert_frame_equal(df, loaded)

    def test_load_nonexistent_returns_none(self, tmp_path):
        """加载不存在的股票代码返回 None"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        assert store.load_price("999999") is None

    def test_save_empty_df_returns_empty_path(self, tmp_path):
        """保存空 DataFrame 返回空路径（不应创建文件）"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        path = store.save_price("300750", pd.DataFrame(), source="test")
        assert str(path) == "."

    def test_save_and_load_financial(self, tmp_path):
        """财务快照保存 -> 加载 循环"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        data = {"roe": 0.18, "pe_ttm": 25.0, "report_date": "2025-06-30"}
        store.save_financial("300750", data, source="test")
        loaded = store.load_financial("300750", latest=True)

        assert loaded is not None
        assert loaded.iloc[0]["roe"] == 0.18

    def test_financial_append(self, tmp_path):
        """多次保存同一股票的财务数据会追加行"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        store.save_financial("300750", {"roe": 0.15, "report_date": "2024-12-31"}, source="test")
        store.save_financial("300750", {"roe": 0.18, "report_date": "2025-06-30"}, source="test")

        loaded = store.load_financial("300750", latest=True)
        assert loaded is not None
        # latest=True + descending by report_date -> only the newest row
        assert len(loaded) == 1
        assert loaded.iloc[0]["roe"] == 0.18

    def test_is_fresh_nonexistent_returns_false(self, tmp_path):
        """不存在的文件返回 False"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        assert store.is_fresh("300750") is False

    def test_is_fresh_recent_data_returns_true(self, tmp_path):
        """刚保存的数据应被视为新鲜"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        df = pd.DataFrame({
            "date": ["2025-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [5000.0],
        })
        store.save_price("300750", df, source="test")
        assert store.is_fresh("300750") is True

    def test_list_stocks(self, tmp_path):
        """list_stocks 返回所有已保存的股票代码"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        df = pd.DataFrame({
            "date": ["2025-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [103.0],
            "volume": [5000.0],
        })
        store.save_price("300750", df, source="test")
        store.save_price("601318", df, source="test")

        stocks = store.list_stocks()
        assert sorted(stocks) == ["300750", "601318"]

    def test_load_financial_nonexistent_returns_none(self, tmp_path):
        """加载不存在的财务数据返回 None"""
        from quant_agent.data.store import DataStore

        store = DataStore(base_dir=str(tmp_path / "parquet"))
        assert store.load_financial("999999") is None


# ═══════════════════════════════════════════════════════════════════════════
# 5. 端到端: DataService + 真实 DataStore
# ═══════════════════════════════════════════════════════════════════════════


class TestDataServiceWithRealStore:
    """DataService 配合真实 DataStore 的端到端测试"""

    def test_price_data_persisted_to_store(self, mock_settings, sample_price_df):
        """从数据源获取后，数据被持久化到真实存储"""
        src = _make_source("mock_source", price_data=sample_price_df)
        ds = _build_service(mock_settings, [src], use_real_store=True)

        result = ds.get_price_data("300750", use_cache=False, clean=False)
        assert result is not None

        # 验证数据已持久化到磁盘
        loaded = ds.store.load_price("300750")
        assert loaded is not None
        assert len(loaded) > 0

    def test_cached_data_returned_when_fresh(self, mock_settings, sample_price_df):
        """缓存新鲜时，第二次请求不调用数据源"""
        src = _make_source("mock_source", price_data=sample_price_df)
        ds = _build_service(mock_settings, [src], use_real_store=True)

        # 第一次调用：填充缓存（days=80 使数据 100 行 >= 80*0.8=64）
        result1 = ds.get_price_data("300750", days=80, use_cache=False, clean=False)
        assert result1 is not None

        # 重置 mock 以追踪新调用
        src.get_price_data.reset_mock()

        # 第二次调用应使用缓存（数据新鲜，100行 >= 80*0.8=64）
        result2 = ds.get_price_data("300750", days=80, use_cache=True, clean=False)
        assert result2 is not None
        src.get_price_data.assert_not_called()

    def test_financial_snapshot_persisted_to_store(self, mock_settings):
        """Tushare 财务快照被持久化到真实存储"""
        from quant_agent.data.sources.base import FinancialSnapshot

        snapshot = FinancialSnapshot("300750", {
            "roe": 0.18, "pe_ttm": 25.0, "report_date": "2025-06-30",
        })
        src = _make_source("source1")
        ds = _build_service(mock_settings, [src], use_real_store=True)

        mock_tushare = MagicMock()
        mock_tushare.available = True
        mock_tushare.get_financial_snapshot.return_value = snapshot

        with patch.object(type(ds), "tushare", new_callable=PropertyMock, return_value=mock_tushare):
            result = ds.get_financial_snapshot("300750")

        assert result is not None
        # 验证财务数据已持久化
        loaded = ds.store.load_financial("300750", latest=True)
        assert loaded is not None
        assert loaded.iloc[0]["roe"] == 0.18


# ═══════════════════════════════════════════════════════════════════════════
# 6. Validation 边界测试
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationEdgeCases:
    """数据校验边界条件"""

    def test_empty_dataframe_fails(self):
        from quant_agent.data.validator import validate_price_data

        report = validate_price_data(pd.DataFrame())
        assert not report.is_valid
        assert len(report.errors) > 0

    def test_negative_close_price_detected_as_error(self):
        """负数收盘价应被检测为错误并记录在 errors 列表中

        NOTE: 当前 validator 将非正收盘价追加到 errors 但不设置
        is_valid=False。此测试验证错误被正确检测。
        """
        from quant_agent.data.validator import validate_price_data

        df = pd.DataFrame({
            "date": ["2025-01-01"],
            "open": [100.0],
            "high": [105.0],
            "low": [99.0],
            "close": [-5.0],
            "volume": [5000.0],
        })
        report = validate_price_data(df)
        assert "存在非正收盘价" in report.errors

    def test_high_null_percentage_fails(self):
        """超过阈值的空值比例应导致校验失败"""
        from quant_agent.data.validator import validate_price_data

        n = 100
        df = pd.DataFrame({
            "date": range(n),
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [99.0] * n,
            "close": [np.nan] * 10 + [100.0] * 90,  # 10% nulls
            "volume": [5000.0] * n,
        })
        report = validate_price_data(df)
        assert not report.is_valid

    def test_clean_removes_stale_rows(self):
        """清洗操作移除停牌日（成交量=0）"""
        from quant_agent.data.validator import clean_price_data

        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02", "2025-01-03"],
            "close": [100.0, 100.0, 101.0],
            "volume": [1000.0, 0.0, 1000.0],
        })
        cleaned = clean_price_data(df, remove_stale=True)
        assert len(cleaned) == 2
        assert 0.0 not in cleaned["volume"].values

    def test_validation_report_summary_format(self):
        """校验报告 summary 包含关键信息"""
        from quant_agent.data.validator import validate_price_data

        df = pd.DataFrame({
            "date": ["2025-01-01", "2025-01-02"],
            "open": [100.0, 101.0],
            "high": [105.0, 106.0],
            "low": [99.0, 100.0],
            "close": [103.0, 104.0],
            "volume": [5000.0, 6000.0],
        })
        report = validate_price_data(df)
        summary = report.summary()
        assert "PASS" in summary
        assert "rows=" in summary


# ═══════════════════════════════════════════════════════════════════════════
# 7. FinancialSnapshot 边界测试
# ═══════════════════════════════════════════════════════════════════════════


class TestFinancialSnapshotEdgeCases:
    """FinancialSnapshot 数据容器边界测试"""

    def test_missing_fields_return_none(self):
        from quant_agent.data.sources.base import FinancialSnapshot

        snap = FinancialSnapshot("300750", {"roe": 0.18})
        assert snap.pe_ttm is None
        assert snap.gross_margin is None
        assert snap.revenue_growth is None

    def test_get_with_default(self):
        from quant_agent.data.sources.base import FinancialSnapshot

        snap = FinancialSnapshot("300750", {"roe": 0.18})
        assert snap.get("roe") == 0.18
        assert snap.get("nonexistent", "fallback") == "fallback"

    def test_getitem_access(self):
        from quant_agent.data.sources.base import FinancialSnapshot

        snap = FinancialSnapshot("300750", {"roe": 0.18, "pe_ttm": 25.0})
        assert snap["roe"] == 0.18
        assert snap["pe_ttm"] == 25.0
        assert snap["missing"] is None

    def test_empty_data(self):
        from quant_agent.data.sources.base import FinancialSnapshot

        snap = FinancialSnapshot("300750", {})
        assert snap.roe is None
        assert snap.to_dict()["stock_code"] == "300750"

    def test_to_dict_includes_all_fields(self):
        from quant_agent.data.sources.base import FinancialSnapshot

        data = {"roe": 0.18, "pe_ttm": 25.0, "pb": 5.6}
        snap = FinancialSnapshot("300750", data)
        d = snap.to_dict()
        assert d["stock_code"] == "300750"
        assert d["roe"] == 0.18
        assert d["pe_ttm"] == 25.0
        assert d["pb"] == 5.6

    def test_repr_contains_code(self):
        from quant_agent.data.sources.base import FinancialSnapshot

        snap = FinancialSnapshot("601318", {"roe": 0.15})
        assert "601318" in repr(snap)

    def test_all_properties(self):
        """所有属性访问器返回正确值"""
        from quant_agent.data.sources.base import FinancialSnapshot

        data = {
            "roe": 0.18, "gross_margin": 0.35, "net_margin": 0.12,
            "debt_ratio": 0.45, "current_ratio": 1.8,
            "pe_ttm": 25.0, "pb": 5.6,
            "revenue_growth": 0.25, "profit_growth": 0.30,
        }
        snap = FinancialSnapshot("300750", data)
        assert snap.roe == 0.18
        assert snap.gross_margin == 0.35
        assert snap.net_margin == 0.12
        assert snap.debt_ratio == 0.45
        assert snap.current_ratio == 1.8
        assert snap.pe_ttm == 25.0
        assert snap.pb == 5.6
        assert snap.revenue_growth == 0.25
        assert snap.profit_growth == 0.30


# ═══════════════════════════════════════════════════════════════════════════
# 8. BaoStock DataSource ABC 合规
# ═══════════════════════════════════════════════════════════════════════════


class TestBaoStockSource:
    """BaoStock 适配器 ABC 合规测试"""

    def test_is_datasource(self):
        from quant_agent.data.sources.base import DataSource
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        assert isinstance(src, DataSource)

    def test_has_required_methods(self):
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        assert hasattr(src, "get_price_data")
        assert hasattr(src, "get_realtime_price")
        assert hasattr(src, "name")

    def test_name(self):
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        assert src.name == "baostock"

    def test_realtime_returns_none(self):
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        assert src.get_realtime_price("300750") is None
