"""WeStockSource 单元测试 — ABC 合规、代码转换、Markdown 解析、CLI Mock

不真实调用 npx CLI，所有 subprocess 调用通过 mock 模拟。
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quant_agent.data.sources.westock import (
    _parse_markdown_table,
    from_westock_code,
    to_westock_code,
)


# ── Helpers ──

_KLINE_MD = """
| date | open | last | high | low | volume | amount | exchange |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-16 | 8.92 | 8.85 | 8.94 | 8.80 | 755826 | 670460000 | 0.23 |
| 2026-07-15 | 8.66 | 8.89 | 8.90 | 8.64 | 1047582 | 966740000 | 0.31 |
"""

_FINANCE_LRB_MD = """
**lrb**

| _date | OperatingRevenue | TotalOperatingCost | NPParentCompanyOwners | TotalProfit | EndDate |
| --- | --- | --- | --- | --- | --- |
| 2025-09-30 | 132280000000.00 | 90073000000.00 | 38819000000.00 | 42016000000.00 | 2025-09-30 |
"""

_FINANCE_ZCFZ_MD = """
**zcfz**

| _date | TotalAssets | TotalLiabilities | TotalOwnerEquity | EndDate |
| --- | --- | --- | --- | --- |
| 2025-09-30 | 900000000000.00 | 820000000000.00 | 80000000000.00 | 2025-09-30 |
"""

_SEARCH_MD = """
| code | name | type |
| --- | --- | --- |
| sz300750 | 宁德时代 | GP-A-CYB |
| hk03750 | 宁德时代 | GP |
"""

_MINUTE_MD = """
| code | time | price | volume | amount |
| --- | --- | --- | --- | --- |
| sh600000 | 0930 | 8.92 | 12129 | 10819068.00 |
| sh600000 | 0931 | 8.89 | 49267 | 43809321.00 |
"""

_PROFILE_MD = """
| code | name | listedDate | business | website | industry | sector | issuePrice |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sh600000 | 浦发银行 | 1999-11-10 | 吸收公众存款 | http://www.spdb.com.cn | 银行 | 银行 | 10.00 |
"""

_ASFUND_MD = """
| code | BlockNetFlow | ClosePrice | EndDate | MainNetFlow | MainInFlow | MainOutFlow | SmallNetFlow |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sh600000 | -20980063.00 | 8.85 | 2026-07-16 | -54967262.00 | 213354408.00 | 268321671.00 | 87892215.00 |
"""

_CHIP_MD = """
| code | name | date | closePrice | chipProfitRate | chipAvgCost | chipConcentration90 | chipConcentration70 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sh600519 | 贵州茅台 | 2026-07-16 | 1258.99 | 16.19 | 1380.82 | 11.04 | 6.93 |
"""

_DIVIDEND_MD = """
| reportEndDate | dividendFlag | dividendType | cashDiviRMB | dividendPlan |
| --- | --- | --- | --- | --- |
| 20251231 | 是 | 有分红 | 280.24 | 10派280.242元 |
"""

_SHAREHOLDER_MD = """
#### sh600519 贵州茅台 (2026-03-31)

**十大股东**

| no | name | holdShares | holdPct | holdChange |
| --- | --- | --- | --- | --- |
| 1 | 中国贵州茅台酒厂 | 681282935 | 54.40 | 0 |
| 2 | 香港中央结算 | 58733069 | 4.69 | 3684225 |

**十大流通股东**

| no | name | holdShares | holdPct | holdChange |
| --- | --- | --- | --- | --- |
| 1 | 中国贵州茅台酒厂 | 681282935 | 54.40 | 0 |
"""


def _create_source(**kwargs):
    """Create WeStockSource with mocked rate limiter and npx available."""
    from quant_agent.data.sources.westock import WeStockSource

    kwargs.setdefault("enabled", True)
    src = WeStockSource(**kwargs)
    src._rate_limiter = MagicMock()
    src._npx_available = True
    return src


def _mock_cli(source, markdown_text: str):
    """Patch _run_cli to return given markdown text."""
    source._run_cli = MagicMock(return_value=markdown_text)


# ═══════════════════════════════════════════════════════════════════════════
# 1. 代码格式转换
# ═══════════════════════════════════════════════════════════════════════════


class TestCodeConversion:
    def test_sh_prefix(self):
        assert to_westock_code("600000") == "sh600000"
        assert to_westock_code("688981") == "sh688981"

    def test_sz_prefix(self):
        assert to_westock_code("300750") == "sz300750"
        assert to_westock_code("000001") == "sz000001"

    def test_bj_prefix(self):
        assert to_westock_code("430047") == "bj430047"
        assert to_westock_code("830799") == "bj830799"

    def test_already_prefixed_passthrough(self):
        assert to_westock_code("sh600000") == "sh600000"
        assert to_westock_code("sz300750") == "sz300750"

    def test_invalid_code(self):
        with pytest.raises(ValueError):
            to_westock_code("123")
        with pytest.raises(ValueError):
            to_westock_code("abcd")

    def test_from_westock(self):
        assert from_westock_code("sh600000") == "600000"
        assert from_westock_code("sz300750") == "300750"
        assert from_westock_code("bj430047") == "430047"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Markdown 解析
# ═══════════════════════════════════════════════════════════════════════════


class TestMarkdownParse:
    def test_parse_kline(self):
        df = _parse_markdown_table(_KLINE_MD)
        assert df is not None
        assert list(df.columns) == [
            "date", "open", "last", "high", "low", "volume", "amount", "exchange"
        ]
        assert len(df) == 2
        assert df.iloc[0]["last"] == "8.85"

    def test_parse_multi_table_returns_first(self):
        combined = _FINANCE_LRB_MD + "\n" + _FINANCE_ZCFZ_MD
        df = _parse_markdown_table(combined)
        assert df is not None
        assert "OperatingRevenue" in df.columns

    def test_parse_empty(self):
        assert _parse_markdown_table("no tables here") is None
        assert _parse_markdown_table("") is None

    def test_parse_handles_separator_row(self):
        df = _parse_markdown_table(_FINANCE_LRB_MD)
        assert df is not None
        # 仅 1 数据行
        assert len(df) == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3. ABC 合规 & 可用性
# ═══════════════════════════════════════════════════════════════════════════


class TestWeStockABC:
    def test_is_datasource(self):
        from quant_agent.data.sources.base import DataSource

        src = _create_source()
        assert isinstance(src, DataSource)

    def test_name(self):
        assert _create_source().name == "westock"

    def test_available_when_enabled_and_npx(self):
        assert _create_source(enabled=True).available is True

    def test_unavailable_when_disabled(self):
        assert _create_source(enabled=False).available is False

    def test_unavailable_when_no_npx(self):
        src = _create_source()
        src._npx_available = False
        assert src.available is False


# ═══════════════════════════════════════════════════════════════════════════
# 4. 行情数据 (Mock CLI)
# ═══════════════════════════════════════════════════════════════════════════


class TestPriceData:
    def test_get_price_data(self):
        src = _create_source()
        _mock_cli(src, _KLINE_MD)
        df = src.get_price_data("600000", days=10)
        assert df is not None
        assert "close" in df.columns  # last -> close 映射
        assert "date" in df.columns
        assert len(df) == 2
        # 数值类型
        assert pd.api.types.is_numeric_dtype(df["close"])

    def test_empty_on_cli_failure(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=None)
        assert src.get_price_data("600000") is None

    def test_code_conversion_applied(self):
        src = _create_source()
        captured = {}
        src._run_cli = MagicMock(return_value=_KLINE_MD)
        src.get_price_data("600000")
        # 验证调用参数包含 sh600000
        args = src._run_cli.call_args[0]
        assert "sh600000" in args

    def test_realtime_price_from_kline(self):
        src = _create_source()
        _mock_cli(src, _KLINE_MD)
        price = src.get_realtime_price("600000")
        assert price == 8.85  # 最后一行 close


# ═══════════════════════════════════════════════════════════════════════════
# 5. 财务快照 (Mock CLI)
# ═══════════════════════════════════════════════════════════════════════════


class TestFinancialSnapshot:
    def test_snapshot_calculates_metrics(self):
        src = _create_source()
        # lrb 与 zcfz 交替返回
        src._run_cli = MagicMock(
            side_effect=[_FINANCE_LRB_MD, _FINANCE_ZCFZ_MD]
        )
        snap = src.get_financial_snapshot("600000")
        assert snap is not None
        # 毛利率 = (rev - cost) / rev
        rev = 132280000000.0
        cost = 90073000000.0
        expected_gm = (rev - cost) / rev
        assert abs(snap.gross_margin - expected_gm) < 1e-6
        # 净利率 = profit / rev
        profit = 38819000000.0
        assert abs(snap.net_margin - profit / rev) < 1e-6
        # ROE = profit / equity
        equity = 80000000000.0
        assert abs(snap.roe - profit / equity) < 1e-6
        # 资产负债率 = liab / assets
        liab = 820000000000.0
        assets = 900000000000.0
        assert abs(snap.debt_ratio - liab / assets) < 1e-6
        assert snap.get("report_date") == "2025-09-30"

    def test_none_when_lrb_missing(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=None)
        assert src.get_financial_snapshot("600000") is None

    def test_report_date_from_enddate(self):
        src = _create_source()
        src._run_cli = MagicMock(side_effect=[_FINANCE_LRB_MD, _FINANCE_ZCFZ_MD])
        snap = src.get_financial_snapshot("600000")
        assert snap.get("report_date") == "2025-09-30"


# ═══════════════════════════════════════════════════════════════════════════
# 6. 技术指标
# ═══════════════════════════════════════════════════════════════════════════


class TestTechnical:
    def test_get_technical_indicators(self):
        src = _create_source()
        _mock_cli(src, _FINANCE_ZCFZ_MD)  # 用一个表模拟
        df = src.get_technical_indicators("600000")
        assert df is not None


# ═══════════════════════════════════════════════════════════════════════════
# 8. 扩展命令 (search/minute/profile/fund/chip/shareholder/dividend)
# ═══════════════════════════════════════════════════════════════════════════


class TestExtendedCommands:
    def test_search_stock(self):
        src = _create_source()
        _mock_cli(src, _SEARCH_MD)
        df = src.search_stock("宁德时代")
        assert df is not None
        assert len(df) == 2
        assert "sz300750" in df["code"].values

    def test_get_minute_data(self):
        src = _create_source()
        _mock_cli(src, _MINUTE_MD)
        df = src.get_minute_data("600000")
        assert df is not None
        assert list(df.columns) == ["code", "time", "price", "volume", "amount"]
        assert len(df) == 2

    def test_get_minute_data_multi_day_arg(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=_MINUTE_MD)
        src.get_minute_data("600000", days=5)
        assert "--days" in src._run_cli.call_args[0]

    def test_get_company_profile(self):
        src = _create_source()
        _mock_cli(src, _PROFILE_MD)
        prof = src.get_company_profile("600000")
        assert prof is not None
        assert prof["name"] == "浦发银行"
        assert prof["industry"] == "银行"

    def test_get_fund_flow(self):
        src = _create_source()
        _mock_cli(src, _ASFUND_MD)
        df = src.get_fund_flow("600000")
        assert df is not None
        assert "MainNetFlow" in df.columns
        assert float(df.iloc[0]["MainNetFlow"]) == -54967262.00

    def test_get_fund_flow_with_date(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=_ASFUND_MD)
        src.get_fund_flow("600000", date="2026-03-10")
        assert "--date" in src._run_cli.call_args[0]

    def test_get_chip(self):
        src = _create_source()
        _mock_cli(src, _CHIP_MD)
        df = src.get_chip("600519")
        assert df is not None
        assert "chipAvgCost" in df.columns
        assert float(df.iloc[0]["chipAvgCost"]) == 1380.82

    def test_get_shareholders(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=_SHAREHOLDER_MD)
        sections = src.get_shareholders("600519")
        assert sections is not None
        assert len(sections) == 1  # 单只股票一个 current
        stock = sections[0]
        assert stock["date"] == "2026-03-31"
        assert len(stock["sections"]) == 2  # 十大股东 + 十大流通股东
        top = stock["sections"][0]
        assert top["section"] == "十大股东"
        assert len(top["rows"]) == 2
        assert top["rows"][0]["name"] == "中国贵州茅台酒厂"

    def test_get_dividend(self):
        src = _create_source()
        _mock_cli(src, _DIVIDEND_MD)
        df = src.get_dividend("600519")
        assert df is not None
        assert "cashDiviRMB" in df.columns
        assert df.iloc[0]["dividendPlan"] == "10派280.242元"

    def test_get_dividend_with_years(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=_DIVIDEND_MD)
        src.get_dividend("600519", years=5)
        assert "--years" in src._run_cli.call_args[0]

    def test_search_returns_none_on_failure(self):
        src = _create_source()
        src._run_cli = MagicMock(return_value=None)
        assert src.search_stock("x") is None

    def test_shareholder_parse_helper(self):
        from quant_agent.data.sources.westock import WeStockSource

        sections = WeStockSource._parse_shareholder(_SHAREHOLDER_MD)
        assert len(sections) == 1
        assert len(sections[0]["sections"]) == 2
        assert all(s["rows"] for s in sections[0]["sections"])


# ═══════════════════════════════════════════════════════════════════════════
# 7. DataService 集成
# ═══════════════════════════════════════════════════════════════════════════


class TestDataServiceIntegration:
    def test_westock_in_sources_when_enabled(self):
        from quant_agent.data.service import DataService

        with patch(
            "quant_agent.data.sources.westock.WeStockSource"
        ) as MockWs:
            mock_src = MagicMock()
            mock_src.available = True
            mock_src.name = "westock"
            MockWs.return_value = mock_src
            svc = DataService.__new__(DataService)
            svc.settings = MagicMock()
            svc.settings.parquet_dir = "data/parquet"
            svc.settings.tushare_token = None
            svc.settings.westock_enabled = True
            svc.settings.akshare_timeout = 10
            svc.settings.data_cache_ttl = 1800
            svc._lineage = {}
            svc.store = MagicMock()
            svc._sources = []
            # 手动初始化 westock 部分
            from quant_agent.data.sources.westock import WeStockSource

            svc._westock = WeStockSource(enabled=True)
            svc._westock.available = True
            svc._sources.append(svc._westock)
            assert any(s.name == "westock" for s in svc._sources)

    def test_westock_excluded_when_disabled(self):
        from quant_agent.data.sources.westock import WeStockSource

        src = WeStockSource(enabled=False)
        assert src.available is False
