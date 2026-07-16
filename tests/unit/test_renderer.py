"""P1.2 — 报告渲染：数据谱系表格 (v2) + 显著警示区块"""

from __future__ import annotations

from quant_agent.data.sources.base import DataProvenance
from quant_agent.orchestrator import AnalysisReport
from quant_agent.reporting.renderer import render_html, render_markdown


def _prov(**kw) -> DataProvenance:
    base = dict(
        source="sample",
        identifier="600519:price",
        fetched_at="2026-07-15T00:00:00",
        data_type="price",
        confidence="low",
    )
    base.update(kw)
    return DataProvenance(**base)


def test_render_markdown_lineage_table_v2_columns():
    """谱系表格应展示 v2 扩展字段列（交易日/复权/缓存年龄/缺失字段/指纹）。"""
    report = AnalysisReport(
        stock_code="600519",
        data_lineage=[
            _prov(
                source="efinance",
                confidence="high",
                trading_day="2026-07-14",
                adjust_status="qfq",
                cache_age_seconds=120.0,
                data_hash="abc123",
            )
        ],
    )
    md = render_markdown(report)
    assert "数据来源 (Data Lineage)" in md
    assert "交易日" in md
    assert "复权" in md
    assert "缓存年龄" in md
    assert "缺失字段" in md
    assert "指纹" in md
    # 具体值应被渲染
    assert "2026-07-14" in md
    assert "qfq" in md
    assert "abc123" in md


def test_render_markdown_shows_lineage_warnings():
    """降级 / 样例场景应在报告显著位置输出警示区块。"""
    report = AnalysisReport(
        stock_code="600519",
        data_lineage=[
            _prov(source="sample", confidence="low", missing_fields=["pe_ttm", "pb"]),
            _prov(source="cache", confidence="partial", cache_age_seconds=86400 * 40),
        ],
    )
    md = render_markdown(report)
    assert "数据谱系警示" in md
    assert "使用内置演示样例" in md
    assert "缺失字段" in md
    assert "缓存较旧" in md
    # 不构成投资建议提示
    assert "不构成任何投资建议" in md


def test_render_markdown_no_warnings_when_clean():
    """干净来源不应出现警示区块。"""
    report = AnalysisReport(
        stock_code="600519",
        data_lineage=[
            _prov(source="efinance", confidence="high", trading_day="2026-07-14"),
        ],
    )
    md = render_markdown(report)
    assert "数据谱系警示" not in md


def test_render_markdown_warnings_dedup():
    """多来源重复警示应去重。"""
    report = AnalysisReport(
        stock_code="600519",
        data_lineage=[_prov(source="sample"), _prov(source="sample")],
    )
    md = render_markdown(report)
    # 仅一条「使用内置演示样例」警示
    assert md.count("使用内置演示样例") == 1


def test_render_html_includes_warnings_blockquote():
    """HTML 渲染应包含警示（blockquote 形式）。"""
    report = AnalysisReport(
        stock_code="600519",
        data_lineage=[_prov(source="sample", confidence="low")],
    )
    html = render_html(report)
    assert "数据谱系警示" in html
    assert "<blockquote>" in html
