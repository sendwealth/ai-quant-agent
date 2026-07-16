"""Tests for 推荐 #2 — 数据可信硬门禁 (data.gate)。

覆盖：
- evaluate_trust 对 sample / low / partial / high / 空谱系 的判定；
- 交易路径 (TradingService) 拦截合成数据；
- 回测路径 (BacktestEngine) 拦截合成数据；
- 报告渲染 (renderer) 的水印（Markdown + HTML）。
"""

from __future__ import annotations

import pandas as pd
import pytest

from quant_agent.backtest.engine import BacktestEngine
from quant_agent.data.gate import (
    DataTrustError,
    evaluate_trust,
)
from quant_agent.data.sources.base import DataProvenance
from quant_agent.orchestrator import AnalysisReport
from quant_agent.reporting.renderer import render_html, render_markdown
from quant_agent.trading.service import TradingService


# ── helpers ──
def _prov(source: str = "tushare", confidence: str = "high") -> DataProvenance:
    return DataProvenance(
        source=source,
        identifier="600519:price",
        fetched_at="2026-07-15T00:00:00",
        data_type="price",
        confidence=confidence,
    )


# ──────────────────────────────────────────────────────────────────────────
# evaluate_trust 单元
# ──────────────────────────────────────────────────────────────────────────
def test_high_confidence_allowed_for_all_purposes():
    prov = [_prov("tushare", "high")]
    for purpose in ("trading", "backtest", "report", "screen"):
        v = evaluate_trust(prov, purpose)
        assert v.allowed is True
        assert v.reasons == []


def test_sample_source_blocked_for_trading():
    v = evaluate_trust([_prov("sample", "low")], "trading")
    assert v.allowed is False
    assert any("合成样例" in r for r in v.reasons)
    with pytest.raises(DataTrustError):
        v.require()


def test_sample_source_blocked_for_backtest():
    v = evaluate_trust([_prov("sample", "low")], "backtest")
    assert v.allowed is False
    with pytest.raises(DataTrustError):
        v.require()


def test_low_confidence_blocked_for_trading():
    v = evaluate_trust([_prov("tushare", "low")], "trading")
    assert v.allowed is False
    assert any("低" in r for r in v.reasons)


def test_sample_allowed_for_report_but_warned():
    """只读用途放行，但记录显著警示（供水印标红）。"""
    v = evaluate_trust([_prov("sample", "low")], "report")
    assert v.allowed is True
    assert v.warning_text is not None
    assert any("不构成任何投资建议" in r for r in v.reasons)


def test_partial_confidence_warned_only():
    v = evaluate_trust([_prov("tushare", "partial")], "report")
    assert v.allowed is True
    assert any("部分" in r for r in v.reasons)


def test_empty_provenance_blocked_for_trading():
    """决策用途缺谱系时 fail closed：默认拒绝执行。"""
    v = evaluate_trust([], "trading")
    assert v.allowed is False
    assert v.blocked is True
    assert v.level == "unknown"
    assert v.reasons  # fail closed 原因
    with pytest.raises(DataTrustError):
        v.require()


def test_empty_provenance_research_exempt():
    """显式研究模式可豁免缺谱系（仅放行并标红，不构成决策依据）。"""
    v = evaluate_trust([], "trading", research_mode=True)
    assert v.allowed is True
    assert v.warning_text is not None
    assert any("研究模式豁免" in r for r in v.reasons)


def test_trading_research_exempt_on_empty_lineage():
    """交易路径缺谱系时，显式研究模式可豁免（仅用于模拟，不下真实决策）。"""
    fake = _FakeExec()
    svc = TradingService(execution=fake, risk=_FakeRisk())
    report = AnalysisReport(stock_code="600519", data_lineage=[])
    # 不应抛 DataTrustError（研究模式豁免）
    svc.execute(report, analysis_results=[], research_mode=True)


def test_worst_confidence_selected():
    v = evaluate_trust([_prov("tushare", "high"), _prov("cache", "partial")], "report")
    assert v.level == "partial"
    assert "cache" in v.sources


# ──────────────────────────────────────────────────────────────────────────
# 交易路径接入
# ──────────────────────────────────────────────────────────────────────────
class _FakeExec:
    positions: dict = {}
    total_equity = 0

    def __init__(self):
        self.calls = 0

    def execute_signal(self, *a, **k):
        self.calls += 1
        return None

    def check_stop_conditions(self, *a, **k):
        return None

    def get_summary(self):
        return {}


class _FakeRisk:
    t1_tracker = None


def test_trading_blocked_on_sample_lineage():
    fake = _FakeExec()
    svc = TradingService(execution=fake, risk=_FakeRisk())
    report = AnalysisReport(stock_code="600519", data_lineage=[_prov("sample", "low")])
    result = svc.execute(report, analysis_results=[])
    assert result is None
    assert fake.calls == 0  # 绝不下单


def test_trading_proceeds_on_real_lineage():
    """真实数据谱系下不应被门禁拦截（此处 risk_result=None 仍会提前返回 None，
    但 execute_signal 不会被门禁误杀，且不抛异常）。"""
    fake = _FakeExec()
    svc = TradingService(execution=fake, risk=_FakeRisk())
    report = AnalysisReport(stock_code="600519", data_lineage=[_prov("tushare", "high")])
    # 不应抛出 DataTrustError
    svc.execute(report, analysis_results=[])


# ──────────────────────────────────────────────────────────────────────────
# 回测路径接入
# ──────────────────────────────────────────────────────────────────────────
def _price_df():
    return pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "close": [10.0, 11.0, 12.0],
            "volume": [1000, 1000, 1000],
        }
    )


def test_backtest_blocks_sample_provenance():
    eng = BacktestEngine()
    with pytest.raises(DataTrustError):
        eng.run(
            _price_df(),
            signals=pd.Series([1, 0, 0]),
            provenance=[_prov("sample", "low")],
        )


def test_backtest_blocks_without_provenance():
    """正式回测缺谱系时 fail closed：默认拒绝执行。"""
    eng = BacktestEngine()
    with pytest.raises(DataTrustError):
        eng.run(_price_df(), signals=pd.Series([1, 0, 0]))


def test_backtest_runs_without_provenance_in_research_mode():
    """显式研究模式豁免缺谱系，回测照常运行。"""
    eng = BacktestEngine()
    res = eng.run(_price_df(), signals=pd.Series([1, 0, 0]), research_mode=True)
    assert res is not None


def test_backtest_runs_with_real_provenance():
    eng = BacktestEngine()
    res = eng.run(
        _price_df(),
        signals=pd.Series([1, 0, 0]),
        provenance=[_prov("tushare", "high")],
    )
    assert res is not None


# ──────────────────────────────────────────────────────────────────────────
# 报告水印
# ──────────────────────────────────────────────────────────────────────────
def test_renderer_watermark_on_sample():
    report = AnalysisReport(stock_code="600519", data_lineage=[_prov("sample", "low")])
    md = render_markdown(report)
    assert "数据可信水印" in md
    html = render_html(report)
    assert 'class="watermark"' in html
    assert "不构成任何投资建议" in html


def test_renderer_no_watermark_on_high():
    report = AnalysisReport(stock_code="600519", data_lineage=[_prov("tushare", "high")])
    md = render_markdown(report)
    assert "数据可信水印" not in md
    html = render_html(report)
    assert 'class="watermark"' not in html
