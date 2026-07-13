"""分析报告渲染 — 将 AnalysisReport 渲染为可读的 Markdown / HTML

面向「易用性」：让分析结果脱离裸日志，产出结构化、可分享、可持久化的报告。
"""

from __future__ import annotations

import html
from typing import Optional

from ..orchestrator import AnalysisReport
from ..agents.base import AgentResult


def _signal_badge(signal: str) -> str:
    return signal or "HOLD"


def _fmt_pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except (TypeError, ValueError):
        return "N/A"


def _agent_block(result: Optional[AgentResult], title: str) -> str:
    if result is None:
        return f"### {title}\n\n- 状态: 未运行\n"
    status = "OK" if result.success else "FAIL"
    lines = [
        f"### {title}",
        "",
        f"- 信号: **{_signal_badge(result.signal)}**",
        f"- 信心度: {_fmt_pct(result.confidence)}",
        f"- 状态: {status}",
    ]
    if result.reasoning:
        lines.append(f"- 理由: {result.reasoning}")
    if result.metrics:
        lines.append("")
        lines.append("**关键指标**")
        for k, v in result.metrics.items():
            if k in ("key_factors",):
                continue
            lines.append(f"- {k}: `{v}`")
    if not result.success and result.error:
        lines.append(f"- 错误: `{result.error}`")
    return "\n".join(lines) + "\n"


def render_markdown(report: AnalysisReport) -> str:
    """渲染为 Markdown 报告"""
    r = report.risk_result
    signal = _signal_badge(r.signal if r else "HOLD")
    conf = _fmt_pct(r.confidence if r else 0.0)
    pos = _fmt_pct(r.metrics.get("position", 0.0) if r else 0.0)

    parts = [
        f"# 量化分析报告 — {report.stock_code}",
        "",
        f"> 生成时间: {report.timestamp}",
        "",
        "## 综合结论",
        "",
        f"- **最终信号**: {signal}",
        f"- **信心度**: {conf}",
        f"- **建议仓位**: {pos}",
        "",
        _agent_block(report.fundamental_result, "基本面分析"),
        _agent_block(report.technical_result, "技术面分析"),
        _agent_block(report.sentiment_result, "情感分析"),
        _agent_block(report.risk_result, "风控汇总"),
    ]

    if report.risk_interpretation:
        parts.append("### 风险解读 (LLM)")
        parts.append("")
        parts.append(report.risk_interpretation)
        parts.append("")

    if report.execution_result:
        er = report.execution_result
        parts.append("### 执行结果")
        parts.append("")
        parts.append(f"- 信号: {_signal_badge(er.signal)}")
        if er.reasoning:
            parts.append(f"- 说明: {er.reasoning}")
        parts.append("")

    if report.summary:
        s = report.summary
        parts.append("## 组合状态")
        parts.append("")
        parts.append(f"- 总资产: `{s.get('total_equity', 0):,.2f}`")
        parts.append(f"- 现金: `{s.get('cash', 0):,.2f}`")
        parts.append(f"- 持仓市值: `{s.get('position_value', 0):,.2f}`")
        parts.append(f"- 总收益: {_fmt_pct(s.get('total_return', 0))}")
        parts.append("")

    if report.llm_analysis:
        parts.append("## LLM 综合报告")
        parts.append("")
        parts.append(report.llm_analysis)
        parts.append("")

    return "\n".join(parts)


def render_html(report: AnalysisReport) -> str:
    """渲染为自包含 HTML 报告（含基础样式）"""
    md = render_markdown(report)
    out: list[str] = ['<div class="report">']
    for line in md.splitlines():
        if line.startswith("# "):
            out.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            out.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("> "):
            out.append(f"<blockquote>{html.escape(line[2:])}</blockquote>")
        elif line.startswith("- "):
            body = html.escape(line[2:])
            body = body.replace("**", "").replace("`", "")
            out.append(f"<li>{body}</li>")
        elif line.strip() == "":
            out.append("")
        else:
            out.append(f"<p>{html.escape(line)}</p>")
    out.append("</div>")
    style = (
        "<style>"
        ".report{max-width:860px;margin:2rem auto;font-family:-apple-system,"
        "Segoe UI,Roboto,Helvetica,Arial,sans-serif;line-height:1.6;color:#1a1a1a;}"
        ".report h1{border-bottom:2px solid #2563eb;padding-bottom:.3rem;}"
        ".report h2{margin-top:1.6rem;color:#2563eb;}"
        ".report blockquote{background:#f1f5f9;padding:.5rem 1rem;border-left:4px solid #94a3b8;}"
        ".report li{margin:.2rem 0;}"
        "</style>"
    )
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Quant Report</title>{style}</head><body>" + "\n".join(out) + "</body></html>"
