"""SMTP 邮件通知器 — 交易信号、每日报告、异常告警"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..config import Settings
    from ..orchestrator import AnalysisReport

logger = logging.getLogger(__name__)

# Signal emojis
_SIGNAL_ICON = {
    "BUY": "&#x1F7E2;",   # green circle
    "SELL": "&#x1F534;",   # red circle
    "HOLD": "&#x1F7E1;",   # yellow circle
}


class EmailNotifier:
    """SMTP 邮件通知器

    支持三种通知场景:
    - 交易信号通知 (BUY/SELL 触发时)
    - 每日分析报告 (批量分析后汇总)
    - 系统异常告警
    """

    def __init__(self, settings: Settings):
        self.enabled = settings.email_enabled
        self.smtp_server = settings.email_smtp_server
        self.smtp_port = settings.email_smtp_port
        self.sender = settings.email_sender
        self.password = settings.email_password
        self.recipients = [
            r.strip() for r in settings.email_recipients.split(",") if r.strip()
        ]

    def send_trade_signal(self, report: AnalysisReport) -> bool:
        """交易信号通知 — 单股分析产生 BUY/SELL 时发送"""
        if not self.enabled:
            return False

        signal = report.risk_result.signal if report.risk_result else "HOLD"
        if signal not in ("BUY", "SELL"):
            return False

        icon = _SIGNAL_ICON.get(signal, "")
        confidence = (
            f"{report.risk_result.confidence:.0%}"
            if report.risk_result
            else "N/A"
        )
        now = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Build agent votes table
        agent_rows = ""
        for result in [report.fundamental_result, report.technical_result, report.risk_result]:
            if result is None:
                continue
            agent_rows += (
                f"<tr><td>{result.agent_name}</td>"
                f"<td>{_SIGNAL_ICON.get(result.signal, '')} {result.signal}</td>"
                f"<td>{result.confidence:.0%}</td>"
                f"<td>{result.reasoning[:80]}</td></tr>"
            )

        # Execution info
        exec_info = ""
        if report.execution_result and report.execution_result.success:
            m = report.execution_result.metrics
            exec_info = (
                f"<p>执行: {m.get('shares', 0)} 股 @ {m.get('price', 0):.2f}"
                f" | 佣金: {m.get('commission', 0):.2f}</p>"
            )

        html = f"""<html><body style="font-family: Arial, sans-serif; max-width: 700px; margin: 0 auto;">
<h2>{icon} 交易信号: {signal} {report.stock_code}</h2>
<p><strong>时间:</strong> {now} | <strong>信心度:</strong> {confidence}</p>

<h3>Agent 投票</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; font-size: 14px;">
<tr style="background:#f0f0f0;"><th>Agent</th><th>信号</th><th>信心度</th><th>理由</th></tr>
{agent_rows}
</table>

{exec_info}

<h3>组合状态</h3>
<p>总权益: {report.summary.get('total_equity', 0):,.2f} |
   收益: {report.summary.get('total_return', 0):.2%}</p>

<hr>
<p style="color: #999; font-size: 12px;">AI Quant Agent v3.0 | 自动通知，不构成投资建议</p>
</body></html>"""

        subject = f"[AI量化] {signal} {report.stock_code} ({confidence})"
        return self._send(subject, html)

    def send_daily_report(
        self,
        reports: list[AnalysisReport],
        portfolio_summary: dict,
    ) -> bool:
        """每日分析报告 — 批量分析后发送汇总"""
        if not self.enabled:
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        total_return = portfolio_summary.get("total_return", 0.0)
        equity = portfolio_summary.get("total_equity", 0.0)

        # Signal summary
        buy_count = sum(
            1 for r in reports
            if r.risk_result and r.risk_result.signal == "BUY"
        )
        sell_count = sum(
            1 for r in reports
            if r.risk_result and r.risk_result.signal == "SELL"
        )
        hold_count = len(reports) - buy_count - sell_count

        # Individual stock rows
        stock_rows = ""
        for report in reports:
            signal = report.risk_result.signal if report.risk_result else "HOLD"
            conf = (
                f"{report.risk_result.confidence:.0%}"
                if report.risk_result
                else "N/A"
            )
            icon = _SIGNAL_ICON.get(signal, "")
            reason = ""
            if report.risk_result:
                reason = report.risk_result.reasoning[:60]
            stock_rows += (
                f"<tr><td>{report.stock_code}</td>"
                f"<td>{icon} {signal}</td>"
                f"<td>{conf}</td>"
                f"<td>{reason}</td></tr>"
            )

        # Position details
        positions = portfolio_summary.get("positions", {})
        pos_rows = ""
        for code, pos in positions.items():
            pnl_color = "green" if pos.get("pnl", 0) >= 0 else "red"
            pos_rows += (
                f"<tr><td>{code}</td>"
                f"<td>{pos.get('shares', 0)}</td>"
                f"<td>{pos.get('avg_price', 0):.2f}</td>"
                f"<td>{pos.get('current_price', 0):.2f}</td>"
                f"<td style='color:{pnl_color}'>{pos.get('pnl', 0):+.2f}</td>"
                f"<td style='color:{pnl_color}'>{pos.get('pnl_pct', 0):+.2%}</td></tr>"
            )

        positions_section = ""
        if pos_rows:
            positions_section = f"""
<h3>当前持仓</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; font-size: 14px;">
<tr style="background:#f0f0f0;"><th>股票</th><th>数量</th><th>成本</th><th>现价</th><th>盈亏</th><th>盈亏%</th></tr>
{pos_rows}
</table>"""

        html = f"""<html><body style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
<h2>AI Quant Agent v3.0 - 每日分析报告</h2>
<p><strong>报告时间:</strong> {now}</p>

<h3>信号汇总</h3>
<ul>
<li>&#x1F7E2; BUY: {buy_count} 只</li>
<li>&#x1F534; SELL: {sell_count} 只</li>
<li>&#x1F7E1; HOLD: {hold_count} 只</li>
</ul>

<h3>个股分析</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; font-size: 14px;">
<tr style="background:#f0f0f0;"><th>股票</th><th>信号</th><th>信心度</th><th>理由</th></tr>
{stock_rows}
</table>

{positions_section}

<h3>组合概况</h3>
<p>总权益: <strong>{equity:,.2f}</strong> | 收益: <strong>{total_return:.2%}</strong> |
   现金: {portfolio_summary.get('cash', 0):,.2f}</p>

<hr>
<p style="color: #999; font-size: 12px;">
AI Quant Agent v3.0 | {now}<br>
信号仅供参考，不构成投资建议
</p>
</body></html>"""

        subject = f"[AI量化] 每日报告 {now} (BUY:{buy_count} SELL:{sell_count})"
        return self._send(subject, html)

    def send_error_alert(self, error_msg: str, context: dict | None = None) -> bool:
        """系统异常告警"""
        if not self.enabled:
            return False

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        ctx_lines = ""
        if context:
            for k, v in context.items():
                ctx_lines += f"<tr><td>{k}</td><td>{v}</td></tr>"

        context_section = ""
        if ctx_lines:
            context_section = f"""
<h3>上下文</h3>
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse;">
{ctx_lines}
</table>"""

        html = f"""<html><body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
<h2>&#x26A0;&#xFE0F; 系统异常告警</h2>
<p><strong>时间:</strong> {now}</p>
<p><strong>错误:</strong> {error_msg}</p>
{context_section}
<hr>
<p style="color: #999; font-size: 12px;">AI Quant Agent v3.0</p>
</body></html>"""

        subject = f"[AI量化] 异常告警 - {error_msg[:50]}"
        return self._send(subject, html)

    def _send(self, subject: str, html: str) -> bool:
        """发送 HTML 邮件"""
        if not self.recipients:
            logger.warning("邮件通知: 无收件人，跳过发送")
            return False

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.sender
        msg["To"] = ", ".join(self.recipients)
        msg.attach(MIMEText(html, "html", "utf-8"))

        try:
            with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port, timeout=30) as server:
                server.login(self.sender, self.password)
                server.sendmail(self.sender, self.recipients, msg.as_string())
            logger.info("邮件发送成功: %s", subject)
            return True
        except Exception as e:
            logger.error("邮件发送失败: %s", e)
            return False
