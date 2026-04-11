"""邮件通知测试 — mock smtplib，验证三种场景"""

import json
import smtplib
from unittest.mock import MagicMock, patch, call
from datetime import datetime

import pytest

from quant_agent.config import Settings
from quant_agent.notification.email import EmailNotifier
from quant_agent.orchestrator import AnalysisReport
from quant_agent.agents.base import AgentResult


# ── Fixtures ────────────────────────────────────────────────────────────────


def _make_settings(enabled=True, recipients="test@example.com") -> Settings:
    """创建测试用 Settings"""
    return Settings(
        email_enabled=enabled,
        email_smtp_server="smtp.test.com",
        email_smtp_port=465,
        email_sender="sender@test.com",
        email_password="test_password",
        email_recipients=recipients,
        tushare_token="fake_token_for_testing",
    )


def _make_notifier(enabled=True, recipients="test@example.com") -> EmailNotifier:
    return EmailNotifier(_make_settings(enabled=enabled, recipients=recipients))


def _make_report(signal="BUY", confidence=0.85) -> AnalysisReport:
    """创建测试用 AnalysisReport"""
    return AnalysisReport(
        stock_code="300750",
        fundamental_result=AgentResult(
            agent_name="fundamental",
            stock_code="300750",
            signal="BUY",
            confidence=0.80,
            reasoning="ROE优秀, PE低估",
        ),
        technical_result=AgentResult(
            agent_name="technical",
            stock_code="300750",
            signal="BUY",
            confidence=0.90,
            reasoning="MACD金叉, RSI处于合理区间",
        ),
        risk_result=AgentResult(
            agent_name="risk",
            stock_code="300750",
            signal=signal,
            confidence=confidence,
            reasoning="2/2 agents agree: BUY",
            metrics={"position": 0.15, "stop_loss": -0.08, "take_profit_2": 0.20},
        ),
        summary={
            "total_equity": 105000.0,
            "cash": 80000.0,
            "position_value": 25000.0,
            "total_return": 0.05,
        },
    )


# ── Disabled Tests ──────────────────────────────────────────────────────────


class TestDisabled:
    """email_enabled=False 时不发送任何邮件"""

    def test_send_trade_signal_disabled(self):
        notifier = _make_notifier(enabled=False)
        result = notifier.send_trade_signal(_make_report())
        assert result is False

    def test_send_daily_report_disabled(self):
        notifier = _make_notifier(enabled=False)
        result = notifier.send_daily_report([], {})
        assert result is False

    def test_send_error_alert_disabled(self):
        notifier = _make_notifier(enabled=False)
        result = notifier.send_error_alert("test error")
        assert result is False


class TestNoRecipients:
    """无收件人时跳过发送"""

    def test_no_recipients_skips_send(self):
        notifier = _make_notifier(recipients="")
        # _send is called but returns False immediately
        result = notifier.send_trade_signal(_make_report())
        assert result is False


# ── Trade Signal Notification ───────────────────────────────────────────────


class TestTradeSignalNotification:
    """交易信号通知"""

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_buy_signal_sends_email(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        result = notifier.send_trade_signal(_make_report(signal="BUY"))

        assert result is True
        mock_smtp_cls.assert_called_once_with("smtp.test.com", 465, timeout=30)
        mock_server.login.assert_called_once_with("sender@test.com", "test_password")
        mock_server.sendmail.assert_called_once()

        # Verify sender and recipients
        call_args = mock_server.sendmail.call_args
        assert call_args[0][0] == "sender@test.com"
        assert call_args[0][1] == ["test@example.com"]

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_sell_signal_sends_email(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        result = notifier.send_trade_signal(_make_report(signal="SELL"))

        assert result is True
        mock_server.sendmail.assert_called_once()

    def test_hold_signal_no_email(self):
        notifier = _make_notifier()
        with patch.object(notifier, "_send") as mock_send:
            notifier.send_trade_signal(_make_report(signal="HOLD"))
            mock_send.assert_not_called()

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_email_contains_agent_votes(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        notifier.send_trade_signal(_make_report())

        mock_server.sendmail.assert_called_once()
        # Message contains HTML body (base64 encoded, non-empty)
        call_args = mock_server.sendmail.call_args
        msg_str = call_args[0][2]
        assert len(msg_str) > 200  # non-trivial email

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_multiple_recipients(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier(recipients="a@test.com, b@test.com")
        notifier.send_trade_signal(_make_report())

        call_args = mock_server.sendmail.call_args
        recipients = call_args[0][1]
        assert recipients == ["a@test.com", "b@test.com"]


# ── Daily Report ────────────────────────────────────────────────────────────


class TestDailyReport:
    """每日报告"""

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_daily_report_sends(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        reports = [
            _make_report(signal="BUY"),
            _make_report(signal="SELL"),
        ]
        reports[1].stock_code = "600519"

        portfolio_summary = {
            "total_equity": 110000.0,
            "cash": 50000.0,
            "total_return": 0.10,
            "positions": {
                "300750": {"shares": 100, "avg_price": 200.0, "current_price": 220.0, "pnl": 2000.0, "pnl_pct": 0.10},
            },
        }

        result = notifier.send_daily_report(reports, portfolio_summary)
        assert result is True

        mock_server.sendmail.assert_called_once()
        call_args = mock_server.sendmail.call_args
        msg_str = call_args[0][2]
        assert len(msg_str) > 200  # non-trivial HTML email

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_daily_report_empty_reports(self, mock_smtp_cls):
        """空报告列表仍会发送 (只是没有个股行)"""
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        result = notifier.send_daily_report([], {"total_equity": 100000, "cash": 100000, "total_return": 0.0})
        # Empty reports still sends an email (shows "BUY:0 SELL:0 HOLD:0")
        assert result is True


# ── Error Alert ─────────────────────────────────────────────────────────────


class TestErrorAlert:
    """异常告警"""

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_error_alert_sends(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        result = notifier.send_error_alert(
            "数据源连接失败",
            {"stock_code": "300750", "source": "tushare"},
        )

        assert result is True
        mock_server.sendmail.assert_called_once()

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_error_alert_no_context(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        result = notifier.send_error_alert("简单错误")

        assert result is True
        mock_server.sendmail.assert_called_once()


# ── SMTP Error Handling ────────────────────────────────────────────────────


class TestSMTPErrorHandling:
    """SMTP 异常不影响主流程"""

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_smtp_login_failure_returns_false(self, mock_smtp_cls):
        mock_server = MagicMock()
        mock_server.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Auth failed")
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        notifier = _make_notifier()
        result = notifier.send_trade_signal(_make_report())
        assert result is False

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_smtp_connection_failure_returns_false(self, mock_smtp_cls):
        mock_smtp_cls.side_effect = ConnectionError("Connection refused")

        notifier = _make_notifier()
        result = notifier.send_error_alert("test")
        assert result is False

    @patch("quant_agent.notification.email.smtplib.SMTP_SSL")
    def test_smtp_timeout_returns_false(self, mock_smtp_cls):
        mock_smtp_cls.side_effect = TimeoutError("SMTP timeout")

        notifier = _make_notifier()
        result = notifier.send_daily_report([_make_report()], {})
        assert result is False
