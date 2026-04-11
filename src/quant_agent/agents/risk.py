"""风险控制 Agent — 信号共识 + 仓位管理 + 组合级风控

Enhanced risk management:
- T+1 enforcement (A-share rule: shares bought today cannot be sold today)
- Portfolio-level risk limits (max_portfolio_risk, sector concentration)
- Daily loss circuit breaker
- Position sizing considers portfolio heat, not just confidence
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Optional

from .base import BaseAgent, AgentResult

if TYPE_CHECKING:
    from ..config import Settings
    from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class T1Tracker:
    """Track buy dates to enforce T+1 settlement rule.

    A-share rule: shares bought on day D can only be sold on day D+1.
    """

    _buy_dates: dict[str, str] = field(default_factory=dict)

    def record_buy(self, stock_code: str, trade_date: str | None = None) -> None:
        """Record a buy so we can enforce T+1."""
        d = trade_date or date.today().isoformat()
        self._buy_dates[stock_code] = d

    def can_sell(self, stock_code: str, current_date: str | None = None) -> bool:
        """Return True if the stock was bought before today (T+1 satisfied)."""
        buy_date = self._buy_dates.get(stock_code)
        if buy_date is None:
            return True  # not bought through this tracker, allow
        today = current_date or date.today().isoformat()
        # Handle both YYYY-MM-DD and YYYYMMDD formats by normalizing
        buy_norm = buy_date.replace("-", "")
        today_norm = today.replace("-", "")
        return buy_norm < today_norm

    def clear(self, stock_code: str) -> None:
        """Remove tracking after position is fully closed."""
        self._buy_dates.pop(stock_code, None)


@dataclass
class DailyPnLTracker:
    """Track daily P&L to enforce max daily loss circuit breaker."""

    _daily_start_equity: float = 0.0
    _current_date: str = ""
    max_daily_loss_pct: float = -0.03

    def update_date(self, current_date: str, equity: float) -> None:
        """Reset daily tracker on a new trading day."""
        if current_date != self._current_date:
            self._current_date = current_date
            self._daily_start_equity = equity

    def check_circuit_breaker(self, current_equity: float) -> tuple[bool, float]:
        """Check if daily loss exceeds threshold.

        Returns (should_stop, daily_pnl_pct).
        """
        if self._daily_start_equity <= 0:
            return False, 0.0
        daily_pnl = (current_equity - self._daily_start_equity) / self._daily_start_equity
        return daily_pnl <= self.max_daily_loss_pct, daily_pnl


class RiskAgent(BaseAgent):
    """风控 Agent — 汇总分析师信号，计算仓位和风险

    Risk controls:
    1. Signal consensus (60% agreement threshold)
    2. Position sizing (max_position_pct, adjusted by confidence)
    3. Portfolio heat check (max_portfolio_risk enforcement)
    4. T+1 settlement enforcement
    5. Daily loss circuit breaker
    6. Stop-loss / take-profit price calculation
    """

    def __init__(
        self,
        max_position: float | None = None,
        stop_loss: float | None = None,
        take_profit_1: float | None = None,
        take_profit_2: float | None = None,
        max_portfolio_risk: float | None = None,
        max_daily_loss_pct: float | None = None,
        settings: Optional["Settings"] = None,
        llm_client: Optional["LLMClient"] = None,
        **kwargs,
    ):
        super().__init__(name="risk", **kwargs)
        self.max_position = max_position if max_position is not None else (
            settings.max_position_pct if settings else 0.20
        )
        self.stop_loss = stop_loss if stop_loss is not None else (
            settings.default_stop_loss if settings else -0.08
        )
        self.take_profit_1 = take_profit_1 if take_profit_1 is not None else (
            settings.default_take_profit_1 if settings else 0.10
        )
        self.take_profit_2 = take_profit_2 if take_profit_2 is not None else (
            settings.default_take_profit_2 if settings else 0.20
        )
        self.max_portfolio_risk = max_portfolio_risk if max_portfolio_risk is not None else (
            settings.max_portfolio_risk if settings else 0.80
        )
        self.max_daily_loss_pct = float(max_daily_loss_pct if max_daily_loss_pct is not None else (
            settings.max_daily_loss_pct if settings else -0.03
        ))
        self._llm = llm_client
        self.t1_tracker = T1Tracker()
        self.daily_tracker = DailyPnLTracker(max_daily_loss_pct=self.max_daily_loss_pct)

    def analyze(
        self,
        stock_code: str,
        results: list[AgentResult] | None = None,
        *,
        current_positions: dict[str, float] | None = None,
        current_equity: float | None = None,
        current_date: str | None = None,
        **kwargs,
    ) -> AgentResult:
        """分析风险 — 输入多个分析师的结果

        Args:
            stock_code: 股票代码
            results: 分析师结果列表
            current_positions: 当前持仓 {code: position_value}
            current_equity: 当前总权益
            current_date: 当前交易日 (YYYY-MM-DD)
        """
        if results is None:
            results = []
        self._logger.info("风控分析: %s (%d 个信号)", stock_code, len(results))

        if not results:
            return AgentResult(
                agent_name=self.name, stock_code=stock_code,
                signal="HOLD", confidence=0.0,
                reasoning="无分析师信号",
                success=False,
            )

        # 1. 信号共识（仅计入成功的分析）
        successful = [r for r in results if r.success]
        if not successful:
            return AgentResult(
                agent_name=self.name, stock_code=stock_code,
                signal="HOLD", confidence=0.0,
                reasoning="所有分析师失败",
                success=False,
            )

        buy_count = sum(1 for r in successful if r.signal == "BUY")
        sell_count = sum(1 for r in successful if r.signal == "SELL")
        hold_count = sum(1 for r in successful if r.signal == "HOLD")

        # 平均信心度（仅成功的分析）
        confidences = [r.confidence for r in successful if r.confidence > 0]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        # 2. 共识信号
        if buy_count >= len(successful) * 0.6:
            consensus = "BUY"
        elif sell_count >= len(successful) * 0.6:
            consensus = "SELL"
        else:
            consensus = "HOLD"

        # 3. 仓位计算
        position = 0.0
        position_warnings: list[str] = []

        if consensus == "BUY":
            position = self.max_position * avg_confidence

            # 3a. Portfolio heat check — reduce position if portfolio is heavily invested
            if current_positions and current_equity and current_equity > 0:
                total_position_pct = sum(current_positions.values()) / current_equity
                available_pct = self.max_portfolio_risk - total_position_pct
                if available_pct <= 0:
                    position = 0.0
                    position_warnings.append(
                        f"组合仓位已达上限 ({total_position_pct:.1%}/{self.max_portfolio_risk:.1%})"
                    )
                else:
                    position = min(position, available_pct)

            position = min(self.max_position, position)

        # 4. T+1 enforcement for SELL
        if consensus == "SELL":
            if not self.t1_tracker.can_sell(stock_code, current_date):
                consensus = "HOLD"
                position_warnings.append(
                    f"T+1 限制: {stock_code} 当日买入不可卖出，信号降级为 HOLD"
                )

        # 5. Daily loss circuit breaker
        if current_equity and current_date:
            self.daily_tracker.update_date(current_date, current_equity)
            circuit_triggered, daily_pnl = self.daily_tracker.check_circuit_breaker(current_equity)
            if circuit_triggered:
                if consensus == "BUY":
                    consensus = "HOLD"
                    position = 0.0
                position_warnings.append(
                    f"日亏损熔断: 当日亏损 {daily_pnl:.2%} 超过阈值 {self.max_daily_loss_pct:.2%}"
                )

        # 6. 风险指标
        price = self._extract_price(results)

        signal_summary = {}
        for r in results:
            signal_summary[r.agent_name] = {
                "signal": r.signal,
                "confidence": r.confidence,
                "success": r.success,
            }

        reasoning_parts = [
            f"共识: {consensus} (买{buy_count}/卖{sell_count}/观{hold_count})",
            f"平均信心度: {avg_confidence:.0%}",
            f"建议仓位: {position:.1%}",
        ]
        if position_warnings:
            reasoning_parts.append("风控警告: " + "; ".join(position_warnings))
        if price:
            reasoning_parts.append(
                f"止损: {price * (1 + self.stop_loss):.2f} | "
                f"止盈1: {price * (1 + self.take_profit_1):.2f} | "
                f"止盈2: {price * (1 + self.take_profit_2):.2f}"
            )

        return AgentResult(
            agent_name=self.name,
            stock_code=stock_code,
            signal=consensus,
            confidence=avg_confidence,
            reasoning="; ".join(reasoning_parts),
            metrics={
                "position": round(position, 4),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "stop_loss": round(self.stop_loss, 4),
                "take_profit_1": self.take_profit_1,
                "take_profit_2": self.take_profit_2,
                "position_warnings": position_warnings,
            },
            scores={"signal_summary": signal_summary},
        )

    def _extract_price(self, results: list[AgentResult]) -> float | None:
        for r in results:
            price = r.metrics.get("current_price") or r.metrics.get("price")
            if price and price > 0:
                return float(price)
        return None

    def interpret_risk(
        self,
        stock_code: str,
        risk_result: AgentResult,
        analysis_results: list[AgentResult],
    ) -> str | None:
        """LLM 生成风险事件的自然语言解读

        仅在 llm_client 存在时调用。返回 Markdown 格式的风险解读文本。
        """
        if self._llm is None:
            return None

        from ..llm.prompts import RISK_INTERPRET_SYSTEM, RISK_INTERPRET_USER

        fund = next((r for r in analysis_results if r.agent_name == "fundamental"), None)
        tech = next((r for r in analysis_results if r.agent_name == "technical"), None)

        user_prompt = RISK_INTERPRET_USER.safe_substitute(
            stock_code=stock_code,
            risk_signal=risk_result.signal,
            confidence=f"{risk_result.confidence:.0%}",
            reasoning=risk_result.reasoning,
            position_pct=risk_result.metrics.get("position", 0) * 100,
            stop_loss=risk_result.metrics.get("stop_loss", -0.08) * 100,
            take_profit=risk_result.metrics.get("take_profit_2", 0.20) * 100,
            fund_signal=fund.signal if fund else "N/A",
            fund_conf=f"{fund.confidence:.0%}" if fund else "N/A",
            tech_signal=tech.signal if tech else "N/A",
            tech_conf=f"{tech.confidence:.0%}" if tech else "N/A",
        )

        return self._llm.invoke(RISK_INTERPRET_SYSTEM, user_prompt)
