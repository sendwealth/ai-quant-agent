"""基本面分析 Agent — 基于真实财务数据"""

from __future__ import annotations

from typing import Any, Optional

from ..data.sources.base import FinancialSnapshot
from ..thresholds import _Thresh, get_thresholds
from .base import BaseAgent, AgentResult


class FundamentalAgent(BaseAgent):
    """基本面分析 Agent — 从真实财务报表计算指标"""

    def __init__(self, threshold_config: _Thresh | None = None, **kwargs):
        super().__init__(name="fundamental", **kwargs)
        self._thresh = threshold_config or get_thresholds().fundamental

    def analyze(self, stock_code: str, **kwargs) -> AgentResult:
        self._logger.info("基本面分析: %s", stock_code)
        self._log_action("analysis_started", stock_code=stock_code)

        try:
            snapshot = self._get_financial_data(stock_code)
            if snapshot is None:
                return AgentResult(
                    agent_name=self.name, stock_code=stock_code,
                    signal="HOLD", confidence=0.0,
                    reasoning="无法获取财务数据",
                    success=False, error="NO_DATA",
                )

            scores = self._calc_scores(snapshot)
            signal, confidence, reasoning = self._generate_signal(snapshot, scores)

            result = AgentResult(
                agent_name=self.name,
                stock_code=stock_code,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics={
                    "roe": snapshot.roe,
                    "pe_ttm": snapshot.pe_ttm,
                    "pb": snapshot.pb,
                    "gross_margin": snapshot.gross_margin,
                    "net_margin": snapshot.net_margin,
                    "debt_ratio": snapshot.debt_ratio,
                    "current_ratio": snapshot.current_ratio,
                    "revenue_growth": snapshot.revenue_growth,
                    "profit_growth": snapshot.profit_growth,
                    "price": snapshot.get("price"),
                    "report_date": snapshot.get("report_date"),
                },
                scores=scores,
            )
            self._log_action("analysis_completed", stock_code=stock_code, signal=signal, confidence=confidence)
            return result

        except Exception as e:
            self._logger.error("基本面分析失败: %s", e)
            self._log_action("analysis_failed", stock_code=stock_code, error=str(e))
            return AgentResult(
                agent_name=self.name, stock_code=stock_code,
                signal="HOLD", confidence=0.0,
                reasoning=f"分析失败: {e}",
                success=False, error=str(e),
            )

    def _get_financial_data(self, stock_code: str) -> Optional[FinancialSnapshot]:
        """获取财务数据"""
        if self.data_service is None:
            self._logger.warning("未配置 DataService")
            return None
        return self.data_service.get_financial_snapshot(stock_code)

    def _calc_scores(self, s: FinancialSnapshot) -> dict[str, Any]:
        """计算各项评分"""
        t = self._thresh

        # 盈利能力 (0-10)
        prof = t.profitability
        profitability = int(prof.get("base", 5))
        roe_exc = float(prof.get("roe_excellent", 0.15))
        roe_good = float(prof.get("roe_good", 0.10))
        nm_exc = float(prof.get("net_margin_excellent", 0.15))
        nm_poor = float(prof.get("net_margin_poor", 0.05))

        if s.roe and s.roe > roe_exc:
            profitability += int(prof.get("roe_excellent_score", 2))
        elif s.roe and s.roe > roe_good:
            profitability += int(prof.get("roe_good_score", 1))
        if s.net_margin and s.net_margin > nm_exc:
            profitability += int(prof.get("net_margin_excellent_score", 1))
        elif s.net_margin and s.net_margin < nm_poor:
            profitability += int(prof.get("net_margin_poor_score", -1))

        # 估值水平 (0-10, 越低越好)
        val = t.valuation
        valuation = int(val.get("base", 5))
        pe_cheap = float(val.get("pe_cheap", 15))
        pe_fair = float(val.get("pe_fair", 25))
        pe_expensive = float(val.get("pe_expensive", 35))
        pe_very_expensive = float(val.get("pe_very_expensive", 50))
        pb_low = float(val.get("pb_low", 2.0))
        pb_high = float(val.get("pb_high", 6.0))

        if s.pe_ttm is not None:
            if s.pe_ttm < 0:
                # Negative PE = loss-making company — penalize, don't treat as cheap
                valuation += int(val.get("pe_negative_score", -3))
            elif s.pe_ttm < pe_cheap:
                valuation += int(val.get("pe_cheap_score", 3))
            elif s.pe_ttm < pe_fair:
                valuation += int(val.get("pe_fair_score", 1))
            elif s.pe_ttm > pe_very_expensive:
                valuation += int(val.get("pe_very_expensive_score", -2))
            elif s.pe_ttm > pe_expensive:
                valuation += int(val.get("pe_expensive_score", -1))
        if s.pb is not None:
            if s.pb < pb_low:
                valuation += int(val.get("pb_low_score", 1))
            elif s.pb > pb_high:
                valuation += int(val.get("pb_high_score", -1))

        # 财务健康 (0-10)
        h = t.health
        health = int(h.get("base", 5))
        debt_low = float(h.get("debt_low", 0.4))
        debt_moderate = float(h.get("debt_moderate", 0.6))
        debt_high = float(h.get("debt_high", 0.8))
        cr_strong = float(h.get("current_ratio_strong", 2.0))
        cr_weak = float(h.get("current_ratio_weak", 1.0))

        if s.debt_ratio is not None:
            if s.debt_ratio < debt_low:
                health += int(h.get("debt_low_score", 2))
            elif s.debt_ratio < debt_moderate:
                health += int(h.get("debt_moderate_score", 1))
            elif s.debt_ratio > debt_high:
                health += int(h.get("debt_high_score", -2))
        if s.current_ratio is not None:
            if s.current_ratio > cr_strong:
                health += int(h.get("current_ratio_strong_score", 1))
            elif s.current_ratio < cr_weak:
                health += int(h.get("current_ratio_weak_score", -2))

        # 成长性 (0-10)
        g = t.growth
        growth = int(g.get("base", 5))
        rev_high = float(g.get("revenue_high", 0.30))
        rev_good = float(g.get("revenue_good", 0.15))
        profit_high = float(g.get("profit_high", 0.30))

        if s.revenue_growth is not None:
            if s.revenue_growth > rev_high:
                growth += int(g.get("revenue_high_score", 3))
            elif s.revenue_growth > rev_good:
                growth += int(g.get("revenue_good_score", 1))
            elif s.revenue_growth < 0:
                growth += int(g.get("revenue_negative_score", -2))
        if s.profit_growth is not None:
            if s.profit_growth > profit_high:
                growth += int(g.get("profit_high_score", 1))
            elif s.profit_growth < 0:
                growth += int(g.get("profit_negative_score", -1))

        prof_min = int(prof.get("min", 1))
        prof_max = int(prof.get("max", 10))
        val_min = int(val.get("min", 1))
        val_max = int(val.get("max", 10))
        h_min = int(h.get("min", 1))
        h_max = int(h.get("max", 10))
        g_min = int(g.get("min", 1))
        g_max = int(g.get("max", 10))

        return {
            "profitability": min(prof_max, max(prof_min, profitability)),
            "valuation": min(val_max, max(val_min, valuation)),
            "health": min(h_max, max(h_min, health)),
            "growth": min(g_max, max(g_min, growth)),
        }

    def _generate_signal(self, s: FinancialSnapshot, scores: dict) -> tuple[str, float, str]:
        """生成信号"""
        t = self._thresh.signal
        total_score = sum(scores.values()) / len(scores)

        excellent_score = float(t.get("excellent_score", 7.5))
        excellent_min_val = int(t.get("excellent_min_valuation", 6))
        exc_cap = float(t.get("excellent_confidence_cap", 0.90))
        exc_base = float(t.get("excellent_confidence_base", 0.60))
        exc_div = float(t.get("excellent_confidence_divisor", 50))

        good_score = float(t.get("good_score", 6.0))
        good_cap = float(t.get("good_confidence_cap", 0.80))
        good_base = float(t.get("good_confidence_base", 0.55))
        good_div = float(t.get("good_confidence_divisor", 50))

        fair_score = float(t.get("fair_score", 4.5))
        fair_conf = float(t.get("fair_confidence", 0.55))

        weak_score = float(t.get("weak_score", 3.0))
        weak_conf = float(t.get("weak_confidence", 0.50))

        poor_sell_conf = float(t.get("poor_sell_confidence", 0.65))

        if total_score >= excellent_score and scores["valuation"] >= excellent_min_val:
            return "BUY", min(exc_cap, exc_base + total_score / exc_div), f"基本面优秀(均分{total_score:.1f})，估值合理"
        elif total_score >= good_score:
            return "BUY", min(good_cap, good_base + total_score / good_div), f"基本面良好(均分{total_score:.1f})"
        elif total_score >= fair_score:
            return "HOLD", fair_conf, f"基本面一般(均分{total_score:.1f})"
        elif total_score >= weak_score:
            return "HOLD", weak_conf, f"基本面偏弱(均分{total_score:.1f})，建议观望"
        else:
            return "SELL", poor_sell_conf, f"基本面较差(均分{total_score:.1f})"
