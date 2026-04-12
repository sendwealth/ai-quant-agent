"""Orchestrator -- 完整分析流水线编排器"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .audit import AuditLogger
from .config import Settings, get_settings
from .data.service import DataService
from .data.validators import validate_stock_code
from .llm.client import LLMClient, LLMError
from .llm.report import LLMReportGenerator
from .agents.base import AgentResult
from .agents.fundamental import FundamentalAgent
from .agents.technical import TechnicalAgent
from .agents.sentiment import SentimentAgent
from .agents.planner import PlannerAgent, ExecutionPlan
from .agents.risk import RiskAgent
from .agents.execution import ExecutionAgent, Order
from .notification.email import EmailNotifier
from .observability.metrics import MetricsCollector, HealthChecker

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """完整分析报告"""

    stock_code: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    fundamental_result: Optional[AgentResult] = None
    technical_result: Optional[AgentResult] = None
    sentiment_result: Optional[AgentResult] = None
    risk_result: Optional[AgentResult] = None
    execution_result: Optional[AgentResult] = None
    llm_analysis: Optional[str] = None
    risk_interpretation: Optional[str] = None
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stock_code": self.stock_code,
            "timestamp": self.timestamp,
            "signal": self.risk_result.signal if self.risk_result else "HOLD",
            "confidence": self.risk_result.confidence if self.risk_result else 0.0,
            "position_pct": self.risk_result.metrics.get("position", 0.0) if self.risk_result else 0.0,
            "fundamental": self.fundamental_result.to_dict() if self.fundamental_result else None,
            "technical": self.technical_result.to_dict() if self.technical_result else None,
            "sentiment": self.sentiment_result.to_dict() if self.sentiment_result else None,
            "risk": self.risk_result.to_dict() if self.risk_result else None,
            "execution": self.execution_result.to_dict() if self.execution_result else None,
            "llm_analysis": self.llm_analysis,
            "risk_interpretation": self.risk_interpretation,
            "summary": self.summary,
        }


class Orchestrator:
    """分析流水线编排器

    将原来 run_pipeline() 中的初始化、分析、执行逻辑封装为可复用的类。
    支持 LLM 增强：情感分析、智能指令解析、综合报告生成、风险解读。
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.metrics = MetricsCollector()
        self.data = DataService(self.settings)

        # Execution lock: serializes the risk+execution phase in analyze()
        # so that concurrent calls via analyze_batch don't corrupt shared
        # portfolio state (positions, cash, orders) in ExecutionAgent.
        self._execution_lock = threading.Lock()

        # 健康检查 — verify actual connectivity, not just object existence
        self.health = HealthChecker()
        self.health.register("data_service", self._check_data_service)
        self.health.register("llm", lambda: self.llm is not None)

        # 审计日志
        audit_dir = f"{self.settings.data_dir}/audit"
        self.audit_logger = AuditLogger(log_dir=audit_dir)

        # LLM 客户端 (可选 — 无 API key 时跳过 LLM 功能)
        self.llm: Optional[LLMClient] = None
        try:
            self.llm = LLMClient(self.settings)
        except LLMError:
            logger.info("LLM 未配置 (无 API key)，跳过 LLM 增强功能")

        # 初始化 Agent 团队
        self.fundamental = FundamentalAgent(data_service=self.data)
        self.technical = TechnicalAgent(data_service=self.data)
        self.sentiment = SentimentAgent(data_service=self.data, llm_client=self.llm)
        self.planner = PlannerAgent(llm_client=self.llm)
        self.risk = RiskAgent(settings=self.settings, llm_client=self.llm)
        self.execution = ExecutionAgent(
            initial_capital=self.settings.initial_capital,
            settings=self.settings,
            audit_logger=self.audit_logger,
        )

        # LLM 报告生成器
        self.report_gen: Optional[LLMReportGenerator] = None
        if self.llm:
            self.report_gen = LLMReportGenerator(self.llm)

        # 邮件通知
        self.notifier = EmailNotifier(self.settings)

        # 选股引擎 (lazy — only instantiated when needed)
        self._screener = None

    def _check_data_service(self) -> bool:
        """Real health check: verify at least one data source is usable."""
        try:
            return bool(self.data and self.data._sources)
        except Exception:
            return False

    @property
    def screener(self):
        """Lazy-loaded ScreeningEngine."""
        if self._screener is None:
            from .screener import ScreeningEngine
            self._screener = ScreeningEngine(
                data_service=self.data,
                settings=self.settings,
            )
        return self._screener

    @staticmethod
    def _run_agent(name: str, agent, stock_code: str) -> AgentResult:
        """Run a single agent, catching exceptions and returning AgentResult."""
        try:
            return agent.analyze(stock_code)
        except Exception as e:
            logger.warning("Agent %s failed: %s", name, e)
            return AgentResult(
                agent_name=name, stock_code=stock_code,
                signal="HOLD", confidence=0.0,
                reasoning=f"Agent failed: {e}", success=False, error=str(e),
            )

    def _analyze_safe(self, code: str, days: int) -> AnalysisReport:
        """Wrapper for batch analysis that propagates exceptions."""
        return self.analyze(code, days)

    def analyze(self, stock_code: str, days: int = 120) -> AnalysisReport:
        """运行完整分析流水线

        Args:
            stock_code: A 股代码 (6 位数字, 沪深创北)
            days: 分析天数 (默认 120)

        Returns:
            AnalysisReport 包含各 Agent 的分析结果

        Raises:
            ValueError: stock_code 格式不合法
        """
        # a. 验证 stock_code
        stock_code = validate_stock_code(stock_code)

        logger.info("=" * 60)
        logger.info("AI Quant Agent v3.0")
        logger.info("=" * 60)
        logger.info("Analyzing %s", stock_code)
        logger.info("-" * 40)

        report = AnalysisReport(stock_code=stock_code)

        # b. 分析师分析 (含 sentiment)
        analysis_results: list[AgentResult] = []
        agent_list = [
            ("fundamental", self.fundamental),
            ("technical", self.technical),
            ("sentiment", self.sentiment),
        ]

        with self.metrics.timer("analysis", {"stock": stock_code}):
            # Run independent analysis agents in parallel
            result_map: dict[str, AgentResult] = {}
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures = {}
                for name, agent in agent_list:
                    futures[pool.submit(self._run_agent, name, agent, stock_code)] = name

                for future in as_completed(futures):
                    name = futures[future]
                    result = future.result()
                    result_map[name] = result

            # Collect results in deterministic order
            for name, _ in agent_list:
                result = result_map[name]
                analysis_results.append(result)
                if name == "fundamental":
                    report.fundamental_result = result
                elif name == "technical":
                    report.technical_result = result
                elif name == "sentiment":
                    report.sentiment_result = result

                status = "OK" if result.success else "FAIL"
                logger.info("  %s %s: %s (%.0f%%)", status, result.agent_name, result.signal, result.confidence)
                self.metrics.counter("analysis.runs", 1, {"agent": name, "success": str(result.success)})

        # c. 风控汇总 — pass portfolio context for portfolio-level risk controls
        #    The execution phase (risk → stop-check → execute → summary) must be
        #    serialized because all threads share the same ExecutionAgent portfolio.
        logger.info("-" * 40)

        with self._execution_lock:
            # Gather current portfolio state for risk agent
            current_positions = {
                code: pos.shares * pos.current_price
                for code, pos in self.execution.positions.items()
            }
            current_equity = self.execution.total_equity
            current_date = datetime.now().strftime("%Y-%m-%d")

            risk_result = self.risk.analyze(
                stock_code, analysis_results,
                current_positions=current_positions,
                current_equity=current_equity,
                current_date=current_date,
            )
            report.risk_result = risk_result
            logger.info("  Risk: %s (position %.1f%%)", risk_result.signal, risk_result.metrics.get("position", 0))
            logger.info("     %s", risk_result.reasoning)

            # c2. LLM 风险解读 (可选)
            if self.llm:
                try:
                    report.risk_interpretation = self.risk.interpret_risk(
                        stock_code, risk_result, analysis_results
                    )
                    if report.risk_interpretation:
                        logger.info("  Risk Interpretation: %s", report.risk_interpretation[:100])
                except LLMError as e:
                    logger.warning("LLM 风险解读失败: %s", e)

            # d. Check stop-loss/take-profit on existing positions BEFORE new trades
            for code in list(self.execution.positions.keys()):
                pos = self.execution.positions[code]
                stop_order = self.execution.check_stop_conditions(code, pos.current_price)
                if stop_order:
                    logger.warning("  Stop triggered for %s: %s %d @ %.2f",
                                   code, stop_order.direction, stop_order.shares, stop_order.filled_price)
                    self.risk.t1_tracker.clear(code)

            # e. 执行 (仅在 risk 信号为 BUY/SELL 时)
            position_pct = risk_result.metrics.get("position", 0.0)
            current_price = 0.0
            for r in analysis_results:
                if current_price > 0:
                    break
                for key in ("current_price", "price"):
                    val = r.metrics.get(key)
                    if val is not None and val > 0:
                        current_price = float(val)
                        break

            all_results = analysis_results + [risk_result]

            if risk_result.signal == "BUY" and position_pct > 0 and current_price > 0:
                order = self.execution.execute_signal(
                    stock_code, "BUY",
                    position_pct=position_pct,
                    current_price=current_price,
                    stop_loss_pct=risk_result.metrics.get("stop_loss", -0.08),
                    take_profit_pct=risk_result.metrics.get("take_profit_2", 0.20),
                    agent_results=all_results,
                )
                if order and order.status == "filled":
                    logger.info("  BUY executed: %s %d shares @ %.2f", stock_code, order.shares, order.filled_price)
                    # Record buy for T+1 enforcement
                    self.risk.t1_tracker.record_buy(stock_code, current_date)
            elif risk_result.signal == "SELL":
                order = self.execution.execute_signal(
                    stock_code, "SELL",
                    current_price=current_price,
                    agent_results=all_results,
                )
                if order and order.status == "filled":
                    logger.info("  SELL executed: %s %d shares @ %.2f", stock_code, order.shares, order.filled_price)
                    self.risk.t1_tracker.clear(stock_code)
            else:
                self.execution.execute_signal(
                    stock_code, risk_result.signal,
                    position_pct=position_pct,
                    current_price=current_price,
                    agent_results=all_results,
                )
                logger.info("  No trade (signal=%s)", risk_result.signal)

            # e. 组合状态
            summary = self.execution.get_summary()
            report.summary = summary
        logger.info("")
        logger.info("Portfolio:")
        logger.info("  Total equity: %.2f", summary["total_equity"])
        logger.info("  Cash: %.2f", summary["cash"])
        logger.info("  Position value: %.2f", summary["position_value"])
        logger.info("  Return: %.2f%%", summary["total_return"] * 100)

        # f. LLM 综合报告 (可选)
        if self.report_gen:
            try:
                report.llm_analysis = self.report_gen.generate(report)
                logger.info("  LLM Report: generated (%d chars)", len(report.llm_analysis))
            except LLMError as e:
                logger.warning("LLM 报告生成失败: %s", e)

        # 指标
        self.metrics.gauge("portfolio.equity", summary["total_equity"], {"stock": stock_code})
        self.metrics.gauge("portfolio.return", summary["total_return"], {"stock": stock_code})

        # 健康检查
        health_status = self.health.check_all()
        logger.info("")
        logger.info("Health: %s", "OK" if health_status["healthy"] else "FAIL")
        logger.info("Pipeline complete (agents use structured logging)")

        # 邮件通知 (BUY/SELL 信号)
        if report.risk_result and report.risk_result.signal in ("BUY", "SELL"):
            self.notifier.send_trade_signal(report)

        return report

    def analyze_prompt(self, user_input: str) -> AnalysisReport:
        """自然语言分析入口

        Args:
            user_input: 自然语言指令，如 "分析宁德时代的买入机会"

        Returns:
            AnalysisReport
        """
        if not self.llm:
            raise LLMError("LLM 未配置，无法解析自然语言指令")

        plan: ExecutionPlan = self.planner.parse_intent(user_input)
        if not plan.stock_code:
            raise ValueError(f"无法从指令中识别股票代码: {user_input}")

        logger.info("Parsed intent: stock=%s days=%d focus=%s",
                     plan.stock_code, plan.days, plan.focus_areas)
        return self.analyze(plan.stock_code, days=plan.days)

    def analyze_batch(self, stock_codes: list[str], days: int = 120) -> list[AnalysisReport]:
        """批量分析多只股票并发送每日报告邮件

        Args:
            stock_codes: 股票代码列表
            days: 分析天数

        Returns:
            各股票的 AnalysisReport 列表（按输入顺序）
        """
        reports: dict[str, AnalysisReport] = {}
        errors: dict[str, str] = {}

        max_workers = min(len(stock_codes), self.settings.fetch_max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for code in stock_codes:
                futures[pool.submit(self._analyze_safe, code, days)] = code

            for future in as_completed(futures):
                code = futures[future]
                try:
                    reports[code] = future.result()
                except Exception as e:
                    logger.error("分析 %s 失败: %s", code, e)
                    errors[code] = str(e)
                    self.notifier.send_error_alert(
                        f"分析 {code} 失败",
                        {"stock_code": code, "error": str(e)},
                    )

        # Return in original order, skipping failures
        ordered = [reports[code] for code in stock_codes if code in reports]

        # 发送每日报告
        if ordered:
            self.notifier.send_daily_report(ordered, self.execution.get_summary())

        return ordered

    def screen_and_analyze(
        self,
        use_full_market: bool = False,
        top_n: int = 10,
        include_fundamentals: bool = False,
        analyze_days: int = 120,
    ) -> tuple:
        """选股 + 深度分析一体化

        Two-phase pipeline:
          Phase 1: ScreeningEngine.screen() → top N stock codes
          Phase 2: Orchestrator.analyze_batch(top codes) → AnalysisReports

        Args:
            use_full_market: Scan all A-shares (slow) vs hardcoded pool
            top_n: How many top-scored stocks to deep-analyze
            include_fundamentals: Include fundamental scoring in screening
            analyze_days: Days of history for deep analysis

        Returns:
            (ScreeningResult, list[AnalysisReport])
        """
        # Phase 1: Screen
        screen_result = self.screener.screen(
            use_full_market=use_full_market,
            top_n=top_n,
            include_fundamentals=include_fundamentals,
            days=analyze_days,
        )

        if not screen_result.top_stocks:
            logger.warning("选股无结果")
            return screen_result, []

        codes = [s.stock_code for s in screen_result.top_stocks]
        logger.info("选股 Top %d: %s", len(codes), ", ".join(codes))

        # Phase 2: Deep analyze
        reports = self.analyze_batch(codes, days=analyze_days)

        return screen_result, reports
