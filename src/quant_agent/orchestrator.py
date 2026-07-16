"""Orchestrator -- 完整分析流水线编排器"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime

from .agents.base import AgentResult, BaseAgent
from .agents.execution import ExecutionAgent
from .agents.fundamental import FundamentalAgent
from .agents.planner import ExecutionPlan, PlannerAgent
from .agents.risk import RiskAgent
from .agents.sentiment import SentimentAgent
from .agents.technical import TechnicalAgent
from .audit import AuditLogger
from .config import Settings, get_settings
from .data.service import DataService
from .data.validators import validate_stock_code
from .llm.client import LLMClient, LLMError, get_llm_client_soft
from .llm.report import LLMReportGenerator
from .notification.email import EmailNotifier
from .observability.metrics import HealthChecker, MetricsCollector
from .trading.service import TradingService

logger = logging.getLogger(__name__)


@dataclass
class AnalysisReport:
    """完整分析报告"""

    stock_code: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    fundamental_result: AgentResult | None = None
    technical_result: AgentResult | None = None
    sentiment_result: AgentResult | None = None
    risk_result: AgentResult | None = None
    execution_result: AgentResult | None = None
    llm_analysis: str | None = None
    risk_interpretation: str | None = None
    summary: dict = field(default_factory=dict)
    # 数据谱系 (P3)：本次分析所依赖数据的来源与时间记录
    data_lineage: list = field(default_factory=list)

    def lineage_warnings(self) -> list[str]:
        """汇总数据谱系中的显著警示（样例/缓存/合并/缺失/降级）。

        用于报告顶部显著提示，确保「用了什么数据、可信度如何」对用户透明。
        """
        warnings: list[str] = []
        seen: set[str] = set()
        for prov in self.data_lineage:
            reasons = prov.warning_reasons() if hasattr(prov, "warning_reasons") else []
            for reason in reasons:
                if reason not in seen:
                    seen.add(reason)
                    warnings.append(reason)
        return warnings

    def to_dict(self) -> dict:
        lineage = [p.to_dict() for p in self.data_lineage] if self.data_lineage else []
        return {
            "stock_code": self.stock_code,
            "timestamp": self.timestamp,
            "signal": self.risk_result.signal if self.risk_result else "HOLD",
            "confidence": self.risk_result.confidence if self.risk_result else 0.0,
            "position_pct": self.risk_result.metrics.get("position", 0.0)
            if self.risk_result
            else 0.0,
            "fundamental": self.fundamental_result.to_dict() if self.fundamental_result else None,
            "technical": self.technical_result.to_dict() if self.technical_result else None,
            "sentiment": self.sentiment_result.to_dict() if self.sentiment_result else None,
            "risk": self.risk_result.to_dict() if self.risk_result else None,
            "execution": self.execution_result.to_dict() if self.execution_result else None,
            "llm_analysis": self.llm_analysis,
            "risk_interpretation": self.risk_interpretation,
            "summary": self.summary,
            "data_lineage": lineage,
            "lineage_warnings": self.lineage_warnings(),
        }


class Orchestrator:
    """分析流水线编排器

    将原来 run_pipeline() 中的初始化、分析、执行逻辑封装为可复用的类。
    支持 LLM 增强：情感分析、智能指令解析、综合报告生成、风险解读。
    """

    def __init__(self, settings: Settings | None = None, offline: bool | None = None):
        base = settings or get_settings()
        # 复制一份 settings，避免覆盖进程级单例（get_settings 被 lru_cache 缓存）。
        # offline 为 None 时沿用全局设置；为 True/False 时以运行时参数为准
        # （Web 端「离线」勾选框按请求实时切换在线/离线）。
        self.settings = base.model_copy(deep=False)
        if offline is not None:
            self.settings.offline_mode = bool(offline)
        self.metrics = MetricsCollector()
        self.data = DataService(self.settings)

        # Execution lock: serializes the risk+execution phase in analyze()
        # so that concurrent calls via analyze_batch don't corrupt shared
        # portfolio state (positions, cash, orders) in ExecutionAgent.
        self._execution_lock = threading.Lock()

        # 健康检查 — verify actual connectivity, not just object existence
        self.health = HealthChecker()
        self.health.register("data_service", self._check_data_service)
        self.health.register("llm", lambda: self.llm is not None and self.llm.enabled)

        # 审计日志
        audit_dir = f"{self.settings.data_dir}/audit"
        self.audit_logger = AuditLogger(log_dir=audit_dir)

        # LLM 客户端 (软降级 — 无 API key 时进入离线规则增强模式，功能不中断)
        self.llm: LLMClient = get_llm_client_soft()

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
            persist_dir=self.settings.data_dir if self.settings.persist_trading else None,
        )

        # LLM 报告生成器
        self.report_gen: LLMReportGenerator | None = None
        if self.llm:
            self.report_gen = LLMReportGenerator(self.llm)

        # 邮件通知
        self.notifier = EmailNotifier(self.settings)

        # 交易执行服务（显式触发，与分析解耦）
        self.trading = TradingService(
            execution=self.execution,
            risk=self.risk,
            notifier=self.notifier,
        )

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
    def _run_agent(name: str, agent: BaseAgent, stock_code: str) -> AgentResult:
        """Run a single agent, catching exceptions and returning AgentResult."""
        try:
            return agent.analyze(stock_code)
        except Exception as e:
            logger.warning("Agent %s failed: %s", name, e)
            return AgentResult(
                agent_name=name,
                stock_code=stock_code,
                signal="HOLD",
                confidence=0.0,
                reasoning=f"Agent failed: {e}",
                success=False,
                error=str(e),
            )

    def _analyze_safe(self, code: str, days: int) -> AnalysisReport:
        """Wrapper for batch analysis that propagates exceptions."""
        return self.analyze(code, days)

    def analyze(
        self,
        stock_code: str,
        days: int = 120,
        execute: bool = True,
        research_mode: bool = False,
    ) -> AnalysisReport:
        """运行完整分析流水线

        Args:
            stock_code: A 股代码 (6 位数字, 沪深创北)
            days: 分析天数 (默认 120)
            execute: 是否根据共识信号显式下单。
                默认 True（CLI/批量等显式操作）；Web 预览等只读场景应传 False，
                避免「点一下分析就建仓」。
            research_mode: 显式研究模式，透传给 :class:`TradingService`；仅当
                报告**无数据谱系**时生效，用于模拟/测试场景的豁免。

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
                logger.info(
                    "  %s %s: %s (%.0f%%)",
                    status,
                    result.agent_name,
                    result.signal,
                    result.confidence,
                )
                self.metrics.counter(
                    "analysis.runs", 1, {"agent": name, "success": str(result.success)}
                )

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
                stock_code,
                analysis_results,
                current_positions=current_positions,
                current_equity=current_equity,
                current_date=current_date,
            )
            report.risk_result = risk_result
            logger.info(
                "  Risk: %s (position %.1f%%)",
                risk_result.signal,
                risk_result.metrics.get("position", 0),
            )
            logger.info("     %s", risk_result.reasoning)

            # c2. LLM 风险解读 (可选)
            if self.llm and self.llm.enabled:
                try:
                    report.risk_interpretation = self.risk.interpret_risk(
                        stock_code, risk_result, analysis_results
                    )
                    if report.risk_interpretation:
                        logger.info("  Risk Interpretation: %s", report.risk_interpretation[:100])
                except LLMError as e:
                    logger.warning("LLM 风险解读失败: %s", e)

            # e2. 数据谱系：先汇总本次分析所依赖数据的来源与时间（P3 透明展示），
            # 必须在交易执行前完成，以便数据可信门禁（fail closed）可校验谱系。
            report.data_lineage = self.data.get_lineage(stock_code)

            # d~e. 仅当显式要求时才执行交易（默认从 Web 预览入口为 False，
            # 避免「点一下分析就建仓」）。交易副作用全部交由 TradingService。
            if execute:
                self.trading.execute(
                    report, analysis_results, current_date, research_mode=research_mode
                )
                if risk_result.signal in ("BUY", "SELL"):
                    self.notifier.send_trade_signal(report)

            # 组合状态（始终读取，便于预览展示；只读不写）
            summary = self.execution.get_summary()
            report.summary = summary
        logger.info("")
        logger.info("Portfolio:")
        logger.info("  Total equity: %.2f", summary["total_equity"])
        logger.info("  Cash: %.2f", summary["cash"])
        logger.info("  Position value: %.2f", summary["position_value"])
        logger.info("  Return: %.2f%%", summary["total_return"] * 100)

        # f. LLM 综合报告 (可选)
        if self.report_gen and self.llm and self.llm.enabled:
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

        return report

    def analyze_prompt(self, user_input: str) -> AnalysisReport:
        """自然语言分析入口

        Args:
            user_input: 自然语言指令，如 "分析宁德时代的买入机会"

        Returns:
            AnalysisReport
        """
        if not self.llm or not self.llm.enabled:
            raise LLMError("LLM 未配置，无法解析自然语言指令（请配置 API key 或本地模型）")

        plan: ExecutionPlan = self.planner.parse_intent(user_input)
        if not plan.stock_code:
            raise ValueError(f"无法从指令中识别股票代码: {user_input}")

        logger.info(
            "Parsed intent: stock=%s days=%d focus=%s", plan.stock_code, plan.days, plan.focus_areas
        )
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
