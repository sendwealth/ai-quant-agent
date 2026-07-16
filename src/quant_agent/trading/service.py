"""TradingService — 显式执行层（应用服务）

把原 ``Orchestrator.analyze`` 中「风控 → 止损检查 → 下单 → 记录 T+1」这一段
交易副作用，从分析流程里剥离出来，成为可独立、显式触发的服务。

设计原则
--------
- **分析是纯只读**（``Orchestrator.analyze(execute=False)``），可无限并发、可缓存、可重放；
- **交易是显式动作**，必须拿到 ``AnalysisReport`` 的共识信号后才下单；
- 不直接读写全局状态，所有副作用都经由注入的 ``ExecutionAgent`` 完成。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from ..agents.execution import Order
from ..data.gate import DataTrustError, evaluate_trust

if TYPE_CHECKING:
    from ..agents.execution import ExecutionAgent
    from ..agents.risk import RiskAgent
    from ..notification.email import EmailNotifier
    from ..orchestrator import AnalysisReport

logger = logging.getLogger(__name__)


class TradingService:
    """显式交易执行服务。"""

    def __init__(
        self,
        execution: ExecutionAgent,
        risk: RiskAgent,
        notifier: EmailNotifier | None = None,
    ):
        self.execution = execution
        self.risk = risk
        self.notifier = notifier

    def execute(
        self,
        report: AnalysisReport,
        analysis_results,
        current_date: str | None = None,
        research_mode: bool = False,
    ) -> Order | None:
        """根据报告里的共识信号显式下单。返回成交订单（若无则为 None）。

        Args:
            research_mode: 显式研究模式。仅当报告**无数据谱系**时生效：开启后
                允许继续（仅用于研究/模拟），关闭（默认）则 fail closed —— 缺
                谱系即禁止进入交易决策路径。
        """
        # 数据可信门禁（推荐 #2）：合成/低可信度数据、或无谱系数据禁止进入交易决策。
        try:
            evaluate_trust(report.data_lineage, "trading", research_mode=research_mode).require()
        except DataTrustError as e:
            logger.warning("交易被数据可信门禁拦截: %s", e)
            return None

        risk_result = report.risk_result
        if risk_result is None:
            logger.info("无风控结果，跳过交易")
            return None

        stock_code = report.stock_code
        current_date = current_date or datetime.now().strftime("%Y-%m-%d")

        # d. 现有持仓止损/止盈检查（下单前）
        for code in list(self.execution.positions.keys()):
            pos = self.execution.positions[code]
            stop_order = self.execution.check_stop_conditions(code, pos.current_price)
            if stop_order:
                logger.warning(
                    "  Stop triggered for %s: %s %d @ %.2f",
                    code,
                    stop_order.direction,
                    stop_order.shares,
                    stop_order.filled_price,
                )
                self.risk.t1_tracker.clear(code)

        # e. 执行信号
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

        all_results = list(analysis_results) + [risk_result]
        order: Order | None = None

        if risk_result.signal == "BUY" and position_pct > 0 and current_price > 0:
            order = self.execution.execute_signal(
                stock_code,
                "BUY",
                position_pct=position_pct,
                current_price=current_price,
                stop_loss_pct=risk_result.metrics.get("stop_loss", -0.08),
                take_profit_pct=risk_result.metrics.get("take_profit_2", 0.20),
                agent_results=all_results,
            )
            if order and order.status == "filled":
                logger.info(
                    "  BUY executed: %s %d shares @ %.2f",
                    stock_code,
                    order.shares,
                    order.filled_price,
                )
                self.risk.t1_tracker.record_buy(stock_code, current_date)
        elif risk_result.signal == "SELL":
            order = self.execution.execute_signal(
                stock_code,
                "SELL",
                current_price=current_price,
                agent_results=all_results,
            )
            if order and order.status == "filled":
                logger.info(
                    "  SELL executed: %s %d shares @ %.2f",
                    stock_code,
                    order.shares,
                    order.filled_price,
                )
                self.risk.t1_tracker.clear(stock_code)
        else:
            self.execution.execute_signal(
                stock_code,
                risk_result.signal,
                position_pct=position_pct,
                current_price=current_price,
                agent_results=all_results,
            )
            logger.info("  No trade (signal=%s)", risk_result.signal)

        return order
