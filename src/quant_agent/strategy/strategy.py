"""策略抽象 — 让「信号生成」成为一等公民，回测与实盘共用同一套逻辑。

设计动机
--------
原系统中「买/卖决策」硬编码在 ``RiskAgent.analyze`` 内部（60% 分析师共识 +
仓位计算），既不可插拔也不可参数化；而回测引擎 ``BacktestEngine`` 走的是另一
套信号阈值逻辑，两条路径互不通用，「回测结果无法代表实盘」。

本模块把「策略」定义为一个清晰的协议：

    Strategy.generate_signal(ctx: StrategyContext) -> Signal

- ``ConsensusStrategy``：复刻原 RiskAgent 的「分析师共识 → 信号 + 仓位」逻辑，
  供实盘流水线（Orchestrator → RiskAgent）使用。它消费分析师 ``AgentResult``。
- ``CrossOverStrategy``：基于价格动量的简单策略，供 ``BacktestEngine`` 直接驱动，
  演示「同一抽象即可注入回测」。

任何新策略只要实现 ``Strategy`` 协议，就能同时被实盘与回测复用。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..agents.base import AgentResult


@dataclass
class StrategyContext:
    """策略生成信号所需的上下文（按需填充，实盘与回测各取所需）。

    - 实盘（ConsensusStrategy）：``results`` + ``current_positions`` +
      ``current_equity`` + ``current_date``。
    - 回测（CrossOverStrategy）：``price`` + ``prev_price`` + ``has_position``。
    """

    results: list[AgentResult] | None = None
    current_positions: dict[str, float] | None = None  # {code: position_value}
    current_equity: float | None = None
    current_date: str | None = None  # YYYY-MM-DD
    price: float | None = None
    prev_price: float | None = None
    has_position: bool = False


@dataclass
class Signal:
    """策略产出的标准化信号。

    Attributes:
        signal: "BUY" / "SELL" / "HOLD"
        confidence: 0~1 置信度
        position_pct: 建议仓位占比（0~1），仅 BUY 时有效
        stop_loss: 止损比例（负，如 -0.08）
        take_profit_1 / take_profit_2: 止盈比例（正）
        metrics: 供上层风控/审计使用的附加指标
    """

    signal: str
    confidence: float = 0.0
    position_pct: float = 0.0
    stop_loss: float = -0.08
    take_profit_1: float = 0.10
    take_profit_2: float = 0.20
    metrics: dict = field(default_factory=dict)


@runtime_checkable
class Strategy(Protocol):
    """策略协议 —— 实盘与回测共用。"""

    name: str

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        """根据上下文生成交易信号。"""
        ...


class ConsensusStrategy:
    """分析师共识策略 —— 复刻原 RiskAgent 的共识 + 仓位逻辑。

    仅负责「该买/卖/观望 + 建议仓位」（策略层），不含 T+1、熔断等
    组合级风控约束（那是 RiskEngine 的职责，在 RiskAgent 中施加）。

    与原 RiskAgent.analyze 的步骤 1~3 完全等价，保证行为一致。
    """

    def __init__(
        self,
        max_position: float = 0.20,
        stop_loss: float = -0.08,
        take_profit_1: float = 0.10,
        take_profit_2: float = 0.20,
        consensus_threshold: float = 0.6,
    ):
        self.max_position = max_position
        self.stop_loss = stop_loss
        self.take_profit_1 = take_profit_1
        self.take_profit_2 = take_profit_2
        self.consensus_threshold = consensus_threshold
        self._name = "consensus"

    @property
    def name(self) -> str:
        return self._name

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        results = ctx.results or []
        # 仅计入成功的分析
        successful = [r for r in results if getattr(r, "success", False)]
        if not successful:
            return Signal(
                signal="HOLD",
                confidence=0.0,
                metrics={"buy_count": 0, "sell_count": 0, "hold_count": 0},
            )

        buy_count = sum(1 for r in successful if r.signal == "BUY")
        sell_count = sum(1 for r in successful if r.signal == "SELL")
        hold_count = sum(1 for r in successful if r.signal == "HOLD")

        confidences = [r.confidence for r in successful if r.confidence > 0]
        avg_confidence = statistics.mean(confidences) if confidences else 0.0

        # 2. 共识信号
        if buy_count >= len(successful) * self.consensus_threshold:
            consensus = "BUY"
        elif sell_count >= len(successful) * self.consensus_threshold:
            consensus = "SELL"
        else:
            consensus = "HOLD"

        # 3. 仓位计算（受组合热度约束）
        position = 0.0
        if consensus == "BUY":
            position = self.max_position * avg_confidence
            cp = ctx.current_positions
            ce = ctx.current_equity
            if cp and ce and ce > 0:
                # 上层会传入 max_portfolio_risk；此处仅做基本热度裁剪占位
                position = min(self.max_position, position)

        return Signal(
            signal=consensus,
            confidence=avg_confidence,
            position_pct=position,
            stop_loss=self.stop_loss,
            take_profit_1=self.take_profit_1,
            take_profit_2=self.take_profit_2,
            metrics={
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
            },
        )


class CrossOverStrategy:
    """价格动量交叉策略 —— 供回测引擎直接驱动的最小示例。

    仅在「无持仓且价格上涨」时买入，在「有持仓且价格下跌」时卖出，
    演示 ``Strategy`` 抽象如何无缝注入 ``BacktestEngine``。
    """

    def __init__(
        self,
        buy_on_rise: bool = True,
        sell_on_fall: bool = True,
        stop_loss: float = -0.08,
        take_profit: float = 0.20,
    ):
        self.buy_on_rise = buy_on_rise
        self.sell_on_fall = sell_on_fall
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self._name = "crossover"

    @property
    def name(self) -> str:
        return self._name

    def generate_signal(self, ctx: StrategyContext) -> Signal:
        price = ctx.price
        prev = ctx.prev_price
        if price is None or prev is None or prev <= 0:
            return Signal(signal="HOLD")

        rising = price > prev
        falling = price < prev

        if not ctx.has_position and self.buy_on_rise and rising:
            return Signal(
                signal="BUY",
                confidence=0.6,
                position_pct=1.0,
                stop_loss=self.stop_loss,
                take_profit_1=self.take_profit,
                take_profit_2=self.take_profit,
            )
        if ctx.has_position and self.sell_on_fall and falling:
            return Signal(signal="SELL", confidence=0.6)
        return Signal(signal="HOLD")
