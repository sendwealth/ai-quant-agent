"""数据可信门禁 (Data Trust Gate) — 推荐 #2

对 ``sample``（合成样例）与 ``low``（低可信度）数据建立硬门禁：

- **禁止**进入「交易决策」与「回测绩效」路径（硬阻断，抛
  :class:`DataTrustError`）；
- **允许**用于分析/选股等只读用途，但必须在报告中显著标红（水印）。

所有判定基于 :class:`quant_agent.data.sources.base.DataProvenance` 的
``source`` 与 ``confidence`` 字段，与现有数据谱系体系一致。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .sources.base import DataProvenance

# 禁止进入「交易决策 / 回测绩效」的来源（合成样例）与可信度（低）
FORBIDDEN_SOURCES_FOR_DECISION = frozenset({"sample"})
FORBIDDEN_CONFIDENCE_FOR_DECISION = frozenset({"low"})

# 仅允许只读用途、但需在报告中高亮（软警示）的来源 / 可信度
WATCH_SOURCES = frozenset({"cache", "merged"})
WATCH_CONFIDENCE = frozenset({"partial"})

# 决策用途：交易 / 回测（受硬门禁约束）；其余为只读用途
DECISION_PURPOSES = frozenset({"trading", "backtest"})

# 可信度排序（数值越大越不可信）
_CONF_RANK = {"high": 0, "partial": 1, "low": 2, "unknown": 3}


class DataTrustError(Exception):
    """数据不可信，禁止进入交易/回测决策路径。"""


@dataclass
class TrustVerdict:
    """数据可信判定结果（可序列化，便于写入报告 / 审计）。"""

    allowed: bool
    purpose: str
    level: str  # 最差可信度：high / partial / low / unknown
    sources: list[str]
    confidence: list[str]
    reasons: list[str] = field(default_factory=list)

    def require(self) -> TrustVerdict:
        """若不可信则抛出 :class:`DataTrustError`。"""
        if not self.allowed:
            raise DataTrustError("; ".join(self.reasons) or "数据可信度不足，禁止进入决策路径")
        return self

    @property
    def blocked(self) -> bool:
        return not self.allowed

    @property
    def warning_text(self) -> str | None:
        """用于报告水印的简短警示（仅软警示 / 只读受限场景）。"""
        if self.allowed and self.reasons:
            return "；".join(self.reasons)
        return None


def evaluate_trust(
    provenance: Iterable[DataProvenance] | None,
    purpose: str,
    research_mode: bool = False,
) -> TrustVerdict:
    """评估一组数据的可信度是否满足给定用途。

    Args:
        provenance: 数据谱系列表（可为空）。
        purpose: 用途，取值 ``trading`` / ``backtest`` / ``report`` / ``screen``。
        research_mode: 是否处于显式研究模式。仅当决策用途（``trading`` /
            ``backtest``）**缺少数据谱系**时生效：开启后允许执行但显著标红，
            关闭（默认）则 fail closed —— 缺谱系即禁止进入交易/正式回测。

    Returns:
        TrustVerdict

    判定规则：

    - 决策用途（``trading`` / ``backtest``）下，出现 ``sample`` 来源或
      ``low`` 可信度 → ``allowed=False``，调用 :meth:`TrustVerdict.require`
      会抛 :class:`DataTrustError`。
    - 决策用途下**缺少数据谱系**（空 ``provenance``）：默认 fail closed
      （``allowed=False``）；仅当显式 ``research_mode=True`` 时放行并标红。
    - 只读用途下，同样数据 ``allowed=True``，但 ``reasons`` 会记录显著警示，
      供报告水印标红。
    """
    provs = list(provenance or [])
    is_decision = purpose in DECISION_PURPOSES

    if not provs:
        # 无谱系：决策用途 fail closed，除非显式研究模式豁免。
        if is_decision:
            if research_mode:
                return TrustVerdict(
                    allowed=True,
                    purpose=purpose,
                    level="unknown",
                    sources=[],
                    confidence=[],
                    reasons=[
                        "研究模式豁免：未提供数据谱系，"
                        f"{purpose} 结果仅供研究参考，不构成任何决策依据"
                    ],
                )
            return TrustVerdict(
                allowed=False,
                purpose=purpose,
                level="unknown",
                sources=[],
                confidence=[],
                reasons=[
                    "未提供数据谱系，fail closed："
                    f"{purpose} 默认拒绝执行；如需研究豁免请显式开启 research_mode"
                ],
            )
        # 只读用途：放行但显著标记
        return TrustVerdict(
            allowed=True,
            purpose=purpose,
            level="unknown",
            sources=[],
            confidence=[],
            reasons=["未提供数据谱系，无法验证可信度（仅供研究/只读参考）"],
        )

    reasons: list[str] = []
    sources: list[str] = []
    confs: list[str] = []
    hard_blocked = False

    for p in provs:
        sources.append(p.source)
        confs.append(p.confidence)
        if p.source in FORBIDDEN_SOURCES_FOR_DECISION:
            hard_blocked = True
            reasons.append(f"数据来源为合成样例({p.source})，禁止用于{purpose}")
        if p.confidence in FORBIDDEN_CONFIDENCE_FOR_DECISION:
            hard_blocked = True
            reasons.append(f"数据可信度为低({p.confidence})，禁止用于{purpose}")
        # 软警示（仅当未被硬阻断时叠加，避免重复噪音）
        if not hard_blocked:
            if p.source in WATCH_SOURCES:
                reasons.append(f"数据来源为{p.source}（非实时），结论仅供参考")
            if p.confidence in WATCH_CONFIDENCE:
                reasons.append(f"数据可信度为部分({p.confidence})，结论仅供参考")

    worst = max(confs, key=lambda c: _CONF_RANK.get(c, 3)) if confs else "unknown"

    if is_decision and hard_blocked:
        # 决策路径：硬阻断
        return TrustVerdict(
            allowed=False,
            purpose=purpose,
            level=worst,
            sources=sources,
            confidence=confs,
            reasons=reasons,
        )

    # 只读用途：放行但显著标记
    if hard_blocked:
        reasons.append("该报告基于受限/合成数据，不构成任何投资建议")
    return TrustVerdict(
        allowed=True,
        purpose=purpose,
        level=worst,
        sources=sources,
        confidence=confs,
        reasons=reasons,
    )
