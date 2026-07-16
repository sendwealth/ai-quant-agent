"""数据源冒烟测试 (P1.5) — 定时校验各数据源可达性。

轻量、可离线运行的「冒烟」检查：对每一个已配置数据源，尝试一次最小
请求（拉取一只已知股票的最近 N 日行情），返回结构化结果。结果可序列化，
便于接入定时工作流 / CI / 可观测性告警。

设计要点：
- 单源检查 :func:`smoke_test_source` 仅做一次 ``get_price_data``，不依赖
  网络以外的状态，可注入 mock 源做单元测试。
- :meth:`quant_agent.data.service.DataService.smoke_test` 遍历已构建数据源
  并聚合结果。
- :func:`smoke_report` 生成可读 + 可机器解析的汇总（含降级提示）。
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from ..observability.health import compute_data_health_score
from .sources.base import DataSource


@dataclass
class SourceSmokeResult:
    """单数据源冒烟结果（可序列化）。"""

    source: str
    ok: bool
    latency_ms: float = 0.0
    rows: int = 0
    error: str | None = None
    skipped: bool = False
    skip_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def smoke_test_source(
    source: DataSource,
    stock_code: str = "600519",
    days: int = 5,
) -> SourceSmokeResult:
    """对一个数据源做一次最小行情请求，返回结构化健康结果。

    仅做读取，不写入任何缓存或状态。网络/解析异常被捕获并记录到
    ``error``，不会抛出，便于定时任务稳定产出报告。
    """
    if not source.available:
        return SourceSmokeResult(
            source=source.name,
            ok=False,
            skipped=True,
            skip_reason="source not available (likely missing credentials/config)",
        )
    start = time.perf_counter()
    try:
        df = source.get_price_data(stock_code, days=days, adjust="qfq")
    except Exception as e:  # 网络/解析/超时等任何异常都视为失败
        elapsed = (time.perf_counter() - start) * 1000
        return SourceSmokeResult(
            source=source.name, ok=False, latency_ms=round(elapsed, 2), error=repr(e)
        )
    elapsed = (time.perf_counter() - start) * 1000
    rows = 0 if df is None else int(len(df))
    ok = df is not None and rows > 0
    error = None if ok else ("empty response" if df is not None else "no data returned")
    return SourceSmokeResult(
        source=source.name,
        ok=ok,
        latency_ms=round(elapsed, 2),
        rows=rows,
        error=error,
    )


def smoke_report(results: list[SourceSmokeResult]) -> dict[str, Any]:
    """聚合冒烟结果，生成汇总（含降级提示）。

    关键语义：
    - ``all_healthy``：无失败且至少 1 个真实源可用（CI 退出码依据）。
    - ``degraded``：没有任何真实源通过，分析将回退到内置样例/缓存；
      结论仅供参考，需在报告/告警中显著提示。
    """
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if not r.ok and not r.skipped)
    degraded = ok == 0
    return {
        "total": total,
        "ok": ok,
        "failed": failed,
        "skipped": skipped,
        "all_healthy": failed == 0 and ok > 0,
        "degraded": degraded,
        "degradation_note": (
            "无可用实时数据源，分析将回退到内置样例/缓存；结论仅供参考，非投资建议。"
            if degraded
            else None
        ),
        "results": [r.to_dict() for r in results],
        # 数据源健康评分（推荐 #3）：把冒烟结果量化为可机读分数，供 CI 告警与
        # 健康端点消费，而不只是文本/artifact。
        "data_health_score": compute_data_health_score({"results": [r.to_dict() for r in results]}),
    }
