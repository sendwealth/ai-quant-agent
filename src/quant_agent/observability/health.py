"""系统健康报告（P3.1）— 聚合组件健康与数据降级状态。

把数据源冒烟结果、LLM 可用性、运行模式等聚合成一份可机读的健康报告，
供 ``/api/health`` 端点与告警模块使用。设计目标是：即使不触网也能给出
「是否降级」的结论（离线模式即视为数据降级）。
"""

from __future__ import annotations

from typing import Any

from ..config import Settings


def score_source_result(result: dict[str, Any]) -> int | None:
    """对单条冒烟结果打分（0–100），可机读供健康评分/告警使用。

    - 成功(ok)：满分 100，按延迟轻微扣分（>1000ms 起每 100ms 扣 1 分，封顶 40）。
    - 失败(非跳过)：0 分。
    - 跳过(skipped，缺凭证/不可用)：中性，返回 ``None``，不计入整体均分，
      避免把「未配置 token」误判为数据源故障。

    对应建议 #3：把每日 smoke 结果变成「数据源健康评分」而非仅文本报告。
    """
    if result.get("skipped"):
        return None
    if result.get("ok"):
        latency = float(result.get("latency_ms") or 0)
        penalty = max(0.0, min(40.0, (latency - 1000.0) / 100.0))
        return int(100 - penalty)
    return 0


def compute_data_health_score(smoke_results: dict[str, Any] | None) -> dict[str, Any]:
    """聚合冒烟结果为数据源健康评分（推荐 #3）。

    Returns:
        dict，含 ``overall_score``（0–100，无真实源时为 0）、
        ``failed_sources``（真实失败源名，供告警）、``healthy/failed/skipped``
        计数与 ``per_source`` 映射。
    """
    if smoke_results is None:
        return {
            "overall_score": 0,
            "healthy_count": 0,
            "failed_count": 0,
            "skipped_count": 0,
            "failed_sources": [],
            "per_source": {},
        }
    results = smoke_results.get("results", []) or []
    scored: dict[str, int | None] = {r.get("source"): score_source_result(r) for r in results}
    real = [sc for sc in scored.values() if sc is not None]
    overall = int(sum(real) / len(real)) if real else 0
    failed_sources = [
        r.get("source") for r in results if not r.get("ok") and not r.get("skipped")
    ]
    return {
        "overall_score": overall,
        "healthy_count": sum(1 for r in results if r.get("ok")),
        "failed_count": len(failed_sources),
        "skipped_count": sum(1 for r in results if r.get("skipped")),
        "failed_sources": failed_sources,
        "per_source": dict(scored),
    }


def _data_source_summary(settings: Settings) -> dict[str, Any]:
    """基于配置推断已启用的数据源（不触网）。"""
    sources: list[dict[str, Any]] = []
    # 免费源默认可用
    for name in ("efinance", "akshare", "baostock"):
        sources.append({"source": name, "configured": True, "kind": "free"})
    # Tushare 需 token
    sources.append(
        {
            "source": "tushare",
            "configured": bool(getattr(settings, "tushare_token", None)),
            "kind": "token",
        }
    )
    configured = [s for s in sources if s["configured"]]
    return {
        "configured_count": len(configured),
        "sources": sources,
        "all_down": len(configured) == 0,
    }


def build_health_report(
    settings: Settings, smoke_results: dict[str, Any] | None = None
) -> dict[str, Any]:
    """聚合健康报告。

    Args:
        settings: 全局配置。
        smoke_results: 可选的数据源冒烟结果（见
            :func:`quant_agent.data.smoke.smoke_report`）。若提供，会以
            其 ``degraded`` 作为数据降级依据；否则以离线模式推断。
    """
    offline = bool(getattr(settings, "offline_mode", False))
    llm_enabled = bool(
        getattr(settings, "openai_api_key", None) or getattr(settings, "zhipu_api_key", None)
    )

    # 数据降级判定
    if smoke_results is not None:
        data_degraded = bool(smoke_results.get("degraded"))
        data_detail = (
            f"ok={smoke_results.get('ok')}/{smoke_results.get('total')} "
            f"failed={smoke_results.get('failed')} skipped={smoke_results.get('skipped')}"
        )
    else:
        data_degraded = offline
        data_detail = "offline_mode" if offline else "在线（未做实时冒烟）"

    # 数据源健康评分（推荐 #3）：把 smoke 结果转成可量化分数与健康源列表
    data_health_score = compute_data_health_score(smoke_results)

    components: list[dict[str, Any]] = [
        {
            "name": "datasource",
            "ok": not data_degraded,
            "degraded": data_degraded,
            "detail": data_detail,
        },
        {
            "name": "llm",
            "ok": llm_enabled,
            "degraded": False,
            "detail": "可用" if llm_enabled else "未配置 API key（规则引擎仍可用）",
        },
        {
            "name": "offline_mode",
            "ok": True,
            "degraded": offline,
            "detail": str(offline),
        },
    ]

    # 整体状态：LLM 未配置属于可选降级（fail-safe，规则引擎仍可用），
    # 不应置为 error；仅当「无任何数据源可用」才算 error。
    ds_summary = _data_source_summary(settings)
    all_down = bool(ds_summary["all_down"])
    if all_down:
        status = "error"
    elif data_degraded:
        status = "degraded"
    else:
        status = "ok"

    return {
        "status": status,
        "degraded": data_degraded,
        "app": getattr(settings, "app_name", "ai-quant-agent"),
        "offline_mode": offline,
        "llm_enabled": llm_enabled,
        "components": components,
        "data_sources": ds_summary,
        "smoke": smoke_results,
        "data_health_score": data_health_score,
    }
