"""系统健康报告（P3.1）— 聚合组件健康与数据降级状态。

把数据源冒烟结果、LLM 可用性、运行模式等聚合成一份可机读的健康报告，
供 ``/api/health`` 端点与告警模块使用。设计目标是：即使不触网也能给出
「是否降级」的结论（离线模式即视为数据降级）。
"""

from __future__ import annotations

from typing import Any

from ..config import Settings


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
    }
