"""P1.5 数据源冒烟测试 CLI — 定时校验所有已配置数据源可达性。

用法::

    uv run python scripts/smoke_test_sources.py [--stock 600519] [--days 5] [--json]

可直接由 cron / CI / GitHub Actions 定时触发。当没有任何真实数据源可用
时退出码为 1（便于告警）；其余情况（含部分源不可用但至少 1 个健康）退出
码为 0。使用 ``--no-fail`` 可强制退出 0（仅报告，用于非阻断性巡检）。
"""

from __future__ import annotations

import argparse
import json
import sys

from quant_agent.config import get_settings
from quant_agent.data.service import DataService


def main() -> int:
    parser = argparse.ArgumentParser(description="数据源冒烟测试 (P1.5)")
    parser.add_argument("--stock", default="600519", help="用于探测的股票代码 (默认 600519)")
    parser.add_argument("--days", type=int, default=5, help="回溯天数 (默认 5)")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出完整报告")
    parser.add_argument(
        "--no-fail",
        action="store_true",
        help="即使全部失败也返回退出码 0（非阻断性巡检）",
    )
    args = parser.parse_args()

    settings = get_settings()
    svc = DataService(settings)
    report = svc.smoke_test(stock_code=args.stock, days=args.days)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(
            f"数据源冒烟测试结果: ok={report['ok']}/{report['total']} "
            f"failed={report['failed']} skipped={report['skipped']}"
        )
        for r in report["results"]:
            if r["skipped"]:
                status = "SKIP"
                extra = r.get("skip_reason") or ""
            elif r["ok"]:
                status = "OK  "
                extra = f"{r['rows']} rows / {r['latency_ms']}ms"
            else:
                status = "FAIL"
                extra = r.get("error") or "unknown"
            print(f"  [{status}] {r['source']}: {extra}")
        if report["degraded"]:
            print(f"  ⚠️ {report['degradation_note']}")

    healthy = report["all_healthy"]
    if args.no_fail:
        return 0
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
