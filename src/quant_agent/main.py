"""CLI 入口 — 完整量化分析流水线"""

import logging

from quant_agent.orchestrator import Orchestrator, AnalysisReport


def run_pipeline(stock_code: str = "300750", days: int = 250) -> dict:
    """完整分析 + 执行流水线

    向后兼容的包装器：内部委托给 Orchestrator.analyze()。
    返回值格式与原始实现一致，确保调用方不受影响。
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    orch = Orchestrator()
    report: AnalysisReport = orch.analyze(stock_code, days=days)

    risk_result = report.risk_result
    summary = report.summary
    health_status = orch.health.check_all()

    return {
        "stock": stock_code,
        "signal": risk_result.signal if risk_result else "HOLD",
        "confidence": risk_result.confidence if risk_result else 0.0,
        "position_pct": risk_result.metrics.get("position", 0.0) if risk_result else 0.0,
        "summary": summary,
        "health": health_status,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Quant Agent v3.0")
    parser.add_argument("--stock", default="300750", help="股票代码")
    parser.add_argument("--days", type=int, default=250, help="分析天数")
    parser.add_argument("--multi", action="store_true", help="分析多只股票")
    parser.add_argument("--notify", action="store_true", help="启用邮件通知 (交易信号)")
    parser.add_argument("--daily-report", action="store_true", help="批量分析 + 发送每日报告")
    parser.add_argument("--prompt", help="自然语言分析指令，如: '分析宁德时代的买入机会'")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    if args.prompt:
        # 自然语言分析
        from quant_agent.llm.client import LLMError
        orch = Orchestrator()
        try:
            report = orch.analyze_prompt(args.prompt)
            if report.llm_analysis:
                print("\n" + report.llm_analysis)
        except LLMError as e:
            print(f"LLM 错误: {e}")
    elif args.daily_report:
        # 批量分析 + 每日报告
        codes = ["300750", "002475", "601318", "600276"]
        orch = Orchestrator()
        orch.analyze_batch(codes, args.days)
    elif args.multi:
        for code in ["300750", "002475", "601318", "600276"]:
            run_pipeline(code, args.days)
    else:
        run_pipeline(args.stock, args.days)
