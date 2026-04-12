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
    parser.add_argument("--screen", action="store_true", help="智能选股模式")
    parser.add_argument("--top", type=int, default=20, help="选股数量 (默认20)")
    parser.add_argument("--full-scan", action="store_true", help="全市场扫描 (约5000只, 较慢)")
    parser.add_argument("--screen-analyze", action="store_true", help="选股后对Top N进行深度分析")
    parser.add_argument("--preload", action="store_true", help="预下载数据到本地缓存")
    parser.add_argument("--offline", action="store_true", help="离线模式 (仅使用缓存数据)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    # Set offline mode via environment variable (picked up by Settings)
    if args.offline:
        import os
        os.environ["QUANT_OFFLINE_MODE"] = "true"

    if args.preload:
        # Pre-download data to local cache
        from quant_agent.config import get_settings
        from quant_agent.data.service import DataService
        settings = get_settings()
        svc = DataService(settings)
        codes = [c.strip() for c in settings.preload_stocks.split(",") if c.strip()]
        print(f"Pre-downloading data for {len(codes)} stocks...")
        ok = 0
        for i, code in enumerate(codes, 1):
            print(f"  [{i}/{len(codes)}] {code}...", end=" ")
            df = svc.get_price_data(code, days=args.days, use_cache=False)
            if df is not None:
                ok += 1
                print(f"OK ({len(df)} bars)")
            else:
                print("FAILED")
        print(f"\nPreload complete: {ok}/{len(codes)} stocks")
    elif args.prompt:
        # 自然语言分析
        from quant_agent.llm.client import LLMError
        orch = Orchestrator()
        try:
            report = orch.analyze_prompt(args.prompt)
            if report.llm_analysis:
                print("\n" + report.llm_analysis)
        except LLMError as e:
            print(f"LLM 错误: {e}")
    elif args.screen:
        # 选股模式
        orch = Orchestrator()
        if args.screen_analyze:
            screen_result, reports = orch.screen_and_analyze(
                use_full_market=args.full_scan,
                top_n=args.top,
                include_fundamentals=True,
            )
        else:
            screen_result = orch.screener.screen(
                use_full_market=args.full_scan,
                top_n=args.top,
            )

        # 打印选股结果表格
        top = screen_result.top_stocks
        if not top:
            print("选股无结果")
        else:
            print(f"\n{'=' * 90}")
            print(f"AI Quant Agent v3.0 — 智能选股 Top {len(top)}")
            print(f"{'=' * 90}")
            print(f"{'#':>2} {'代码':<8} {'价格':>8} {'评分':>5} "
                  f"{'技术':>4} {'动量':>4} {'流动性':>4} {'基本':>4}")
            print("-" * 90)
            for i, s in enumerate(top):
                print(f"{i + 1:>2} {s.stock_code:<8} {s.price:>8.2f} "
                      f"{s.total_score:>5.1f} {s.technical_score:>4.0f} "
                      f"{s.momentum_score:>4.0f} {s.liquidity_score:>4.0f} "
                      f"{s.fundamental_score:>4.0f}")
            print("=" * 90)
            codes = ", ".join(s.stock_code for s in top)
            print(f"\n入选: {codes}")

            if args.screen_analyze and reports:
                print(f"\n{'=' * 90}")
                print(f"深度分析结果 ({len(reports)} 只)")
                print(f"{'=' * 90}")
                for r in reports:
                    risk = r.risk_result
                    sig = risk.signal if risk else "N/A"
                    conf = f"{risk.confidence:.0%}" if risk else "N/A"
                    pos = f"{risk.metrics.get('position', 0):.0%}" if risk else "N/A"
                    print(f"  {r.stock_code}: {sig} (信心 {conf}, 仓位 {pos})")
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
