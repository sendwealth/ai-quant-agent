"""CLI 入口 — Typer 子命令（易用性优先）

子命令:
  analyze  单股深度分析（规则 + LLM 增强，离线可用）
  screen   智能选股
  batch    批量分析
  report   历史报告查看 / 导出
  init     配置向导（交互生成 .env，可选下载样例）
  preload  预下载行情到本地缓存

零配置开箱即用：无 API key / 无网络时自动走样例兜底 + LLM 离线规则增强。
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer

from .config import get_settings
from .orchestrator import Orchestrator
from .reporting import (
    render_markdown,
    render_html,
    plot_price_chart,
    save_report,
    list_reports,
    load_report,
    latest_for_stock,
)

app = typer.Typer(
    help="AI Quant Agent v3.1 — LLM 增强的 A 股量化分析 CLI",
    no_args_is_help=True,
    add_completion=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("quant.cli")


def _build_orchestrator() -> Orchestrator:
    return Orchestrator()


def _print_report_block(report) -> None:
    md = render_markdown(report)
    # 控制台输出 Markdown（清晰可读）
    typer.echo(md)


# ──────────────────────────────────────────────────────────────────────────
# analyze
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def analyze(
    stock_code: str = typer.Argument(..., help="股票代码，如 600519"),
    days: int = typer.Option(120, "--days", "-d", help="分析天数"),
    report: bool = typer.Option(True, "--report/--no-report", help="保存 Markdown 报告"),
    fmt: str = typer.Option("md", "--format", help="报告格式: md / html"),
    chart: bool = typer.Option(False, "--chart/--no-chart", help="生成价格走势图"),
    offline: bool = typer.Option(False, "--offline", help="离线模式（仅用本地/样例数据）"),
    out_dir: str = typer.Option("data/reports", "--out-dir", help="报告输出目录"),
):
    """单股深度分析"""
    if offline:
        os.environ["QUANT_OFFLINE_MODE"] = "true"

    orch = _build_orchestrator()
    result = orch.analyze(stock_code, days=days)

    _print_report_block(result)

    if report:
        if fmt == "html":
            text = render_html(result)
            ext = "html"
        else:
            text = render_markdown(result)
            ext = "md"
        ts = result.timestamp.replace(":", "").replace("-", "").replace(".", "_")
        out_path = Path(out_dir) / f"{stock_code}_{ts}.{ext}"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        typer.echo(f"\n[报告已保存] {out_path}")

    if chart:
        try:
            price = orch.data.get_price_data(stock_code, days=days)
            chart_path = Path(out_dir) / f"{stock_code}_chart.png"
            plot_price_chart(price, result, chart_path)
            typer.echo(f"[图表已保存] {chart_path}")
        except Exception as e:
            typer.echo(f"[图表生成跳过] {e}", err=True)

    # 同时写入历史索引（便于 report 子命令查看）
    try:
        save_report(result, base_dir=out_dir)
    except Exception as e:
        logger.warning("历史记录写入失败: %s", e)


# ──────────────────────────────────────────────────────────────────────────
# screen
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def screen(
    top: int = typer.Option(10, "--top", "-n", help="选股数量"),
    full_scan: bool = typer.Option(False, "--full-scan", help="全市场扫描（较慢）"),
    fundamentals: bool = typer.Option(False, "--fundamentals/--no-fundamentals", help="含基本面评分"),
    deep: bool = typer.Option(False, "--deep", help="对 Top N 进行深度分析"),
    report: bool = typer.Option(False, "--report/--no-report", help="保存选股报告"),
):
    """智能选股"""
    orch = _build_orchestrator()
    if deep:
        screen_result, reports = orch.screen_and_analyze(
            use_full_market=full_scan,
            top_n=top,
            include_fundamentals=fundamentals,
        )
    else:
        screen_result = orch.screener.screen(
            use_full_market=full_scan,
            top_n=top,
            include_fundamentals=fundamentals,
        )

    top_stocks = screen_result.top_stocks
    typer.echo("\n" + "=" * 78)
    typer.echo(f"智能选股 Top {len(top_stocks)}")
    typer.echo("=" * 78)
    header = f"{'#':>2} {'代码':<8} {'价格':>9} {'评分':>6} {'技术':>5} {'动量':>5} {'流动':>5} {'基本':>5}"
    typer.echo(header)
    typer.echo("-" * 78)
    for i, s in enumerate(top_stocks):
        typer.echo(
            f"{i + 1:>2} {s.stock_code:<8} {s.price:>9.2f} {s.total_score:>6.1f} "
            f"{s.technical_score:>5.0f} {s.momentum_score:>5.0f} "
            f"{s.liquidity_score:>5.0f} {s.fundamental_score:>5.0f}"
        )
    typer.echo("=" * 78)
    typer.echo("入选: " + ", ".join(s.stock_code for s in top_stocks))

    if deep and reports:
        typer.echo("\n深度分析:")
        for r in reports:
            risk = r.risk_result
            sig = risk.signal if risk else "N/A"
            conf = f"{risk.confidence:.0%}" if risk else "N/A"
            typer.echo(f"  {r.stock_code}: {sig} (信心 {conf})")

    if report:
        lines = [f"# 智能选股 Top {len(top_stocks)}", ""]
        for i, s in enumerate(top_stocks):
            lines.append(
                f"{i + 1}. {s.stock_code} 价格 {s.price:.2f} 评分 {s.total_score:.1f}"
            )
        out = Path("data/reports") / "screen_result.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        typer.echo(f"\n[选股报告已保存] {out}")


# ──────────────────────────────────────────────────────────────────────────
# batch
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def batch(
    stocks: str = typer.Argument(..., help="逗号分隔的股票代码，如 600519,300750"),
    days: int = typer.Option(120, "--days", "-d", help="分析天数"),
    report: bool = typer.Option(True, "--report/--no-report", help="保存各股报告"),
    out_dir: str = typer.Option("data/reports", "--out-dir", help="报告输出目录"),
):
    """批量分析多只股票"""
    codes = [c.strip() for c in stocks.split(",") if c.strip()]
    orch = _build_orchestrator()
    for code in codes:
        typer.echo(f"\n{'=' * 70}\n分析 {code}\n{'=' * 70}")
        res = orch.analyze(code, days=days)
        _print_report_block(res)
        if report:
            try:
                save_report(res, base_dir=out_dir)
            except Exception as e:
                logger.warning("保存失败 %s: %s", code, e)


# ──────────────────────────────────────────────────────────────────────────
# report (历史)
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def report(
    action: str = typer.Argument("list", help="list / show / latest"),
    target: str = typer.Argument(None, help="show 用文件名；latest 用股票代码"),
    out_dir: str = typer.Option("data/reports", "--out-dir", help="报告目录"),
):
    """查看历史报告：list | show <file> | latest <stock_code>"""
    if action == "list":
        entries = list_reports(out_dir)
        if not entries:
            typer.echo("暂无历史报告")
            return
        typer.echo(f"{'文件':<32} {'代码':<9} {'信号':<6} {'信心':>6} {'时间'}")
        typer.echo("-" * 70)
        for e in entries[:30]:
            typer.echo(
                f"{e['file']:<32} {e['stock_code']:<9} {e['signal']:<6} "
                f"{e['confidence']:>6.0%} {e['timestamp']}"
            )
    elif action == "show":
        if not target:
            typer.echo("请提供报告文件名", err=True)
            raise typer.Exit(code=1)
        data = load_report(target, out_dir)
        if data is None:
            typer.echo(f"未找到报告: {target}", err=True)
            raise typer.Exit(code=1)
        typer.echo(render_markdown_from_dict(data))
    elif action == "latest":
        if not target:
            typer.echo("请提供股票代码", err=True)
            raise typer.Exit(code=1)
        data = latest_for_stock(target, out_dir)
        if data is None:
            typer.echo(f"未找到 {target} 的历史报告", err=True)
            raise typer.Exit(code=1)
        typer.echo(render_markdown_from_dict(data))
    else:
        typer.echo(f"未知 action: {action}（可用: list/show/latest）", err=True)
        raise typer.Exit(code=1)


def render_markdown_from_dict(data: dict) -> str:
    """将历史报告 dict 渲染为 Markdown（复用渲染器需 AnalysisReport，这里轻量重建）"""
    from .orchestrator import AnalysisReport

    try:
        rep = AnalysisReport(
            stock_code=data.get("stock_code", "?"),
            timestamp=data.get("timestamp", ""),
        )
        # 直接复用 to_dict 往返不完美，简化输出关键字段
    except Exception:
        rep = None
    # 简单可读输出
    lines = [
        f"# 历史报告 — {data.get('stock_code', '?')}",
        f"> 时间: {data.get('timestamp', '')}",
        "",
        f"- 信号: {data.get('signal')}",
        f"- 信心度: {data.get('confidence')}",
        f"- 建议仓位: {data.get('position_pct')}",
    ]
    if data.get("llm_analysis"):
        lines.append("\n## LLM 报告\n" + str(data.get("llm_analysis")))
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# init（配置向导）
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def init(
    force: bool = typer.Option(False, "--force", help="覆盖已存在的 .env"),
    download_samples: bool = typer.Option(
        True, "--download-samples/--no-download-samples", help="联网下载真实历史行情样例"
    ),
):
    """配置向导：交互生成 .env，校验并（可选）下载样例数据"""
    env_path = Path(".env")
    if env_path.exists() and not force:
        typer.echo(f".env 已存在（{env_path}）。使用 --force 覆盖。")
    else:
        typer.echo("=== AI Quant Agent 配置向导 ===\n")
        tushare = typer.prompt("Tushare token（留空=使用免费源）", default="")
        openai = typer.prompt("OpenAI API key（留空=离线规则增强）", default="")
        zhipu = typer.prompt("智谱 API key（留空跳过）", default="")
        local_url = typer.prompt(
            "本地模型 base_url（Ollama 等，留空跳过）", default=""
        )
        local_model = typer.prompt("本地模型名（留空跳过）", default="")

        lines = [
            "# AI Quant Agent 配置（由 init 生成）",
            f"QUANT_TUSHARE_TOKEN={tushare}",
            f"QUANT_OPENAI_API_KEY={openai}",
            f"QUANT_ZHIPU_API_KEY={zhipu}",
        ]
        if local_url:
            lines.append(f"QUANT_LLM_BASE_URL={local_url}")
        if local_model:
            lines.append(f"QUANT_LLM_LOCAL_MODEL={local_model}")
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        typer.echo(f"\n[已生成] {env_path}")

    # 校验
    try:
        settings = get_settings()
        typer.echo(f"配置加载成功：app={settings.app_name}")
    except Exception as e:
        typer.echo(f"[配置校验失败] {e}", err=True)
        raise typer.Exit(code=1)

    # 下载样例
    if download_samples:
        typer.echo("正在下载样例行情（联网）...")
        try:
            from .data.service import DataService

            svc = DataService(settings)
            codes = [c.strip() for c in settings.preload_stocks.split(",") if c.strip()]
            ok = 0
            for code in codes:
                df = svc.get_price_data(code, days=250, use_cache=False)
                if df is not None:
                    ok += 1
            typer.echo(f"样例下载完成：{ok}/{len(codes)} 只（存入 data/parquet/price）")
        except Exception as e:
            typer.echo(f"[样例下载跳过] {e}", err=True)

    typer.echo("\n配置完成！可运行 `quant-agent analyze 600519` 开始分析。")


# ──────────────────────────────────────────────────────────────────────────
# preload
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def preload(
    days: int = typer.Option(250, "--days", "-d", help="下载天数"),
    stocks: str = typer.Option(None, "--stocks", help="指定代码，逗号分隔"),
):
    """预下载行情到本地缓存"""
    settings = get_settings()
    from .data.service import DataService

    svc = DataService(settings)
    if stocks:
        codes = [c.strip() for c in stocks.split(",") if c.strip()]
    else:
        codes = [c.strip() for c in settings.preload_stocks.split(",") if c.strip()]
    typer.echo(f"预下载 {len(codes)} 只...")
    ok = 0
    for i, code in enumerate(codes, 1):
        typer.echo(f"  [{i}/{len(codes)}] {code}...", nl=False)
        df = svc.get_price_data(code, days=days, use_cache=False)
        if df is not None:
            ok += 1
            typer.echo(f" OK ({len(df)} bars)")
        else:
            typer.echo(" FAILED")
    typer.echo(f"\n完成：{ok}/{len(codes)}")


# ──────────────────────────────────────────────────────────────────────────
# web（Web UI）
# ──────────────────────────────────────────────────────────────────────────


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    offline: bool = typer.Option(False, "--offline", help="离线模式（仅用本地/样例数据）"),
):
    """启动 Web UI（零依赖，标准库实现；浏览器打开 http://<host>:<port>）"""
    from .web.server import run_web

    run_web(host=host, port=port, offline=offline)


if __name__ == "__main__":
    app()
