"""分析图表 — 行情 + 信号可视化（matplotlib，按需懒加载）

图表为可选增强：未安装 matplotlib 时调用会给出明确提示，不影响核心分析。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..orchestrator import AnalysisReport

logger = logging.getLogger(__name__)


def plot_price_chart(
    price_df: pd.DataFrame,
    report: AnalysisReport,
    output_path: str | Path,
    title: str | None = None,
) -> str:
    """绘制收盘价走势图并标注最终信号。

    Args:
        price_df: 归一化行情（需含 date/close 列）
        report: 分析报告（读取最终信号）
        output_path: 输出 PNG 路径
        title: 图表标题

    Returns:
        实际保存的文件路径
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as err:
        raise RuntimeError(
            "未安装 matplotlib，无法生成图表。请运行 `uv add matplotlib` 或 "
            "`pip install matplotlib` 后重试。"
        ) from err

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    if price_df is not None and not price_df.empty and "close" in price_df.columns:
        dates = price_df["date"] if "date" in price_df.columns else price_df.index
        ax.plot(dates, price_df["close"], label="收盘价", color="#2563eb", linewidth=1.2)
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "无行情数据", ha="center", va="center")

    signal = report.risk_result.signal if report.risk_result else "HOLD"
    color = {"BUY": "#16a34a", "SELL": "#dc2626"}.get(signal, "#64748b")
    ax.set_title(title or f"{report.stock_code} - Signal: {signal}")
    ax.annotate(
        f"Signal: {signal}",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        color=color,
        fontsize=12,
        fontweight="bold",
        bbox={"boxstyle": "round", "fc": "white", "ec": color},
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    logger.info("图表已保存: %s", output_path)
    return str(output_path)
