"""生成内置「演示样例」行情 / 财务 parquet 文件（可复现）。

这些样例用于离线兜底：当用户未配置任何数据源 token、且无本地缓存时，
``SamplePriceSource`` 会读取 ``quant_agent/data/samples`` 下与股票代码同名的
parquet 文件，使 ``quant-agent --offline`` 在全新安装后也能跑通端到端流程。

重要：这些是**确定性合成演示数据**（固定随机种子），仅用于演示与测试，
**不代表任何真实行情或财务表现**。真实分析请配置数据源 token 并通过
``quant-agent preload`` 预下载数据。

输出目录：``src/quant_agent/data/samples/{price,financial}/``
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

SEED = 20260411
TRADING_DAYS = 250

# (代码, 起始价, 年化漂移, 年化波动)
DEMO_STOCKS = [
    ("600519", 1680.0, 0.12, 0.28),  # 贵州茅台（演示）
    ("300750", 245.0, 0.05, 0.45),  # 宁德时代（演示）
    ("000001", 12.3, 0.03, 0.22),  # 平安银行（演示）
]

# 演示用财务快照（确定性、非真实）
DEMO_FINANCIALS = {
    "600519": {
        "roe": 0.31,
        "gross_margin": 0.91,
        "net_margin": 0.52,
        "debt_ratio": 0.22,
        "current_ratio": 3.1,
        "revenue_growth": 0.16,
        "profit_growth": 0.19,
        "pe_ttm": 28.5,
        "pb": 9.2,
        "ps_ttm": 15.1,
        "total_mv": 2_100_000.0,
        "price": 1680.0,
        "report_date": "2025-12-31",
    },
    "300750": {
        "roe": 0.18,
        "gross_margin": 0.22,
        "net_margin": 0.10,
        "debt_ratio": 0.58,
        "current_ratio": 1.4,
        "revenue_growth": 0.09,
        "profit_growth": 0.07,
        "pe_ttm": 22.0,
        "pb": 4.0,
        "ps_ttm": 2.2,
        "total_mv": 1_080_000.0,
        "price": 245.0,
        "report_date": "2025-12-31",
    },
    "000001": {
        "roe": 0.11,
        "gross_margin": 0.0,
        "net_margin": 0.18,
        "debt_ratio": 0.92,
        "current_ratio": 0.0,
        "revenue_growth": 0.04,
        "profit_growth": 0.02,
        "pe_ttm": 4.8,
        "pb": 0.5,
        "ps_ttm": 1.1,
        "total_mv": 240_000.0,
        "price": 12.3,
        "report_date": "2025-12-31",
    },
}


def _price_frame(code: str, start: float, drift: float, vol: float) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + int(code))
    end = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end, periods=TRADING_DAYS)
    daily_ret = rng.normal(drift / TRADING_DAYS, vol / np.sqrt(TRADING_DAYS), size=len(dates))
    price = start * np.cumprod(1 + daily_ret)
    # 复权收盘价
    close = np.round(price, 2)
    # 由收盘价构造开高低量，保证技术形态合理
    open_ = np.round(close * (1 + rng.normal(0, 0.005, size=len(dates))), 2)
    high = np.round(np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, size=len(dates)))), 2)
    low = np.round(np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, size=len(dates)))), 2)
    volume = rng.integers(5_000_000, 30_000_000, size=len(dates)).astype("int64")
    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    base = Path(__file__).resolve().parent.parent / "src" / "quant_agent" / "data" / "samples"
    price_dir = base / "price"
    fin_dir = base / "financial"
    price_dir.mkdir(parents=True, exist_ok=True)
    fin_dir.mkdir(parents=True, exist_ok=True)

    for code, start, drift, vol in DEMO_STOCKS:
        pf = _price_frame(code, start, drift, vol)
        pf.to_parquet(price_dir / f"{code}.parquet", index=False)
        print(f"price  -> {price_dir / (code + '.parquet')} ({len(pf)} rows)")

        ff = pd.DataFrame([DEMO_FINANCIALS[code]])
        ff.to_parquet(fin_dir / f"{code}.parquet", index=False)
        print(f"fin    -> {fin_dir / (code + '.parquet')}")

    print("\n演示样例已生成。注意：这些是确定性合成数据，仅用于离线演示，非真实行情/财务。")


if __name__ == "__main__":
    main()
