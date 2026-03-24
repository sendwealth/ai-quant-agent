#!/usr/bin/env python3
"""Technical Analyst - 技术分析（简化版）"""

import sys

from utils.logger import get_logger

logger = get_logger(__name__)

import argparse
import json
from datetime import datetime
from pathlib import Path

from config.settings import Settings

PROJECT_ROOT = Path(__file__).parent.parent


def analyze_technical(stock_code: str) -> dict:
    """技术分析"""

    # 生成信号
    if data["rsi"] <= 60 and data["macd"] == "金叉" and data["ema_trend"] == "上升":
        signal, confidence = "BUY", 0.75
    elif data["rsi"] > 80 or data["macd"] == "死叉":
        signal, confidence = "SELL", 0.70
    else:
        signal, confidence = "HOLD", 0.60

    return {
        "analyst": "technical",
        "stock": data["stock"],
        "stock_code": stock_code,
        "signal": signal,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "rsi": data["rsi"],
            "macd": data["macd"],
            "ema_trend": data["ema_trend"],
            "volume": data["volume"],
        },
        "reasoning": f"RSI {data['rsi']}，MACD {data['macd']}，EMA趋势{data['ema_trend']}，成交量{data['volume']}。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    args = parser.parse_args()

    report = analyze_technical(args.stock)
    output_path = PROJECT_ROOT / "data" / "signals" / f"technical_{args.stock}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Technical Analyst: {report['stock']} → {report['signal']} ({report['confidence']:.0%})"
    )


if __name__ == "__main__":
    main()
