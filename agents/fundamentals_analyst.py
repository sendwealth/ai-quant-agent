#!/usr/bin/env python3
"""Fundamentals Analyst - 基本面分析（简化版）"""

import sys

from utils.logger import get_logger

logger = get_logger(__name__)

import argparse
import json
from datetime import datetime
from pathlib import Path

from config.settings import Settings

PROJECT_ROOT = Path(__file__).parent.parent


def analyze_fundamentals(stock_code: str) -> dict:
    """基本面分析"""

    # 生成信号
    if data["de_ratio"] < 0.6 and data["financial_health"] >= 7:
        signal, confidence = "BUY", 0.80
    elif data["de_ratio"] > 0.8 or data["financial_health"] < 5:
        signal, confidence = "SELL", 0.75
    else:
        signal, confidence = "HOLD", 0.65

    return {
        "analyst": "fundamentals",
        "stock": data["stock"],
        "stock_code": stock_code,
        "signal": signal,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "pe_ratio": data["pe_ratio"],
            "pb_ratio": data["pb_ratio"],
            "de_ratio": data["de_ratio"],
            "financial_health": data["financial_health"],
        },
        "reasoning": f"P/E {data['pe_ratio']}倍，P/B {data['pb_ratio']}倍，"
        f"负债率{data['de_ratio']*100:.0f}%，财务健康度{data['financial_health']}/10。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    args = parser.parse_args()

    report = analyze_fundamentals(args.stock)
    output_path = PROJECT_ROOT / "data" / "signals" / f"fundamentals_{args.stock}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Fundamentals Analyst: {report['stock']} → {report['signal']} ({report['confidence']:.0%})"
    )


if __name__ == "__main__":
    main()
