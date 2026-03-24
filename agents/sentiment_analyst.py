#!/usr/bin/env python3
"""Sentiment Analyst - 情绪分析（简化版）"""

import sys

from utils.logger import get_logger

logger = get_logger(__name__)

import argparse
import json
from datetime import datetime
from pathlib import Path

from config.settings import Settings

PROJECT_ROOT = Path(__file__).parent.parent


def analyze_sentiment(stock_code: str) -> dict:
    """情绪分析"""

    # 计算综合情绪
    avg_sentiment = (data["news_sentiment"] + data["social_sentiment"]) / 2

    # 生成信号
    if avg_sentiment >= 0.3 and data["insider_trading"] == "增持":
        signal, confidence = "BUY", 0.70
    elif avg_sentiment < -0.3:
        signal, confidence = "SELL", 0.65
    else:
        signal, confidence = "HOLD", 0.60

    return {
        "analyst": "sentiment",
        "stock": data["stock"],
        "stock_code": stock_code,
        "signal": signal,
        "confidence": confidence,
        "timestamp": datetime.now().isoformat(),
        "analysis": {
            "news_sentiment": data["news_sentiment"],
            "social_sentiment": data["social_sentiment"],
            "insider_trading": data["insider_trading"],
            "analyst_rating": data["analyst_rating"],
        },
        "reasoning": f"新闻情绪{data['news_sentiment']:.2f}，社交情绪{data['social_sentiment']:.2f}，"
        f"内部人{data['insider_trading']}，分析师评级{data['analyst_rating']}。",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", required=True)
    args = parser.parse_args()

    report = analyze_sentiment(args.stock)
    output_path = PROJECT_ROOT / "data" / "signals" / f"sentiment_{args.stock}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info(
        "Sentiment Analyst: {report['stock']} → {report['signal']} ({report['confidence']:.0%})"
    )


if __name__ == "__main__":
    main()
