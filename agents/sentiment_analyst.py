#!/usr/bin/env python3
"""
Sentiment Analyst - 情绪分析(真实数据版)

策略:基于价格走势和市场数据推断情绪
维度:价格动量、成交量变化、技术面情绪、波动率

注意:不使用外部新闻API,完全基于可获取的真实市场数据
"""

import logging
import sys

# 设置 logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pandas as pd
    import numpy as np
    from utils.multi_source_data_fetcher import MultiSourceDataFetcher
except ImportError as e:
    logger.error(f"缺少依赖: {e}")
    sys.exit(1)

# 导入真实数据获取器
# from utils.real_data_fetcher import RealDataFetcher  # 已弃用,改用 MultiSourceDataFetcher


class SentimentAnalyst:
    """情绪分析师"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.analyst_name = "sentiment"
        self.data_fetcher = MultiSourceDataFetcher()

    def get_market_data(self, days: int = 60) -> Dict[str, Any]:
        """获取市场数据(多数据源)"""
        data = {}

        # 1. 获取当前价格
        price = self.data_fetcher.get_stock_price(self.stock_code)
        if not price:
            raise RuntimeError(f"❌ 无法获取 {self.stock_code} 的实时价格")
        data["current_price"] = price

        # 2. 获取历史数据(多数据源)
        df = self.data_fetcher.get_stock_hist_data(self.stock_code, days=days)

        if df is None or df.empty:
            raise RuntimeError(f"❌ 无法从任何数据源获取 {self.stock_code} 的历史数据")

        logger.info(f"✅ 获取到 {len(df)} 天历史数据")
        data["history_df"] = df

        return data

    def calculate_price_momentum(self, df: pd.DataFrame) -> tuple:
        """计算价格动量情绪"""
        close = df["close"]

        # 5日、10日、20日收益率
        ret_5d = (close.iloc[-1] / close.iloc[-5] - 1) if len(df) >= 5 else 0
        ret_10d = (close.iloc[-1] / close.iloc[-10] - 1) if len(df) >= 10 else 0
        ret_20d = (close.iloc[-1] / close.iloc[-20] - 1) if len(df) >= 20 else 0

        # 加权平均(近期权重更高)
        weighted_return = ret_5d * 0.5 + ret_10d * 0.3 + ret_20d * 0.2

        # 转换为情绪分数(-1 到 1)
        if weighted_return > 0.10:
            momentum_sentiment = 0.8
            status = "强势上涨"
        elif weighted_return > 0.05:
            momentum_sentiment = 0.5
            status = "温和上涨"
        elif weighted_return > 0:
            momentum_sentiment = 0.2
            status = "小幅上涨"
        elif weighted_return > -0.05:
            momentum_sentiment = -0.2
            status = "小幅下跌"
        elif weighted_return > -0.10:
            momentum_sentiment = -0.5
            status = "温和下跌"
        else:
            momentum_sentiment = -0.8
            status = "强势下跌"

        return (
            momentum_sentiment,
            status,
            {
                "5d": ret_5d * 100,
                "10d": ret_10d * 100,
                "20d": ret_20d * 100,
            },
        )

    def calculate_volume_sentiment(self, df: pd.DataFrame) -> tuple:
        """计算成交量情绪（市场关注度）"""
        volume = df["volume"]
        avg_volume_20d = volume.rolling(window=20).mean().iloc[-1]
        current_volume = volume.iloc[-1]

        # 成交量比率
        volume_ratio = current_volume / avg_volume_20d

        # 转换为情绪分数
        if volume_ratio > 2.0:
            volume_sentiment = 0.8
            status = "极高关注"
        elif volume_ratio > 1.5:
            volume_sentiment = 0.5
            status = "高关注"
        elif volume_ratio > 1.0:
            volume_sentiment = 0.2
            status = "正常关注"
        elif volume_ratio > 0.7:
            volume_sentiment = -0.2
            status = "低关注"
        else:
            volume_sentiment = -0.5
            status = "极低关注"

        return volume_sentiment, status, volume_ratio

    def calculate_technical_sentiment(self, df: pd.DataFrame) -> tuple:
        """计算技术面情绪"""
        close = df["close"]

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        # RSI 情绪
        if rsi > 70:
            tech_sentiment = -0.5  # 超买,负面情绪
            rsi_status = "超买"
        elif rsi > 60:
            tech_sentiment = 0.3
            rsi_status = "偏强"
        elif rsi > 40:
            tech_sentiment = 0.0
            rsi_status = "中性"
        elif rsi > 30:
            tech_sentiment = 0.3  # 超卖,可能反弹
            rsi_status = "偏弱"
        else:
            tech_sentiment = 0.5  # 严重超卖,反弹概率高
            rsi_status = "超卖"

        # MACD
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=9, adjust=False).mean()

        macd_status = "金叉" if macd.iloc[-1] > signal_line.iloc[-1] else "死叉"
        if macd_status == "金叉":
            tech_sentiment += 0.3
        else:
            tech_sentiment -= 0.3

        # 限制在 -1 到 1
        tech_sentiment = max(-1, min(1, tech_sentiment))

        return tech_sentiment, {"rsi": rsi, "rsi_status": rsi_status, "macd": macd_status}

    def calculate_volatility_sentiment(self, df: pd.DataFrame) -> tuple:
        """计算波动率情绪"""
        close = df["close"]
        returns = close.pct_change()

        # 20日波动率
        volatility_20d = returns.rolling(window=20).std().iloc[-1]

        # 转换为年化波动率
        annual_vol = volatility_20d * np.sqrt(252)

        # 波动率情绪(高波动 = 不确定性 = 负面情绪)
        if annual_vol > 0.50:
            vol_sentiment = -0.8
            status = "极高波动"
        elif annual_vol > 0.35:
            vol_sentiment = -0.5
            status = "高波动"
        elif annual_vol > 0.25:
            vol_sentiment = -0.2
            status = "中波动"
        elif annual_vol > 0.15:
            vol_sentiment = 0.2
            status = "低波动"
        else:
            vol_sentiment = 0.5
            status = "极低波动"

        return vol_sentiment, status, annual_vol

    def generate_signal(self, sentiments: Dict[str, float]) -> tuple:
        """生成综合情绪信号"""
        # 加权平均
        weights = {
            "price_momentum": 0.35,
            "volume": 0.20,
            "technical": 0.30,
            "volatility": 0.15,
        }

        overall_sentiment = sum(
            sentiments[key] * weights[key] for key in weights if key in sentiments
        )

        # 生成信号
        if overall_sentiment > 0.3:
            return "BUY", 0.70, f"市场情绪积极(综合得分{overall_sentiment:.2f})"
        elif overall_sentiment < -0.3:
            return "SELL", 0.65, f"市场情绪消极(综合得分{overall_sentiment:.2f})"
        else:
            return "HOLD", 0.60, f"市场情绪中性(综合得分{overall_sentiment:.2f})"

    def analyze(self) -> Dict[str, Any]:
        """执行分析"""
        logger.info(f"🔍 Sentiment Analyst 分析 {self.stock_code}...")

        # 1. 获取数据
        data = self.get_market_data()
        df = data["history_df"]

        # 确保列名标准化
        column_mapping = {
            "收盘": "close",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
        }
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})

        # 使用标准化后的列名
        close = df["close"]
        volume = df["volume"]

        # 2. 计算各维度情绪
        momentum_sentiment, momentum_status, momentum_returns = self.calculate_price_momentum(df)
        volume_sentiment, volume_status, volume_ratio = self.calculate_volume_sentiment(df)
        tech_sentiment, tech_details = self.calculate_technical_sentiment(df)
        vol_sentiment, vol_status, annual_vol = self.calculate_volatility_sentiment(df)

        # 3. 汇总情绪
        sentiments = {
            "price_momentum": momentum_sentiment,
            "volume": volume_sentiment,
            "technical": tech_sentiment,
            "volatility": vol_sentiment,
        }

        # 4. 生成信号
        signal, confidence, reasoning = self.generate_signal(sentiments)

        # 5. 构建报告
        report = {
            "analyst": "Sentiment Analyst",
            "stock_code": self.stock_code,
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "current_price": data["current_price"],
                "overall_sentiment": sum(sentiments.values()) / len(sentiments),
            },
            "analysis": {
                "price_momentum": momentum_status,
                "volume": volume_status,
                "technical": f"RSI {tech_details['rsi']:.0f} ({tech_details['rsi_status']}), MACD {tech_details['macd']}",
                "volatility": vol_status,
            },
            "scores": {
                "momentum_sentiment": round(momentum_sentiment, 2),
                "volume_sentiment": round(volume_sentiment, 2),
                "technical_sentiment": round(tech_sentiment, 2),
                "volatility_sentiment": round(vol_sentiment, 2),
            },
            "details": {
                "returns": momentum_returns,
                "volume_ratio": round(volume_ratio, 2),
                "annual_volatility": f"{annual_vol*100:.1f}%",
            },
        }

        logger.info(f"✅ 分析完成: {signal} (信心度: {confidence:.0%})")
        return report

    def save_report(self, report: Dict[str, Any]) -> Path:
        """保存报告到文件"""
        output_dir = PROJECT_ROOT / "data" / "signals"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.analyst_name}_{self.stock_code}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"报告已保存: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Sentiment Analyst - 情绪分析")
    parser.add_argument("--stock", required=True, help="股票代码(如 300750)")
    parser.add_argument("--send", action="store_true", help="发送给 Risk Manager")

    args = parser.parse_args()

    # 执行分析
    analyst = SentimentAnalyst(args.stock)
    report = analyst.analyze()

    # 保存报告
    output_path = analyst.save_report(report)

    # 打印摘要
    print("\n" + "=" * 60)
    logger.info(f"📊 情绪分析结果: {report['stock_code']}")
    print("=" * 60)
    logger.info(f"信号: {report['signal']}")
    logger.info(f"信心度: {report['confidence']:.0%}")
    logger.info(f"理由: {report['reasoning']}")
    logger.info(f"综合情绪: {report['metrics']['overall_sentiment']:.2f}")
    logger.info(f"\n分项情绪:")
    logger.info(
        f"  价格动量: {report['analysis']['price_momentum']} ({report['scores']['momentum_sentiment']:.2f})"
    )
    logger.info(
        f"  成交量: {report['analysis']['volume']} ({report['scores']['volume_sentiment']:.2f})"
    )
    logger.info(
        f"  技术面: {report['analysis']['technical']} ({report['scores']['technical_sentiment']:.2f})"
    )
    logger.info(
        f"  波动率: {report['analysis']['volatility']} ({report['scores']['volatility_sentiment']:.2f})"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
