#!/usr/bin/env python3
"""
Technical Analyst - 技术分析（真实数据版）

策略：技术指标分析
维度：RSI、MACD、MA趋势、成交量
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


class TechnicalAnalyst:
    """技术分析师"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.analyst_name = "technical"
        self.data_fetcher = MultiSourceDataFetcher()

    def get_price_data(self, days: int = 250) -> pd.DataFrame:
        """获取历史价格数据（多数据源）"""
        df = self.data_fetcher.get_stock_hist_data(self.stock_code, days=days)

        if df is None or df.empty:
            raise RuntimeError(f"❌ 无法从任何数据源获取 {self.stock_code} 的历史数据")

        logger.info(f"✅ 获取到 {len(df)} 天历史数据")
        return df

    def calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """计算 RSI"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi.iloc[-1])

    def calculate_macd(self, close: pd.Series) -> tuple:
        """计算 MACD"""
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        # 判断金叉/死叉
        if len(histogram) >= 2:
            if histogram.iloc[-1] > 0 and histogram.iloc[-2] <= 0:
                macd_signal = "金叉"
            elif histogram.iloc[-1] < 0 and histogram.iloc[-2] >= 0:
                macd_signal = "死叉"
            else:
                macd_signal = "金叉" if histogram.iloc[-1] > 0 else "死叉"
        else:
            macd_signal = "未知"

        return float(macd.iloc[-1]), float(signal.iloc[-1]), float(histogram.iloc[-1]), macd_signal

    def analyze_ema_trend(self, close: pd.Series) -> str:
        """分析 EMA 趋势"""
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        # 检查趋势
        if ema_20.iloc[-1] > ema_50.iloc[-1]:
            return "上升"
        elif ema_20.iloc[-1] < ema_50.iloc[-1]:
            return "下降"
        else:
            return "横盘"

    def analyze_volume(self, volume: pd.Series) -> str:
        """分析成交量"""
        avg_volume = volume.rolling(window=20).mean().iloc[-1]
        current_volume = volume.iloc[-1]

        ratio = current_volume / avg_volume
        if ratio > 1.5:
            return "放量"
        elif ratio < 0.7:
            return "缩量"
        else:
            return "正常"

    def generate_signal(
        self, rsi: float, macd_signal: str, ema_trend: str, volume_status: str
    ) -> tuple:
        """生成交易信号"""
        # 综合判断
        buy_score = 0
        sell_score = 0

        # RSI
        if rsi < 30:
            buy_score += 2  # 超卖
        elif rsi < 50:
            buy_score += 1
        elif rsi > 70:
            sell_score += 2  # 超买
        elif rsi > 60:
            sell_score += 1

        # MACD
        if macd_signal == "金叉":
            buy_score += 2
        elif macd_signal == "死叉":
            sell_score += 2

        # EMA 趋势
        if ema_trend == "上升":
            buy_score += 1
        elif ema_trend == "下降":
            sell_score += 1

        # 成交量
        if volume_status == "放量":
            if buy_score > sell_score:
                buy_score += 1  # 放量上涨
            else:
                sell_score += 1  # 放量下跌

        # 决策
        if buy_score >= 4 and buy_score > sell_score + 1:
            return "BUY", 0.75, f"技术指标偏多（买入信号{buy_score}分）"
        elif sell_score >= 4 and sell_score > buy_score + 1:
            return "SELL", 0.70, f"技术指标偏空（卖出信号{sell_score}分）"
        else:
            return "HOLD", 0.60, f"技术指标中性（买入{buy_score}分，卖出{sell_score}分）"

    def analyze(self) -> Dict[str, Any]:
        """执行技术分析"""
        logger.info(f"🔍 Technical Analyst 分析 {self.stock_code}...")

        # 1. 获取数据
        df = self.get_price_data()

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

        # 2. 计算指标
        rsi = self.calculate_rsi(close)
        macd, macd_signal_line, macd_histogram, macd_status = self.calculate_macd(close)
        ema_trend = self.analyze_ema_trend(close)
        volume_status = self.analyze_volume(volume)

        # 3. 生成信号
        signal, confidence, reasoning = self.generate_signal(
            rsi, macd_status, ema_trend, volume_status
        )

        # 4. 获取股票名称
        stock_name = ""
        try:
            info_df = ak.stock_individual_info_em(symbol=self.stock_code)
            if not info_df.empty:
                info_dict = dict(zip(info_df["item"], info_df["value"]))
                stock_name = info_dict.get("股票简称", "")
        except:
            pass

        # 5. 构建报告
        report = {
            "analyst": "Technical Analyst",
            "stock_code": self.stock_code,
            "stock": stock_name,
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "current_price": float(close.iloc[-1]),
                "rsi": round(rsi, 2),
                "macd": round(macd, 4),
                "macd_signal": round(macd_signal_line, 4),
                "macd_histogram": round(macd_histogram, 4),
                "macd_status": macd_status,
                "ema_trend": ema_trend,
                "volume_status": volume_status,
                "volume": int(volume.iloc[-1]),
            },
            "analysis": {
                "rsi": round(rsi, 2),
                "macd": macd_status,
                "ema_trend": ema_trend,
                "volume": volume_status,
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
    parser = argparse.ArgumentParser(description="Technical Analyst - 技术分析")
    parser.add_argument("--stock", required=True, help="股票代码（如 300750）")
    parser.add_argument("--send", action="store_true", help="发送给 Risk Manager")

    args = parser.parse_args()

    # 执行分析
    analyst = TechnicalAnalyst(args.stock)
    report = analyst.analyze()

    # 保存报告
    output_path = analyst.save_report(report)

    # 打印摘要
    print("\n" + "=" * 60)
    logger.info(f"📊 技术分析结果: {report['stock_code']}")
    print("=" * 60)
    logger.info(f"信号: {report['signal']}")
    logger.info(f"信心度: {report['confidence']:.0%}")
    logger.info(f"理由: {report['reasoning']}")
    logger.info(f"当前价格: {report['metrics']['current_price']:.2f}")
    logger.info(f"RSI: {report['metrics']['rsi']:.2f}")
    logger.info(f"MACD: {report['metrics']['macd_status']}")
    logger.info(f"EMA趋势: {report['metrics']['ema_trend']}")
    logger.info(f"成交量: {report['metrics']['volume_status']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
