"""
分析智能体
实时市场分析、趋势识别、情绪分析
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

from utils.config import get_config
from utils.indicators import adx, atr, bollinger_bands, ema, macd, rsi, sma


class AnalysisAgent:
    """分析智能体 - 市场分析和趋势识别"""

    def __init__(self):
        """初始化分析智能体"""
        self.config = get_config()

    def analyze_market(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        综合市场分析

        Args:
            df: 历史价格数据 (包含open, high, low, close, volume)

        Returns:
            分析结果字典
        """
        logger.info("开始市场分析...")

        analysis = {
            "trend": self.analyze_trend(df),
            "momentum": self.analyze_momentum(df),
            "volatility": self.analyze_volatility(df),
            "volume": self.analyze_volume(df),
            "support_resistance": self.find_support_resistance(df),
            "overall_sentiment": None,
        }

        # 综合判断市场情绪
        analysis["overall_sentiment"] = self.calculate_overall_sentiment(analysis)

        logger.info(f"市场分析完成，情绪: {analysis['overall_sentiment']}")

        return analysis

    def analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        趋势分析

        Args:
            df: 价格数据

        Returns:
            趋势分析结果
        """
        close = df["close"]

        # 移动平均线
        sma20 = sma(close, 20)
        sma60 = sma(close, 60)

        # 价格与均线关系
        price_above_sma20 = close.iloc[-1] > sma20.iloc[-1]
        price_above_sma60 = close.iloc[-1] > sma60.iloc[-1]
        sma20_above_sma60 = sma20.iloc[-1] > sma60.iloc[-1]

        # ADX趋势强度
        adx_value = adx(df["high"], df["low"], close, 14).iloc[-1]

        # 判断趋势
        if sma20_above_sma60 and price_above_sma20:
            trend = "strong_uptrend"
            strength = "强" if adx_value > 40 else "中等"
        elif sma20_above_sma60:
            trend = "uptrend"
            strength = "中等"
        elif not sma20_above_sma60 and not price_above_sma20:
            trend = "strong_downtrend"
            strength = "强" if adx_value > 40 else "中等"
        elif not sma20_above_sma60:
            trend = "downtrend"
            strength = "中等"
        else:
            trend = "sideways"
            strength = "弱" if adx_value < 20 else "中等"

        return {
            "direction": trend,
            "strength": strength,
            "adx": adx_value,
            "price_above_sma20": price_above_sma20,
            "price_above_sma60": price_above_sma60,
            "sma20": sma20.iloc[-1],
            "sma60": sma60.iloc[-1],
        }

    def analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        动量分析

        Args:
            df: 价格数据

        Returns:
            动量分析结果
        """
        close = df["close"]

        # RSI
        rsi_value = rsi(close, 14).iloc[-1]

        # MACD
        macd_line, signal_line, histogram = macd(close)
        macd_signal = "bullish" if histogram.iloc[-1] > 0 else "bearish"

        # 动量信号
        if rsi_value < 30 and macd_signal == "bullish":
            momentum = "strong_bullish_reversal"
        elif rsi_value < 30:
            momentum = "oversold"
        elif rsi_value > 70 and macd_signal == "bearish":
            momentum = "strong_bearish_reversal"
        elif rsi_value > 70:
            momentum = "overbought"
        elif macd_signal == "bullish":
            momentum = "bullish"
        else:
            momentum = "bearish"

        return {
            "state": momentum,
            "rsi": rsi_value,
            "rsi_signal": (
                "oversold" if rsi_value < 30 else "overbought" if rsi_value > 70 else "neutral"
            ),
            "macd_signal": macd_signal,
            "macd_histogram": histogram.iloc[-1],
        }

    def analyze_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        波动率分析

        Args:
            df: 价格数据

        Returns:
            波动率分析结果
        """
        close = df["close"]

        # ATR
        atr_value = atr(df["high"], df["low"], close, 14).iloc[-1]
        atr_percent = (atr_value / close.iloc[-1]) * 100

        # 布林带宽度
        upper, middle, lower = bollinger_bands(close)
        bb_width = (upper - lower) / middle
        bb_width_pct = bb_width.iloc[-1] * 100

        # 波动率信号
        if atr_percent > 3:
            volatility = "very_high"
        elif atr_percent > 2:
            volatility = "high"
        elif atr_percent > 1:
            volatility = "normal"
        else:
            volatility = "low"

        return {
            "level": volatility,
            "atr": atr_value,
            "atr_percent": atr_percent,
            "bb_width": bb_width_pct,
            "signal": (
                "high_volatility" if volatility in ["high", "very_high"] else "low_volatility"
            ),
        }

    def analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        成交量分析

        Args:
            df: 价格数据（包含volume）

        Returns:
            成交量分析结果
        """
        if "volume" not in df.columns:
            return {"message": "无成交量数据", "signal": "unknown"}

        volume = df["volume"]

        # 成交量移动平均
        vol_ma20 = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1

        # 价格变化
        price_change = df["close"].pct_change()
        last_price_change = price_change.iloc[-1]

        # 成交量信号
        if vol_ratio > 2 and last_price_change > 0:
            signal = "high_volume_rise"
        elif vol_ratio > 2 and last_price_change < 0:
            signal = "high_volume_fall"
        elif vol_ratio > 1.5:
            signal = "increased_volume"
        elif vol_ratio < 0.5:
            signal = "low_volume"
        else:
            signal = "normal_volume"

        return {
            "current_volume": volume.iloc[-1],
            "avg_volume": vol_ma20.iloc[-1],
            "volume_ratio": vol_ratio,
            "signal": signal,
        }

    def find_support_resistance(
        self, df: pd.DataFrame, window: int = 20, num_levels: int = 3
    ) -> Dict[str, List[float]]:
        """
        寻找支撑位和阻力位

        Args:
            df: 价格数据
            window: 窗口大小
            num_levels: 返回的支撑/阻力位数量

        Returns:
            支撑位和阻力位
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 寻找局部极值点
        from scipy.signal import argrelextrema

        # 局部高点（阻力位）
        max_idx = argrelextrema(high.values, np.greater, order=window)[0]
        resistance_levels = sorted(high.iloc[max_idx].values, reverse=True)[:num_levels]

        # 局部低点（支撑位）
        min_idx = argrelextrema(low.values, np.less, order=window)[0]
        support_levels = sorted(low.iloc[min_idx].values)[:num_levels]

        return {
            "support": support_levels,
            "resistance": resistance_levels,
            "current_price": close.iloc[-1],
        }

    def calculate_overall_sentiment(self, analysis: Dict[str, Any]) -> str:
        """
        计算综合市场情绪

        Args:
            analysis: 各项分析结果

        Returns:
            综合情绪: very_bullish / bullish / neutral / bearish / very_bearish
        """
        score = 0

        # 趋势得分
        trend = analysis["trend"]["direction"]
        if trend == "strong_uptrend":
            score += 2
        elif trend == "uptrend":
            score += 1
        elif trend == "downtrend":
            score -= 1
        elif trend == "strong_downtrend":
            score -= 2

        # 动量得分
        momentum = analysis["momentum"]["state"]
        if "bullish" in momentum:
            score += 1 if "strong" not in momentum else 2
        elif "bearish" in momentum:
            score -= 1 if "strong" not in momentum else -2

        # 波动率得分（高波动率通常意味着不确定性）
        volatility = analysis["volatility"]["signal"]
        if volatility == "high_volatility":
            score -= 0.5

        # 成交量得分
        volume_signal = analysis["volume"].get("signal", "normal_volume")
        if volume_signal == "high_volume_rise":
            score += 0.5
        elif volume_signal == "high_volume_fall":
            score -= 0.5

        # 综合判断
        if score >= 3:
            return "very_bullish"
        elif score >= 1:
            return "bullish"
        elif score <= -3:
            return "very_bearish"
        elif score <= -1:
            return "bearish"
        else:
            return "neutral"

    def generate_trading_signals(self, df: pd.DataFrame) -> Tuple[str, str]:
        """
        生成交易信号

        Args:
            df: 价格数据

        Returns:
            (方向, 强度) - 方向: long/short/neutral, 强度: strong/medium/weak
        """
        analysis = self.analyze_market(df)
        sentiment = analysis["overall_sentiment"]

        # 根据综合情绪生成信号
        if sentiment == "very_bullish":
            return "long", "strong"
        elif sentiment == "bullish":
            return "long", "medium"
        elif sentiment == "very_bearish":
            return "short", "strong"
        elif sentiment == "bearish":
            return "short", "medium"
        else:
            return "neutral", "weak"

    def print_analysis_report(self, analysis: Dict[str, Any]):
        """
        打印分析报告

        Args:
            analysis: 分析结果
        """
        print("\n" + "=" * 60)
        logger.info("市场分析报告")
        print("=" * 60)

        # 趋势
        trend = analysis["trend"]
        logger.info("\n【趋势分析】")
        logger.info("方向: {trend['direction']}")
        logger.info("强度: {trend['strength']}")
        logger.info("ADX: {trend['adx']:.2f}")
        logger.info("价格 vs SMA20: {'上方' if trend['price_above_sma20'] else '下方'}")
        logger.info("价格 vs SMA60: {'上方' if trend['price_above_sma60'] else '下方'}")

        # 动量
        momentum = analysis["momentum"]
        logger.info("\n【动量分析】")
        logger.info("状态: {momentum['state']}")
        logger.info("RSI: {momentum['rsi']:.2f} ({momentum['rsi_signal']})")
        logger.info("MACD: {momentum['macd_signal']}")

        # 波动率
        volatility = analysis["volatility"]
        logger.info("\n【波动率分析】")
        logger.info("级别: {volatility['level']}")
        logger.info("ATR: {volatility['atr']:.2f} ({volatility['atr_percent']:.2f}%)")
        logger.info("布林带宽度: {volatility['bb_width']:.2f}%")

        # 成交量
        volume = analysis["volume"]
        logger.info("\n【成交量分析】")
        if "message" not in volume:
            logger.info("信号: {volume['signal']}")
            logger.info("成交量比率: {volume['volume_ratio']:.2f}")
        else:
            print(volume["message"])

        # 支撑/阻力
        sr = analysis["support_resistance"]
        logger.info("\n【支撑/阻力】")
        logger.info("当前价格: {sr['current_price']:.2f}")
        logger.info("支撑位: {', '.join([f'{x:.2f}' for x in sr['support']])}")
        logger.info("阻力位: {', '.join([f'{x:.2f}' for x in sr['resistance']])}")

        # 综合情绪
        logger.info("\n【综合情绪】")
        logger.info("🎯 {analysis['overall_sentiment'].upper()}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # 测试分析智能体
    agent = AnalysisAgent()

    # 生成测试数据（仅用于演示，请使用真实数据）
    logger.warning("⚠️  以下使用测试数据仅用于演示，请使用真实数据")
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    close = np.cumsum(np.random.randn(100) * 0.01) + 100
    high = close + np.random.rand(100) * 0.5
    low = close - np.random.rand(100) * 0.5
    volume = np.random.randint(10000, 50000, 100)

    df = pd.DataFrame(
        {"open": close, "high": high, "low": low, "close": close, "volume": volume}, index=dates
    )

    # 执行分析
    analysis = agent.analyze_market(df)
    agent.print_analysis_report(analysis)

    # 生成交易信号
    direction, strength = agent.generate_trading_signals(df)
    logger.info("\n交易信号: {direction.upper()} ({strength})")
