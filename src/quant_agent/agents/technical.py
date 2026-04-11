"""技术分析 Agent — 基于真实行情数据"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..strategy.indicators import rsi, macd, bollinger_bands, atr, ema, adx
from ..thresholds import _Thresh, get_thresholds
from .base import BaseAgent, AgentResult


class TechnicalAgent(BaseAgent):
    """技术分析 Agent"""

    def __init__(self, threshold_config: _Thresh | None = None, **kwargs):
        super().__init__(name="technical", **kwargs)
        self._thresh = threshold_config or get_thresholds().technical

    def analyze(self, stock_code: str, days: int = 250, **kwargs) -> AgentResult:
        self._logger.info(f"技术分析: {stock_code}")
        self._log_action("analysis_started", stock_code=stock_code)

        try:
            df = self._get_price_data(stock_code, days)
            if df is None or df.empty:
                return AgentResult(
                    agent_name=self.name, stock_code=stock_code,
                    signal="HOLD", confidence=0.0,
                    reasoning="无法获取行情数据",
                    success=False, error="NO_DATA",
                )

            close = df["close"]
            high = df["high"]
            low = df["low"]
            volume = df["volume"]

            # 计算指标
            rsi_val = float(rsi(close, 14).iloc[-1])
            macd_line, signal_line, histogram = macd(close)
            macd_val = float(macd_line.iloc[-1])
            signal_val = float(signal_line.iloc[-1])
            hist_val = float(histogram.iloc[-1])
            upper, middle, lower = bollinger_bands(close, 20, 2.0)
            atr_val = float(atr(high, low, close, 14).iloc[-1])
            ema_20 = float(ema(close, 20).iloc[-1])
            ema_50 = float(ema(close, 50).iloc[-1])
            adx_val = float(adx(high, low, close, 14).iloc[-1])

            # 趋势判断
            ema_trend = "上升" if ema_20 > ema_50 else "下降"

            # MACD 金叉/死叉
            macd_status = "金叉" if hist_val > 0 else "死叉"

            # RSI 状态
            t = self._thresh
            rsi_oversold = float(t.rsi.oversold or 30)
            rsi_overbought = float(t.rsi.overbought or 70)
            if rsi_val > rsi_overbought:
                rsi_status = "超买"
            elif rsi_val < rsi_oversold:
                rsi_status = "超卖"
            else:
                rsi_status = "中性"

            # 成交量
            vol_high_ratio = float(t.volume.high_ratio or 1.5)
            vol_low_ratio = float(t.volume.low_ratio or 0.7)
            avg_vol = volume.rolling(20).mean().iloc[-1]
            vol_ratio = float(volume.iloc[-1] / avg_vol) if avg_vol > 0 else 1.0
            if vol_ratio > vol_high_ratio:
                vol_status = "放量"
            elif vol_ratio < vol_low_ratio:
                vol_status = "缩量"
            else:
                vol_status = "正常"

            # 综合评分
            signal, confidence, reasoning = self._generate_signal(
                rsi_val, macd_status, ema_trend, vol_status, adx_val
            )

            result = AgentResult(
                agent_name=self.name,
                stock_code=stock_code,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                metrics={
                    "current_price": float(close.iloc[-1]),
                    "rsi": round(rsi_val, 2),
                    "rsi_status": rsi_status,
                    "macd": round(macd_val, 4),
                    "macd_signal": round(signal_val, 4),
                    "macd_histogram": round(hist_val, 4),
                    "macd_status": macd_status,
                    "ema_20": round(ema_20, 2),
                    "ema_50": round(ema_50, 2),
                    "ema_trend": ema_trend,
                    "bollinger_upper": round(float(upper.iloc[-1]), 2),
                    "bollinger_lower": round(float(lower.iloc[-1]), 2),
                    "atr": round(atr_val, 2),
                    "adx": round(adx_val, 2),
                    "volume_ratio": round(vol_ratio, 2),
                    "vol_status": vol_status,
                },
                scores={
                    "trend": 1 if ema_trend == "上升" else -1,
                    "momentum": rsi_val / 100,
                    "volatility": 1 - min(atr_val / (close.iloc[-1] * 0.05), 1.0),
                },
            )
            self._log_action("analysis_completed", stock_code=stock_code, signal=signal, confidence=confidence)
            return result

        except Exception as e:
            self._logger.error(f"技术分析失败: {e}")
            self._log_action("analysis_failed", stock_code=stock_code, error=str(e))
            return AgentResult(
                agent_name=self.name, stock_code=stock_code,
                signal="HOLD", confidence=0.0, reasoning=f"分析失败: {e}",
                success=False, error=str(e),
            )

    def _get_price_data(self, stock_code: str, days: int) -> Optional[pd.DataFrame]:
        if self.data_service is None:
            return None
        return self.data_service.get_price_data(stock_code, days)

    def _generate_signal(
        self, rsi: float, macd_status: str, ema_trend: str,
        vol_status: str, adx: float,
    ) -> tuple[str, float, str]:
        t = self._thresh
        buy_score = 0.0
        sell_score = 0.0

        # RSI
        rsi_oversold = float(t.rsi.oversold or 30)
        rsi_near_oversold = float(t.rsi.near_oversold or 45)
        rsi_near_overbought = float(t.rsi.near_overbought or 60)
        rsi_overbought = float(t.rsi.overbought or 70)

        if rsi < rsi_oversold:
            buy_score += float(t.rsi.oversold_buy_score or 2.0)
        elif rsi < rsi_near_oversold:
            buy_score += float(t.rsi.near_oversold_buy_score or 0.5)
        elif rsi > rsi_overbought:
            sell_score += float(t.rsi.overbought_sell_score or 2.0)
        elif rsi > rsi_near_overbought:
            sell_score += float(t.rsi.near_overbought_sell_score or 0.5)

        # MACD
        if macd_status == "金叉":
            buy_score += float(t.macd.golden_cross_score or 2.0)
        else:
            sell_score += float(t.macd.death_cross_score or 2.0)

        # EMA 趋势
        if ema_trend == "上升":
            buy_score += float(t.ema.uptrend_score or 1.5)
        else:
            sell_score += float(t.ema.downtrend_score or 1.5)

        # ADX（趋势强度）
        adx_threshold = float(t.adx.strong_threshold or 25)
        adx_multiplier = float(t.adx.multiplier or 1.2)
        if adx > adx_threshold:
            buy_score *= adx_multiplier
            sell_score *= adx_multiplier

        # 成交量确认
        vol_score = float(t.volume.confirmation_score or 1.0)
        if vol_status == "放量":
            if buy_score > sell_score:
                buy_score += vol_score
            else:
                sell_score += vol_score

        # Signal generation
        sig = t.signal
        diff = buy_score - sell_score
        strong_buy_diff = float(sig.strong_buy_diff or 3.0)
        strong_sell_diff = float(sig.strong_sell_diff or -3.0)
        weak_buy_diff = float(sig.weak_buy_diff or 1.0)
        weak_sell_diff = float(sig.weak_sell_diff or -1.0)

        if diff >= strong_buy_diff:
            return "BUY", float(sig.strong_buy_confidence or 0.80), f"技术面偏多(+{diff:.1f})"
        elif diff <= strong_sell_diff:
            return "SELL", float(sig.strong_sell_confidence or 0.75), f"技术面偏空({diff:.1f})"
        elif diff >= weak_buy_diff:
            return "BUY", float(sig.weak_buy_confidence or 0.65), f"技术面略偏多(+{diff:.1f})"
        elif diff <= weak_sell_diff:
            return "SELL", float(sig.weak_sell_confidence or 0.60), f"技术面略偏空({diff:.1f})"
        else:
            return "HOLD", float(sig.hold_confidence or 0.50), f"技术面中性({diff:.1f})"
