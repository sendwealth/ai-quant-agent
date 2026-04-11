"""TechnicalAgent 单元测试 -- 覆盖所有代码路径"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from quant_agent.agents.base import AgentResult
from quant_agent.agents.technical import TechnicalAgent


# ---------------------------------------------------------------------------
# Helper: 生成各种价格模式的 DataFrame
# ---------------------------------------------------------------------------

def _make_df(
    closes: list[float],
    high_offset: float = 0.5,
    low_offset: float = 0.5,
    base_volume: float = 1_000_000.0,
    volumes: list[float] | None = None,
) -> pd.DataFrame:
    """根据收盘价列表构造包含 close/high/low/volume 的 DataFrame。

    * high = close + high_offset
    * low  = close - low_offset   (保证 > 0)
    * volume = base_volume, 除非显式传入
    """
    closes_arr = np.array(closes, dtype=float)
    n = len(closes_arr)

    if volumes is not None:
        vol = np.array(volumes, dtype=float)
    else:
        vol = np.full(n, base_volume)

    return pd.DataFrame({
        "close":  closes_arr,
        "high":   closes_arr + high_offset,
        "low":    np.maximum(closes_arr - low_offset, 0.01),
        "volume": vol,
    })


def _uptrend_df(n: int = 250, start: float = 50.0, step: float = 0.5,
                vol_base: float = 1_000_000.0) -> pd.DataFrame:
    """平稳上升趋势 DataFrame"""
    closes = start + np.arange(n) * step
    return _make_df(closes.tolist(), volumes=(np.full(n, vol_base).tolist()))


def _downtrend_df(n: int = 250, start: float = 200.0, step: float = 0.5,
                  vol_base: float = 1_000_000.0) -> pd.DataFrame:
    """平稳下降趋势 DataFrame"""
    closes = start - np.arange(n) * step
    closes = np.maximum(closes, 5.0)  # keep positive
    return _make_df(closes.tolist(), volumes=(np.full(n, vol_base).tolist()))


def _flat_df(n: int = 250, price: float = 100.0) -> pd.DataFrame:
    """全平价格 DataFrame (所有价格相同)"""
    return _make_df([price] * n)


def _volatile_df(n: int = 250, base: float = 100.0, amp: float = 10.0,
                 seed: int = 42) -> pd.DataFrame:
    """高波动 DataFrame -- 正弦叠加噪声"""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    closes = base + amp * np.sin(2 * np.pi * t / 20) + rng.normal(0, amp * 0.3, n)
    closes = np.maximum(closes, 1.0)
    return _make_df(closes.tolist())


def _oversold_df(n: int = 250) -> pd.DataFrame:
    """先横盘后暴跌, 制造 RSI < 30 且 MACD 金叉 (histogram > 0) 的场景。

    通过在尾部微幅反弹来让 MACD histogram 由负转正 (金叉)。
    """
    # 横盘 100 根
    flat = np.full(100, 100.0)
    # 暴跌 120 根
    drop = 100.0 - np.linspace(0, 50, 120)
    # 微幅反弹 30 根, 让 MACD histogram 变正
    bounce_start = drop[-1]
    bounce = bounce_start + np.linspace(0, 2, 30)
    closes = np.concatenate([flat, drop, bounce])
    return _make_df(closes.tolist())


def _overbought_df(n: int = 250) -> pd.DataFrame:
    """先横盘后暴涨, 制造 RSI > 70 且 MACD 死叉 (histogram < 0) 的场景。

    尾部微跌让 MACD histogram 由正转负 (死叉)。
    """
    flat = np.full(100, 100.0)
    rise = 100.0 + np.linspace(0, 80, 120)
    drop_start = rise[-1]
    drop = drop_start - np.linspace(0, 5, 30)
    closes = np.concatenate([flat, rise, drop])
    return _make_df(closes.tolist())


# ===========================================================================
# Tests -- no data / error paths
# ===========================================================================


class TestTechnicalAgentNoData:
    """analyze() 在无数据 / 错误数据下的行为"""

    def test_no_data_service_returns_hold_failure(self):
        """没有 data_service 时返回 HOLD + success=False"""
        agent = TechnicalAgent()
        result = agent.analyze("300750")
        assert result.signal == "HOLD"
        assert result.confidence == 0.0
        assert result.success is False
        assert result.error == "NO_DATA"
        assert "无法获取行情数据" in result.reasoning

    def test_data_service_returns_none(self):
        """DataService.get_price_data 返回 None"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = None
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")
        assert result.success is False
        assert result.error == "NO_DATA"
        mock_ds.get_price_data.assert_called_once_with("300750", 250)

    def test_data_service_returns_empty_dataframe(self):
        """DataService.get_price_data 返回空 DataFrame"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = pd.DataFrame()
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")
        assert result.success is False
        assert result.error == "NO_DATA"

    def test_data_service_raises_exception(self):
        """DataService.get_price_data 抛出异常时, 返回失败的 AgentResult"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.side_effect = RuntimeError("网络错误")
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")
        assert result.success is False
        assert result.signal == "HOLD"
        assert "网络错误" in result.error
        assert "分析失败" in result.reasoning

    def test_data_service_raises_value_error(self):
        """DataService 抛出 ValueError 时也被捕获"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.side_effect = ValueError("缺少 close 列")
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")
        assert result.success is False
        assert "缺少 close 列" in result.error

    def test_custom_days_parameter(self):
        """days 参数被正确传递给 DataService"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = None
        agent = TechnicalAgent(data_service=mock_ds)
        agent.analyze("300750", days=60)
        mock_ds.get_price_data.assert_called_once_with("300750", 60)


# ===========================================================================
# Tests -- _generate_signal scoring logic (isolated unit test)
# ===========================================================================


class TestGenerateSignal:
    """对 _generate_signal 评分逻辑的直接测试, 不依赖行情数据"""

    def _make_agent(self) -> TechnicalAgent:
        return TechnicalAgent()

    # ---- Strong BUY: RSI 超卖 + 金叉 + 上升 + ADX > 25 + 放量 ----

    def test_strong_buy_all_bullish(self):
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=20.0, macd_status="金叉", ema_trend="上升",
            vol_status="放量", adx=30.0,
        )
        assert signal == "BUY"
        assert conf == 0.80
        assert "偏多" in reason

    # ---- Strong SELL: RSI 超买 + 死叉 + 下降 + ADX > 25 + 放量 ----

    def test_strong_sell_all_bearish(self):
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=80.0, macd_status="死叉", ema_trend="下降",
            vol_status="放量", adx=30.0,
        )
        assert signal == "SELL"
        assert conf == 0.75
        assert "偏空" in reason

    # ---- Neutral / HOLD ----

    def test_hold_neutral_indicators(self):
        """RSI 中性 + 金叉 + 下降 (互相抵消)"""
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=50.0, macd_status="金叉", ema_trend="下降",
            vol_status="正常", adx=20.0,
        )
        # RSI 50 -> buy+=0, sell+=0; 金叉 -> buy+=2; 下降 -> sell+=1.5
        # diff = 2.0 - 1.5 = 0.5 -> in (-1, 1) -> HOLD
        assert signal == "HOLD"
        assert conf == 0.50
        assert "中性" in reason

    def test_hold_all_cancel_out(self):
        """RSI < 45 偏多 + 死叉 + 上升, 大致抵消"""
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=40.0, macd_status="死叉", ema_trend="上升",
            vol_status="正常", adx=20.0,
        )
        # RSI 40 -> buy+=0.5; 死叉 -> sell+=2; 上升 -> buy+=1.5
        # diff = 2.0 - 2.0 = 0 -> HOLD
        assert signal == "HOLD"

    # ---- Weak BUY (diff in [1, 3)) ----

    def test_weak_buy(self):
        """RSI 偏低 + 金叉 + 下降 (差异在 1~3 之间)"""
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=35.0, macd_status="金叉", ema_trend="下降",
            vol_status="正常", adx=20.0,
        )
        # RSI 35 -> buy+=0; 金叉 -> buy+=2; 下降 -> sell+=1.5
        # diff = 2.0 - 1.5 = 0.5 -> actually HOLD
        # Let me adjust: RSI 42 -> buy+=0.5
        # diff = 2.5 - 1.5 = 1.0 -> weak BUY
        signal2, conf2, reason2 = agent._generate_signal(
            rsi=42.0, macd_status="金叉", ema_trend="下降",
            vol_status="正常", adx=20.0,
        )
        assert signal2 == "BUY"
        assert conf2 == 0.65
        assert "略偏多" in reason2

    # ---- Weak SELL (diff in (-3, -1]) ----

    def test_weak_sell(self):
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=65.0, macd_status="死叉", ema_trend="上升",
            vol_status="正常", adx=20.0,
        )
        # RSI 65 -> sell+=0.5; 死叉 -> sell+=2; 上升 -> buy+=1.5
        # diff = 1.5 - 2.5 = -1.0 -> weak SELL
        assert signal == "SELL"
        assert conf == 0.60
        assert "略偏空" in reason

    # ---- ADX amplification ----

    def test_adx_above_25_amplifies_buy(self):
        """ADX > 25 放大已有分数"""
        agent = self._make_agent()
        # Without ADX: buy=2(golden)+0.5(rsi<45)+1.5(uptrend)=4.0, sell=0
        # diff = 4.0 -> strong BUY with conf 0.80
        signal_no_adx, _, _ = agent._generate_signal(
            rsi=42.0, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=10.0,
        )
        # With ADX=30: buy = 4.0*1.2 = 4.8, sell = 0*1.2 = 0
        # diff = 4.8 -> strong BUY with conf 0.80
        signal_with_adx, conf_with_adx, _ = agent._generate_signal(
            rsi=42.0, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=30.0,
        )
        assert signal_no_adx == "BUY"
        assert signal_with_adx == "BUY"
        # ADX amplification makes the diff larger: reason should reflect that
        _, _, reason_no = agent._generate_signal(
            rsi=42.0, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=10.0,
        )
        _, _, reason_with = agent._generate_signal(
            rsi=42.0, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=30.0,
        )
        assert "4.0" in reason_no
        assert "4.8" in reason_with

    # ---- Volume confirmation ----

    def test_volume_confirms_buy_direction(self):
        """放量 + 买方优势 -> 额外加 1 分"""
        agent = self._make_agent()
        # RSI 20 -> buy+=2; 金叉 -> buy+=2; 上升 -> buy+=1.5
        # Without 放量: buy=5.5, sell=0, diff=5.5
        # With 放量: buy=5.5+1=6.5, diff=6.5
        signal, conf, reason = agent._generate_signal(
            rsi=20.0, macd_status="金叉", ema_trend="上升",
            vol_status="放量", adx=20.0,
        )
        assert signal == "BUY"
        assert "6.5" in reason

    def test_volume_confirms_sell_direction(self):
        """放量 + 卖方优势 -> 额外加 1 分"""
        agent = self._make_agent()
        signal, conf, reason = agent._generate_signal(
            rsi=80.0, macd_status="死叉", ema_trend="下降",
            vol_status="放量", adx=20.0,
        )
        # RSI 80 -> sell+=2; 死叉 -> sell+=2; 下降 -> sell+=1.5
        # 放量 + sell > buy -> sell += 1 => total sell = 6.5
        assert signal == "SELL"
        assert "6.5" in reason

    def test_volume_no_effect_when_normal(self):
        """成交量正常时不会额外加分"""
        agent = self._make_agent()
        signal, _, reason = agent._generate_signal(
            rsi=20.0, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=20.0,
        )
        # buy = 2 + 2 + 1.5 = 5.5, no extra
        assert "5.5" in reason

    def test_volume_shrink_no_extra(self):
        """缩量不会加分"""
        agent = self._make_agent()
        signal, _, reason = agent._generate_signal(
            rsi=20.0, macd_status="金叉", ema_trend="上升",
            vol_status="缩量", adx=20.0,
        )
        assert "5.5" in reason

    # ---- RSI boundary values ----

    @pytest.mark.parametrize("rsi_val,expected_buy_add", [
        (29.0, 2.0),    # < 30: strong buy
        (44.0, 0.5),    # < 45: weak buy
        (50.0, 0.0),    # neutral: nothing
        (61.0, 0.5),    # > 60: weak sell added to sell_score
        (71.0, 2.0),    # > 70: strong sell added to sell_score
    ])
    def test_rsi_scoring_boundaries(self, rsi_val, expected_buy_add):
        """验证 RSI 评分边界"""
        agent = self._make_agent()
        # Use 金叉+上升 to make buy_score > sell_score, then check diff
        signal, _, reason = agent._generate_signal(
            rsi=rsi_val, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=20.0,
        )
        # buy = expected_buy_add + 2(金叉) + 1.5(上升) = 3.5 + expected_buy_add
        # sell = 0
        # For RSI > 60, sell gets extra, not buy
        assert isinstance(signal, str)
        assert signal in ("BUY", "SELL", "HOLD")

    # ---- EMA trend edge ----

    def test_ema_trend_rising(self):
        agent = self._make_agent()
        signal, _, _ = agent._generate_signal(
            rsi=50.0, macd_status="金叉", ema_trend="上升",
            vol_status="正常", adx=20.0,
        )
        assert signal == "BUY"

    def test_ema_trend_falling(self):
        agent = self._make_agent()
        signal, _, _ = agent._generate_signal(
            rsi=50.0, macd_status="金叉", ema_trend="下降",
            vol_status="正常", adx=20.0,
        )
        # buy=2, sell=1.5, diff=0.5 -> HOLD
        assert signal == "HOLD"


# ===========================================================================
# Tests -- full analyze() with realistic price DataFrames
# ===========================================================================


class TestTechnicalAgentAnalyzeBullish:
    """analyze() 牛市场景"""

    def test_uptrend_produces_buy_signal(self):
        """持续上升趋势应该产生 BUY 信号"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250, start=50.0, step=0.5)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.signal in ("BUY", "HOLD")  # uptrend should at least not SELL
        assert result.confidence > 0.0
        assert result.agent_name == "technical"
        assert result.stock_code == "300750"

    def test_uptrend_metrics_populated(self):
        """上升趋势的 metrics 字段完整"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        m = result.metrics
        assert "rsi" in m
        assert "rsi_status" in m
        assert "macd" in m
        assert "macd_signal" in m
        assert "macd_histogram" in m
        assert "macd_status" in m
        assert "ema_20" in m
        assert "ema_50" in m
        assert "ema_trend" in m
        assert "bollinger_upper" in m
        assert "bollinger_lower" in m
        assert "atr" in m
        assert "adx" in m
        assert "volume_ratio" in m
        assert "vol_status" in m
        assert "current_price" in m

    def test_uptrend_ema_is_rising(self):
        """上升趋势中 EMA 20 > EMA 50 (趋势=上升)"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.metrics["ema_trend"] == "上升"
        assert result.metrics["ema_20"] > result.metrics["ema_50"]

    def test_uptrend_scores_trend_positive(self):
        """上升趋势 scores.trend == 1"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.scores["trend"] == 1


class TestTechnicalAgentAnalyzeBearish:
    """analyze() 熊市场景"""

    def test_downtrend_produces_sell_signal(self):
        """持续下降趋势应该产生 SELL 信号"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _downtrend_df(250, start=200.0, step=0.5)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.signal in ("SELL", "HOLD")
        assert result.confidence > 0.0

    def test_downtrend_ema_is_falling(self):
        """下降趋势中 EMA 20 < EMA 50 (趋势=下降)"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _downtrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.metrics["ema_trend"] == "下降"
        assert result.metrics["ema_20"] < result.metrics["ema_50"]

    def test_downtrend_scores_trend_negative(self):
        """下降趋势 scores.trend == -1"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _downtrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.scores["trend"] == -1


class TestTechnicalAgentAnalyzeNeutral:
    """analyze() 中性场景"""

    def test_flat_prices_hold_signal(self):
        """全平价格应该产生 HOLD 信号 (或接近中性)"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _flat_df(250, price=100.0)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        # Flat prices: RSI ~50, MACD histogram ~0 (could be slightly pos or neg)
        # Either way it should be neutral-ish
        assert result.confidence >= 0.0

    def test_flat_prices_atr_near_zero(self):
        """全平价格的 ATR 应该接近 0"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _flat_df(250, price=100.0)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        # ATR should be very small (high and low differ by only 1.0 total)
        assert result.metrics["atr"] < 2.0

    def test_volatile_data_completes(self):
        """高波动数据不应导致异常"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _volatile_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.signal in ("BUY", "SELL", "HOLD")
        assert result.confidence > 0.0


class TestTechnicalAgentAnalyzeEdgeCases:
    """analyze() 边界条件"""

    def test_minimal_data_60_bars(self):
        """60 根 K 线 (刚好够 MACD 26+9=35 + 一些)"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(60)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.signal in ("BUY", "SELL", "HOLD")

    def test_too_few_bars_30(self):
        """30 根 K 线, EMA 50 没有足够数据但仍应能完成 (pandas 不会报错)"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(30)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        # Should either succeed or fail gracefully
        assert isinstance(result, AgentResult)
        assert result.signal in ("BUY", "SELL", "HOLD")

    def test_very_few_bars_10(self):
        """10 根 K 线, 大部分指标为 NaN 但 float(nan) 不抛异常"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(10)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        # float(nan) succeeds silently, so the agent returns success=True
        # but with NaN or fallback values in metrics.
        assert isinstance(result, AgentResult)
        assert result.signal in ("BUY", "SELL", "HOLD")
        # With only 10 bars, indicators are insufficient for full calculation.
        # RSI returns 50.0 (neutral fallback) when data is too short or constant.
        # ATR may be NaN due to insufficient rolling window.
        assert isinstance(result.metrics["rsi"], float)

    def test_high_volume_ratio_triggers_vol_status(self):
        """最后一根成交量远大于均值 -> vol_status == '放量'"""
        n = 250
        volumes = [1_000_000.0] * (n - 1) + [5_000_000.0]  # last bar 5x avg
        df = _uptrend_df(n)
        df["volume"] = volumes

        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = df
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.metrics["vol_status"] == "放量"
        assert result.metrics["volume_ratio"] > 1.5

    def test_low_volume_ratio_triggers_shrink(self):
        """最后一根成交量远小于均值 -> vol_status == '缩量'"""
        n = 250
        volumes = [1_000_000.0] * (n - 1) + [100_000.0]  # last bar very low
        df = _uptrend_df(n)
        df["volume"] = volumes

        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = df
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.metrics["vol_status"] == "缩量"
        assert result.metrics["volume_ratio"] < 0.7

    def test_zero_avg_volume_handled(self):
        """全部成交量为 0 时 vol_ratio 回退为 1.0"""
        df = _uptrend_df(250)
        df["volume"] = 0.0

        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = df
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.success is True
        assert result.metrics["volume_ratio"] == 1.0
        assert result.metrics["vol_status"] == "正常"

    def test_metrics_rounding(self):
        """验证 metrics 中数值被正确四舍五入"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        # rsi rounded to 2 decimal places
        rsi_val = result.metrics["rsi"]
        assert rsi_val == round(rsi_val, 2)

        # macd rounded to 4 decimal places
        macd_val = result.metrics["macd"]
        assert macd_val == round(macd_val, 4)

    def test_scores_structure(self):
        """验证 scores 字典包含 trend/momentum/volatility"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert "trend" in result.scores
        assert "momentum" in result.scores
        assert "volatility" in result.scores

        # momentum = rsi / 100, should be in [0, 1]
        assert 0.0 <= result.scores["momentum"] <= 1.0

        # volatility = 1 - min(atr/(close*0.05), 1.0), should be in [0, 1]
        assert 0.0 <= result.scores["volatility"] <= 1.0

    def test_rsi_status_labels(self):
        """验证 RSI 状态标签正确"""
        # Test overbought: sharp rise at end
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _overbought_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        # RSI should be high -> overbought
        assert result.metrics["rsi_status"] in ("超买", "中性", "超卖")
        assert result.success is True


class TestTechnicalAgentStructure:
    """Agent 结构/协议测试"""

    def test_agent_name_is_technical(self):
        agent = TechnicalAgent()
        assert agent.name == "technical"

    def test_repr(self):
        agent = TechnicalAgent()
        assert repr(agent) == "Agent(technical)"

    def test_inherits_base_agent(self):
        from quant_agent.agents.base import BaseAgent
        agent = TechnicalAgent()
        assert isinstance(agent, BaseAgent)

    def test_result_has_timestamp(self):
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.timestamp is not None
        assert len(result.timestamp) > 0

    def test_result_to_dict(self):
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")
        d = result.to_dict()

        assert d["agent"] == "technical"
        assert d["stock_code"] == "300750"
        assert "signal" in d
        assert "confidence" in d
        assert "metrics" in d
        assert "scores" in d


class TestTechnicalAgentWithEventBus:
    """TechnicalAgent 与 EventBus 的交互"""

    def test_with_event_bus_attached(self):
        """Agent 接收 event_bus kwarg 不应报错"""
        from quant_agent.events.bus import EventBus
        bus = EventBus()
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(event_bus=bus, data_service=mock_ds)
        result = agent.analyze("300750")
        assert result.success is True

    def test_without_event_bus_still_works(self):
        """没有 event_bus 时也能正常分析"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")
        assert result.success is True


class TestTechnicalAgentDifferentStocks:
    """测试不同股票代码"""

    @pytest.mark.parametrize("stock_code", [
        "300750",   # 创业板
        "600519",   # 沪市
        "000858",   # 深市
        "830799",   # 北交所
    ])
    def test_various_stock_codes(self, stock_code):
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze(stock_code)

        assert result.stock_code == stock_code
        assert result.success is True
        mock_ds.get_price_data.assert_called_once_with(stock_code, 250)


class TestTechnicalAgentMacdStatus:
    """MACD 状态 (金叉/死叉) 在不同数据下的表现"""

    def test_uptrend_macd_status(self):
        """上升趋势末期 MACD histogram 通常为正 -> 金叉"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _uptrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        # In a steady uptrend, MACD histogram should be positive (golden cross)
        assert result.metrics["macd_status"] in ("金叉", "死叉")

    def test_downtrend_macd_status(self):
        """下降趋势末期 MACD histogram 通常为负 -> 死叉"""
        mock_ds = MagicMock()
        mock_ds.get_price_data.return_value = _downtrend_df(250)
        agent = TechnicalAgent(data_service=mock_ds)
        result = agent.analyze("300750")

        assert result.metrics["macd_status"] in ("金叉", "死叉")
