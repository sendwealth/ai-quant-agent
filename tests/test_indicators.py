#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
技术指标测试
Technical Indicators Tests
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import (
    sma, ema, atr, rsi, macd, 
    bollinger_bands, stochastic, williams_r,
    momentum, rate_of_change, detect_crossover
)


class TestSMA:
    """SMA测试"""
    
    def test_basic_calculation(self):
        """测试基本计算"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = sma(data, 3)
        
        # 第3个值应该是 (1+2+3)/3 = 2
        assert result.iloc[2] == 2.0
        
        # 第5个值应该是 (3+4+5)/3 = 4
        assert result.iloc[4] == 4.0
    
    def test_with_nan(self):
        """测试包含NaN的数据"""
        data = pd.Series([1, 2, np.nan, 4, 5])
        result = sma(data, 3)
        
        # SMA应该能处理NaN（但结果可能包含NaN）
        assert len(result) == len(data)
    
    def test_period_equals_length(self):
        """测试周期等于数据长度"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = sma(data, 5)
        
        # 只有一个有效值
        assert result.iloc[4] == 3.0


class TestEMA:
    """EMA测试"""
    
    def test_basic_calculation(self):
        """测试基本计算"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = ema(data, 3)
        
        # EMA应该比SMA反应更快
        assert not result.isnull().all()
    
    def test_ema_vs_sma(self):
        """测试EMA与SMA的区别"""
        data = pd.Series([10, 20, 30, 40, 50])
        
        ema_result = ema(data, 3)
        sma_result = sma(data, 3)
        
        # EMA应该比SMA更快上升
        assert ema_result.iloc[-1] > sma_result.iloc[-1]


class TestATR:
    """ATR测试"""
    
    def test_basic_calculation(self):
        """测试基本计算"""
        high = pd.Series([105, 106, 107, 108, 109])
        low = pd.Series([95, 96, 97, 98, 99])
        close = pd.Series([100, 101, 102, 103, 104])
        
        result = atr(high, low, close, period=3)
        
        # ATR应该为正数
        assert (result.dropna() > 0).all()
    
    def test_high_volatility(self):
        """测试高波动率"""
        # 高波动数据
        high = pd.Series([110, 90, 120, 80, 130])
        low = pd.Series([90, 70, 100, 60, 110])
        close = pd.Series([100, 80, 110, 70, 120])
        
        result_high = atr(high, low, close, period=3)
        
        # 低波动数据
        high2 = pd.Series([102, 101, 103, 102, 104])
        low2 = pd.Series([98, 99, 97, 98, 96])
        close2 = pd.Series([100, 100, 100, 100, 100])
        
        result_low = atr(high2, low2, close2, period=3)
        
        # 高波动应该有更高的ATR
        assert result_high.iloc[-1] > result_low.iloc[-1]


class TestRSI:
    """RSI测试"""
    
    def test_range(self):
        """测试RSI范围 (0-100)"""
        data = pd.Series([100, 101, 102, 101, 100, 99, 100, 101, 102, 103,
                          104, 105, 104, 103, 102])
        result = rsi(data, period=14)
        
        # RSI应该在0-100之间
        assert (result.dropna() >= 0).all()
        assert (result.dropna() <= 100).all()
    
    def test_uptrend(self):
        """测试上升趋势 (RSI应该较高)"""
        # 持续上涨
        data = pd.Series(range(100, 120))
        result = rsi(data, period=14)
        
        # RSI应该>70 (超买)
        assert result.iloc[-1] > 70
    
    def test_downtrend(self):
        """测试下降趋势 (RSI应该较低)"""
        # 持续下跌
        data = pd.Series(range(120, 100, -1))
        result = rsi(data, period=14)
        
        # RSI应该<30 (超卖)
        assert result.iloc[-1] < 30


class TestMACD:
    """MACD测试"""
    
    def test_basic_calculation(self):
        """测试基本计算"""
        data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        
        macd_line, signal_line, histogram = macd(data)
        
        # 应该返回三个序列
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
        
        # MACD应该等于快线减慢线
        assert not macd_line.isnull().all()
    
    def test_histogram(self):
        """测试柱状图计算"""
        data = pd.Series([100, 102, 104, 106, 108, 110, 112, 114, 116, 118,
                          120, 118, 116, 114, 112, 110, 108, 106, 104, 102])
        
        macd_line, signal_line, histogram = macd(data)
        
        # 柱状图应该等于MACD减去信号线
        expected_histogram = macd_line - signal_line
        
        # 允许浮点数误差
        assert np.allclose(histogram.dropna(), expected_histogram.dropna(), rtol=1e-10)


class TestBollingerBands:
    """布林带测试"""
    
    def test_basic_calculation(self):
        """测试基本计算"""
        data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        
        upper, middle, lower = bollinger_bands(data, period=20, std_dev=2.0)
        
        # 上轨 > 中轨 > 下轨
        assert (upper.dropna() > middle.dropna()).all()
        assert (middle.dropna() > lower.dropna()).all()
    
    def test_middle_is_sma(self):
        """测试中轨是否为SMA"""
        data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119])
        
        upper, middle, lower = bollinger_bands(data, period=20)
        expected_middle = sma(data, 20)
        
        # 中轨应该等于SMA
        assert np.allclose(middle.dropna(), expected_middle.dropna(), rtol=1e-10)


class TestStochastic:
    """随机指标测试"""
    
    def test_range(self):
        """测试K线和D线范围 (0-100)"""
        high = pd.Series([105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                          115, 116, 117, 118, 119])
        low = pd.Series([95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                         105, 106, 107, 108, 109])
        close = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                           110, 111, 112, 113, 114])
        
        k_line, d_line = stochastic(high, low, close, k_period=14, d_period=3)
        
        # K线和D线应该在0-100之间
        assert (k_line.dropna() >= 0).all()
        assert (k_line.dropna() <= 100).all()
        assert (d_line.dropna() >= 0).all()
        assert (d_line.dropna() <= 100).all()


class TestDetectCrossover:
    """交叉检测测试"""
    
    def test_bullish_crossover(self):
        """测试上穿信号"""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([3, 3, 3, 3, 3])
        
        signals = detect_crossover(series1, series2)
        
        # 应该在第2个位置检测到上穿
        assert signals.iloc[2] == 1
    
    def test_bearish_crossover(self):
        """测试下穿信号"""
        series1 = pd.Series([5, 4, 3, 2, 1])
        series2 = pd.Series([3, 3, 3, 3, 3])
        
        signals = detect_crossover(series1, series2)
        
        # 应该在第2个位置检测到下穿
        assert signals.iloc[2] == -1
    
    def test_no_crossover(self):
        """测试无交叉"""
        series1 = pd.Series([1, 2, 3, 4, 5])
        series2 = pd.Series([10, 11, 12, 13, 14])
        
        signals = detect_crossover(series1, series2)
        
        # 应该没有交叉信号
        assert (signals == 0).all()


# ========== 运行测试 ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
