#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能测试
Performance Tests

测试系统性能和缓存效果
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cache import get_cache, get_indicator_cache
from core.indicators import ema, macd, rsi, sma
from utils.performance import get_monitor, monitor


class TestIndicatorPerformance:
    """指标性能测试"""

    @pytest.fixture
    def large_data(self):
        """生成大数据集"""
        np.random.seed(42)
        size = 10000  # 1万条数据

        return pd.DataFrame(
            {
                "open": np.random.randn(size) * 10 + 100,
                "high": np.random.randn(size) * 10 + 105,
                "low": np.random.randn(size) * 10 + 95,
                "close": np.random.randn(size) * 10 + 100,
                "volume": np.random.randint(1000, 10000, size),
            }
        )

    def test_sma_performance(self, large_data):
        """测试SMA性能"""
        start = time.time()

        for period in [5, 10, 20, 30, 60]:
            result = sma(large_data["close"], period)

        elapsed = time.time() - start

        # 1万条数据，5个周期，应该在0.1秒内完成
        assert elapsed < 0.1, f"SMA性能测试失败: {elapsed:.3f}s"
        print(f"✅ SMA性能: {elapsed:.3f}s (1万条×5周期)")

    def test_ema_performance(self, large_data):
        """测试EMA性能"""
        start = time.time()

        for period in [5, 10, 20, 30, 60]:
            result = ema(large_data["close"], period)

        elapsed = time.time() - start

        assert elapsed < 0.1, f"EMA性能测试失败: {elapsed:.3f}s"
        print(f"✅ EMA性能: {elapsed:.3f}s (1万条×5周期)")

    def test_rsi_performance(self, large_data):
        """测试RSI性能"""
        start = time.time()

        result = rsi(large_data["close"], 14)

        elapsed = time.time() - start

        assert elapsed < 0.1, f"RSI性能测试失败: {elapsed:.3f}s"
        print(f"✅ RSI性能: {elapsed:.3f}s (1万条)")

    def test_macd_performance(self, large_data):
        """测试MACD性能"""
        start = time.time()

        macd_line, signal_line, histogram = macd(large_data["close"])

        elapsed = time.time() - start

        assert elapsed < 0.1, f"MACD性能测试失败: {elapsed:.3f}s"
        print(f"✅ MACD性能: {elapsed:.3f}s (1万条)")

    def test_all_indicators_performance(self, large_data):
        """测试所有指标性能"""
        start = time.time()

        # 计算所有指标
        large_data["ma10"] = sma(large_data["close"], 10)
        large_data["ma30"] = sma(large_data["close"], 30)
        large_data["ema10"] = ema(large_data["close"], 10)
        large_data["rsi"] = rsi(large_data["close"], 14)
        macd_line, signal_line, histogram = macd(large_data["close"])
        large_data["macd"] = macd_line
        large_data["atr"] = large_data["high"] - large_data["low"]  # 简化

        elapsed = time.time() - start

        # 所有指标应该在0.5秒内完成
        assert elapsed < 0.5, f"综合性能测试失败: {elapsed:.3f}s"
        print(f"✅ 综合性能: {elapsed:.3f}s (1万条×7指标)")


class TestCachePerformance:
    """缓存性能测试"""

    def test_cache_hit_rate(self):
        """测试缓存命中率"""
        cache = get_cache()
        cache.clear()

        # 第一次访问 (miss)
        data1 = cache.get("test_key")
        assert data1 is None

        # 设置缓存
        test_data = pd.Series([1, 2, 3, 4, 5])
        cache.set("test_key", test_data)

        # 第二次访问
        data2 = cache.get("test_key")
        assert data2 is not None

        stats = cache.get_stats()

        # 命中率应该>0
        assert stats["hit_rate"] > 0, f"缓存命中率测试失败: {stats['hit_rate']}"
        print(f"✅ 缓存命中率: {stats['hit_rate']*100:.1f}%")

    def test_indicator_cache_performance(self):
        """测试指标缓存性能"""
        cache = get_indicator_cache()

        # 生成测试数据
        np.random.seed(42)
        close = pd.Series(np.random.randn(1000) * 10 + 100)

        # 第一次计算 (无缓存)
        start = time.time()
        result1 = sma(close, 10)
        time_without_cache = time.time() - start

        # 设置缓存
        cache.set_indicator("TEST", "sma", {"period": 10}, result1)

        # 从缓存读取
        start = time.time()
        result2 = cache.get_indicator("TEST", "sma", {"period": 10})
        time_with_cache = time.time() - start

        # 缓存应该更快
        speedup = time_without_cache / time_with_cache if time_with_cache > 0 else 0

        print(
            f"✅ 缓存加速: {speedup:.1f}x (无缓存:{time_without_cache:.6f}s, 有缓存:{time_with_cache:.6f}s)"
        )


class TestSystemPerformance:
    """系统性能测试"""

    def test_memory_usage(self):
        """测试内存使用"""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 加载数据
        data_frames = []
        for i in range(10):
            df = pd.DataFrame({"close": np.random.randn(1000) * 10 + 100})
            data_frames.append(df)

        final_memory = process.memory_info().rss / 1024 / 1024

        memory_increase = final_memory - initial_memory

        # 内存增加应该<50MB
        assert memory_increase < 50, f"内存使用测试失败: {memory_increase:.1f}MB"
        print(f"✅ 内存使用: {memory_increase:.1f}MB (10个DataFrame)")

    def test_concurrent_processing(self):
        """测试并发处理"""
        import time
        from concurrent.futures import ThreadPoolExecutor

        def calculate_indicator(data):
            return sma(data, 10)

        # 准备数据
        datasets = []
        for i in range(10):
            datasets.append(pd.Series(np.random.randn(1000) * 10 + 100))

        # 串行处理
        start = time.time()
        results_serial = [calculate_indicator(data) for data in datasets]
        serial_time = time.time() - start

        # 并行处理
        start = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results_parallel = list(executor.map(calculate_indicator, datasets))
        parallel_time = time.time() - start

        speedup = serial_time / parallel_time if parallel_time > 0 else 0

        print(f"✅ 并发加速: {speedup:.2f}x (串行:{serial_time:.3f}s, 并行:{parallel_time:.3f}s)")


# ========== 运行测试 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
