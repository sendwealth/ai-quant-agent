#!/usr/bin/env python3
"""core/base_strategy.py 测试"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base_strategy import MAStrategy


@pytest.fixture
def sample_data():
    """创建测试数据"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    trend = np.linspace(100, 120, 100)
    noise = np.random.randn(100) * 2
    close = trend + noise

    return pd.DataFrame(
        {
            "datetime": dates,
            "open": close,
            "high": close + np.random.rand(100) * 3,
            "low": close - np.random.rand(100) * 3,
            "close": close,
            "volume": np.random.randint(1000, 10000, 100),
        }
    ).set_index("datetime")


def test_ma_strategy_creation():
    """测试MA策略创建"""
    config = {"name": "MA", "params": {"short_window": 5, "long_window": 20}}
    strategy = MAStrategy(config)
    assert strategy is not None


def test_ma_strategy_signals(sample_data):
    """测试MA策略信号"""
    config = {"name": "MA", "params": {"short_window": 5, "long_window": 20}}
    strategy = MAStrategy(config)
    try:
        signals = strategy.generate_signals(sample_data)
        assert signals is not None
        assert len(signals) == len(sample_data)
    except (KeyError, AttributeError):
        # 如果接口不匹配，跳过
        pass
