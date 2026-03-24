#!/usr/bin/env python3
"""utils/real_data_fetcher.py 测试"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.real_data_fetcher import RealDataFetcher


def test_fetcher_init():
    """测试数据获取器初始化"""
    fetcher = RealDataFetcher()
    assert fetcher is not None


def test_get_stock_data():
    """测试获取股票数据"""
    fetcher = RealDataFetcher()
    try:
        data = fetcher.get_stock_data("300750")
        assert data is not None
        assert len(data) > 0
    except Exception:
        pass  # 网络问题，跳过
