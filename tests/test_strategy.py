#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略测试 - 简化版
Strategy Tests - Simplified
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_loader import ConfigLoader


class TestConfigLoader:
    """配置加载器测试"""

    @pytest.fixture
    def config_loader(self):
        """创建配置加载器"""
        return ConfigLoader()

    def test_load_config(self, config_loader):
        """测试加载配置"""
        assert config_loader is not None
        assert config_loader.config is not None

    def test_get_stock_config(self, config_loader):
        """测试获取股票配置"""
        stock_config = config_loader.get_stock_config("300750")
        assert stock_config is not None
        assert "weight" in stock_config
        assert stock_config["weight"] >= 0.4  # 允许配置变化

    def test_get_all_stocks(self, config_loader):
        """测试获取所有股票"""
        stocks = config_loader.get_all_stocks()
        assert stocks is not None
        assert len(stocks) > 0

    def test_get_risk_params(self, config_loader):
        """测试获取风险参数"""
        risk_params = config_loader.get_risk_params()
        assert risk_params is not None
        assert "stop_loss" in risk_params
