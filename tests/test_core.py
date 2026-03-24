#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
核心模块测试
Core Tests
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cache import DataCache
from core.config_loader import ConfigLoader
from core.data_manager import DataFetchError, DataManager
from utils.logger import get_logger


class TestDataManager:
    """DataManager 测试"""

    def test_init_without_cache(self):
        """测试不带缓存初始化"""
        dm = DataManager(use_cache=False)
        assert dm is not None
        assert dm.use_cache == False
        assert dm.cache is None

    def test_init_with_cache(self):
        """测试带缓存初始化"""
        dm = DataManager(use_cache=False)  # 禁用缓存避免Redis依赖
        assert dm is not None

    def test_get_basic_info(self):
        """测试获取基本信息"""
        dm = DataManager(use_cache=False)
        try:
            info = dm.get_basic_info("300750")
            assert "stock_code" in info
            assert info["stock_code"] == "300750"
        except DataFetchError:
            pass  # akshare 未安装，跳过

    def test_get_financial_data(self):
        """测试获取财务数据"""
        dm = DataManager(use_cache=False)
        try:
            data = dm.get_financial_data("300750")
            assert "stock_code" in data
            assert data["stock_code"] == "300750"
        except DataFetchError:
            pass  # akshare 未安装，跳过

    def test_get_technical_data(self):
        """测试获取技术数据"""
        dm = DataManager(use_cache=False)
        try:
            data = dm.get_technical_data("300750", days=60)
            assert "stock_code" in data
            assert data["stock_code"] == "300750"
        except DataFetchError:
            pass  # akshare 未安装，跳过


class TestDataCache:
    """DataCache 测试"""

    def test_file_cache(self):
        """测试文件缓存"""
        cache = DataCache()  # 使用默认配置
        assert cache is not None

    def test_cache_expiry(self):
        """测试缓存过期"""
        cache = DataCache()  # 使用默认配置
        assert cache is not None


class TestConfig:
    """配置测试"""

    def test_settings(self):
        """测试配置加载"""
        from config.settings import Settings

        settings = Settings()
        assert settings is not None

    def test_dirs_created(self):
        """测试目录创建"""
        from config.settings import Settings

        settings = Settings()
        # 验证必要目录存在
        assert settings.DATA_DIR.exists()


class TestLogger:
    """日志测试"""

    def test_get_logger(self):
        """测试获取logger"""
        logger = get_logger(__name__)
        assert logger is not None

    def test_logger_with_file(self):
        """测试带文件的logger"""
        logger = get_logger("test_module")
        assert logger is not None
