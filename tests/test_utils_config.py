#!/usr/bin/env python3
"""utils/config.py 测试"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import Config, get_config


def test_get_config():
    """测试获取配置"""
    config = get_config()
    assert config is not None


def test_config_singleton():
    """测试配置单例"""
    config1 = get_config()
    config2 = get_config()
    assert config1 is config2
