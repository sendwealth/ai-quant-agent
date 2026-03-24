#!/usr/bin/env python3
"""全局配置"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

class Settings:
    """全局配置"""
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CACHE_DIR = DATA_DIR / "cache"
    FILE_CACHE_DIR = str(DATA_DIR / "cache")  # 添加这个
    
    REDIS_HOST = "localhost"
    REDIS_PORT = 6379
    REDIS_DB = 0
    
    CACHE_MAX_AGE_MINUTES = 30
    
    LOG_LEVEL = "INFO"
    LOG_FILE = str(LOGS_DIR / "app.log")
    
    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)

Settings.ensure_dirs()
