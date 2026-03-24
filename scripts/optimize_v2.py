#!/usr/bin/env python3
"""代码优化脚本 v2"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent  # 项目根目录

print(f"项目根目录: {PROJECT_ROOT}")
print(f"存在: {PROJECT_ROOT.exists()}")

# 创建 utils/logger.py
logger_code = '''#!/usr/bin/env python3
"""统一的日志配置"""

import logging
from pathlib import Path

def get_logger(name: str, log_file: str = None, level=logging.INFO):
    """获取配置好的 logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
'''

utils_dir = PROJECT_ROOT / "utils"
utils_dir.mkdir(parents=True, exist_ok=True)
logger_file = utils_dir / "logger.py"

with open(logger_file, "w", encoding="utf-8") as f:
    f.write(logger_code)

print(f"✅ 创建: {logger_file}")

# 创建 config/settings.py
config_code = '''#!/usr/bin/env python3
"""全局配置"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

class Settings:
    """全局配置"""
    PROJECT_ROOT = PROJECT_ROOT
    DATA_DIR = PROJECT_ROOT / "data"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CACHE_DIR = DATA_DIR / "cache"
    
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
'''

config_dir = PROJECT_ROOT / "config"
config_dir.mkdir(parents=True, exist_ok=True)
config_file = config_dir / "settings.py"

with open(config_file, "w", encoding="utf-8") as f:
    f.write(config_code)

print(f"✅ 创建: {config_file}")

print("\n✅ 优化完成！")
