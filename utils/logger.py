"""
日志模块
配置和管理系统日志
"""

import sys
from pathlib import Path
from loguru import logger
from .config import get_config


def setup_logger(config_path: str = None):
    """
    设置日志系统

    Args:
        config_path: 配置文件路径
    """
    # 移除默认处理器
    logger.remove()

    # 获取配置
    config = get_config()
    log_level = config.get('logging', 'level', default='INFO')
    log_file = config.get('logging', 'file', default='logs/quant_agent.log')
    rotation = config.get('logging', 'rotation', default='10 MB')
    retention = config.get('logging', 'retention', default='30 days')

    # 创建日志目录
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # 控制台输出（带颜色）
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True,
        enqueue=True
    )

    # 文件输出（详细）
    logger.add(
        log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True
    )

    # 错误日志单独文件
    logger.add(
        log_path.parent / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        compression="zip",
        enqueue=True
    )

    logger.info(f"日志系统已初始化，级别: {log_level}")


def get_logger(name: str = None):
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    return logger.bind(name=name) if name else logger


# 自动初始化
setup_logger()
