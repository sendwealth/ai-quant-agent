#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日志配置
Logging Configuration

统一的日志配置和管理
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    log_level: str = "INFO",
    log_file: str = "logs/trading.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
):
    """
    配置日志系统

    Args:
        log_level: 日志级别
        log_file: 日志文件
        rotation: 日志轮转大小
        retention: 日志保留时间
    """
    # 移除默认处理器
    logger.remove()

    # 控制台输出
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    # 文件输出
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    # 错误日志单独文件
    error_log = log_path.parent / "error.log"
    logger.add(
        str(error_log),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=rotation,
        retention=retention,
        compression="zip",
    )

    logger.info(f"日志系统配置完成: {log_file}")


def get_logger(name: str = None):
    """
    获取logger实例

    Args:
        name: logger名称

    Returns:
        logger实例
    """
    if name:
        return logger.bind(name=name)
    return logger


# ========== 导出 ==========
__all__ = [
    "setup_logging",
    "get_logger",
]
