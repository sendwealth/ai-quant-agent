#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异常处理模块
Exception Handler Module

统一的异常处理和错误恢复
"""

from typing import Callable, Any, Optional
from loguru import logger
import functools
import traceback


class QuantException(Exception):
    """量化交易异常基类"""
    pass


class DataError(QuantException):
    """数据错误"""
    pass


class StrategyError(QuantException):
    """策略错误"""
    pass


class RiskError(QuantException):
    """风险控制错误"""
    pass


class ConfigError(QuantException):
    """配置错误"""
    pass


def retry(max_attempts: int = 3, 
         delay: float = 1.0,
         exceptions: tuple = (Exception,)):
    """
    重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 重试延迟(秒)
        exceptions: 需要重试的异常类型
    
    Returns:
        装饰器
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            import time
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                
                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} 失败 (尝试{attempt}次): {e}")
                        raise
                    
                    logger.warning(f"{func.__name__} 失败 (尝试{attempt}/{max_attempts}): {e}, {delay}秒后重试")
                    time.sleep(delay)
            
            return None
        
        return wrapper
    
    return decorator


def safe_execute(default: Any = None, 
                log_error: bool = True):
    """
    安全执行装饰器
    
    Args:
        default: 发生异常时的默认返回值
        log_error: 是否记录错误
    
    Returns:
        装饰器
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                if log_error:
                    logger.error(f"{func.__name__} 执行失败: {e}\n{traceback.format_exc()}")
                
                return default
        
        return wrapper
    
    return decorator


class ErrorHandler:
    """
    错误处理器
    
    统一处理各种错误
    """
    
    def __init__(self):
        """初始化错误处理器"""
        self.errors = []
        logger.info("错误处理器初始化")
    
    def handle(self, 
               error: Exception, 
               context: dict = None,
               reraise: bool = False) -> bool:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            reraise: 是否重新抛出
        
        Returns:
            是否处理成功
        """
        error_info = {
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }
        
        self.errors.append(error_info)
        
        # 根据错误类型处理
        if isinstance(error, DataError):
            logger.error(f"数据错误: {error}")
            # 可以尝试重新加载数据
        
        elif isinstance(error, StrategyError):
            logger.error(f"策略错误: {error}")
            # 可以使用备用策略
        
        elif isinstance(error, RiskError):
            logger.error(f"风险错误: {error}")
            # 紧急平仓
        
        elif isinstance(error, ConfigError):
            logger.error(f"配置错误: {error}")
            # 使用默认配置
        
        else:
            logger.error(f"未知错误: {error}\n{traceback.format_exc()}")
        
        if reraise:
            raise error
        
        return True
    
    def get_errors(self, limit: int = 100) -> list:
        """获取错误列表"""
        return self.errors[-limit:]
    
    def clear_errors(self):
        """清空错误列表"""
        self.errors = []
        logger.info("错误列表已清空")


# ========== 全局错误处理器 ==========
_global_handler = None


def get_handler() -> ErrorHandler:
    """获取全局错误处理器"""
    global _global_handler
    
    if _global_handler is None:
        _global_handler = ErrorHandler()
    
    return _global_handler


# ========== 导出 __all__ = [
    'QuantException',
    'DataError',
    'StrategyError',
    'RiskError',
    'ConfigError',
    'retry',
    'safe_execute',
    'ErrorHandler',
    'get_handler',
]
