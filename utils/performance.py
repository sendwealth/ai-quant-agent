#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控模块
Performance Monitor Module

监控系统性能，记录执行时间和资源使用
"""

import functools
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict

import psutil
from loguru import logger


class PerformanceMonitor:
    """
    性能监控器

    功能:
    - 记录函数执行时间
    - 监控内存使用
    - 生成性能报告
    """

    def __init__(self, log_file: str = "logs/performance.json"):
        """
        初始化性能监控器

        Args:
            log_file: 性能日志文件
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # 性能数据
        self.metrics = {"functions": {}, "system": []}

        logger.info(f"性能监控器初始化: {self.log_file}")

    def timeit(self, func_name: str = None):
        """
        计时装饰器

        Args:
            func_name: 函数名称

        Returns:
            装饰器
        """

        def decorator(func: Callable) -> Callable:
            name = func_name or func.__name__

            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

                try:
                    result = func(*args, **kwargs)

                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024

                    # 记录性能
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory

                    self._record_function(name, execution_time, memory_delta, success=True)

                    logger.debug(f"{name}: {execution_time:.3f}s, 内存: {memory_delta:+.2f}MB")

                    return result

                except Exception as e:
                    end_time = time.time()
                    execution_time = end_time - start_time

                    self._record_function(name, execution_time, 0, success=False, error=str(e))

                    raise

            return wrapper

        return decorator

    def _record_function(
        self,
        name: str,
        execution_time: float,
        memory_delta: float,
        success: bool,
        error: str = None,
    ):
        """记录函数性能"""
        if name not in self.metrics["functions"]:
            self.metrics["functions"][name] = {
                "calls": 0,
                "total_time": 0,
                "avg_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
                "total_memory": 0,
                "errors": 0,
            }

        metrics = self.metrics["functions"][name]
        metrics["calls"] += 1
        metrics["total_time"] += execution_time
        metrics["avg_time"] = metrics["total_time"] / metrics["calls"]
        metrics["min_time"] = min(metrics["min_time"], execution_time)
        metrics["max_time"] = max(metrics["max_time"], execution_time)
        metrics["total_memory"] += memory_delta

        if not success:
            metrics["errors"] += 1

    def record_system(self):
        """记录系统状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        system_metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_used_gb": memory.used / (1024**3),
        }

        self.metrics["system"].append(system_metrics)

        # 只保留最近100条
        if len(self.metrics["system"]) > 100:
            self.metrics["system"] = self.metrics["system"][-100:]

    def get_summary(self) -> dict:
        """获取性能摘要"""
        summary = {"functions": {}, "system": {}}

        # 函数性能
        for name, metrics in self.metrics["functions"].items():
            summary["functions"][name] = {
                "calls": metrics["calls"],
                "avg_time": f"{metrics['avg_time']:.3f}s",
                "total_time": f"{metrics['total_time']:.3f}s",
                "min_time": f"{metrics['min_time']:.3f}s",
                "max_time": f"{metrics['max_time']:.3f}s",
                "errors": metrics["errors"],
            }

        # 系统性能
        if self.metrics["system"]:
            latest = self.metrics["system"][-1]
            summary["system"] = {
                "cpu_percent": latest["cpu_percent"],
                "memory_percent": latest["memory_percent"],
                "memory_available_gb": f"{latest['memory_available_gb']:.2f}GB",
            }

        return summary

    def save(self):
        """保存性能数据"""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, ensure_ascii=False, indent=2)

        logger.info(f"性能数据已保存: {self.log_file}")

    def load(self):
        """加载性能数据"""
        if not self.log_file.exists():
            return

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                self.metrics = json.load(f)

            logger.info(f"性能数据已加载: {self.log_file}")
        except Exception as e:
            logger.warning(f"加载性能数据失败: {e}")


# ========== 全局监控实例 ==========
_global_monitor = None


def get_monitor() -> PerformanceMonitor:
    """获取全局监控实例"""
    global _global_monitor

    if _global_monitor is None:
        log_file = Path(__file__).parent.parent / "logs" / "performance.json"
        _global_monitor = PerformanceMonitor(str(log_file))

    return _global_monitor


# ========== 便捷装饰器 ==========
def monitor(func: Callable) -> Callable:
    """性能监控装饰器"""
    monitor = get_monitor()
    return monitor.timeit()(func)


# ========== 导出 ==========
__all__ = [
    "PerformanceMonitor",
    "get_monitor",
    "monitor",
]
