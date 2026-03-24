#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器
Configuration Loader

从YAML文件加载策略配置
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger


class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_path: str = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "strategy_v5.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

        logger.info(f"配置加载完成: {self.config_path}")

    def _load_config(self) -> dict:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, *keys, default=None) -> Any:
        """
        获取配置值

        Args:
            *keys: 配置键路径
            default: 默认值

        Returns:
            配置值

        Example:
            loader.get('strategy', 'name')  # 获取strategy.name
        """
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_strategy_config(self) -> dict:
        """获取策略配置"""
        return self.config.get("strategy", {})

    def get_stock_config(self, code: str) -> Optional[dict]:
        """
        获取单只股票配置

        Args:
            code: 股票代码

        Returns:
            股票配置字典
        """
        stocks = self.config.get("stocks", [])

        for stock in stocks:
            if stock["code"] == code:
                return stock

        return None

    def get_all_stocks(self, enabled_only: bool = True) -> List[dict]:
        """
        获取所有股票配置

        Args:
            enabled_only: 是否只返回启用的股票

        Returns:
            股票配置列表
        """
        stocks = self.config.get("stocks", [])

        if enabled_only:
            return [s for s in stocks if s.get("enabled", True)]

        return stocks

    def get_risk_params(self) -> dict:
        """获取风险参数"""
        return self.config.get("risk", {})

    def get_capital_config(self) -> dict:
        """获取资金配置"""
        return self.config.get("capital", {})

    def get_monitor_config(self) -> dict:
        """获取监控配置"""
        return self.config.get("monitor", {})

    def get_backtest_config(self) -> dict:
        """获取回测配置"""
        return self.config.get("backtest", {})

    def update_stock_weight(self, code: str, weight: float):
        """
        更新股票权重

        Args:
            code: 股票代码
            weight: 新权重
        """
        stocks = self.config.get("stocks", [])

        for stock in stocks:
            if stock["code"] == code:
                stock["weight"] = weight
                logger.info(f"更新权重: {code} -> {weight}")
                break

        # 保存配置
        self._save_config()

    def enable_stock(self, code: str, enabled: bool = True):
        """
        启用/禁用股票

        Args:
            code: 股票代码
            enabled: 是否启用
        """
        stocks = self.config.get("stocks", [])

        for stock in stocks:
            if stock["code"] == code:
                stock["enabled"] = enabled
                status = "启用" if enabled else "禁用"
                logger.info(f"{status}股票: {code}")
                break

        # 保存配置
        self._save_config()

    def _save_config(self):
        """保存配置到文件"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)

        logger.info(f"配置已保存: {self.config_path}")

    def validate(self) -> bool:
        """
        验证配置有效性

        Returns:
            配置是否有效
        """
        try:
            # 检查必要字段
            if not self.config.get("strategy"):
                logger.error("缺少strategy配置")
                return False

            if not self.config.get("stocks"):
                logger.error("缺少stocks配置")
                return False

            # 检查权重总和
            stocks = self.get_all_stocks(enabled_only=True)
            total_weight = sum(s.get("weight", 0) for s in stocks)

            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"权重总和不为1: {total_weight}")

            # 检查风险参数
            risk = self.get_risk_params()
            if risk.get("stop_loss", 0) >= 0:
                logger.error("止损必须为负数")
                return False

            logger.info("配置验证通过")
            return True

        except Exception as e:
            logger.error(f"配置验证失败: {e}")
            return False

    def reload(self):
        """重新加载配置"""
        self.config = self._load_config()
        logger.info("配置已重新加载")


# ========== 全局配置实例 ==========
_global_loader = None


def get_config(reload: bool = False) -> ConfigLoader:
    """
    获取全局配置实例

    Args:
        reload: 是否重新加载

    Returns:
        ConfigLoader实例
    """
    global _global_loader

    if _global_loader is None or reload:
        _global_loader = ConfigLoader()

    return _global_loader


# ========== 导出 ==========
__all__ = [
    "ConfigLoader",
    "get_config",
]
