#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略基类
Base Strategy Class

所有策略必须继承此类，统一接口
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indicators import atr, ema, macd, rsi, sma


class BaseStrategy(ABC):
    """
    策略基类

    所有交易策略必须继承此类并实现以下方法:
    - generate_signals(): 生成交易信号
    - calculate_position_size(): 计算仓位大小
    """

    def __init__(self, config: dict):
        """
        初始化策略

        Args:
            config: 策略配置字典
        """
        self.config = config
        self.name = config.get("name", "BaseStrategy")
        self.params = config.get("params", {})

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号 (必须实现)

        Args:
            data: 股票数据 (OHLCV)

        Returns:
            信号序列: 1=买入, -1=卖出, 0=持有
        """
        pass

    @abstractmethod
    def calculate_position_size(self, capital: float, price: float, **kwargs) -> int:
        """
        计算仓位大小 (必须实现)

        Args:
            capital: 可用资金
            price: 当前价格
            **kwargs: 其他参数

        Returns:
            买入股数
        """
        pass

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        验证数据完整性

        Args:
            data: 股票数据

        Returns:
            数据是否有效
        """
        required_cols = ["open", "high", "low", "close", "volume"]

        # 检查列是否存在
        if not all(col in data.columns for col in required_cols):
            return False

        # 检查数据是否为空
        if data.empty:
            return False

        # 检查是否有NaN
        if data[required_cols].isnull().any().any():
            return False

        return True

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算常用技术指标

        Args:
            data: 股票数据

        Returns:
            添加了指标的数据
        """
        df = data.copy()

        # MA
        if "ma_fast" in self.params and "ma_slow" in self.params:
            df["ma_fast"] = sma(df["close"], self.params["ma_fast"])
            df["ma_slow"] = sma(df["close"], self.params["ma_slow"])

        # ATR
        df["atr"] = atr(df["high"], df["low"], df["close"], period=14)

        # RSI
        df["rsi"] = rsi(df["close"], period=14)

        # MACD
        macd_line, signal_line, histogram = macd(df["close"])
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        return df

    def check_risk(self, position: dict, current_price: float) -> Tuple[bool, str]:
        """
        检查风险控制

        Args:
            position: 持仓信息
            current_price: 当前价格

        Returns:
            (是否需要平仓, 原因)
        """
        if not position:
            return False, ""

        entry_price = position.get("entry_price", 0)
        if entry_price <= 0:
            return False, ""

        # 计算盈亏比例
        pnl_pct = (current_price - entry_price) / entry_price

        # 止损
        stop_loss = self.config.get("risk", {}).get("stop_loss", -0.05)
        if pnl_pct <= stop_loss:
            return True, f"触发止损 (亏损{pnl_pct*100:.2f}%)"

        # 止盈1
        take_profit_1 = self.config.get("risk", {}).get("take_profit_1", 0.10)
        if pnl_pct >= take_profit_1:
            return True, f"触发止盈1 (盈利{pnl_pct*100:.2f}%)"

        # 止盈2
        take_profit_2 = self.config.get("risk", {}).get("take_profit_2", 0.20)
        if pnl_pct >= take_profit_2:
            return True, f"触发止盈2 (盈利{pnl_pct*100:.2f}%)"

        return False, ""

    def save_signals(self, signals: pd.Series, filepath: str):
        """
        保存信号到文件

        Args:
            signals: 信号序列
            filepath: 文件路径
        """
        signals_df = pd.DataFrame({"signal": signals, "timestamp": datetime.now().isoformat()})

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        signals_df.to_csv(path, index=True)

    def load_signals(self, filepath: str) -> Optional[pd.Series]:
        """
        从文件加载信号

        Args:
            filepath: 文件路径

        Returns:
            信号序列
        """
        path = Path(filepath)
        if not path.exists():
            return None

        df = pd.read_csv(path, index_col=0)
        return df["signal"]

    def get_strategy_info(self) -> dict:
        """
        获取策略信息

        Returns:
            策略信息字典
        """
        return {
            "name": self.name,
            "params": self.params,
            "config": self.config,
        }


class MAStrategy(BaseStrategy):
    """
    MA均线策略 - V4版本

    基于MA金叉/死叉 + MACD确认 + RSI过滤
    """

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """生成交易信号"""
        if not self.validate_data(data):
            raise ValueError("数据不完整或无效")

        # 计算指标
        df = self.calculate_indicators(data)

        # 初始化信号
        signals = pd.Series(0, index=df.index)

        # 买入条件: MA金叉 + MACD正 + RSI适中
        buy_condition = (
            (df["ma_fast"] > df["ma_slow"])
            & (df["macd"] > df["macd_signal"])
            & (df["rsi"] > 30)
            & (df["rsi"] < 70)
        )
        signals[buy_condition] = 1

        # 卖出条件: MA死叉 或 RSI超买
        sell_condition = (df["ma_fast"] < df["ma_slow"]) | (df["rsi"] > 80)
        signals[sell_condition] = -1

        return signals

    def calculate_position_size(self, capital: float, price: float, **kwargs) -> int:
        """根据权重计算仓位"""
        weight = self.config.get("weight", 0.25)
        position_value = capital * weight
        return int(position_value / price)


class StrategyFactory:
    """策略工厂 - 根据配置创建策略实例"""

    @staticmethod
    def create(config: dict) -> BaseStrategy:
        """
        创建策略实例

        Args:
            config: 策略配置

        Returns:
            策略实例
        """
        strategy_type = config.get("type", "ma")

        if strategy_type == "ma":
            return MAStrategy(config)
        else:
            raise ValueError(f"不支持的策略类型: {strategy_type}")


# ========== 导出 ==========
__all__ = [
    "BaseStrategy",
    "MAStrategy",
    "StrategyFactory",
]
