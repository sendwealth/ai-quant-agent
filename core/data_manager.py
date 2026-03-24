#!/usr/bin/env python3
"""
数据管理器 - 真实数据源和缓存层

功能：
1. 股票数据获取（akshare）
2. Redis 缓存层
3. 数据健康检查
4. 真实数据强制使用（无降级）
"""

import json
import sys

from utils.logger import get_logger

logger = get_logger(__name__)

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from config.settings import Settings

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入 akshare
try:
    import akshare as ak

    AKSHARE_AVAILABLE = True
    logger.info("akshare 已加载")
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("akshare 未安装，数据获取功能受限")


class DataFetchError(Exception):
    """数据获取错误"""

    pass


class DataManager:
    """数据管理器 - 统一数据获取接口"""

    def __init__(self, use_cache: bool = True):
        """
        初始化数据管理器

        Args:
            use_cache: 是否使用缓存
        """
        self.akshare_available = AKSHARE_AVAILABLE
        self.use_cache = use_cache
        self.cache = None

        # 尝试初始化缓存
        if use_cache:
            try:
                from core.cache import RedisCache

                self.cache = RedisCache()
                logger.info("Redis 缓存已启用")
            except Exception as e:
                logger.warning(f"Redis 缓存初始化失败: {e}")
                self.cache = None

    def get_basic_info(self, stock_code: str) -> Dict[str, Any]:
        """
        获取股票基本信息

        Args:
            stock_code: 股票代码

        Returns:
            基本信息 dict

        Raises:
            DataFetchError: 获取失败时抛出
        """
        data_type = "basic_info"

        # 1. 尝试从缓存获取
        if self.cache:
            cached = self.cache.get(stock_code, data_type, max_age_minutes=60)
            if cached:
                logger.info(f"从缓存获取 {stock_code} 基本信息")
                return cached

        # 2. 从 akshare 获取
        if self.akshare_available:
            try:
                df = ak.stock_individual_info_em(symbol=stock_code)

                if not df.empty:
                    # 转换为 dict
                    info_dict = dict(zip(df["item"], df["value"]))

                    data = {
                        "stock_code": stock_code,
                        "stock_name": info_dict.get("股票简称", ""),
                        "industry": info_dict.get("行业", ""),
                        "list_date": info_dict.get("上市时间", ""),
                        "total_market_value": info_dict.get("总市值", ""),
                        "circulation_market_value": info_dict.get("流通市值", ""),
                    }

                    # 缓存数据
                    if self.cache:
                        self.cache.set(stock_code, data_type, data)

                    return data
            except Exception as e:
                logger.error(f"获取真实数据失败: {e}")
                raise DataFetchError(f"无法获取股票 {stock_code} 的真实数据: {e}")

        # 3. akshare 不可用
        raise DataFetchError("akshare 未安装，无法获取真实数据")

    def get_financial_data(self, stock_code: str) -> Dict[str, Any]:
        """获取财务数据"""
        data_type = "financial"

        # 1. 尝试从缓存获取
        if self.cache:
            cached = self.cache.get(stock_code, data_type, max_age_minutes=30)
            if cached:
                return cached

        # 2. 从 akshare 获取
        if self.akshare_available:
            try:
                # 获取财务指标
                df = ak.stock_financial_analysis_indicator(symbol=stock_code)

                if not df.empty:
                    latest = df.iloc[0]

                    # 安全的数据提取和验证
                    def safe_float(value, default=0.0, min_val=None, max_val=None):
                        """安全转换为浮点数"""
                        try:
                            result = float(value) if value else default
                            if min_val is not None:
                                result = max(result, min_val)
                            if max_val is not None:
                                result = min(result, max_val)
                            return result
                        except (ValueError, TypeError):
                            return default

                    data = {
                        "stock_code": stock_code,
                        "roe": safe_float(latest.get("净资产收益率"), 0.0, 0.0, 100.0) / 100,
                        "gross_margin": safe_float(latest.get("销售毛利率"), 0.0, 0.0, 100.0) / 100,
                        "net_margin": safe_float(latest.get("销售净利率"), 0.0, 0.0, 100.0) / 100,
                        "current_ratio": safe_float(latest.get("流动比率"), 1.0, 0.0, 100.0),
                        "debt_ratio": safe_float(latest.get("资产负债率"), 0.0, 0.0, 100.0) / 100,
                        "timestamp": datetime.now().isoformat(),
                    }

                    # 缓存数据
                    if self.cache:
                        self.cache.set(stock_code, data_type, data)

                    return data
            except Exception as e:
                logger.error(f"获取财务数据失败: {e}")
                raise DataFetchError(f"无法获取股票 {stock_code} 的财务数据: {e}")

        raise DataFetchError("akshare 未安装，无法获取真实数据")

    def get_technical_data(self, stock_code: str, days: int = 250) -> Dict[str, Any]:
        """获取技术数据"""
        data_type = f"technical_{days}"

        # 1. 尝试从缓存获取
        if self.cache:
            cached = self.cache.get(stock_code, data_type, max_age_minutes=5)
            if cached:
                return cached

        # 2. 从 akshare 获取
        if self.akshare_available:
            try:
                # 获取历史数据
                end_date = datetime.now().strftime("%Y%m%d")
                start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

                df = ak.stock_zh_a_hist(
                    symbol=stock_code,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )

                if not df.empty:
                    # 计算技术指标
                    close = df["收盘"]
                    high = df["最高"]
                    low = df["最低"]
                    volume = df["成交量"]

                    # MA
                    ma_5 = close.rolling(5).mean().iloc[-1]
                    ma_10 = close.rolling(10).mean().iloc[-1]
                    ma_20 = close.rolling(20).mean().iloc[-1]
                    ma_50 = close.rolling(50).mean().iloc[-1]

                    # RSI
                    delta = close.diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]

                    # MACD
                    exp1 = close.ewm(span=12, adjust=False).mean()
                    exp2 = close.ewm(span=26, adjust=False).mean()
                    macd = exp1 - exp2
                    signal = macd.ewm(span=9, adjust=False).mean()
                    histogram = macd - signal

                    data = {
                        "stock_code": stock_code,
                        "current_price": float(close.iloc[-1]),
                        "ma_5": float(ma_5),
                        "ma_10": float(ma_10),
                        "ma_20": float(ma_20),
                        "ma_50": float(ma_50),
                        "rsi": float(rsi),
                        "macd": float(macd.iloc[-1]),
                        "macd_signal": float(signal.iloc[-1]),
                        "macd_histogram": float(histogram.iloc[-1]),
                        "volume": float(volume.iloc[-1]),
                        "timestamp": datetime.now().isoformat(),
                    }

                    # 缓存数据
                    if self.cache:
                        self.cache.set(stock_code, data_type, data)

                    return data
            except Exception as e:
                logger.error(f"获取技术数据失败: {e}")
                raise DataFetchError(f"无法获取股票 {stock_code} 的技术数据: {e}")

        raise DataFetchError("akshare 未安装，无法获取真实数据")


if __name__ == "__main__":
    # 测试
    dm = DataManager(use_cache=False)

    try:
        # 测试获取基本信息
        info = dm.get_basic_info("300750")
        logger.info("基本信息: {info}")
    except DataFetchError as e:
        logger.info("错误: {e}")
