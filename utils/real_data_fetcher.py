#!/usr/bin/env python3
"""
真实数据获取器 - 使用可靠的数据源

优先级：
1. 腾讯财经（已验证可用）
2. 东方财富
3. 新浪财经
"""

import json
import logging
from typing import Any, Dict, Optional

import requests

# 简单的 logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RealDataFetcher:
    """真实数据获取器"""

    @staticmethod
    def get_stock_price(stock_code: str) -> Optional[float]:
        """
        获取股票当前价格

        Args:
            stock_code: 股票代码（如 300750）

        Returns:
            股价（失败返回 None）
        """
        # 1. 尝试腾讯数据源（最可靠）
        price = RealDataFetcher._get_from_tencent(stock_code)
        if price:
            return price

        # 2. 尝试东方财富
        price = RealDataFetcher._get_from_eastmoney(stock_code)
        if price:
            return price

        # 3. 尝试新浪
        price = RealDataFetcher._get_from_sina(stock_code)
        if price:
            return price

        logger.error(f"❌ 所有数据源失败，无法获取 {stock_code} 价格")
        return None

    @staticmethod
    def _get_from_tencent(stock_code: str) -> Optional[float]:
        """腾讯数据源"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://web.sqt.gtimg.cn/q={market}{stock_code}"

            response = requests.get(url, timeout=5)
            text = response.text

            # 解析格式: v_sz300750="51~宁德时代~300750~..."
            if "~" in text:
                parts = text.split("~")
                if len(parts) > 3:
                    price = float(parts[3])
                    logger.info(f"✅ 腾讯数据: {stock_code} = {price}")
                    return price
        except Exception as e:
            logger.warning(f"腾讯数据源失败: {e}")
        return None

    @staticmethod
    def _get_from_eastmoney(stock_code: str) -> Optional[float]:
        """东方财富数据源"""
        try:
            market = "SZ" if stock_code.startswith(("0", "3")) else "SH"
            secid = f"{market}.{stock_code}"

            url = "https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                "secid": secid,
                "fields": "f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f57,f58,f60,f170,f171",
            }

            response = requests.get(url, params=params, timeout=5)
            data = response.json()

            if data and "data" in data and data["data"]:
                price = data["data"]["f43"] / 100  # 价格需要除以100
                logger.info(f"✅ 东方财富数据: {stock_code} = {price}")
                return price
        except Exception as e:
            logger.warning(f"东方财富数据源失败: {e}")
        return None

    @staticmethod
    def _get_from_sina(stock_code: str) -> Optional[float]:
        """新浪数据源"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://hq.sinajs.cn/list={market}{stock_code}"

            headers = {"Referer": "https://finance.sina.com.cn"}
            response = requests.get(url, headers=headers, timeout=5)
            text = response.text

            # 解析格式: var hq_str_sz300750="宁德时代,395.500,..."
            if "=" in text and "," in text:
                data_str = text.split('"')[1]
                parts = data_str.split(",")
                if len(parts) > 3:
                    price = float(parts[3])
                    logger.info(f"✅ 新浪数据: {stock_code} = {price}")
                    return price
        except Exception as e:
            logger.warning(f"新浪数据源失败: {e}")
        return None

    @staticmethod
    def get_stock_data(stock_code: str) -> Dict[str, Any]:
        """
        获取股票完整数据（价格 + 基本信息）

        Args:
            stock_code: 股票代码

        Returns:
            包含 price, name 等字段的字典
        """
        data = {
            "code": stock_code,
            "price": None,
            "name": None,
            "change_pct": None,
        }

        # 获取价格
        price = RealDataFetcher.get_stock_price(stock_code)
        if price:
            data["price"] = price

        # 尝试获取更多信息（从腾讯）
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://web.sqt.gtimg.cn/q={market}{stock_code}"
            response = requests.get(url, timeout=5)
            text = response.text

            if "~" in text:
                parts = text.split("~")
                if len(parts) > 30:
                    data["name"] = parts[1]
                    data["change_pct"] = float(parts[5]) if parts[5] else None
        except Exception as e:
            logger.warning(f"获取股票信息失败: {e}")

        return data


if __name__ == "__main__":
    # 测试
    import sys

    if len(sys.argv) < 2:
        logger.info("用法: python3 real_data_fetcher.py <股票代码> [股票代码2] ...")
        sys.exit(1)

    logger.info("📊 获取真实股价...")
    print("=" * 50)

    for code in sys.argv[1:]:
        data = RealDataFetcher.get_stock_data(code)
        logger.info("\n{data.get('name', code)} ({code})")
        logger.info("  价格: {data.get('price', 'N/A')}")
        logger.info("  涨跌幅: {data.get('change_pct', 'N/A')}%")

    print("=" * 50)
