#!/usr/bin/env python3
"""
多数据源获取器 - AkShare + Tushare + 新浪财经

优先级：
1. AkShare（免费）
2. Tushare（需token）
3. 新浪财经（备用）
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class MultiSourceDataFetcher:
    """多数据源获取器"""

    def __init__(self):
        self.tushare_token = os.getenv("TUSHARE_TOKEN")
        self.tushare_available = False

        # 检查 tushare
        if self.tushare_token:
            try:
                import tushare as ts

                ts.set_token(self.tushare_token)
                self.pro = ts.pro_api()
                self.tushare_available = True
                logger.info("✅ Tushare 已启用")
            except Exception as e:
                logger.warning(f"Tushare 初始化失败: {e}")
                self.tushare_available = False

    def get_stock_hist_data(
        self, stock_code: str, days: int = 250, retry: int = 2
    ) -> Optional[pd.DataFrame]:
        """
        获取股票历史数据（多数据源）

        Args:
            stock_code: 股票代码
            days: 获取天数
            retry: 重试次数

        Returns:
            DataFrame 或 None
        """
        for attempt in range(retry + 1):
            # 1. 尝试 AkShare
            df = self._get_from_akshare(stock_code, days)
            if df is not None and not df.empty:
                logger.info(f"✅ AkShare: 获取到 {len(df)} 天数据")
                return df

            # 2. 尝试 Tushare
            df = self._get_from_tushare(stock_code, days)
            if df is not None and not df.empty:
                logger.info(f"✅ Tushare: 获取到 {len(df)} 天数据")
                return df

            # 3. 尝试新浪财经
            df = self._get_from_sina(stock_code, days)
            if df is not None and not df.empty:
                logger.info(f"✅ 新浪财经: 获取到 {len(df)} 天数据")
                return df

            if attempt < retry:
                logger.warning(f"第 {attempt + 1} 次尝试失败，重试...")
                import time

                time.sleep(2)

        logger.error(f"❌ 所有数据源失败，无法获取 {stock_code} 历史数据")
        return None

    def _get_from_akshare(self, stock_code: str, days: int) -> Optional[pd.DataFrame]:
        """从 AkShare 获取数据"""
        try:
            import akshare as ak

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
                # 标准化列名
                df = df.rename(
                    columns={
                        "日期": "date",
                        "开盘": "open",
                        "收盘": "close",
                        "最高": "high",
                        "最低": "low",
                        "成交量": "volume",
                        "成交额": "amount",
                        "振幅": "amplitude",
                        "涨跌幅": "pct_change",
                        "涨跌额": "change",
                        "换手率": "turnover",
                    }
                )
                return df
        except Exception as e:
            logger.warning(f"AkShare 失败: {e}")
        return None

    def _get_from_tushare(self, stock_code: str, days: int) -> Optional[pd.DataFrame]:
        """从 Tushare 获取数据"""
        if not self.tushare_available:
            return None

        try:
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

            # 转换股票代码格式（300750 -> 300750.SZ）
            if stock_code.startswith("6"):
                ts_code = f"{stock_code}.SH"
            else:
                ts_code = f"{stock_code}.SZ"

            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

            if not df.empty:
                # 标准化列名
                df = df.rename(
                    columns={
                        "trade_date": "date",
                        "vol": "volume",
                        "amount": "amount",
                        "pct_chg": "pct_change",
                    }
                )
                # 按日期升序排列
                df = df.sort_values("date").reset_index(drop=True)
                return df
        except Exception as e:
            logger.warning(f"Tushare 失败: {e}")
        return None

    def _get_from_sina(self, stock_code: str, days: int) -> Optional[pd.DataFrame]:
        """从新浪财经获取数据（备用）"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"

            params = {
                "symbol": f"{market}{stock_code}",
                "scale": "240",  # 日线
                "ma": "no",
                "datalen": str(days),
            }

            response = requests.get(url, params=params, timeout=10)
            data = response.json()

            if data and isinstance(data, list):
                # 转换为 DataFrame
                df = pd.DataFrame(data)
                df = df.rename(
                    columns={
                        "day": "date",
                        "open": "open",
                        "close": "close",
                        "high": "high",
                        "low": "low",
                        "volume": "volume",
                    }
                )

                # 转换数据类型
                for col in ["open", "close", "high", "low", "volume"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

                return df
        except Exception as e:
            logger.warning(f"新浪财经失败: {e}")
        return None

    def get_stock_price(self, stock_code: str) -> Optional[float]:
        """获取实时价格（多数据源）"""
        # 1. 腾讯
        price = self._get_price_from_tencent(stock_code)
        if price:
            return price

        # 2. 东方财富
        price = self._get_price_from_eastmoney(stock_code)
        if price:
            return price

        # 3. 新浪
        price = self._get_price_from_sina(stock_code)
        if price:
            return price

        return None

    def _get_price_from_tencent(self, stock_code: str) -> Optional[float]:
        """从腾讯获取实时价格"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://web.sqt.gtimg.cn/q={market}{stock_code}"

            response = requests.get(url, timeout=5)
            text = response.text

            if "~" in text:
                parts = text.split("~")
                if len(parts) > 3:
                    return float(parts[3])
        except Exception as e:
            logger.warning(f"腾讯价格获取失败: {e}")
        return None

    def _get_price_from_eastmoney(self, stock_code: str) -> Optional[float]:
        """从东方财富获取实时价格"""
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
                return data["data"]["f43"] / 100
        except Exception as e:
            logger.warning(f"东方财富价格获取失败: {e}")
        return None

    def _get_price_from_sina(self, stock_code: str) -> Optional[float]:
        """从新浪获取实时价格"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://hq.sinajs.cn/list={market}{stock_code}"

            headers = {"Referer": "https://finance.sina.com.cn"}
            response = requests.get(url, headers=headers, timeout=5)
            text = response.text

            if "=" in text and "," in text:
                data_str = text.split('"')[1]
                parts = data_str.split(",")
                if len(parts) > 3:
                    return float(parts[3])
        except Exception as e:
            logger.warning(f"新浪价格获取失败: {e}")
        return None


if __name__ == "__main__":
    # 测试
    fetcher = MultiSourceDataFetcher()

    print("测试获取历史数据...")
    df = fetcher.get_stock_hist_data("300750", days=10)
    if df is not None:
        print(f"✅ 成功获取 {len(df)} 天数据")
        print(df.head())
    else:
        print("❌ 获取失败")

    print("\n测试获取实时价格...")
    price = fetcher.get_stock_price("300750")
    if price:
        print(f"✅ 价格: {price}")
    else:
        print("❌ 获取失败")
