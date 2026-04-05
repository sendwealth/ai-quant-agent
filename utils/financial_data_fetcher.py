#!/usr/bin/env python3
"""
财务数据获取器 - 优先使用Tushare

优先级：
1. Tushare（更准确、更稳定）
2. AkShare（免费）
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FinancialDataFetcher:
    """财务数据获取器"""

    def __init__(self):
        self.tushare_token = os.getenv("TUSHARE_TOKEN")
        self.tushare_available = False
        self.akshare_available = False

        # 初始化 Tushare
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

        # 初始化 AkShare
        try:
            import akshare as ak

            self.akshare_available = True
            logger.info("✅ AkShare 可用")
        except Exception as e:
            logger.warning(f"AkShare 不可用: {e}")
            self.akshare_available = False

    def get_financial_indicators(self, stock_code: str) -> Dict[str, Any]:
        """
        获取财务指标（多数据源）

        Returns:
            {
                'pe_ratio': 市盈率,
                'pb_ratio': 市净率,
                'de_ratio': 资产负债率,
                'roe': 净资产收益率,
                'gross_margin': 销售毛利率,
                'net_margin': 销售净利率,
                'current_ratio': 流动比率,
                'quick_ratio': 速动比率,
            }
        """
        # 1. 尝试 Tushare
        data = self._get_from_tushare(stock_code)
        if data and self._validate_data(data):
            logger.info(f"✅ Tushare: {stock_code} 财务数据获取成功")
            return data

        # 2. 尝试 AkShare
        data = self._get_from_akshare(stock_code)
        if data and self._validate_data(data):
            logger.info(f"✅ AkShare: {stock_code} 财务数据获取成功")
            return data

        # 3. 使用保守估计
        logger.warning(f"⚠️  所有数据源失败，{stock_code} 使用保守估计")
        return self._get_conservative_estimate()

    def _get_from_tushare(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """从 Tushare 获取财务数据"""
        if not self.tushare_available:
            return None

        try:
            # 转换股票代码格式
            ts_code = self._convert_stock_code(stock_code)

            # 获取每日基本面指标
            df = self.pro.daily_basic(ts_code=ts_code, fields="pe,pb,ps,dv_ratio,total_mv,circ_mv")

            if df.empty:
                return None

            latest = df.iloc[0]

            # 获取财务指标
            df_fin = self.pro.fina_indicator(ts_code=ts_code, fields="roe,roa,grossprofit_margin,netprofit_margin,debt_to_assets,current_ratio,quick_ratio")

            if df_fin.empty:
                # 只返回基础指标
                return {
                    "pe_ratio": float(latest.get("pe", 25)) if pd.notna(latest.get("pe")) else 25.0,
                    "pb_ratio": float(latest.get("pb", 3)) if pd.notna(latest.get("pb")) else 3.0,
                    "de_ratio": 0.5,
                    "roe": 0.10,
                    "gross_margin": 0.30,
                    "net_margin": 0.10,
                    "current_ratio": 1.5,
                    "quick_ratio": 1.0,
                }

            latest_fin = df_fin.iloc[0]

            # 安全提取数据
            def safe_float(value, default=0.0):
                try:
                    result = float(value) if pd.notna(value) else default
                    return result
                except (ValueError, TypeError):
                    return default

            data = {
                "pe_ratio": safe_float(latest.get("pe"), 25.0),
                "pb_ratio": safe_float(latest.get("pb"), 3.0),
                "de_ratio": safe_float(latest_fin.get("debt_to_assets"), 50.0) / 100,
                "roe": safe_float(latest_fin.get("roe"), 10.0) / 100,
                "gross_margin": safe_float(latest_fin.get("grossprofit_margin"), 30.0) / 100,
                "net_margin": safe_float(latest_fin.get("netprofit_margin"), 10.0) / 100,
                "current_ratio": safe_float(latest_fin.get("current_ratio"), 1.5),
                "quick_ratio": safe_float(latest_fin.get("quick_ratio"), 1.0),
            }

            return data

        except Exception as e:
            logger.warning(f"Tushare 获取失败: {e}")
            return None

    def _get_from_akshare(self, stock_code: str) -> Optional[Dict[str, Any]]:
        """从 AkShare 获取财务数据"""
        if not self.akshare_available:
            return None

        try:
            import akshare as ak

            df = ak.stock_financial_analysis_indicator(symbol=stock_code)

            if df.empty:
                return None

            latest = df.iloc[0]

            # 安全提取数据
            def safe_float(value, default=0.0):
                try:
                    result = float(value) if value else default
                    return result
                except (ValueError, TypeError):
                    return default

            data = {
                "pe_ratio": safe_float(latest.get("市盈率"), 25.0),
                "pb_ratio": safe_float(latest.get("市净率"), 3.0),
                "de_ratio": safe_float(latest.get("资产负债率"), 50.0) / 100,
                "roe": safe_float(latest.get("净资产收益率"), 10.0) / 100,
                "gross_margin": safe_float(latest.get("销售毛利率"), 30.0) / 100,
                "net_margin": safe_float(latest.get("销售净利率"), 10.0) / 100,
                "current_ratio": safe_float(latest.get("流动比率"), 1.5),
                "quick_ratio": safe_float(latest.get("速动比率"), 1.0),
            }

            return data

        except Exception as e:
            logger.warning(f"AkShare 获取失败: {e}")
            return None

    def _validate_data(self, data: Dict[str, Any]) -> bool:
        """验证数据合理性"""
        if not data:
            return False

        # 检查关键字段
        if data.get("pe_ratio", 0) <= 0:
            logger.warning("P/E 比率异常")
            return False

        if data.get("pb_ratio", 0) <= 0:
            logger.warning("P/B 比率异常")
            return False

        if not (0 <= data.get("de_ratio", 0) <= 1):
            logger.warning("资产负债率异常")
            return False

        return True

    def _get_conservative_estimate(self) -> Dict[str, Any]:
        """获取保守估计"""
        return {
            "pe_ratio": 25.0,
            "pb_ratio": 3.0,
            "de_ratio": 0.50,
            "roe": 0.10,
            "gross_margin": 0.30,
            "net_margin": 0.10,
            "current_ratio": 1.5,
            "quick_ratio": 1.0,
        }

    def _convert_stock_code(self, stock_code: str) -> str:
        """转换股票代码格式（300750 -> 300750.SZ）"""
        if stock_code.startswith("6"):
            return f"{stock_code}.SH"
        else:
            return f"{stock_code}.SZ"


if __name__ == "__main__":
    # 测试
    fetcher = FinancialDataFetcher()

    print("\n测试获取财务数据...")
    for stock in ["300750", "002475", "601318", "600276"]:
        print(f"\n{stock}:")
        data = fetcher.get_financial_indicators(stock)
        print(f"  P/E: {data['pe_ratio']:.1f}")
        print(f"  P/B: {data['pb_ratio']:.1f}")
        print(f"  资产负债率: {data['de_ratio']*100:.1f}%")
        print(f"  ROE: {data['roe']*100:.1f}%")
        print(f"  毛利率: {data['gross_margin']*100:.1f}%")
