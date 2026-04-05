#!/usr/bin/env python3
"""
财务数据获取器 v2 - 使用实时行情数据

数据源：
1. 新浪财经实时行情（免费、无需token）
2. 腾讯财经（备用）
"""

import logging
import re
from typing import Any, Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class FinancialDataFetcherV2:
    """财务数据获取器 v2 - 基于实时行情"""

    @staticmethod
    def get_real_time_indicators(stock_code: str) -> Dict[str, Any]:
        """
        获取实时行情中的财务指标

        Returns:
            {
                'pe_ratio': 市盈率（动态）,
                'pb_ratio': 市净率,
                'total_value': 总市值,
                'circulation_value': 流通市值,
                'de_ratio': 资产负债率（估计）,
                'roe': ROE（估计）,
                'gross_margin': 毛利率（估计）,
            }
        """
        # 尝试腾讯财经
        data = FinancialDataFetcherV2._get_from_tencent(stock_code)
        if data:
            return data

        # 尝试新浪财经
        data = FinancialDataFetcherV2._get_from_sina(stock_code)
        if data:
            return data

        # 使用保守估计
        logger.warning(f"⚠️  所有数据源失败，{stock_code} 使用保守估计")
        return FinancialDataFetcherV2._get_conservative_estimate()

    @staticmethod
    def _get_from_tencent(stock_code: str) -> Optional[Dict[str, Any]]:
        """从腾讯财经获取实时数据"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://web.sqt.gtimg.cn/q={market}{stock_code}"

            response = requests.get(url, timeout=5)
            text = response.text

            # 解析格式: v_sz300750="51~宁德时代~300750~386.46~..."
            if "~" in text:
                parts = text.split("~")

                if len(parts) > 40:
                    # 实时行情数据
                    price = float(parts[3])
                    pe_ratio = float(parts[39]) if parts[39] else 25.0  # 市盈率
                    pb_ratio = float(parts[46]) if parts[46] else 3.0  # 市净率

                    # 总市值（亿元）
                    total_value_str = parts[45] if len(parts) > 45 else "0"
                    total_value = FinancialDataFetcherV2._parse_market_cap(total_value_str)

                    logger.info(
                        f"✅ 腾讯数据: {stock_code} P/E={pe_ratio:.1f}, P/B={pb_ratio:.1f}, 市值={total_value:.0f}亿"
                    )

                    return {
                        "pe_ratio": pe_ratio if pe_ratio > 0 else 25.0,
                        "pb_ratio": pb_ratio if pb_ratio > 0 else 3.0,
                        "total_value": total_value,
                        "circulation_value": total_value * 0.7,  # 估计
                        "de_ratio": FinancialDataFetcherV2._estimate_de_ratio(pe_ratio, pb_ratio),
                        "roe": FinancialDataFetcherV2._estimate_roe(pb_ratio),
                        "gross_margin": 0.30,  # 保守估计
                        "net_margin": 0.10,
                        "current_ratio": 1.5,
                        "quick_ratio": 1.0,
                    }
        except Exception as e:
            logger.warning(f"腾讯数据源失败: {e}")
        return None

    @staticmethod
    def _get_from_sina(stock_code: str) -> Optional[Dict[str, Any]]:
        """从新浪财经获取实时数据"""
        try:
            market = "sz" if stock_code.startswith(("0", "3")) else "sh"
            url = f"https://hq.sinajs.cn/list={market}{stock_code}"

            headers = {"Referer": "https://finance.sina.com.cn"}
            response = requests.get(url, headers=headers, timeout=5)
            text = response.text

            # 解析格式: var hq_str_sz300750="宁德时代,386.460,..."
            if "=" in text and "," in text:
                data_str = text.split('"')[1]
                parts = data_str.split(",")

                if len(parts) > 30:
                    price = float(parts[3])

                    # 新浪财经的市盈率和市净率在第35和36位
                    pe_ratio = float(parts[35]) if len(parts) > 35 and parts[35] else 25.0
                    pb_ratio = float(parts[36]) if len(parts) > 36 and parts[36] else 3.0

                    logger.info(f"✅ 新浪数据: {stock_code} P/E={pe_ratio:.1f}, P/B={pb_ratio:.1f}")

                    return {
                        "pe_ratio": pe_ratio if pe_ratio > 0 else 25.0,
                        "pb_ratio": pb_ratio if pb_ratio > 0 else 3.0,
                        "total_value": 0,  # 新浪不提供
                        "circulation_value": 0,
                        "de_ratio": FinancialDataFetcherV2._estimate_de_ratio(pe_ratio, pb_ratio),
                        "roe": FinancialDataFetcherV2._estimate_roe(pb_ratio),
                        "gross_margin": 0.30,
                        "net_margin": 0.10,
                        "current_ratio": 1.5,
                        "quick_ratio": 1.0,
                    }
        except Exception as e:
            logger.warning(f"新浪数据源失败: {e}")
        return None

    @staticmethod
    def _parse_market_cap(value_str: str) -> float:
        """解析市值字符串"""
        try:
            # 去除单位
            value_str = value_str.replace("亿", "").strip()
            return float(value_str) if value_str else 0
        except:
            return 0

    @staticmethod
    def _estimate_de_ratio(pe_ratio: float, pb_ratio: float) -> float:
        """根据P/E和P/B估计资产负债率"""
        # 简化估计：P/B较低说明资产较多，负债率可能较低
        if pb_ratio < 2:
            return 0.30  # 低负债
        elif pb_ratio < 4:
            return 0.50  # 中等负债
        else:
            return 0.65  # 较高负债

    @staticmethod
    def _estimate_roe(pb_ratio: float) -> float:
        """根据P/B估计ROE"""
        # 简化估计：P/B较高说明市场认可度好，ROE可能较高
        if pb_ratio < 2:
            return 0.08  # 较低ROE
        elif pb_ratio < 4:
            return 0.12  # 中等ROE
        else:
            return 0.18  # 较高ROE

    @staticmethod
    def _get_conservative_estimate() -> Dict[str, Any]:
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


if __name__ == "__main__":
    # 测试
    print("\n测试获取实时财务数据...")
    print("=" * 60)

    for stock in ["300750", "002475", "601318", "600276"]:
        print(f"\n{stock}:")
        data = FinancialDataFetcherV2.get_real_time_indicators(stock)
        print(f"  P/E: {data['pe_ratio']:.1f}")
        print(f"  P/B: {data['pb_ratio']:.1f}")
        print(f"  资产负债率: {data['de_ratio']*100:.1f}%")
        print(f"  ROE: {data['roe']*100:.1f}%")
        print(f"  毛利率: {data['gross_margin']*100:.1f}%")
