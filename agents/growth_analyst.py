#!/usr/bin/env python3
"""
Growth Analyst - 成长投资分析（真实数据版）

策略：成长投资理念
维度：营收增长、利润增长、市场份额、创新能力
"""

import logging
import sys

# 设置 logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import akshare as ak
    import pandas as pd
    import numpy as np
except ImportError as e:
    logger.error(f"缺少依赖: {e}")
    sys.exit(1)

# 导入真实数据获取器
from utils.real_data_fetcher import RealDataFetcher


class GrowthAnalyst:
    """成长投资分析师"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.analyst_name = "growth"

    def get_growth_data(self) -> Dict[str, Any]:
        """获取成长数据（真实数据）"""
        data = {}

        # 1. 获取当前价格
        price = RealDataFetcher.get_stock_price(self.stock_code)
        if not price:
            raise RuntimeError(f"❌ 无法获取 {self.stock_code} 的实时价格")
        data["current_price"] = price

        # 2. 获取财务指标（包含营收、利润增长）
        try:
            df = ak.stock_financial_analysis_indicator(symbol=self.stock_code)

            if not df.empty:
                # 取最近2年数据计算增长率
                if len(df) >= 2:
                    latest = df.iloc[0]
                    previous = df.iloc[1]

                    # 安全提取数据
                    def safe_float(value, default=0.0, min_val=None, max_val=None):
                        try:
                            result = float(value) if value else default
                            if min_val is not None:
                                result = max(result, min_val)
                            if max_val is not None:
                                result = min(result, max_val)
                            return result
                        except (ValueError, TypeError):
                            return default

                    # 营收增长率
                    latest_revenue = safe_float(latest.get("营业收入"), 0)
                    previous_revenue = safe_float(previous.get("营业收入"), 0)
                    if previous_revenue > 0:
                        revenue_growth = (latest_revenue - previous_revenue) / previous_revenue
                    else:
                        revenue_growth = 0.15  # 默认15%

                    # 利润增长率
                    latest_profit = safe_float(latest.get("净利润"), 0)
                    previous_profit = safe_float(previous.get("净利润"), 0)
                    if previous_profit > 0:
                        profit_growth = (latest_profit - previous_profit) / previous_profit
                    else:
                        profit_growth = 0.15  # 默认15%

                    data.update(
                        {
                            "revenue_growth": revenue_growth,
                            "profit_growth": profit_growth,
                            "roe": safe_float(latest.get("净资产收益率"), 10.0, 0.0, 100.0) / 100,
                            "gross_margin": safe_float(latest.get("销售毛利率"), 30.0, 0.0, 100.0) / 100,
                            "revenue": latest_revenue,
                            "profit": latest_profit,
                        }
                    )
                    logger.info(
                        f"✅ 获取到成长数据: 营收增长{revenue_growth*100:.1f}%, 利润增长{profit_growth*100:.1f}%"
                    )
                else:
                    # 数据不足，使用保守估计
                    logger.warning("历史数据不足，使用保守估计")
                    data.update(
                        {
                            "revenue_growth": 0.10,
                            "profit_growth": 0.10,
                            "roe": 0.10,
                            "gross_margin": 0.30,
                            "revenue": 0,
                            "profit": 0,
                        }
                    )
            else:
                raise ValueError("无财务数据")

        except Exception as e:
            logger.warning(f"获取成长数据失败: {e}，使用保守估计")
            # 使用保守的默认值
            data.update(
                {
                    "revenue_growth": 0.10,
                    "profit_growth": 0.10,
                    "roe": 0.10,
                    "gross_margin": 0.30,
                    "revenue": 0,
                    "profit": 0,
                }
            )

        # 3. 估算TAM和市场份额（简化版）
        # 由于没有行业数据，使用保守估计
        data["tam"] = "未知"  # Total Addressable Market
        data["market_share"] = "未知"
        data["innovation_score"] = self._estimate_innovation(data)

        return data

    def _estimate_innovation(self, data: Dict[str, Any]) -> int:
        """估算创新能力（1-10分）"""
        score = 5  # 基础分

        # 高毛利率 = 产品创新能力
        if data.get("gross_margin", 0) > 0.4:
            score += 2
        elif data.get("gross_margin", 0) > 0.3:
            score += 1

        # 高营收增长 = 市场扩张能力
        if data.get("revenue_growth", 0) > 0.3:
            score += 2
        elif data.get("revenue_growth", 0) > 0.2:
            score += 1

        # 高ROE = 资本效率
        if data.get("roe", 0) > 0.15:
            score += 1

        return min(10, max(1, score))

    def calculate_sustainable_growth_rate(self, data: Dict[str, Any]) -> float:
        """计算可持续增长率（SGR）"""
        # SGR = ROE × (1 - Dividend Payout Ratio)
        # 简化版：假设不分红
        roe = data.get("roe", 0.10)
        sgr = roe  # 简化：假设利润全部再投资

        logger.info(f"可持续增长率: {sgr*100:.1f}%")
        return sgr

    def evaluate_growth_quality(
        self, revenue_growth: float, profit_growth: float, roe: float, innovation_score: int
    ) -> tuple:
        """评估成长质量"""
        # 评级
        if revenue_growth > 0.3 and profit_growth > 0.25 and roe > 0.15:
            return "优质成长", "高"
        elif revenue_growth > 0.2 and profit_growth > 0.15:
            return "良好成长", "中高"
        elif revenue_growth > 0.1 and profit_growth > 0.10:
            return "稳定成长", "中"
        elif revenue_growth > 0.05:
            return "缓慢成长", "中低"
        else:
            return "缺乏成长", "低"

    def generate_signal(
        self, revenue_growth: float, profit_growth: float, roe: float, innovation_score: int
    ) -> tuple:
        """生成交易信号"""
        # 评估成长质量
        quality, quality_level = self.evaluate_growth_quality(
            revenue_growth, profit_growth, roe, innovation_score
        )

        # 决策规则
        if revenue_growth > 0.30 and innovation_score >= 8:
            return "BUY", 0.90, f"{quality}：营收增长{revenue_growth*100:.0f}%，创新能力{innovation_score}/10"
        elif revenue_growth > 0.25 and profit_growth > 0.20:
            return "BUY", 0.85, f"{quality}：营收和利润双高增长"
        elif revenue_growth > 0.20:
            return "BUY", 0.80, f"{quality}：营收增长稳健{revenue_growth*100:.0f}%"
        elif revenue_growth > 0.10:
            return "HOLD", 0.65, f"{quality}：增长一般{revenue_growth*100:.0f}%"
        elif revenue_growth > 0.05:
            return "HOLD", 0.60, f"{quality}：增长缓慢{revenue_growth*100:.0f}%"
        else:
            return "SELL", 0.70, f"{quality}：营收停滞或下降"

    def analyze(self) -> Dict[str, Any]:
        """执行分析"""
        logger.info(f"🔍 Growth Analyst 分析 {self.stock_code}...")

        # 1. 获取数据
        data = self.get_growth_data()

        # 2. 计算SGR
        sgr = self.calculate_sustainable_growth_rate(data)

        # 3. 生成信号
        signal, confidence, reasoning = self.generate_signal(
            data.get("revenue_growth", 0),
            data.get("profit_growth", 0),
            data.get("roe", 0),
            data.get("innovation_score", 5),
        )

        # 4. 评估成长质量
        quality, quality_level = self.evaluate_growth_quality(
            data.get("revenue_growth", 0),
            data.get("profit_growth", 0),
            data.get("roe", 0),
            data.get("innovation_score", 5),
        )

        # 5. 构建报告
        report = {
            "analyst": "Growth Analyst",
            "stock_code": self.stock_code,
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "current_price": data.get("current_price", 0),
                "revenue_growth": f"{data.get('revenue_growth', 0)*100:.1f}%",
                "profit_growth": f"{data.get('profit_growth', 0)*100:.1f}%",
                "sgr": f"{sgr*100:.1f}%",
                "roe": f"{data.get('roe', 0)*100:.1f}%",
                "gross_margin": f"{data.get('gross_margin', 0)*100:.1f}%",
            },
            "scores": {
                "innovation_score": data.get("innovation_score", 5),
                "quality_level": quality_level,
            },
            "analysis": {
                "revenue_growth": data.get("revenue_growth", 0),
                "profit_growth": data.get("profit_growth", 0),
                "tam": data.get("tam", "未知"),
                "market_share": data.get("market_share", "未知"),
                "innovation_score": data.get("innovation_score", 5),
            },
            "quality": quality,
        }

        logger.info(f"✅ 分析完成: {signal} (信心度: {confidence:.0%})")
        return report

    def save_report(self, report: Dict[str, Any]) -> Path:
        """保存报告到文件"""
        output_dir = PROJECT_ROOT / "data" / "signals"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{self.analyst_name}_{self.stock_code}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"报告已保存: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Growth Analyst - 成长投资分析")
    parser.add_argument("--stock", required=True, help="股票代码（如 300750）")
    parser.add_argument("--send", action="store_true", help="发送给 Risk Manager")

    args = parser.parse_args()

    # 执行分析
    analyst = GrowthAnalyst(args.stock)
    report = analyst.analyze()

    # 保存报告
    output_path = analyst.save_report(report)

    # 打印摘要
    print("\n" + "=" * 60)
    logger.info(f"📊 成长分析结果: {report['stock_code']}")
    print("=" * 60)
    logger.info(f"信号: {report['signal']}")
    logger.info(f"信心度: {report['confidence']:.0%}")
    logger.info(f"理由: {report['reasoning']}")
    logger.info(f"当前价格: {report['metrics']['current_price']:.2f}")
    logger.info(f"成长质量: {report['quality']} ({report['scores']['quality_level']})")
    logger.info(f"\n成长指标:")
    logger.info(f"  营收增长: {report['metrics']['revenue_growth']}")
    logger.info(f"  利润增长: {report['metrics']['profit_growth']}")
    logger.info(f"  可持续增长率: {report['metrics']['sgr']}")
    logger.info(f"  ROE: {report['metrics']['roe']}")
    logger.info(f"  创新能力: {report['scores']['innovation_score']}/10")
    print("=" * 60)


if __name__ == "__main__":
    main()
