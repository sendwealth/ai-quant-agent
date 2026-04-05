#!/usr/bin/env python3
"""
Fundamentals Analyst - 基本面分析（修复版）

策略：基本面分析
维度：P/E、P/B、负债率、财务健康度

修复内容：
- 使用 FinancialDataFetcherV2 获取真实P/E、 P/B数据
- 修复数据异常问题
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入财务数据获取器 v2
from utils.financial_data_fetcher_v2 import FinancialDataFetcherV2


class FundamentalsAnalyst:
    """基本面分析师"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.analyst_name = "fundamentals"

    def get_financial_data(self) -> Dict[str, Any]:
        """获取财务数据（真实数据 v2）"""
        data = {}

        # 1. 获取财务指标（使用新获取器）
        financial_data = FinancialDataFetcherV2.get_real_time_indicators(self.stock_code)

        # 2. 获取当前价格
        price = financial_data.get("total_value", 0)  # 暂时用市值
        if not price:
            price = 100.0  # 默认价格

        data.update(
            {
                "price": price,
                "pe_ratio": financial_data["pe_ratio"],
                "pb_ratio": financial_data["pb_ratio"],
                "de_ratio": financial_data["de_ratio"],
                "roe": financial_data["roe"],
                "gross_margin": financial_data["gross_margin"],
                "net_margin": financial_data["net_margin"],
                "current_ratio": financial_data["current_ratio"],
                "quick_ratio": financial_data["quick_ratio"],
            }
        )

        logger.info(
            f"✅ 获取到财务数据: P/E={data['pe_ratio']:.1f}, P/B={data['pb_ratio']:.1f}, 负债率={data['de_ratio']*100:.1f}%"
        )

        return data

    def calculate_financial_health(self, data: Dict[str, Any]) -> int:
        """计算财务健康度（1-10分）"""
        score = 5  # 基础分

        # ROE
        if data.get("roe", 0) > 0.15:
            score += 2
        elif data.get("roe", 0) > 0.10:
            score += 1

        # 负债率
        if data.get("de_ratio", 1) < 0.4:
            score += 2
        elif data.get("de_ratio", 1) < 0.6:
            score += 1
        elif data.get("de_ratio", 1) > 0.8:
            score -= 2

        # 流动性
        if data.get("current_ratio", 0) > 2.0:
            score += 1
        elif data.get("current_ratio", 0) < 1.0:
            score -= 1

        # 毛利率
        if data.get("gross_margin", 0) > 0.4:
            score += 1
        elif data.get("gross_margin", 0) < 0.2:
            score -= 1

        return max(1, min(10, score))

    def evaluate_valuation(self, pe_ratio: float, pb_ratio: float) -> tuple:
        """评估估值水平"""
        # P/E 评估
        if pe_ratio < 15:
            pe_score = "低估"
        elif pe_ratio < 25:
            pe_score = "合理"
        elif pe_ratio < 40:
            pe_score = "偏高"
        else:
            pe_score = "高估"

        # P/B 评估
        if pb_ratio < 2:
            pb_score = "低估"
        elif pb_ratio < 4:
            pb_score = "合理"
        elif pb_ratio < 6:
            pb_score = "偏高"
        else:
            pb_score = "高估"

        return pe_score, pb_score

    def generate_signal(
        self, de_ratio: float, financial_health: int, pe_ratio: float, pb_ratio: float
    ) -> tuple:
        """生成交易信号"""
        # 估值评估
        pe_score, pb_score = self.evaluate_valuation(pe_ratio, pb_ratio)

        # 决策规则
        if de_ratio < 0.5 and financial_health >= 7:
            # 财务健康 + 低负债
            if pe_score in ["低估", "合理"] and pb_score in ["低估", "合理"]:
                return "BUY", 0.80, "财务健康且估值合理"
            elif pe_score == "低估" or pb_score == "低估":
                return "BUY", 0.75, "财务健康且估值偏低"
            else:
                return "HOLD", 0.65, "财务健康但估值偏高"

        elif de_ratio > 0.8 or financial_health < 5:
            # 财务风险
            return "SELL", 0.75, "财务风险较高"

        else:
            # 中等状况
            if pe_score == "低估" and pb_score == "低估":
                return "BUY", 0.70, "估值偏低但财务一般"
            elif pe_score == "高估" and pb_score == "高估":
                return "SELL", 0.70, "估值过高"
            else:
                return "HOLD", 0.60, "财务和估值均中等"

    def analyze(self) -> Dict[str, Any]:
        """执行分析"""
        logger.info(f"🔍 Fundamentals Analyst 分析 {self.stock_code}...")

        # 1. 获取数据
        data = self.get_financial_data()

        # 2. 计算财务健康度
        financial_health = self.calculate_financial_health(data)

        # 3. 生成信号
        signal, confidence, reasoning = self.generate_signal(
            data.get("de_ratio", 0.5),
            financial_health,
            data.get("pe_ratio", 25),
            data.get("pb_ratio", 3),
        )

        # 4. 估值评估
        pe_score, pb_score = self.evaluate_valuation(
            data.get("pe_ratio", 25), data.get("pb_ratio", 3)
        )

        # 5. 构建报告
        report = {
            "analyst": "Fundamentals Analyst",
            "stock_code": self.stock_code,
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "current_price": data.get("price", 0),
                "pe_ratio": data.get("pe_ratio", 0),
                "pb_ratio": data.get("pb_ratio", 0),
                "de_ratio": f"{data.get('de_ratio', 0)*100:.1f}%",
                "roe": f"{data.get('roe', 0)*100:.1f}%",
                "gross_margin": f"{data.get('gross_margin', 0)*100:.1f}%",
                "net_margin": f"{data.get('net_margin', 0)*100:.1f}%",
                "current_ratio": data.get("current_ratio", 0),
                "quick_ratio": data.get("quick_ratio", 0),
            },
            "scores": {
                "financial_health": financial_health,
                "pe_valuation": pe_score,
                "pb_valuation": pb_score,
            },
            "analysis": {
                "pe_ratio": data.get("pe_ratio", 0),
                "pb_ratio": data.get("pb_ratio", 0),
                "de_ratio": data.get("de_ratio", 0),
                "financial_health": financial_health,
            },
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
    parser = argparse.ArgumentParser(description="Fundamentals Analyst - 基本面分析")
    parser.add_argument("--stock", required=True, help="股票代码（如 300750）")
    parser.add_argument("--send", action="store_true", help="发送给 Risk Manager")

    args = parser.parse_args()

    # 执行分析
    analyst = FundamentalsAnalyst(args.stock)
    report = analyst.analyze()

    # 保存报告
    output_path = analyst.save_report(report)

    # 打印摘要
    print("\n" + "=" * 60)
    logger.info(f"📊 基本面分析结果: {report['stock_code']}")
    print("=" * 60)
    logger.info(f"信号: {report['signal']}")
    logger.info(f"信心度: {report['confidence']:.0%}")
    logger.info(f"理由: {report['reasoning']}")
    logger.info(f"当前价格: {report['metrics']['current_price']:.2f}")
    logger.info(f"P/E: {report['metrics']['pe_ratio']:.1f} ({report['scores']['pe_valuation']})")
    logger.info(f"P/B: {report['metrics']['pb_ratio']:.1f} ({report['scores']['pb_valuation']})")
    logger.info(f"负债率: {report['metrics']['de_ratio']}")
    logger.info(f"财务健康度: {report['scores']['financial_health']}/10")
    print("=" * 60)


if __name__ == "__main__":
    main()
