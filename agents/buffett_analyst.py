#!/usr/bin/env python3
"""
Buffett Analyst - 价值投资分析师

策略：巴菲特价值投资理念
维度：护城河、ROE、DCF、安全边际、管理层质量
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

try:
    import akshare as ak
except ImportError:
    ak = None

# 导入真实数据获取器
from utils.real_data_fetcher import RealDataFetcher


class BuffettAnalyst:
    """价值投资分析师"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.analyst_name = "buffett"

    def get_financial_data(self) -> Dict[str, Any]:
        """获取财务数据（真实数据）"""
        data = {}

        # 1. 获取当前价格（使用真实数据获取器）
        price = RealDataFetcher.get_stock_price(self.stock_code)
        if not price:
            raise RuntimeError(f"❌ 无法获取 {self.stock_code} 的实时价格")
        data["price"] = price

        # 2. 获取财务指标（如果 akshare 可用）
        if ak:
            try:
                df = ak.stock_financial_analysis_indicator(symbol=self.stock_code)
                if not df.empty:
                    latest = df.iloc[0]
                    data.update(
                        {
                            "roe": float(latest.get("净资产收益率", 0)) / 100,
                            "gross_margin": float(latest.get("销售毛利率", 0)) / 100,
                            "net_margin": float(latest.get("销售净利率", 0)) / 100,
                            "debt_ratio": float(latest.get("资产负债率", 0)) / 100,
                            "current_ratio": float(latest.get("流动比率", 0)),
                        }
                    )
                    logger.info(f"✅ 获取到财务数据: ROE={data['roe']*100:.1f}%")
            except Exception as e:
                logger.warning(f"获取财务指标失败: {e}，使用默认值")
                # 使用保守的默认值
                data.update(
                    {
                        "roe": 0.12,
                        "gross_margin": 0.25,
                        "net_margin": 0.10,
                        "debt_ratio": 0.50,
                        "current_ratio": 1.5,
                    }
                )
        else:
            logger.warning("akshare 不可用，使用默认财务指标")
            data.update(
                {
                    "roe": 0.12,
                    "gross_margin": 0.25,
                    "net_margin": 0.10,
                    "debt_ratio": 0.50,
                    "current_ratio": 1.5,
                }
            )

        return data

    def analyze_competitive_mote(self, data: Dict[str, Any]) -> int:
        """分析护城河（1-10分）"""
        score = 5  # 基础分

        # 高毛利率 = 强护城河
        if data.get("gross_margin", 0) > 0.5:
            score += 2
        elif data.get("gross_margin", 0) > 0.3:
            score += 1

        # 高 ROE = 竞争优势
        if data.get("roe", 0) > 0.15:
            score += 2
        elif data.get("roe", 0) > 0.10:
            score += 1

        # 低负债率 = 财务稳健
        if data.get("debt_ratio", 0) < 0.5:
            score += 1

        return min(10, score)

    def calculate_dcf_value(self, data: Dict[str, Any]) -> float:
        """计算 DCF 估值（简化版）"""
        # 使用保守的估值方法：基于 ROE 和当前价格
        current_price = data.get("price", 0)
        roe = data.get("roe", 0.12)

        # 简化 DCF：如果 ROE > 15%，给予 1.2 倍溢价
        if roe > 0.15:
            dcf_value = current_price * 1.2
        elif roe > 0.12:
            dcf_value = current_price * 1.1
        else:
            dcf_value = current_price * 0.95

        logger.info(f"DCF 估值: {dcf_value:.2f} (基于 ROE={roe*100:.1f}%)")
        return dcf_value

    def calculate_safety_margin(self, current_price: float, dcf_value: float) -> float:
        """计算安全边际"""
        if current_price <= 0:
            return 0.0
        return (dcf_value - current_price) / current_price

    def analyze_management_quality(self, data: Dict[str, Any]) -> int:
        """分析管理层质量（1-10分）"""
        score = 5  # 基础分

        # 高 ROE = 管理层能力
        if data.get("roe", 0) > 0.15:
            score += 2
        elif data.get("roe", 0) > 0.10:
            score += 1

        # 高净利率 = 运营效率
        if data.get("net_margin", 0) > 0.15:
            score += 2
        elif data.get("net_margin", 0) > 0.10:
            score += 1

        # 低负债率 = 财务审慎
        if data.get("debt_ratio", 0) < 0.5:
            score += 1

        # 高流动比率 = 流动性好
        if data.get("current_ratio", 0) > 1.5:
            score += 1

        return min(10, score)

    def generate_signal(
        self, roe: float, safety_margin: float, mote_score: int, management_score: int
    ) -> tuple:
        """生成交易信号"""
        # 决策规则
        if roe < 0.10:
            return "SELL", 0.6, "ROE < 10%，盈利能力不足"
        elif roe < 0.15:
            if safety_margin < 0:
                return "HOLD", 0.5, "ROE 一般，安全边际不足"
            else:
                return "HOLD", 0.6, "ROE 一般，但有安全边际"
        else:  # roe >= 0.15
            if safety_margin >= 0.30 and mote_score >= 7 and management_score >= 7:
                return "BUY", 0.90, "ROE优秀，安全边际充足，护城河强，管理层优秀"
            elif safety_margin >= 0.20 and mote_score >= 6:
                return "BUY", 0.85, "ROE优秀，安全边际较好，护城河较强"
            elif safety_margin >= 0.10:
                return "BUY", 0.75, "ROE优秀，有一定安全边际"
            else:
                return "HOLD", 0.65, "ROE优秀，但安全边际不足"

    def analyze(self) -> Dict[str, Any]:
        """执行分析"""
        logger.info(f"🔍 Buffett Analyst 分析 {self.stock_code}...")

        # 1. 获取数据
        data = self.get_financial_data()

        # 2. 分析各维度
        competitive_mote = self.analyze_competitive_mote(data)
        management_quality = self.analyze_management_quality(data)
        dcf_value = self.calculate_dcf_value(data)
        current_price = data.get("price")
        if not current_price:
            raise RuntimeError("❌ 缺少当前价格数据")

        safety_margin = self.calculate_safety_margin(current_price, dcf_value)
        roe = data.get("roe", 0)

        # 3. 生成信号
        signal, confidence, reasoning = self.generate_signal(
            roe, safety_margin, competitive_mote, management_quality
        )

        # 4. 构建报告
        report = {
            "analyst": "Buffett Analyst",
            "stock_code": self.stock_code,
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "metrics": {
                "current_price": current_price,
                "dcf_value": dcf_value,
                "safety_margin": f"{safety_margin*100:.1f}%",
                "roe": f"{roe*100:.1f}%",
                "gross_margin": f"{data.get('gross_margin', 0)*100:.1f}%",
                "net_margin": f"{data.get('net_margin', 0)*100:.1f}%",
                "debt_ratio": f"{data.get('debt_ratio', 0)*100:.1f}%",
                "current_ratio": data.get("current_ratio", 0),
            },
            "scores": {
                "competitive_mote": competitive_mote,
                "management_quality": management_quality,
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
    parser = argparse.ArgumentParser(description="Buffett Analyst - 价值投资分析")
    parser.add_argument("--stock", required=True, help="股票代码（如 300750）")
    parser.add_argument("--send", action="store_true", help="发送给 Risk Manager")

    args = parser.parse_args()

    # 执行分析
    analyst = BuffettAnalyst(args.stock)
    report = analyst.analyze()

    # 保存报告
    output_path = analyst.save_report(report)

    # 打印摘要
    print("\n" + "=" * 60)
    logger.info(f"📊 分析结果: {report['stock_code']}")
    print("=" * 60)
    logger.info(f"信号: {report['signal']}")
    logger.info(f"信心度: {report['confidence']:.0%}")
    logger.info(f"理由: {report['reasoning']}")
    logger.info(f"当前价格: {report['metrics']['current_price']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
