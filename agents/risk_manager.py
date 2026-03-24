#!/usr/bin/env python3
"""
Risk Manager - 风险管理师

职责：
1. 汇总所有分析师信号
2. 计算组合风险敞口
3. 设置止损/止盈位
4. 仓位限制
5. 相关性分析
"""

import sys

from utils.logger import get_logger

logger = get_logger(__name__)

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config.settings import Settings

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class RiskManager:
    """风险管理师"""

    def __init__(self):
        self.analyst_name = "risk_manager"
        self.signals_dir = PROJECT_ROOT / "data" / "signals"

        # 风险控制规则
        self.max_position = 0.30  # 单只股票最大仓位
        self.max_total_risk = 0.80  # 总风险敞口上限
        self.stop_loss_pct = -0.06  # 止损 -6%
        self.take_profit_1 = 0.10  # 止盈1 +10%
        self.take_profit_2 = 0.20  # 止盈2 +20%

    def load_signals(self) -> Dict[str, List[Dict[str, Any]]]:
        """加载所有分析师信号"""
        signals_by_stock = {}

        if not self.signals_dir.exists():
            logger.warning(f"信号目录不存在: {self.signals_dir}")
            return signals_by_stock

        for signal_file in self.signals_dir.glob("*.json"):
            try:
                with open(signal_file, "r", encoding="utf-8") as f:
                    signal = json.load(f)

                stock_code = signal.get("stock_code")
                if stock_code not in signals_by_stock:
                    signals_by_stock[stock_code] = []

                signals_by_stock[stock_code].append(signal)
                logger.info(f"加载信号: {signal_file.name}")
            except Exception as e:
                logger.warning(f"加载 {signal_file} 失败: {e}")

        return signals_by_stock

    def calculate_consensus(self, signals: List[Dict[str, Any]]) -> tuple:
        """计算共识信号"""
        if not signals:
            return "HOLD", 0.0

        # 统计信号
        buy_count = sum(1 for s in signals if s.get("signal") == "BUY")
        sell_count = sum(1 for s in signals if s.get("signal") == "SELL")
        hold_count = sum(1 for s in signals if s.get("signal") == "HOLD")

        # 平均信心度
        avg_confidence = statistics.mean([s.get("confidence", 0) for s in signals])

        # 确定共识
        if buy_count >= 3:
            return "BUY", avg_confidence
        elif sell_count >= 3:
            return "SELL", avg_confidence
        else:
            return "HOLD", avg_confidence

    def calculate_position_size(self, consensus: str, confidence: float) -> float:
        """计算仓位大小"""
        if consensus != "BUY":
            return 0.0

        # 基于信心度调整仓位
        base_position = self.max_position * confidence

        # 限制在最大仓位内
        return min(base_position, self.max_position)

    def calculate_risk_metrics(
        self, stock_code: str, position: float, entry_price: float
    ) -> Dict[str, Any]:
        """计算风险指标"""
        return {
            "position": round(position, 4),
            "entry_price": entry_price,
            "stop_loss": round(entry_price * (1 + self.stop_loss_pct), 2),
            "take_profit_1": round(entry_price * (1 + self.take_profit_1), 2),
            "take_profit_2": round(entry_price * (1 + self.take_profit_2), 2),
            "risk_amount": round(entry_price * position * abs(self.stop_loss_pct), 2),
        }

    def analyze_correlation(self, stocks: List[str]) -> Dict[str, float]:
        """分析股票相关性（简化版）"""
        # 使用预设的相关性矩阵
        correlation_matrix = {
            "300750-002475": 0.65,  # 宁德时代 - 立讯精密（科技股相关）
            "300750-601318": 0.35,  # 宁德时代 - 中国平安
            "300750-600276": 0.25,  # 宁德时代 - 恒瑞医药
            "002475-601318": 0.40,  # 立讯精密 - 中国平安
            "002475-600276": 0.30,  # 立讯精密 - 恒瑞医药
            "601318-600276": 0.45,  # 中国平安 - 恒瑞医药
        }

        result = {}
        for i, stock1 in enumerate(stocks):
            for stock2 in stocks[i + 1 :]:
                key = f"{stock1}-{stock2}"
                if key in correlation_matrix:
                    result[key] = correlation_matrix[key]

        return result

    def generate_report(self) -> Dict[str, Any]:
        """生成风险报告"""
        logger.info("🔍 Risk Manager 汇总分析...")

        # 1. 加载信号
        signals_by_stock = self.load_signals()

        if not signals_by_stock:
            logger.warning("没有找到任何信号")
            return {}

        # 2. 分析每只股票
        portfolio = {}
        total_risk = 0.0
        stock_list = []

        for stock_code, signals in signals_by_stock.items():
            logger.info(f"\n📊 分析 {stock_code}...")

            # 计算共识
            consensus, avg_confidence = self.calculate_consensus(signals)

            # 获取当前价格（从第一个信号中获取）
            entry_price = signals[0].get("analysis", {}).get("current_price", 100.0)

            # 计算仓位
            position = self.calculate_position_size(consensus, avg_confidence)

            # 计算风险指标
            risk_metrics = self.calculate_risk_metrics(stock_code, position, entry_price)

            # 构建信号汇总
            signal_summary = {}
            for s in signals:
                analyst = s.get("analyst", "unknown")
                signal_summary[analyst] = {
                    "signal": s.get("signal"),
                    "confidence": s.get("confidence"),
                }

            portfolio[signals[0].get("stock", stock_code)] = {
                "stock_code": stock_code,
                "signals": signal_summary,
                "consensus": consensus,
                "avg_confidence": round(avg_confidence, 2),
                **risk_metrics,
            }

            total_risk += position
            stock_list.append(stock_code)

        # 3. 相关性分析
        correlation_matrix = self.analyze_correlation(stock_list)

        # 4. 生成建议
        recommendations = self._generate_recommendations(portfolio, total_risk, correlation_matrix)

        # 5. 构建报告
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio": portfolio,
            "total_positions": len(portfolio),
            "total_risk": round(total_risk, 4),
            "correlation_matrix": correlation_matrix,
            "risk_control": {
                "max_position": self.max_position,
                "max_total_risk": self.max_total_risk,
                "stop_loss_pct": self.stop_loss_pct,
                "take_profit_strategy": f"{self.take_profit_1*100:.0f}%卖50%，{self.take_profit_2*100:.0f}%清仓",
            },
            "recommendations": recommendations,
        }

        return report

    def _generate_recommendations(
        self,
        portfolio: Dict[str, Any],
        total_risk: float,
        correlation_matrix: Dict[str, float],
    ) -> str:
        """生成建议"""
        buy_count = sum(1 for data in portfolio.values() if data.get("consensus") == "BUY")
        total_count = len(portfolio)

        # 检查风险敞口
        risk_warning = ""
        if total_risk > self.max_total_risk:
            risk_warning = f"⚠️  总风险敞口 {total_risk*100:.1f}% 超过上限 {self.max_total_risk*100:.0f}%，建议降低仓位。"
        else:
            risk_warning = f"✅ 总风险敞口 {total_risk*100:.1f}% 在合理范围内。"

        # 检查高相关性
        high_correlation_warning = ""
        high_corr_pairs = [pair for pair, corr in correlation_matrix.items() if corr > 0.7]
        if high_corr_pairs:
            high_correlation_warning = (
                f"⚠️  高相关性股票对: {', '.join(high_corr_pairs)}，建议降低仓位。"
            )

        # 生成建议
        if buy_count == total_count:
            consensus = f"所有 {total_count} 只股票均获得买入共识。"
        elif buy_count >= total_count * 0.6:
            consensus = f"大部分股票（{buy_count}/{total_count}）获得买入共识。"
        else:
            consensus = f"买入共识不足（{buy_count}/{total_count}），建议观望。"

        return f"{consensus} {risk_warning} {high_correlation_warning} 建议均衡配置，单只股票仓位≤30%，止损统一设为-6%，止盈分两档+10%和+20%。"

    def save_report(self, report: Dict[str, Any]) -> Path:
        """保存报告"""
        output_dir = PROJECT_ROOT / "data" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risk_report_{timestamp}.json"
        output_path = output_dir / filename

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"\n✅ 风险报告已保存: {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Risk Manager - 风险管理")
    parser.add_argument("--send", action="store_true", help="发送给 Portfolio Manager")

    args = parser.parse_args()

    # 执行分析
    manager = RiskManager()
    report = manager.generate_report()

    if not report:
        logger.error("无法生成报告")
        return

    # 保存报告
    output_path = manager.save_report(report)

    # 打印摘要
    print("\n" + "=" * 60)
    logger.info("📊 风险报告摘要")
    print("=" * 60)
    logger.info(f"总持仓数: {report['total_positions']}")
    logger.info(f"总风险敞口: {report['total_risk']:.1%}")
    logger.info(f"\n建议: {report['recommendations']}")
    print("=" * 60)

    # 打印每只股票
    logger.info("\n📋 持仓详情:")
    for stock, data in report["portfolio"].items():
        logger.info(f"\n{stock}:")
        logger.info(f"  共识: {data['consensus']}")
        logger.info(f"  信心度: {data['avg_confidence']:.0%}")
        logger.info(f"  仓位: {data['position']:.1%}")
        logger.info(f"  止损: {data['stop_loss']}")
        logger.info(f"  止盈: {data['take_profit_1']} / {data['take_profit_2']}")

    # 可选：发送给 Portfolio Manager
    if args.send:
        import subprocess

        cmd = f"clawteam inbox send quant-fund portfolio-manager @{output_path}"
        logger.info("\n📤 发送报告给 Portfolio Manager...")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("报告已发送")
        else:
            logger.error(f"发送失败: {result.stderr}")


if __name__ == "__main__":
    main()
