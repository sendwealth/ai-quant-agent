#!/usr/bin/env python3
"""
测试所有监控的股票 - 修复版
"""

import json
import sys
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class FullBacktest:
    """完整回测系统"""

    def __init__(self):
        self.signals_dir = PROJECT_ROOT / "data" / "signals"
        self.agents = ["buffett", "technical", "fundamentals", "growth", "sentiment"]
        self.stocks = self.load_monitored_stocks()

    def load_monitored_stocks(self) -> List[Dict]:
        """从配置文件加载监控股票列表"""
        config_file = PROJECT_ROOT / "config" / "data_sources.yaml"
        
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config.get("monitored_stocks", [])
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {e}")
            # 默认股票列表
            return [
                {"code": "300750", "name": "宁德时代"},
                {"code": "002475", "name": "立讯精密"},
                {"code": "601318", "name": "中国平安"},
                {"code": "600276", "name": "恒瑞医药"},
            ]

    def load_signals(self, stock_code: str) -> Dict:
        """加载所有agent的信号"""
        signals = {}

        for agent in self.agents:
            signal_file = self.signals_dir / f"{agent}_{stock_code}.json"

            if signal_file.exists():
                try:
                    with open(signal_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        signals[agent] = {
                            "signal": data.get("signal", "HOLD"),
                            "confidence": data.get("confidence", 0.5),
                            "reasoning": data.get("reasoning", ""),
                        }
                except Exception as e:
                    print(f"⚠️  加载 {agent} 信号失败: {e}")

        return signals

    def calculate_consensus(self, signals: Dict) -> tuple:
        """计算共识信号"""
        if not signals:
            return "HOLD", 0.5, "无信号"

        buy_count = sum(1 for s in signals.values() if s["signal"] == "BUY")
        sell_count = sum(1 for s in signals.values() if s["signal"] == "SELL")
        hold_count = sum(1 for s in signals.values() if s["signal"] == "HOLD")

        avg_confidence = sum(s["confidence"] for s in signals.values()) / len(signals)
        total = len(signals)

        if buy_count >= total * 0.6:
            return "BUY", avg_confidence, f"强烈买入（{buy_count}/{total} agents看涨）"
        elif sell_count >= total * 0.6:
            return "SELL", avg_confidence, f"强烈卖出（{sell_count}/{total} agents看跌）"
        elif buy_count > sell_count:
            return "BUY", avg_confidence * 0.8, f"偏多（{buy_count}/{total} agents看涨）"
        elif sell_count > buy_count:
            return "SELL", avg_confidence * 0.8, f"偏空（{sell_count}/{total} agents看跌）"
        else:
            return "HOLD", avg_confidence, f"中性（{hold_count}/{total} agents观望）"

    def run_backtest_all(self) -> List[Dict]:
        """运行所有股票的回测"""
        reports = []

        print(f"\n{'='*80}")
        print(f"📊 完整回测报告 - {len(self.stocks)}只股票")
        print(f"{'='*80}\n")

        for stock in self.stocks:
            code = stock["code"]
            name = stock["name"]

            print(f"\n{'='*60}")
            print(f"🔍 {name} ({code})")
            print(f"{'='*60}")

            # 加载信号
            signals = self.load_signals(code)

            if not signals:
                print(f"❌ 无信号数据，跳过")
                continue

            # 显示信号
            print(f"\n📊 Agent信号汇总:")
            print(f"{'Agent':<15} {'信号':<10} {'信心度':<10} {'理由'}")
            print("-" * 80)

            for agent, data in signals.items():
                signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(data["signal"], "⚪")
                print(
                    f"{agent:<15} {signal_emoji} {data['signal']:<8} {data['confidence']:.0%}       {data['reasoning'][:40]}"
                )

            # 计算共识
            consensus_signal, consensus_confidence, consensus_reason = self.calculate_consensus(signals)

            # 显示共识
            print(f"\n{'='*60}")
            print(f"🎯 共识信号")
            print(f"{'='*60}")

            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(consensus_signal, "⚪")
            print(f"信号: {signal_emoji} {consensus_signal}")
            print(f"信心度: {consensus_confidence:.0%}")
            print(f"理由: {consensus_reason}")

            # 构建报告
            report = {
                "stock_code": code,
                "stock_name": name,
                "timestamp": datetime.now().isoformat(),
                "consensus": {
                    "signal": consensus_signal,
                    "confidence": consensus_confidence,
                    "reasoning": consensus_reason,
                },
                "agents": signals,
                "summary": {
                    "buy_count": sum(1 for s in signals.values() if s["signal"] == "BUY"),
                    "sell_count": sum(1 for s in signals.values() if s["signal"] == "SELL"),
                    "hold_count": sum(1 for s in signals.values() if s["signal"] == "HOLD"),
                    "total_agents": len(signals),
                },
            }

            reports.append(report)

        # 保存汇总报告
        self.save_summary_report(reports)

        return reports

    def save_summary_report(self, reports: List[Dict]):
        """保存汇总报告"""
        if not reports:
            return

        output_dir = PROJECT_ROOT / "data" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"full_backtest_{timestamp_str}.json"

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_stocks": len(reports),
            "stocks": reports,
            "portfolio_recommendation": self.generate_portfolio_recommendation(reports),
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"\n{'='*80}")
        print(f"✅ 汇总报告已保存: {output_file}")
        print(f"{'='*80}\n")

    def generate_portfolio_recommendation(self, reports: List[Dict]) -> Dict:
        """生成组合建议"""
        buy_stocks = [r for r in reports if r["consensus"]["signal"] == "BUY"]
        sell_stocks = [r for r in reports if r["consensus"]["signal"] == "SELL"]
        hold_stocks = [r for r in reports if r["consensus"]["signal"] == "HOLD"]

        recommendation = {
            "buy": [
                {"code": s["stock_code"], "name": s["stock_name"], "confidence": s["consensus"]["confidence"]}
                for s in buy_stocks
            ],
            "sell": [
                {"code": s["stock_code"], "name": s["stock_name"], "confidence": s["consensus"]["confidence"]}
                for s in sell_stocks
            ],
            "hold": [
                {"code": s["stock_code"], "name": s["stock_name"], "confidence": s["consensus"]["confidence"]}
                for s in hold_stocks
            ],
            "suggested_allocation": {},
        }

        # 生成建议仓位
        if buy_stocks:
            for stock in buy_stocks:
                confidence = stock["consensus"]["confidence"]
                # 基于信心度计算仓位（信心度70% → 20%仓位，80% → 25%仓位，90% → 30%仓位）
                position = min(0.30, confidence * 0.35)
                recommendation["suggested_allocation"][stock["stock_code"]] = {
                    "name": stock["stock_name"],
                    "position": round(position, 2),
                    "reason": f"基于{stock['summary']['buy_count']}/{stock['summary']['total_agents']}agents看涨",
                }

        return recommendation


def main():
    """运行完整回测"""
    backtest = FullBacktest()
    reports = backtest.run_backtest_all()

    # 显示最终汇总
    print(f"\n{'='*80}")
    print(f"📊 投资组合建议")
    print(f"{'='*80}\n")

    if reports:
        buy_stocks = [r for r in reports if r["consensus"]["signal"] == "BUY"]
        sell_stocks = [r for r in reports if r["consensus"]["signal"] == "SELL"]

        if buy_stocks:
            print(f"🟢 建议买入 ({len(buy_stocks)}只):")
            for stock in buy_stocks:
                print(
                    f"  • {stock['stock_name']} ({stock['stock_code']}): {stock['consensus']['confidence']:.0%}信心度"
                )

        if sell_stocks:
            print(f"\n🔴 建议卖出 ({len(sell_stocks)}只):")
            for stock in sell_stocks:
                print(
                    f"  • {stock['stock_name']} ({stock['stock_code']}): {stock['consensus']['confidence']:.0%}信心度"
                )

        hold_stocks = [r for r in reports if r["consensus"]["signal"] == "HOLD"]
        if hold_stocks:
            print(f"\n🟡 观望 ({len(hold_stocks)}只):")
            for stock in hold_stocks:
                print(
                    f"  • {stock['stock_name']} ({stock['stock_code']}): {stock['consensus']['confidence']:.0%}信心度"
                )


if __name__ == "__main__":
    main()
