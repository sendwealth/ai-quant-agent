#!/usr/bin/env python3
"""
快速回测 - 使用修复后的Agent信号
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class QuickBacktest:
    """快速回测系统"""

    def __init__(self, stock_code: str):
        self.stock_code = stock_code
        self.signals_dir = PROJECT_ROOT / "data" / "signals"
        self.agents = ["buffett", "technical", "fundamentals", "growth", "sentiment"]

    def load_signals(self) -> Dict:
        """加载所有agent的信号"""
        signals = {}

        for agent in self.agents:
            signal_file = self.signals_dir / f"{agent}_{self.stock_code}.json"

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

        # 统计信号
        buy_count = sum(1 for s in signals.values() if s["signal"] == "BUY")
        sell_count = sum(1 for s in signals.values() if s["signal"] == "SELL")
        hold_count = sum(1 for s in signals.values() if s["signal"] == "HOLD")

        # 平均信心度
        avg_confidence = sum(s["confidence"] for s in signals.values()) / len(signals)

        # 决策逻辑
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

    def run_backtest(self) -> Dict:
        """运行回测"""
        print(f"\n{'='*60}")
        print(f"🔍 {self.stock_code} 快速回测")
        print(f"{'='*60}")

        # 1. 加载信号
        signals = self.load_signals()

        if not signals:
            print("❌ 无信号数据")
            return {}

        # 2. 显示各agent信号
        print(f"\n📊 Agent信号汇总:")
        print(f"{'Agent':<15} {'信号':<10} {'信心度':<10} {'理由'}")
        print("-" * 80)

        for agent, data in signals.items():
            signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(data["signal"], "⚪")
            print(
                f"{agent:<15} {signal_emoji} {data['signal']:<8} {data['confidence']:.0%}       {data['reasoning'][:40]}"
            )

        # 3. 计算共识
        consensus_signal, consensus_confidence, consensus_reason = self.calculate_consensus(signals)

        # 4. 显示共识
        print(f"\n{'='*60}")
        print(f"🎯 共识信号")
        print(f"{'='*60}")

        signal_emoji = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(consensus_signal, "⚪")
        print(f"信号: {signal_emoji} {consensus_signal}")
        print(f"信心度: {consensus_confidence:.0%}")
        print(f"理由: {consensus_reason}")

        # 5. 构建报告
        report = {
            "stock_code": self.stock_code,
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

        # 6. 保存报告
        output_dir = PROJECT_ROOT / "data" / "reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"quick_backtest_{self.stock_code}_{timestamp_str}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 报告已保存: {output_file}")

        return report


def main():
    """测试所有监控股票"""
    stocks = ["300750", "002475"]  # 宁德时代、立讯精密

    for stock in stocks:
        backtest = QuickBacktest(stock)
        report = backtest.run_backtest()

        if report:
            print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
