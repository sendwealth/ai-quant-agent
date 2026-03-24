#!/usr/bin/env python3
"""
实时监控脚本 - ClawTeam AI 对冲基金

功能：
1. 实时监控投资组合
2. 自动更新信号
3. 风险告警
4. 性能追踪
"""

import json
import time
from datetime import datetime
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).parent.parent
SIGNALS_DIR = PROJECT_ROOT / "data" / "signals"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
LOGS_DIR = PROJECT_ROOT / "logs" / "monitor"

# 创建日志目录
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class PortfolioMonitor:
    """投资组合监控器"""

    def __init__(self):
        self.positions = {}
        self.alerts = []

    def load_latest_report(self):
        """加载最新的风险报告"""
        if not REPORTS_DIR.exists():
            return None

        reports = sorted(REPORTS_DIR.glob("risk_report_*.json"), reverse=True)
        if not reports:
            return None

        with open(reports[0], "r", encoding="utf-8") as f:
            return json.load(f)

    def check_risk_alerts(self, report):
        """检查风险告警"""
        if not report:
            return

        # 检查总风险敞口
        total_risk = report.get("total_risk", 0)
        if total_risk > 0.8:
            self.alerts.append(
                {
                    "level": "ERROR",
                    "message": f"总风险敞口过高: {total_risk:.1%} > 80%",
                    "timestamp": datetime.now().isoformat(),
                }
            )
        elif total_risk > 0.6:
            self.alerts.append(
                {
                    "level": "WARNING",
                    "message": f"总风险敞口偏高: {total_risk:.1%}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # 检查单只股票仓位
        for stock, data in report.get("portfolio", {}).items():
            position = data.get("position", 0)
            if position > 0.3:
                self.alerts.append(
                    {
                        "level": "ERROR",
                        "message": f"{stock} 仓位超标: {position:.1%} > 30%",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            # 检查信号一致性
            consensus = data.get("consensus")
            avg_confidence = data.get("avg_confidence", 0)
            if consensus == "SELL" and avg_confidence > 0.7:
                self.alerts.append(
                    {
                        "level": "WARNING",
                        "message": f"{stock} 强烈卖出信号: 信心度 {avg_confidence:.0%}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    def generate_monitoring_report(self):
        """生成监控报告"""
        report = self.load_latest_report()

        if not report:
            return {
                "status": "NO_DATA",
                "message": "没有找到风险报告",
                "timestamp": datetime.now().isoformat(),
            }

        # 检查风险告警
        self.check_risk_alerts(report)

        # 生成监控报告
        monitoring_report = {
            "timestamp": datetime.now().isoformat(),
            "status": "OK" if not self.alerts else "ALERT",
            "portfolio_summary": {
                "total_positions": report.get("total_positions", 0),
                "total_risk": report.get("total_risk", 0),
            },
            "positions": {},
            "alerts": self.alerts,
        }

        # 添加持仓详情
        for stock, data in report.get("portfolio", {}).items():
            monitoring_report["positions"][stock] = {
                "consensus": data.get("consensus"),
                "confidence": data.get("avg_confidence"),
                "position": data.get("position"),
                "stop_loss": data.get("stop_loss"),
                "take_profit": [data.get("take_profit_1"), data.get("take_profit_2")],
            }

        return monitoring_report

    def print_dashboard(self, report):
        """打印仪表盘"""
        print("\n" + "=" * 70)
        print("🦞 AI 对冲基金 - 实时监控仪表盘")
        print("=" * 70)
        print(f"时间: {report['timestamp']}")
        print(f"状态: {report['status']}")

        # 投资组合摘要
        summary = report.get("portfolio_summary", {})
        print(f"\n📊 投资组合摘要:")
        print(f"  总持仓数: {summary.get('total_positions', 0)}")
        print(f"  总风险敞口: {summary.get('total_risk', 0):.1%}")

        # 持仓详情
        print(f"\n📋 持仓详情:")
        for stock, data in report.get("positions", {}).items():
            print(f"\n  {stock}:")
            print(f"    共识: {data['consensus']}")
            print(f"    信心度: {data['confidence']:.0%}")
            print(f"    仓位: {data['position']:.1%}")
            print(f"    止损: {data['stop_loss']}")
            print(f"    止盈: {data['take_profit'][0]} / {data['take_profit'][1]}")

        # 告警
        if report.get("alerts"):
            print(f"\n⚠️  告警 ({len(report['alerts'])} 个):")
            for alert in report["alerts"]:
                level = alert["level"]
                message = alert["message"]
                if level == "ERROR":
                    print(f"  🔴 {message}")
                else:
                    print(f"  🟡 {message}")
        else:
            print(f"\n✅ 无告警")

        print("=" * 70)

    def save_monitoring_log(self, report):
        """保存监控日志"""
        log_file = LOGS_DIR / f"monitor_{datetime.now().strftime('%Y%m%d')}.jsonl"

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

    def run_continuous_monitoring(self, interval_seconds=300):
        """持续监控（每5分钟）"""
        print("🦞 启动实时监控...")
        print(f"监控间隔: {interval_seconds} 秒")
        print("按 Ctrl+C 停止\n")

        try:
            while True:
                # 生成监控报告
                report = self.generate_monitoring_report()

                # 打印仪表盘
                self.print_dashboard(report)

                # 保存日志
                self.save_monitoring_log(report)

                # 如果有告警，发送 ClawTeam 消息
                if report.get("alerts"):
                    self.send_alert_notification(report)

                # 等待
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n✅ 监控已停止")

    def send_alert_notification(self, report):
        """发送告警通知（通过 ClawTeam）"""
        alerts = report.get("alerts", [])
        if not alerts:
            return

        # 构建告警消息
        alert_messages = []
        for alert in alerts:
            alert_messages.append(f"[{alert['level']}] {alert['message']}")

        message = f"⚠️ 风险告警 ({len(alerts)}个):\n" + "\n".join(alert_messages)

        # 通过 ClawTeam 广播
        cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && clawteam inbox broadcast quant-fund '{message}'"
        subprocess.run(cmd, shell=True, capture_output=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="AI 对冲基金实时监控")
    parser.add_argument("--interval", type=int, default=300, help="监控间隔（秒）")
    parser.add_argument("--once", action="store_true", help="只运行一次")

    args = parser.parse_args()

    monitor = PortfolioMonitor()

    if args.once:
        # 只运行一次
        report = monitor.generate_monitoring_report()
        monitor.print_dashboard(report)
        monitor.save_monitoring_log(report)
    else:
        # 持续监控
        monitor.run_continuous_monitoring(args.interval)


if __name__ == "__main__":
    main()
