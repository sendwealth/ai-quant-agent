#!/usr/bin/env python3
"""
实时风险监控系统
监控持仓风险和市场状态
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List

class RiskMonitor:
    """风险监控器"""

    def __init__(self):
        self.risk_params = {
            'stop_loss': -0.06,
            'take_profit_1': 0.10,
            'take_profit_2': 0.20,
            'max_position_pct': 0.30,
            'max_total_position_pct': 0.80,
            'daily_loss_limit': -0.03,
            'weekly_loss_limit': -0.05
        }

        self.alerts = []

    def check_position_risk(self, symbol: str, position: Dict, current_price: float) -> Dict:
        """检查单个持仓风险"""

        avg_cost = position['avg_cost']
        shares = position['shares']
        pnl_pct = (current_price - avg_cost) / avg_cost
        pnl_amount = (current_price - avg_cost) * shares

        # 止损检查
        if pnl_pct <= self.risk_params['stop_loss']:
            return {
                'level': 'CRITICAL',
                'symbol': symbol,
                'type': 'STOP_LOSS',
                'message': f'触发止损线！亏损{pnl_pct:.2%}（¥{pnl_amount:,.2f}）',
                'action': f'建议立即卖出{shares}股',
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount
            }

        # 止盈检查
        if pnl_pct >= self.risk_params['take_profit_2']:
            return {
                'level': 'WARNING',
                'symbol': symbol,
                'type': 'TAKE_PROFIT_2',
                'message': f'触发止盈2！盈利{pnl_pct:.2%}（¥{pnl_amount:,.2f}）',
                'action': f'建议清仓卖出{shares}股',
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount
            }

        if pnl_pct >= self.risk_params['take_profit_1']:
            return {
                'level': 'INFO',
                'symbol': symbol,
                'type': 'TAKE_PROFIT_1',
                'message': f'触发止盈1！盈利{pnl_pct:.2%}（¥{pnl_amount:,.2f}）',
                'action': f'建议卖出{shares//2}股（50%）',
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount
            }

        # 接近止损警告
        if pnl_pct <= self.risk_params['stop_loss'] * 0.8:
            return {
                'level': 'WARNING',
                'symbol': symbol,
                'type': 'NEAR_STOP_LOSS',
                'message': f'接近止损线！亏损{pnl_pct:.2%}',
                'action': '密切关注',
                'pnl_pct': pnl_pct,
                'pnl_amount': pnl_amount
            }

        return {
            'level': 'OK',
            'symbol': symbol,
            'pnl_pct': pnl_pct,
            'pnl_amount': pnl_amount
        }

    def check_portfolio_risk(self, portfolio: Dict) -> List[Dict]:
        """检查组合风险"""

        alerts = []

        # 总仓位检查
        position_value = portfolio['position_value']
        total_value = portfolio['total_value']
        position_pct = position_value / total_value

        if position_pct > self.risk_params['max_total_position_pct']:
            alerts.append({
                'level': 'WARNING',
                'type': 'HIGH_POSITION',
                'message': f'总仓位过高：{position_pct:.1%}（上限{self.risk_params["max_total_position_pct"]:.1%}）',
                'action': '建议降低仓位'
            })

        # 现金比例检查
        cash_ratio = portfolio['cash'] / total_value
        if cash_ratio < 0.15:
            alerts.append({
                'level': 'WARNING',
                'type': 'LOW_CASH',
                'message': f'现金比例过低：{cash_ratio:.1%}（建议>15%）',
                'action': '风险较高'
            })

        return alerts

    def check_market_risk(self, signals: Dict) -> Dict:
        """检查市场风险"""

        buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
        sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')
        total = len(signals)

        # 市场情绪
        if sell_count > total * 0.6:
            sentiment = 'BEARISH'
            message = '市场偏空，多数股票发出卖出信号'
        elif buy_count > total * 0.6:
            sentiment = 'BULLISH'
            message = '市场偏多，多数股票发出买入信号'
        else:
            sentiment = 'NEUTRAL'
            message = '市场中性，信号分化'

        return {
            'sentiment': sentiment,
            'message': message,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': total - buy_count - sell_count
        }

    def run_risk_check(self):
        """运行风险检查"""

        print("="*70)
        print("⚠️  风险监控系统")
        print("="*70)
        print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 加载账户状态
        state_file = Path('data/paper_trading_state.json')
        if not state_file.exists():
            print("❌ 未找到账户状态")
            return

        with open(state_file, 'r', encoding='utf-8') as f:
            account = json.load(f)

        # 加载最新价格
        analysis_file = Path('data/daily_analysis_results.json')
        if not analysis_file.exists():
            print("❌ 未找到分析结果")
            return

        with open(analysis_file, 'r', encoding='utf-8') as f:
            analysis = json.load(f)

        signals = analysis['signals']

        # 1. 检查持仓风险
        print("📊 持仓风险检查:")
        print("-"*70)

        position_alerts = []

        for symbol, position in account['positions'].items():
            if symbol in signals:
                current_price = signals[symbol]['price']
                name = signals[symbol]['name']

                risk = self.check_position_risk(symbol, position, current_price)

                level_icon = {
                    'CRITICAL': '🔴',
                    'WARNING': '🟡',
                    'INFO': '🔵',
                    'OK': '🟢'
                }.get(risk['level'], '⚪')

                print(f"{level_icon} {name} ({symbol})")
                print(f"  成本: ¥{position['avg_cost']:.2f}")
                print(f"  现价: ¥{current_price:.2f}")
                print(f"  盈亏: {risk['pnl_pct']:+.2%} (¥{risk['pnl_amount']:+,.2f})")

                if risk['level'] != 'OK':
                    print(f"  ⚠️  {risk['message']}")
                    print(f"  💡 {risk['action']}")
                    position_alerts.append(risk)

                print()

        # 2. 检查组合风险
        print("📦 组合风险检查:")
        print("-"*70)

        portfolio = {
            'cash': account['cash'],
            'position_value': sum(
                pos['shares'] * signals[sym]['price']
                for sym, pos in account['positions'].items()
                if sym in signals
            ),
            'total_value': account['cash'] + sum(
                pos['shares'] * signals[sym]['price']
                for sym, pos in account['positions'].items()
                if sym in signals
            )
        }

        portfolio_alerts = self.check_portfolio_risk(portfolio)

        print(f"现金: ¥{portfolio['cash']:,.2f}")
        print(f"持仓: ¥{portfolio['position_value']:,.2f}")
        print(f"总资产: ¥{portfolio['total_value']:,.2f}")
        print(f"仓位: {portfolio['position_value']/portfolio['total_value']:.1%}")

        if portfolio_alerts:
            print()
            for alert in portfolio_alerts:
                level_icon = '🟡' if alert['level'] == 'WARNING' else '🔴'
                print(f"{level_icon} {alert['message']}")
                print(f"  💡 {alert['action']}")

        print()

        # 3. 检查市场风险
        print("🌍 市场风险检查:")
        print("-"*70)

        market = self.check_market_risk(signals)

        sentiment_icon = {
            'BULLISH': '📈',
            'BEARISH': '📉',
            'NEUTRAL': '➡️'
        }.get(market['sentiment'], '❓')

        print(f"{sentiment_icon} 市场情绪: {market['sentiment']}")
        print(f"  {market['message']}")
        print(f"  买入: {market['buy_count']} | 卖出: {market['sell_count']} | 持有: {market['hold_count']}")
        print()

        # 4. 风险汇总
        print("="*70)
        print("📋 风险汇总:")
        print("="*70)

        total_alerts = len(position_alerts) + len(portfolio_alerts)

        if total_alerts == 0:
            print("✅ 无风险告警，持仓健康")
        else:
            print(f"⚠️  发现{total_alerts}个风险告警:")

            for alert in position_alerts:
                level_icon = {
                    'CRITICAL': '🔴',
                    'WARNING': '🟡',
                    'INFO': '🔵'
                }.get(alert['level'], '⚪')
                print(f"{level_icon} [{alert['type']}] {alert['symbol']}: {alert['message']}")

            for alert in portfolio_alerts:
                level_icon = '🟡' if alert['level'] == 'WARNING' else '🔴'
                print(f"{level_icon} [{alert['type']}] {alert['message']}")

        print("="*70)

        # 保存风险报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio': portfolio,
            'position_alerts': position_alerts,
            'portfolio_alerts': portfolio_alerts,
            'market': market,
            'total_alerts': total_alerts
        }

        report_file = Path('data/risk_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n✅ 风险报告已保存: {report_file}")

        return report


def main():
    monitor = RiskMonitor()
    monitor.run_risk_check()


if __name__ == '__main__':
    main()
