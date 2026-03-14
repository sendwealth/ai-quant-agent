#!/usr/bin/env python3
"""
量化系统综合监控
整合数据健康、策略信号、风险监控
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd):
    """运行命令并返回输出"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    return result.stdout, result.returncode

def check_data_health():
    """检查数据健康"""
    output, code = run_command('python3 scripts/check_data_health.py')
    if code == 0:
        data = json.loads(output)
        return {
            'status': 'ok',
            'message': data['message'],
            'latest_date': data.get('latest_date', 'N/A')
        }
    else:
        try:
            data = json.loads(output)
            return {
                'status': 'critical',
                'message': data['message'],
                'latest_date': data.get('latest_date', 'N/A')
            }
        except:
            return {
                'status': 'error',
                'message': '检查失败',
                'latest_date': 'N/A'
            }

def check_strategy_signals():
    """检查策略信号"""
    analysis_file = Path(__file__).parent.parent / 'data' / 'daily_analysis_results.json'
    if not analysis_file.exists():
        return {'status': 'error', 'message': '未找到分析结果'}

    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    signals = analysis['signals']
    buy_count = sum(1 for s in signals.values() if s['signal'] == 'BUY')
    sell_count = sum(1 for s in signals.values() if s['signal'] == 'SELL')

    return {
        'status': 'ok',
        'buy_count': buy_count,
        'sell_count': sell_count,
        'hold_count': len(signals) - buy_count - sell_count,
        'signals': signals
    }

def check_portfolio():
    """检查账户"""
    state_file = Path(__file__).parent.parent / 'data' / 'paper_trading_state.json'
    if not state_file.exists():
        return {'status': 'empty', 'message': '未找到账户'}

    with open(state_file, 'r', encoding='utf-8') as f:
        account = json.load(f)

    # 获取最新价格
    analysis_file = Path(__file__).parent.parent / 'data' / 'daily_analysis_results.json'
    if not analysis_file.exists():
        return {'status': 'error', 'message': '未找到价格数据'}

    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    signals = analysis['signals']

    # 计算当前资产
    position_value = sum(
        pos['shares'] * signals[sym]['price']
        for sym, pos in account['positions'].items()
        if sym in signals
    )

    total_value = account['cash'] + position_value
    return_pct = (total_value - account['initial_capital']) / account['initial_capital']

    return {
        'status': 'ok',
        'cash': account['cash'],
        'position_value': position_value,
        'total_value': total_value,
        'return_pct': return_pct,
        'positions': account['positions'],
        'trade_count': len(account['trades'])
    }

def check_risk():
    """检查风险"""
    report_file = Path(__file__).parent.parent / 'data' / 'risk_report.json'
    if not report_file.exists():
        return {'status': 'no_report', 'message': '未找到风险报告，请运行 risk_monitor.py'}

    with open(report_file, 'r', encoding='utf-8') as f:
        report = json.load(f)

    return {
        'status': 'ok' if report['total_alerts'] == 0 else 'warning',
        'total_alerts': report['total_alerts'],
        'position_alerts': len(report['position_alerts']),
        'portfolio_alerts': len(report['portfolio_alerts']),
        'market_sentiment': report['market']['sentiment']
    }

def main():
    """主函数"""
    print("="*70)
    print("📊 量化系统综合监控")
    print("="*70)
    print(f"监控时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. 数据健康
    print("1️⃣ 数据健康:")
    print("-"*70)
    data_health = check_data_health()

    status_icon = {
        'ok': '🟢',
        'warning': '🟡',
        'error': '🔴',
        'critical': '🔴'
    }.get(data_health['status'], '⚪')

    print(f"{status_icon} {data_health['message']}")
    if 'latest_date' in data_health:
        print(f"  最新数据: {data_health['latest_date']}")
    print()

    # 2. 策略信号
    print("2️⃣ 策略信号:")
    print("-"*70)
    signals = check_strategy_signals()

    if signals['status'] == 'ok':
        print(f"买入: {signals['buy_count']} | 卖出: {signals['sell_count']} | 持有: {signals['hold_count']}")
        print()

        for code, data in signals['signals'].items():
            signal_icon = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡'
            }.get(data['signal'], '⚪')

            print(f"{signal_icon} {data['name']} ({code}): {data['signal']} @ ¥{data['price']}")
    else:
        print(f"❌ {signals['message']}")
    print()

    # 3. 账户状态
    print("3️⃣ 账户状态:")
    print("-"*70)
    portfolio = check_portfolio()

    if portfolio['status'] == 'ok':
        print(f"现金: ¥{portfolio['cash']:,.2f}")
        print(f"持仓: ¥{portfolio['position_value']:,.2f}")
        print(f"总资产: ¥{portfolio['total_value']:,.2f}")
        print(f"收益率: {portfolio['return_pct']:+.2%}")
        print(f"交易次数: {portfolio['trade_count']}")

        if portfolio['positions']:
            print(f"\n持仓明细:")
            for symbol, pos in portfolio['positions'].items():
                name = signals['signals'].get(symbol, {}).get('name', symbol)
                current_price = signals['signals'].get(symbol, {}).get('price', 0)
                pnl_pct = (current_price - pos['avg_cost']) / pos['avg_cost'] if pos['avg_cost'] > 0 else 0
                print(f"  {name}: {pos['shares']}股 @ ¥{pos['avg_cost']:.2f} ({pnl_pct:+.2%})")
    elif portfolio['status'] == 'empty':
        print("📝 账户未创建")
    else:
        print(f"❌ {portfolio['message']}")
    print()

    # 4. 风险监控
    print("4️⃣ 风险监控:")
    print("-"*70)
    risk = check_risk()

    if risk['status'] == 'ok':
        print("✅ 无风险告警")
    elif risk['status'] == 'warning':
        print(f"⚠️  发现{risk['total_alerts']}个风险告警")
        print(f"  持仓风险: {risk['position_alerts']}个")
        print(f"  组合风险: {risk['portfolio_alerts']}个")
    else:
        print(f"⚠️  {risk['message']}")

    if 'market_sentiment' in risk:
        sentiment_icon = {
            'BULLISH': '📈',
            'BEARISH': '📉',
            'NEUTRAL': '➡️'
        }.get(risk['market_sentiment'], '❓')
        print(f"  市场情绪: {sentiment_icon} {risk['market_sentiment']}")
    print()

    # 5. 系统状态汇总
    print("="*70)
    print("📋 系统状态汇总:")
    print("="*70)

    issues = []

    if data_health['status'] in ['error', 'critical']:
        issues.append(f"数据健康: {data_health['message']}")

    if portfolio['status'] == 'ok' and portfolio['return_pct'] < -0.05:
        issues.append(f"账户亏损: {portfolio['return_pct']:.2%}")

    if risk['status'] == 'warning':
        issues.append(f"风险告警: {risk['total_alerts']}个")

    if issues:
        print("⚠️  需要关注:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("✅ 系统运行正常，无异常")

    print("="*70)

    # 返回状态
    return {
        'data_health': data_health,
        'signals': signals,
        'portfolio': portfolio,
        'risk': risk,
        'issues': issues
    }


if __name__ == '__main__':
    result = main()

    # 保存监控结果
    output_file = Path(__file__).parent.parent / 'data' / 'monitor_summary.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'result': result
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 监控结果已保存: {output_file}")
