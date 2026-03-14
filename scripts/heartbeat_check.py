#!/usr/bin/env python3
"""
心跳检查脚本
用于OpenClaw心跳系统调用
返回简洁的状态报告
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_quick_check():
    """快速检查"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'status': 'ok',
        'alerts': [],
        'summary': {}
    }

    # 1. 数据健康
    try:
        proc = subprocess.run(
            ['python3', 'scripts/check_data_health.py'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )

        output = proc.stdout
        code = proc.returncode

        if code == 0:
            data = json.loads(output)
            result['summary']['data_health'] = 'ok'
            result['summary']['latest_date'] = data.get('latest_date', 'N/A')
        else:
            try:
                data = json.loads(output)
                result['status'] = 'warning'
                result['alerts'].append(f"数据过期: {data['message']}")
                result['summary']['data_health'] = 'critical'
            except:
                result['status'] = 'error'
                result['alerts'].append("数据检查失败")
    except Exception as e:
        result['status'] = 'error'
        result['alerts'].append(f"数据检查异常: {str(e)[:50]}")

    # 2. 账户状态
    try:
        state_file = Path(__file__).parent.parent / 'data' / 'paper_trading_state.json'
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                account = json.load(f)

            # 获取价格
            analysis_file = Path(__file__).parent.parent / 'data' / 'daily_analysis_results.json'
            if analysis_file.exists():
                with open(analysis_file, 'r', encoding='utf-8') as f:
                    analysis = json.load(f)

                signals = analysis['signals']
                position_value = sum(
                    pos['shares'] * signals[sym]['price']
                    for sym, pos in account['positions'].items()
                    if sym in signals
                )
                total_value = account['cash'] + position_value
                return_pct = (total_value - account['initial_capital']) / account['initial_capital']

                result['summary']['portfolio'] = {
                    'total_value': total_value,
                    'return_pct': return_pct,
                    'position_count': len(account['positions'])
                }

                # 大额亏损告警
                if return_pct < -0.05:
                    result['status'] = 'warning'
                    result['alerts'].append(f"账户亏损{return_pct:.2%}")
    except Exception as e:
        result['alerts'].append(f"账户检查异常: {str(e)[:50]}")

    # 3. 风险告警
    try:
        risk_file = Path(__file__).parent.parent / 'data' / 'risk_report.json'
        if risk_file.exists():
            with open(risk_file, 'r', encoding='utf-8') as f:
                risk = json.load(f)

            if risk['total_alerts'] > 0:
                result['status'] = 'warning'
                for alert in risk['position_alerts'][:3]:  # 最多显示3个
                    result['alerts'].append(f"{alert['symbol']}: {alert['message'][:50]}")

            result['summary']['risk_alerts'] = risk['total_alerts']
    except Exception as e:
        result['alerts'].append(f"风险检查异常: {str(e)[:50]}")

    return result


def main():
    """主函数"""
    result = run_quick_check()

    # 输出结果
    if result['status'] == 'ok':
        print("✅ 量化系统正常")
        if 'portfolio' in result['summary']:
            portfolio = result['summary']['portfolio']
            print(f"  总资产: ¥{portfolio['total_value']:,.2f} ({portfolio['return_pct']:+.2%})")
        if 'latest_date' in result['summary']:
            print(f"  数据: {result['summary']['latest_date']}")
    elif result['status'] == 'warning':
        print(f"⚠️  量化系统告警:")
        for alert in result['alerts']:
            print(f"  - {alert}")
    else:
        print(f"❌ 量化系统异常:")
        for alert in result['alerts']:
            print(f"  - {alert}")

    # 保存结果
    output_file = Path(__file__).parent.parent / 'data' / 'heartbeat_status.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 返回状态码
    return {
        'ok': 0,
        'warning': 1,
        'error': 2
    }.get(result['status'], 2)


if __name__ == '__main__':
    exit(main())
