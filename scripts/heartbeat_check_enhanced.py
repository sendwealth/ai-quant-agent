#!/usr/bin/env python3
"""
增强版心跳检查脚本

功能：
1. 数据健康检查
2. 自动触发数据更新（如果过期）
3. 账户状态检查
4. 风险告警
5. 邮件通知（可选）
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_data_update():
    """运行容错数据更新"""
    try:
        proc = subprocess.run(
            ['python3', 'scripts/data_updater_robust.py'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=120  # 2分钟超时
        )

        if proc.returncode == 0:
            result = json.loads(proc.stdout)
            return True, result
        else:
            return False, {'error': proc.stderr}

    except subprocess.TimeoutExpired:
        return False, {'error': '数据更新超时（2分钟）'}
    except Exception as e:
        return False, {'error': str(e)}


def run_quick_check():
    """快速检查"""
    result = {
        'timestamp': datetime.now().isoformat(),
        'status': 'ok',
        'alerts': [],
        'summary': {}
    }

    # 1. 数据健康检查
    try:
        proc = subprocess.run(
            ['python3', 'scripts/check_data_health.py'],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=30
        )

        if proc.returncode == 0:
            data = json.loads(proc.stdout)
            result['summary']['data_health'] = 'ok'
            result['summary']['latest_date'] = data.get('latest_date', 'N/A')
        else:
            try:
                data = json.loads(proc.stdout)
                stale_days = 0

                # 提取过期天数
                msg = data.get('message', '')
                if '过期' in msg:
                    import re
                    match = re.search(r'(\d+)天', msg)
                    if match:
                        stale_days = int(match.group(1))

                # 如果数据过期>1天，自动触发更新
                if stale_days > 1:
                    result['alerts'].append(f"数据过期{stale_days}天，触发自动更新")
                    result['summary']['data_health'] = 'updating'

                    # 运行数据更新
                    success, update_result = run_data_update()
                    if success:
                        result['summary']['data_health'] = 'updated'
                        result['summary']['update_result'] = update_result
                    else:
                        result['status'] = 'error'
                        result['alerts'].append(f"数据更新失败: {update_result.get('error', '未知错误')}")
                else:
                    result['status'] = 'warning'
                    result['alerts'].append(f"数据过期: {msg}")
                    result['summary']['data_health'] = 'warning'

            except:
                result['status'] = 'error'
                result['alerts'].append("数据检查失败")

    except Exception as e:
        result['status'] = 'error'
        result['alerts'].append(f"数据检查异常: {str(e)[:50]}")

    # 2. 账户状态检查
    try:
        state_file = PROJECT_ROOT / 'data' / 'paper_trading_state.json'
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                account = json.load(f)

            # 获取价格
            analysis_file = PROJECT_ROOT / 'data' / 'daily_analysis_results.json'
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
                    result['alerts'].append(f"账户亏损{return_pct:.1%}")
                elif return_pct < -0.10:
                    result['status'] = 'error'
                    result['alerts'].append(f"账户严重亏损{return_pct:.1%}")

    except Exception as e:
        result['alerts'].append(f"账户检查失败: {str(e)[:50]}")

    # 3. 风险检查
    try:
        risk_file = PROJECT_ROOT / 'data' / 'reports' / 'risk_report_latest.json'
        if risk_file.exists():
            with open(risk_file, 'r', encoding='utf-8') as f:
                risk = json.load(f)

            if risk.get('risk_level') == 'high':
                result['status'] = 'warning'
                result['alerts'].append(f"高风险: {risk.get('message', '')}")

    except Exception:
        pass

    return result


def main():
    """主函数"""
    result = run_quick_check()

    # 输出结果
    if result['status'] == 'ok':
        if result['summary'].get('data_health') == 'updated':
            print("✅ 量化系统正常（数据已自动更新）")
        else:
            print("✅ 量化系统正常")
    elif result['status'] == 'warning':
        print(f"⚠️ 量化系统警告:")
        for alert in result['alerts']:
            print(f"  - {alert}")
    else:
        print(f"❌ 量化系统异常:")
        for alert in result['alerts']:
            print(f"  - {alert}")

    # 输出详细信息
    if result['summary']:
        print("\n📊 系统状态:")
        if 'latest_date' in result['summary']:
            print(f"  数据: {result['summary']['latest_date']}")
        if 'portfolio' in result['summary']:
            portfolio = result['summary']['portfolio']
            print(f"  账户: ¥{portfolio['total_value']:.2f} ({portfolio['return_pct']:+.2%})")

    # 返回状态码
    return {
        'ok': 0,
        'warning': 1,
        'error': 2
    }.get(result['status'], 2)


if __name__ == '__main__':
    exit(main())
