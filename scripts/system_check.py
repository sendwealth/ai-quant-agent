#!/usr/bin/env python3
"""
系统状态检查脚本
快速检查量化系统的健康状态
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_env_file():
    """检查 .env 文件"""
    env_path = PROJECT_ROOT / '.env'
    if not env_path.exists():
        return {
            'status': 'error',
            'message': '.env 文件不存在',
            'action': '创建 .env 文件并配置环境变量'
        }

    required_vars = ['TUSHARE_TOKEN', 'EMAIL_PASSWORD']
    missing_vars = []

    with open(env_path, 'r') as f:
        content = f.read()
        for var in required_vars:
            if var not in content or f'{var}=\n' in content or f'{var}=""\n' in content:
                missing_vars.append(var)

    if missing_vars:
        return {
            'status': 'warning',
            'message': f'缺少环境变量: {", ".join(missing_vars)}',
            'action': f'在 .env 中配置: {", ".join(missing_vars)}'
        }

    return {
        'status': 'ok',
        'message': '.env 文件配置完整'
    }


def check_config_file():
    """检查配置文件"""
    config_path = PROJECT_ROOT / 'config' / 'data_sources.yaml'
    if not config_path.exists():
        return {
            'status': 'error',
            'message': '配置文件不存在',
            'action': '创建 config/data_sources.yaml'
        }

    return {
        'status': 'ok',
        'message': '配置文件存在'
    }


def check_data_files():
    """检查数据文件"""
    data_dir = PROJECT_ROOT / 'data'

    required_stocks = ['300750', '002475', '601318', '600276']
    missing_files = []

    for code in required_stocks:
        csv_files = list(data_dir.glob(f'{code}_*.csv')) + list(data_dir.glob(f'real_{code}.csv'))
        if not csv_files:
            missing_files.append(code)

    if missing_files:
        return {
            'status': 'warning',
            'message': f'缺少数据文件: {", ".join(missing_files)}',
            'action': '运行 python3 scripts/data_updater_robust.py'
        }

    return {
        'status': 'ok',
        'message': f'数据文件完整 ({len(required_stocks)} 只股票)'
    }


def check_data_freshness():
    """检查数据新鲜度"""
    data_dir = PROJECT_ROOT / 'data'

    # 检查第一个数据文件
    csv_files = list(data_dir.glob('300750_*.csv'))
    if not csv_files:
        return {
            'status': 'error',
            'message': '无数据文件',
            'action': '运行 python3 scripts/data_updater_robust.py'
        }

    try:
        import pandas as pd
        df = pd.read_csv(csv_files[0])
        latest_date = pd.to_datetime(df['date'].iloc[-1])
        age_days = (datetime.now() - latest_date).days

        if age_days > 7:
            return {
                'status': 'error',
                'message': f'数据已过期 {age_days} 天',
                'action': '运行 python3 scripts/data_updater_robust.py'
            }
        elif age_days > 3:
            return {
                'status': 'warning',
                'message': f'数据 {age_days} 天前更新',
                'action': '建议更新数据'
            }
        else:
            return {
                'status': 'ok',
                'message': f'数据新鲜（{age_days} 天前）'
            }
    except Exception as e:
        return {
            'status': 'error',
            'message': f'数据检查失败: {e}',
            'action': '检查数据文件格式'
        }


def check_dependencies():
    """检查依赖包"""
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yaml': 'pyyaml',
        'akshare': 'akshare',
        'tushare': 'tushare'
    }

    missing_packages = []

    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)

    if missing_packages:
        return {
            'status': 'error',
            'message': f'缺少依赖包: {", ".join(missing_packages)}',
            'action': f'pip install {" ".join(missing_packages)}'
        }

    return {
        'status': 'ok',
        'message': '依赖包完整'
    }


def check_directory_structure():
    """检查目录结构"""
    required_dirs = ['config', 'data', 'scripts', 'docs', 'logs']

    missing_dirs = []
    for dir_name in required_dirs:
        if not (PROJECT_ROOT / dir_name).exists():
            missing_dirs.append(dir_name)

    if missing_dirs:
        return {
            'status': 'warning',
            'message': f'缺少目录: {", ".join(missing_dirs)}',
            'action': f'创建目录: mkdir -p {" ".join(missing_dirs)}'
        }

    return {
        'status': 'ok',
        'message': '目录结构完整'
    }


def run_system_check():
    """运行系统检查"""
    checks = {
        '环境变量': check_env_file(),
        '配置文件': check_config_file(),
        '数据文件': check_data_files(),
        '数据新鲜度': check_data_freshness(),
        '依赖包': check_dependencies(),
        '目录结构': check_directory_structure()
    }

    # 计算总体状态
    has_error = any(c['status'] == 'error' for c in checks.values())
    has_warning = any(c['status'] == 'warning' for c in checks.values())

    if has_error:
        overall_status = 'error'
    elif has_warning:
        overall_status = 'warning'
    else:
        overall_status = 'ok'

    return {
        'timestamp': datetime.now().isoformat(),
        'overall_status': overall_status,
        'checks': checks
    }


def print_report(result):
    """打印报告"""
    print("=" * 60)
    print("🔍 量化系统健康检查报告")
    print("=" * 60)
    print(f"检查时间: {result['timestamp']}")
    print(f"总体状态: {result['overall_status'].upper()}")
    print("=" * 60)

    status_emoji = {
        'ok': '✅',
        'warning': '⚠️',
        'error': '❌'
    }

    for name, check in result['checks'].items():
        emoji = status_emoji.get(check['status'], '❓')
        print(f"{emoji} {name}: {check['message']}")
        if check['status'] != 'ok' and 'action' in check:
            print(f"   💡 建议: {check['action']}")

    print("=" * 60)

    if result['overall_status'] == 'ok':
        print("✅ 系统状态良好，可以正常运行")
    elif result['overall_status'] == 'warning':
        print("⚠️ 系统存在警告，建议优化")
    else:
        print("❌ 系统存在问题，需要修复")

    print("=" * 60)


def main():
    """主函数"""
    result = run_system_check()
    print_report(result)

    # 保存 JSON 结果
    json_output = json.dumps(result, ensure_ascii=False, indent=2)
    print("\n📊 JSON 输出:")
    print(json_output)

    # 返回状态码
    return 0 if result['overall_status'] == 'ok' else 1


if __name__ == '__main__':
    exit(main())
