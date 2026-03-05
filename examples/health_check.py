#!/usr/bin/env python3
"""
系统健康检查工具
================
全面检查系统状态，生成报告
"""
import json
from pathlib import Path
from datetime import datetime
import sys

def check_data_files():
    """检查数据文件"""
    print("📊 检查数据文件...")
    data_dir = Path('data')
    issues = []

    # 检查股票数据
    stock_files = list(data_dir.glob('real_*.csv'))
    if len(stock_files) < 4:
        issues.append(f"股票数据不足: {len(stock_files)}/4")

    # 检查持仓文件
    portfolio_file = data_dir / 'auto_portfolio.json'
    if not portfolio_file.exists():
        issues.append("持仓文件不存在")

    # 检查配置文件
    config_files = [
        'smart_screening_v2.json',
        'weight_optimization_results.json',
        'param_optimization_results.json'
    ]
    for config_file in config_files:
        if not (data_dir / config_file).exists():
            issues.append(f"配置文件缺失: {config_file}")

    if issues:
        for issue in issues:
            print(f"  ❌ {issue}")
        return False
    else:
        print("  ✅ 数据文件完整")
        return True

def check_cron_jobs():
    """检查cron任务"""
    print("\n⏰ 检查cron任务...")
    import subprocess

    try:
        result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)

        if result.returncode != 0:
            print("  ❌ 未设置cron任务")
            return False

        cron_content = result.stdout

        # 检查每日运行
        if 'daily_run.sh' not in cron_content:
            print("  ❌ 未设置每日自动运行")
            return False

        # 检查每周更新
        if 'fetch_tushare_auto.py' not in cron_content:
            print("  ⚠️  未设置每周数据更新")

        print("  ✅ Cron任务正常")
        print(f"\n当前cron任务:")
        for line in cron_content.split('\n'):
            if line and not line.startswith('#'):
                print(f"    {line}")

        return True

    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

def check_logs():
    """检查日志"""
    print("\n📝 检查日志...")
    log_dir = Path('logs')
    issues = []

    # 检查日志目录
    if not log_dir.exists():
        print("  ⚠️  日志目录不存在")
        return True  # 不影响运行

    # 检查最近的日志
    today = datetime.now().strftime('%Y%m%d')
    system_log = log_dir / f'system_{today}.log'
    auto_trading_log = log_dir / 'auto_trading.log'

    if not system_log.exists():
        issues.append("今日系统日志不存在")

    if auto_trading_log.exists():
        # 检查最近的错误
        try:
            with open(auto_trading_log, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-100:]  # 最后100行

            errors = [line for line in lines if 'ERROR' in line or '❌' in line]
            if errors:
                issues.append(f"发现{len(errors)}个错误")

                # 检查错误文件
                error_file = log_dir / 'errors.json'
                if error_file.exists():
                    with open(error_file, 'r', encoding='utf-8') as f:
                        error_data = json.load(f)
                    print(f"  ⚠️  累计错误: {len(error_data)}次")

        except Exception as e:
            issues.append(f"读取日志失败: {e}")

    if issues:
        for issue in issues:
            print(f"  ⚠️  {issue}")
    else:
        print("  ✅ 日志正常")

    return len([i for i in issues if 'ERROR' in i or '❌' in i]) == 0

def check_backups():
    """检查备份"""
    print("\n💾 检查备份...")
    backup_dir = Path('backups')

    if not backup_dir.exists():
        print("  ⚠️  备份目录不存在（首次运行时正常）")
        return True

    backup_files = list(backup_dir.glob('portfolio_*.json'))

    if len(backup_files) == 0:
        print("  ⚠️  暂无备份文件")
    else:
        print(f"  ✅ 备份文件: {len(backup_files)}个")

        # 检查最新备份
        latest = max(backup_files, key=lambda x: x.stat().st_mtime)
        mtime = datetime.fromtimestamp(latest.stat().st_mtime)
        age = datetime.now() - mtime

        if age.days > 7:
            print(f"  ⚠️  最新备份已{age.days}天前")
        else:
            print(f"  ✅ 最新备份: {mtime.strftime('%Y-%m-%d %H:%M')}")

    return True

def check_portfolio():
    """检查持仓"""
    print("\n💰 检查持仓...")
    portfolio_file = Path('data/auto_portfolio.json')

    if not portfolio_file.exists():
        print("  ⚠️  持仓文件不存在（首次运行时正常）")
        return True

    try:
        with open(portfolio_file, 'r', encoding='utf-8') as f:
            portfolio = json.load(f)

        cash = portfolio.get('cash', 0)
        positions = portfolio.get('positions', {})
        trades = portfolio.get('trades', [])
        update_time = portfolio.get('update_time', 'N/A')

        print(f"  现金: {cash:,.2f}元")
        print(f"  持仓: {len(positions)}只股票")
        print(f"  交易: {len(trades)}次")
        print(f"  更新: {update_time}")

        # 检查更新时间
        if update_time != 'N/A':
            try:
                last_update = datetime.fromisoformat(update_time)
                age = datetime.now() - last_update

                if age.days > 1:
                    print(f"  ⚠️  数据已{age.days}天未更新")
                else:
                    print(f"  ✅ 数据最新")
            except:
                pass

        return True

    except Exception as e:
        print(f"  ❌ 读取失败: {e}")
        return False

def check_disk_space():
    """检查磁盘空间"""
    print("\n💾 检查磁盘空间...")
    import shutil

    try:
        total, used, free = shutil.disk_usage('/')
        free_gb = free // (2**30)

        if free_gb < 5:
            print(f"  ❌ 磁盘空间不足: {free_gb}GB")
            return False
        elif free_gb < 20:
            print(f"  ⚠️  磁盘空间较少: {free_gb}GB")
        else:
            print(f"  ✅ 磁盘空间充足: {free_gb}GB")

        return True

    except Exception as e:
        print(f"  ⚠️  检查失败: {e}")
        return True  # 不影响运行

def check_python_version():
    """检查Python版本"""
    print("\n🐍 检查Python版本...")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"  ❌ Python版本过低: {version.major}.{version.minor}")
        return False

    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """检查依赖包"""
    print("\n📦 检查依赖包...")

    required = ['pandas', 'numpy']
    missing = []

    for package in required:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} 未安装")
            missing.append(package)

    if missing:
        print(f"\n安装缺失的包: pip install {' '.join(missing)}")
        return False

    return True

def generate_health_report():
    """生成健康报告"""
    print("="*70)
    print("🏥 系统健康检查")
    print("="*70)
    print(f"检查时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {
        'Python版本': check_python_version(),
        '依赖包': check_dependencies(),
        '磁盘空间': check_disk_space(),
        '数据文件': check_data_files(),
        'Cron任务': check_cron_jobs(),
        '日志': check_logs(),
        '备份': check_backups(),
        '持仓': check_portfolio()
    }

    print("\n" + "="*70)
    print("📊 检查结果汇总")
    print("="*70)

    all_passed = True
    for check, passed in results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
        if not passed:
            all_passed = False

    print("="*70)

    if all_passed:
        print("\n✅ 所有检查通过！系统健康！")
        return 0
    else:
        print("\n⚠️  部分检查未通过，请修复后再运行")
        return 1

if __name__ == '__main__':
    exit_code = generate_health_report()
    sys.exit(exit_code)
