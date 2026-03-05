#!/usr/bin/env python3
"""
数据验证工具
============
验证股票数据质量和完整性
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json

def validate_stock_data(csv_file: Path) -> dict:
    """验证单个股票数据"""
    result = {
        'file': csv_file.name,
        'status': 'ok',
        'issues': [],
        'stats': {}
    }

    try:
        df = pd.read_csv(csv_file)

        # 1. 检查数据量
        if len(df) < 100:
            result['issues'].append(f"数据量不足: {len(df)}行")
            result['status'] = 'error'

        # 2. 检查列
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            result['issues'].append(f"缺少列: {missing}")
            result['status'] = 'error'

        # 3. 检查空值
        null_count = df[required_cols].isnull().sum().sum()
        if null_count > 0:
            result['issues'].append(f"空值: {null_count}个")
            result['status'] = 'warning'

        # 4. 检查价格合理性
        if (df['close'] <= 0).any():
            result['issues'].append("发现无效价格（≤0）")
            result['status'] = 'error'

        if (df['high'] < df['low']).any():
            result['issues'].append("最高价<最低价")
            result['status'] = 'error'

        # 5. 检查时间序列
        if 'datetime' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')

                # 检查时间间隔
                time_diff = df['datetime'].diff()
                irregular = time_diff[time_diff > timedelta(days=10)]
                if len(irregular) > 0:
                    result['issues'].append(f"时间序列不规则: {len(irregular)}处")
                    result['status'] = 'warning'

                # 检查数据新鲜度
                latest = df['datetime'].max()
                age = datetime.now() - latest.to_pydatetime()
                if age.days > 7:
                    result['issues'].append(f"数据过期: {age.days}天前")
                    result['status'] = 'warning'

            except Exception as e:
                result['issues'].append(f"时间解析失败: {e}")
                result['status'] = 'error'

        # 统计信息
        result['stats'] = {
            'rows': len(df),
            'null_count': int(null_count),
            'date_range': f"{df['datetime'].min()} to {df['datetime'].max()}" if 'datetime' in df.columns else 'N/A'
        }

    except Exception as e:
        result['status'] = 'error'
        result['issues'].append(f"读取失败: {e}")

    return result

def main():
    """主函数"""
    print("="*70)
    print("🔍 数据验证工具")
    print("="*70)
    print(f"验证时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    data_dir = Path('data')
    stock_files = sorted(data_dir.glob('real_*.csv'))

    if not stock_files:
        print("❌ 未找到股票数据文件")
        return

    print(f"发现{len(stock_files)}个数据文件\n")

    results = []
    for csv_file in stock_files:
        result = validate_stock_data(csv_file)
        results.append(result)

        # 显示状态
        status_icon = {
            'ok': '✅',
            'warning': '⚠️',
            'error': '❌'
        }[result['status']]

        print(f"{status_icon} {csv_file.stem}")
        print(f"  数据量: {result['stats'].get('rows', 0)}行")
        print(f"  时间范围: {result['stats'].get('date_range', 'N/A')}")

        if result['issues']:
            print(f"  问题:")
            for issue in result['issues']:
                print(f"    - {issue}")

        print()

    # 汇总
    print("="*70)
    print("📊 验证汇总")
    print("="*70)

    ok_count = sum(1 for r in results if r['status'] == 'ok')
    warning_count = sum(1 for r in results if r['status'] == 'warning')
    error_count = sum(1 for r in results if r['status'] == 'error')

    print(f"✅ 正常: {ok_count}个")
    print(f"⚠️  警告: {warning_count}个")
    print(f"❌ 错误: {error_count}个")
    print("="*70)

    if error_count > 0:
        print("\n⚠️  发现数据错误，建议重新获取数据:")
        print("python3 examples/fetch_tushare_auto.py")
    elif warning_count > 0:
        print("\n⚠️  数据有警告，建议检查")
    else:
        print("\n✅ 所有数据验证通过！")

    # 保存验证结果
    report_file = data_dir / 'validation_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            'time': datetime.now().isoformat(),
            'summary': {
                'total': len(results),
                'ok': ok_count,
                'warning': warning_count,
                'error': error_count
            },
            'details': results
        }, f, ensure_ascii=False, indent=2)

    print(f"\n验证报告已保存: {report_file}")

if __name__ == '__main__':
    main()
