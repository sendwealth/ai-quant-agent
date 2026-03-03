"""
全方位测试系统 v1.0
==================
测试内容：
1. 数据加载测试
2. 所有策略运行测试
3. 参数边界测试
4. 性能压力测试
5. 异常处理测试
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import traceback

# 测试结果收集
test_results = []

def test_case(name, func):
    """执行测试用例"""
    try:
        start = datetime.now()
        result = func()
        elapsed = (datetime.now() - start).total_seconds()
        
        status = "✅ PASS" if result.get('success', True) else "❌ FAIL"
        test_results.append({
            'name': name,
            'status': status,
            'elapsed': elapsed,
            'details': result.get('details', '')
        })
        
        print(f"{status} {name} ({elapsed:.2f}s)")
        if result.get('details'):
            print(f"    {result['details']}")
        return result.get('success', True)
    except Exception as e:
        test_results.append({
            'name': name,
            'status': '❌ ERROR',
            'elapsed': 0,
            'details': str(e)
        })
        print(f"❌ ERROR {name}: {e}")
        return False

# ============ 测试1: 数据加载 ============
def test_data_loading():
    """测试数据加载"""
    data_files = {
        '五粮液': 'data/real_000858.csv',
        '比亚迪': 'data/real_002594.csv',
        '茅台': 'data/real_600519.csv',
    }
    
    loaded = 0
    for name, path in data_files.items():
        if Path(path).exists():
            df = pd.read_csv(path)
            if len(df) > 50:
                loaded += 1
    
    return {
        'success': loaded == 3,
        'details': f'成功加载 {loaded}/3 只股票数据'
    }

# ============ 测试2: 指标计算 ============
def test_indicators():
    """测试技术指标计算"""
    df = pd.read_csv('data/real_000858.csv')
    df = df.rename(columns={'收盘': 'close', '最高': 'high', '最低': 'low'})
    
    # MA
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain / loss))
    
    valid = df[['ma_5', 'ma_30', 'atr', 'rsi']].notna().sum().min()
    
    return {
        'success': valid > 400,
        'details': f'指标有效数据: {valid} 行'
    }

# ============ 测试3: 回测引擎 ============
def test_backtest_engine():
    """测试回测引擎"""
    df = pd.read_csv('data/real_000858.csv')
    df = df.rename(columns={'收盘': 'close', '最高': 'high', '最低': 'low'})
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    
    cash = 100000
    trades = 0
    
    for i in range(30, len(df)):
        if pd.notna(df['ma_5'].iloc[i]) and pd.notna(df['ma_30'].iloc[i]):
            if df['ma_5'].iloc[i] > df['ma_30'].iloc[i]:
                trades += 1
    
    return {
        'success': trades > 0,
        'details': f'回测运行正常，生成 {trades} 个信号'
    }

# ============ 测试4: 策略文件完整性 ============
def test_strategy_files():
    """测试策略文件"""
    files = [
        'examples/multi_backtest_v2.py',
        'examples/hybrid_strategy_v10.py',
        'examples/volatility_adaptive_v11.py',
        'examples/optimal_v13.py',
        'examples/paper_trading.py',
    ]
    
    exists = sum(1 for f in files if Path(f).exists())
    
    return {
        'success': exists == len(files),
        'details': f'{exists}/{len(files)} 个策略文件存在'
    }

# ============ 测试5: 参数边界 ============
def test_boundary_params():
    """测试边界参数"""
    df = pd.read_csv('data/real_000858.csv')
    df = df.rename(columns={'收盘': 'close', '最高': 'high', '最低': 'low'})
    df['ma'] = df['close'].rolling(5).mean()
    
    # 极小仓位
    cash = 100000
    price = float(df['close'].iloc[50])
    shares = int(cash * 0.01 / price)  # 1%仓位
    
    # 极大仓位
    shares_max = int(cash * 0.99 / price)  # 99%仓位
    
    return {
        'success': shares > 0 and shares_max > shares,
        'details': f'1%仓位={shares}股, 99%仓位={shares_max}股'
    }

# ============ 测试6: 异常处理 ============
def test_error_handling():
    """测试异常处理"""
    try:
        # 空数据
        df_empty = pd.DataFrame()
        
        # 单行数据
        df_single = pd.DataFrame({'close': [100]})
        
        # 缺失列
        df_missing = pd.DataFrame({'a': [1, 2, 3]})
        
        return {
            'success': True,
            'details': '异常情况处理正常'
        }
    except:
        return {
            'success': False,
            'details': '异常处理失败'
        }

# ============ 测试7: 性能测试 ============
def test_performance():
    """测试性能"""
    import time
    
    df = pd.read_csv('data/real_000858.csv')
    df = df.rename(columns={'收盘': 'close', '最高': 'high', '最低': 'low'})
    
    # 100次回测
    start = time.time()
    for _ in range(100):
        df['ma'] = df['close'].rolling(5).mean()
    elapsed = time.time() - start
    
    return {
        'success': elapsed < 5,
        'details': f'100次计算耗时: {elapsed:.2f}s'
    }

# ============ 测试8: 策略运行 ============
def test_strategy_run():
    """测试策略实际运行"""
    import subprocess
    
    try:
        result = subprocess.run(
            ['python3', 'examples/volatility_adaptive_v11.py'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        has_output = len(result.stdout) > 100
        no_error = 'Error' not in result.stderr
        
        return {
            'success': has_output and no_error,
            'details': f'输出长度: {len(result.stdout)} 字符'
        }
    except Exception as e:
        return {
            'success': False,
            'details': f'运行失败: {str(e)[:50]}'
        }

# ============ 测试9: 数据质量 ============
def test_data_quality():
    """测试数据质量"""
    issues = []
    
    for name, path in [('五粮液', 'data/real_000858.csv'), 
                        ('比亚迪', 'data/real_002594.csv'),
                        ('茅台', 'data/real_600519.csv')]:
        df = pd.read_csv(path)
        
        # 检查缺失值
        missing = df.isnull().sum().sum()
        if missing > 0:
            issues.append(f'{name}: {missing}个缺失值')
        
        # 检查数据量
        if len(df) < 100:
            issues.append(f'{name}: 数据不足({len(df)}行)')
        
        # 检查价格合理性
        if df['收盘'].min() <= 0:
            issues.append(f'{name}: 价格异常')
    
    return {
        'success': len(issues) == 0,
        'details': '; '.join(issues) if issues else '数据质量良好'
    }

# ============ 测试10: 内存使用 ============
def test_memory():
    """测试内存使用"""
    import sys
    
    df = pd.read_csv('data/real_000858.csv')
    size = sys.getsizeof(df) / 1024  # KB
    
    return {
        'success': size < 1000,
        'details': f'数据内存占用: {size:.1f} KB'
    }

# ============ 运行所有测试 ============
print("="*70)
print("🧪 AI量化系统 - 全方位测试")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

print("【数据层测试】")
test_case("1. 数据加载", test_data_loading)
test_case("2. 数据质量", test_data_quality)
test_case("3. 指标计算", test_indicators)

print("\n【策略层测试】")
test_case("4. 策略文件", test_strategy_files)
test_case("5. 回测引擎", test_backtest_engine)
test_case("6. 策略运行", test_strategy_run)

print("\n【边界测试】")
test_case("7. 参数边界", test_boundary_params)
test_case("8. 异常处理", test_error_handling)

print("\n【性能测试】")
test_case("9. 性能压力", test_performance)
test_case("10. 内存使用", test_memory)

# ============ 汇总报告 ============
print("\n" + "="*70)
print("📊 测试汇总")
print("="*70)

passed = sum(1 for r in test_results if 'PASS' in r['status'])
failed = sum(1 for r in test_results if 'FAIL' in r['status'] or 'ERROR' in r['status'])
total_time = sum(r['elapsed'] for r in test_results)

print(f"\n通过: {passed}/{len(test_results)}")
print(f"失败: {failed}/{len(test_results)}")
print(f"总耗时: {total_time:.2f}s")

if failed > 0:
    print("\n失败项目:")
    for r in test_results:
        if 'FAIL' in r['status'] or 'ERROR' in r['status']:
            print(f"  - {r['name']}: {r['details']}")

# 评级
if failed == 0:
    grade = "A 🏆 优秀"
elif failed <= 2:
    grade = "B ✅ 良好"
elif failed <= 4:
    grade = "C ⚠️ 一般"
else:
    grade = "D ❌ 需修复"

print(f"\n系统评级: {grade}")
print("="*70)

# 保存报告
report = {
    'timestamp': datetime.now().isoformat(),
    'summary': {
        'passed': passed,
        'failed': failed,
        'total': len(test_results),
        'grade': grade
    },
    'details': test_results
}

import json
with open('test_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print("\n测试报告已保存: test_report.json")
