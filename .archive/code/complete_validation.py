"""
完整系统验证 v2.0
================
运行所有测试并生成100%通过报告
"""
import subprocess
import json
from datetime import datetime

print("="*70)
print("🔍 完整系统验证")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 运行基础测试
print("【1/2】运行基础功能测试...")
result1 = subprocess.run(['python3', 'examples/full_system_test.py'],
                         capture_output=True, text=True)

# 解析结果
basic_passed = '10/10' in result1.stdout
print(f"  基础测试: {'✅ 10/10 通过' if basic_passed else '❌ 失败'}")

# 运行压力测试
print("\n【2/2】运行深度压力测试...")
result2 = subprocess.run(['python3', 'examples/stress_test.py'],
                         capture_output=True, text=True)

# 解析结果
stress_passed = '8/8' in result2.stdout
print(f"  压力测试: {'✅ 8/8 通过' if stress_passed else '❌ 失败'}")

# 汇总
print("\n" + "="*70)
print("📊 最终验证结果")
print("="*70)

total_tests = 18
passed_tests = 10 + 8 if basic_passed and stress_passed else 0
pass_rate = (passed_tests / total_tests) * 100

print(f"\n总测试数: {total_tests}")
print(f"通过: {passed_tests}")
print(f"失败: {total_tests - passed_tests}")
print(f"通过率: {pass_rate:.1f}%")

if pass_rate == 100:
    print("\n🏆 系统评级: A+ 完美")
    print("✅ 所有测试100%通过")
    print("✅ 系统已完全验证")
    print("✅ 生产环境就绪")
else:
    print(f"\n⚠️ 系统评级: B ({pass_rate:.1f}%通过)")

print("\n" + "="*70)

# 保存结果
report = {
    'timestamp': datetime.now().isoformat(),
    'total_tests': total_tests,
    'passed': passed_tests,
    'failed': total_tests - passed_tests,
    'pass_rate': pass_rate,
    'grade': 'A+' if pass_rate == 100 else 'A' if pass_rate >= 90 else 'B',
    'details': {
        'basic_tests': '10/10 PASS' if basic_passed else 'FAIL',
        'stress_tests': '8/8 PASS' if stress_passed else 'FAIL'
    }
}

with open('final_test_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"报告已保存: final_test_report.json")
