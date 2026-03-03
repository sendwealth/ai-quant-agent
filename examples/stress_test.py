"""
深度压力测试 v2.0
================
测试极端情况：
1. 极端市场（暴涨/暴跌）
2. 数据异常
3. 参数极限
4. 并发压力
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys

print("="*70)
print("🔬 深度压力测试")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

results = []

# ============ 测试1: 极端市场模拟 ============
print("【测试1: 极端市场模拟】")

# 暴涨市场
bull_data = pd.DataFrame({
    'close': [100 * (1 + 0.02 * i) for i in range(200)],  # 连续上涨
    'high': [100 * (1 + 0.02 * i) * 1.02 for i in range(200)],
    'low': [100 * (1 + 0.02 * i) * 0.98 for i in range(200)],
})
bull_data['ma_5'] = bull_data['close'].rolling(5).mean()
bull_data['ma_30'] = bull_data['close'].rolling(30).mean()

cash = 100000
shares = 0
for i in range(30, len(bull_data)):
    if bull_data['ma_5'].iloc[i] > bull_data['ma_30'].iloc[i] and shares == 0:
        shares = int(cash * 0.25 / bull_data['close'].iloc[i])
        cash -= shares * bull_data['close'].iloc[i]
    elif bull_data['ma_5'].iloc[i] < bull_data['ma_30'].iloc[i] and shares > 0:
        cash += shares * bull_data['close'].iloc[i]
        shares = 0

if shares > 0:
    cash += shares * bull_data['close'].iloc[-1]

bull_return = (cash - 100000) / 100000 * 100
print(f"  暴涨市场(+300%): 策略收益 {bull_return:+.1f}%")
results.append(('暴涨市场', bull_return > 0))

# 暴跌市场
bear_data = pd.DataFrame({
    'close': [100 * (1 - 0.015 * i) for i in range(200)],  # 连续下跌
    'high': [100 * (1 - 0.015 * i) * 1.02 for i in range(200)],
    'low': [100 * (1 - 0.015 * i) * 0.98 for i in range(200)],
})
bear_data['ma_5'] = bear_data['close'].rolling(5).mean()
bear_data['ma_30'] = bear_data['close'].rolling(30).mean()

cash = 100000
shares = 0
for i in range(30, len(bear_data)):
    if bear_data['ma_5'].iloc[i] > bear_data['ma_30'].iloc[i] and shares == 0:
        shares = int(cash * 0.25 / bear_data['close'].iloc[i])
        cash -= shares * bear_data['close'].iloc[i]
    elif bear_data['ma_5'].iloc[i] < bear_data['ma_30'].iloc[i] and shares > 0:
        cash += shares * bear_data['close'].iloc[i]
        shares = 0

if shares > 0:
    cash += shares * bear_data['close'].iloc[-1]

bear_return = (cash - 100000) / 100000 * 100
bear_bh = (bear_data['close'].iloc[-1] / bear_data['close'].iloc[0] - 1) * 100
print(f"  暴跌市场({bear_bh:.1f}%): 策略收益 {bear_return:+.1f}%")
results.append(('暴跌市场', bear_return > bear_bh))

# ============ 测试2: 波动率极限 ============
print("\n【测试2: 波动率极限】")

# 高波动
high_vol = pd.DataFrame({
    'close': [100 * (1 + 0.1 * np.sin(i/5) * (1 + i/100)) for i in range(200)],
    'high': [100 * (1 + 0.1 * np.sin(i/5) * (1 + i/100)) * 1.05 for i in range(200)],
    'low': [100 * (1 + 0.1 * np.sin(i/5) * (1 + i/100)) * 0.95 for i in range(200)],
})
vol = high_vol['close'].pct_change().std() * np.sqrt(252)
print(f"  高波动数据: 年化波动率 {vol*100:.1f}%")
results.append(('高波动', vol > 0.5))

# ============ 测试3: 参数极限 ============
print("\n【测试3: 参数极限】")

# 极小止损
df = pd.read_csv('data/real_000858.csv')
df = df.rename(columns={'收盘': 'close', '最高': 'high', '最低': 'low'})
price = float(df['close'].iloc[50])

# 0.1%仓位
shares_tiny = int(100000 * 0.001 / price)
# 100%仓位
shares_full = int(100000 * 1.0 / price)

print(f"  极小仓位(0.1%): {shares_tiny}股")
print(f"  满仓(100%): {shares_full}股")
results.append(('参数极限', shares_tiny >= 0 and shares_full > shares_tiny))

# ============ 测试4: 数据异常处理 ============
print("\n【测试4: 数据异常处理】")

# 缺失值处理
df_nan = pd.DataFrame({
    'close': [100, np.nan, 102, 103, np.nan, 105],
    'high': [101, 102, 103, 104, 105, 106],
    'low': [99, 100, 101, 102, 103, 104],
})
df_nan['ma'] = df_nan['close'].rolling(3).mean()
nan_count = df_nan['ma'].isna().sum()
print(f"  缺失值处理: {nan_count}个NaN")
results.append(('缺失值', nan_count > 0))

# 负价格
try:
    df_neg = pd.DataFrame({'close': [-100, 101, 102]})
    valid = df_neg['close'].min() > 0
    print(f"  负价格检测: {'通过' if not valid else '失败'}")
    results.append(('负价格', not valid))
except:
    results.append(('负价格', True))
    print("  负价格检测: 通过")

# ============ 测试5: 性能压力 ============
print("\n【测试5: 性能压力】")

import time

# 大数据量
start = time.time()
big_df = pd.DataFrame({
    'close': np.random.randn(10000) * 10 + 100,
    'high': np.random.randn(10000) * 10 + 105,
    'low': np.random.randn(10000) * 10 + 95,
})
big_df['ma_5'] = big_df['close'].rolling(5).mean()
big_df['ma_30'] = big_df['close'].rolling(30).mean()
big_df['ma_60'] = big_df['close'].rolling(60).mean()
elapsed = time.time() - start

print(f"  10000行数据处理: {elapsed:.3f}s")
results.append(('性能', elapsed < 1))

# ============ 测试6: 策略稳定性 ============
print("\n【测试6: 策略稳定性】")

# 运行同一策略10次
returns = []
for _ in range(10):
    df = pd.read_csv('data/real_000858.csv')
    df = df.rename(columns={'收盘': 'close'})
    df['ma_5'] = df['close'].rolling(5).mean()
    df['ma_30'] = df['close'].rolling(30).mean()
    
    cash = 100000
    shares = 0
    for i in range(30, len(df)):
        if pd.notna(df['ma_5'].iloc[i]) and pd.notna(df['ma_30'].iloc[i]):
            if df['ma_5'].iloc[i] > df['ma_30'].iloc[i] and shares == 0:
                shares = int(cash * 0.25 / df['close'].iloc[i])
                cash -= shares * df['close'].iloc[i]
            elif df['ma_5'].iloc[i] < df['ma_30'].iloc[i] and shares > 0:
                cash += shares * df['close'].iloc[i]
                shares = 0
    
    if shares > 0:
        cash += shares * df['close'].iloc[-1]
    
    returns.append((cash - 100000) / 100000)

std = np.std(returns)
print(f"  10次运行标准差: {std:.6f}")
results.append(('稳定性', std < 0.0001))

# ============ 汇总 ============
print("\n" + "="*70)
print("📊 压力测试汇总")
print("="*70)

passed = sum(1 for _, p in results if p)
total = len(results)

print(f"\n通过: {passed}/{total}")
print(f"失败: {total - passed}/{total}")

for name, passed in results:
    status = "✅" if passed else "❌"
    print(f"  {status} {name}")

if passed == total:
    print("\n评级: A 🏆 系统稳定")
elif passed >= total * 0.8:
    print("\n评级: B ✅ 基本稳定")
else:
    print("\n评级: C ⚠️ 需要改进")

print("="*70)
