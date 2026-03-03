"""
模拟数据生成器 v1.0
==================
生成不同市场环境的模拟数据用于策略测试
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_trending_data(days=500, trend='up', volatility=0.02):
    """
    生成趋势数据
    trend: 'up'(上涨), 'down'(下跌), 'sideways'(震荡), 'mixed'(混合)
    """
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    if trend == 'up':
        # 上涨趋势
        drift = 0.0005
        returns = np.random.randn(days) * volatility + drift
    elif trend == 'down':
        # 下跌趋势
        drift = -0.0005
        returns = np.random.randn(days) * volatility + drift
    elif trend == 'sideways':
        # 震荡
        returns = np.random.randn(days) * volatility * 0.5
    else:
        # 混合：先涨后跌
        mid = days // 2
        returns = np.concatenate([
            np.random.randn(mid) * volatility + 0.0003,
            np.random.randn(days - mid) * volatility - 0.0003
        ])
    
    # 生成价格
    price = 100
    prices = [price]
    for r in returns[1:]:
        price *= (1 + r)
        prices.append(price)
    
    close = np.array(prices)
    high = close * (1 + np.random.rand(days) * 0.02)
    low = close * (1 - np.random.rand(days) * 0.02)
    open_price = close * (1 + (np.random.rand(days) - 0.5) * 0.01)
    volume = np.random.randint(1000000, 10000000, days)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    return df

def generate_crash_recovery(days=500):
    """生成崩盘后恢复的数据"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    # 前1/3正常，中间1/3崩盘，后1/3恢复
    normal_days = days // 3
    
    returns = np.concatenate([
        np.random.randn(normal_days) * 0.015,  # 正常
        np.random.randn(normal_days) * 0.03 - 0.02,  # 崩盘
        np.random.randn(days - 2*normal_days) * 0.02 + 0.001  # 恢复
    ])
    
    price = 100
    prices = [price]
    for r in returns[1:]:
        price *= (1 + r)
        prices.append(price)
    
    close = np.array(prices)
    high = close * (1 + np.random.rand(days) * 0.02)
    low = close * (1 - np.random.rand(days) * 0.02)
    
    return pd.DataFrame({
        'datetime': dates,
        'open': close * (1 + (np.random.rand(days) - 0.5) * 0.01),
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, days)
    })

print("="*70)
print("📊 模拟数据生成器")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# 生成不同场景的数据
scenarios = {
    'uptrend': ('上涨趋势', generate_trending_data(500, 'up')),
    'downtrend': ('下跌趋势', generate_trending_data(500, 'down')),
    'sideways': ('震荡市场', generate_trending_data(500, 'sideways')),
    'mixed': ('混合市场', generate_trending_data(500, 'mixed')),
    'crash_recovery': ('崩盘恢复', generate_crash_recovery(500)),
}

# 保存数据
for key, (name, df) in scenarios.items():
    filepath = f'data/sim_{key}.csv'
    df.to_csv(filepath, index=False)
    
    ret = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
    vol = df['close'].pct_change().std() * np.sqrt(252) * 100
    
    print(f"{name}:")
    print(f"  数据: {len(df)}天")
    print(f"  收益: {ret:+.1f}%")
    print(f"  波动率: {vol:.1f}%")
    print(f"  文件: {filepath}\n")

print("="*70)
print("✅ 模拟数据生成完成")
print("="*70)
print("\n可用于测试策略在不同市场环境下的表现")
