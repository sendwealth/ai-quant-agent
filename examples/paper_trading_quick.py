"""
快速模拟盘验证系统
使用优化后的策略：MA(5/40)
快速完成10个周期的验证（模拟）
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


print("\n" + "="*70)
print("AI智能体量化交易系统 - 快速模拟盘验证")
print("="*70)
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 配置参数
STOCK_CODE = '600519'
INITIAL_CAPITAL = 100000

# 优化后的策略参数
MA_SHORT = 5
MA_LONG = 40
RSI_PERIOD = 14

print(f"\n策略配置:")
print(f"  股票代码: {STOCK_CODE}")
print(f"  初始资金: ¥{INITIAL_CAPITAL:,.2f}")
print(f"  短期均线: {MA_SHORT}")
print(f"  长期均线: {MA_LONG}")
print(f"  RSI周期: {RSI_PERIOD}")

# 导入模块
from data.astock_fetcher import AStockDataFetcher
from utils.indicators import sma, rsi

# 初始化数据获取器
fetcher = AStockDataFetcher()

# 获取数据
print(f"\n获取数据...")
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

df = fetcher.fetch_stock_daily(STOCK_CODE, start_date, end_date, source='akshare')

if df is None or len(df) < 200:
    print(f"⚠️  实时数据获取失败，使用模拟数据")
    np.random.seed(42)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    price = 1500.0
    prices = []
    for _ in range(len(dates)):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(max(price, 100))
    
    df = pd.DataFrame({
        'datetime': dates,
        'close': prices,
        'volume': [int(np.random.uniform(1000000, 5000000)) for _ in prices],
    })

print(f"✓ 获取数据: {len(df)} 条")

# 计算指标
df['sma_short'] = sma(df['close'], MA_SHORT)
df['sma_long'] = sma(df['close'], MA_LONG)
df['rsi'] = rsi(df['close'], RSI_PERIOD)

# 填充NaN值
df['sma_short'] = df['sma_short'].ffill()
df['sma_long'] = df['sma_long'].ffill()
df['rsi'] = df['rsi'].fillna(50)

# 生成信号
df['signal'] = 0
df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1

print(f"✓ 计算指标完成")
print(f"  买入信号: {(df['signal'] == 1).sum()}")
print(f"  卖出信号: {(df['signal'] == -1).sum()}")

# 模拟交易（10个周期，每个周期50个交易日）
CYCLES = 10
TRADES_PER_CYCLE = 50

print(f"\n开始模拟交易（{CYCLES}个周期）")
print("="*70)

state = {
    'capital': INITIAL_CAPITAL,
    'position': 0,
    'entry_price': 0,
    'equity_curve': [INITIAL_CAPITAL],
    'last_signal': 0,
    'trades': []
}

for cycle in range(CYCLES):
    start_idx = cycle * TRADES_PER_CYCLE
    end_idx = min(start_idx + TRADES_PER_CYCLE, len(df))
    
    if start_idx >= len(df):
        break
    
    print(f"\n周期 {cycle + 1}/{CYCLES}: 第{start_idx+1}-{end_idx}个交易日")
    
    cycle_capital = state['capital']
    cycle_trades = 0
    
    for i in range(start_idx, end_idx):
        price = df['close'].iloc[i]
        signal = df['signal'].iloc[i]
        prev_signal = df['signal'].iloc[i-1] if i > 0 else 0
        
        # 金叉：买入
        if signal == 1 and prev_signal == -1 and state['position'] == 0:
            shares = int(state['capital'] / price)
            if shares > 0:
                state['position'] = shares
                state['entry_price'] = price
                state['capital'] -= shares * price
                cycle_trades += 1
        
        # 死叉：卖出
        elif signal == -1 and prev_signal == 1 and state['position'] > 0:
            capital = state['position'] * price
            state['capital'] += capital
            pnl = (price - state['entry_price']) / state['entry_price'] * 100
            
            state['trades'].append({
                'cycle': cycle + 1,
                'action': '卖出',
                'price': price,
                'shares': state['position'],
                'pnl': pnl
            })
            
            state['position'] = 0
            state['entry_price'] = 0
            cycle_trades += 1
        
        # 更新权益曲线
        equity = state['capital'] + state['position'] * price
        state['equity_curve'].append(equity)
    
    # 最终平仓
    if state['position'] > 0:
        capital = state['position'] * df['close'].iloc[end_idx-1]
        state['capital'] += capital
        state['position'] = 0
        state['entry_price'] = 0
    
    equity = state['equity_curve'][-1]
    cycle_return = (equity - cycle_capital) / cycle_capital * 100
    
    print(f"  资金: ¥{equity:,.2f}")
    print(f"  收益: {cycle_return:+.2f}%")
    print(f"  交易: {cycle_trades}次")

# 计算最终性能
print(f"\n" + "="*70)
print("最终报告")
print("="*70)

equity = state['equity_curve'][-1]
total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
annual_return = (1 + total_return/100) ** (365 / len(df)) - 1

equity_values = pd.Series(state['equity_curve'])
daily_returns = equity_values.pct_change().dropna()
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

cummax = equity_values.cummax()
drawdown = (equity_values - cummax) / cummax
max_drawdown = drawdown.min() * 100

win_trades = sum(1 for t in state['trades'] if t.get('pnl', 0) > 0)
loss_trades = sum(1 for t in state['trades'] if t.get('pnl', 0) <= 0)
win_rate = win_trades / len(state['trades']) * 100 if state['trades'] else 0

avg_pnl = np.mean([t.get('pnl', 0) for t in state['trades']]) if state['trades'] else 0

print(f"\n最终权益: ¥{equity:,.2f}")
print(f"总收益: {total_return:+.2f}%")
print(f"年化收益: {annual_return*100:+.2f}%")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2f}%")
print(f"交易次数: {len(state['trades'])}")
print(f"胜率: {win_rate:.2f}%")
print(f"平均盈亏: {avg_pnl:+.2f}%")

# 评级
grade = 'C'
if sharpe_ratio > 1.0 and max_drawdown > -15:
    grade = 'A'
elif sharpe_ratio > 0.8 and max_drawdown > -20:
    grade = 'B'

print(f"\n策略评级: {grade}")
if grade == 'A':
    print("  🏆 优秀！可以考虑实盘")
elif grade == 'B':
    print("  ✅ 良好，建议继续观察")
else:
    print("  ⚠️  一般，需要进一步优化")

print("="*70)

sys.exit(0)
