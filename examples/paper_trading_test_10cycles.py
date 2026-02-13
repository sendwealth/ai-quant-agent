"""
快速实时模拟盘测试 - 10个周期
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
print("AI智能体量化交易系统 - 实时模拟盘测试（10个周期）")
print("="*70)
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================
# 配置参数
# ============================================
STOCK_CODE = '600519'
INITIAL_CAPITAL = 100000

MA_SHORT = 5
MA_LONG = 40
ATR_MULTIPLIER = 2.0
MAX_POSITION_RATIO = 0.3
MAX_DAILY_LOSS = -0.03
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.2
ADX_THRESHOLD = 25

# ============================================
# 导入模块
# ============================================
from data.astock_fetcher import AStockDataFetcher
from utils.indicators import sma, rsi

fetcher = AStockDataFetcher()

# ============================================
# 自定义指标
# ============================================
def atr(data, period=14):
    high = data['high'] if 'high' in data.columns else data['close']
    low = data['low'] if 'low' in data.columns else data['close']
    close = data['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()

def adx(data, period=14):
    high = data['high'] if 'high' in data.columns else data['close'] * 1.02
    low = data['low'] if 'low' in data.columns else data['close'] * 0.98
    close = data['close']

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.zeros(len(data))
    minus_dm = np.zeros(len(data))

    mask_plus = (up_move > down_move) & (up_move > 0)
    mask_minus = (down_move > up_move) & (down_move > 0)

    plus_dm[mask_plus] = up_move[mask_plus].values
    minus_dm[mask_minus] = down_move[mask_minus].values

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr_series = tr.rolling(window=period).mean()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period).mean()

    plus_di = 100 * plus_dm_smooth / atr_series
    minus_di = 100 * minus_dm_smooth / atr_series

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_series = dx.rolling(window=period).mean()

    return adx_series

# ============================================
# 获取数据
# ============================================
print(f"\n获取数据...")
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')

df = fetcher.fetch_stock_daily(STOCK_CODE, start_date, end_date, source='akshare')

if df is None or len(df) < 200:
    print(f"  使用模拟数据...")
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
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'volume': [int(np.random.uniform(1000000, 5000000)) for _ in prices],
    })

print(f"✓ 获取数据: {len(df)} 条")

# ============================================
# 计算指标
# ============================================
print(f"\n计算指标...")

df['sma_short'] = sma(df['close'], MA_SHORT)
df['sma_long'] = sma(df['close'], MA_LONG)
df['rsi'] = rsi(df['close'], 14)
df['atr'] = atr(df, 14)
df['adx'] = adx(df, 14)
df['volume_ma'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()

df['sma_short'] = df['sma_short'].ffill()
df['sma_long'] = df['sma_long'].ffill()
df['rsi'] = df['rsi'].fillna(50)
df['atr'] = df['atr'].ffill()
df['adx'] = df['adx'].fillna(0)
df['volume_ma'] = df['volume_ma'].ffill()

# 生成信号
df['base_signal'] = 0
df.loc[df['sma_short'] > df['sma_long'], 'base_signal'] = 1
df.loc[df['sma_short'] < df['sma_long'], 'base_signal'] = -1

df['volume_confirm'] = df['volume'] > df['volume_ma'] * VOLUME_MULTIPLIER
df['trend_confirm'] = df['adx'] > ADX_THRESHOLD

df['final_signal'] = 0
df.loc[
    (df['base_signal'] == 1) &
    (df['volume_confirm']) &
    (df['trend_confirm']),
    'final_signal'
] = 1

df.loc[
    (df['base_signal'] == -1),
    'final_signal'
] = -1

print(f"✓ 计算指标完成")
print(f"  买入信号: {(df['final_signal'] == 1).sum()}")
print(f"  卖出信号: {(df['final_signal'] == -1).sum()}")

# ============================================
# 模拟交易（10个周期）
# ============================================
print(f"\n开始模拟交易（10个周期）")
print(f"{'='*70}")

state = {
    'capital': INITIAL_CAPITAL,
    'position': 0,
    'entry_price': 0,
    'stop_loss': 0,
    'equity_curve': [INITIAL_CAPITAL],
    'last_signal': 0,
    'trades': [],
    'stop_loss_trades': 0,
    'daily_loss_trades': 0
}

cycles = 10
trades_per_cycle = 50

for cycle in range(cycles):
    start_idx = cycle * trades_per_cycle
    end_idx = min(start_idx + trades_per_cycle, len(df))

    if start_idx >= len(df):
        break

    print(f"\n周期 {cycle + 1}/{cycles}: 第{start_idx+1}-{end_idx}个交易日")
    print(f"  时间范围: {df['datetime'].iloc[start_idx].strftime('%Y-%m-%d')} - {df['datetime'].iloc[end_idx-1].strftime('%Y-%m-%d')}")

    cycle_trades = 0
    cycle_stop_loss = 0
    cycle_daily_loss = 0

    for i in range(start_idx, end_idx):
        price = df['close'].iloc[i]
        signal = df['final_signal'].iloc[i]
        prev_signal = df['final_signal'].iloc[i-1] if i > 0 else 0
        atr_val = df['atr'].iloc[i]

        # 止损检查
        if state['position'] > 0 and price <= state['stop_loss']:
            capital = state['position'] * price
            state['capital'] += capital
            pnl = (price - state['entry_price']) / state['entry_price'] * 100

            state['trades'].append({
                'cycle': cycle + 1,
                'action': '止损卖出',
                'price': price,
                'shares': state['position'],
                'pnl': pnl,
                'type': 'stop_loss'
            })

            state['position'] = 0
            state['entry_price'] = 0
            state['stop_loss'] = 0
            state['last_signal'] = 0
            state['stop_loss_trades'] += 1

            cycle_stop_loss += 1
            cycle_trades += 1
            continue

        # 日亏损检查
        if state['position'] > 0:
            current_pnl = (price - state['entry_price']) / state['entry_price']
            if current_pnl <= MAX_DAILY_LOSS:
                capital = state['position'] * price
                state['capital'] += capital
                pnl = current_pnl * 100

                state['trades'].append({
                    'cycle': cycle + 1,
                    'action': '日风控卖出',
                    'price': price,
                    'shares': state['position'],
                    'pnl': pnl,
                    'type': 'daily_loss'
                })

                state['position'] = 0
                state['entry_price'] = 0
                state['stop_loss'] = 0
                state['last_signal'] = 0
                state['daily_loss_trades'] += 1

                cycle_daily_loss += 1
                cycle_trades += 1
                continue

        # 买入
        if signal == 1 and prev_signal != 1 and state['position'] == 0:
            max_position_value = state['capital'] * MAX_POSITION_RATIO
            shares = int(max_position_value / price)

            if shares > 0:
                state['position'] = shares
                state['entry_price'] = price
                state['stop_loss'] = price - atr_val * ATR_MULTIPLIER
                state['capital'] -= shares * price
                state['last_signal'] = 1

                print(f"    ✓ 买入: {shares}股 @ ¥{price:.2f} (止损: ¥{state['stop_loss']:.2f})")
                cycle_trades += 1

        # 卖出
        elif signal == -1 and prev_signal != -1 and state['position'] > 0:
            capital = state['position'] * price
            state['capital'] += capital
            pnl = (price - state['entry_price']) / state['entry_price'] * 100

            state['trades'].append({
                'cycle': cycle + 1,
                'action': '信号卖出',
                'price': price,
                'shares': state['position'],
                'pnl': pnl,
                'type': 'signal'
            })

            state['position'] = 0
            state['entry_price'] = 0
            state['stop_loss'] = 0
            state['last_signal'] = -1

            print(f"    ✓ 卖出: {shares}股 @ ¥{price:.2f} (盈亏: {pnl:+.2f}%)")
            cycle_trades += 1

        # 更新权益
        equity = state['capital'] + state['position'] * price
        state['equity_curve'].append(equity)

    # 最终平仓
    if state['position'] > 0:
        capital = state['position'] * df['close'].iloc[end_idx-1]
        state['capital'] += capital
        state['position'] = 0
        state['entry_price'] = 0
        state['stop_loss'] = 0

    equity = state['equity_curve'][-1]
    cycle_return = (equity / INITIAL_CAPITAL - 1) * 100 - (sum([t.get('pnl', 0) for t in state['trades'] if t['cycle'] == cycle+1]) if cycle > 0 else 0)

    print(f"  资金: ¥{equity:,.2f}")
    print(f"  收益: {(equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100:+.2f}%")
    print(f"  交易: {cycle_trades}次 (止损: {cycle_stop_loss}次, 日风控: {cycle_daily_loss}次)")

# ============================================
# 最终报告
# ============================================
print(f"\n{'='*70}")
print("最终报告")
print(f"{'='*70}")

equity = state['equity_curve'][-1]
total_return = (equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
annual_return = (1 + total_return/100) ** (365 / len(df)) - 1

equity_values = pd.Series(state['equity_curve'])
daily_returns = equity_values.pct_change().dropna()
sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

max_equity = max(state['equity_curve'])
min_equity = min(state['equity_curve'])
max_drawdown = ((min_equity - max_equity) / max_equity) * 100 if max_equity > 0 else 0

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
print(f"  - 信号交易: {len(state['trades']) - state['stop_loss_trades'] - state['daily_loss_trades']}次")
print(f"  - 止损: {state['stop_loss_trades']}次")
print(f"  - 日风控: {state['daily_loss_trades']}次")
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
    print(f"  🏆 优秀！可以考虑实盘")
elif grade == 'B':
    print(f"  ✅ 良好，建议继续观察")
else:
    print(f"  ⚠️  一般，需要进一步优化")

print(f"{'='*70}")

sys.exit(0)
