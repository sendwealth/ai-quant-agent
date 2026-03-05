"""
3个月实时模拟验证 - 监控脚本
查看当前状态和性能
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

data_dir = Path("data/paper_trading_3month")

print("\n" + "="*70)
print("3个月实时模拟验证 - 状态监控")
print("="*70)

# 加载状态
state_file = data_dir / "state_600519.json"
trades_file = data_dir / "trades_600519.csv"
report_file = data_dir / "report_600519.json"

if not state_file.exists():
    print("\n❌ 状态文件不存在，系统可能尚未启动")
    exit(1)

with open(state_file, 'r') as f:
    state = json.load(f)

# 计算性能
equity_values = pd.Series(state['equity_curve'])
total_return = (equity_values.iloc[-1] - 100000) / 100000 * 100

daily_returns = equity_values.pct_change().dropna()
if len(daily_returns) > 0 and daily_returns.std() > 0:
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
else:
    sharpe_ratio = 0

max_drawdown = ((state['min_equity'] - state['max_equity']) / state['max_equity']) * 100 if state['max_equity'] > 0 else 0

win_rate = state['win_trades'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0
avg_pnl = state['total_pnl'] / state['total_trades'] if state['total_trades'] > 0 else 0

# 评级
grade = 'C'
if sharpe_ratio > 1.0 and max_drawdown > -15:
    grade = 'A'
elif sharpe_ratio > 0.8 and max_drawdown > -20:
    grade = 'B'

# 显示信息
print(f"\n运行时间: {(datetime.now() - pd.to_datetime(state['start_time'])).days} 天")
print(f"当前周期: {state['cycle_count']}/66")

print(f"\n" + "-"*70)
print("当前持仓")
print("-"*70)
print(f"  现金: ¥{state['capital']:,.2f}")
print(f"  持仓: {state['position']}股")
print(f"  持仓价值: ¥{state['position'] * (state['entry_price'] if state['position'] > 0 else 0):,.2f}")
print(f"  总权益: ¥{equity_values.iloc[-1]:,.2f}")

print(f"\n" + "-"*70)
print("性能指标")
print("-"*70)
print(f"  总收益: {total_return:+.2f}%")
print(f"  年化收益: {(1 + total_return/100) ** (365 / max(1, (datetime.now() - pd.to_datetime(state['start_time'])).days)) - 1:.2%}")
print(f"  夏普比率: {sharpe_ratio:.2f}")
print(f"  最大回撤: {max_drawdown:.2f}%")
print(f"  胜率: {win_rate:.2f}%")
print(f"  平均盈亏: {avg_pnl:+.2f}%")

print(f"\n" + "-"*70)
print("交易统计")
print("-"*70)
print(f"  总交易次数: {state['total_trades']}")
print(f"  盈利交易: {state['win_trades']}")
print(f"  亏损交易: {state['loss_trades']}")
print(f"  信号交易: {state['signal_trades']}")
print(f"  追踪止损: {state['trailing_stop_trades']}")
print(f"  初始止损: {state['stop_loss_trades']}")
print(f"  日风控: {state['daily_loss_trades']}")

print(f"\n" + "-"*70)
print("策略评级")
print("-"*70)
print(f"  等级: {grade}")
if grade == 'A':
    print(f"  🏆 优秀！")
elif grade == 'B':
    print(f"  ✅ 良好")
else:
    print(f"  ⚠️  一般")

# 显示最近交易
if trades_file.exists():
    trades_df = pd.read_csv(trades_file)
    if len(trades_df) > 0:
        print(f"\n" + "-"*70)
        print("最近5笔交易")
        print("-"*70)
        print(trades_df.tail(5)[['datetime', 'action', 'price', 'shares', 'pnl', 'type']].to_string(index=False))

print(f"\n{'='*70}")
print(f"数据目录: {data_dir.absolute()}")
print(f"状态文件: {state_file.absolute()}")
print(f"交易记录: {trades_file.absolute()}")
print(f"性能报告: {report_file.absolute()}")
print(f"{'='*70}\n")
