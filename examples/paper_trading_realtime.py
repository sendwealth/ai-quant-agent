"""
实时模拟盘验证系统
使用优化后的策略：MA(5/40)
持续运行，实时获取数据并执行交易
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import time
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


print("\n" + "="*70)
print("AI智能体量化交易系统 - 实时模拟盘验证")
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

# 数据目录
data_dir = Path("data/paper_trading")
data_dir.mkdir(exist_ok=True)

# 状态文件
state_file = data_dir / f"paper_trading_{STOCK_CODE}.json"
trades_file = data_dir / f"trades_{STOCK_CODE}.csv"
log_file = data_dir / f"log_{STOCK_CODE}.txt"

# 导入模块
from data.astock_fetcher import AStockDataFetcher
from utils.indicators import sma, rsi

# 初始化数据获取器
fetcher = AStockDataFetcher()

# 加载或初始化交易状态
def load_state():
    """加载交易状态"""
    if state_file.exists():
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            'capital': INITIAL_CAPITAL,
            'position': 0,
            'entry_price': 0,
            'equity_curve': [INITIAL_CAPITAL],
            'last_signal': 0,
            'start_time': datetime.now().isoformat(),
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'total_pnl': 0
        }

# 保存交易状态
def save_state(state):
    """保存交易状态"""
    # 转换numpy类型为Python原生类型
    state_to_save = {}
    for key, value in state.items():
        if isinstance(value, (np.integer, np.floating)):
            state_to_save[key] = float(value)
        elif isinstance(value, (list, tuple)):
            state_to_save[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
        else:
            state_to_save[key] = value
    
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_to_save, f, indent=2, ensure_ascii=False)

# 记录交易
def record_trade(trade):
    """记录交易到CSV"""
    df = pd.DataFrame([trade])
    df.to_csv(trades_file, mode='a', header=not trades_file.exists(), index=False)

# 写入日志
def write_log(message):
    """写入日志文件"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}\n"
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_line)
    print(log_line.strip())

# 获取最新数据
def get_latest_data(stock_code, days=100):
    """获取最近的数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    df = fetcher.fetch_stock_daily(stock_code, start_date, end_date, source='akshare')

    if df is None or len(df) < MA_LONG + 20:
        # 使用模拟数据
        write_log(f"⚠️  实时数据获取失败，使用模拟数据")
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

    return df

# 生成交易信号
def generate_signals(df):
    """生成交易信号"""
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

    return df

# 执行交易
def execute_trade(signal, price, state):
    """执行交易"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    trade = None

    # 金叉：买入
    if signal == 1 and state['last_signal'] == -1 and state['position'] == 0:
        shares = int(state['capital'] / price)
        if shares > 0:
            state['position'] = shares
            state['entry_price'] = price
            state['capital'] -= shares * price

            trade = {
                'datetime': current_time,
                'action': '买入',
                'price': price,
                'shares': shares,
                'capital': state['capital'],
                'position_value': state['position'] * price
            }

            write_log(f"✓ 买入: {shares}股 @ ¥{price:.2f}")

    # 死叉：卖出
    elif signal == -1 and state['last_signal'] == 1 and state['position'] > 0:
        capital = state['position'] * price
        state['capital'] += capital
        pnl = (price - state['entry_price']) / state['entry_price'] * 100
        state['total_pnl'] += pnl
        state['total_trades'] += 1

        if pnl > 0:
            state['win_trades'] += 1
        else:
            state['loss_trades'] += 1

        trade = {
            'datetime': current_time,
            'action': '卖出',
            'price': price,
            'shares': state['position'],
            'capital': state['capital'],
            'position_value': 0,
            'pnl': pnl
        }

        write_log(f"✓ 卖出: {state['position']}股 @ ¥{price:.2f} (盈亏: {pnl:+.2f}%)")

        state['position'] = 0
        state['entry_price'] = 0

    # 更新信号
    state['last_signal'] = signal

    # 更新权益曲线
    equity = state['capital'] + state['position'] * price
    state['equity_curve'].append(equity)

    # 记录交易
    if trade:
        record_trade(trade)

    return trade

# 计算性能指标
def calculate_performance(state, df):
    """计算性能指标"""
    if len(state['equity_curve']) < 2:
        return {}

    equity_values = pd.Series(state['equity_curve'])

    # 总收益
    total_return = (equity_values.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 日收益率
    daily_returns = equity_values.pct_change().dropna()

    # 夏普比率
    if daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 最大回撤
    cummax = equity_values.cummax()
    drawdown = (equity_values - cummax) / cummax
    max_drawdown = drawdown.min() * 100

    # 胜率
    win_rate = state['win_trades'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0

    # 平均盈亏
    avg_pnl = state['total_pnl'] / state['total_trades'] if state['total_trades'] > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_trades': state['total_trades'],
        'win_trades': state['win_trades'],
        'loss_trades': state['loss_trades']
    }

# 主循环
def main_loop(cycles=10, interval=30):
    """主循环"""
    write_log("="*70)
    write_log("实时模拟盘验证启动")
    write_log("="*70)
    write_log(f"策略参数: MA({MA_SHORT}/{MA_LONG})")
    write_log(f"初始资金: ¥{INITIAL_CAPITAL:,.2f}")

    # 加载状态
    state = load_state()
    write_log(f"加载状态: 资金=¥{state['capital']:,.2f}, 持仓={state['position']}股")

    for cycle in range(cycles):
        write_log(f"\n--- 周期 {cycle + 1}/{cycles} ---")

        # 获取数据
        df = get_latest_data(STOCK_CODE, days=100)
        write_log(f"✓ 获取数据: {len(df)} 条")

        # 生成信号
        df = generate_signals(df)
        current_signal = df['signal'].iloc[-1]
        current_price = df['close'].iloc[-1]

        write_log(f"当前价格: ¥{current_price:.2f}")
        write_log(f"当前信号: {current_signal}")

        # 执行交易
        trade = execute_trade(current_signal, current_price, state)

        # 保存状态
        save_state(state)

        # 计算性能
        perf = calculate_performance(state, df)

        if perf:
            write_log("\n当前持仓:")
            write_log(f"  现金: ¥{state['capital']:,.2f}")
            write_log(f"  持仓: {state['position']}股")
            write_log(f"  持仓价值: ¥{state['position'] * current_price:,.2f}")
            write_log(f"  总权益: ¥{state['equity_curve'][-1]:,.2f}")

            write_log("\n性能指标:")
            write_log(f"  总收益: {perf['total_return']:+.2f}%")
            write_log(f"  夏普比率: {perf['sharpe_ratio']:.2f}")
            write_log(f"  最大回撤: {perf['max_drawdown']:.2f}%")
            write_log(f"  胜率: {perf['win_rate']:.2f}%")
            write_log(f"  交易次数: {perf['total_trades']}")

        # 等待下一个周期
        if cycle < cycles - 1:
            write_log(f"\n等待 {interval} 秒...")
            time.sleep(interval)

    # 最终报告
    write_log("\n" + "="*70)
    write_log("最终报告")
    write_log("="*70)

    if perf:
        equity = state['equity_curve'][-1]
        write_log(f"\n最终权益: ¥{equity:,.2f}")
        write_log(f"总收益: {perf['total_return']:+.2f}%")
        write_log(f"夏普比率: {perf['sharpe_ratio']:.2f}")
        write_log(f"最大回撤: {perf['max_drawdown']:.2f}%")
        write_log(f"胜率: {perf['win_rate']:.2f}%")
        write_log(f"总交易次数: {perf['total_trades']}")
        write_log(f"盈利交易: {perf['win_trades']}")
        write_log(f"亏损交易: {perf['loss_trades']}")

        # 评级
        grade = 'C'
        if perf['sharpe_ratio'] > 1.0 and perf['max_drawdown'] > -15:
            grade = 'A'
        elif perf['sharpe_ratio'] > 0.8 and perf['max_drawdown'] > -20:
            grade = 'B'

        write_log(f"\n策略评级: {grade}")
        if grade == 'A':
            write_log("  🏆 优秀！可以考虑实盘")
        elif grade == 'B':
            write_log("  ✅ 良好，建议继续观察")
        else:
            write_log("  ⚠️  一般，需要进一步优化")

    write_log("="*70)

    return state, perf

# 运行主程序
if __name__ == '__main__':
    try:
        state, perf = main_loop(cycles=10, interval=30)
        print("\n✓ 模拟盘验证完成")
        print(f"最终权益: ¥{state['equity_curve'][-1]:,.2f}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
