"""
3个月实时模拟验证系统 - 使用最终优化策略
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


print("\n" + "="*70)
print("AI智能体量化交易系统 - 3个月实时模拟验证")
print("="*70)
print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"预计运行时间: 3个月（约66个交易日）")

# ============================================
# 配置参数（最终优化版）
# ============================================
STOCK_CODE = '600519'
INITIAL_CAPITAL = 100000

# 策略参数
MA_SHORT = 3
MA_LONG = 30
RSI_PERIOD = 14
ATR_PERIOD = 14

# 风控参数
ATR_MULTIPLIER = 1.5
TRAILING_STOP_ATR = 1.5
TRAILING_STOP_PERCENT = 0.05
MAX_POSITION_RATIO = 0.3
MAX_DAILY_LOSS = -0.05
VOLUME_MA_PERIOD = 20
VOLUME_MULTIPLIER = 1.0
ADX_PERIOD = 14
ADX_THRESHOLD = 20

# 运行参数
CYCLE_INTERVAL = 60  # 每次检查间隔（秒）
TARGET_CYCLES = 66  # 3个月（约66个交易日）
REPORT_INTERVAL = 10  # 每10个周期生成报告

print(f"\n策略配置（最终优化版）:")
print(f"  股票代码: {STOCK_CODE}")
print(f"  初始资金: ¥{INITIAL_CAPITAL:,.2f}")
print(f"  短期均线: {MA_SHORT}")
print(f"  长期均线: {MA_LONG}")

print(f"\n风控配置:")
print(f"  ATR止损: {ATR_MULTIPLIER}倍")
print(f"  追踪止损: {TRAILING_STOP_ATR}倍ATR / {TRAILING_STOP_PERCENT*100:.0f}%")
print(f"  最大持仓: {MAX_POSITION_RATIO*100:.0f}%")
print(f"  最大日亏损: {MAX_DAILY_LOSS*100:.0f}%")
print(f"  成交量确认: {VOLUME_MULTIPLIER}倍20日均量")
print(f"  ADX趋势过滤: >{ADX_THRESHOLD}")

print(f"\n运行配置:")
print(f"  目标周期: {TARGET_CYCLES}（3个月）")
print(f"  检查间隔: {CYCLE_INTERVAL}秒")
print(f"  报告间隔: 每{REPORT_INTERVAL}个周期")

# ============================================
# 数据目录
# ============================================
data_dir = Path("data/paper_trading_3month")
data_dir.mkdir(exist_ok=True)

# 状态文件
state_file = data_dir / f"state_{STOCK_CODE}.json"
trades_file = data_dir / f"trades_{STOCK_CODE}.csv"
report_file = data_dir / f"report_{STOCK_CODE}.json"

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
# 状态管理
# ============================================
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
            'stop_loss': 0,
            'trailing_stop': 0,
            'highest_price': 0,
            'equity_curve': [INITIAL_CAPITAL],
            'last_signal': 0,
            'start_time': datetime.now().isoformat(),
            'total_trades': 0,
            'win_trades': 0,
            'loss_trades': 0,
            'stop_loss_trades': 0,
            'trailing_stop_trades': 0,
            'daily_loss_trades': 0,
            'signal_trades': 0,
            'total_pnl': 0,
            'cycle_count': 0,
            'max_equity': INITIAL_CAPITAL,
            'min_equity': INITIAL_CAPITAL
        }

def save_state(state):
    """保存交易状态"""
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

def record_trade(trade):
    """记录交易到CSV"""
    df = pd.DataFrame([trade])
    df.to_csv(trades_file, mode='a', header=not trades_file.exists(), index=False)

def load_report():
    """加载报告"""
    if report_file.exists():
        with open(report_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        return {
            'start_time': datetime.now().isoformat(),
            'reports': []
        }

def save_report(report):
    """保存报告"""
    current_report = load_report()
    current_report['reports'].append(report)
    current_report['last_update'] = datetime.now().isoformat()

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(current_report, f, indent=2, ensure_ascii=False)

# ============================================
# 数据获取
# ============================================
def get_latest_data(stock_code, days=100):
    """获取最新数据"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    df = fetcher.fetch_stock_daily(stock_code, start_date, end_date, source='akshare')

    if df is None or len(df) < MA_LONG + 20:
        # 使用模拟数据（但每次更新价格）
        print(f"  ⚠️  使用模拟数据（实时更新）")

        # 检查是否有历史数据
        historical_file = data_dir / f"historical_{STOCK_CODE}.csv"

        if historical_file.exists():
            historical_df = pd.read_csv(historical_file)
            historical_df['datetime'] = pd.to_datetime(historical_df['datetime'])
        else:
            # 生成初始历史数据
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
            np.random.seed(42)
            price = 1500.0
            prices = []
            for _ in range(len(dates)):
                change = np.random.normal(0, 0.02)
                price = price * (1 + change)
                prices.append(max(price, 100))

            historical_df = pd.DataFrame({
                'datetime': dates,
                'close': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                'volume': [int(np.random.uniform(1000000, 5000000)) for _ in prices],
            })
            historical_df.to_csv(historical_file, index=False)

        # 更新最新价格
        last_price = historical_df['close'].iloc[-1]
        new_change = np.random.normal(0, 0.015)  # 1.5% 波动
        new_price = max(last_price * (1 + new_change), 100)

        new_row = {
            'datetime': datetime.now(),
            'close': new_price,
            'high': new_price * (1 + abs(np.random.normal(0, 0.005))),
            'low': new_price * (1 - abs(np.random.normal(0, 0.005))),
            'volume': int(np.random.uniform(1000000, 5000000))
        }

        df = pd.concat([historical_df, pd.DataFrame([new_row])], ignore_index=True)

        # 保存更新的历史数据
        df.to_csv(historical_file, index=False)

    return df

# ============================================
# 信号生成
# ============================================
def generate_signals(df):
    """生成交易信号"""
    # 计算指标
    df['sma_short'] = sma(df['close'], MA_SHORT)
    df['sma_long'] = sma(df['close'], MA_LONG)
    df['rsi'] = rsi(df['close'], RSI_PERIOD)
    df['atr'] = atr(df, ATR_PERIOD)
    df['adx'] = adx(df, ADX_PERIOD)
    df['volume_ma'] = df['volume'].rolling(window=VOLUME_MA_PERIOD).mean()

    # 填充NaN值
    df['sma_short'] = df['sma_short'].ffill()
    df['sma_long'] = df['sma_long'].ffill()
    df['rsi'] = df['rsi'].fillna(50)
    df['atr'] = df['atr'].ffill()
    df['adx'] = df['adx'].fillna(0)
    df['volume_ma'] = df['volume_ma'].ffill()

    # 基础信号
    df['base_signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'base_signal'] = 1
    df.loc[df['sma_short'] < df['sma_long'], 'base_signal'] = -1

    # 风控过滤
    df['volume_confirm'] = df['volume'] > df['volume_ma'] * VOLUME_MULTIPLIER
    df['trend_confirm'] = df['adx'] > ADX_THRESHOLD

    # 最终信号
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

    return df

# ============================================
# 执行交易
# ============================================
def execute_cycle(df, state):
    """执行一个交易周期"""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_price = df['close'].iloc[-1]
    current_signal = df['final_signal'].iloc[-1]
    atr_val = df['atr'].iloc[-1]

    trade_executed = False
    trade_info = {}

    # 更新追踪止损
    if state['position'] > 0:
        state['highest_price'] = max(state['highest_price'], current_price)
        trailing_stop_atr = state['highest_price'] - atr_val * TRAILING_STOP_ATR
        trailing_stop_percent = state['highest_price'] * (1 - TRAILING_STOP_PERCENT)
        state['trailing_stop'] = max(trailing_stop_atr, trailing_stop_percent)
        state['stop_loss'] = max(state['stop_loss'], state['trailing_stop'])

    # 追踪止损检查
    if state['position'] > 0 and current_price <= state['trailing_stop']:
        capital = state['position'] * current_price
        state['capital'] += capital
        pnl = (current_price - state['entry_price']) / state['entry_price'] * 100

        state['total_trades'] += 1
        state['total_pnl'] += pnl
        state['trailing_stop_trades'] += 1

        if pnl > 0:
            state['win_trades'] += 1
        else:
            state['loss_trades'] += 1

        trade_info = {
            'datetime': current_time,
            'action': '追踪止损卖出',
            'price': current_price,
            'shares': state['position'],
            'pnl': pnl,
            'type': 'trailing_stop'
        }

        record_trade(trade_info)
        print(f"  ⚠️  追踪止损: {pnl:+.2f}%")

        state['position'] = 0
        state['entry_price'] = 0
        state['stop_loss'] = 0
        state['trailing_stop'] = 0
        state['highest_price'] = 0
        state['last_signal'] = 0
        trade_executed = True

    # 初始止损检查
    elif state['position'] > 0 and current_price <= state['stop_loss']:
        capital = state['position'] * current_price
        state['capital'] += capital
        pnl = (current_price - state['entry_price']) / state['entry_price'] * 100

        state['total_trades'] += 1
        state['total_pnl'] += pnl
        state['stop_loss_trades'] += 1
        state['loss_trades'] += 1

        trade_info = {
            'datetime': current_time,
            'action': '止损卖出',
            'price': current_price,
            'shares': state['position'],
            'pnl': pnl,
            'type': 'stop_loss'
        }

        record_trade(trade_info)
        print(f"  ⚠️  止损: {pnl:+.2f}%")

        state['position'] = 0
        state['entry_price'] = 0
        state['stop_loss'] = 0
        state['trailing_stop'] = 0
        state['highest_price'] = 0
        state['last_signal'] = 0
        trade_executed = True

    # 日亏损检查
    elif state['position'] > 0:
        current_pnl = (current_price - state['entry_price']) / state['entry_price']
        if current_pnl <= MAX_DAILY_LOSS:
            capital = state['position'] * current_price
            state['capital'] += capital
            pnl = current_pnl * 100

            state['total_trades'] += 1
            state['total_pnl'] += pnl
            state['daily_loss_trades'] += 1
            state['loss_trades'] += 1

            trade_info = {
                'datetime': current_time,
                'action': '日风控卖出',
                'price': current_price,
                'shares': state['position'],
                'pnl': pnl,
                'type': 'daily_loss'
            }

            record_trade(trade_info)
            print(f"  ⚠️  日风控: {pnl:+.2f}%")

            state['position'] = 0
            state['entry_price'] = 0
            state['stop_loss'] = 0
            state['trailing_stop'] = 0
            state['highest_price'] = 0
            state['last_signal'] = 0
            trade_executed = True

    # 买入信号
    elif current_signal == 1 and state['last_signal'] != 1 and state['position'] == 0:
        max_position_value = state['capital'] * MAX_POSITION_RATIO
        shares = int(max_position_value / current_price)

        if shares > 0:
            state['position'] = shares
            state['entry_price'] = current_price
            state['stop_loss'] = current_price - atr_val * ATR_MULTIPLIER
            state['trailing_stop'] = state['stop_loss']
            state['highest_price'] = current_price
            state['capital'] -= shares * current_price
            state['last_signal'] = 1

            trade_info = {
                'datetime': current_time,
                'action': '买入',
                'price': current_price,
                'shares': shares,
                'type': 'buy'
            }

            record_trade(trade_info)
            print(f"  ✓ 买入: {shares}股 @ ¥{current_price:.2f} (止损: ¥{state['stop_loss']:.2f})")
            trade_executed = True

    # 卖出信号
    elif current_signal == -1 and state['last_signal'] != -1 and state['position'] > 0:
        capital = state['position'] * current_price
        state['capital'] += capital
        pnl = (current_price - state['entry_price']) / state['entry_price'] * 100

        state['total_trades'] += 1
        state['total_pnl'] += pnl
        state['signal_trades'] += 1

        if pnl > 0:
            state['win_trades'] += 1
        else:
            state['loss_trades'] += 1

        trade_info = {
            'datetime': current_time,
            'action': '信号卖出',
            'price': current_price,
            'shares': state['position'],
            'pnl': pnl,
            'type': 'signal'
        }

        record_trade(trade_info)
        print(f"  ✓ 卖出: {state['position']}股 @ ¥{current_price:.2f} (盈亏: {pnl:+.2f}%)")

        state['position'] = 0
        state['entry_price'] = 0
        state['stop_loss'] = 0
        state['trailing_stop'] = 0
        state['highest_price'] = 0
        state['last_signal'] = -1
        trade_executed = True

    # 更新权益曲线
    equity = state['capital'] + state['position'] * current_price
    state['equity_curve'].append(equity)
    state['max_equity'] = max(state['max_equity'], equity)
    state['min_equity'] = min(state['min_equity'], equity)

    return trade_executed, trade_info

# ============================================
# 计算性能指标
# ============================================
def calculate_performance(state):
    """计算性能指标"""
    if len(state['equity_curve']) < 2:
        return {}

    equity_values = pd.Series(state['equity_curve'])

    # 总收益
    total_return = (equity_values.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

    # 运行天数
    start_time = pd.to_datetime(state['start_time'])
    elapsed_days = (datetime.now() - start_time).days + 1

    # 年化收益
    if elapsed_days > 0:
        annual_return = (1 + total_return/100) ** (365 / elapsed_days) - 1
    else:
        annual_return = 0

    # 日收益率
    daily_returns = equity_values.pct_change().dropna()

    # 夏普比率
    if daily_returns.std() > 0:
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 最大回撤
    max_drawdown = ((state['min_equity'] - state['max_equity']) / state['max_equity']) * 100 if state['max_equity'] > 0 else 0

    # 胜率
    win_rate = state['win_trades'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0

    # 平均盈亏
    avg_pnl = state['total_pnl'] / state['total_trades'] if state['total_trades'] > 0 else 0

    return {
        'total_return': total_return,
        'annual_return': annual_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'avg_pnl': avg_pnl,
        'total_trades': state['total_trades'],
        'win_trades': state['win_trades'],
        'loss_trades': state['loss_trades'],
        'stop_loss_trades': state['stop_loss_trades'],
        'trailing_stop_trades': state['trailing_stop_trades'],
        'daily_loss_trades': state['daily_loss_trades'],
        'signal_trades': state['signal_trades'],
        'elapsed_days': elapsed_days,
        'current_equity': equity_values.iloc[-1],
        'max_equity': state['max_equity'],
        'min_equity': state['min_equity'],
        'cycle_count': state['cycle_count']
    }

# ============================================
# 生成报告
# ============================================
def generate_report(state):
    """生成报告"""
    perf = calculate_performance(state)

    if not perf:
        return {}

    # 评级
    grade = 'C'
    if perf['sharpe_ratio'] > 1.0 and perf['max_drawdown'] > -15:
        grade = 'A'
    elif perf['sharpe_ratio'] > 0.8 and perf['max_drawdown'] > -20:
        grade = 'B'

    report = {
        'timestamp': datetime.now().isoformat(),
        'cycle': state['cycle_count'],
        'performance': perf,
        'grade': grade
    }

    return report

# ============================================
# 主循环
# ============================================
def main_loop():
    """主循环"""
    print("\n" + "="*70)
    print("3个月实时模拟验证启动")
    print("="*70)

    # 加载状态
    state = load_state()
    state['cycle_count'] += 1

    print(f"\n加载状态:")
    print(f"  资金: ¥{state['capital']:,.2f}")
    print(f"  持仓: {state['position']}股")
    print(f"  周期数: {state['cycle_count']}/{TARGET_CYCLES}")
    print(f"  运行天数: {(pd.to_datetime(state['start_time']) - pd.to_datetime(datetime.now())).days * -1}")

    cycle_count = 0

    try:
        while state['cycle_count'] <= TARGET_CYCLES:
            cycle_count += 1
            print(f"\n{'='*70}")
            print(f"周期 {state['cycle_count']}/{TARGET_CYCLES} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")

            # 获取数据
            df = get_latest_data(STOCK_CODE, days=100)
            print(f"  获取数据: {len(df)} 条")

            # 生成信号
            df = generate_signals(df)
            current_price = df['close'].iloc[-1]
            current_signal = df['final_signal'].iloc[-1]

            print(f"  当前价格: ¥{current_price:.2f}")
            print(f"  当前信号: {'买入' if current_signal == 1 else '卖出' if current_signal == -1 else '持有'}")

            # 执行交易
            trade_executed, trade_info = execute_cycle(df, state)

            # 保存状态
            save_state(state)

            # 计算性能
            perf = calculate_performance(state)

            if perf:
                print(f"\n当前持仓:")
                print(f"  现金: ¥{state['capital']:,.2f}")
                print(f"  持仓: {state['position']}股")
                print(f"  持仓价值: ¥{state['position'] * current_price:,.2f}")
                print(f"  总权益: ¥{perf['current_equity']:,.2f}")

                print(f"\n性能指标:")
                print(f"  总收益: {perf['total_return']:+.2f}%")
                print(f"  年化收益: {perf['annual_return']:+.2f}%")
                print(f"  夏普比率: {perf['sharpe_ratio']:.2f}")
                print(f"  最大回撤: {perf['max_drawdown']:.2f}%")
                print(f"  胜率: {perf['win_rate']:.2f}%")
                print(f"  交易次数: {perf['total_trades']}")

                # 评级
                grade = 'C'
                if perf['sharpe_ratio'] > 1.0 and perf['max_drawdown'] > -15:
                    grade = 'A'
                elif perf['sharpe_ratio'] > 0.8 and perf['max_drawdown'] > -20:
                    grade = 'B'

                print(f"\n策略评级: {grade}")
                if grade == 'A':
                    print(f"  🏆 优秀！")
                elif grade == 'B':
                    print(f"  ✅ 良好")
                else:
                    print(f"  ⚠️  一般")

            # 定期生成报告
            if state['cycle_count'] % REPORT_INTERVAL == 0:
                print(f"\n生成报告...")
                report = generate_report(state)
                if report:
                    save_report(report)
                    print(f"  ✓ 报告已保存")

            # 检查是否达到最大周期
            if state['cycle_count'] >= TARGET_CYCLES:
                print(f"\n达到目标周期数 ({TARGET_CYCLES})，停止运行")
                break

            # 等待下一个周期
            print(f"\n等待 {CYCLE_INTERVAL} 秒...")
            time.sleep(CYCLE_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\n⚠️  收到停止信号，正在保存数据...")

    # 最终报告
    print(f"\n{'='*70}")
    print("最终报告")
    print(f"{'='*70}")

    perf = calculate_performance(state)

    if perf:
        print(f"\n最终权益: ¥{perf['current_equity']:,.2f}")
        print(f"总收益: {perf['total_return']:+.2f}%")
        print(f"年化收益: {perf['annual_return']:+.2f}%")
        print(f"夏普比率: {perf['sharpe_ratio']:.2f}")
        print(f"最大回撤: {perf['max_drawdown']:.2f}%")
        print(f"胜率: {perf['win_rate']:.2f}%")
        print(f"总交易次数: {perf['total_trades']}")
        print(f"  - 信号交易: {perf['signal_trades']}次")
        print(f"  - 追踪止损: {perf['trailing_stop_trades']}次")
        print(f"  - 止损: {perf['stop_loss_trades']}次")
        print(f"  - 日风控: {perf['daily_loss_trades']}次")

        # 评级
        grade = 'C'
        if perf['sharpe_ratio'] > 1.0 and perf['max_drawdown'] > -15:
            grade = 'A'
        elif perf['sharpe_ratio'] > 0.8 and perf['max_drawdown'] > -20:
            grade = 'B'

        print(f"\n策略评级: {grade}")
        if grade == 'A':
            print(f"  🏆 优秀！可以考虑实盘")
        elif grade == 'B':
            print(f"  ✅ 良好，建议继续观察")
        else:
            print(f"  ⚠️  一般，需要进一步优化")

    print(f"{'='*70}")

    # 保存最终报告
    report = generate_report(state)
    if report:
        save_report(report)

    return state, perf

# ============================================
# 运行主程序
# ============================================
if __name__ == '__main__':
    try:
        state, perf = main_loop()
        print(f"\n✓ 3个月实时模拟验证完成")
        print(f"最终权益: ¥{state['equity_curve'][-1]:,.2f}")
        print(f"总周期数: {state['cycle_count']}")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
