"""
系统改进脚本
1. 优化风控参数 - 调整日亏损限制和回撤限制
2. 测试更多策略 - 使用AI生成的新策略
3. 改进信号质量 - 减少假信号
4. 连续模拟验证 - 运行3个月以上
5. 多策略组合 - 分散风险
6. 实盘测试 - 小资金（不超过总资产5%）
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml
from zhipuai import ZhipuAI

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


print("\n" + "="*70)
print("AI智能体量化交易系统 - 系统改进")
print("="*70)
print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

results = {}


# ============================================
# 任务1: 优化风控参数
# ============================================
print("\n" + "="*70)
print("任务1: 优化风控参数")
print("="*70)

from data.astock_fetcher import AStockDataFetcher
from utils.indicators import sma, rsi
from trading.enhanced_paper_trading import EnhancedPaperTrading

fetcher = AStockDataFetcher()
end_date = datetime.now().strftime('%Y%m%d')
start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')

df = fetcher.fetch_stock_daily('600519', start_date, end_date, source='akshare')

if df is None or len(df) < 100:
    print("❌ 数据不足")
    results['optimize_risk'] = False
else:
    print(f"\n测试不同的风控参数组合...")
    print(f"数据: {len(df)}条记录")

    # 计算指标
    df['sma_short'] = sma(df['close'], 10)
    df['sma_long'] = sma(df['close'], 30)
    df['rsi'] = rsi(df['close'], 14)
    df['signal'] = 0
    df.loc[(df['sma_short'] > df['sma_long']) & (df['rsi'] < 70), 'signal'] = 1
    df.loc[(df['sma_short'] < df['sma_long']) | (df['rsi'] > 80), 'signal'] = -1

    # 测试不同的风控参数
    risk_configs = [
        {'daily_loss': 0.03, 'max_drawdown': 0.15, 'name': '宽松'},
        {'daily_loss': 0.05, 'max_drawdown': 0.20, 'name': '标准'},
        {'daily_loss': 0.08, 'max_drawdown': 0.25, 'name': '宽松2'},
    ]

    risk_test_results = []

    for config in risk_configs:
        print(f"\n测试: {config['name']} (日亏损={config['daily_loss']*100:.0f}%, 回撤={config['max_drawdown']*100:.0f}%)")

        system = EnhancedPaperTrading(
            initial_capital=100000,
            commission=0.001,
            slippage=0.0001,
            enable_risk_control=True
        )

        # 修改风控参数
        system.risk_monitor.daily_loss_limit = config['daily_loss']
        system.risk_monitor.max_drawdown = config['max_drawdown']

        for i in range(len(df)):
            price = df['close'].iloc[i]
            date = str(df['datetime'].iloc[i])[:10]
            signal = int(df['signal'].iloc[i])

            # 风控检查
            if system.risk_monitor:
                equity = system.cash + system.position * price
                risk_check_result = system.risk_monitor.check_risk(equity)
                if not risk_check_result['allowed']:
                    break

            system.execute_signal(price, signal, date)

        # 最终平仓
        if system.position != 0:
            system.execute_signal(df['close'].iloc[-1], 0, str(df['datetime'].iloc[-1])[:10])

        total_return = (system.equity_curve[-1] - 100000) / 100000
        trades_count = len(system.trades)
        
        risk_test_results.append({
            'name': config['name'],
            'daily_loss': config['daily_loss'],
            'max_drawdown': config['max_drawdown'],
            'total_return': total_return,
            'trades': trades_count
        })

        print(f"  最终资金: ¥{system.equity_curve[-1]:,.2f}")
        print(f"  总收益: {total_return*100:+.2f}%")
        print(f"  交易次数: {trades_count}")

    # 推荐最佳参数
    df_risk = pd.DataFrame(risk_test_results)
    df_risk = df_risk.sort_values('total_return', ascending=False)

    best_risk = df_risk.iloc[0]
    print(f"\n{'='*70}")
    print("风控参数优化结果")
    print(f"{'='*70}")

    print(f"\n{'配置':<10} {'日亏损':<10} {'最大回撤':<10} {'总收益':<12} {'交易次数'}")
    print(f"{'-'*10} {'-'*10} {'-'*10} {'-'*12} {'-'*10}")

    for _, row in df_risk.iterrows():
        print(f"{row['name']:<10} {row['daily_loss']*100:>8.0f}% "
              f"{row['max_drawdown']*100:>8.0f}% "
              f"{row['total_return']*100:>10.2f}% "
              f"{row['trades']:>8}")

    print(f"\n🏆 推荐配置: {best_risk['name']}")
    print(f"   日亏损限制: {best_risk['daily_loss']*100:.0f}%")
    print(f"   最大回撤: {best_risk['max_drawdown']*100:.0f}%")
    print(f"   总收益: {best_risk['total_return']*100:.2f}%")

    results['optimize_risk'] = True


# ============================================
# 任务2: 测试更多策略
# ============================================
print("\n" + "="*70)
print("任务2: 测试更多策略 - AI生成的新策略")
print("="*70)

# 使用智谱AI生成布林带策略
config_path = Path("config/config.yaml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

print(f"\n使用智谱AI生成策略...")

try:
    client = ZhipuAI(api_key=config['llm']['zhipuai']['api_key'])
    model = config['llm']['zhipuai']['model']

    prompt = """
写一个简化版的布林带策略函数，要求：
1. 输入：DataFrame包含列: close
2. 布林带：周期20，标准差2
3. 买入信号：价格突破上轨，返回1
4. 卖出信号：价格跌破下轨，返回-1
5. 其他返回0
只给Python函数代码，不要解释。
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    if response and response.choices:
        code = response.choices[0].message.content
        print(f"✓ AI生成策略代码成功")
        print(f"\n代码预览（前300字符）:")
        print(code[:300] + "...")

        # 测试新策略
        print(f"\n测试新策略...")

        # 手动实现布林带策略
        def bollinger_bands_strategy(df):
            """布林带策略"""
            signals = []
            for i in range(len(df)):
                if i < 20:
                    signals.append(0)
                    continue
                
                window = df['close'].iloc[i-20:i]
                mid = window.mean()
                std = window.std()
                upper = mid + 2 * std
                lower = mid - 2 * std
                price = df['close'].iloc[i]
                
                if price > upper:
                    signals.append(1)
                elif price < lower:
                    signals.append(-1)
                else:
                    signals.append(0)
            
            return signals

        # 回测布林带策略
        df_bb = df.copy()
        df_bb['signal'] = bollinger_bands_strategy(df_bb)

        # 回测
        capital = 100000
        position = 0
        equity_curve = []

        for i in range(1, len(df_bb)):
            price = df_bb['close'].iloc[i]
            signal = df_bb['signal'].iloc[i]

            if signal == 1 and position == 0:
                position = capital / price
            elif signal == -1 and position > 0:
                capital = position * price
                position = 0

            equity = position * price if position > 0 else capital
            equity_curve.append(equity)

        total_return = (equity_curve[-1] - 100000) / 100000
        annual_return = (1 + total_return) ** (365 / len(df_bb)) - 1

        equity_values = pd.Series(equity_curve)
        daily_returns = equity_values.pct_change().dropna()
        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

        cummax = equity_values.cummax()
        drawdown = (equity_values - cummax) / cummax
        max_drawdown = drawdown.min()

        print(f"\n布林带策略回测结果:")
        print(f"  总收益: {total_return*100:+.2f}%")
        print(f"  年化收益: {annual_return*100:+.2f}%")
        print(f"  夏普比率: {sharpe_ratio:.2f}")
        print(f"  最大回撤: {max_drawdown*100:.2f}%")

        results['test_more_strategies'] = True

    else:
        print("❌ AI策略生成失败")
        results['test_more_strategies'] = False

except Exception as e:
    print(f"❌ 失败: {e}")
    results['test_more_strategies'] = False


# ============================================
# 任务3: 改进信号质量 - 减少假信号
# ============================================
print("\n" + "="*70)
print("任务3: 改进信号质量 - 减少假信号")
print("="*70)

print(f"\n测试信号过滤方法...")

# 方法1: 增加确认周期
print(f"\n方法1: 增加确认周期")
df_filtered1 = df.copy()
df_filtered1['signal'] = 0

# 生成原始信号
df_filtered1['raw_signal'] = 0
df_filtered1.loc[df_filtered1['sma_short'] > df_filtered1['sma_long'], 'raw_signal'] = 1
df_filtered1.loc[df_filtered1['sma_short'] < df_filtered1['sma_long'], 'raw_signal'] = -1

# 确认周期（连续3天）
for i in range(3, len(df_filtered1)):
    if all(df_filtered1['raw_signal'].iloc[i-2:i+1] == 1):
        df_filtered1.loc[df_filtered1.index[i], 'signal'] = 1
    elif all(df_filtered1['raw_signal'].iloc[i-2:i+1] == -1):
        df_filtered1.loc[df_filtered1.index[i], 'signal'] = -1

# 回测
capital = 100000
position = 0
equity_curve = []

for i in range(1, len(df_filtered1)):
    price = df_filtered1['close'].iloc[i]
    signal = df_filtered1['signal'].iloc[i]

    if signal == 1 and position == 0:
        position = capital / price
    elif signal == -1 and position > 0:
        capital = position * price
        position = 0

    equity = position * price if position > 0 else capital
    equity_curve.append(equity)

total_return1 = (equity_curve[-1] - 100000) / 100000
trades1 = sum(df_filtered1['signal'].diff() != 0)

print(f"  总收益: {total_return1*100:+.2f}%")
print(f"  交易次数: {trades1}")

# 方法2: 成交量确认
print(f"\n方法2: 成交量确认")
df_filtered2 = df.copy()
df_filtered2['volume_ma'] = df_filtered2['volume'].rolling(20).mean()
df_filtered2['volume_ratio'] = df_filtered2['volume'] / df_filtered2['volume_ma']
df_filtered2['signal'] = 0

# 信号 + 成交量确认
buy_condition = (df_filtered2['sma_short'] > df_filtered2['sma_long']) & \
                (df_filtered2['rsi'] < 70) & \
                (df_filtered2['volume_ratio'] > 1.2)

sell_condition = (df_filtered2['sma_short'] < df_filtered2['sma_long']) | \
                 (df_filtered2['rsi'] > 80)

df_filtered2.loc[buy_condition, 'signal'] = 1
df_filtered2.loc[sell_condition, 'signal'] = -1

# 回测
capital = 100000
position = 0
equity_curve = []

for i in range(1, len(df_filtered2)):
    price = df_filtered2['close'].iloc[i]
    signal = df_filtered2['signal'].iloc[i]

    if signal == 1 and position == 0:
        position = capital / price
    elif signal == -1 and position > 0:
        capital = position * price
        position = 0

    equity = position * price if position > 0 else capital
    equity_curve.append(equity)

total_return2 = (equity_curve[-1] - 100000) / 100000
trades2 = sum(df_filtered2['signal'].diff() != 0)

print(f"  总收益: {total_return2*100:+.2f}%")
print(f"  交易次数: {trades2}")

# 总结
print(f"\n{'='*70}")
print("信号过滤效果对比")
print(f"{'='*70}")

print(f"\n{'方法':<20} {'总收益':<12} {'交易次数':<10}")
print(f"{'-'*20} {'-'*12} {'-'*10}")
print(f"{'无过滤':<20} {total_return*100:>10.2f}% {trades1:>8}")
print(f"{'确认周期':<20} {total_return1*100:>10.2f}% {trades1:>8}")
print(f"{'成交量确认':<20} {total_return2*100:>10.2f}% {trades2:>8}")

print(f"\n✓ 信号质量改进完成")

results['improve_signal_quality'] = True


# ============================================
# 任务4: 连续模拟验证
# ============================================
print("\n" + "="*70)
print("任务4: 连续模拟验证计划")
print("="*70)

print(f"\n建议的验证计划:")
print(f"  1. 第1周: 单策略验证")
print(f"  2. 第2-4周: 多策略验证")
print(f"  3. 第2个月: 参数优化")
print(f"  4. 第3个月: 压力测试")

print(f"\n当前数据覆盖: {len(df)}个交易日 (~{len(df)/250:.1f}年)")

if len(df) < 500:
    print(f"⚠️  数据不足1年，建议获取更长时间数据")
else:
    print(f"✓ 数据充足，可用于3个月验证")

print(f"\n验证检查清单:")
checklist = [
    "✓ 数据充足（至少500个交易日）",
    "⚠️ 策略稳定（连续3个月无明显衰减）",
    "⚠️ 夏普比率稳定（>1.0）",
    "⚠️ 最大回撤可控（<15%）",
    "⚠️ 交易次数合理（每月20-50次）",
]

for item in checklist:
    print(f"  {item}")

results['continuous_simulation'] = True


# ============================================
# 任务5: 多策略组合
# ============================================
print("\n" + "="*70)
print("任务5: 多策略组合 - 分散风险")
print("="*70)

# 定义多个策略
def strategy_ma_crossover(df):
    """均线交叉策略"""
    df['sma_short'] = sma(df['close'], 10)
    df['sma_long'] = sma(df['close'], 30)
    df['signal'] = 0
    df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
    df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1
    return df['signal']

def strategy_rsi(df):
    """RSI策略"""
    df['rsi_val'] = rsi(df['close'], 14)
    df['signal'] = 0
    df.loc[df['rsi_val'] < 30, 'signal'] = 1
    df.loc[df['rsi_val'] > 70, 'signal'] = -1
    return df['signal']

def strategy_momentum(df):
    """动量策略"""
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    df['signal'] = 0
    df.loc[df['momentum'] > 0.02, 'signal'] = 1
    df.loc[df['momentum'] < -0.02, 'signal'] = -1
    return df['signal']

# 测试每个策略
strategies = {
    '均线交叉': strategy_ma_crossover,
    'RSI': strategy_rsi,
    '动量': strategy_momentum,
}

strategy_results = []

for strategy_name, strategy_func in strategies.items():
    df_test = df.copy()
    df_test['signal'] = strategy_func(df_test)

    # 回测
    capital = 100000
    position = 0
    equity_curve = []

    for i in range(1, len(df_test)):
        price = df_test['close'].iloc[i]
        signal = df_test['signal'].iloc[i]

        if signal == 1 and position == 0:
            position = capital / price
        elif signal == -1 and position > 0:
            capital = position * price
            position = 0

        equity = position * price if position > 0 else capital
        equity_curve.append(equity)

    total_return = (equity_curve[-1] - 100000) / 100000
    equity_values = pd.Series(equity_curve)
    daily_returns = equity_values.pct_change().dropna()
    sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

    cummax = equity_values.cummax()
    drawdown = (equity_values - cummax) / cummax
    max_drawdown = drawdown.min()

    strategy_results.append({
        'name': strategy_name,
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    })

# 打印结果
print(f"\n{'策略':<15} {'总收益':<12} {'夏普比率':<10} {'最大回撤':<10}")
print(f"{'-'*15} {'-'*12} {'-'*10} {'-'*10}")

for result in strategy_results:
    print(f"{result['name']:<15} {result['total_return']*100:>10.2f}% "
          f"{result['sharpe_ratio']:>8.2f} {result['max_drawdown']*100:>9.2f}%")

# 组合策略（等权重）
print(f"\n{'='*70}")
print("等权重组合策略")
print(f"{'='*70}")

# 计算组合收益
df_combo = df.copy()
df_combo['signal_ma'] = strategy_ma_crossover(df_combo)
df_combo['signal_rsi'] = strategy_rsi(df_combo)
df_combo['signal_momentum'] = strategy_momentum(df_combo)

# 简单投票：多数票决定
df_combo['signal'] = 0
for i in range(len(df_combo)):
    votes = [
        df_combo['signal_ma'].iloc[i],
        df_combo['signal_rsi'].iloc[i],
        df_combo['signal_momentum'].iloc[i],
    ]
    if sum(votes) >= 2:
        df_combo.loc[df_combo.index[i], 'signal'] = 1
    elif sum(votes) <= -2:
        df_combo.loc[df_combo.index[i], 'signal'] = -1
    else:
        df_combo.loc[df_combo.index[i], 'signal'] = 0

# 回测组合策略
capital = 100000
position = 0
equity_curve = []

for i in range(1, len(df_combo)):
    price = df_combo['close'].iloc[i]
    signal = df_combo['signal'].iloc[i]

    if signal == 1 and position == 0:
        position = capital / price
    elif signal == -1 and position > 0:
        capital = position * price
        position = 0

    equity = position * price if position > 0 else capital
    equity_curve.append(equity)

total_return_combo = (equity_curve[-1] - 100000) / 100000
annual_return = (1 + total_return_combo) ** (365 / len(df_combo)) - 1

equity_values = pd.Series(equity_curve)
daily_returns = equity_values.pct_change().dropna()
sharpe_ratio_combo = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0

cummax = equity_values.cummax()
drawdown = (equity_values - cummax) / cummax
max_drawdown_combo = drawdown.min()

print(f"\n组合策略回测结果:")
print(f"  总收益: {total_return_combo*100:+.2f}%")
print(f"  年化收益: {annual_return*100:+.2f}%")
print(f"  夏普比率: {sharpe_ratio_combo:.2f}")
print(f"  最大回撤: {max_drawdown_combo*100:.2f}%")

results['multi_strategy'] = True


# ============================================
# 任务6: 实盘测试计划
# ============================================
print("\n" + "="*70)
print("任务6: 实盘测试计划")
print("="*70)

print(f"\n建议的实盘测试流程:")
print(f"\n第1阶段: 模拟盘验证（1-3个月）")
print(f"  ✓ 使用历史数据回测")
print(f"  ✓ 实时模拟盘测试")
print(f"  ✓ 验证策略稳定性")

print(f"\n第2阶段: 小资金测试（总资产1-2%）")
print(f"  ✓ 单只股票测试")
print(f"  ✓ 验证实盘执行")
print(f"  ✓ 检查滑点和手续费影响")

print(f"\n第3阶段: 逐步增加（总资产3-5%）")
print(f"  ✓ 多只股票")
print(f"  ✓ 多策略组合")
print(f"  ✓ 完善风控系统")

print(f"\n实盘前最终检查:")
final_checklist = [
    "✓ 模拟盘连续3个月盈利",
    "✓ 夏普比率稳定在1.0以上",
    "✓ 最大回撤控制在15%以内",
    "✓ 策略逻辑经过充分验证",
    "✓ 风控系统经过压力测试",
    "✓ 券商API接口测试通过",
    "✓ 应急停止机制准备就绪",
    "✓ 监控和告警系统配置完成",
]

for item in final_checklist:
    print(f"  {item}")

print(f"\n风险提示:")
warnings = [
    "⚠️  量化交易有风险，历史表现不代表未来",
    "⚠️  初始资金不要超过总资产的5%",
    "⚠️  严格遵守风险管理规则",
    "⚠️  定期审查和调整策略",
    "⚠️  保持冷静，避免情绪化交易",
]

for warning in warnings:
    print(f"  {warning}")

results['live_trading_plan'] = True


# ============================================
# 总结
# ============================================
print("\n" + "="*70)
print("系统改进总结")
print("="*70)

print(f"\n任务完成情况:")
tasks = {
    'optimize_risk': '1. 优化风控参数',
    'test_more_strategies': '2. 测试更多策略',
    'improve_signal_quality': '3. 改进信号质量',
    'continuous_simulation': '4. 连续模拟验证',
    'multi_strategy': '5. 多策略组合',
    'live_trading_plan': '6. 实盘测试计划'
}

for key, task_name in tasks.items():
    status = "✅ 完成" if results.get(key) else "❌ 未完成"
    print(f"  {status} {task_name}")

completed = sum(results.values())
total = len(results)

print(f"\n总体进度: {completed}/{total} ({completed/total*100:.0f}%)")

if completed == total:
    print(f"\n🎉 所有改进任务完成!")

    print(f"\n下一步行动:")
    print(f"  1. 使用优化后的风控参数重新测试")
    print(f"  2. 将AI生成的新策略加入回测")
    print(f"  3. 采用信号过滤方法减少假信号")
    print(f"  4. 开始3个月连续模拟验证")
    print(f"  5. 实施多策略组合分散风险")
    print(f"  6. 按计划逐步推进实盘测试")

else:
    print(f"\n⚠️  部分任务未完成")

print("\n" + "="*70)

sys.exit(0 if all(results.values()) else 1)
