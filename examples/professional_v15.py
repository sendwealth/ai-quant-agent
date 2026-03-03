"""
专业级策略 v15.0
================
基于2024-2025量化交易最佳实践：
1. 避免过拟合 - 样本外测试
2. 现实假设 - 交易成本+滑点
3. 风险管理 - 1%规则
4. 多指标评估 - Profit Factor, Calmar Ratio
5. 动态调整 - 波动率自适应
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ============ 配置 ============
class Config:
    # 资金管理
    INITIAL_CAPITAL = 100000
    MAX_RISK_PER_TRADE = 0.01  # 1%规则：单笔交易最大风险1%
    
    # 交易成本
    COMMISSION = 0.0003  # 0.03%手续费
    SLIPPAGE = 0.001     # 0.1%滑点
    
    # 策略参数
    MA_SHORT = 10  # 改为10减少噪音
    MA_LONG = 30
    ATR_PERIOD = 14
    ATR_STOP_MULT = 3.0  # 放宽止损
    ATR_TRAIL_MULT = 2.5
    
    # 过滤器
    MIN_VOLUME_RATIO = 1.2  # 成交量放大20%
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70

# ============ 工具函数 ============
def sma(data, period):
    return data.rolling(window=period).mean()

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

def calculate_position_size(capital, risk_per_trade, entry_price, stop_loss_price):
    """基于1%规则计算仓位"""
    risk_per_share = entry_price - stop_loss_price
    if risk_per_share <= 0:
        return 0
    
    # 计算可以买多少股
    max_risk_amount = capital * risk_per_trade
    shares = int(max_risk_amount / risk_per_share)
    
    # 确保不超过总资金的40%
    max_shares = int(capital * 0.4 / entry_price)
    return min(shares, max_shares)

# ============ 策略引擎 ============
def professional_backtest(df, config, train_ratio=0.7):
    """专业级回测，包含样本外测试"""
    df = df.copy()
    
    # 计算指标
    df['ma_s'] = sma(df['close'], config.MA_SHORT)
    df['ma_l'] = sma(df['close'], config.MA_LONG)
    df['ema_s'] = ema(df['close'], config.MA_SHORT)
    df['atr'] = atr(df['high'], df['low'], df['close'], config.ATR_PERIOD)
    df['rsi'] = rsi(df['close'], 14)
    df['vol_ma'] = df['volume'].rolling(20).mean()
    
    # 分割训练集和测试集
    split_point = int(len(df) * train_ratio)
    
    results = {'train': None, 'test': None}
    
    for phase, start_idx in [('train', 50), ('test', split_point)]:
        end_idx = split_point if phase == 'train' else len(df)
        
        cash = config.INITIAL_CAPITAL
        shares = 0
        entry_price = 0
        stop_loss = 0
        highest = 0
        
        trades = []
        equity = [cash]
        total_commission = 0
        total_slippage = 0
        
        for i in range(start_idx, end_idx):
            price = float(df['close'].iloc[i])
            high = float(df['high'].iloc[i])
            low = float(df['low'].iloc[i])
            atr_val = float(df['atr'].iloc[i])
            ma_s = float(df['ma_s'].iloc[i])
            ma_l = float(df['ma_l'].iloc[i])
            ema_s = float(df['ema_s'].iloc[i])
            rsi_val = float(df['rsi'].iloc[i])
            volume = float(df['volume'].iloc[i])
            vol_ma = float(df['vol_ma'].iloc[i])
            
            if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
                equity.append(cash + shares * price)
                continue
            
            # 追踪止损
            if shares > 0 and high > highest:
                highest = high
                new_stop = highest - atr_val * config.ATR_TRAIL_MULT
                if new_stop > stop_loss:
                    stop_loss = new_stop
            
            # 止损
            if shares > 0 and low <= stop_loss:
                sell_price = stop_loss * (1 - config.SLIPPAGE)
                commission = shares * sell_price * config.COMMISSION
                pnl = (sell_price - entry_price) / entry_price
                
                trades.append({'action': 'sell', 'type': 'stop', 'pnl': pnl})
                cash += shares * sell_price - commission
                total_commission += commission
                total_slippage += shares * stop_loss * config.SLIPPAGE
                
                shares = 0
                equity.append(cash)
                continue
            
            # 买入信号（增强版）
            should_buy = (
                shares == 0 and
                ma_s > ma_l and  # 趋势向上
                ema_s > ma_l and  # EMA确认
                rsi_val < config.RSI_OVERBOUGHT and  # 不超买
                volume > vol_ma * config.MIN_VOLUME_RATIO  # 成交量放大
            )
            
            if should_buy:
                stop_loss_price = price - atr_val * config.ATR_STOP_MULT
                position_size = calculate_position_size(
                    cash, config.MAX_RISK_PER_TRADE, price, stop_loss_price
                )
                
                if position_size > 0:
                    buy_price = price * (1 + config.SLIPPAGE)
                    commission = position_size * buy_price * config.COMMISSION
                    cost = position_size * buy_price + commission
                    
                    if cost <= cash:
                        shares = position_size
                        cash -= cost
                        entry_price = buy_price
                        stop_loss = stop_loss_price
                        highest = high
                        total_commission += commission
                        total_slippage += position_size * price * config.SLIPPAGE
                        trades.append({'action': 'buy', 'price': buy_price})
            
            # 卖出信号
            elif ma_s < ma_l and shares > 0:
                sell_price = price * (1 - config.SLIPPAGE)
                commission = shares * sell_price * config.COMMISSION
                pnl = (sell_price - entry_price) / entry_price
                
                trades.append({'action': 'sell', 'type': 'signal', 'pnl': pnl})
                cash += shares * sell_price - commission
                total_commission += commission
                total_slippage += shares * price * config.SLIPPAGE
                shares = 0
            
            equity.append(cash + shares * price)
        
        # 平仓
        if shares > 0:
            final_price = float(df['close'].iloc[end_idx-1])
            sell_price = final_price * (1 - config.SLIPPAGE)
            commission = shares * sell_price * config.COMMISSION
            cash += shares * sell_price - commission
            total_commission += commission
        
        # 计算指标
        final = cash
        ret = (final - config.INITIAL_CAPITAL) / config.INITIAL_CAPITAL
        
        eq = pd.Series(equity)
        rets = eq.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        
        peak = eq.expanding().max()
        dd = (eq - peak) / peak
        max_dd = dd.min()
        
        # Profit Factor
        sells = [t for t in trades if t['action'] == 'sell']
        profits = sum(t['pnl'] for t in sells if t['pnl'] > 0)
        losses = abs(sum(t['pnl'] for t in sells if t['pnl'] < 0))
        profit_factor = profits / losses if losses > 0 else 0
        
        # Calmar Ratio
        calmar = ret / abs(max_dd) if max_dd != 0 else 0
        
        results[phase] = {
            'return': ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'profit_factor': profit_factor,
            'calmar': calmar,
            'trades': len([t for t in trades if t['action'] == 'buy']),
            'commission': total_commission,
            'slippage': total_slippage
        }
    
    return results

# ============ 主程序 ============
print("="*70)
print("📊 专业级策略 v15.0")
print("="*70)
print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n基于2024-2025最佳实践:")
print("  ✓ 1%风险规则")
print("  ✓ 交易成本+滑点")
print("  ✓ 样本外测试(70/30)")
print("  ✓ 多维度评估")

config = Config()

data_files = {
    '五粮液': 'data/real_000858.csv',
    '比亚迪': 'data/real_002594.csv',
    '茅台': 'data/real_600519.csv',
}

all_results = []

for name, filepath in data_files.items():
    if not Path(filepath).exists():
        continue
    
    df = load_data(filepath)
    results = professional_backtest(df, config)
    
    print(f"\n{'='*70}")
    print(f"{name} ({len(df)}天)")
    print("="*70)
    
    for phase in ['train', 'test']:
        r = results[phase]
        status = "✅" if r['return'] > 0 else "❌"
        
        print(f"\n【{phase.upper()} SET】")
        print(f"  {status} 收益: {r['return']*100:+.2f}%")
        print(f"  夏普: {r['sharpe']:.2f}")
        print(f"  最大回撤: {r['max_dd']*100:.1f}%")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")
        print(f"  Calmar Ratio: {r['calmar']:.2f}")
        print(f"  交易次数: {r['trades']}")
        print(f"  总手续费: ¥{r['commission']:.2f}")
        print(f"  总滑点: ¥{r['slippage']:.2f}")
    
    all_results.append({'name': name, 'results': results})

# 汇总
print("\n" + "="*70)
print("📈 汇总对比")
print("="*70)

print("\n【训练集表现】")
train_returns = [r['results']['train']['return'] for r in all_results]
avg_train = np.mean(train_returns) * 100
print(f"平均收益: {avg_train:+.2f}%")

print("\n【测试集表现】(样本外)")
test_returns = [r['results']['test']['return'] for r in all_results]
avg_test = np.mean(test_returns) * 100
print(f"平均收益: {avg_test:+.2f}%")

print("\n【过拟合检测】")
overfitting = avg_train - avg_test
if overfitting > 5:
    print(f"⚠️ 可能过拟合 (训练-测试={overfitting:+.1f}%)")
elif overfitting > 0:
    print(f"✅ 轻微过拟合 (训练-测试={overfitting:+.1f}%)")
else:
    print(f"✅ 无过拟合 (测试优于训练)")

print("="*70)
