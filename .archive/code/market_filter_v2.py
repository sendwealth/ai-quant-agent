"""
市场环境过滤策略 - 使用akshare获取大盘
=========================================
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

try:
    import akshare as ak
    HAS_AKSHARE = True
    print("✅ akshare已安装")
except:
    HAS_AKSHARE = False
    print("❌ akshare未安装")

def get_market_index_akshare():
    """使用akshare获取上证指数"""
    if not HAS_AKSHARE:
        return None
    
    try:
        print("\n获取大盘指数数据(akshare)...")
        
        df = ak.stock_zh_index_daily(symbol="sh000001")
        
        if df is None or len(df) < 100:
            print("❌ 获取失败")
            return None
        
        # 过滤最近3年
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= '2021-03-01'].reset_index(drop=True)
        
        # 计算MA
        df['ma20'] = df['close'].rolling(20).mean()
        df['ma60'] = df['close'].rolling(60).mean()
        
        # 趋势判断
        df['trend_up'] = df['ma20'] > df['ma60']
        
        # 格式化日期
        df['trade_date'] = df['date'].dt.strftime('%Y%m%d')
        
        # 保存
        df.to_csv('data/market_index.csv', index=False)
        
        print(f"✅ 获取成功: {len(df)}天")
        
        return df
        
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return None

# 导入策略函数
def sma(data, period):
    return data.rolling(window=period).mean()

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def detect_market_state(df, lookback=50):
    if len(df) < lookback:
        return 'unknown'
    recent = df['close'].iloc[-lookback:]
    returns = recent.pct_change().dropna()
    ma_short = sma(df['close'], 20).iloc[-1]
    ma_long = sma(df['close'], 50).iloc[-1]
    volatility = returns.std()
    trend = (recent.iloc[-1] / recent.iloc[0] - 1)
    if trend > 0.1 and ma_short > ma_long:
        return 'bull'
    elif trend < -0.1 and ma_short < ma_long:
        return 'bear'
    else:
        return 'range'

def calculate_position_size(volatility, market_state):
    base_position = 0.30
    if volatility > 0.03:
        vol_adj = 0.7
    elif volatility > 0.02:
        vol_adj = 0.85
    else:
        vol_adj = 1.0
    if market_state == 'bull':
        market_adj = 1.2
    elif market_state == 'bear':
        market_adj = 0.5
    else:
        market_adj = 0.8
    return min(base_position * vol_adj * market_adj, 0.50)

def advanced_backtest_with_filter(df, market_df, params, use_filter=True):
    """带市场过滤的回测"""
    df = df.copy()
    
    # 计算指标
    df['ma_fast'] = sma(df['close'], params['ma_fast'])
    df['ma_slow'] = sma(df['close'], params['ma_slow'])
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    partial_sold = False
    
    trades = 0
    wins = 0
    filtered_trades = 0
    equity = [cash]
    
    for i in range(60, len(df)):
        date = str(df['datetime'].iloc[i])
        price = float(df['close'].iloc[i])
        atr_val = float(df['atr'].iloc[i])
        ma_fast = float(df['ma_fast'].iloc[i])
        ma_slow = float(df['ma_slow'].iloc[i])
        rsi_val = float(df['rsi'].iloc[i])
        volatility = float(df['volatility'].iloc[i])
        macd_hist = float(df['macd_hist'].iloc[i])
        
        if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(atr_val):
            equity.append(cash + shares * price)
            continue
        
        market_state = detect_market_state(df.iloc[:i+1])
        
        if shares > 0:
            if price > highest:
                highest = price
                new_stop = highest - atr_val * params['atr_trail_mult']
                if new_stop > stop_loss:
                    stop_loss = new_stop
            
            if price <= stop_loss:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
                equity.append(cash)
                continue
            
            if not partial_sold and params.get('take_profit_1'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_1']:
                    sell_shares = int(shares * params['partial_exit_1'])
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_sold = True
                    wins += 1
            
            if partial_sold and params.get('take_profit_2'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_2']:
                    cash += shares * price
                    shares = 0
                    partial_sold = False
                    equity.append(cash)
                    continue
            
            if ma_fast < ma_slow and macd_hist < 0:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
        
        if ma_fast > ma_slow and shares == 0:
            if params['use_macd'] and macd_hist < 0:
                equity.append(cash)
                continue
            
            if params['use_rsi'] and (rsi_val > 70 or rsi_val < 30):
                equity.append(cash)
                continue
            
            # 市场环境过滤
            if use_filter and market_df is not None:
                market_row = market_df[market_df['trade_date'] == date]
                
                if len(market_row) > 0:
                    market_trend_up = market_row['trend_up'].iloc[0]
                    
                    if not market_trend_up:
                        filtered_trades += 1
                        equity.append(cash)
                        continue
            
            position_pct = calculate_position_size(volatility, market_state)
            position_value = cash * position_pct
            shares = int(position_value / price)
            
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * params['atr_stop_mult']
                highest = price
                trades += 1
        
        equity.append(cash + shares * price)
    
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
        if float(df['close'].iloc[-1]) > entry_price:
            wins += 1
    
    final = cash
    ret = (final - 100000) / 100000
    
    eq = pd.Series(equity)
    rets = eq.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    
    peak = eq.expanding().max()
    dd = (eq - peak) / peak
    max_dd = dd.min()
    
    win_rate = wins / trades if trades > 0 else 0
    
    return {
        'return': ret,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'trades': trades,
        'win_rate': win_rate,
        'filtered_trades': filtered_trades
    }

def main():
    """主函数"""
    print("="*70)
    print("市场环境过滤策略验证")
    print("="*70)
    
    # 获取大盘数据
    market_df = get_market_index_akshare()
    
    if market_df is None:
        print("\n❌ 无法获取大盘数据")
        return
    
    # 策略参数
    params = {
        'ma_fast': 10, 'ma_slow': 30,
        'atr_stop_mult': 2.5, 'atr_trail_mult': 2.0,
        'use_dynamic_position': True,
        'use_macd': True, 'use_rsi': True,
        'take_profit_1': 0.10, 'take_profit_2': 0.20,
        'partial_exit_1': 0.5, 'partial_exit_2': 0.5
    }
    
    print(f"\n【策略参数】")
    print(f"  MA: {params['ma_fast']}/{params['ma_slow']}, ATR: {params['atr_stop_mult']}x")
    print(f"  ✨ 市场过滤: 大盘MA20>MA60时才交易")
    
    # 测试股票
    test_stocks = [
        ('300750', '宁德时代'), ('601318', '中国平安'), ('002475', '立讯精密'),
        ('002594', '比亚迪'), ('000333', '美的集团'), ('601398', '工商银行'),
        ('002230', '科大讯飞'), ('600276', '恒瑞医药'), ('000651', '格力电器'),
        ('600887', '伊利股份'), ('002415', '海康威视'), ('601012', '隆基绿能')
    ]
    
    print(f"\n【测试{len(test_stocks)}只股票】\n")
    
    results_without = []
    results_with = []
    
    for code, name in test_stocks:
        filepath = Path(f'data/real_{code}.csv')
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        if 'close' not in df.columns or len(df) < 60:
            continue
        
        bh_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # 无过滤
        try:
            r1 = advanced_backtest_with_filter(df, market_df, params, use_filter=False)
            results_without.append({
                'code': code, 'name': name,
                'strategy_return': r1['return'] * 100,
                'sharpe': r1['sharpe'],
                'win_rate': r1['win_rate'] * 100
            })
        except:
            pass
        
        # 有过滤
        try:
            r2 = advanced_backtest_with_filter(df, market_df, params, use_filter=True)
            results_with.append({
                'code': code, 'name': name,
                'strategy_return': r2['return'] * 100,
                'sharpe': r2['sharpe'],
                'win_rate': r2['win_rate'] * 100,
                'filtered_trades': r2['filtered_trades']
            })
        except:
            pass
        
        # 对比
        if len(results_without) > 0 and len(results_with) > 0:
            last_without = results_without[-1]
            last_with = results_with[-1]
            
            status = "✅" if last_with['sharpe'] >= 0.5 else "⚠️"
            improve = "↑" if last_with['sharpe'] > last_without['sharpe'] else "↓"
            
            print(f"  {status} {name:<8} {last_without['strategy_return']:+5.1f}%→{last_with['strategy_return']:+5.1f}% | 夏普{last_without['sharpe']:.2f}→{last_with['sharpe']:.2f} {improve}")
    
    # 汇总
    if results_without and results_with:
        print("\n" + "="*70)
        print("对比汇总")
        print("="*70)
        
        df_without = pd.DataFrame(results_without)
        df_with = pd.DataFrame(results_with)
        
        print(f"\n【无市场过滤】")
        print(f"  平均收益: {df_without['strategy_return'].mean():+.2f}%")
        print(f"  平均夏普: {df_without['sharpe'].mean():.3f}")
        print(f"  平均胜率: {df_without['win_rate'].mean():.1f}%")
        
        print(f"\n【有市场过滤】✨")
        print(f"  平均收益: {df_with['strategy_return'].mean():+.2f}%")
        print(f"  平均夏普: {df_with['sharpe'].mean():.3f}")
        print(f"  平均胜率: {df_with['win_rate'].mean():.1f}%")
        
        # 提升幅度
        return_improve = df_with['strategy_return'].mean() - df_without['strategy_return'].mean()
        sharpe_improve = df_with['sharpe'].mean() - df_without['sharpe'].mean()
        winrate_improve = df_with['win_rate'].mean() - df_without['win_rate'].mean()
        
        print(f"\n【改进效果】")
        print(f"  收益提升: {return_improve:+.2f}%")
        print(f"  夏普提升: {sharpe_improve:+.3f}")
        print(f"  胜率提升: {winrate_improve:+.1f}%")
        
        # 达标检查
        avg_sharpe = df_with['sharpe'].mean()
        avg_return = df_with['strategy_return'].mean()
        avg_winrate = df_with['win_rate'].mean()
        
        print(f"\n【实盘条件】")
        print(f"  {'✅' if avg_sharpe >= 0.5 else '❌'} 夏普≥0.5: {avg_sharpe:.3f}")
        print(f"  {'✅' if avg_winrate >= 60 else '❌'} 胜率≥60%: {avg_winrate:.1f}%")
        print(f"  {'✅' if avg_return >= 5 else '❌'} 收益≥5%: {avg_return:+.2f}%")
        
        # 保存
        report = {
            'test_time': datetime.now().isoformat(),
            'without_filter': {
                'avg_return': float(df_without['strategy_return'].mean()),
                'avg_sharpe': float(df_without['sharpe'].mean()),
                'avg_win_rate': float(df_without['win_rate'].mean())
            },
            'with_filter': {
                'avg_return': float(df_with['strategy_return'].mean()),
                'avg_sharpe': float(df_with['sharpe'].mean()),
                'avg_win_rate': float(df_with['win_rate'].mean())
            },
            'improvement': {
                'return': float(return_improve),
                'sharpe': float(sharpe_improve),
                'win_rate': float(winrate_improve)
            }
        }
        
        with open('data/market_filter_results.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存: data/market_filter_results.json")

if __name__ == '__main__':
    Path('data').mkdir(exist_ok=True)
    main()
