"""
进化版智能选股系统 V2
=====================
特点：
1. 选股评分 + 回测验证双重筛选
2. 动态参数自适应
3. 组合优化配置
4. 实时推荐更新
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# ============ 基础函数 ============

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

# ============ 选股评分 ============

def calculate_volatility(close_prices, period=20):
    """波动率"""
    returns = close_prices.pct_change()
    return returns.rolling(window=period).std().iloc[-1]

def calculate_trend_strength(df, lookback=60):
    """趋势强度"""
    if len(df) < lookback:
        return 0
    
    close = df['close']
    
    # 价格变化
    price_change = abs(close.iloc[-1] / close.iloc[-lookback] - 1)
    
    # MA排列
    ma20 = sma(close, 20).iloc[-1]
    ma60 = sma(close, 60).iloc[-1] if len(df) >= 60 else ma20
    ma_alignment = 1 if ma20 > ma60 else 0.5
    
    # 连续性
    returns = close.pct_change()
    consecutive = 0
    for i in range(-1, -min(lookback, len(returns)), -1):
        if i == -1:
            consecutive = 1 if returns.iloc[i] > 0 else -1
        else:
            if (returns.iloc[i] > 0 and consecutive > 0) or (returns.iloc[i] < 0 and consecutive < 0):
                consecutive += 1 if returns.iloc[i] > 0 else -1
            else:
                break
    
    continuity = min(abs(consecutive) / lookback, 1)
    
    score = (price_change * 5 + ma_alignment + continuity) / 3
    return min(max(score, 0), 1)

def calculate_max_drawdown(close_prices):
    """最大回撤"""
    peak = close_prices.expanding().max()
    drawdown = (close_prices - peak) / peak
    return drawdown.min()

# ============ 回测引擎 ============

def backtest_strategy(df, params):
    """回测策略"""
    df = df.copy()
    
    # 计算指标
    df['ma_fast'] = sma(df['close'], params['ma_fast'])
    df['ma_slow'] = sma(df['close'], params['ma_slow'])
    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
    df['rsi'] = rsi(df['close'], 14)
    df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    
    # 初始化
    cash = 100000
    shares = 0
    entry_price = 0
    stop_loss = 0
    highest = 0
    partial_sold = False
    
    trades = 0
    wins = 0
    equity = [cash]
    
    # 交易循环
    for i in range(60, len(df)):
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
        
        # 持仓管理
        if shares > 0:
            # 更新最高价和止损
            if price > highest:
                highest = price
                new_stop = highest - atr_val * params['atr_trail_mult']
                if new_stop > stop_loss:
                    stop_loss = new_stop
            
            # 止损
            if price <= stop_loss:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
                equity.append(cash)
                continue
            
            # 分批止盈1
            if not partial_sold and params.get('take_profit_1'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_1']:
                    sell_shares = int(shares * params['partial_exit_1'])
                    cash += sell_shares * price
                    shares -= sell_shares
                    partial_sold = True
                    wins += 1
            
            # 分批止盈2
            if partial_sold and params.get('take_profit_2'):
                profit_pct = (price - entry_price) / entry_price
                if profit_pct >= params['take_profit_2']:
                    cash += shares * price
                    shares = 0
                    partial_sold = False
                    equity.append(cash)
                    continue
            
            # 趋势反转
            if ma_fast < ma_slow and macd_hist < 0:
                cash += shares * price
                if price > entry_price:
                    wins += 1
                shares = 0
                partial_sold = False
        
        # 买入信号
        if ma_fast > ma_slow and shares == 0:
            # MACD确认
            if params['use_macd'] and macd_hist < 0:
                equity.append(cash)
                continue
            
            # RSI过滤
            if params['use_rsi'] and (rsi_val > 70 or rsi_val < 30):
                equity.append(cash)
                continue
            
            # 动态仓位
            position_pct = params.get('base_position', 0.30)
            if params['use_dynamic_position']:
                if volatility > 0.03:
                    position_pct *= 0.7
                elif volatility > 0.02:
                    position_pct *= 0.85
            
            position_pct = min(position_pct, 0.50)
            position_value = cash * position_pct
            shares = int(position_value / price)
            
            if shares > 0:
                cash -= shares * price
                entry_price = price
                stop_loss = price - atr_val * params['atr_stop_mult']
                highest = price
                trades += 1
        
        equity.append(cash + shares * price)
    
    # 清仓
    if shares > 0:
        cash += shares * float(df['close'].iloc[-1])
        if float(df['close'].iloc[-1]) > entry_price:
            wins += 1
    
    # 计算指标
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
        'win_rate': win_rate
    }

# ============ 智能评分 ============

def smart_score_stock(df, code, name):
    """智能评分（选股+回测双重验证）"""
    
    # 1. 基础评分
    volatility = calculate_volatility(df['close'])
    trend_strength = calculate_trend_strength(df)
    max_dd = calculate_max_drawdown(df['close'])
    
    # 波动率评分
    if 0.015 <= volatility <= 0.025:
        vol_score = 100
    elif 0.025 < volatility <= 0.035:
        vol_score = 80
    elif 0.010 <= volatility < 0.015:
        vol_score = 60
    else:
        vol_score = 20
    
    # 趋势评分
    if trend_strength >= 0.6:
        trend_score = 100
    elif trend_strength >= 0.4:
        trend_score = 70
    elif trend_strength >= 0.3:
        trend_score = 50
    else:
        trend_score = 20
    
    # 回撤评分
    if max_dd > -0.30:
        dd_score = 100
    elif max_dd > -0.40:
        dd_score = 80
    elif max_dd > -0.50:
        dd_score = 60
    elif max_dd > -0.60:
        dd_score = 40
    else:
        dd_score = 20
    
    # 基础总分
    base_score = (vol_score * 0.25 + trend_score * 0.30 + dd_score * 0.25 + 60)
    
    # 2. 回测验证
    params = {
        'ma_fast': 10, 'ma_slow': 30,
        'atr_stop_mult': 2.5, 'atr_trail_mult': 2.0,
        'use_dynamic_position': True,
        'use_macd': True, 'use_rsi': True,
        'take_profit_1': 0.10, 'take_profit_2': 0.20,
        'partial_exit_1': 0.5, 'partial_exit_2': 0.5,
        'base_position': 0.30
    }
    
    backtest_result = backtest_strategy(df, params)
    
    # 回测评分
    sharpe = backtest_result['sharpe']
    ret = backtest_result['return']
    win_rate = backtest_result['win_rate']
    
    if sharpe >= 0.5:
        sharpe_score = 100
    elif sharpe >= 0.3:
        sharpe_score = 70
    elif sharpe >= 0:
        sharpe_score = 50
    else:
        sharpe_score = 20
    
    if win_rate >= 0.60:
        win_score = 100
    elif win_rate >= 0.55:
        win_score = 70
    elif win_rate >= 0.50:
        win_score = 50
    else:
        win_score = 20
    
    # 3. 综合评分（基础40% + 回测60%）
    total_score = base_score * 0.4 + (sharpe_score + win_score) / 2 * 0.6
    
    # 判断是否推荐
    suitable = (
        sharpe >= 0.3 and
        win_rate >= 0.50 and
        total_score >= 60
    )
    
    return {
        'code': code,
        'name': name,
        'score': float(total_score),
        'base_score': float(base_score),
        'volatility': float(volatility),
        'trend_strength': float(trend_strength),
        'max_dd': float(max_dd),
        'sharpe': float(sharpe),
        'return': float(ret),
        'win_rate': float(win_rate),
        'trades': int(backtest_result['trades']),
        'suitable': bool(suitable)
    }

# ============ 主程序 ============

def main():
    """主函数"""
    print("="*70)
    print("进化版智能选股系统 V2")
    print("="*70)
    
    # 股票名称
    names = {
        '600519': '茅台', '000858': '五粮液', '000568': '泸州老窖', '000596': '古井贡酒',
        '002304': '洋河股份', '002594': '比亚迪', '300750': '宁德时代', '601012': '隆基绿能',
        '002129': 'TCL中环', '600438': '通威股份', '601318': '中国平安', '601398': '工商银行',
        '600036': '招商银行', '601166': '兴业银行', '000001': '平安银行', '600276': '恒瑞医药',
        '000538': '云南白药', '300760': '迈瑞医疗', '002007': '华兰生物', '000661': '长春高新',
        '002415': '海康威视', '002230': '科大讯飞', '600588': '用友网络', '000725': '京东方A',
        '002475': '立讯精密', '000333': '美的集团', '000651': '格力电器', '600887': '伊利股份',
        '000895': '双汇发展'
    }
    
    results = []
    
    print("\n【智能评分】(选股40% + 回测60%)\n")
    print(f"{'股票':<10} {'综合':<6} {'夏普':<6} {'收益':<8} {'胜率':<6} {'交易':<6} {'状态'}")
    print("-" * 70)
    
    for code, name in names.items():
        filepath = Path(f'data/real_{code}.csv')
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        if 'close' not in df.columns or len(df) < 60:
            continue
        
        # 智能评分
        result = smart_score_stock(df, code, name)
        results.append(result)
        
        # 显示
        status = "✅" if result['suitable'] else "❌"
        print(f"{name:<10} {result['score']:5.0f}  {result['sharpe']:5.2f}  {result['return']*100:+6.1f}%  {result['win_rate']*100:5.1f}%  {result['trades']:3d}次  {status}")
    
    # 排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 统计
    suitable = [r for r in results if r['suitable']]
    excellent = [r for r in results if r['sharpe'] >= 0.5]
    
    print("\n" + "="*70)
    print("筛选结果")
    print("="*70)
    
    print(f"\n总股票数: {len(results)}")
    print(f"推荐股票: {len(suitable)}只 ({len(suitable)/len(results)*100:.1f}%)")
    print(f"优秀股票: {len(excellent)}只 (夏普≥0.5)")
    
    if suitable:
        print("\n【推荐股票池】")
        print(f"\n{'股票':<10} {'综合评分':<10} {'夏普':<8} {'收益':<10} {'胜率':<8} {'推荐等级'}")
        print("-" * 70)
        
        for r in suitable[:10]:
            if r['sharpe'] >= 0.5:
                level = "⭐⭐⭐ 强烈推荐"
            elif r['sharpe'] >= 0.3:
                level = "⭐⭐ 推荐"
            else:
                level = "⭐ 可选"
            
            print(f"{r['name']:<10} {r['score']:<10.0f} {r['sharpe']:<8.2f} {r['return']*100:<10.1f}% {r['win_rate']*100:<8.1f}% {level}")
    
    # 保存结果
    output = {
        'screen_time': datetime.now().isoformat(),
        'total_stocks': len(results),
        'suitable_stocks': len(suitable),
        'excellent_stocks': len(excellent),
        'suitable_rate': len(suitable) / len(results),
        'stocks': results,
        'recommended': suitable
    }
    
    with open('data/smart_screening_v2.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: data/smart_screening_v2.json")
    
    # 生成组合建议
    if suitable:
        print("\n" + "="*70)
        print("组合配置建议")
        print("="*70)
        
        # 选前5只
        top5 = suitable[:5]
        
        print(f"\n推荐组合 (前{len(top5)}只):")
        for i, r in enumerate(top5, 1):
            print(f"  {i}. {r['name']}: 评分{r['score']:.0f}, 夏普{r['sharpe']:.2f}, 收益{r['return']*100:+.1f}%")
        
        # 等权重配置
        print(f"\n建议配置:")
        print(f"  总资金: 100,000元")
        print(f"  每只股票: {100000/len(top5):.0f}元 ({100/len(top5):.1f}%)")
        print(f"  预期组合夏普: {np.mean([r['sharpe'] for r in top5]):.2f}")
        print(f"  预期组合收益: {np.mean([r['return'] for r in top5])*100:+.1f}%")
    
    return suitable

if __name__ == '__main__':
    main()
