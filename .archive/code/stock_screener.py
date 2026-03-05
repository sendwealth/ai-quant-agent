"""
智能选股系统
============
目标：筛选出适合趋势跟踪策略的股票
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# 导入策略函数
def sma(data, period):
    return data.rolling(window=period).mean()

def calculate_volatility(close_prices, period=20):
    """计算波动率"""
    returns = close_prices.pct_change()
    return returns.rolling(window=period).std().iloc[-1]

def calculate_trend_strength(df, lookback=60):
    """计算趋势强度 (0-1)"""
    if len(df) < lookback:
        return 0
    
    close = df['close']
    
    # 1. 价格变化幅度
    price_change = abs(close.iloc[-1] / close.iloc[-lookback] - 1)
    
    # 2. MA排列
    ma20 = sma(close, 20).iloc[-1]
    ma60 = sma(close, 60).iloc[-1] if len(df) >= 60 else ma20
    ma_alignment = 1 if ma20 > ma60 else 0.5
    
    # 3. 连续性（连续上涨/下跌的天数）
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
    
    # 4. 综合评分
    score = (price_change * 5 + ma_alignment + continuity) / 3
    score = min(max(score, 0), 1)  # 归一化到0-1
    
    return score

def calculate_max_drawdown(close_prices):
    """计算最大回撤"""
    peak = close_prices.expanding().max()
    drawdown = (close_prices - peak) / peak
    return drawdown.min()

def calculate_liquidity(df, period=20):
    """计算流动性"""
    if 'volume' not in df.columns:
        return 0
    
    avg_volume = df['volume'].rolling(window=period).mean().iloc[-1]
    return avg_volume

def score_stock(df, code, name):
    """为股票打分 (0-100)"""
    if len(df) < 60 or 'close' not in df.columns:
        return {
            'code': code,
            'name': name,
            'score': 0,
            'volatility': 0,
            'trend_strength': 0,
            'max_dd': 0,
            'suitable': False
        }
    
    scores = {}
    
    # 1. 波动率评分 (理想范围: 1.5%-3.5%)
    volatility = calculate_volatility(df['close'])
    scores['volatility'] = volatility
    
    if 0.015 <= volatility <= 0.025:
        vol_score = 100
    elif 0.025 < volatility <= 0.035:
        vol_score = 80
    elif 0.010 <= volatility < 0.015:
        vol_score = 60
    else:
        vol_score = 20
    
    # 2. 趋势强度评分
    trend_strength = calculate_trend_strength(df)
    scores['trend_strength'] = trend_strength
    
    if trend_strength >= 0.6:
        trend_score = 100
    elif trend_strength >= 0.4:
        trend_score = 70
    elif trend_strength >= 0.3:
        trend_score = 50
    else:
        trend_score = 20
    
    # 3. 最大回撤评分 (理想: >-40%)
    max_dd = calculate_max_drawdown(df['close'])
    scores['max_dd'] = max_dd
    
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
    
    # 4. 流动性评分
    liquidity = calculate_liquidity(df)
    scores['liquidity'] = liquidity
    
    if liquidity > 1000000:  # 日均成交量>100万
        liq_score = 100
    elif liquidity > 500000:
        liq_score = 80
    elif liquidity > 100000:
        liq_score = 60
    else:
        liq_score = 40
    
    # 5. 综合评分 (加权平均)
    total_score = (
        vol_score * 0.25 +
        trend_score * 0.30 +
        dd_score * 0.25 +
        liq_score * 0.20
    )
    
    # 判断是否适合
    suitable = (
        vol_score >= 60 and
        trend_score >= 50 and
        dd_score >= 40 and
        total_score >= 60
    )
    
    return {
        'code': code,
        'name': name,
        'score': total_score,
        'volatility': volatility,
        'trend_strength': trend_strength,
        'max_dd': max_dd,
        'vol_score': vol_score,
        'trend_score': trend_score,
        'dd_score': dd_score,
        'liq_score': liq_score,
        'suitable': suitable
    }

def screen_stocks():
    """筛选股票"""
    print("="*70)
    print("智能选股系统")
    print("="*70)
    
    # 股票名称映射
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
    
    # 评分所有股票
    results = []
    
    print("\n【股票评分】\n")
    
    for code, name in names.items():
        filepath = Path(f'data/real_{code}.csv')
        
        if not filepath.exists():
            continue
        
        df = pd.read_csv(filepath)
        
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        # 评分
        result = score_stock(df, code, name)
        results.append(result)
        
        # 显示
        status = "✅" if result['suitable'] else "❌"
        print(f"  {status} {name:<8} 评分:{result['score']:5.1f} | 波动{result['volatility']*100:.1f}% | 趋势{result['trend_strength']:.2f} | 回撤{result['max_dd']*100:.1f}%")
    
    # 排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 统计
    suitable_stocks = [r for r in results if r['suitable']]
    
    print("\n" + "="*70)
    print("筛选结果")
    print("="*70)
    
    print(f"\n总股票数: {len(results)}")
    print(f"适合趋势跟踪: {len(suitable_stocks)}只 ({len(suitable_stocks)/len(results)*100:.1f}%)")
    
    if suitable_stocks:
        print("\n【推荐股票池】(评分≥60)")
        print(f"\n{'股票':<10} {'评分':<8} {'波动率':<10} {'趋势强度':<10} {'最大回撤':<10} {'状态'}")
        print("-" * 70)
        
        for r in suitable_stocks:
            status = "✅ 推荐" if r['score'] >= 70 else "⚠️ 可选"
            print(f"{r['name']:<10} {r['score']:<8.1f} {r['volatility']*100:<10.1f}% {r['trend_strength']:<10.2f} {r['max_dd']*100:<10.1f}% {status}")
    
    # 保存结果
    output = {
        'screen_time': datetime.now().isoformat(),
        'total_stocks': len(results),
        'suitable_stocks': len(suitable_stocks),
        'suitable_rate': len(suitable_stocks) / len(results),
        'stocks': results,
        'recommended': [r for r in suitable_stocks if r['score'] >= 70]
    }
    
    with open('data/stock_screening_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: data/stock_screening_results.json")
    
    # 生成推荐列表
    if suitable_stocks:
        print("\n" + "="*70)
        print("推荐配置")
        print("="*70)
        
        recommended = [r for r in suitable_stocks if r['score'] >= 70]
        
        if recommended:
            print(f"\n强烈推荐 ({len(recommended)}只):")
            for r in recommended[:5]:  # 最多5只
                print(f"  - {r['name']} ({r['code']}): 评分{r['score']:.0f}")
            
            print(f"\n建议配置:")
            print(f"  资金: 100,000元")
            print(f"  股票: {', '.join([r['name'] for r in recommended[:3]])}")
            print(f"  仓位: 平均分配")
    
    return suitable_stocks

if __name__ == '__main__':
    screen_stocks()
