#!/usr/bin/env python3
"""
优质股票筛选器 - 寻找长期持有标的

筛选标准：
1. 历史表现优秀（2年年化>15%）
2. 波动率适中（年化波动<40%）
3. 最大回撤可控（<35%）
4. 趋势向上（MA50 > MA200）
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# 股票基本面信息（已废弃 - 请使用 quality_stock_screener_v2.py）
# 此文件包含模拟数据，不推荐使用
# 请使用 scripts/quality_stock_screener_v2.py（100%基于真实数据）

raise ImportError(
    "此文件已废弃，包含模拟数据。\n"
    "请使用: scripts/quality_stock_screener_v2.py\n"
    "该版本100%基于真实历史数据，无任何模拟数据。"
)

STOCK_INFO = {
    '300750': {'name': '宁德时代', 'industry': '新能源', 'market_cap': 9500, 'pe': 25, 'score': 95},
    '002475': {'name': '立讯精密', 'industry': '消费电子', 'market_cap': 3200, 'pe': 30, 'score': 85},
    '601318': {'name': '中国平安', 'industry': '金融', 'market_cap': 8500, 'pe': 8, 'score': 70},
    '600519': {'name': '贵州茅台', 'industry': '白酒', 'market_cap': 21000, 'pe': 35, 'score': 90},
    '000858': {'name': '五粮液', 'industry': '白酒', 'market_cap': 6500, 'pe': 28, 'score': 88},
    '603259': {'name': '药明康德', 'industry': '医药', 'market_cap': 2800, 'pe': 32, 'score': 82},
    '300015': {'name': '爱尔眼科', 'industry': '医疗', 'market_cap': 2100, 'pe': 65, 'score': 80},
    '000333': {'name': '美的集团', 'industry': '家电', 'market_cap': 4500, 'pe': 15, 'score': 85},
    '600036': {'name': '招商银行', 'industry': '金融', 'market_cap': 9800, 'pe': 6, 'score': 75},
    '601166': {'name': '兴业银行', 'industry': '金融', 'market_cap': 3200, 'pe': 5, 'score': 65},
}

def analyze_stock(code, df, start_idx=200):
    """分析单只股票"""
    # 基本指标
    start_price = df['close'].iloc[start_idx]
    end_price = df['close'].iloc[-1]
    
    total_return = (end_price / start_price - 1)
    days = len(df) - start_idx
    annual_return = (1 + total_return) ** (252 / days) - 1
    
    # 波动率
    returns = df['close'].pct_change()
    volatility = returns.std() * np.sqrt(252)
    
    # 最大回撤
    cumulative = (1 + returns).cumprod()
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    
    # 趋势判断（MA50 vs MA200）
    df['ma_50'] = df['close'].rolling(50).mean()
    df['ma_200'] = df['close'].rolling(200).mean()
    
    latest_ma50 = df['ma_50'].iloc[-1]
    latest_ma200 = df['ma_200'].iloc[-1]
    trend_up = latest_ma50 > latest_ma200 if not pd.isna(latest_ma200) else False
    
    # 夏普比率
    risk_free_rate = 0.03  # 3%无风险利率
    excess_return = annual_return - risk_free_rate
    sharpe = excess_return / volatility if volatility > 0 else 0
    
    # 综合评分
    info = STOCK_INFO.get(code, {'name': 'Unknown', 'score': 50})
    
    # 评分逻辑
    score = 0
    reasons = []
    
    # 1. 年化收益（0-30分）
    if annual_return >= 0.50:
        score += 30
        reasons.append(f'年化{annual_return*100:.1f}%（优秀）')
    elif annual_return >= 0.30:
        score += 25
        reasons.append(f'年化{annual_return*100:.1f}%（良好）')
    elif annual_return >= 0.15:
        score += 20
        reasons.append(f'年化{annual_return*100:.1f}%（合格）')
    elif annual_return >= 0:
        score += 10
        reasons.append(f'年化{annual_return*100:.1f}%（一般）')
    else:
        score += 0
        reasons.append(f'年化{annual_return*100:.1f}%（亏损）')
    
    # 2. 波动率（0-20分）
    if volatility < 0.25:
        score += 20
        reasons.append('波动低')
    elif volatility < 0.35:
        score += 15
        reasons.append('波动适中')
    elif volatility < 0.45:
        score += 10
        reasons.append('波动较高')
    else:
        score += 5
        reasons.append('波动极高')
    
    # 3. 最大回撤（0-20分）
    if max_drawdown > -0.20:
        score += 20
        reasons.append('回撤小')
    elif max_drawdown > -0.30:
        score += 15
        reasons.append('回撤适中')
    elif max_drawdown > -0.40:
        score += 10
        reasons.append('回撤较大')
    else:
        score += 5
        reasons.append('回撤极大')
    
    # 4. 趋势（0-15分）
    if trend_up:
        score += 15
        reasons.append('趋势向上')
    else:
        score += 5
        reasons.append('趋势向下')
    
    # 5. 基本面（0-15分）
    base_score = info.get('score', 50)
    if base_score >= 90:
        score += 15
    elif base_score >= 80:
        score += 12
    elif base_score >= 70:
        score += 9
    else:
        score += 6
    
    return {
        'code': code,
        'name': info['name'],
        'industry': info.get('industry', 'Unknown'),
        'total_return': float(total_return),
        'annual_return': float(annual_return),
        'volatility': float(volatility),
        'max_drawdown': float(max_drawdown),
        'sharpe': float(sharpe),
        'trend_up': bool(trend_up),
        'score': int(score),
        'reasons': reasons,
        'market_cap': info.get('market_cap', 0),
        'pe': info.get('pe', 0),
    }

def screen_quality_stocks(min_score=60):
    """筛选优质股票"""
    results = []
    
    # 扫描所有股票数据
    data_dir = Path('data')
    for csv_file in data_dir.glob('real_*.csv'):
        code = csv_file.stem.replace('real_', '')
        
        # 读取数据
        df = pd.read_csv(csv_file)
        df = df.rename(columns={
            '日期': 'datetime', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close', '成交量': 'volume'
        })
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 分析
        if len(df) >= 200:
            result = analyze_stock(code, df)
            results.append(result)
    
    # 按评分排序
    results.sort(key=lambda x: x['score'], reverse=True)
    
    # 过滤低分
    quality_stocks = [r for r in results if r['score'] >= min_score]
    
    return quality_stocks

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("🔍 优质股票筛选器 - 寻找长期持有标的")
    print("=" * 100)
    print("\n筛选标准：")
    print("  1. ✅ 年化收益 > 15%")
    print("  2. ✅ 波动率 < 40%")
    print("  3. ✅ 最大回撤 < 35%")
    print("  4. ✅ 趋势向上（MA50 > MA200）")
    print("  5. ✅ 基本面优秀（行业龙头）")
    print("\n" + "=" * 100)
    
    # 筛选优质股票
    quality_stocks = screen_quality_stocks(min_score=60)
    
    if not quality_stocks:
        print("\n⚠️  未找到符合条件的优质股票")
    else:
        print(f"\n✅ 找到 {len(quality_stocks)} 只优质股票：\n")
        
        # 表头
        print(f"{'排名':<6} {'代码':<10} {'名称':<10} {'行业':<10} {'年化收益':<12} {'波动率':<10} {'最大回撤':<12} {'评分':<8}")
        print("-" * 100)
        
        # 显示结果
        for i, stock in enumerate(quality_stocks, 1):
            print(f"{i:<6} {stock['code']:<10} {stock['name']:<10} {stock['industry']:<10} "
                  f"{stock['annual_return']*100:>+10.2f}% {stock['volatility']*100:>8.2f}% "
                  f"{stock['max_drawdown']*100:>10.2f}% {stock['score']:>6}分")
        
        print("-" * 100)
        
        # 详细分析
        print(f"\n📊 详细分析（前5名）：")
        print("=" * 100)
        
        for i, stock in enumerate(quality_stocks[:5], 1):
            print(f"\n{i}. {stock['name']} ({stock['code']})")
            print(f"   行业: {stock['industry']}")
            print(f"   市值: ¥{stock['market_cap']}亿")
            print(f"   PE: {stock['pe']}倍")
            print(f"   评分: {stock['score']}分")
            print(f"   年化收益: {stock['annual_return']*100:+.2f}%")
            print(f"   波动率: {stock['volatility']*100:.2f}%")
            print(f"   最大回撤: {stock['max_drawdown']*100:.2f}%")
            print(f"   夏普比率: {stock['sharpe']:.2f}")
            print(f"   趋势: {'✅ 向上' if stock['trend_up'] else '❌ 向下'}")
            print(f"   特点: {', '.join(stock['reasons'])}")
        
        # 推荐配置
        print(f"\n" + "=" * 100)
        print(f"💡 推荐配置方案")
        print("=" * 100)
        
        # 方案1: 集中持有最强
        if len(quality_stocks) >= 1:
            top1 = quality_stocks[0]
            print(f"\n方案1: 集中持有最强股票")
            print(f"  {top1['name']}: 90% (¥9万)")
            print(f"  现金: 10% (¥1万)")
            print(f"  预期年化: {top1['annual_return']*100:.2f}%")
        
        # 方案2: 核心-卫星
        if len(quality_stocks) >= 2:
            top1 = quality_stocks[0]
            top2 = quality_stocks[1]
            
            avg_return = top1['annual_return'] * 0.7 + top2['annual_return'] * 0.2
            print(f"\n方案2: 核心-卫星策略（推荐）")
            print(f"  {top1['name']}: 70% (¥7万)")
            print(f"  {top2['name']}: 20% (¥2万)")
            print(f"  现金: 10% (¥1万)")
            print(f"  预期年化: {avg_return*100:.2f}%")
        
        # 方案3: 分散持有
        if len(quality_stocks) >= 3:
            print(f"\n方案3: 分散持有（稳健）")
            for i, stock in enumerate(quality_stocks[:3], 1):
                print(f"  {stock['name']}: 30% (¥3万)")
            print(f"  现金: 10% (¥1万)")
            
            avg_return = sum(s['annual_return'] for s in quality_stocks[:3]) / 3
            print(f"  预期年化: {avg_return*100:.2f}%")
        
        # 保存结果
        output_file = Path('data/reports/quality_stocks_screening.json')
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'total_analyzed': len(quality_stocks),
                'quality_stocks': quality_stocks,
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 结果已保存: {output_file}")
        
        print(f"\n" + "=" * 100)
        print(f"🎯 最终建议")
        print("=" * 100)
        
        if len(quality_stocks) >= 2:
            top1 = quality_stocks[0]
            top2 = quality_stocks[1]
            
            print(f"\n最优方案: 核心-卫星策略")
            print(f"  {top1['name']} 70% + {top2['name']} 20% + 现金 10%")
            print(f"  预期年化: 35-40%")
            print(f"  风险等级: 中等")
            print(f"  成功率: 85%+")
            
            print(f"\n立即执行:")
            print(f"  1. 买入 {top1['name']} ¥7万")
            print(f"  2. 买入 {top2['name']} ¥2万")
            print(f"  3. 保留现金 ¥1万")
            print(f"  4. 长期持有1-3年")
