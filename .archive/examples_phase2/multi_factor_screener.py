#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子选股系统
Multi-Factor Stock Screening System

功能:
1. 多因子评分（估值、成长、质量、动量）
2. 智能筛选
3. 行业分散
4. 动态权重
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class MultiFactorScreener:
    """多因子选股系统"""
    
    def __init__(self):
        self.pro = ts.pro_api(Config.TUSHARE_TOKEN)
        self.factors = {
            'valuation': {
                'PE': {'weight': 0.3, 'direction': 'low'},      # 市盈率，越低越好
                'PB': {'weight': 0.3, 'direction': 'low'},      # 市净率，越低越好
                'PS': {'weight': 0.2, 'direction': 'low'},      # 市销率，越低越好
                'PCF': {'weight': 0.2, 'direction': 'low'},     # 市现率，越低越好
            },
            'growth': {
                'revenue_growth': {'weight': 0.4, 'direction': 'high'},  # 营收增长率
                'profit_growth': {'weight': 0.4, 'direction': 'high'},   # 利润增长率
                'roe_growth': {'weight': 0.2, 'direction': 'high'},      # ROE增长
            },
            'quality': {
                'ROE': {'weight': 0.4, 'direction': 'high'},    # 净资产收益率
                'ROA': {'weight': 0.3, 'direction': 'high'},    # 总资产收益率
                'gross_margin': {'weight': 0.3, 'direction': 'high'},  # 毛利率
            },
            'momentum': {
                'price_momentum_1m': {'weight': 0.3, 'direction': 'high'},  # 1个月动量
                'price_momentum_3m': {'weight': 0.3, 'direction': 'high'},  # 3个月动量
                'volume_ratio': {'weight': 0.2, 'direction': 'high'},       # 量比
                'turnover_rate': {'weight': 0.2, 'direction': 'mid'},      # 换手率
            }
        }
        
    def get_stock_data(self, ts_code, start_date, end_date):
        """获取股票数据"""
        try:
            # 获取日线数据
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            if df.empty:
                return None
            
            # 获取财务数据
            df_basic = self.pro.daily_basic(ts_code=ts_code, start_date=start_date, end_date=end_date)
            df_fina = self.pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            return {
                'price': df,
                'basic': df_basic,
                'financial': df_fina
            }
        except Exception as e:
            print(f"获取数据失败 {ts_code}: {e}")
            return None
    
    def calculate_factors(self, stock_data):
        """计算因子值"""
        if not stock_data or stock_data['price'].empty:
            return None
        
        factors = {}
        price_df = stock_data['price'].sort_values('trade_date')
        basic_df = stock_data['basic']
        fina_df = stock_data['financial']
        
        # 估值因子
        if not basic_df.empty:
            latest_basic = basic_df.iloc[-1]
            factors['PE'] = latest_basic.get('pe', np.nan)
            factors['PB'] = latest_basic.get('pb', np.nan)
            factors['PS'] = latest_basic.get('ps', np.nan)
            factors['PCF'] = latest_basic.get('pcf', np.nan)
        
        # 质量因子
        if not fina_df.empty:
            latest_fina = fina_df.iloc[-1]
            factors['ROE'] = latest_fina.get('roe', np.nan)
            factors['ROA'] = latest_fina.get('roa', np.nan)
            factors['gross_margin'] = latest_fina.get('grossprofit_margin', np.nan)
            
            # 成长因子
            if len(fina_df) >= 2:
                current = fina_df.iloc[-1]
                previous = fina_df.iloc[-2]
                
                # 营收增长率
                if 'revenue' in current and 'revenue' in previous and previous['revenue'] != 0:
                    factors['revenue_growth'] = (current['revenue'] - previous['revenue']) / abs(previous['revenue'])
                
                # 利润增长率
                if 'profit_dedt' in current and 'profit_dedt' in previous and previous['profit_dedt'] != 0:
                    factors['profit_growth'] = (current['profit_dedt'] - previous['profit_dedt']) / abs(previous['profit_dedt'])
        
        # 动量因子
        if len(price_df) >= 60:
            # 1个月动量
            price_1m_ago = price_df.iloc[-20]['close']
            price_now = price_df.iloc[-1]['close']
            factors['price_momentum_1m'] = (price_now - price_1m_ago) / price_1m_ago
            
            # 3个月动量
            price_3m_ago = price_df.iloc[-60]['close']
            factors['price_momentum_3m'] = (price_now - price_3m_ago) / price_3m_ago
            
            # 量比（最近5日平均成交量 / 前20日平均成交量）
            vol_5d = price_df.tail(5)['vol'].mean()
            vol_20d = price_df.tail(20)['vol'].mean()
            factors['volume_ratio'] = vol_5d / vol_20d if vol_20d > 0 else np.nan
            
            # 换手率
            factors['turnover_rate'] = price_df.tail(5)['turnover_rate'].mean()
        
        return factors
    
    def normalize_factor(self, value, all_values, direction='high'):
        """归一化因子"""
        if np.isnan(value):
            return 0.5
        
        # 去除NaN值
        valid_values = [v for v in all_values if not np.isnan(v)]
        if not valid_values:
            return 0.5
        
        min_val = min(valid_values)
        max_val = max(valid_values)
        
        if max_val == min_val:
            return 0.5
        
        # 归一化到0-1
        normalized = (value - min_val) / (max_val - min_val)
        
        # 根据方向调整
        if direction == 'low':
            normalized = 1 - normalized
        elif direction == 'mid':
            # 中间值最优
            mid = (min_val + max_val) / 2
            if value >= mid:
                normalized = 1 - (value - mid) / (max_val - mid)
            else:
                normalized = (value - min_val) / (mid - min_val)
        
        return normalized
    
    def score_stock(self, stock_code, stock_data, all_stocks_data):
        """评分单只股票"""
        factors = self.calculate_factors(stock_data)
        if not factors:
            return None
        
        # 收集所有股票的因子值（用于归一化）
        all_factor_values = {factor: [] for factor in factors.keys()}
        for data in all_stocks_data.values():
            if data and 'factors' in data:
                for factor, value in data['factors'].items():
                    if not np.isnan(value):
                        all_factor_values[factor].append(value)
        
        # 计算各维度得分
        scores = {}
        total_score = 0
        
        for category, category_factors in self.factors.items():
            category_score = 0
            category_weight_sum = 0
            
            for factor_name, factor_config in category_factors.items():
                if factor_name in factors:
                    # 归一化
                    normalized = self.normalize_factor(
                        factors[factor_name],
                        all_factor_values.get(factor_name, [factors[factor_name]]),
                        factor_config['direction']
                    )
                    
                    # 加权
                    weight = factor_config['weight']
                    category_score += normalized * weight
                    category_weight_sum += weight
            
            if category_weight_sum > 0:
                category_score /= category_weight_sum
                scores[category] = category_score
                total_score += category_score
        
        # 总分（平均）
        if scores:
            total_score /= len(scores)
        
        return {
            'code': stock_code,
            'factors': factors,
            'scores': scores,
            'total_score': total_score
        }
    
    def screen_stocks(self, stock_pool, top_n=10):
        """筛选股票"""
        print(f"\n{'='*70}")
        print(f"多因子选股 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 获取数据
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
        
        all_stocks_data = {}
        
        # 第一轮：获取所有股票数据
        print("📊 获取股票数据...")
        for stock in stock_pool:
            code = stock['code']
            print(f"  {stock['name']} ({code})...", end=' ')
            
            data = self.get_stock_data(code, start_date, end_date)
            if data:
                # 预计算因子
                factors = self.calculate_factors(data)
                if factors:
                    all_stocks_data[code] = {
                        'data': data,
                        'factors': factors,
                        'info': stock
                    }
                    print("✅")
                else:
                    print("⚠️ 因子计算失败")
            else:
                print("❌ 数据获取失败")
        
        print(f"\n成功获取 {len(all_stocks_data)}/{len(stock_pool)} 只股票数据\n")
        
        # 第二轮：评分
        print("📈 计算因子得分...")
        scored_stocks = []
        
        for code, data in all_stocks_data.items():
            score_result = self.score_stock(code, data['data'], all_stocks_data)
            if score_result:
                score_result['info'] = data['info']
                scored_stocks.append(score_result)
        
        # 排序
        scored_stocks.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 输出结果
        print(f"\n{'='*70}")
        print(f"筛选结果 TOP {top_n}")
        print(f"{'='*70}\n")
        
        for i, stock in enumerate(scored_stocks[:top_n], 1):
            info = stock['info']
            print(f"#{i} {info['name']} ({info['code']})")
            print(f"   总分: {stock['total_score']:.3f}")
            print(f"   估值: {stock['scores'].get('valuation', 0):.3f} | "
                  f"成长: {stock['scores'].get('growth', 0):.3f} | "
                  f"质量: {stock['scores'].get('quality', 0):.3f} | "
                  f"动量: {stock['scores'].get('momentum', 0):.3f}")
            print()
        
        # 保存结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(stock_pool),
            'scored_stocks': len(scored_stocks),
            'top_stocks': [
                {
                    'rank': i,
                    'code': s['code'],
                    'name': s['info']['name'],
                    'sector': s['info'].get('sector', 'N/A'),
                    'total_score': s['total_score'],
                    'scores': s['scores']
                }
                for i, s in enumerate(scored_stocks[:top_n], 1)
            ]
        }
        
        output_file = 'data/multi_factor_screening_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到 {output_file}\n")
        
        return scored_stocks[:top_n]

def main():
    """主函数"""
    # 加载股票池
    with open('data/stock_pool_extended.json', 'r', encoding='utf-8') as f:
        pool_data = json.load(f)
    
    # 合并所有股票
    stock_pool = []
    for category, data in pool_data['categories'].items():
        stock_pool.extend(data['stocks'])
    
    # 创建筛选器
    screener = MultiFactorScreener()
    
    # 执行筛选
    top_stocks = screener.screen_stocks(stock_pool, top_n=15)
    
    print(f"\n{'='*70}")
    print("选股完成！建议关注以上股票。")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
