#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多因子选股系统（演示版）
Multi-Factor Stock Screening System (Demo Version)

使用模拟数据演示功能
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

class MultiFactorScreenerDemo:
    """多因子选股系统（演示版）"""
    
    def __init__(self):
        self.demo_mode = True
        
    def generate_demo_factors(self, stock_info):
        """生成演示因子数据"""
        factors = {
            # 估值因子
            'PE': np.random.uniform(10, 50),
            'PB': np.random.uniform(1, 5),
            'PS': np.random.uniform(2, 10),
            'PCF': np.random.uniform(5, 20),
            
            # 质量因子
            'ROE': np.random.uniform(5, 25),
            'ROA': np.random.uniform(2, 15),
            'gross_margin': np.random.uniform(20, 60),
            
            # 成长因子
            'revenue_growth': np.random.uniform(-10, 40),
            'profit_growth': np.random.uniform(-20, 50),
            
            # 动量因子
            'price_momentum_1m': np.random.uniform(-15, 20),
            'price_momentum_3m': np.random.uniform(-30, 50),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'turnover_rate': np.random.uniform(1, 8)
        }
        
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
        
        return normalized
    
    def score_stock(self, stock_info, all_factor_values):
        """评分单只股票"""
        factors = self.generate_demo_factors(stock_info)
        
        # 因子配置
        factor_config = {
            'PE': {'weight': 0.3, 'direction': 'low'},
            'PB': {'weight': 0.3, 'direction': 'low'},
            'PS': {'weight': 0.2, 'direction': 'low'},
            'PCF': {'weight': 0.2, 'direction': 'low'},
            'ROE': {'weight': 0.4, 'direction': 'high'},
            'ROA': {'weight': 0.3, 'direction': 'high'},
            'gross_margin': {'weight': 0.3, 'direction': 'high'},
            'revenue_growth': {'weight': 0.4, 'direction': 'high'},
            'profit_growth': {'weight': 0.4, 'direction': 'high'},
            'price_momentum_1m': {'weight': 0.3, 'direction': 'high'},
            'price_momentum_3m': {'weight': 0.3, 'direction': 'high'},
            'volume_ratio': {'weight': 0.2, 'direction': 'high'},
            'turnover_rate': {'weight': 0.2, 'direction': 'high'}
        }
        
        # 计算各维度得分
        dimension_scores = {
            'valuation': 0,
            'quality': 0,
            'growth': 0,
            'momentum': 0
        }
        
        dimension_factors = {
            'valuation': ['PE', 'PB', 'PS', 'PCF'],
            'quality': ['ROE', 'ROA', 'gross_margin'],
            'growth': ['revenue_growth', 'profit_growth'],
            'momentum': ['price_momentum_1m', 'price_momentum_3m', 'volume_ratio', 'turnover_rate']
        }
        
        for dimension, factor_list in dimension_factors.items():
            total_weight = 0
            weighted_score = 0
            
            for factor_name in factor_list:
                if factor_name in factors and factor_name in factor_config:
                    value = factors[factor_name]
                    config = factor_config[factor_name]
                    
                    # 归一化
                    normalized = self.normalize_factor(
                        value,
                        all_factor_values.get(factor_name, [value]),
                        config['direction']
                    )
                    
                    weighted_score += normalized * config['weight']
                    total_weight += config['weight']
            
            if total_weight > 0:
                dimension_scores[dimension] = weighted_score / total_weight
        
        # 总分（平均）
        total_score = np.mean(list(dimension_scores.values()))
        
        return {
            'code': stock_info['code'],
            'name': stock_info['name'],
            'sector': stock_info.get('sector', 'N/A'),
            'factors': factors,
            'scores': dimension_scores,
            'total_score': total_score
        }
    
    def screen_stocks(self, stock_pool, top_n=10):
        """筛选股票"""
        print(f"\n{'='*70}")
        print(f"多因子选股（演示模式） - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 生成所有股票的因子数据
        all_factor_values = {}
        
        # 第一轮：生成所有因子值
        print("📊 生成因子数据...")
        stock_factors = []
        
        for stock in stock_pool:
            factors = self.generate_demo_factors(stock)
            stock_factors.append({
                'info': stock,
                'factors': factors
            })
            
            # 收集所有因子值
            for factor_name, value in factors.items():
                if factor_name not in all_factor_values:
                    all_factor_values[factor_name] = []
                all_factor_values[factor_name].append(value)
        
        print(f"✅ 生成 {len(stock_factors)} 只股票的因子数据\n")
        
        # 第二轮：评分
        print("📈 计算因子得分...")
        scored_stocks = []
        
        for stock_data in stock_factors:
            score_result = self.score_stock(stock_data['info'], all_factor_values)
            scored_stocks.append(score_result)
        
        # 排序
        scored_stocks.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 输出结果
        print(f"\n{'='*70}")
        print(f"筛选结果 TOP {top_n}")
        print(f"{'='*70}\n")
        
        for i, stock in enumerate(scored_stocks[:top_n], 1):
            print(f"#{i} {stock['name']} ({stock['code']}) - {stock['sector']}")
            print(f"   总分: {stock['total_score']:.3f}")
            print(f"   估值: {stock['scores']['valuation']:.3f} | "
                  f"质量: {stock['scores']['quality']:.3f} | "
                  f"成长: {stock['scores']['growth']:.3f} | "
                  f"动量: {stock['scores']['momentum']:.3f}")
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
                    'name': s['name'],
                    'sector': s['sector'],
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
    screener = MultiFactorScreenerDemo()
    
    # 执行筛选
    top_stocks = screener.screen_stocks(stock_pool, top_n=15)
    
    print(f"\n{'='*70}")
    print("✅ 多因子选股完成！")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
