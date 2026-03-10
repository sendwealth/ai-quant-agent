#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实数据多因子选股系统
Real Data Multi-Factor Stock Screening System

使用真实TuShare数据进行分析
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

class RealMultiFactorScreener:
    """真实数据多因子选股"""
    
    def __init__(self):
        self.load_real_data()
    
    def load_real_data(self):
        """加载真实数据"""
        with open('data/real_stock_data.json', 'r', encoding='utf-8') as f:
            self.stock_data = json.load(f)
    
    def calculate_factors(self, code):
        """计算因子"""
        data = self.stock_data.get(code)
        if not data or 'price_data' not in data:
            return None
        
        price_df = pd.DataFrame(data['price_data'])
        price_df = price_df.sort_values('trade_date')
        
        if len(price_df) < 60:
            return None
        
        factors = {}
        
        # 价格数据
        latest = price_df.iloc[-1]
        
        # 1. 动量因子
        # 1个月动量
        if len(price_df) >= 20:
            price_20d_ago = price_df.iloc[-20]['close']
            factors['momentum_1m'] = (latest['close'] - price_20d_ago) / price_20d_ago
        else:
            factors['momentum_1m'] = 0
        
        # 3个月动量
        if len(price_df) >= 60:
            price_60d_ago = price_df.iloc[-60]['close']
            factors['momentum_3m'] = (latest['close'] - price_60d_ago) / price_60d_ago
        else:
            factors['momentum_3m'] = 0
        
        # 2. 波动率因子
        returns = price_df['close'].pct_change()
        factors['volatility'] = returns.std() * np.sqrt(252)  # 年化波动率
        
        # 3. 成交量因子
        vol_ma_20 = price_df['vol'].tail(20).mean()
        factors['volume_ratio'] = latest['vol'] / vol_ma_20 if vol_ma_20 > 0 else 1
        
        # 4. 均线因子
        price_df['ma5'] = price_df['close'].rolling(5).mean()
        price_df['ma10'] = price_df['close'].rolling(10).mean()
        price_df['ma20'] = price_df['close'].rolling(20).mean()
        
        factors['price_vs_ma5'] = (latest['close'] - price_df['ma5'].iloc[-1]) / price_df['ma5'].iloc[-1]
        factors['price_vs_ma20'] = (latest['close'] - price_df['ma20'].iloc[-1]) / price_df['ma20'].iloc[-1]
        
        # 5. RSI因子
        delta = price_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        factors['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # 6. 涨跌幅
        factors['pct_chg'] = latest['pct_chg'] / 100
        
        # 7. 基本面因子（如果有）
        basic = data.get('basic_info', {})
        factors['pe'] = basic.get('pe', 0)
        factors['pb'] = basic.get('pb', 0)
        factors['turnover_rate'] = basic.get('turnover_rate', 0)
        
        return factors
    
    def normalize_factor(self, value, all_values, direction='high'):
        """归一化因子"""
        if np.isnan(value) or value == 0:
            return 0.5
        
        valid_values = [v for v in all_values if not np.isnan(v) and v != 0]
        if not valid_values:
            return 0.5
        
        min_val = min(valid_values)
        max_val = max(valid_values)
        
        if max_val == min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        
        if direction == 'low':
            normalized = 1 - normalized
        
        return max(0, min(1, normalized))
    
    def score_stock(self, code, all_factors):
        """评分股票"""
        factors = self.calculate_factors(code)
        if not factors:
            return None
        
        data = self.stock_data[code]
        
        # 因子配置
        factor_config = {
            'momentum_1m': {'weight': 0.2, 'direction': 'high'},
            'momentum_3m': {'weight': 0.15, 'direction': 'high'},
            'volatility': {'weight': 0.1, 'direction': 'low'},
            'volume_ratio': {'weight': 0.1, 'direction': 'high'},
            'price_vs_ma5': {'weight': 0.1, 'direction': 'high'},
            'price_vs_ma20': {'weight': 0.1, 'direction': 'high'},
            'rsi': {'weight': 0.1, 'direction': 'mid'},
            'pct_chg': {'weight': 0.05, 'direction': 'high'},
            'pe': {'weight': 0.05, 'direction': 'low'},
            'pb': {'weight': 0.05, 'direction': 'low'}
        }
        
        # 计算得分
        scores = {}
        total_score = 0
        total_weight = 0
        
        for factor_name, config in factor_config.items():
            if factor_name in factors:
                value = factors[factor_name]
                all_values = [all_factors[code][factor_name] for code in all_factors if factor_name in all_factors[code]]
                
                normalized = self.normalize_factor(value, all_values, config['direction'])
                
                weight = config['weight']
                total_score += normalized * weight
                total_weight += weight
                scores[factor_name] = normalized
        
        if total_weight > 0:
            total_score /= total_weight
        
        return {
            'code': code,
            'name': data['name'],
            'sector': data.get('sector', 'N/A'),
            'current_price': data['price_data'][0]['close'],  # 最新价格
            'factors': factors,
            'scores': scores,
            'total_score': total_score
        }
    
    def screen_stocks(self, top_n=15):
        """筛选股票"""
        print(f"\n{'='*70}")
        print(f"真实数据多因子选股 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 计算所有股票的因子
        print("📊 计算因子...")
        all_factors = {}
        
        for code in self.stock_data.keys():
            factors = self.calculate_factors(code)
            if factors:
                all_factors[code] = factors
        
        print(f"✅ 成功计算 {len(all_factors)} 只股票的因子\n")
        
        # 评分
        print("📈 评分排序...")
        scored_stocks = []
        
        for code in all_factors.keys():
            score_result = self.score_stock(code, all_factors)
            if score_result:
                scored_stocks.append(score_result)
        
        # 排序
        scored_stocks.sort(key=lambda x: x['total_score'], reverse=True)
        
        # 输出结果
        print(f"\n{'='*70}")
        print(f"筛选结果 TOP {top_n}")
        print(f"{'='*70}\n")
        
        for i, stock in enumerate(scored_stocks[:top_n], 1):
            print(f"#{i} {stock['name']} ({stock['code']}) - {stock['sector']}")
            print(f"   价格: {stock['current_price']:.2f}")
            print(f"   总分: {stock['total_score']:.3f}")
            
            # 关键因子
            key_factors = []
            if 'momentum_1m' in stock['factors']:
                key_factors.append(f"1月动量 {stock['factors']['momentum_1m']:+.1%}")
            if 'momentum_3m' in stock['factors']:
                key_factors.append(f"3月动量 {stock['factors']['momentum_3m']:+.1%}")
            if 'rsi' in stock['factors']:
                key_factors.append(f"RSI {stock['factors']['rsi']:.1f}")
            if stock['factors'].get('pe', 0) > 0:
                key_factors.append(f"PE {stock['factors']['pe']:.1f}")
            
            print(f"   关键指标: {' | '.join(key_factors)}")
            print()
        
        # 保存结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(self.stock_data),
            'scored_stocks': len(scored_stocks),
            'top_stocks': [
                {
                    'rank': i,
                    'code': s['code'],
                    'name': s['name'],
                    'sector': s['sector'],
                    'price': s['current_price'],
                    'total_score': s['total_score'],
                    'momentum_1m': s['factors'].get('momentum_1m', 0),
                    'momentum_3m': s['factors'].get('momentum_3m', 0),
                    'rsi': s['factors'].get('rsi', 0),
                    'pe': s['factors'].get('pe', 0)
                }
                for i, s in enumerate(scored_stocks[:top_n], 1)
            ]
        }
        
        output_file = 'data/real_multi_factor_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 结果已保存到 {output_file}\n")
        
        return scored_stocks[:top_n]

def main():
    """主函数"""
    screener = RealMultiFactorScreener()
    top_stocks = screener.screen_stocks(top_n=15)
    
    print(f"\n{'='*70}")
    print("✅ 真实数据选股完成！")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
