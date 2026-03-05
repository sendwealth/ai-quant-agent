#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统改进统一执行器（演示版）
System Improvement Executor (Demo Version)

整合执行所有改进模块的演示版本
"""

import json
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.market_monitor_demo import MarketMonitorDemo
from examples.multi_factor_screener_demo import MultiFactorScreenerDemo
from examples.risk_manager import RiskManager

class SystemImproverDemo:
    """系统改进执行器（演示版）"""
    
    def __init__(self):
        self.monitor = MarketMonitorDemo()
        self.screener = MultiFactorScreenerDemo()
        self.risk_manager = RiskManager()
        
        self.stock_pool = self.load_stock_pool()
    
    def load_stock_pool(self):
        """加载股票池"""
        with open('data/stock_pool_extended.json', 'r', encoding='utf-8') as f:
            pool_data = json.load(f)
        
        stocks = []
        for category, data in pool_data['categories'].items():
            stocks.extend(data['stocks'])
        
        return stocks
    
    def run_all_improvements(self):
        """运行所有改进"""
        print(f"\n{'='*70}")
        print(f"系统全面改进（演示模式） - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 1. 市场监控
        print("Step 1/4: 市场监控...")
        print("=" * 70)
        market_data = self.monitor.monitor()
        
        # 2. 多因子选股
        print("\nStep 2/4: 多因子选股...")
        print("=" * 70)
        top_stocks = self.screener.screen_stocks(self.stock_pool, top_n=10)
        
        # 3. 风险评估
        print("\nStep 3/4: 风险评估...")
        print("=" * 70)
        portfolio_risk = self.risk_manager.generate_risk_report()
        
        # 4. 生成改进报告
        print("\nStep 4/4: 生成改进报告...")
        print("=" * 70)
        self.generate_improvement_report(market_data, top_stocks, portfolio_risk)
        
        print(f"\n{'='*70}")
        print("✅ 所有改进模块执行完成！")
        print(f"{'='*70}\n")
    
    def generate_improvement_report(self, market_data, top_stocks, portfolio_risk):
        """生成改进报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'market': {
                'sentiment': market_data.get('sentiment', {}) if market_data else {},
                'indices_count': len(market_data.get('indices', [])) if market_data else 0,
                'breadth': market_data.get('breadth', {}) if market_data else {}
            },
            'stock_selection': {
                'total_scored': len(top_stocks) if top_stocks else 0,
                'top_5': [
                    {
                        'code': s['code'],
                        'name': s['name'],
                        'score': s['total_score'],
                        'sector': s.get('sector', 'N/A')
                    }
                    for s in (top_stocks[:5] if top_stocks else [])
                ]
            },
            'risk': portfolio_risk,
            'recommendations': self.generate_recommendations(market_data, top_stocks, portfolio_risk)
        }
        
        # 保存报告
        output_file = 'data/system_improvement_report.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 改进报告已保存到 {output_file}")
        
        # 输出摘要
        print("\n📊 改进摘要:")
        print("-" * 70)
        
        # 市场情绪
        if market_data and 'sentiment' in market_data:
            sentiment = market_data['sentiment']
            print(f"市场情绪: {sentiment['score']}/100")
        
        # 选股结果
        if top_stocks:
            print(f"高分股票: {len(top_stocks)} 只")
            if top_stocks:
                best = top_stocks[0]
                print(f"最佳股票: {best['name']} ({best['code']}) - 得分 {best['total_score']:.3f}")
        
        # 风险等级
        print(f"风险等级: {portfolio_risk['risk_level']}")
        
        # 投资建议
        print("\n💡 投资建议:")
        print("-" * 70)
        for i, rec in enumerate(report['recommendations'][:5], 1):
            emoji = {'MARKET': '📊', 'RISK': '⚠️', 'STOCK': '📈'}.get(rec['type'], '💡')
            print(f"{i}. {emoji} {rec['message']}")
            print(f"   行动: {rec['action']}")
        
        print()

    def generate_recommendations(self, market_data, top_stocks, portfolio_risk):
        """生成投资建议"""
        recommendations = []
        
        # 基于市场情绪
        if market_data and 'sentiment' in market_data:
            sentiment_score = market_data['sentiment']['score']
            
            if sentiment_score >= 75:
                recommendations.append({
                    'type': 'MARKET',
                    'level': 'AGGRESSIVE',
                    'message': '市场极度乐观，可积极做多',
                    'action': '增加仓位至70-80%'
                })
            elif sentiment_score >= 50:
                recommendations.append({
                    'type': 'MARKET',
                    'level': 'MODERATE',
                    'message': '市场乐观，可适度加仓',
                    'action': '增加仓位至50-60%'
                })
            elif sentiment_score < 25:
                recommendations.append({
                    'type': 'MARKET',
                    'level': 'DEFENSIVE',
                    'message': '市场悲观，谨慎防守',
                    'action': '降低仓位至20-30%'
                })
        
        # 基于风险等级
        if portfolio_risk['risk_level'] == 'CRITICAL':
            recommendations.append({
                'type': 'RISK',
                'level': 'URGENT',
                'message': '风险极高，建议立即减仓',
                'action': '清仓或保留20%以下仓位'
            })
        elif portfolio_risk['risk_level'] == 'HIGH':
            recommendations.append({
                'type': 'RISK',
                'level': 'WARNING',
                'message': '风险较高，建议控制仓位',
                'action': '降低仓位至30-40%'
            })
        
        # 基于选股结果
        if top_stocks and len(top_stocks) >= 3:
            best_stock = top_stocks[0]
            recommendations.append({
                'type': 'STOCK',
                'level': 'OPPORTUNITY',
                'message': f"推荐关注 {best_stock['name']} ({best_stock['sector']})",
                'action': f"可考虑建仓，综合得分 {best_stock['total_score']:.3f}"
            })
        
        # 基于市场宽度
        if market_data and 'breadth' in market_data:
            breadth = market_data['breadth']
            if breadth['breadth'] > 65:
                recommendations.append({
                    'type': 'MARKET',
                    'level': 'STRONG',
                    'message': f"市场宽度强势 {breadth['breadth']:.1f}%",
                    'action': '市场普涨，可积极参与'
                })
            elif breadth['breadth'] < 35:
                recommendations.append({
                    'type': 'MARKET',
                    'level': 'WEAK',
                    'message': f"市场宽度弱势 {breadth['breadth']:.1f}%",
                    'action': '市场普跌，谨慎观望'
                })
        
        return recommendations

def main():
    """主函数"""
    improver = SystemImproverDemo()
    
    # 运行所有改进
    improver.run_all_improvements()

if __name__ == '__main__':
    main()
