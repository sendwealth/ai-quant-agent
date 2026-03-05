#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统全面改进执行器
System Improvement Executor

整合执行所有改进模块:
1. 多因子选股
2. 市场监控
3. 风险管理
4. 多策略组合
5. 自动交易
"""

import json
import sys
import os
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.multi_factor_screener import MultiFactorScreener
from examples.market_monitor import MarketMonitor
from examples.risk_manager import RiskManager
from examples.strategy_combiner import StrategyCombiner
from examples.enhanced_auto_trading_bot import EnhancedAutoTrader

class SystemImprover:
    """系统改进执行器"""
    
    def __init__(self):
        self.screener = MultiFactorScreener()
        self.monitor = MarketMonitor()
        self.risk_manager = RiskManager()
        self.combiner = StrategyCombiner()
        self.trader = EnhancedAutoTrader()
        
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
        print(f"系统全面改进 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 1. 市场监控
        print("Step 1/5: 市场监控...")
        print("-" * 70)
        market_data = self.monitor.monitor()
        
        # 2. 多因子选股
        print("\nStep 2/5: 多因子选股...")
        print("-" * 70)
        top_stocks = self.screener.screen_stocks(self.stock_pool, top_n=10)
        
        # 3. 风险评估
        print("\nStep 3/5: 风险评估...")
        print("-" * 70)
        portfolio_risk = self.risk_manager.generate_risk_report()
        
        # 4. 多策略信号（选股结果前5只）
        print("\nStep 4/5: 多策略信号...")
        print("-" * 70)
        if top_stocks:
            test_codes = [s['code'] for s in top_stocks[:5]]
            strategy_results = self.combiner.analyze_multiple_stocks(test_codes)
        else:
            print("⚠️ 无选股结果，跳过策略分析")
            strategy_results = []
        
        # 5. 生成改进报告
        print("\nStep 5/5: 生成改进报告...")
        print("-" * 70)
        self.generate_improvement_report(market_data, top_stocks, portfolio_risk, strategy_results)
        
        print(f"\n{'='*70}")
        print("✅ 所有改进模块执行完成！")
        print(f"{'='*70}\n")
    
    def generate_improvement_report(self, market_data, top_stocks, portfolio_risk, strategy_results):
        """生成改进报告"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'market': {
                'sentiment': market_data.get('sentiment', {}) if market_data else {},
                'indices': market_data.get('indices', []) if market_data else [],
                'breadth': market_data.get('breadth', {}) if market_data else {}
            },
            'stock_selection': {
                'total_scored': len(top_stocks) if top_stocks else 0,
                'top_5': [
                    {
                        'code': s['code'],
                        'name': s['info']['name'],
                        'score': s['total_score']
                    }
                    for s in (top_stocks[:5] if top_stocks else [])
                ]
            },
            'risk': portfolio_risk,
            'strategy_signals': [
                {
                    'code': r['code'],
                    'price': r['price'],
                    'combined_score': r['combined_score'],
                    'final_signal': r['final_signal']
                }
                for r in strategy_results
            ],
            'recommendations': self.generate_recommendations(market_data, top_stocks, portfolio_risk, strategy_results)
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
                print(f"最佳股票: {best['info']['name']} ({best['code']}) - 得分 {best['total_score']:.3f}")
        
        # 风险等级
        print(f"风险等级: {portfolio_risk['risk_level']}")
        
        # 交易信号
        buy_signals = [r for r in strategy_results if r['final_signal'] == 'BUY']
        if buy_signals:
            print(f"买入信号: {len(buy_signals)} 只")
            for s in buy_signals[:3]:
                print(f"  - {s['code']}: 得分 {s['combined_score']:+.3f}")
        
        print()
    
    def generate_recommendations(self, market_data, top_stocks, portfolio_risk, strategy_results):
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
                'message': f"推荐关注 {best_stock['info']['name']}",
                'action': f"可考虑建仓，得分 {best_stock['total_score']:.3f}"
            })
        
        # 基于策略信号
        buy_signals = [r for r in strategy_results if r['final_signal'] == 'BUY']
        if len(buy_signals) >= 2:
            recommendations.append({
                'type': 'STRATEGY',
                'level': 'SIGNAL',
                'message': f"多策略确认 {len(buy_signals)} 只股票买入信号",
                'action': '可考虑分批建仓'
            })
        
        return recommendations
    
    def quick_analysis(self, stock_code):
        """快速分析单只股票"""
        print(f"\n{'='*70}")
        print(f"快速分析 - {stock_code}")
        print(f"{'='*70}\n")
        
        # 多策略信号
        result = self.combiner.combine_signals(stock_code)
        
        if result:
            # 风险评估（模拟买入100股）
            print("\n风险评估 (模拟买入100股):")
            print("-" * 70)
            
            risks = self.risk_manager.assess_trade_risk(
                stock_code, 'BUY', 100, result['price']
            )
            
            if risks:
                for risk in risks:
                    emoji = '🔴' if risk['level'] == 'HIGH' else '🟡'
                    print(f"{emoji} {risk['message']}")
            else:
                print("✅ 无重大风险")
        
        print(f"\n{'='*70}\n")
        
        return result

def main():
    """主函数"""
    improver = SystemImprover()
    
    # 运行所有改进
    improver.run_all_improvements()
    
    # 可选：快速分析特定股票
    # improver.quick_analysis('300750.SZ')

if __name__ == '__main__':
    main()
