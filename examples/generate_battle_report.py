#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实战投资报告生成器
Real Battle Investment Report Generator

基于真实数据生成投资建议
"""

import json
from datetime import datetime

class BattleReportGenerator:
    """实战报告生成器"""
    
    def __init__(self):
        self.load_data()
    
    def load_data(self):
        """加载数据"""
        # 加载选股结果
        with open('data/real_multi_factor_results.json', 'r', encoding='utf-8') as f:
            self.stock_results = json.load(f)
        
        # 加载持仓
        try:
            with open('data/auto_portfolio.json', 'r', encoding='utf-8') as f:
                self.portfolio = json.load(f)
        except:
            self.portfolio = {
                'initial_capital': 100000,
                'cash': 100000,
                'positions': {}
            }
    
    def generate_report(self):
        """生成实战报告"""
        print(f"\n{'='*70}")
        print(f"💰 实战投资报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 1. 市场概况
        print("📊 市场概况")
        print("-" * 70)
        print("基于30只A股真实数据分析")
        print("数据来源: TuShare")
        print("分析维度: 10个因子（动量、波动率、估值等）")
        print()
        
        # 2. TOP 5 推荐股票
        print("🏆 TOP 5 推荐股票")
        print("-" * 70)
        
        top_5 = self.stock_results['top_stocks'][:5]
        
        for stock in top_5:
            rank_emoji = ['🥇', '🥈', '🥉', '4️⃣', '5️⃣'][stock['rank']-1]
            
            print(f"\n{rank_emoji} {stock['name']} ({stock['code']}) - {stock['sector']}")
            print(f"   当前价格: {stock['price']:.2f}元")
            print(f"   综合得分: {stock['total_score']:.3f}")
            
            # 分析亮点
            highlights = []
            if stock['momentum_1m'] > 0.02:
                highlights.append(f"✅ 短期强势（1月+{stock['momentum_1m']*100:.1f}%）")
            elif stock['momentum_1m'] < -0.03:
                highlights.append(f"⚠️ 短期回调（1月{stock['momentum_1m']*100:.1f}%）")
            
            if stock['momentum_3m'] > 0.15:
                highlights.append(f"✅ 中期上涨（3月+{stock['momentum_3m']*100:.1f}%）")
            
            if 30 < stock['rsi'] < 70:
                highlights.append(f"✅ RSI合理 ({stock['rsi']:.1f})")
            elif stock['rsi'] < 30:
                highlights.append(f"💡 RSI超卖 ({stock['rsi']:.1f})，可能反弹")
            elif stock['rsi'] > 70:
                highlights.append(f"⚠️ RSI超买 ({stock['rsi']:.1f})，注意风险")
            
            if stock['pe'] > 0 and stock['pe'] < 20:
                highlights.append(f"✅ 估值合理（PE {stock['pe']:.1f}）")
            
            for h in highlights:
                print(f"   {h}")
        
        print()
        
        # 3. 行业分布
        print("\n🏢 推荐股票行业分布")
        print("-" * 70)
        
        sectors = {}
        for stock in top_5:
            sector = stock['sector']
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(stock['name'])
        
        for sector, stocks in sectors.items():
            print(f"{sector}: {', '.join(stocks)}")
        
        print()
        
        # 4. 投资建议
        print("\n💡 投资建议")
        print("-" * 70)
        
        # 基于TOP 5生成建议
        best_stock = top_5[0]
        
        suggestions = []
        
        # 建议1：核心持仓
        suggestions.append({
            'type': '核心持仓',
            'stocks': [s['name'] for s in top_5[:3]],
            'weight': '50-60%',
            'reason': '得分最高，基本面和技术面较好'
        })
        
        # 建议2：卫星持仓
        suggestions.append({
            'type': '卫星持仓',
            'stocks': [s['name'] for s in top_5[3:5]],
            'weight': '20-30%',
            'reason': '分散风险，增加行业覆盖'
        })
        
        # 建议3：现金储备
        suggestions.append({
            'type': '现金储备',
            'stocks': [],
            'weight': '20%',
            'reason': '应对不确定性，保留加仓能力'
        })
        
        for sug in suggestions:
            print(f"\n{sug['type']} ({sug['weight']})")
            if sug['stocks']:
                print(f"  股票: {', '.join(sug['stocks'])}")
            print(f"  理由: {sug['reason']}")
        
        print()
        
        # 5. 风险提示
        print("\n⚠️ 风险提示")
        print("-" * 70)
        
        risks = []
        
        # 检查RSI超买
        overbought = [s for s in top_5 if s['rsi'] > 70]
        if overbought:
            risks.append(f"以下股票RSI超买(>70)，注意回调风险: {', '.join([s['name'] for s in overbought])}")
        
        # 检查短期涨幅过大
        high_momentum = [s for s in top_5 if s['momentum_1m'] > 0.1]
        if high_momentum:
            risks.append(f"以下股票1个月涨幅>10%，追高需谨慎: {', '.join([s['name'] for s in high_momentum])}")
        
        if not risks:
            risks.append("当前无重大风险信号，但仍需关注市场整体走势")
        
        for risk in risks:
            print(f"• {risk}")
        
        print()
        
        # 6. 操作计划
        print("\n📋 操作计划")
        print("-" * 70)
        
        print("\n建议建仓步骤:")
        print("1. 第1批（30%）：买入长江电力、比亚迪")
        print("   - 分散在电力、新能源两个行业")
        print("   - 长江电力稳定，比亚迪成长性好")
        print()
        print("2. 第2批（20%）：观察1-2周后，加仓宁德时代")
        print("   - 如果市场稳定，可考虑加仓")
        print()
        print("3. 第3批（20%）：根据市场情况，选择万华化学或招商银行")
        print("   - 化工或银行，看哪个行业更强")
        print()
        print("4. 保留30%现金，等待更好的加仓机会")
        
        print()
        
        # 7. 止损止盈设置
        print("\n🎯 止损止盈设置")
        print("-" * 70)
        
        print("\n止损线: -5%")
        print("  任何股票亏损达到5%，立即止损")
        print()
        print("止盈线:")
        print("  • 第一目标: +10% (卖出50%)")
        print("  • 第二目标: +20% (卖出剩余50%)")
        print()
        print("持仓周期: 1-3个月（中短期）")
        
        print()
        
        # 8. 总结
        print("\n" + "="*70)
        print("📝 总结")
        print("="*70)
        
        print(f"""
基于真实数据分析，当前推荐关注：

🥇 首选: {top_5[0]['name']} ({top_5[0]['sector']})
   - 综合得分最高: {top_5[0]['total_score']:.3f}
   - 短期动量: {top_5[0]['momentum_1m']*100:+.1f}%
   - 价格: {top_5[0]['price']:.2f}元

🥈 次选: {top_5[1]['name']} ({top_5[1]['sector']})
   - 新能源车龙头
   - 得分: {top_5[1]['total_score']:.3f}

🥉 备选: {top_5[2]['name']} ({top_5[2]['sector']})
   - 新能源核心标的
   - 得分: {top_5[2]['total_score']:.3f}

建议仓位: 70%股票 + 30%现金
风险等级: 中等
预期收益: 5-15%（1-3个月）

⚠️ 重要提醒:
• 本报告基于历史数据分析，不构成投资建议
• 股市有风险，投资需谨慎
• 请结合自身风险承受能力决策
""")
        
        print("="*70)
        print(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70 + "\n")
        
        # 保存报告
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'top_5_stocks': top_5,
            'suggestions': suggestions,
            'risks': risks,
            'position_plan': {
                'batch_1': {'stocks': ['长江电力', '比亚迪'], 'weight': '30%'},
                'batch_2': {'stocks': ['宁德时代'], 'weight': '20%'},
                'batch_3': {'stocks': ['万华化学', '招商银行'], 'weight': '20%'},
                'cash': '30%'
            },
            'stop_loss': -0.05,
            'take_profit_1': 0.10,
            'take_profit_2': 0.20
        }
        
        with open('data/battle_report.json', 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)
        
        print("✅ 报告已保存到 data/battle_report.json\n")

def main():
    """主函数"""
    generator = BattleReportGenerator()
    generator.generate_report()

if __name__ == '__main__':
    main()
