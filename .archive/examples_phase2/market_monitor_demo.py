#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场监控系统（演示版）
Market Monitoring System (Demo Version)

使用模拟数据演示功能
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class MarketMonitorDemo:
    """市场监控系统（演示版）"""
    
    def __init__(self):
        self.demo_mode = True
        
    def generate_demo_index_data(self, base_value, volatility=0.02):
        """生成演示指数数据"""
        dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
        values = [base_value]
        
        for i in range(1, 60):
            change = np.random.randn() * volatility * values[-1]
            new_value = values[-1] + change
            values.append(max(new_value, base_value * 0.8))  # 防止跌太多
        
        df = pd.DataFrame({
            'trade_date': dates,
            'close': values,
            'pct_chg': np.random.randn(60) * 2,
            'vol': np.random.uniform(1e8, 5e8, 60),
            'amount': np.random.uniform(1e11, 5e11, 60)
        })
        
        return df
    
    def analyze_index(self, df, name):
        """分析指数"""
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 均线
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        
        # 趋势
        trend = 'UP' if latest['close'] > df['ma5'].iloc[-1] else 'DOWN'
        
        # 成交量变化
        vol_change = (latest['vol'] - prev['vol']) / prev['vol'] * 100 if prev['vol'] > 0 else 0
        
        return {
            'name': name,
            'close': latest['close'],
            'change_pct': latest['pct_chg'],
            'volume': latest['vol'],
            'volume_change': vol_change,
            'ma5': df['ma5'].iloc[-1],
            'ma10': df['ma10'].iloc[-1],
            'ma20': df['ma20'].iloc[-1],
            'trend': trend,
            'amount': latest['amount']
        }
    
    def get_market_breadth_demo(self):
        """演示市场宽度"""
        total = 4000
        up_count = int(np.random.uniform(1500, 2500))
        down_count = total - up_count - int(np.random.uniform(100, 300))
        flat_count = total - up_count - down_count
        
        return {
            'up_count': up_count,
            'down_count': down_count,
            'flat_count': flat_count,
            'limit_up': int(np.random.uniform(20, 80)),
            'limit_down': int(np.random.uniform(5, 40)),
            'total': total,
            'breadth': up_count / total * 100
        }
    
    def get_north_flow_demo(self):
        """演示北向资金"""
        return {
            'today': np.random.uniform(-100, 100),
            'recent_5d': np.random.uniform(-300, 300),
            'recent_20d': np.random.uniform(-1000, 1000),
            'trend': 'INFLOW' if np.random.random() > 0.5 else 'OUTFLOW'
        }
    
    def get_sector_performance_demo(self):
        """演示板块表现"""
        sectors = [
            {'name': '电子', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '计算机', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '医药生物', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '食品饮料', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '电气设备', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '化工', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '机械设备', 'change_pct': np.random.uniform(-3, 5)},
            {'name': '银行', 'change_pct': np.random.uniform(-2, 2)},
            {'name': '非银金融', 'change_pct': np.random.uniform(-2, 3)},
            {'name': '房地产', 'change_pct': np.random.uniform(-3, 3)}
        ]
        
        for s in sectors:
            s['code'] = f"DEMO{sectors.index(s)}"
            s['amount'] = np.random.uniform(5e10, 2e11)
        
        sectors.sort(key=lambda x: x['change_pct'], reverse=True)
        return sectors
    
    def monitor(self):
        """执行市场监控"""
        print(f"\n{'='*70}")
        print(f"市场监控（演示模式） - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        results = {}
        
        # 1. 主要指数（使用演示数据）
        print("📊 主要指数:")
        print("-" * 70)
        
        indices = [
            (3300, '上证指数'),
            (11000, '深证成指'),
            (2200, '创业板指'),
            (4000, '沪深300')
        ]
        
        index_results = []
        for base_value, name in indices:
            df = self.generate_demo_index_data(base_value)
            analysis = self.analyze_index(df, name)
            
            if analysis:
                index_results.append(analysis)
                
                trend_emoji = '📈' if analysis['trend'] == 'UP' else '📉'
                change_emoji = '🔺' if analysis['change_pct'] > 0 else '🔻'
                
                print(f"{trend_emoji} {name}: {analysis['close']:.2f} "
                      f"{change_emoji} {analysis['change_pct']:+.2f}% "
                      f"| 成交额 {analysis['amount']/1e8:.0f}亿")
        
        results['indices'] = index_results
        print()
        
        # 2. 市场宽度
        print("📐 市场宽度:")
        print("-" * 70)
        
        breadth = self.get_market_breadth_demo()
        results['breadth'] = breadth
        
        print(f"上涨: {breadth['up_count']} | "
              f"下跌: {breadth['down_count']} | "
              f"平盘: {breadth['flat_count']}")
        print(f"涨停: {breadth['limit_up']} | "
              f"跌停: {breadth['limit_down']}")
        print(f"市场宽度: {breadth['breadth']:.1f}% "
              f"({'强势' if breadth['breadth'] > 60 else '弱势' if breadth['breadth'] < 40 else '平衡'})")
        print()
        
        # 3. 北向资金
        print("💰 北向资金:")
        print("-" * 70)
        
        north = self.get_north_flow_demo()
        results['north_flow'] = north
        
        flow_emoji = '🔵' if north['today'] > 0 else '🔴'
        trend_emoji = '⬆️' if north['trend'] == 'INFLOW' else '⬇️'
        
        print(f"今日: {flow_emoji} {north['today']:+.2f}亿")
        print(f"近5日: {north['recent_5d']:+.2f}亿 | "
              f"近20日: {north['recent_20d']:+.2f}亿")
        print(f"趋势: {trend_emoji} {north['trend']}")
        print()
        
        # 4. 板块表现
        print("🏢 板块表现 (TOP 5):")
        print("-" * 70)
        
        sectors = self.get_sector_performance_demo()
        results['sectors'] = sectors
        
        for i, sector in enumerate(sectors[:5], 1):
            emoji = '🔥' if sector['change_pct'] > 2 else '✅' if sector['change_pct'] > 0 else '❄️'
            print(f"{i}. {emoji} {sector['name']}: {sector['change_pct']:+.2f}% "
                  f"| 成交额 {sector['amount']/1e8:.0f}亿")
        print()
        
        # 5. 市场情绪评估
        print("🎯 市场情绪:")
        print("-" * 70)
        
        sentiment_score = 0
        signals = []
        
        # 指数趋势
        up_count = sum(1 for idx in index_results if idx['trend'] == 'UP')
        if up_count >= 3:
            sentiment_score += 25
            signals.append("✅ 多数指数上涨")
        elif up_count <= 1:
            signals.append("❄️ 多数指数下跌")
        
        # 市场宽度
        if breadth['breadth'] > 60:
            sentiment_score += 25
            signals.append("✅ 市场宽度强势")
        elif breadth['breadth'] < 40:
            signals.append("❄️ 市场宽度弱势")
        else:
            sentiment_score += 12.5
            signals.append("⚖️ 市场宽度平衡")
        
        # 北向资金
        if north['today'] > 0 and north['recent_5d'] > 0:
            sentiment_score += 25
            signals.append("✅ 北向资金持续流入")
        elif north['today'] < 0 and north['recent_5d'] < 0:
            signals.append("❄️ 北向资金持续流出")
        else:
            sentiment_score += 12.5
            signals.append("⚖️ 北向资金分歧")
        
        # 板块表现
        positive_sectors = sum(1 for s in sectors[:5] if s['change_pct'] > 0)
        if positive_sectors >= 4:
            sentiment_score += 25
            signals.append("✅ 板块普涨")
        elif positive_sectors <= 1:
            signals.append("❄️ 板块普跌")
        else:
            sentiment_score += 12.5
            signals.append("⚖️ 板块分化")
        
        results['sentiment'] = {
            'score': sentiment_score,
            'signals': signals
        }
        
        for signal in signals:
            print(signal)
        
        # 情绪评级
        if sentiment_score >= 75:
            level = "🔥 极度乐观"
            action = "积极做多"
        elif sentiment_score >= 50:
            level = "😊 乐观"
            action = "逢低加仓"
        elif sentiment_score >= 25:
            level = "😐 中性"
            action = "观望为主"
        else:
            level = "😰 悲观"
            action = "谨慎防守"
        
        print(f"\n情绪得分: {sentiment_score}/100 | {level}")
        print(f"建议操作: {action}")
        
        print(f"\n{'='*70}\n")
        
        # 保存结果
        output_file = 'data/market_monitor_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 结果已保存到 {output_file}\n")
        
        return results

def main():
    """主函数"""
    monitor = MarketMonitorDemo()
    monitor.monitor()

if __name__ == '__main__':
    main()
