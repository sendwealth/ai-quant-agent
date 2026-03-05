#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
市场监控系统
Market Monitoring System

功能:
1. 大盘指数监控
2. 板块轮动分析
3. 市场情绪指标
4. 资金流向监控
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class MarketMonitor:
    """市场监控系统"""
    
    def __init__(self):
        self.pro = ts.pro_api(Config.TUSHARE_TOKEN)
        
    def get_index_data(self, index_code, days=60):
        """获取指数数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            df = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
            return df.sort_values('trade_date')
        except Exception as e:
            print(f"获取指数数据失败 {index_code}: {e}")
            return None
    
    def analyze_index(self, df, name):
        """分析指数"""
        if df is None or df.empty:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        
        # 涨跌幅
        change_pct = latest['pct_chg']
        
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
            'change_pct': change_pct,
            'volume': latest['vol'],
            'volume_change': vol_change,
            'ma5': df['ma5'].iloc[-1],
            'ma10': df['ma10'].iloc[-1],
            'ma20': df['ma20'].iloc[-1],
            'trend': trend,
            'amount': latest['amount']
        }
    
    def get_market_breadth(self):
        """获取市场宽度（涨跌家数）"""
        try:
            # 获取当日涨跌停统计
            end_date = datetime.now().strftime('%Y%m%d')
            
            # 涨停
            limit_up = self.pro.limit_list(trade_date=end_date, limit_type='U')
            # 跌停
            limit_down = self.pro.limit_list(trade_date=end_date, limit_type='D')
            
            # 获取全部股票涨跌情况
            all_stocks = self.pro.daily(trade_date=end_date)
            
            if all_stocks.empty:
                return None
            
            up_count = len(all_stocks[all_stocks['pct_chg'] > 0])
            down_count = len(all_stocks[all_stocks['pct_chg'] < 0])
            flat_count = len(all_stocks[all_stocks['pct_chg'] == 0])
            
            return {
                'up_count': up_count,
                'down_count': down_count,
                'flat_count': flat_count,
                'limit_up': len(limit_up) if not limit_up.empty else 0,
                'limit_down': len(limit_down) if not limit_down.empty else 0,
                'total': len(all_stocks),
                'breadth': up_count / len(all_stocks) * 100 if len(all_stocks) > 0 else 0
            }
        except Exception as e:
            print(f"获取市场宽度失败: {e}")
            return None
    
    def get_north_flow(self, days=30):
        """获取北向资金流向"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            df = self.pro.moneyflow_hsgt(start_date=start_date, end_date=end_date)
            
            if df.empty:
                return None
            
            df = df.sort_values('trade_date')
            latest = df.iloc[-1]
            
            # 近5日净流入
            recent_5d = df.tail(5)['north_money'].sum()
            # 近20日净流入
            recent_20d = df.tail(20)['north_money'].sum()
            
            return {
                'today': latest['north_money'],
                'recent_5d': recent_5d,
                'recent_20d': recent_20d,
                'trend': 'INFLOW' if recent_5d > 0 else 'OUTFLOW'
            }
        except Exception as e:
            print(f"获取北向资金失败: {e}")
            return None
    
    def get_sector_performance(self):
        """获取板块表现"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            
            # 获取行业板块数据
            df = self.pro.index_classify(level='L1', src='SW')
            
            if df.empty:
                return None
            
            # 获取每个板块的涨跌幅
            sectors = []
            for _, row in df.head(10).iterrows():  # 只取前10个主要板块
                index_code = row['index_code']
                
                # 获取板块指数
                index_df = self.pro.index_daily(ts_code=f"{index_code}.SI", 
                                               start_date=end_date, 
                                               end_date=end_date)
                
                if not index_df.empty:
                    latest = index_df.iloc[-1]
                    sectors.append({
                        'name': row['industry_name'],
                        'code': index_code,
                        'change_pct': latest['pct_chg'],
                        'amount': latest['amount']
                    })
            
            # 按涨跌幅排序
            sectors.sort(key=lambda x: x['change_pct'], reverse=True)
            
            return sectors
        except Exception as e:
            print(f"获取板块表现失败: {e}")
            return None
    
    def monitor(self):
        """执行市场监控"""
        print(f"\n{'='*70}")
        print(f"市场监控 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        results = {}
        
        # 1. 主要指数
        print("📊 主要指数:")
        print("-" * 70)
        
        indices = [
            ('000001.SH', '上证指数'),
            ('399001.SZ', '深证成指'),
            ('399006.SZ', '创业板指'),
            ('000300.SH', '沪深300')
        ]
        
        index_results = []
        for code, name in indices:
            df = self.get_index_data(code, days=60)
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
        
        breadth = self.get_market_breadth()
        if breadth:
            results['breadth'] = breadth
            
            print(f"上涨: {breadth['up_count']} | "
                  f"下跌: {breadth['down_count']} | "
                  f"平盘: {breadth['flat_count']}")
            print(f"涨停: {breadth['limit_up']} | "
                  f"跌停: {breadth['limit_down']}")
            print(f"市场宽度: {breadth['breadth']:.1f}% "
                  f"({'强势' if breadth['breadth'] > 60 else '弱势' if breadth['breadth'] < 40 else '平衡'})")
        else:
            print("⚠️ 市场宽度数据获取失败")
        
        print()
        
        # 3. 北向资金
        print("💰 北向资金:")
        print("-" * 70)
        
        north = self.get_north_flow(days=30)
        if north:
            results['north_flow'] = north
            
            flow_emoji = '🔵' if north['today'] > 0 else '🔴'
            trend_emoji = '⬆️' if north['trend'] == 'INFLOW' else '⬇️'
            
            print(f"今日: {flow_emoji} {north['today']:+.2f}亿")
            print(f"近5日: {north['recent_5d']:+.2f}亿 | "
                  f"近20日: {north['recent_20d']:+.2f}亿")
            print(f"趋势: {trend_emoji} {north['trend']}")
        else:
            print("⚠️ 北向资金数据获取失败")
        
        print()
        
        # 4. 板块表现
        print("🏢 板块表现 (TOP 5):")
        print("-" * 70)
        
        sectors = self.get_sector_performance()
        if sectors:
            results['sectors'] = sectors
            
            for i, sector in enumerate(sectors[:5], 1):
                emoji = '🔥' if sector['change_pct'] > 2 else '✅' if sector['change_pct'] > 0 else '❄️'
                print(f"{i}. {emoji} {sector['name']}: {sector['change_pct']:+.2f}% "
                      f"| 成交额 {sector['amount']/1e8:.0f}亿")
        else:
            print("⚠️ 板块数据获取失败")
        
        print()
        
        # 5. 市场情绪评估
        print("🎯 市场情绪:")
        print("-" * 70)
        
        sentiment_score = 0
        signals = []
        
        # 指数趋势
        if index_results:
            up_count = sum(1 for idx in index_results if idx['trend'] == 'UP')
            if up_count >= 3:
                sentiment_score += 25
                signals.append("✅ 多数指数上涨")
            elif up_count <= 1:
                signals.append("❄️ 多数指数下跌")
        
        # 市场宽度
        if breadth:
            if breadth['breadth'] > 60:
                sentiment_score += 25
                signals.append("✅ 市场宽度强势")
            elif breadth['breadth'] < 40:
                signals.append("❄️ 市场宽度弱势")
            else:
                sentiment_score += 12.5
                signals.append("⚖️ 市场宽度平衡")
        
        # 北向资金
        if north:
            if north['today'] > 0 and north['recent_5d'] > 0:
                sentiment_score += 25
                signals.append("✅ 北向资金持续流入")
            elif north['today'] < 0 and north['recent_5d'] < 0:
                signals.append("❄️ 北向资金持续流出")
            else:
                sentiment_score += 12.5
                signals.append("⚖️ 北向资金分歧")
        
        # 板块表现
        if sectors:
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
    monitor = MarketMonitor()
    monitor.monitor()

if __name__ == '__main__':
    main()
