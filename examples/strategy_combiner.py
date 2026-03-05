#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多策略组合系统
Multi-Strategy Portfolio System

功能:
1. 多策略信号整合
2. 信号强度评估
3. 投票决策机制
4. 动态权重调整
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

class StrategyCombiner:
    """多策略组合"""
    
    def __init__(self):
        self.pro = ts.pro_api(Config.TUSHARE_TOKEN)
        
        # 策略权重
        self.strategy_weights = {
            'trend_following': 0.30,      # 趋势跟踪
            'mean_reversion': 0.25,       # 均值回归
            'breakout': 0.25,             # 突破
            'momentum': 0.20              # 动量
        }
        
        # 信号阈值
        self.signal_threshold = 0.6  # 60%以上才执行
        
    def get_stock_data(self, ts_code, days=120):
        """获取股票数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df.empty:
                return None
            
            df = df.sort_values('trade_date')
            
            # 计算指标
            df['ma5'] = df['close'].rolling(5).mean()
            df['ma10'] = df['close'].rolling(10).mean()
            df['ma20'] = df['close'].rolling(20).mean()
            df['ma60'] = df['close'].rolling(60).mean()
            
            # ATR
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['pre_close']),
                    abs(df['low'] - df['pre_close'])
                )
            )
            df['atr'] = df['tr'].rolling(14).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['macd'] = ema12 - ema26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 布林带
            df['boll_mid'] = df['close'].rolling(20).mean()
            df['boll_std'] = df['close'].rolling(20).std()
            df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
            df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']
            
            # 成交量MA
            df['vol_ma5'] = df['vol'].rolling(5).mean()
            df['vol_ma10'] = df['vol'].rolling(10).mean()
            
            return df
        except Exception as e:
            print(f"获取数据失败 {ts_code}: {e}")
            return None
    
    def trend_following_signal(self, df):
        """趋势跟踪策略信号"""
        if df is None or len(df) < 60:
            return 0, 'NEUTRAL'
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        signals = []
        
        # 1. MA多头排列
        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            score += 30
            signals.append("MA多头排列")
        elif latest['ma5'] < latest['ma10'] < latest['ma20']:
            score -= 30
            signals.append("MA空头排列")
        
        # 2. 价格在MA之上
        if latest['close'] > latest['ma20']:
            score += 20
            signals.append("价格>MA20")
        else:
            score -= 20
            signals.append("价格<MA20")
        
        # 3. MACD金叉/死叉
        if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
            score += 25
            signals.append("MACD金叉")
        elif latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
            score -= 25
            signals.append("MACD死叉")
        
        # 4. 成交量放大
        if latest['vol'] > latest['vol_ma5'] * 1.5:
            score += 15
            signals.append("放量")
        elif latest['vol'] < latest['vol_ma5'] * 0.5:
            score -= 10
            signals.append("缩量")
        
        # 归一化到-1到1
        normalized = score / 100
        signal = 'BUY' if score >= 50 else 'SELL' if score <= -50 else 'NEUTRAL'
        
        return normalized, signal
    
    def mean_reversion_signal(self, df):
        """均值回归策略信号"""
        if df is None or len(df) < 20:
            return 0, 'NEUTRAL'
        
        latest = df.iloc[-1]
        
        score = 0
        signals = []
        
        # 1. RSI超卖/超买
        if latest['rsi'] < 30:
            score += 40
            signals.append(f"RSI超卖({latest['rsi']:.1f})")
        elif latest['rsi'] > 70:
            score -= 40
            signals.append(f"RSI超买({latest['rsi']:.1f})")
        elif latest['rsi'] < 40:
            score += 20
            signals.append(f"RSI偏低({latest['rsi']:.1f})")
        elif latest['rsi'] > 60:
            score -= 20
            signals.append(f"RSI偏高({latest['rsi']:.1f})")
        
        # 2. 布林带下轨/上轨
        if latest['close'] < latest['boll_lower']:
            score += 30
            signals.append("触及布林下轨")
        elif latest['close'] > latest['boll_upper']:
            score -= 30
            signals.append("触及布林上轨")
        elif latest['close'] < latest['boll_mid']:
            score += 15
            signals.append("价格低于布林中轨")
        else:
            score -= 15
            signals.append("价格高于布林中轨")
        
        # 3. 距离MA20的偏离度
        deviation = (latest['close'] - latest['ma20']) / latest['ma20']
        if deviation < -0.05:
            score += 20
            signals.append(f"偏离MA20 {deviation:.1%}")
        elif deviation > 0.05:
            score -= 20
            signals.append(f"偏离MA20 +{deviation:.1%}")
        
        # 归一化
        normalized = score / 100
        signal = 'BUY' if score >= 50 else 'SELL' if score <= -50 else 'NEUTRAL'
        
        return normalized, signal
    
    def breakout_signal(self, df):
        """突破策略信号"""
        if df is None or len(df) < 60:
            return 0, 'NEUTRAL'
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        score = 0
        signals = []
        
        # 1. 价格突破20日高点/低点
        high_20 = df['high'].iloc[-21:-1].max()
        low_20 = df['low'].iloc[-21:-1].min()
        
        if latest['close'] > high_20:
            score += 40
            signals.append("突破20日高点")
        elif latest['close'] < low_20:
            score -= 40
            signals.append("跌破20日低点")
        
        # 2. 价格突破60日高点/低点
        high_60 = df['high'].iloc[-61:-1].max()
        low_60 = df['low'].iloc[-61:-1].min()
        
        if latest['close'] > high_60:
            score += 30
            signals.append("突破60日高点")
        elif latest['close'] < low_60:
            score -= 30
            signals.append("跌破60日低点")
        
        # 3. 成交量确认
        if latest['vol'] > prev['vol'] * 1.5:
            if score > 0:
                score += 20
                signals.append("突破放量确认")
            elif score < 0:
                score -= 20
                signals.append("跌破放量确认")
        
        # 归一化
        normalized = score / 100
        signal = 'BUY' if score >= 50 else 'SELL' if score <= -50 else 'NEUTRAL'
        
        return normalized, signal
    
    def momentum_signal(self, df):
        """动量策略信号"""
        if df is None or len(df) < 60:
            return 0, 'NEUTRAL'
        
        latest = df.iloc[-1]
        
        score = 0
        signals = []
        
        # 1. 短期动量（5日）
        momentum_5d = (latest['close'] - df.iloc[-5]['close']) / df.iloc[-5]['close']
        if momentum_5d > 0.05:
            score += 30
            signals.append(f"5日动量 +{momentum_5d:.1%}")
        elif momentum_5d < -0.05:
            score -= 30
            signals.append(f"5日动量 {momentum_5d:.1%}")
        
        # 2. 中期动量（20日）
        momentum_20d = (latest['close'] - df.iloc[-20]['close']) / df.iloc[-20]['close']
        if momentum_20d > 0.10:
            score += 25
            signals.append(f"20日动量 +{momentum_20d:.1%}")
        elif momentum_20d < -0.10:
            score -= 25
            signals.append(f"20日动量 {momentum_20d:.1%}")
        
        # 3. 长期动量（60日）
        momentum_60d = (latest['close'] - df.iloc[-60]['close']) / df.iloc[-60]['close']
        if momentum_60d > 0.20:
            score += 20
            signals.append(f"60日动量 +{momentum_60d:.1%}")
        elif momentum_60d < -0.20:
            score -= 20
            signals.append(f"60日动量 {momentum_60d:.1%}")
        
        # 4. 成交量动量
        vol_momentum = (latest['vol'] - df['vol'].iloc[-10:-1].mean()) / df['vol'].iloc[-10:-1].mean()
        if vol_momentum > 0.5 and score > 0:
            score += 15
            signals.append(f"成交量动量 +{vol_momentum:.1%}")
        
        # 归一化
        normalized = score / 100
        signal = 'BUY' if score >= 50 else 'SELL' if score <= -50 else 'NEUTRAL'
        
        return normalized, signal
    
    def combine_signals(self, ts_code):
        """整合多策略信号"""
        print(f"\n{'='*70}")
        print(f"多策略信号整合 - {ts_code}")
        print(f"{'='*70}\n")
        
        # 获取数据
        df = self.get_stock_data(ts_code)
        
        if df is None or df.empty:
            print("❌ 数据获取失败")
            return None
        
        latest = df.iloc[-1]
        
        # 各策略信号
        strategies = {
            'trend_following': self.trend_following_signal(df),
            'mean_reversion': self.mean_reversion_signal(df),
            'breakout': self.breakout_signal(df),
            'momentum': self.momentum_signal(df)
        }
        
        # 显示各策略信号
        print(f"当前价格: {latest['close']:.2f}")
        print()
        
        for strategy_name, (score, signal) in strategies.items():
            weight = self.strategy_weights[strategy_name]
            emoji = '🟢' if signal == 'BUY' else '🔴' if signal == 'SELL' else '⚪'
            
            print(f"{emoji} {strategy_name:20s} | 得分: {score:+.2f} | "
                  f"信号: {signal:8s} | 权重: {weight:.0%}")
        
        print()
        
        # 加权平均
        combined_score = sum(
            strategies[name][0] * self.strategy_weights[name]
            for name in strategies
        )
        
        # 投票
        buy_votes = sum(1 for s in strategies.values() if s[1] == 'BUY')
        sell_votes = sum(1 for s in strategies.values() if s[1] == 'SELL')
        
        # 最终信号
        if combined_score >= self.signal_threshold:
            final_signal = 'BUY'
            signal_emoji = '🟢'
        elif combined_score <= -self.signal_threshold:
            final_signal = 'SELL'
            signal_emoji = '🔴'
        else:
            final_signal = 'NEUTRAL'
            signal_emoji = '⚪'
        
        # 输出结果
        print(f"{'='*70}")
        print(f"综合得分: {combined_score:+.3f}")
        print(f"投票结果: 买入 {buy_votes} 票 | 卖出 {sell_votes} 票 | 中性 {4-buy_votes-sell_votes} 票")
        print(f"{signal_emoji} 最终信号: {final_signal}")
        print(f"{'='*70}\n")
        
        return {
            'code': ts_code,
            'price': latest['close'],
            'strategies': {
                name: {
                    'score': score,
                    'signal': signal,
                    'weight': self.strategy_weights[name]
                }
                for name, (score, signal) in strategies.items()
            },
            'combined_score': combined_score,
            'votes': {
                'buy': buy_votes,
                'sell': sell_votes,
                'neutral': 4 - buy_votes - sell_votes
            },
            'final_signal': final_signal
        }
    
    def analyze_multiple_stocks(self, stock_codes):
        """分析多只股票"""
        results = []
        
        for code in stock_codes:
            result = self.combine_signals(code)
            if result:
                results.append(result)
        
        # 排序（按综合得分）
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # 输出汇总
        print(f"\n{'='*70}")
        print(f"多股票信号汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        buy_stocks = [r for r in results if r['final_signal'] == 'BUY']
        sell_stocks = [r for r in results if r['final_signal'] == 'SELL']
        
        if buy_stocks:
            print("🟢 买入信号:")
            for r in buy_stocks:
                print(f"  {r['code']}: {r['price']:.2f} | 得分 {r['combined_score']:+.3f}")
        
        if sell_stocks:
            print("\n🔴 卖出信号:")
            for r in sell_stocks:
                print(f"  {r['code']}: {r['price']:.2f} | 得分 {r['combined_score']:+.3f}")
        
        if not buy_stocks and not sell_stocks:
            print("⚪ 无明确信号")
        
        print(f"\n{'='*70}\n")
        
        # 保存结果
        output_file = 'data/multi_strategy_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': results
            }, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"✅ 结果已保存到 {output_file}\n")
        
        return results

def main():
    """主函数"""
    # 加载股票池
    with open('data/stock_pool_extended.json', 'r', encoding='utf-8') as f:
        pool_data = json.load(f)
    
    # 合并所有股票
    stock_codes = []
    for category, data in pool_data['categories'].items():
        for stock in data['stocks']:
            stock_codes.append(stock['code'])
    
    # 创建组合器
    combiner = StrategyCombiner()
    
    # 分析（测试前5只）
    test_codes = stock_codes[:5]
    results = combiner.analyze_multiple_stocks(test_codes)

if __name__ == '__main__':
    main()
