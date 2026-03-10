#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日监控脚本
Daily Monitor Script

运行时间: 每个交易日 15:30 (收盘后)
功能:
1. 更新股票数据
2. 生成交易信号
3. 市场情绪分析
4. 生成每日报告
5. 保存监控状态供心跳读取
"""

import json
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import tushare as ts
    import pandas as pd
    import numpy as np
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("⚠️ tushare/pandas not available, using mock data")

from config import Config

# ========== 配置 ==========
DATA_DIR = Path(__file__).parent.parent / 'data'
LOG_DIR = Path(__file__).parent.parent / 'logs'

# 监控股票池 (基于battle_report)
WATCH_LIST = [
    {'code': '600900', 'name': '长江电力', 'role': 'core'},
    {'code': '002594', 'name': '比亚迪', 'role': 'core'},
    {'code': '300750', 'name': '宁德时代', 'role': 'core'},
    {'code': '600309', 'name': '万华化学', 'role': 'satellite'},
    {'code': '600036', 'name': '招商银行', 'role': 'satellite'},
]

# 风险参数
RISK_PARAMS = {
    'stop_loss': -0.05,          # 止损 -5%
    'take_profit_1': 0.10,       # 止盈1 +10%
    'take_profit_2': 0.20,       # 止盈2 +20%
    'partial_exit': 0.5,         # 分批卖出比例
}

# 策略参数 (最优)
STRATEGY_PARAMS = {
    'ma_fast': 5,
    'ma_slow': 25,
    'atr_stop_mult': 2.5,
    'use_dynamic_position': True,
    'use_macd': True,
    'use_rsi': True,
}


class DailyMonitor:
    """每日监控器"""
    
    def __init__(self):
        self.timestamp = datetime.now().isoformat()
        self.state = {
            'timestamp': self.timestamp,
            'status': 'ok',
            'market': {},
            'positions': [],
            'signals': [],
            'alerts': [],
            'summary': {},
        }
        
        if TUSHARE_AVAILABLE:
            try:
                self.pro = ts.pro_api(Config.TUSHARE_TOKEN)
            except Exception as e:
                print(f"⚠️ Tushare初始化失败: {e}")
                self.pro = None
        else:
            self.pro = None
    
    def update_stock_data(self, code):
        """更新单只股票数据"""
        if not self.pro:
            return None
        
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=100)).strftime('%Y%m%d')
            
            df = self.pro.daily(ts_code=code, start_date=start_date, end_date=end_date)
            if df.empty:
                return None
            
            df = df.sort_values('trade_date')
            
            # 保存到文件
            file_path = DATA_DIR / f'real_{code}.csv'
            df.to_csv(file_path, index=False)
            
            return df
        except Exception as e:
            print(f"❌ 更新{code}数据失败: {e}")
            return None
    
    def analyze_stock(self, code, name, role):
        """分析单只股票"""
        result = {
            'code': code,
            'name': name,
            'role': role,
            'price': 0,
            'change_pct': 0,
            'signal': 'HOLD',
            'alerts': [],
        }
        
        # 尝试读取数据
        file_path = DATA_DIR / f'real_{code}.csv'
        if not file_path.exists():
            # 尝试更新
            df = self.update_stock_data(code)
            if df is None:
                result['signal'] = 'NO_DATA'
                result['alerts'].append('数据缺失')
                return result
        else:
            try:
                df = pd.read_csv(file_path)
            except:
                result['signal'] = 'DATA_ERROR'
                return result
        
        if df.empty:
            result['signal'] = 'NO_DATA'
            return result
        
        # 最新数据
        latest = df.iloc[-1]
        result['price'] = float(latest['close'])
        result['change_pct'] = float(latest['pct_chg'])
        result['volume'] = float(latest['vol'])
        
        # 计算指标
        df['ma_fast'] = df['close'].rolling(STRATEGY_PARAMS['ma_fast']).mean()
        df['ma_slow'] = df['close'].rolling(STRATEGY_PARAMS['ma_slow']).mean()
        
        # 计算RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 生成信号
        ma_fast = df['ma_fast'].iloc[-1]
        ma_slow = df['ma_slow'].iloc[-1]
        rsi = df['rsi'].iloc[-1]
        
        # 买入信号: 快线上穿慢线 + RSI不超买
        if ma_fast > ma_slow and 30 < rsi < 70:
            result['signal'] = 'BUY'
        # 卖出信号: 快线下穿慢线 + RSI超买
        elif ma_fast < ma_slow or rsi > 80:
            result['signal'] = 'SELL'
        else:
            result['signal'] = 'HOLD'
        
        # 检查风险
        if result['change_pct'] < -5:
            result['alerts'].append(f'单日大跌 {result["change_pct"]:.2f}%')
        elif result['change_pct'] > 5:
            result['alerts'].append(f'单日大涨 {result["change_pct"]:.2f}%')
        
        if rsi > 80:
            result['alerts'].append(f'RSI超买 {rsi:.1f}')
        elif rsi < 20:
            result['alerts'].append(f'RSI超卖 {rsi:.1f}')
        
        return result
    
    def analyze_market(self):
        """分析市场整体情况"""
        market = {
            'trend': 'UNKNOWN',
            'sentiment': 'NEUTRAL',
            'indices': [],
        }
        
        if not self.pro:
            return market
        
        try:
            # 获取主要指数
            indices = [
                ('000001.SH', '上证指数'),
                ('399001.SZ', '深证成指'),
                ('399006.SZ', '创业板指'),
            ]
            
            end_date = datetime.now().strftime('%Y%m%d')
            
            for code, name in indices:
                try:
                    df = self.pro.index_daily(ts_code=code, start_date=end_date, end_date=end_date)
                    if not df.empty:
                        latest = df.iloc[0]
                        market['indices'].append({
                            'name': name,
                            'close': float(latest['close']),
                            'change_pct': float(latest['pct_chg']),
                        })
                except:
                    pass
            
            # 判断市场趋势
            if market['indices']:
                avg_change = sum(i['change_pct'] for i in market['indices']) / len(market['indices'])
                if avg_change > 1:
                    market['trend'] = 'STRONG_UP'
                    market['sentiment'] = 'GREED'
                elif avg_change > 0:
                    market['trend'] = 'UP'
                    market['sentiment'] = 'OPTIMISTIC'
                elif avg_change > -1:
                    market['trend'] = 'DOWN'
                    market['sentiment'] = 'FEAR'
                else:
                    market['trend'] = 'STRONG_DOWN'
                    market['sentiment'] = 'PANIC'
        
        except Exception as e:
            print(f"❌ 市场分析失败: {e}")
        
        return market
    
    def run(self):
        """运行监控"""
        print(f"\n{'='*50}")
        print(f"📊 每日量化监控")
        print(f"⏰ 时间: {self.timestamp}")
        print(f"{'='*50}\n")
        
        # 1. 市场分析
        print("1️⃣ 市场分析...")
        self.state['market'] = self.analyze_market()
        print(f"   趋势: {self.state['market']['trend']}")
        print(f"   情绪: {self.state['market']['sentiment']}")
        
        # 2. 股票分析
        print("\n2️⃣ 股票分析...")
        for stock in WATCH_LIST:
            result = self.analyze_stock(stock['code'], stock['name'], stock['role'])
            self.state['positions'].append(result)
            
            alert_str = f" ⚠️ {', '.join(result['alerts'])}" if result['alerts'] else ""
            print(f"   {result['name']}: {result['signal']} | {result['change_pct']:+.2f}%{alert_str}")
            
            # 收集预警
            if result['alerts']:
                self.state['alerts'].extend([f"{result['name']}: {a}" for a in result['alerts']])
        
        # 3. 汇总信号
        print("\n3️⃣ 信号汇总...")
        buy_signals = [p for p in self.state['positions'] if p['signal'] == 'BUY']
        sell_signals = [p for p in self.state['positions'] if p['signal'] == 'SELL']
        
        self.state['signals'] = {
            'buy': [s['name'] for s in buy_signals],
            'sell': [s['name'] for s in sell_signals],
            'hold': [p['name'] for p in self.state['positions'] if p['signal'] == 'HOLD'],
        }
        
        print(f"   买入: {', '.join(self.state['signals']['buy']) or '无'}")
        print(f"   卖出: {', '.join(self.state['signals']['sell']) or '无'}")
        
        # 4. 生成摘要
        self.state['summary'] = {
            'total_stocks': len(WATCH_LIST),
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'alert_count': len(self.state['alerts']),
            'market_trend': self.state['market']['trend'],
            'market_sentiment': self.state['market']['sentiment'],
        }
        
        # 5. 保存状态
        state_file = DATA_DIR / 'monitor_state.json'
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        print(f"\n✅ 状态已保存: {state_file}")
        
        # 6. 保存每日报告
        report_file = DATA_DIR / f"daily_report_{datetime.now().strftime('%Y%m%d')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        print(f"✅ 报告已保存: {report_file}")
        
        # 7. 输出摘要
        print(f"\n{'='*50}")
        print("📋 今日摘要:")
        print(f"   市场趋势: {self.state['market']['trend']}")
        print(f"   市场情绪: {self.state['market']['sentiment']}")
        print(f"   买入信号: {len(buy_signals)}个")
        print(f"   卖出信号: {len(sell_signals)}个")
        print(f"   预警数量: {len(self.state['alerts'])}个")
        if self.state['alerts']:
            print("\n⚠️ 预警:")
            for alert in self.state['alerts']:
                print(f"   - {alert}")
        print(f"{'='*50}\n")
        
        return self.state


def main():
    """主函数"""
    # 确保目录存在
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    monitor = DailyMonitor()
    state = monitor.run()
    
    # 返回状态码
    if state['alerts']:
        return 1  # 有预警
    return 0


if __name__ == '__main__':
    sys.exit(main())
