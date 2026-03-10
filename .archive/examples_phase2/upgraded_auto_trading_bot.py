#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
升级版自动交易系统 - 整合多因子选股
Upgraded Auto Trading System with Multi-Factor Selection

整合新策略到自动交易
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tushare as ts
import sys
import os

# 设置TuShare
ts.set_token('33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b')
pro = ts.pro_api()

class UpgradedAutoTrader:
    """升级版自动交易系统"""
    
    def __init__(self):
        self.initial_capital = 100000
        self.cash = 100000
        self.positions = {}
        self.trades = []
        
        # 加载持仓
        self.load_portfolio()
        
        # 风险参数
        self.risk_params = {
            'max_position_pct': 0.15,      # 单股最大15%
            'max_total_position': 0.80,    # 总仓位最大80%
            'stop_loss': -0.05,            # 止损-5%
            'take_profit_1': 0.10,         # 止盈1: 10%
            'take_profit_2': 0.20,         # 止盈2: 20%
            'min_cash_reserve': 0.20       # 最小现金20%
        }
        
        # 选股参数
        self.top_n_stocks = 10  # 从TOP 10中选股
        self.position_stocks = 5  # 最多持有5只
        
    def load_portfolio(self):
        """加载持仓"""
        try:
            with open('data/auto_portfolio.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.initial_capital = data.get('initial_capital', 100000)
                self.cash = data.get('cash', 100000)
                self.positions = data.get('positions', {})
                self.trades = data.get('trades', [])
        except:
            pass
    
    def save_portfolio(self):
        """保存持仓"""
        data = {
            'update_time': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'trades': self.trades
        }
        
        with open('data/auto_portfolio.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def fetch_stock_data(self, ts_code, days=120):
        """获取股票数据"""
        try:
            end_date = datetime.now().strftime('%Y%m%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')
            
            df = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
            
            if df.empty:
                return None
            
            return df.sort_values('trade_date')
        except:
            return None
    
    def calculate_factors(self, ts_code):
        """计算因子（简化版）"""
        df = self.fetch_stock_data(ts_code)
        
        if df is None or len(df) < 60:
            return None
        
        latest = df.iloc[-1]
        factors = {}
        
        # 动量因子
        if len(df) >= 20:
            price_20d_ago = df.iloc[-20]['close']
            factors['momentum_1m'] = (latest['close'] - price_20d_ago) / price_20d_ago
        else:
            factors['momentum_1m'] = 0
        
        if len(df) >= 60:
            price_60d_ago = df.iloc[-60]['close']
            factors['momentum_3m'] = (latest['close'] - price_60d_ago) / price_60d_ago
        else:
            factors['momentum_3m'] = 0
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        factors['rsi'] = 100 - (100 / (1 + rs.iloc[-1]))
        
        # 价格
        factors['price'] = latest['close']
        factors['pct_chg'] = latest['pct_chg'] / 100
        
        return factors
    
    def score_stock(self, ts_code, all_factors):
        """评分股票"""
        factors = self.calculate_factors(ts_code)
        
        if not factors:
            return None
        
        # 简化评分
        score = 0
        
        # 动量（权重50%）
        if factors['momentum_1m'] > 0.03:
            score += 30
        elif factors['momentum_1m'] > 0:
            score += 20
        elif factors['momentum_1m'] > -0.03:
            score += 10
        
        if factors['momentum_3m'] > 0.15:
            score += 20
        elif factors['momentum_3m'] > 0:
            score += 10
        
        # RSI（权重30%）
        if 30 < factors['rsi'] < 70:
            score += 20
        elif factors['rsi'] < 30:
            score += 10  # 超卖可能反弹
        
        # 涨跌幅（权重20%）
        if factors['pct_chg'] > 0.02:
            score += 10
        elif factors['pct_chg'] > 0:
            score += 5
        
        return {
            'code': ts_code,
            'score': score,
            'factors': factors
        }
    
    def select_stocks(self):
        """选股"""
        print(f"\n{'='*70}")
        print(f"多因子选股 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 加载股票池
        with open('data/stock_pool_extended.json', 'r', encoding='utf-8') as f:
            pool_data = json.load(f)
        
        all_stocks = []
        for category, data in pool_data['categories'].items():
            all_stocks.extend(data['stocks'])
        
        print(f"📊 分析 {len(all_stocks)} 只股票...")
        
        # 计算所有股票的因子
        all_factors = {}
        for stock in all_stocks:
            code = stock['code']
            # 转换代码格式
            if code.startswith('6'):
                ts_code = f"{code}.SH"
            else:
                ts_code = f"{code}.SZ"
            
            factors = self.calculate_factors(ts_code)
            if factors:
                all_factors[ts_code] = {
                    'factors': factors,
                    'info': stock
                }
        
        print(f"✅ 成功计算 {len(all_factors)} 只股票的因子\n")
        
        # 评分
        print("📈 评分排序...")
        scored_stocks = []
        
        for ts_code, data in all_factors.items():
            result = self.score_stock(ts_code, all_factors)
            if result:
                result['info'] = data['info']
                scored_stocks.append(result)
        
        # 排序
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        
        # 输出TOP N
        print(f"\n{'='*70}")
        print(f"TOP {self.top_n_stocks} 推荐股票")
        print(f"{'='*70}\n")
        
        for i, stock in enumerate(scored_stocks[:self.top_n_stocks], 1):
            info = stock['info']
            factors = stock['factors']
            
            print(f"#{i} {info['name']} ({info['code']}) - {info.get('sector', 'N/A')}")
            print(f"   价格: {factors['price']:.2f} | 得分: {stock['score']}")
            print(f"   1月动量: {factors['momentum_1m']:+.1%} | "
                  f"3月动量: {factors['momentum_3m']:+.1%} | "
                  f"RSI: {factors['rsi']:.1f}")
            print()
        
        return scored_stocks[:self.top_n_stocks]
    
    def check_risk(self):
        """风险检查"""
        # 检查持仓风险
        for code, pos in self.positions.items():
            current_price = pos.get('current_price', pos['cost_price'])
            pnl_pct = (current_price - pos['cost_price']) / pos['cost_price']
            
            # 止损检查
            if pnl_pct <= self.risk_params['stop_loss']:
                return {
                    'action': 'STOP_LOSS',
                    'code': code,
                    'message': f"{pos['name']} 触发止损 ({pnl_pct:.2%})"
                }
            
            # 止盈检查
            if pnl_pct >= self.risk_params['take_profit_1']:
                return {
                    'action': 'TAKE_PROFIT',
                    'code': code,
                    'message': f"{pos['name']} 触发止盈 ({pnl_pct:.2%})"
                }
        
        return None
    
    def execute_trade(self, signal):
        """执行交易"""
        if signal['action'] == 'STOP_LOSS':
            # 止损卖出
            code = signal['code']
            pos = self.positions[code]
            
            # 获取当前价格
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            df = self.fetch_stock_data(ts_code, days=1)
            if df is not None and not df.empty:
                current_price = df.iloc[-1]['close']
            else:
                current_price = pos['cost_price'] * (1 + self.risk_params['stop_loss'])
            
            # 卖出
            sell_amount = pos['shares'] * current_price
            self.cash += sell_amount
            
            # 记录交易
            self.trades.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'action': 'SELL',
                'code': code,
                'name': pos['name'],
                'shares': pos['shares'],
                'price': current_price,
                'amount': sell_amount,
                'reason': 'STOP_LOSS'
            })
            
            # 删除持仓
            del self.positions[code]
            
            print(f"⚠️ 止损卖出: {pos['name']} {pos['shares']}股 @ {current_price:.2f}")
            
        elif signal['action'] == 'TAKE_PROFIT':
            # 止盈卖出
            code = signal['code']
            pos = self.positions[code]
            
            # 获取当前价格
            ts_code = f"{code}.SH" if code.startswith('6') else f"{code}.SZ"
            df = self.fetch_stock_data(ts_code, days=1)
            if df is not None and not df.empty:
                current_price = df.iloc[-1]['close']
            else:
                current_price = pos['cost_price'] * (1 + self.risk_params['take_profit_1'])
            
            # 计算卖出数量（卖出50%）
            sell_shares = int(pos['shares'] * 0.5)
            sell_amount = sell_shares * current_price
            self.cash += sell_amount
            
            # 更新持仓
            self.positions[code]['shares'] -= sell_shares
            
            # 记录交易
            self.trades.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'action': 'SELL',
                'code': code,
                'name': pos['name'],
                'shares': sell_shares,
                'price': current_price,
                'amount': sell_amount,
                'reason': 'TAKE_PROFIT_1'
            })
            
            print(f"✅ 止盈卖出: {pos['name']} {sell_shares}股 @ {current_price:.2f}")
    
    def buy_stocks(self, top_stocks):
        """买入股票"""
        print(f"\n{'='*70}")
        print("执行买入")
        print(f"{'='*70}\n")
        
        # 计算可用资金
        max_total_position = self.initial_capital * self.risk_params['max_total_position']
        current_position_value = sum(
            pos['shares'] * pos.get('current_price', pos['cost_price'])
            for pos in self.positions.values()
        )
        
        available_for_stocks = max_total_position - current_position_value
        available_cash = min(self.cash * 0.8, available_for_stocks)  # 保留20%现金
        
        # 当前持仓数
        current_holdings = len(self.positions)
        can_buy = self.position_stocks - current_holdings
        
        if can_buy <= 0:
            print("⚠️ 已达到最大持仓数量")
            return
        
        print(f"可用资金: {available_cash:,.0f}元")
        print(f"可买入: {can_buy}只股票\n")
        
        # 选择要买的股票
        for stock in top_stocks[:can_buy]:
            code = stock['code']
            
            # 检查是否已持有
            if code in self.positions:
                continue
            
            # 检查是否在黑名单
            # TODO: 添加黑名单检查
            
            # 计算买入金额
            position_size = available_cash / can_buy
            position_size = min(position_size, self.initial_capital * self.risk_params['max_position_pct'])
            
            # 获取价格
            ts_code = stock['code']
            factors = stock['factors']
            price = factors['price']
            
            # 计算股数
            shares = int(position_size / price / 100) * 100  # 整手
            
            if shares <= 0:
                continue
            
            # 检查资金
            buy_amount = shares * price
            if buy_amount > self.cash:
                continue
            
            # 买入
            self.cash -= buy_amount
            self.positions[code] = {
                'name': stock['info']['name'],
                'shares': shares,
                'cost_price': price,
                'current_price': price,
                'buy_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            # 记录交易
            self.trades.append({
                'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'action': 'BUY',
                'code': code,
                'name': stock['info']['name'],
                'shares': shares,
                'price': price,
                'amount': buy_amount,
                'reason': 'MULTI_FACTOR_SELECTION'
            })
            
            print(f"✅ 买入: {stock['info']['name']} {shares}股 @ {price:.2f} = {buy_amount:,.0f}元")
        
        print()
    
    def run(self):
        """运行交易系统"""
        print(f"\n{'='*70}")
        print(f"升级版自动交易系统 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 1. 风险检查
        print("🔍 风险检查...")
        risk_signal = self.check_risk()
        
        if risk_signal:
            print(f"⚠️ {risk_signal['message']}")
            self.execute_trade(risk_signal)
        else:
            print("✅ 风险检查通过\n")
        
        # 2. 选股
        top_stocks = self.select_stocks()
        
        if not top_stocks:
            print("❌ 选股失败")
            return
        
        # 3. 买入
        self.buy_stocks(top_stocks)
        
        # 4. 保存
        self.save_portfolio()
        
        # 5. 汇总
        self.summary()
    
    def summary(self):
        """持仓汇总"""
        print(f"\n{'='*70}")
        print("持仓汇总")
        print(f"{'='*70}\n")
        
        if not self.positions:
            print("空仓")
        else:
            total_position_value = 0
            
            for code, pos in self.positions.items():
                current_price = pos.get('current_price', pos['cost_price'])
                position_value = pos['shares'] * current_price
                pnl = (current_price - pos['cost_price']) * pos['shares']
                pnl_pct = (current_price - pos['cost_price']) / pos['cost_price']
                
                total_position_value += position_value
                
                pnl_emoji = '🟢' if pnl >= 0 else '🔴'
                
                print(f"{pos['name']} ({code})")
                print(f"  持仓: {pos['shares']}股 @ {pos['cost_price']:.2f} → {current_price:.2f}")
                print(f"  市值: {position_value:,.0f}元 | {pnl_emoji} 盈亏: {pnl:+,.0f}元 ({pnl_pct:+.2%})")
                print()
            
            total_asset = self.cash + total_position_value
            total_pnl = total_asset - self.initial_capital
            total_pnl_pct = total_pnl / self.initial_capital
            
            print(f"总资产: {total_asset:,.0f}元")
            print(f"现金: {self.cash:,.0f}元 ({self.cash/total_asset:.1%})")
            print(f"持仓: {total_position_value:,.0f}元 ({total_position_value/total_asset:.1%})")
            print(f"总盈亏: {total_pnl:+,.0f}元 ({total_pnl_pct:+.2%})")
        
        print(f"\n📊 交易统计:")
        print(f"  总交易: {len(self.trades)}次")
        print(f"  当前持仓: {len(self.positions)}只")
        
        print(f"\n{'='*70}\n")
        print("✅ 运行完成！\n")

def main():
    """主函数"""
    trader = UpgradedAutoTrader()
    trader.run()

if __name__ == '__main__':
    main()
