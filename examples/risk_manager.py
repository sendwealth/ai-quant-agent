#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
风险管理系统
Risk Management System

功能:
1. 仓位风险控制
2. 止损止盈管理
3. 回撤控制
4. 风险预警
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
import os

class RiskManager:
    """风险管理器"""
    
    def __init__(self, portfolio_file='data/auto_portfolio.json'):
        self.portfolio_file = portfolio_file
        self.load_portfolio()
        
        # 风险参数
        self.risk_params = {
            'max_position_pct': 0.15,        # 单股最大仓位15%
            'max_sector_pct': 0.30,           # 单行业最大仓位30%
            'max_total_position': 0.80,       # 最大总仓位80%
            'stop_loss_pct': -0.05,           # 止损-5%
            'take_profit_pct': 0.15,          # 止盈15%
            'max_drawdown_pct': -0.10,        # 最大回撤-10%
            'max_daily_loss_pct': -0.03,      # 单日最大亏损-3%
            'min_cash_reserve': 0.20,         # 最小现金储备20%
        }
        
        # 黑名单（风险股）
        self.blacklist = []
        
    def load_portfolio(self):
        """加载持仓"""
        try:
            with open(self.portfolio_file, 'r', encoding='utf-8') as f:
                self.portfolio = json.load(f)
        except:
            self.portfolio = {
                'initial_capital': 100000,
                'cash': 100000,
                'positions': {},
                'trades': []
            }
    
    def save_portfolio(self):
        """保存持仓"""
        with open(self.portfolio_file, 'w', encoding='utf-8') as f:
            json.dump(self.portfolio, f, ensure_ascii=False, indent=2)
    
    def calculate_position_risk(self, stock_code, position_size, current_price):
        """计算仓位风险"""
        total_capital = self.portfolio['initial_capital']
        position_value = position_size * current_price
        position_pct = position_value / total_capital
        
        risks = []
        
        # 检查单股仓位
        if position_pct > self.risk_params['max_position_pct']:
            risks.append({
                'type': 'POSITION_SIZE',
                'level': 'HIGH',
                'message': f"单股仓位 {position_pct:.1%} 超过限制 {self.risk_params['max_position_pct']:.1%}",
                'current': position_pct,
                'limit': self.risk_params['max_position_pct']
            })
        
        # 检查总仓位
        current_total_position = sum(
            p['shares'] * p.get('current_price', p['cost_price'])
            for p in self.portfolio['positions'].values()
        )
        new_total_position = current_total_position + position_value
        total_position_pct = new_total_position / total_capital
        
        if total_position_pct > self.risk_params['max_total_position']:
            risks.append({
                'type': 'TOTAL_POSITION',
                'level': 'HIGH',
                'message': f"总仓位 {total_position_pct:.1%} 超过限制 {self.risk_params['max_total_position']:.1%}",
                'current': total_position_pct,
                'limit': self.risk_params['max_total_position']
            })
        
        # 检查现金储备
        new_cash = self.portfolio['cash'] - position_value
        cash_pct = new_cash / total_capital
        
        if cash_pct < self.risk_params['min_cash_reserve']:
            risks.append({
                'type': 'CASH_RESERVE',
                'level': 'MEDIUM',
                'message': f"现金储备 {cash_pct:.1%} 低于要求 {self.risk_params['min_cash_reserve']:.1%}",
                'current': cash_pct,
                'limit': self.risk_params['min_cash_reserve']
            })
        
        # 检查黑名单
        if stock_code in self.blacklist:
            risks.append({
                'type': 'BLACKLIST',
                'level': 'CRITICAL',
                'message': f"股票 {stock_code} 在黑名单中",
                'current': stock_code,
                'limit': 'N/A'
            })
        
        return risks
    
    def check_stop_loss(self, stock_code, cost_price, current_price):
        """检查止损"""
        if cost_price <= 0:
            return None
        
        pnl_pct = (current_price - cost_price) / cost_price
        
        if pnl_pct <= self.risk_params['stop_loss_pct']:
            return {
                'action': 'STOP_LOSS',
                'stock_code': stock_code,
                'cost_price': cost_price,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'message': f"触发止损: 亏损 {pnl_pct:.2%}，建议立即卖出"
            }
        
        return None
    
    def check_take_profit(self, stock_code, cost_price, current_price):
        """检查止盈"""
        if cost_price <= 0:
            return None
        
        pnl_pct = (current_price - cost_price) / cost_price
        
        if pnl_pct >= self.risk_params['take_profit_pct']:
            return {
                'action': 'TAKE_PROFIT',
                'stock_code': stock_code,
                'cost_price': cost_price,
                'current_price': current_price,
                'pnl_pct': pnl_pct,
                'message': f"触发止盈: 盈利 {pnl_pct:.2%}，建议部分或全部卖出"
            }
        
        return None
    
    def calculate_portfolio_risk(self):
        """计算组合风险"""
        total_capital = self.portfolio['initial_capital']
        current_capital = self.calculate_total_capital()
        
        # 计算回撤
        drawdown = (current_capital - total_capital) / total_capital
        
        # 计算持仓集中度
        positions = self.portfolio['positions']
        if not positions:
            return {
                'drawdown': 0,
                'position_count': 0,
                'concentration': 0,
                'total_position_pct': 0,
                'cash_pct': 1.0,
                'risk_level': 'LOW'
            }
        
        # 持仓市值
        position_values = []
        for code, pos in positions.items():
            value = pos['shares'] * pos.get('current_price', pos['cost_price'])
            position_values.append(value)
        
        total_position = sum(position_values)
        position_count = len(positions)
        
        # 集中度（HHI指数）
        if total_position > 0:
            weights = [v / total_position for v in position_values]
            hhi = sum(w ** 2 for w in weights)
        else:
            hhi = 0
        
        # 风险等级
        risk_level = 'LOW'
        if drawdown < self.risk_params['max_drawdown_pct']:
            risk_level = 'CRITICAL'
        elif drawdown < self.risk_params['max_drawdown_pct'] * 0.7:
            risk_level = 'HIGH'
        elif drawdown < self.risk_params['max_drawdown_pct'] * 0.4:
            risk_level = 'MEDIUM'
        
        return {
            'drawdown': drawdown,
            'position_count': position_count,
            'concentration': hhi,
            'total_position_pct': total_position / total_capital,
            'cash_pct': self.portfolio['cash'] / total_capital,
            'risk_level': risk_level
        }
    
    def calculate_total_capital(self):
        """计算总资产"""
        cash = self.portfolio['cash']
        
        position_value = sum(
            pos['shares'] * pos.get('current_price', pos['cost_price'])
            for pos in self.portfolio['positions'].values()
        )
        
        return cash + position_value
    
    def assess_trade_risk(self, stock_code, action, shares, price):
        """评估交易风险"""
        risks = []
        
        if action == 'BUY':
            # 买入风险
            risks = self.calculate_position_risk(stock_code, shares, price)
        elif action == 'SELL':
            # 卖出检查
            if stock_code not in self.portfolio['positions']:
                risks.append({
                    'type': 'NO_POSITION',
                    'level': 'HIGH',
                    'message': f"没有持有 {stock_code}，无法卖出",
                    'current': 0,
                    'limit': 'N/A'
                })
        
        return risks
    
    def generate_risk_report(self):
        """生成风险报告"""
        print(f"\n{'='*70}")
        print(f"风险报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"{'='*70}\n")
        
        # 组合风险
        portfolio_risk = self.calculate_portfolio_risk()
        
        print("📊 组合风险:")
        print("-" * 70)
        
        total_capital = self.calculate_total_capital()
        initial_capital = self.portfolio['initial_capital']
        pnl = total_capital - initial_capital
        pnl_pct = pnl / initial_capital
        
        print(f"总资产: {total_capital:,.2f} | "
              f"盈亏: {pnl:+,.2f} ({pnl_pct:+.2%})")
        print(f"持仓数: {portfolio_risk['position_count']} | "
              f"仓位: {portfolio_risk['total_position_pct']:.1%} | "
              f"现金: {portfolio_risk['cash_pct']:.1%}")
        
        # 回撤
        dd_emoji = '🔴' if portfolio_risk['drawdown'] < self.risk_params['max_drawdown_pct'] * 0.7 else '🟡' if portfolio_risk['drawdown'] < 0 else '🟢'
        print(f"{dd_emoji} 回撤: {portfolio_risk['drawdown']:.2%}")
        
        # 集中度
        conc_emoji = '🔴' if portfolio_risk['concentration'] > 0.25 else '🟡' if portfolio_risk['concentration'] > 0.15 else '🟢'
        print(f"{conc_emoji} 集中度: {portfolio_risk['concentration']:.3f} "
              f"({'高集中' if portfolio_risk['concentration'] > 0.25 else '中集中' if portfolio_risk['concentration'] > 0.15 else '分散'})")
        
        # 风险等级
        risk_emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}
        print(f"{risk_emoji.get(portfolio_risk['risk_level'], '⚪')} 风险等级: {portfolio_risk['risk_level']}")
        
        print()
        
        # 持仓风险
        print("📋 持仓风险:")
        print("-" * 70)
        
        for code, pos in self.portfolio['positions'].items():
            current_price = pos.get('current_price', pos['cost_price'])
            pnl_pct = (current_price - pos['cost_price']) / pos['cost_price']
            
            # 止损检查
            stop_loss = self.check_stop_loss(code, pos['cost_price'], current_price)
            take_profit = self.check_take_profit(code, pos['cost_price'], current_price)
            
            pnl_emoji = '🔴' if pnl_pct < -0.03 else '🟡' if pnl_pct < 0 else '🟢'
            alert = ''
            
            if stop_loss:
                alert = f" ⚠️ {stop_loss['message']}"
            elif take_profit:
                alert = f" ✅ {take_profit['message']}"
            
            print(f"{code}: {pos['shares']}股 @ {pos['cost_price']:.2f} → {current_price:.2f} "
                  f"{pnl_emoji} {pnl_pct:+.2%}{alert}")
        
        if not self.portfolio['positions']:
            print("空仓")
        
        print()
        
        # 风险建议
        print("💡 风险建议:")
        print("-" * 70)
        
        suggestions = []
        
        if portfolio_risk['drawdown'] < self.risk_params['max_drawdown_pct']:
            suggestions.append("🚨 紧急: 回撤超限，建议减仓或清仓")
        elif portfolio_risk['drawdown'] < self.risk_params['max_drawdown_pct'] * 0.7:
            suggestions.append("⚠️ 警告: 接近最大回撤，建议控制仓位")
        
        if portfolio_risk['cash_pct'] < self.risk_params['min_cash_reserve']:
            suggestions.append("💰 现金储备不足，建议保留更多现金")
        
        if portfolio_risk['concentration'] > 0.25:
            suggestions.append("📊 持仓过于集中，建议分散投资")
        
        if portfolio_risk['position_count'] > 10:
            suggestions.append("📉 持仓过多，建议精简到5-10只")
        
        if not suggestions:
            suggestions.append("✅ 风险控制良好，继续保持")
        
        for suggestion in suggestions:
            print(suggestion)
        
        print(f"\n{'='*70}\n")
        
        return portfolio_risk
    
    def add_to_blacklist(self, stock_code, reason=''):
        """添加到黑名单"""
        if stock_code not in self.blacklist:
            self.blacklist.append(stock_code)
            print(f"✅ 已将 {stock_code} 加入黑名单: {reason}")
    
    def remove_from_blacklist(self, stock_code):
        """从黑名单移除"""
        if stock_code in self.blacklist:
            self.blacklist.remove(stock_code)
            print(f"✅ 已将 {stock_code} 从黑名单移除")

def main():
    """主函数"""
    rm = RiskManager()
    rm.generate_risk_report()

if __name__ == '__main__':
    main()
