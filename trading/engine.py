#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一交易引擎
Unified Trading Engine

使用新的core模块，整合所有交易功能
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional, Tuple
from loguru import logger

# 导入核心模块
from core.indicators import sma, ema, atr, rsi, macd
from core.base_strategy import MAStrategy, StrategyFactory
from core.config_loader import ConfigLoader, get_config


class TradingEngine:
    """
    交易引擎
    
    统一的交易系统，整合:
    - 策略执行
    - 风险管理
    - 仓位管理
    - 交易记录
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化交易引擎
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config_loader = get_config() if config_path is None else ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # 资金
        capital_config = self.config_loader.get_capital_config()
        self.initial_capital = capital_config.get('initial', 100000)
        self.cash = self.initial_capital
        
        # 持仓
        self.positions: Dict[str, dict] = {}
        
        # 交易记录
        self.trades: List[dict] = []
        
        # 创建策略
        self.strategies: Dict[str, MAStrategy] = {}
        self._init_strategies()
        
        # 数据
        self.stock_data: Dict[str, pd.DataFrame] = {}
        self._load_data()
        
        # 加载已有持仓
        self._load_positions()
        
        logger.info(f"交易引擎初始化完成 | 初始资金: {self.initial_capital}")
    
    def _init_strategies(self):
        """初始化策略"""
        stocks = self.config_loader.get_all_stocks(enabled_only=True)
        
        for stock in stocks:
            code = stock['code']
            strategy_config = {
                'name': stock['name'],
                'weight': stock['weight'],
                'params': stock['params'],
                'risk': self.config_loader.get_risk_params()
            }
            
            self.strategies[code] = MAStrategy(strategy_config)
            logger.info(f"策略创建: {code} {stock['name']}")
    
    def _load_data(self):
        """加载股票数据"""
        logger.info("加载股票数据...")
        
        stocks = self.config_loader.get_all_stocks(enabled_only=True)
        data_dir = Path(__file__).parent.parent / 'data'
        
        for stock in stocks:
            code = stock['code']
            filepath = data_dir / f'real_{code}.csv'
            
            if filepath.exists():
                try:
                    df = pd.read_csv(filepath)
                    
                    # 标准化列名
                    if 'datetime' not in df.columns and 'trade_date' in df.columns:
                        df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
                    
                    df = df.sort_values('datetime').reset_index(drop=True)
                    self.stock_data[code] = df
                    
                    logger.info(f"  ✅ {stock['name']}: {len(df)}天")
                except Exception as e:
                    logger.error(f"  ❌ {stock['name']}加载失败: {e}")
            else:
                logger.warning(f"  ⚠️  {stock['name']}: 数据文件不存在")
    
    def _load_positions(self):
        """加载已有持仓"""
        filepath = Path(__file__).parent.parent / 'data' / 'auto_portfolio.json'
        
        if filepath.exists():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.cash = data.get('cash', self.initial_capital)
                self.positions = data.get('positions', {})
                self.trades = data.get('trades', [])
                
                logger.info(f"加载已有持仓: 现金{self.cash:.2f}, {len(self.positions)}只股票")
            except Exception as e:
                logger.error(f"加载持仓失败: {e}")
    
    def _save_positions(self):
        """保存持仓"""
        filepath = Path(__file__).parent.parent / 'data' / 'auto_portfolio.json'
        
        data = {
            'update_time': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'trades': self.trades[-100:],  # 只保留最近100条
            'performance': self._calculate_performance()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"持仓已保存: {filepath}")
    
    def _calculate_performance(self) -> dict:
        """计算绩效"""
        total_value = self.cash
        
        # 计算持仓市值
        for code, position in self.positions.items():
            if code in self.stock_data and not self.stock_data[code].empty:
                current_price = self.stock_data[code]['close'].iloc[-1]
                total_value += position['shares'] * current_price
        
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        return {
            'total_value': total_value,
            'total_return': total_return,
            'cash_ratio': self.cash / total_value if total_value > 0 else 0
        }
    
    def run_daily_analysis(self) -> dict:
        """
        运行每日分析
        
        Returns:
            分析结果
        """
        logger.info("\n" + "="*50)
        logger.info("📊 每日量化分析")
        logger.info(f"⏰ 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*50 + "\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'positions': {},
            'alerts': [],
            'summary': {}
        }
        
        # 1. 分析每只股票
        for code, strategy in self.strategies.items():
            if code not in self.stock_data:
                logger.warning(f"⚠️  {code}: 无数据")
                continue
            
            data = self.stock_data[code]
            name = strategy.name
            
            # 生成信号
            signals = strategy.generate_signals(data)
            latest_signal = signals.iloc[-1]
            
            # 最新价格
            latest_price = data['close'].iloc[-1]
            latest_change = data['pct_chg'].iloc[-1] if 'pct_chg' in data.columns else 0
            
            # 保存结果
            results['signals'][code] = {
                'name': name,
                'signal': 'BUY' if latest_signal == 1 else ('SELL' if latest_signal == -1 else 'HOLD'),
                'price': latest_price,
                'change_pct': latest_change
            }
            
            # 检查持仓风险
            if code in self.positions:
                should_close, reason = strategy.check_risk(self.positions[code], latest_price)
                
                if should_close:
                    results['alerts'].append({
                        'code': code,
                        'name': name,
                        'type': 'RISK',
                        'reason': reason,
                        'price': latest_price
                    })
            
            # 打印信号
            signal_str = results['signals'][code]['signal']
            logger.info(f"  {name}: {signal_str} | {latest_change:+.2f}%")
        
        # 2. 生成摘要
        buy_count = sum(1 for s in results['signals'].values() if s['signal'] == 'BUY')
        sell_count = sum(1 for s in results['signals'].values() if s['signal'] == 'SELL')
        
        results['summary'] = {
            'total_stocks': len(self.strategies),
            'buy_count': buy_count,
            'sell_count': sell_count,
            'alert_count': len(results['alerts']),
            'performance': self._calculate_performance()
        }
        
        # 3. 打印摘要
        logger.info(f"\n{'='*50}")
        logger.info("📋 今日摘要:")
        logger.info(f"  买入信号: {buy_count}个")
        logger.info(f"  卖出信号: {sell_count}个")
        logger.info(f"  预警数量: {len(results['alerts'])}个")
        
        if results['alerts']:
            logger.info("\n⚠️  预警:")
            for alert in results['alerts']:
                logger.info(f"  - {alert['name']}: {alert['reason']}")
        
        logger.info(f"\n💰 绩效:")
        perf = results['summary']['performance']
        logger.info(f"  总资产: {perf['total_value']:.2f}")
        logger.info(f"  总收益: {perf['total_return']*100:+.2f}%")
        logger.info(f"{'='*50}\n")
        
        return results
    
    def execute_signal(self, code: str, signal: int, price: float):
        """
        执行交易信号
        
        Args:
            code: 股票代码
            signal: 信号 (1=买入, -1=卖出)
            price: 价格
        """
        if code not in self.strategies:
            logger.warning(f"未知股票: {code}")
            return
        
        strategy = self.strategies[code]
        
        if signal == 1:  # 买入
            # 检查是否已持仓
            if code in self.positions:
                logger.info(f"  {strategy.name}: 已持仓，跳过买入")
                return
            
            # 计算仓位
            shares = strategy.calculate_position_size(self.cash, price)
            
            if shares <= 0:
                logger.warning(f"  {strategy.name}: 资金不足")
                return
            
            # 执行买入
            cost = shares * price
            self.cash -= cost
            self.positions[code] = {
                'code': code,
                'name': strategy.name,
                'shares': shares,
                'entry_price': price,
                'entry_time': datetime.now().isoformat()
            }
            
            # 记录交易
            self.trades.append({
                'time': datetime.now().isoformat(),
                'code': code,
                'name': strategy.name,
                'action': 'BUY',
                'price': price,
                'shares': shares,
                'cost': cost
            })
            
            logger.info(f"  ✅ 买入 {strategy.name}: {shares}股 @ {price:.2f} = {cost:.2f}")
        
        elif signal == -1:  # 卖出
            # 检查是否持仓
            if code not in self.positions:
                logger.info(f"  {strategy.name}: 无持仓，跳过卖出")
                return
            
            # 执行卖出
            position = self.positions[code]
            revenue = position['shares'] * price
            self.cash += revenue
            
            # 记录交易
            pnl = (price - position['entry_price']) * position['shares']
            self.trades.append({
                'time': datetime.now().isoformat(),
                'code': code,
                'name': strategy.name,
                'action': 'SELL',
                'price': price,
                'shares': position['shares'],
                'revenue': revenue,
                'pnl': pnl
            })
            
            logger.info(f"  ✅ 卖出 {strategy.name}: {position['shares']}股 @ {price:.2f} = {revenue:.2f} (盈亏: {pnl:+.2f})")
            
            # 删除持仓
            del self.positions[code]
        
        # 保存持仓
        self._save_positions()
    
    def run(self, auto_trade: bool = False):
        """
        运行交易引擎
        
        Args:
            auto_trade: 是否自动交易
        """
        results = self.run_daily_analysis()
        
        if auto_trade:
            logger.info("\n开始自动交易...")
            
            for code, signal_info in results['signals'].items():
                signal_str = signal_info['signal']
                price = signal_info['price']
                
                if signal_str == 'BUY':
                    self.execute_signal(code, 1, price)
                elif signal_str == 'SELL':
                    self.execute_signal(code, -1, price)
        
        return results


def main():
    """主函数"""
    # 创建交易引擎
    engine = TradingEngine()
    
    # 运行分析
    results = engine.run(auto_trade=False)
    
    # 保存结果
    output_path = Path(__file__).parent.parent / 'data' / 'daily_analysis_results.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n结果已保存: {output_path}")


if __name__ == '__main__':
    main()
