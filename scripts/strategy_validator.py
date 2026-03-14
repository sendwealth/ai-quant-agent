#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
策略验证器
Strategy Validator

对比测试不同策略版本的性能
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import sma, ema, atr, rsi, macd
from core.config_loader import ConfigLoader


class StrategyValidator:
    """策略验证器"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.config_dir = Path(__file__).parent.parent / 'config'
    
    def load_stock_data(self, code: str) -> pd.DataFrame:
        """加载股票数据"""
        filepath = self.data_dir / f'real_{code}.csv'
        
        if not filepath.exists():
            return None
        
        df = pd.read_csv(filepath)
        
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    
    def backtest_strategy(self, 
                         data: pd.DataFrame,
                         params: Dict,
                         initial_capital: float = 100000) -> Dict:
        """
        回测策略
        
        Args:
            data: 股票数据
            params: 策略参数
            initial_capital: 初始资金
        
        Returns:
            回测结果
        """
        df = data.copy()
        
        # 计算指标
        df['ma_fast'] = sma(df['close'], params['ma_fast'])
        df['ma_slow'] = sma(df['close'], params['ma_slow'])
        df['atr'] = atr(df['high'], df['low'], df['close'], 14)
        df['macd'], df['macd_signal'], _ = macd(df['close'])
        df['rsi'] = rsi(df['close'], 14)
        
        # 模拟交易
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        daily_values = []
        
        for i in range(50, len(df)):
            row = df.iloc[i]
            
            # 记录每日价值
            total_value = capital + position * row['close']
            daily_values.append({
                'date': row['datetime'],
                'value': total_value
            })
            
            # 买入信号
            if position == 0:
                buy_signal = (
                    row['ma_fast'] > row['ma_slow'] and
                    row['macd'] > row['macd_signal'] and
                    30 < row['rsi'] < 70
                )
                
                if buy_signal:
                    position = capital / row['close']
                    entry_price = row['close']
                    capital = 0
                    trades.append({
                        'type': 'BUY',
                        'price': row['close'],
                        'date': row['datetime']
                    })
            
            # 卖出信号
            elif position > 0:
                # ATR止损
                stop_price = entry_price - params['atr_stop'] * row['atr']
                
                # 止盈检查
                pnl_pct = (row['close'] - entry_price) / entry_price
                
                sell_signal = (
                    row['ma_fast'] < row['ma_slow'] or
                    row['close'] < stop_price or
                    pnl_pct > 0.20  # 20%止盈
                )
                
                if sell_signal:
                    capital = position * row['close']
                    pnl = (row['close'] - entry_price) * position
                    trades.append({
                        'type': 'SELL',
                        'price': row['close'],
                        'date': row['datetime'],
                        'pnl': pnl
                    })
                    position = 0
                    entry_price = 0
        
        # 计算绩效
        final_value = capital + position * df['close'].iloc[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        # 计算夏普比率
        values_df = pd.DataFrame(daily_values)
        values_df['returns'] = values_df['value'].pct_change()
        
        if len(values_df) > 1:
            avg_return = values_df['returns'].mean()
            std_return = values_df['returns'].std()
            sharpe = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
        else:
            sharpe = 0
        
        # 最大回撤
        values_df['cummax'] = values_df['value'].cummax()
        values_df['drawdown'] = (values_df['value'] - values_df['cummax']) / values_df['cummax']
        max_drawdown = values_df['drawdown'].min()
        
        # 胜率
        winning_trades = [t for t in trades if t['type'] == 'SELL' and t.get('pnl', 0) > 0]
        total_sells = len([t for t in trades if t['type'] == 'SELL'])
        win_rate = len(winning_trades) / total_sells if total_sells > 0 else 0
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': len(trades) // 2
        }
    
    def compare_strategies(self) -> Dict:
        """对比不同策略版本"""
        logger.info("\n" + "="*60)
        logger.info("📊 策略对比测试")
        logger.info("="*60)
        
        results = {}
        
        # 加载V4和V5配置
        v4_config = ConfigLoader(self.config_dir / 'strategy_v4.yaml')
        v5_config = ConfigLoader(self.config_dir / 'strategy_v5.yaml')
        
        stocks = v4_config.get_all_stocks(enabled_only=True)
        
        for stock in stocks:
            code = stock['code']
            name = stock['name']
            
            # 加载数据
            data = self.load_stock_data(code)
            if data is None:
                continue
            
            logger.info(f"\n📈 {name} ({code})")
            
            # V4回测
            v4_params = stock['params']
            v4_result = self.backtest_strategy(data, v4_params)
            
            # V5回测
            v5_stock = next((s for s in v5_config.get_all_stocks(enabled_only=True) 
                           if s['code'] == code), None)
            
            if v5_stock:
                v5_params = v5_stock['params']
                v5_result = self.backtest_strategy(data, v5_params)
            else:
                v5_result = None
            
            results[code] = {
                'name': name,
                'v4': v4_result,
                'v5': v5_result
            }
            
            # 打印对比
            logger.info(f"  V4: 收益 {v4_result['total_return']*100:+.2f}%, "
                       f"夏普 {v4_result['sharpe']:.3f}, "
                       f"胜率 {v4_result['win_rate']*100:.1f}%")
            
            if v5_result:
                improvement = v5_result['total_return'] - v4_result['total_return']
                logger.info(f"  V5: 收益 {v5_result['total_return']*100:+.2f}%, "
                           f"夏普 {v5_result['sharpe']:.3f}, "
                           f"胜率 {v5_result['win_rate']*100:.1f}%")
                logger.info(f"  📊 提升: {improvement*100:+.2f}%")
        
        # 生成报告
        self._generate_comparison_report(results)
        
        return results
    
    def _generate_comparison_report(self, results: Dict):
        """生成对比报告"""
        report = ["# 策略对比报告\n\n"]
        report.append(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        report.append("## 📊 性能对比\n\n")
        report.append("| 股票 | V4收益 | V5收益 | 提升 | V4夏普 | V5夏普 |\n")
        report.append("|------|--------|--------|------|--------|--------|\n")
        
        for code, result in results.items():
            v4_return = result['v4']['total_return'] * 100
            v5_return = result['v5']['total_return'] * 100 if result['v5'] else 0
            improvement = v5_return - v4_return
            v4_sharpe = result['v4']['sharpe']
            v5_sharpe = result['v5']['sharpe'] if result['v5'] else 0
            
            report.append(
                f"| {result['name']} | {v4_return:+.2f}% | {v5_return:+.2f}% | "
                f"{improvement:+.2f}% | {v4_sharpe:.3f} | {v5_sharpe:.3f} |\n"
            )
        
        report.append("\n## 💡 建议\n\n")
        report.append("- 使用V5优化配置替换V4\n")
        report.append("- 定期运行优化脚本更新参数\n")
        report.append("- 监控实际表现，必要时微调\n")
        
        # 保存报告
        report_path = Path(__file__).parent.parent / 'docs' / 'STRATEGY_COMPARISON.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logger.info(f"\n📄 对比报告已生成: {report_path}")


def main():
    """主函数"""
    validator = StrategyValidator()
    results = validator.compare_strategies()
    
    logger.info("\n✅ 策略对比完成!")


if __name__ == '__main__':
    main()
