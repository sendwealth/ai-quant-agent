#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统优化器
System Optimizer

自动优化策略参数、生成优化报告、更新配置
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger

# 添加父目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.indicators import sma, ema, atr, rsi, macd


class SystemOptimizer:
    """系统优化器"""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent / 'data'
        self.config_dir = Path(__file__).parent.parent / 'config'
        self.results = {}
    
    def load_stock_data(self, code: str) -> pd.DataFrame:
        """加载股票数据"""
        filepath = self.data_dir / f'real_{code}.csv'
        
        if not filepath.exists():
            logger.error(f"数据文件不存在: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        
        # 标准化列名
        if 'datetime' not in df.columns and 'trade_date' in df.columns:
            df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
        
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    
    def grid_search_parameters(self, 
                               data: pd.DataFrame,
                               code: str) -> Dict:
        """
        网格搜索最优参数
        
        Args:
            data: 股票数据
            code: 股票代码
        
        Returns:
            最优参数
        """
        logger.info(f"\n🔍 开始参数优化: {code}")
        
        # 参数范围
        ma_fast_range = [5, 7, 10, 12, 15]
        ma_slow_range = [20, 25, 30, 35, 40]
        atr_stop_range = [1.5, 2.0, 2.5, 3.0]
        
        best_params = None
        best_sharpe = -999
        best_return = -999
        
        total_combinations = len(ma_fast_range) * len(ma_slow_range) * len(atr_stop_range)
        current = 0
        
        for ma_fast in ma_fast_range:
            for ma_slow in ma_slow_range:
                if ma_fast >= ma_slow:
                    continue
                
                for atr_stop in atr_stop_range:
                    current += 1
                    
                    # 计算指标
                    df = data.copy()
                    df['ma_fast'] = sma(df['close'], ma_fast)
                    df['ma_slow'] = sma(df['close'], ma_slow)
                    df['atr'] = atr(df['high'], df['low'], df['close'], 14)
                    df['macd'], df['macd_signal'], _ = macd(df['close'])
                    
                    # 回测
                    result = self._backtest(df, atr_stop)
                    
                    # 更新最优
                    if result['sharpe'] > best_sharpe:
                        best_sharpe = result['sharpe']
                        best_return = result['return']
                        best_params = {
                            'ma_fast': ma_fast,
                            'ma_slow': ma_slow,
                            'atr_stop': atr_stop
                        }
        
        logger.info(f"  ✅ 最优参数: MA{best_params['ma_fast']}/{best_params['ma_slow']}, ATR {best_params['atr_stop']}x")
        logger.info(f"  📊 夏普: {best_sharpe:.3f}, 收益: {best_return*100:+.2f}%")
        
        return {
            'code': code,
            'params': best_params,
            'sharpe': best_sharpe,
            'return': best_return
        }
    
    def _backtest(self, df: pd.DataFrame, atr_stop: float) -> Dict:
        """简单回测"""
        # 初始资金
        capital = 100000
        position = 0
        entry_price = 0
        
        returns = []
        
        for i in range(50, len(df)):  # 从50开始，确保指标稳定
            row = df.iloc[i]
            
            # 买入信号
            if position == 0:
                buy_signal = (
                    row['ma_fast'] > row['ma_slow'] and
                    row['macd'] > row['macd_signal']
                )
                
                if buy_signal:
                    position = capital / row['close']
                    entry_price = row['close']
                    capital = 0
            
            # 卖出信号
            elif position > 0:
                # 止损
                stop_price = entry_price * (1 - atr_stop * row['atr'] / entry_price)
                
                sell_signal = (
                    row['ma_fast'] < row['ma_slow'] or
                    row['close'] < stop_price
                )
                
                if sell_signal:
                    capital = position * row['close']
                    returns.append((capital / 100000 - 1))
                    position = 0
                    entry_price = 0
        
        # 计算绩效
        if len(returns) == 0:
            return {'sharpe': -999, 'return': 0}
        
        total_return = sum(returns)
        avg_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 0.001
        
        sharpe = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
        
        return {
            'sharpe': sharpe,
            'return': total_return
        }
    
    def optimize_all_stocks(self, stocks: List[Dict]) -> Dict:
        """
        优化所有股票参数
        
        Args:
            stocks: 股票列表
        
        Returns:
            优化结果
        """
        logger.info("\n" + "="*50)
        logger.info("🚀 开始系统优化")
        logger.info("="*50)
        
        results = {}
        
        for stock in stocks:
            code = stock['code']
            
            # 加载数据
            data = self.load_stock_data(code)
            if data is None:
                continue
            
            # 优化参数
            result = self.grid_search_parameters(data, code)
            results[code] = result
        
        # 保存结果
        self._save_results(results)
        
        # 生成报告
        self._generate_report(results, stocks)
        
        return results
    
    def _save_results(self, results: Dict):
        """保存优化结果"""
        output_path = self.data_dir / 'optimization_results.json'
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        logger.info(f"\n✅ 结果已保存: {output_path}")
    
    def _generate_report(self, results: Dict, stocks: List[Dict]):
        """生成优化报告"""
        report = ["# 系统优化报告\n"]
        report.append(f"**优化时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        report.append("## 📊 优化结果\n\n")
        report.append("| 股票 | MA快/慢 | ATR止损 | 夏普 | 预期收益 |\n")
        report.append("|------|---------|---------|------|----------|\n")
        
        for stock in stocks:
            code = stock['code']
            if code not in results:
                continue
            
            result = results[code]
            params = result['params']
            
            report.append(
                f"| {stock['name']} | {params['ma_fast']}/{params['ma_slow']} | "
                f"{params['atr_stop']}x | {result['sharpe']:.3f} | "
                f"{result['return']*100:+.2f}% |\n"
            )
        
        report.append("\n## 🔧 建议更新\n\n")
        report.append("将以上参数更新到 `config/strategy_v4.yaml`\n\n")
        
        # 保存报告
        report_path = Path(__file__).parent.parent / 'docs' / 'OPTIMIZATION_REPORT.md'
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        logger.info(f"📄 报告已生成: {report_path}")


def main():
    """主函数"""
    # 股票列表 (从当前配置)
    stocks = [
        {'code': '300750', 'name': '宁德时代'},
        {'code': '002475', 'name': '立讯精密'},
        {'code': '601318', 'name': '中国平安'},
        {'code': '600276', 'name': '恒瑞医药'}
    ]
    
    # 创建优化器
    optimizer = SystemOptimizer()
    
    # 运行优化
    results = optimizer.optimize_all_stocks(stocks)
    
    logger.info("\n✅ 优化完成!")


if __name__ == '__main__':
    main()
