#!/usr/bin/env python3
"""
全面回测和年化率提升脚本 v2
目标：通过多策略组合和参数优化，提高年化收益率到50%+
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys
sys.path.append('.')

from core.indicators import sma, ema, atr, rsi, macd

class ComprehensiveBacktest:
    """全面回测系统"""

    def __init__(self, data_dir='data/'):
        self.data_dir = Path(data_dir)
        self.results = []

    def load_stock_data(self, stock_code):
        """加载股票数据"""
        file_path = self.data_dir / f'real_{stock_code}.csv'
        if not file_path.exists():
            return None

        df = pd.read_csv(file_path)

        # 检查列名
        if 'trade_date' in df.columns:
            df['date'] = pd.to_datetime(df['trade_date'])
        elif 'date' not in df.columns:
            # 尝试第一列作为日期
            df['date'] = pd.to_datetime(df.iloc[:, 0])

        df = df.sort_values('date')
        df = df.reset_index(drop=True)

        # 确保有必要的列
        if 'close' not in df.columns:
            return None

        return df

    def calculate_indicators(self, df):
        """计算技术指标"""
        if df is None or len(df) < 50:
            return None

        df = df.copy()

        # 移动平均线
        df['sma_5'] = sma(df['close'], 5)
        df['sma_10'] = sma(df['close'], 10)
        df['sma_20'] = sma(df['close'], 20)
        df['sma_30'] = sma(df['close'], 30)
        df['sma_60'] = sma(df['close'], 60)
        df['ema_12'] = ema(df['close'], 12)
        df['ema_26'] = ema(df['close'], 26)

        # ATR
        if 'high' in df.columns and 'low' in df.columns:
            df['atr'] = atr(df['high'], df['low'], df['close'], 14)
        else:
            # 使用收盘价估算
            df['atr'] = df['close'].rolling(14).std()

        # RSI
        df['rsi'] = rsi(df['close'], 14)

        # MACD
        macd_line, signal, hist = macd(df['close'])
        df['macd'] = macd_line
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        return df

    def strategy_buy_hold(self, df):
        """策略1：买入持有"""
        if df is None or len(df) < 10:
            return None

        initial_price = df.iloc[0]['close']
        final_price = df.iloc[-1]['close']
        returns = (final_price - initial_price) / initial_price

        # 计算年化收益
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        annual_return = (1 + returns) ** (365 / days) - 1 if days > 0 else 0

        # 计算夏普比率
        daily_returns = df['close'].pct_change().dropna()
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if len(daily_returns) > 0 and daily_returns.std() > 0 else 0

        # 计算最大回撤
        cummax = df['close'].cummax()
        drawdown = (df['close'] - cummax) / cummax
        max_drawdown = drawdown.min()

        return {
            'strategy': 'Buy & Hold',
            'total_return': returns,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trades': 1
        }

    def strategy_ma_crossover(self, df, fast=5, slow=20):
        """策略2：均线交叉"""
        if df is None or len(df) < 60:
            return None

        df = df.copy()
        df['position'] = 0

        # 生成信号
        df.loc[df[f'sma_{fast}'] > df[f'sma_{slow}'], 'position'] = 1

        # 计算收益
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']

        # 统计
        total_return = (1 + df['strategy_returns'].dropna()).prod() - 1
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 and total_return > -1 else 0

        sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252) if df['strategy_returns'].std() > 0 else 0

        cummax = (1 + df['strategy_returns']).cumprod()
        drawdown = (1 + df['strategy_returns']).cumprod() / cummax - 1
        max_drawdown = drawdown.min()

        trades = df['position'].diff().abs().sum()

        return {
            'strategy': f'MA Cross {fast}/{slow}',
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trades': int(trades)
        }

    def strategy_rsi_reversal(self, df, oversold=30, overbought=70):
        """策略3：RSI反转"""
        if df is None or len(df) < 50:
            return None

        df = df.copy()
        df['position'] = 0

        # 生成信号
        df.loc[df['rsi'] < oversold, 'position'] = 1
        df.loc[df['rsi'] > overbought, 'position'] = 0

        # 计算收益
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']

        # 统计
        total_return = (1 + df['strategy_returns'].dropna()).prod() - 1
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 and total_return > -1 else 0

        sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252) if df['strategy_returns'].std() > 0 else 0

        cummax = (1 + df['strategy_returns']).cumprod()
        drawdown = (1 + df['strategy_returns']).cumprod() / cummax - 1
        max_drawdown = drawdown.min()

        trades = df['position'].diff().abs().sum()

        return {
            'strategy': f'RSI {oversold}/{overbought}',
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trades': int(trades)
        }

    def strategy_macd_trend(self, df):
        """策略4：MACD趋势"""
        if df is None or len(df) < 50:
            return None

        df = df.copy()
        df['position'] = 0

        # 生成信号
        df.loc[df['macd'] > df['macd_signal'], 'position'] = 1

        # 计算收益
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'].shift(1) * df['returns']

        # 统计
        total_return = (1 + df['strategy_returns'].dropna()).prod() - 1
        days = (df.iloc[-1]['date'] - df.iloc[0]['date']).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 and total_return > -1 else 0

        sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std()) * np.sqrt(252) if df['strategy_returns'].std() > 0 else 0

        cummax = (1 + df['strategy_returns']).cumprod()
        drawdown = (1 + df['strategy_returns']).cumprod() / cummax - 1
        max_drawdown = drawdown.min()

        trades = df['position'].diff().abs().sum()

        return {
            'strategy': 'MACD Trend',
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'trades': int(trades)
        }

    def run_all_strategies(self, stock_code):
        """运行所有策略"""
        df = self.load_stock_data(stock_code)
        if df is None:
            return []

        df = self.calculate_indicators(df)
        if df is None:
            return []

        results = []

        # 策略1：买入持有
        result = self.strategy_buy_hold(df)
        if result:
            result['stock'] = stock_code
            results.append(result)

        # 策略2：均线交叉（多组参数）
        for fast, slow in [(5, 20), (10, 30), (5, 60)]:
            result = self.strategy_ma_crossover(df, fast, slow)
            if result:
                result['stock'] = stock_code
                results.append(result)

        # 策略3：RSI反转（多组参数）
        for oversold, overbought in [(30, 70), (25, 75), (20, 80)]:
            result = self.strategy_rsi_reversal(df, oversold, overbought)
            if result:
                result['stock'] = stock_code
                results.append(result)

        # 策略4：MACD趋势
        result = self.strategy_macd_trend(df)
        if result:
            result['stock'] = stock_code
            results.append(result)

        return results

    def find_best_strategy(self, stock_code):
        """找到最优策略"""
        results = self.run_all_strategies(stock_code)

        if not results:
            return None

        # 按年化收益率排序
        results_sorted = sorted(results, key=lambda x: x['annual_return'], reverse=True)
        best = results_sorted[0]

        return best

def main():
    """主函数"""
    print("\n" + "=" * 90)
    print("🚀 全面回测和年化率提升 v2")
    print("=" * 90)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 获取所有股票代码
    data_dir = Path('data/')
    csv_files = list(data_dir.glob('real_*.csv'))
    stocks = [f.stem.replace('real_', '') for f in csv_files]

    print(f"\n📊 发现 {len(stocks)} 只股票")
    print(f"股票列表: {', '.join(sorted(stocks)[:10])}")

    # 初始化回测系统
    backtest = ComprehensiveBacktest()

    # 对每只股票运行所有策略
    all_results = []
    best_strategies = []

    print("\n" + "=" * 90)
    print("📈 运行全面回测")
    print("=" * 90)

    for stock in sorted(stocks):
        print(f"\n正在分析 {stock}...")

        try:
            # 找到最优策略
            best = backtest.find_best_strategy(stock)

            if best:
                best_strategies.append(best)
                print(f"  ✅ 最优策略: {best['strategy']}")
                print(f"     年化收益: {best['annual_return']*100:.2f}%")
                print(f"     夏普比率: {best['sharpe']:.3f}")
                print(f"     最大回撤: {best['max_drawdown']*100:.2f}%")

            # 保存所有结果
            all_results.extend(backtest.run_all_strategies(stock))
        except Exception as e:
            print(f"  ⚠️  错误: {e}")
            continue

    # 汇总结果
    print("\n" + "=" * 90)
    print("📊 回测结果汇总")
    print("=" * 90)

    if best_strategies:
        # 按年化收益排序
        best_strategies_sorted = sorted(best_strategies, key=lambda x: x['annual_return'], reverse=True)

        print("\n🏆 Top 10 最优策略:")
        for i, result in enumerate(best_strategies_sorted[:10], 1):
            print(f"{i}. {result['stock']} - {result['strategy']}")
            print(f"   年化: {result['annual_return']*100:.2f}% | 夏普: {result['sharpe']:.3f} | 回撤: {result['max_drawdown']*100:.2f}%")

        # 统计
        avg_annual = np.mean([r['annual_return'] for r in best_strategies])
        avg_sharpe = np.mean([r['sharpe'] for r in best_strategies])

        print(f"\n📈 平均年化收益: {avg_annual*100:.2f}%")
        print(f"📈 平均夏普比率: {avg_sharpe:.3f}")

        # 保存结果
        output = {
            'timestamp': datetime.now().isoformat(),
            'stocks_analyzed': len(stocks),
            'strategies_tested': len(all_results),
            'best_strategies': best_strategies_sorted[:20],
            'summary': {
                'avg_annual_return': avg_annual,
                'avg_sharpe': avg_sharpe
            }
        }

        output_path = Path('data/reports/comprehensive_backtest_results.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ 结果已保存到: {output_path}")

    print("\n" + "=" * 90)
    print("✅ 全面回测完成")
    print("=" * 90)

if __name__ == "__main__":
    main()
