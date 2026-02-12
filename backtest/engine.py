"""
回测引擎
验证交易策略的盈利能力
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from datetime import datetime
from loguru import logger

from utils.indicators import *


class BacktestEngine:
    """回测引擎 - 验证策略盈利能力"""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0001):
        """
        初始化回测引擎

        Args:
            initial_capital: 初始资金
            commission: 手续费率（如0.001 = 0.1%）
            slippage: 滑点（如0.0001 = 0.01%）
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # 回测状态
        self.current_capital = initial_capital
        self.position = 0.0  # 持仓数量（正数多头，负数空头）
        self.cash = initial_capital

        # 交易记录
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = []
        self.drawdowns: List[float] = []

        logger.info(f"回测引擎初始化 - 初始资金: ${initial_capital:,.2f}, "
                   f"手续费: {commission*100:.3f}%, 滑点: {slippage*100:.3f}%")

    def run(self,
            df: pd.DataFrame,
            strategy_func: Callable,
            signal_col: str = 'signal') -> Dict[str, float]:
        """
        运行回测

        Args:
            df: 历史价格数据 (必须包含open, high, low, close, volume)
            strategy_func: 策略函数，接收df，返回信号series
            signal_col: 信号列名

        Returns:
            回测结果字典
        """
        logger.info("开始回测...")

        # 重置状态
        self._reset()

        # 生成信号
        logger.info("生成交易信号...")
        signals = strategy_func(df)

        # 确保信号对齐
        if len(signals) != len(df):
            signals = signals.reindex(df.index).fillna(0)

        df = df.copy()
        df['signal'] = signals

        # 逐日回测
        logger.info(f"回测 {len(df)} 个交易日...")
        for i in range(len(df)):
            row = df.iloc[i]

            # 更新权益曲线
            self._update_equity(row)

            # 执行交易
            self._execute_trade(row)

        # 计算最终结果
        results = self._calculate_results(df)

        logger.info(f"回测完成 - 最终资金: ${self.current_capital:,.2f}, "
                   f"总收益: {results['total_return']*100:.2f}%")

        return results

    def _reset(self):
        """重置回测状态"""
        self.current_capital = self.initial_capital
        self.position = 0.0
        self.cash = self.initial_capital
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []

    def _update_equity(self, row: pd.Series):
        """更新权益曲线"""
        current_price = row['close']
        equity = self.cash + self.position * current_price

        self.equity_curve.append(equity)
        self.current_capital = equity

        # 计算回撤
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            drawdown = (equity - peak) / peak if peak > 0 else 0
            self.drawdowns.append(drawdown)

    def _execute_trade(self, row: pd.Series):
        """
        执行交易

        信号: 1=买入, -1=卖出, 0=持有
        """
        signal = row['signal']
        current_price = row['close']

        # 计算成交价格（考虑滑点）
        execution_price = self._calculate_execution_price(current_price, signal)

        # 买入
        if signal == 1 and self.position <= 0:
            self._buy(execution_price, row.name)

        # 卖出
        elif signal == -1 and self.position >= 0:
            self._sell(execution_price, row.name)

        # 平仓
        elif signal == 0 and self.position != 0:
            self._close_position(execution_price, row.name)

    def _calculate_execution_price(self, current_price: float, signal: int) -> float:
        """计算成交价格（考虑滑点）"""
        if signal == 1:  # 买入，价格向上滑
            return current_price * (1 + self.slippage)
        elif signal == -1:  # 卖出，价格向下滑
            return current_price * (1 - self.slippage)
        else:
            return current_price

    def _buy(self, price: float, date):
        """买入"""
        # 计算可买数量（考虑手续费）
        max_quantity = self.cash / price
        commission_amount = max_quantity * price * self.commission

        # 可用数量
        quantity = (self.cash - commission_amount) / price

        if quantity > 0:
            total_cost = quantity * price + commission_amount
            self.cash -= total_cost
            self.position += quantity

            self.trades.append({
                'date': date,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'cost': total_cost,
                'commission': commission_amount
            })

    def _sell(self, price: float, date):
        """卖出（做空）"""
        # 简单实现：先平仓再做空
        if self.position > 0:
            self._close_position(price, date)

        # 做空数量
        quantity = self.cash / price
        commission_amount = quantity * price * self.commission

        total_proceeds = quantity * price - commission_amount
        self.cash += total_proceeds
        self.position -= quantity

        self.trades.append({
            'date': date,
            'action': 'sell',
            'price': price,
            'quantity': quantity,
            'proceeds': total_proceeds,
            'commission': commission_amount
        })

    def _close_position(self, price: float, date):
        """平仓"""
        if self.position > 0:
            # 平多
            total_proceeds = self.position * price
            commission = total_proceeds * self.commission
            net_proceeds = total_proceeds - commission

            self.cash += net_proceeds
            quantity = self.position
            self.position = 0

            self.trades.append({
                'date': date,
                'action': 'close_long',
                'price': price,
                'quantity': quantity,
                'proceeds': net_proceeds,
                'commission': commission
            })

        elif self.position < 0:
            # 平空
            quantity = abs(self.position)
            total_cost = quantity * price
            commission = total_cost * self.commission
            net_cost = total_cost + commission

            self.cash -= net_cost
            self.position = 0

            self.trades.append({
                'date': date,
                'action': 'close_short',
                'price': price,
                'quantity': quantity,
                'cost': net_cost,
                'commission': commission
            })

    def _calculate_results(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算回测结果"""
        # 基础收益
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        # 年化收益率
        days = len(df)
        years = days / 252  # 假设一年252个交易日
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 每日收益率
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna()

        # 波动率
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

        # 夏普比率（假设无风险利率2%）
        risk_free_rate = 0.02
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0

        # 最大回撤
        max_drawdown = min(self.drawdowns) if self.drawdowns else 0

        # 交易统计
        num_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.get('action') in ['close_long', 'close_short']])

        # 盈亏比
        if winning_trades > 0:
            win_rate = 1.0  # 简化计算
        else:
            win_rate = 0.0

        # 基准收益（买入持有）
        buy_hold_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.current_capital,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'buy_hold_return': buy_hold_return,
            'excess_return': total_return - buy_hold_return
        }

    def print_report(self, results: Dict[str, float]):
        """打印回测报告"""
        print("\n" + "="*60)
        print("回测报告")
        print("="*60)

        print(f"\n【资金情况】")
        print(f"初始资金: ${results['initial_capital']:,.2f}")
        print(f"最终资金: ${results['final_capital']:,.2f}")
        print(f"总收益: {results['total_return']*100:+.2f}%")

        print(f"\n【收益指标】")
        print(f"年化收益: {results['annual_return']*100:+.2f}%")
        print(f"超额收益: {results['excess_return']*100:+.2f}% (vs 买入持有)")
        print(f"买入持有收益: {results['buy_hold_return']*100:+.2f}%")

        print(f"\n【风险指标】")
        print(f"波动率: {results['volatility']*100:.2f}%")
        print(f"最大回撤: {results['max_drawdown']*100:.2f}%")

        print(f"\n【性能指标】")
        print(f"夏普比率: {results['sharpe_ratio']:.2f}")

        print(f"\n【交易统计】")
        print(f"交易次数: {results['num_trades']}")

        print("\n" + "="*60)


# 预定义策略函数
def sma_crossover_strategy(df: pd.DataFrame,
                            short_period: int = 20,
                            long_period: int = 60) -> pd.Series:
    """
    均线交叉策略

    Args:
        df: 价格数据
        short_period: 短期均线周期
        long_period: 长期均线周期

    Returns:
        信号series
    """
    # 计算均线
    short_ma = sma(df['close'], short_period)
    long_ma = sma(df['close'], long_period)

    # 生成信号
    # 金叉（短期上穿长期）= 买入
    # 死叉（短期下穿长期）= 卖出
    signals = pd.Series(0, index=df.index)

    # 上穿
    signals[(short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))] = 1

    # 下穿
    signals[(short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))] = -1

    return signals


def rsi_strategy(df: pd.DataFrame,
                  oversold: int = 30,
                  overbought: int = 70) -> pd.Series:
    """
    RSI策略

    Args:
        df: 价格数据
        oversold: 超卖阈值
        overbought: 超买阈值

    Returns:
        信号series
    """
    rsi_values = rsi(df['close'], 14)

    signals = pd.Series(0, index=df.index)

    # RSI从超卖区间反弹 = 买入
    signals[(rsi_values > oversold) & (rsi_values.shift(1) <= oversold)] = 1

    # RSI从超买区间回落 = 卖出
    signals[(rsi_values < overbought) & (rsi_values.shift(1) >= overbought)] = -1

    return signals


def macd_strategy(df: pd.DataFrame) -> pd.Series:
    """
    MACD策略

    Args:
        df: 价格数据

    Returns:
        信号series
    """
    macd_line, signal_line, histogram = macd(df['close'])

    signals = pd.Series(0, index=df.index)

    # MACD柱状图从负转正 = 买入
    signals[(histogram > 0) & (histogram.shift(1) <= 0)] = 1

    # MACD柱状图从正转负 = 卖出
    signals[(histogram < 0) & (histogram.shift(1) >= 0)] = -1

    return signals


if __name__ == "__main__":
    # 测试回测引擎
    from data.fetcher import DataFetcher

    print("="*60)
    print("测试回测引擎")
    print("="*60)

    # 获取数据
    fetcher = DataFetcher()
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

    print(f"\n获取SPY数据 ({start_date} -> {end_date})...")
    df = fetcher.fetch_stock_data('SPY', start_date, end_date)

    # 测试均线交叉策略
    print("\n" + "="*60)
    print("策略1: 均线交叉 (20日 / 60日)")
    print("="*60)

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(df, sma_crossover_strategy)
    engine.print_report(results)

    # 测试RSI策略
    print("\n" + "="*60)
    print("策略2: RSI策略 (30 / 70)")
    print("="*60)

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(df, rsi_strategy)
    engine.print_report(results)

    # 测试MACD策略
    print("\n" + "="*60)
    print("策略3: MACD策略")
    print("="*60)

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run(df, macd_strategy)
    engine.print_report(results)
