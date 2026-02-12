"""
模拟交易系统
使用历史数据验证策略的实时表现
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class PaperTradingSystem:
    """模拟交易系统"""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0001):
        """
        初始化模拟交易系统

        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

        # 账户状态
        self.cash = initial_capital
        self.position = 0.0  # 持仓数量
        self.equity_curve: List[float] = [initial_capital]

        # 交易记录
        self.trades: List[Dict] = []

        print(f"\n{'='*70}")
        print(f"模拟交易系统初始化")
        print(f"{'='*70}")
        print(f"初始资金: ${initial_capital:,.2f}")
        print(f"手续费率: {commission*100:.3f}%")
        print(f"滑点: {slippage*100:.3f}%")
        print(f"{'='*70}\n")

    def execute_signal(self,
                      price: float,
                      signal: int,
                      date: str) -> Optional[Dict]:
        """
        执行交易信号

        Args:
            price: 当前价格
            signal: 信号 (1=买入, -1=卖出, 0=持有)
            date: 日期

        Returns:
            交易记录
        """
        # 计算成交价格（考虑滑点）
        if signal == 1:
            execution_price = price * (1 + self.slippage)
        elif signal == -1:
            execution_price = price * (1 - self.slippage)
        else:
            return None

        trade = None

        # 买入信号
        if signal == 1 and self.position <= 0:
            # 先平空
            if self.position < 0:
                self._close_short(execution_price, date)

            # 买入
            if self.cash > 0:
                trade = self._buy(execution_price, date)

        # 卖出信号
        elif signal == -1 and self.position >= 0:
            # 先平多
            if self.position > 0:
                self._close_long(execution_price, date)

            # 卖出（做空）
            if self.cash > 0:
                trade = self._sell(execution_price, date)

        # 持有信号，但有持仓则平仓
        elif signal == 0 and self.position != 0:
            if self.position > 0:
                self._close_long(execution_price, date)
            elif self.position < 0:
                self._close_short(execution_price, date)

        return trade

    def _buy(self, price: float, date: str) -> Dict:
        """买入"""
        # 计算可买数量
        commission_amount = self.cash * self.commission
        available_cash = self.cash - commission_amount

        quantity = available_cash / price

        if quantity > 0:
            total_cost = quantity * price + commission_amount
            self.cash -= total_cost
            self.position += quantity

            trade = {
                'date': date,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'cost': total_cost
            }
            self.trades.append(trade)
            print(f"✓ 买入: {quantity:.2f}股 @ ${price:.2f} (成本: ${total_cost:.2f})")
            return trade

        return None

    def _sell(self, price: float, date: str) -> Dict:
        """卖出（做空）"""
        # 计算可卖数量
        available_cash = self.cash / (1 + self.commission)

        quantity = available_cash / price
        commission_amount = quantity * price * self.commission

        if quantity > 0:
            total_proceeds = quantity * price - commission_amount
            self.cash += total_proceeds
            self.position -= quantity

            trade = {
                'date': date,
                'action': 'sell',
                'price': price,
                'quantity': quantity,
                'proceeds': total_proceeds
            }
            self.trades.append(trade)
            print(f"✓ 卖出: {quantity:.2f}股 @ ${price:.2f} (收入: ${total_proceeds:.2f})")
            return trade

        return None

    def _close_long(self, price: float, date: str):
        """平多头"""
        if self.position > 0:
            total_proceeds = self.position * price
            commission = total_proceeds * self.commission
            net_proceeds = total_proceeds - commission

            self.cash += net_proceeds

            print(f"✓ 平多: {self.position:.2f}股 @ ${price:.2f} (收入: ${net_proceeds:.2f})")

            self.trades.append({
                'date': date,
                'action': 'close_long',
                'price': price,
                'quantity': self.position,
                'proceeds': net_proceeds,
                'pnl': net_proceeds - (self.position * price * (1 - self.commission))
            })

            self.position = 0

    def _close_short(self, price: float, date: str):
        """平空头"""
        if self.position < 0:
            quantity = abs(self.position)
            total_cost = quantity * price
            commission = total_cost * self.commission
            net_cost = total_cost + commission

            self.cash -= net_cost

            print(f"✓ 平空: {quantity:.2f}股 @ ${price:.2f} (成本: ${net_cost:.2f})")

            self.trades.append({
                'date': date,
                'action': 'close_short',
                'price': price,
                'quantity': quantity,
                'cost': net_cost
            })

            self.position = 0

    def update_equity(self, price: float):
        """更新权益"""
        equity = self.cash + self.position * price
        self.equity_curve.append(equity)
        return equity

    def get_performance_metrics(self) -> Dict[str, float]:
        """计算性能指标"""
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna()

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        days = len(self.equity_curve)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        peak = equity_series.expanding().max()
        drawdowns = (equity_series - peak) / peak
        max_drawdown = drawdowns.min()

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades)
        }

    def print_report(self):
        """打印交易报告"""
        metrics = self.get_performance_metrics()

        print(f"\n{'='*70}")
        print(f"模拟交易报告")
        print(f"{'='*70}")

        print(f"\n【资金情况】")
        print(f"初始资金: ${metrics['initial_capital']:,.2f}")
        print(f"最终资金: ${metrics['final_capital']:,.2f}")
        print(f"总盈亏: ${metrics['final_capital'] - metrics['initial_capital']:,.2f}")
        print(f"总收益: {metrics['total_return']*100:+.2f}%")

        print(f"\n【收益指标】")
        print(f"年化收益: {metrics['annual_return']*100:+.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")

        print(f"\n【风险指标】")
        print(f"波动率: {metrics['volatility']*100:.2f}%")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")

        print(f"\n【交易统计】")
        print(f"交易次数: {metrics['num_trades']}")

        print(f"\n{'='*70}")

        return metrics


def run_paper_test(strategy_name: str,
                   df: pd.DataFrame,
                   signal_generator,
                   initial_capital: float = 100000) -> Dict[str, float]:
    """
    运行模拟交易

    Args:
        strategy_name: 策略名称
        df: 历史数据
        signal_generator: 信号生成函数
        initial_capital: 初始资金

    Returns:
        性能指标
    """
    print(f"\n{'#'*70}")
    print(f"# {strategy_name}")
    print(f"{'#'*70}\n")

    # 初始化模拟系统
    system = PaperTradingSystem(initial_capital=initial_capital)

    # 生成信号
    print("生成交易信号...")
    signals = signal_generator(df)

    print(f"\n开始模拟交易 ({len(df)} 个交易日)...\n")

    # 逐日交易
    for i, (idx, row) in enumerate(df.iterrows()):
        if pd.isna(signals.iloc[i]):
            continue

        price = row['close']
        date = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
        signal = int(signals.iloc[i])

        # 执行信号
        trade = system.execute_signal(price, signal, date)

        # 更新权益
        equity = system.update_equity(price)

        # 每50天打印一次状态
        if (i + 1) % 50 == 0:
            print(f"第 {i+1} 天: 权益 ${equity:,.2f}, 持仓 {system.position:.2f}")

    # 最终平仓
    if system.position != 0:
        final_price = df['close'].iloc[-1]
        final_date = df.index[-1]
        if hasattr(final_date, 'strftime'):
            final_date = final_date.strftime('%Y-%m-%d')
        system.execute_signal(final_price, 0, final_date)

    # 打印报告
    metrics = system.print_report()

    return metrics


if __name__ == "__main__":
    # 简单测试
    print("模拟交易系统测试")
