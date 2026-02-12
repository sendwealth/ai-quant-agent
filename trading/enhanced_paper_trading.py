"""
完善的模拟交易系统
整合风控、监控、告警功能
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.indicators import sma, ema, rsi, macd, atr, adx
except ImportError:
    print("⚠️  指标模块未使用")


class RiskMonitor:
    """风险监控器"""

    def __init__(self,
                 daily_loss_limit: float = 0.05,
                 max_drawdown: float = 0.20,
                 volatility_threshold: float = 0.03):
        """
        初始化风险监控

        Args:
            daily_loss_limit: 日亏损限制
            max_drawdown: 最大回撤
            volatility_threshold: 波动率阈值
        """
        self.daily_loss_limit = daily_loss_limit
        self.max_drawdown = max_drawdown
        self.volatility_threshold = volatility_threshold

        self.start_capital = 0
        self.daily_start_capital = 0
        self.peak_equity = 0
        self.daily_trades = 0

    def reset(self, initial_capital: float):
        """重置监控"""
        self.start_capital = initial_capital
        self.daily_start_capital = initial_capital
        self.peak_equity = initial_capital
        self.daily_trades = 0

    def check_risk(self,
                   current_equity: float,
                   is_new_day: bool = False) -> Dict[str, bool]:
        """
        检查风险

        Args:
            current_equity: 当前权益
            is_new_day: 是否新的一天

        Returns:
            风险检查结果
        """
        results = {
            'daily_loss': True,
            'max_drawdown': True,
            'allowed': True
        }

        # 更新峰值
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        # 新的一天，重置日数据
        if is_new_day:
            self.daily_start_capital = current_equity
            self.daily_trades = 0

        # 检查日亏损
        daily_pnl = (current_equity - self.daily_start_capital) / self.daily_start_capital
        if daily_pnl < -self.daily_loss_limit:
            results['daily_loss'] = False
            results['allowed'] = False
            print(f"\n⚠️  风险触发: 日亏损超过限制 ({daily_pnl*100:.2f}% > {-self.daily_loss_limit*100:.2f}%)")

        # 检查最大回撤
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        if drawdown < -self.max_drawdown:
            results['max_drawdown'] = False
            results['allowed'] = False
            print(f"\n⚠️  风险触发: 回撤超过限制 ({drawdown*100:.2f}% < {-self.max_drawdown*100:.2f}%)")

        return results


class EnhancedPaperTrading:
    """增强的模拟交易系统"""

    def __init__(self,
                 initial_capital: float = 100000,
                 commission: float = 0.001,
                 slippage: float = 0.0001,
                 enable_risk_control: bool = True):
        """
        初始化模拟交易系统

        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点
            enable_risk_control: 启用风控
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.enable_risk_control = enable_risk_control

        # 账户状态
        self.cash = initial_capital
        self.position = 0.0
        self.entry_price = 0.0
        self.stop_loss = None
        self.take_profit = None

        # 监控
        self.risk_monitor = RiskMonitor() if enable_risk_control else None
        self.equity_curve: List[float] = [initial_capital]
        self.trades: List[Dict] = []
        self.equity_curve: List[float] = [initial_capital]
        self.drawdowns: List[float] = []

        # 统计
        self.total_pnl = 0.0
        self.win_trades = 0
        self.loss_trades = 0

        print(f"\n{'='*70}")
        print(f"增强模拟交易系统初始化")
        print(f"{'='*70}")
        print(f"初始资金: ¥{initial_capital:,.2f}" if initial_capital > 50000 else f"${initial_capital:,.2f}")
        print(f"手续费率: {commission*100:.3f}%")
        print(f"滑点: {slippage*100:.3f}%")
        print(f"风控系统: {'启用' if enable_risk_control else '禁用'}")
        print(f"{'='*70}\n")

        if self.risk_monitor:
            self.risk_monitor.reset(initial_capital)

    def execute_signal(self,
                      price: float,
                      signal: int,
                      date: str,
                      volatility: float = None,
                      atr: float = None) -> Optional[Dict]:
        """
        执行交易信号（含风控）

        Args:
            price: 当前价格
            signal: 信号 (1=买入, -1=卖出, 0=持有)
            date: 日期
            volatility: 波动率
            atr: ATR值

        Returns:
            交易记录
        """
        # 风险检查
        if self.risk_monitor:
            equity = self.cash + self.position * price
            risk_results = self.risk_monitor.check_risk(equity)

            if not risk_results['allowed']:
                print(f"\n⛔ 风控触发，暂停交易")
                return None

        # 检查止损止盈
        if self.position != 0 and self.stop_loss is not None:
            if self.position > 0 and price <= self.stop_loss:
                print(f"\n🛡️  触发止损: ¥{price:.2f}")
                return self._close_position(price, date, reason='stop_loss')
            elif self.position < 0 and price >= self.stop_loss:
                print(f"\n🛡️  触发止损: ¥{price:.2f}")
                return self._close_position(price, date, reason='stop_loss')

        if self.position != 0 and self.take_profit is not None:
            if self.position > 0 and price >= self.take_profit:
                print(f"\n🎯 触发止盈: ¥{price:.2f}")
                return self._close_position(price, date, reason='take_profit')
            elif self.position < 0 and price <= self.take_profit:
                print(f"\n🎯 触发止盈: ¥{price:.2f}")
                return self._close_position(price, date, reason='take_profit')

        # 执行信号
        trade = None

        if signal == 1 and self.position <= 0:
            # 先平空
            if self.position < 0:
                self._close_position(price, date, reason='signal')
            # 买入
            if self.cash > 0:
                trade = self._buy(price, date, volatility, atr)

        elif signal == -1 and self.position >= 0:
            # 先平多
            if self.position > 0:
                self._close_position(price, date, reason='signal')
            # 卖出
            if self.cash > 0:
                trade = self._sell(price, date, volatility, atr)

        elif signal == 0 and self.position != 0:
            # 平仓
            self._close_position(price, date, reason='signal')

        # 更新权益
        equity = self._update_equity(price)

        return trade

    def _buy(self, price: float, date: str, volatility: float = None, atr: float = None):
        """买入"""
        # 计算止损止盈
        if atr is not None:
            stop_loss_pct = (atr * 2) / price
            take_profit_pct = stop_loss_pct * 3
        elif volatility is not None:
            stop_loss_pct = volatility * 1.5
            take_profit_pct = stop_loss_pct * 2
        else:
            stop_loss_pct = 0.05
            take_profit_pct = 0.10

        self.stop_loss = price * (1 - stop_loss_pct)
        self.take_profit = price * (1 + take_profit_pct)

        # 买入
        execution_price = price * (1 + self.slippage)
        commission_amount = self.cash * self.commission
        available_cash = self.cash - commission_amount

        quantity = available_cash / execution_price

        if quantity > 0:
            total_cost = quantity * execution_price
            self.cash -= total_cost
            self.position += quantity
            self.entry_price = price

            trade = {
                'date': date,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'cost': total_cost,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
            self.trades.append(trade)

            print(f"✓ 买入: {quantity:.2f}股 @ ¥{price:.2f} "
                  f"(止损: ¥{self.stop_loss:.2f}, 止盈: ¥{self.take_profit:.2f})")

            return trade

        return None

    def _sell(self, price: float, date: str, volatility: float = None, atr: float = None):
        """卖出（做空）"""
        # 计算止损止盈
        if atr is not None:
            stop_loss_pct = (atr * 2) / price
            take_profit_pct = stop_loss_pct * 3
        elif volatility is not None:
            stop_loss_pct = volatility * 1.5
            take_profit_pct = stop_loss_pct * 2
        else:
            stop_loss_pct = 0.05
            take_profit_pct = 0.10

        self.stop_loss = price * (1 + stop_loss_pct)
        self.take_profit = price * (1 - take_profit_pct)

        # 卖出
        execution_price = price * (1 - self.slippage)
        commission_amount = self.cash * self.commission
        available_cash = self.cash - commission_amount

        quantity = available_cash / execution_price

        if quantity > 0:
            total_proceeds = quantity * execution_price
            self.cash += total_proceeds
            self.position -= quantity
            self.entry_price = price

            trade = {
                'date': date,
                'action': 'sell',
                'price': price,
                'quantity': quantity,
                'proceeds': total_proceeds,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            }
            self.trades.append(trade)

            print(f"✓ 卖出: {quantity:.2f}股 @ ¥{price:.2f} "
                  f"(止损: ¥{self.stop_loss:.2f}, 止盈: ¥{self.take_profit:.2f})")

            return trade

        return None

    def _close_position(self, price: float, date: str, reason: str = 'signal'):
        """平仓"""
        if self.position > 0:
            total_proceeds = self.position * price * (1 - self.commission)
            self.cash += total_proceeds

            pnl = (price - self.entry_price) * self.position - (self.position * price * self.commission)

            if pnl > 0:
                self.win_trades += 1
            else:
                self.loss_trades += 1

            self.total_pnl += pnl

            print(f"✓ 平多: {self.position:.2f}股 @ ¥{price:.2f} "
                  f"(盈亏: ¥{pnl:.2f}, 原因: {reason})")

            self.trades.append({
                'date': date,
                'action': 'close_long',
                'price': price,
                'quantity': self.position,
                'pnl': pnl,
                'reason': reason
            })

            self.position = 0
            self.entry_price = 0
            self.stop_loss = None
            self.take_profit = None

        elif self.position < 0:
            quantity = abs(self.position)
            total_cost = quantity * price * (1 + self.commission)
            self.cash -= total_cost

            pnl = (self.entry_price - price) * quantity - (quantity * price * self.commission)

            if pnl > 0:
                self.win_trades += 1
            else:
                self.loss_trades += 1

            self.total_pnl += pnl

            print(f"✓ 平空: {quantity:.2f}股 @ ¥{price:.2f} "
                  f"(盈亏: ¥{pnl:.2f}, 原因: {reason})")

            self.trades.append({
                'date': date,
                'action': 'close_short',
                'price': price,
                'quantity': quantity,
                'pnl': pnl,
                'reason': reason
            })

            self.position = 0
            self.entry_price = 0
            self.stop_loss = None
            self.take_profit = None

    def _update_equity(self, price: float) -> float:
        """更新权益"""
        equity = self.cash + self.position * price
        self.equity_curve.append(equity)

        # 计算回撤
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            drawdown = (equity - peak) / peak if peak > 0 else 0
            self.drawdowns.append(drawdown)

        return equity

    def get_performance_metrics(self) -> Dict[str, float]:
        """获取性能指标"""
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna()

        total_return = (self.equity_curve[-1] - self.initial_capital) / self.initial_capital

        days = len(self.equity_curve)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0

        max_drawdown = min(self.drawdowns) if self.drawdowns else 0

        win_rate = self.win_trades / (self.win_trades + self.loss_trades) if (self.win_trades + self.loss_trades) > 0 else 0

        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.equity_curve[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': win_rate,
            'win_trades': self.win_trades,
            'loss_trades': self.loss_trades,
            'total_pnl': self.total_pnl
        }

    def print_report(self):
        """打印交易报告"""
        metrics = self.get_performance_metrics()

        print(f"\n{'='*70}")
        print(f"增强模拟交易报告")
        print(f"{'='*70}")

        print(f"\n【资金情况】")
        currency = '¥' if metrics['initial_capital'] > 50000 else '$'
        print(f"初始资金: {currency}{metrics['initial_capital']:,.2f}")
        print(f"最终资金: {currency}{metrics['final_capital']:,.2f}")
        print(f"总盈亏: {currency}{metrics['final_capital'] - metrics['initial_capital']:,.2f}")
        print(f"总收益: {metrics['total_return']*100:+.2f}%")

        print(f"\n【收益指标】")
        print(f"年化收益: {metrics['annual_return']*100:+.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")

        print(f"\n【风险指标】")
        print(f"波动率: {metrics['volatility']*100:.2f}%")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")

        print(f"\n【交易统计】")
        print(f"交易次数: {metrics['num_trades']}")
        print(f"盈利次数: {metrics['win_trades']}")
        print(f"亏损次数: {metrics['loss_trades']}")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        print(f"总盈亏: {currency}{metrics['total_pnl']:,.2f}")

        print("\n" + "="*70)


if __name__ == "__main__":
    # 测试增强模拟交易系统
    print("增强模拟交易系统测试")
