"""
风控智能体
动态风险管理、止损止盈、仓位管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from loguru import logger

from utils.config import get_config


class RiskAgent:
    """风控智能体 - 风险管理和仓位控制"""

    def __init__(self):
        """初始化风控智能体"""
        self.config = get_config()
        self.daily_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_equity = 0.0

    def calculate_position_size(self,
                                  signal_direction: str,
                                  signal_strength: str,
                                  current_price: float,
                                  account_value: float,
                                  volatility: float = None) -> float:
        """
        计算建议仓位大小

        Args:
            signal_direction: 信号方向 (long/short/neutral)
            signal_strength: 信号强度 (strong/medium/weak)
            current_price: 当前价格
            account_value: 账户价值
            volatility: 波动率（可选，用于调整仓位）

        Returns:
            建议持仓数量
        """
        # 基础风险比例
        base_risk_per_trade = self.config.get('strategy', 'default', 'risk_per_trade', default=0.02)
        max_position_size = self.config.get('strategy', 'default', 'max_position_size', default=0.3)

        # 根据信号强度调整
        if signal_strength == "strong":
            risk_multiplier = 1.5
        elif signal_strength == "medium":
            risk_multiplier = 1.0
        else:  # weak
            risk_multiplier = 0.5

        # 根据波动率调整（高波动率降低仓位）
        if volatility and volatility > 2:  # 2%以上
            volatility_multiplier = 0.7
        elif volatility and volatility > 1:
            volatility_multiplier = 0.85
        else:
            volatility_multiplier = 1.0

        # 计算风险金额
        risk_amount = account_value * base_risk_per_trade * risk_multiplier * volatility_multiplier

        # 限制最大仓位
        risk_amount = min(risk_amount, account_value * max_position_size)

        # 计算持仓数量
        if signal_direction == "neutral":
            return 0.0

        position_value = risk_amount * 2  # 2倍杠杆（可配置）
        position_size = position_value / current_price

        logger.info(f"仓位计算: 信号={signal_direction}, 强度={signal_strength}, "
                   f"波动率={volatility:.2f}%, 仓位={position_size:.2f}")

        return position_size

    def calculate_stop_loss(self,
                            entry_price: float,
                            signal_direction: str,
                            volatility: float = None,
                            atr: float = None) -> float:
        """
        计算止损位

        Args:
            entry_price: 入场价格
            signal_direction: 信号方向 (long/short)
            volatility: 波动率（可选）
            atr: ATR值（可选）

        Returns:
            止损价格
        """
        # 默认止损比例
        default_stop_loss = self.config.get('strategy', 'default', 'stop_loss', default=0.05)

        # 使用ATR或波动率计算动态止损
        if atr is not None:
            # ATR止损（更灵活）
            stop_distance = atr * 2  # 2倍ATR
            stop_loss_pct = stop_distance / entry_price
        elif volatility is not None:
            # 基于波动率的止损
            stop_loss_pct = min(volatility * 1.5, default_stop_loss * 2)
        else:
            # 固定止损
            stop_loss_pct = default_stop_loss

        # 计算止损价格
        if signal_direction == "long":
            stop_loss = entry_price * (1 - stop_loss_pct)
        elif signal_direction == "short":
            stop_loss = entry_price * (1 + stop_loss_pct)
        else:
            stop_loss = entry_price

        logger.info(f"止损计算: 入场={entry_price:.2f}, 方向={signal_direction}, "
                   f"止损={stop_loss:.2f} ({stop_loss_pct*100:.2f}%)")

        return stop_loss

    def calculate_take_profit(self,
                              entry_price: float,
                              stop_loss: float,
                              signal_strength: str) -> float:
        """
        计算止盈位

        Args:
            entry_price: 入场价格
            stop_loss: 止损价格
            signal_strength: 信号强度

        Returns:
            止盈价格
        """
        # 风险收益比
        if signal_strength == "strong":
            risk_reward_ratio = 3.0  # 强信号，更高的风险收益比
        elif signal_strength == "medium":
            risk_reward_ratio = 2.0
        else:
            risk_reward_ratio = 1.5

        # 计算风险距离
        risk = abs(entry_price - stop_loss)

        # 计算止盈价格
        if entry_price > stop_loss:  # 多头
            take_profit = entry_price + risk * risk_reward_ratio
        else:  # 空头
            take_profit = entry_price - risk * risk_reward_ratio

        logger.info(f"止盈计算: 入场={entry_price:.2f}, 止损={stop_loss:.2f}, "
                   f"止盈={take_profit:.2f} (风险收益比={risk_reward_ratio})")

        return take_profit

    def check_risk_limits(self, account_value: float,
                         daily_pnl: float,
                         current_drawdown: float) -> Tuple[bool, str]:
        """
        检查风险限制

        Args:
            account_value: 账户价值
            daily_pnl: 当日盈亏
            current_drawdown: 当前回撤

        Returns:
            (是否允许交易, 原因)
        """
        # 日亏损限制
        daily_loss_limit = self.config.get('risk', 'daily_loss_limit', default=0.05)
        if daily_pnl < 0 and abs(daily_pnl / account_value) > daily_loss_limit:
            reason = f"当日亏损超过限制 ({abs(daily_pnl/account_value)*100:.2f}% > {daily_loss_limit*100:.2f}%)"
            logger.warning(reason)
            return False, reason

        # 最大回撤限制
        max_drawdown_limit = self.config.get('risk', 'max_drawdown', default=0.2)
        if current_drawdown < -max_drawdown_limit:
            reason = f"回撤超过限制 ({abs(current_drawdown)*100:.2f}% > {max_drawdown_limit*100:.2f}%)"
            logger.warning(reason)
            return False, reason

        return True, "风险检查通过"

    def dynamic_trailing_stop(self,
                             entry_price: float,
                             current_price: float,
                             signal_direction: str,
                             highest_profit_pct: float = 0) -> float:
        """
        动态移动止损（追踪止损）

        Args:
            entry_price: 入场价格
            current_price: 当前价格
            signal_direction: 信号方向
            highest_profit_pct: 最高盈利百分比

        Returns:
            新的止损价格
        """
        # 计算当前盈亏
        if signal_direction == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # 盈利时启动移动止损
        if profit_pct > 0.05:  # 盈利超过5%时启动
            # 追踪回撤比例
            trail_pct = 0.3  # 追踪30%的盈利

            if signal_direction == "long":
                trailing_stop = current_price * (1 - trail_pct * profit_pct)
            else:
                trailing_stop = current_price * (1 + trail_pct * profit_pct)

            logger.info(f"移动止损: 当前价格={current_price:.2f}, "
                       f"盈利={profit_pct*100:.2f}%, 追踪止损={trailing_stop:.2f}")

            return trailing_stop

        # 未达到盈利阈值，返回入场价格作为止损
        return entry_price

    def check_position_concentration(self,
                                     positions: Dict[str, Dict],
                                     symbol: str,
                                     max_positions: int = None) -> Tuple[bool, str]:
        """
        检查持仓集中度

        Args:
            positions: 当前持仓字典
            symbol: 要交易的标的
            max_positions: 最大持仓数

        Returns:
            (是否允许交易, 原因)
        """
        if max_positions is None:
            max_positions = self.config.get('risk', 'max_positions', default=10)

        num_positions = len(positions)

        if num_positions >= max_positions:
            reason = f"持仓数量已达上限 ({num_positions}/{max_positions})"
            logger.warning(reason)
            return False, reason

        # 检查是否已持有该标的
        if symbol in positions:
            reason = f"已持有该标的 {symbol}"
            logger.warning(reason)
            return False, reason

        return True, "持仓集中度检查通过"

    def manage_risk_on_profit(self,
                              current_price: float,
                              entry_price: float,
                              position_size: float,
                              signal_direction: str) -> Tuple[float, float]:
        """
        盈利时动态管理仓位

        Args:
            current_price: 当前价格
            entry_price: 入场价格
            position_size: 当前仓位
            signal_direction: 信号方向

        Returns:
            (新仓位, 止损价格)
        """
        # 计算盈利百分比
        if signal_direction == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price

        # 盈利加仓策略
        if profit_pct >= 0.1:  # 盈利10%以上
            # 加仓50%
            new_position_size = position_size * 1.5

            # 更新止损到盈亏平衡点
            if signal_direction == "long":
                new_stop_loss = entry_price
            else:
                new_stop_loss = entry_price

            logger.info(f"盈利加仓: 盈利={profit_pct*100:.2f}%, "
                       f"仓位={position_size:.2f} -> {new_position_size:.2f}")

            return new_position_size, new_stop_loss

        # 未达到加仓条件，保持原仓位
        return position_size, None

    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """
        计算VaR (Value at Risk)

        Args:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            VaR值
        """
        var = np.percentile(returns, (1 - confidence) * 100)
        return var

    def calculate_portfolio_risk(self,
                                  positions: Dict[str, Dict],
                                  correlation_matrix: pd.DataFrame = None) -> Dict[str, float]:
        """
        计算投资组合风险

        Args:
            positions: 持仓字典 {symbol: {size, entry_price, current_price}}
            correlation_matrix: 标的相关性矩阵

        Returns:
            风险指标字典
        """
        total_value = sum([pos['size'] * pos['current_price'] for pos in positions.values()])

        # 计算单个持仓的风险
        position_risks = {}
        for symbol, pos in positions.items():
            if signal_direction := pos.get('direction'):
                if signal_direction == 'long':
                    pct_change = (pos['current_price'] - pos['entry_price']) / pos['entry_price']
                else:
                    pct_change = (pos['entry_price'] - pos['current_price']) / pos['entry_price']

                position_value = pos['size'] * pos['current_price']
                weight = position_value / total_value
                position_risks[symbol] = {
                    'weight': weight,
                    'pnl_pct': pct_change,
                    'contribution_to_risk': weight * abs(pct_change)
                }

        # 如果有相关性矩阵，计算组合风险
        if correlation_matrix is not None:
            # TODO: 实现相关性调整的风险计算
            pass

        return {
            'total_value': total_value,
            'position_risks': position_risks,
            'portfolio_risk': sum([r['contribution_to_risk'] for r in position_risks.values()])
        }


if __name__ == "__main__":
    # 测试风控智能体
    agent = RiskAgent()

    # 测试仓位计算
    position_size = agent.calculate_position_size(
        signal_direction="long",
        signal_strength="strong",
        current_price=100.0,
        account_value=100000,
        volatility=1.5
    )
    print(f"建议仓位: {position_size:.2f}")

    # 测试止损计算
    stop_loss = agent.calculate_stop_loss(
        entry_price=100.0,
        signal_direction="long",
        atr=2.0
    )
    print(f"止损价格: {stop_loss:.2f}")

    # 测试止盈计算
    take_profit = agent.calculate_take_profit(
        entry_price=100.0,
        stop_loss=stop_loss,
        signal_strength="strong"
    )
    print(f"止盈价格: {take_profit:.2f}")

    # 测试风险限制检查
    allowed, reason = agent.check_risk_limits(
        account_value=100000,
        daily_pnl=-3000,
        current_drawdown=-0.08
    )
    print(f"风险检查: {'通过' if allowed else '拒绝'} - {reason}")
