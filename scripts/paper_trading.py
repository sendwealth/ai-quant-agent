#!/usr/bin/env python3
"""
模拟交易执行器
用于小资金实盘测试
"""
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class PaperTrader:
    """模拟交易账户"""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {symbol: {shares, avg_cost}}
        self.trades = []
        self.daily_values = []

        # 策略参数
        self.stop_loss = -0.06  # -6%
        self.take_profit_1 = 0.10  # +10%卖50%
        self.take_profit_2 = 0.20  # +20%清仓

        # 风控参数
        self.max_position_pct = 0.30  # 单只最大30%
        self.max_total_position_pct = 0.80  # 总仓位最大80%

    def execute_signal(self, symbol: str, signal: str, price: float, shares: int = None):
        """执行交易信号"""

        # 计算可用仓位
        max_shares = int((self.cash * self.max_position_pct) / price)

        if shares is None:
            shares = max_shares

        if signal == 'BUY':
            return self._buy(symbol, price, shares)
        elif signal == 'SELL':
            return self._sell(symbol, price, shares)
        else:
            return None

    def _buy(self, symbol: str, price: float, shares: int):
        """买入"""
        cost = price * shares

        if cost > self.cash:
            shares = int(self.cash / price)
            cost = price * shares

        if shares <= 0:
            return None

        # 更新持仓
        if symbol in self.positions:
            old_shares = self.positions[symbol]['shares']
            old_cost = self.positions[symbol]['avg_cost']
            total_cost = old_cost * old_shares + cost
            total_shares = old_shares + shares
            self.positions[symbol] = {
                'shares': total_shares,
                'avg_cost': total_cost / total_shares
            }
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_cost': price
            }

        self.cash -= cost

        # 记录交易
        trade = {
            'time': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'cost': cost
        }
        self.trades.append(trade)

        return trade

    def _sell(self, symbol: str, price: float, shares: int = None):
        """卖出"""
        if symbol not in self.positions:
            return None

        if shares is None:
            shares = self.positions[symbol]['shares']

        shares = min(shares, self.positions[symbol]['shares'])

        if shares <= 0:
            return None

        # 更新持仓
        self.positions[symbol]['shares'] -= shares
        if self.positions[symbol]['shares'] == 0:
            del self.positions[symbol]

        # 更新现金
        revenue = price * shares
        self.cash += revenue

        # 记录交易
        trade = {
            'time': datetime.now().isoformat(),
            'symbol': symbol,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'revenue': revenue
        }
        self.trades.append(trade)

        return trade

    def check_risk(self, symbol: str, current_price: float):
        """检查风险（止损/止盈）"""

        if symbol not in self.positions:
            return None

        position = self.positions[symbol]
        avg_cost = position['avg_cost']
        pnl_pct = (current_price - avg_cost) / avg_cost

        # 止损
        if pnl_pct <= self.stop_loss:
            return {
                'action': 'STOP_LOSS',
                'symbol': symbol,
                'shares': position['shares'],
                'reason': f'触发止损 ({pnl_pct:.2%})'
            }

        # 止盈1
        if pnl_pct >= self.take_profit_1 and position['shares'] > 0:
            sell_shares = position['shares'] // 2
            if sell_shares > 0:
                return {
                    'action': 'TAKE_PROFIT_1',
                    'symbol': symbol,
                    'shares': sell_shares,
                    'reason': f'触发止盈1 ({pnl_pct:.2%})'
                }

        # 止盈2
        if pnl_pct >= self.take_profit_2:
            return {
                'action': 'TAKE_PROFIT_2',
                'symbol': symbol,
                'shares': position['shares'],
                'reason': f'触发止盈2 ({pnl_pct:.2%})'
            }

        return None

    def get_portfolio_value(self, prices: Dict[str, float]):
        """计算总资产"""

        position_value = 0
        for symbol, position in self.positions.items():
            if symbol in prices:
                position_value += position['shares'] * prices[symbol]

        total_value = self.cash + position_value

        # 记录每日价值
        self.daily_values.append({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'return_pct': (total_value - self.initial_capital) / self.initial_capital
        })

        return {
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value,
            'return_pct': (total_value - self.initial_capital) / self.initial_capital,
            'positions': self.positions
        }

    def save_state(self, filepath: str = 'data/paper_trading_state.json'):
        """保存状态"""
        state = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'trades': self.trades,
            'daily_values': self.daily_values[-30:]  # 只保留最近30天
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, filepath: str = 'data/paper_trading_state.json'):
        """加载状态"""
        if not Path(filepath).exists():
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)

        self.initial_capital = state['initial_capital']
        self.cash = state['cash']
        self.positions = state['positions']
        self.trades = state['trades']
        self.daily_values = state.get('daily_values', [])

        return True


def main():
    """主函数 - 测试V5策略"""
    print("="*70)
    print("📊 V5策略模拟交易测试")
    print("="*70)

    # 初始化
    trader = PaperTrader(initial_capital=10000)

    # 加载已有状态
    if trader.load_state():
        print("✅ 加载已有账户状态")
    else:
        print("📝 创建新账户")

    # 获取今日信号
    analysis_file = Path('data/daily_analysis_results.json')
    if not analysis_file.exists():
        print("❌ 未找到分析结果，请先运行 python3 run.py")
        return

    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis = json.load(f)

    signals = analysis['signals']

    print(f"\n📋 今日信号:")
    for code, data in signals.items():
        print(f"  {data['name']} ({code}): {data['signal']} @ ¥{data['price']}")

    # 执行交易
    print(f"\n💹 执行交易:")

    for code, data in signals.items():
        signal = data['signal']
        price = data['price']
        name = data['name']

        if signal == 'BUY':
            trade = trader.execute_signal(code, 'BUY', price)
            if trade:
                print(f"  ✅ 买入 {name}: {trade['shares']}股 @ ¥{price}")

        elif signal == 'SELL':
            trade = trader.execute_signal(code, 'SELL', price)
            if trade:
                print(f"  ✅ 卖出 {name}: {trade['shares']}股 @ ¥{price}")

    # 检查风险
    print(f"\n⚠️  风险检查:")
    for code in list(trader.positions.keys()):
        risk = trader.check_risk(code, signals.get(code, {}).get('price', 0))
        if risk:
            print(f"  🔴 {risk['symbol']}: {risk['reason']}")
            # 执行风控
            trader.execute_signal(code, 'SELL', signals[code]['price'], risk['shares'])

    # 计算当前资产
    prices = {code: data['price'] for code, data in signals.items()}
    portfolio = trader.get_portfolio_value(prices)

    print(f"\n💰 账户状态:")
    print(f"  现金: ¥{portfolio['cash']:,.2f}")
    print(f"  持仓: ¥{portfolio['position_value']:,.2f}")
    print(f"  总资产: ¥{portfolio['total_value']:,.2f}")
    print(f"  收益率: {portfolio['return_pct']:+.2%}")

    if portfolio['positions']:
        print(f"\n📊 持仓明细:")
        for symbol, pos in portfolio['positions'].items():
            name = signals.get(symbol, {}).get('name', symbol)
            current_price = signals.get(symbol, {}).get('price', 0)
            pnl_pct = (current_price - pos['avg_cost']) / pos['avg_cost']
            print(f"  {name}: {pos['shares']}股 @ ¥{pos['avg_cost']:.2f} (现价¥{current_price}, {pnl_pct:+.2%})")

    # 保存状态
    trader.save_state()
    print(f"\n✅ 状态已保存")

    print("="*70)


if __name__ == '__main__':
    main()
