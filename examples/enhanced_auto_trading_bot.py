"""
增强版自动交易机器人
===================
集成可靠性模块、错误处理、数据验证、备份机制
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import sys

# 导入基础函数
from smart_screener_v2 import sma, ema, atr, rsi, macd

# 导入可靠性模块
from system_reliability import SystemReliability

class EnhancedAutoTradingBot:
    """增强版自动交易机器人"""

    def __init__(self, initial_capital=100000):
        # 初始化可靠性模块
        self.reliability = SystemReliability()
        self.logger = self.reliability.logger

        self.logger.info("增强版自动交易机器人启动")

        # 初始资金
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # 持仓
        self.positions = {}

        # 交易记录
        self.trades = []

        # V4配置
        self.config = {
            '300750': {
                'name': '宁德时代',
                'weight': 0.45,
                'ma_fast': 10,
                'ma_slow': 35,
                'atr_stop': 2.0
            },
            '002475': {
                'name': '立讯精密',
                'weight': 0.30,
                'ma_fast': 10,
                'ma_slow': 35,
                'atr_stop': 3.0
            },
            '601318': {
                'name': '中国平安',
                'weight': 0.15,
                'ma_fast': 8,
                'ma_slow': 25,
                'atr_stop': 2.5
            },
            '600276': {
                'name': '恒瑞医药',
                'weight': 0.10,
                'ma_fast': 8,
                'ma_slow': 30,
                'atr_stop': 2.0
            }
        }

        # 交易限制
        self.daily_trades = {}  # 每日交易次数限制
        self.max_daily_trades = 3  # 每只股票每天最多3次
        self.min_trade_interval = timedelta(hours=1)  # 最小交易间隔
        self.last_trade_time = {}  # 上次交易时间

        # 加载数据
        self.load_data()

        # 加载持仓
        self.load_positions()

    def load_data(self):
        """加载股票数据（带验证）"""
        self.logger.info("加载股票数据...")
        self.stock_data = {}

        for code in self.config.keys():
            filepath = Path(f'data/real_{code}.csv')

            if not filepath.exists():
                self.logger.error(f"  ❌ {self.config[code]['name']}: 数据文件不存在")
                continue

            try:
                df = pd.read_csv(filepath)

                # 标准化列名
                if 'datetime' not in df.columns and 'trade_date' in df.columns:
                    df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})

                df = df.sort_values('datetime').reset_index(drop=True)

                # 数据验证
                if self.reliability.validate_data(df, code):
                    self.stock_data[code] = df
                    self.logger.info(f"  ✅ {self.config[code]['name']}: {len(df)}天")
                else:
                    self.logger.error(f"  ❌ {self.config[code]['name']}: 数据验证失败")

            except Exception as e:
                self.logger.error(f"  ❌ {self.config[code]['name']}: 加载失败 - {e}")

        if len(self.stock_data) < len(self.config):
            self.reliability.send_alert(f"只有{len(self.stock_data)}/{len(self.config)}只股票数据加载成功")

    def load_positions(self):
        """加载已有持仓（带备份）"""
        filepath = Path('data/auto_portfolio.json')

        if not filepath.exists():
            self.logger.info("首次运行，创建新持仓文件")
            return

        try:
            # 先备份
            self.reliability.backup_portfolio()

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.cash = data.get('cash', self.initial_capital)
            self.positions = data.get('positions', {})
            self.trades = data.get('trades', [])

            self.logger.info(f"加载已有持仓:")
            self.logger.info(f"  现金: {self.cash:.2f}")
            self.logger.info(f"  持仓: {len(self.positions)}只股票")

        except Exception as e:
            self.logger.error(f"加载持仓失败: {e}")
            self.reliability.send_alert(f"加载持仓失败，使用默认值")

    def save_positions(self):
        """保存持仓（带备份）"""
        try:
            # 保存前先备份
            self.reliability.backup_portfolio()

            data = {
                'update_time': datetime.now().isoformat(),
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'positions': self.positions,
                'trades': self.trades[-100:]  # 只保留最近100条
            }

            filepath = Path('data/auto_portfolio.json')
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info("持仓已保存")

        except Exception as e:
            self.logger.error(f"保存持仓失败: {e}")
            self.reliability.send_alert(f"保存持仓失败")

    def can_trade(self, code: str) -> bool:
        """检查是否可以交易（频率限制）"""
        today = datetime.now().date().isoformat()

        # 检查每日交易次数
        if code not in self.daily_trades:
            self.daily_trades[code] = {}

        if today not in self.daily_trades[code]:
            self.daily_trades[code][today] = 0

        if self.daily_trades[code][today] >= self.max_daily_trades:
            self.logger.warning(f"{self.config[code]['name']}: 今日交易次数已达上限")
            return False

        # 检查交易间隔
        if code in self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time[code]
            if time_since_last < self.min_trade_interval:
                self.logger.warning(f"{self.config[code]['name']}: 交易间隔不足")
                return False

        return True

    def record_trade_time(self, code: str):
        """记录交易时间"""
        today = datetime.now().date().isoformat()

        if code not in self.daily_trades:
            self.daily_trades[code] = {}

        if today not in self.daily_trades[code]:
            self.daily_trades[code][today] = 0

        self.daily_trades[code][today] += 1
        self.last_trade_time[code] = datetime.now()

    def check_signals(self):
        """检查交易信号（带错误处理）"""
        self.logger.info("="*70)
        self.logger.info(f"信号检测 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.logger.info("="*70)

        signals = {}

        for code, cfg in self.config.items():
            if code not in self.stock_data:
                self.logger.warning(f"{cfg['name']}: 无数据，跳过")
                continue

            try:
                df = self.stock_data[code]

                # 计算指标（带错误处理）
                df['ma_fast'] = self.reliability.safe_execute(sma, df['close'], cfg['ma_fast'])
                df['ma_slow'] = self.reliability.safe_execute(sma, df['close'], cfg['ma_slow'])
                df['atr'] = self.reliability.safe_execute(atr, df['high'], df['low'], df['close'], 14)
                df['rsi'] = self.reliability.safe_execute(rsi, df['close'], 14)
                df['macd'], df['macd_signal'], df['macd_hist'] = self.reliability.safe_execute(
                    macd, df['close']
                ) or (None, None, None)

                # 检查计算是否成功
                if df['ma_fast'].isnull().all() or df['ma_slow'].isnull().all():
                    self.logger.error(f"{cfg['name']}: 指标计算失败")
                    continue

                # 最新数据
                latest = df.iloc[-1]
                prev = df.iloc[-2]

                ma_fast = latest['ma_fast']
                ma_slow = latest['ma_slow']
                rsi_val = latest['rsi']
                macd_hist = latest['macd_hist']
                close = latest['close']

                # 信号判断
                buy_signal = (
                    ma_fast > ma_slow and
                    prev['ma_fast'] <= prev['ma_slow'] and
                    macd_hist > 0 and
                    30 < rsi_val < 70
                )

                # 卖出信号
                sell_signal = False
                sell_reason = ""

                if code in self.positions:
                    pos = self.positions[code]

                    # 止损
                    if close <= pos['stop_loss']:
                        sell_signal = True
                        sell_reason = "止损"

                    # 趋势反转
                    elif ma_fast < ma_slow and macd_hist < 0:
                        sell_signal = True
                        sell_reason = "趋势反转"

                signals[code] = {
                    'name': cfg['name'],
                    'price': close,
                    'ma_fast': ma_fast,
                    'ma_slow': ma_slow,
                    'rsi': rsi_val,
                    'macd_hist': macd_hist,
                    'buy_signal': buy_signal,
                    'sell_signal': sell_signal,
                    'sell_reason': sell_reason,
                    'trend': 'UP' if ma_fast > ma_slow else 'DOWN'
                }

                # 显示
                status = "🟢" if buy_signal else "🔴" if sell_signal else "⚪"
                self.logger.info(f"\n{status} {cfg['name']} ({code})")
                self.logger.info(f"  价格: {close:.2f}")
                self.logger.info(f"  MA{cfg['ma_fast']}: {ma_fast:.2f} | MA{cfg['ma_slow']}: {ma_slow:.2f}")
                self.logger.info(f"  趋势: {signals[code]['trend']}")
                self.logger.info(f"  RSI: {rsi_val:.1f} | MACD: {macd_hist:.2f}")

                if buy_signal:
                    self.logger.info(f"  ⚡ 买入信号！")
                elif sell_signal:
                    self.logger.info(f"  ⚠️ 卖出信号: {sell_reason}")

            except Exception as e:
                self.logger.error(f"{cfg['name']}: 信号检测失败 - {e}")
                continue

        return signals

    def execute_trades(self, signals):
        """执行交易（带频率限制）"""
        self.logger.info("\n" + "="*70)
        self.logger.info("执行交易")
        self.logger.info("="*70)

        # 先处理卖出
        for code, signal in signals.items():
            if signal['sell_signal'] and code in self.positions:
                if self.can_trade(code):
                    self.sell(code, signal['price'], signal['sell_reason'])

        # 再处理买入
        for code, signal in signals.items():
            if signal['buy_signal'] and code not in self.positions:
                if self.can_trade(code):
                    self.buy(code, signal['price'])

        # 保存
        self.save_positions()

    def buy(self, code: str, price: float):
        """买入（带验证）"""
        try:
            cfg = self.config[code]

            # 计算买入金额
            target_value = self.initial_capital * cfg['weight']
            buy_value = target_value * 0.30

            # 检查现金
            if buy_value > self.cash:
                buy_value = self.cash * 0.50

            # 计算股数
            shares = int(buy_value / price)

            if shares <= 0:
                self.logger.warning(f"{cfg['name']}: 现金不足，无法买入")
                return False

            # 扣除现金
            cost = shares * price
            self.cash -= cost

            # 计算止损位
            df = self.stock_data[code]
            atr_val = df['atr'].iloc[-1]
            stop_loss = price - atr_val * cfg['atr_stop']

            # 记录持仓
            self.positions[code] = {
                'name': cfg['name'],
                'shares': shares,
                'buy_price': price,
                'buy_time': datetime.now().isoformat(),
                'stop_loss': stop_loss,
                'highest': price,
                'cost': cost
            }

            # 记录交易
            self.trades.append({
                'time': datetime.now().isoformat(),
                'code': code,
                'name': cfg['name'],
                'action': 'BUY',
                'shares': shares,
                'price': price,
                'amount': cost
            })

            # 记录交易时间
            self.record_trade_time(code)

            self.logger.info(f"\n  ✅ 买入 {cfg['name']}")
            self.logger.info(f"     数量: {shares}股")
            self.logger.info(f"     价格: {price:.2f}")
            self.logger.info(f"     金额: {cost:.2f}")
            self.logger.info(f"     止损: {stop_loss:.2f} (-{(price-stop_loss)/price*100:.1f}%)")

            return True

        except Exception as e:
            self.logger.error(f"买入失败: {e}")
            return False

    def sell(self, code: str, price: float, reason: str):
        """卖出（带验证）"""
        try:
            if code not in self.positions:
                return False

            pos = self.positions[code]
            shares = pos['shares']
            revenue = shares * price
            profit = revenue - pos['cost']
            profit_pct = profit / pos['cost'] * 100

            # 增加现金
            self.cash += revenue

            # 记录交易
            self.trades.append({
                'time': datetime.now().isoformat(),
                'code': code,
                'name': pos['name'],
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'amount': revenue,
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': reason
            })

            # 记录交易时间
            self.record_trade_time(code)

            # 删除持仓
            del self.positions[code]

            status = "✅" if profit > 0 else "❌"
            self.logger.info(f"\n  {status} 卖出 {pos['name']}: {reason}")
            self.logger.info(f"     数量: {shares}股")
            self.logger.info(f"     价格: {price:.2f}")
            self.logger.info(f"     金额: {revenue:.2f}")
            self.logger.info(f"     盈亏: {profit:+.2f} ({profit_pct:+.1f}%)")

            return True

        except Exception as e:
            self.logger.error(f"卖出失败: {e}")
            return False

    def run(self):
        """运行交易机器人（带健康检查）"""
        try:
            self.logger.info("="*70)
            self.logger.info("增强版自动交易机器人 V4")
            self.logger.info("="*70)

            # 系统健康检查
            health = self.reliability.health_check()
            if health['status'] == 'error':
                self.logger.error("系统健康检查失败，停止运行")
                return

            # 检查信号
            signals = self.check_signals()

            # 执行交易
            self.execute_trades(signals)

            # 显示汇总
            self.show_summary()

            # 保存
            self.save_positions()

            # 显示系统统计
            stats = self.reliability.get_system_stats()
            self.logger.info(f"\n📊 系统统计:")
            self.logger.info(f"  总交易: {stats.get('trades_count', 0)}次")
            self.logger.info(f"  当前持仓: {stats.get('positions_count', 0)}只")
            self.logger.info(f"  备份文件: {stats.get('backup_count', 0)}个")
            self.logger.info(f"  错误次数: {stats.get('current_errors', 0)}次")

            self.logger.info(f"\n✅ 运行完成！")

        except Exception as e:
            self.logger.error(f"系统运行失败: {e}")
            self.reliability.send_alert(f"系统运行失败: {e}")

    def show_summary(self):
        """显示持仓汇总"""
        self.logger.info("\n" + "="*70)
        self.logger.info("持仓汇总")
        self.logger.info("="*70)

        if not self.positions:
            self.logger.info("\n空仓")
            self.logger.info(f"现金: {self.cash:.2f}")
            self.logger.info(f"总资产: {self.cash:.2f}")
            return

        total_value = self.cash

        self.logger.info(f"\n{'股票':<10} {'持仓':<8} {'成本':<10} {'现价':<10} {'市值':<12} {'盈亏':<15}")
        self.logger.info("-" * 70)

        for code, pos in self.positions.items():
            if code in self.stock_data:
                price = self.stock_data[code]['close'].iloc[-1]
            else:
                price = pos['buy_price']

            market_value = pos['shares'] * price
            profit = market_value - pos['cost']
            profit_pct = profit / pos['cost'] * 100

            total_value += market_value

            status = "✅" if profit > 0 else "❌"
            self.logger.info(
                f"{pos['name']:<10} {pos['shares']:<8} {pos['buy_price']:<10.2f} "
                f"{price:<10.2f} {market_value:<12.2f} {status} {profit:+.2f} ({profit_pct:+.1f}%)"
            )

        total_return = (total_value - self.initial_capital) / self.initial_capital * 100

        self.logger.info("-" * 70)
        self.logger.info(f"现金: {self.cash:.2f}")
        self.logger.info(f"持仓市值: {total_value - self.cash:.2f}")
        self.logger.info(f"总资产: {total_value:.2f}")
        self.logger.info(f"总收益: {total_return:+.2f}%")
        self.logger.info("="*70)

def main():
    """主函数"""
    bot = EnhancedAutoTradingBot(initial_capital=100000)
    bot.run()

if __name__ == '__main__':
    main()
