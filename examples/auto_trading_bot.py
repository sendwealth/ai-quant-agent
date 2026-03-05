"""
自动化模拟交易系统
==================
自动运行V4策略，无需手动操作
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json

# 导入基础函数
from smart_screener_v2 import sma, ema, atr, rsi, macd

class AutoTradingBot:
    """自动化交易机器人"""
    
    def __init__(self, initial_capital=100000):
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
        
        # 加载数据
        self.load_data()
        
        # 加载持仓
        self.load_positions()
    
    def load_data(self):
        """加载股票数据"""
        print("\n加载股票数据...")
        self.stock_data = {}
        
        for code in self.config.keys():
            filepath = Path(f'data/real_{code}.csv')
            if filepath.exists():
                df = pd.read_csv(filepath)
                if 'datetime' not in df.columns and 'trade_date' in df.columns:
                    df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
                df = df.sort_values('datetime').reset_index(drop=True)
                self.stock_data[code] = df
                print(f"  ✅ {self.config[code]['name']}: {len(df)}天")
    
    def load_positions(self):
        """加载已有持仓"""
        filepath = Path('data/auto_portfolio.json')
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.cash = data.get('cash', self.initial_capital)
                self.positions = data.get('positions', {})
                self.trades = data.get('trades', [])
            print(f"\n加载已有持仓:")
            print(f"  现金: {self.cash:.2f}")
            print(f"  持仓: {len(self.positions)}只股票")
    
    def save_positions(self):
        """保存持仓"""
        data = {
            'update_time': datetime.now().isoformat(),
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'trades': self.trades[-100:]  # 只保留最近100条
        }
        
        with open('data/auto_portfolio.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def check_signals(self):
        """检查交易信号"""
        print("\n" + "="*70)
        print(f"信号检测 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
        signals = {}
        
        for code, cfg in self.config.items():
            if code not in self.stock_data:
                continue
            
            df = self.stock_data[code]
            
            # 计算指标
            df['ma_fast'] = sma(df['close'], cfg['ma_fast'])
            df['ma_slow'] = sma(df['close'], cfg['ma_slow'])
            df['atr'] = atr(df['high'], df['low'], df['close'], 14)
            df['rsi'] = rsi(df['close'], 14)
            df['macd'], df['macd_signal'], df['macd_hist'] = macd(df['close'])
            
            # 最新数据
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            ma_fast = latest['ma_fast']
            ma_slow = latest['ma_slow']
            rsi_val = latest['rsi']
            macd_hist = latest['macd_hist']
            close = latest['close']
            
            # 信号判断
            # 买入：MA金叉
            buy_signal = (
                ma_fast > ma_slow and  # 快线在慢线上方
                prev['ma_fast'] <= prev['ma_slow'] and  # 前一根K线快线在慢线下方或相等
                macd_hist > 0 and  # MACD为正
                30 < rsi_val < 70  # RSI适中
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
            print(f"\n{status} {cfg['name']} ({code})")
            print(f"  价格: {close:.2f}")
            print(f"  MA{cfg['ma_fast']}: {ma_fast:.2f} | MA{cfg['ma_slow']}: {ma_slow:.2f}")
            print(f"  趋势: {signals[code]['trend']}")
            print(f"  RSI: {rsi_val:.1f} | MACD: {macd_hist:.2f}")
            
            if buy_signal:
                print(f"  ⚡ 买入信号！")
            elif sell_signal:
                print(f"  ⚠️ 卖出信号: {sell_reason}")
        
        return signals
    
    def execute_trades(self, signals):
        """执行交易"""
        print("\n" + "="*70)
        print("执行交易")
        print("="*70)
        
        # 先处理卖出
        for code, signal in signals.items():
            if signal['sell_signal'] and code in self.positions:
                self.sell(code, signal['price'], signal['sell_reason'])
        
        # 再处理买入
        for code, signal in signals.items():
            if signal['buy_signal'] and code not in self.positions:
                self.buy(code, signal['price'])
        
        # 保存
        self.save_positions()
    
    def buy(self, code, price):
        """买入"""
        cfg = self.config[code]
        
        # 计算买入金额（初始仓位30%）
        target_value = self.initial_capital * cfg['weight']
        buy_value = target_value * 0.30  # 初始仓位30%
        
        # 检查现金
        if buy_value > self.cash:
            buy_value = self.cash * 0.50  # 最多用50%现金
        
        # 计算股数
        shares = int(buy_value / price)
        
        if shares <= 0:
            print(f"  ⚠️ {cfg['name']}: 现金不足，无法买入")
            return
        
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
        
        print(f"\n  ✅ 买入 {cfg['name']}")
        print(f"     数量: {shares}股")
        print(f"     价格: {price:.2f}")
        print(f"     金额: {cost:.2f}")
        print(f"     止损: {stop_loss:.2f} (-{(price-stop_loss)/price*100:.1f}%)")
    
    def sell(self, code, price, reason):
        """卖出"""
        if code not in self.positions:
            return
        
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
        
        # 删除持仓
        del self.positions[code]
        
        status = "✅" if profit > 0 else "❌"
        print(f"\n  {status} 卖出 {pos['name']}: {reason}")
        print(f"     数量: {shares}股")
        print(f"     价格: {price:.2f}")
        print(f"     金额: {revenue:.2f}")
        print(f"     盈亏: {profit:+.2f} ({profit_pct:+.1f}%)")
    
    def update_positions(self, signals):
        """更新持仓（止损止盈检查）"""
        for code, pos in list(self.positions.items()):
            if code not in signals:
                continue
            
            price = signals[code]['price']
            
            # 更新最高价
            if price > pos['highest']:
                pos['highest'] = price
                # 更新止损位（移动止损）
                df = self.stock_data[code]
                atr_val = df['atr'].iloc[-1]
                new_stop = pos['highest'] - atr_val * self.config[code]['atr_stop']
                if new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop
            
            # 检查止盈1 (10%)
            if price >= pos['buy_price'] * 1.10:
                # 卖出50%
                sell_shares = int(pos['shares'] * 0.5)
                if sell_shares > 0:
                    revenue = sell_shares * price
                    self.cash += revenue
                    pos['shares'] -= sell_shares
                    pos['cost'] -= pos['cost'] * 0.5
                    
                    print(f"\n  💰 {pos['name']}: 止盈1 (+10%)")
                    print(f"     卖出: {sell_shares}股 @ {price:.2f}")
                    print(f"     剩余: {pos['shares']}股")
                    
                    if pos['shares'] <= 0:
                        del self.positions[code]
            
            # 检查止盈2 (20%)
            if code in self.positions and price >= pos['buy_price'] * 1.20:
                # 清仓
                self.sell(code, price, "止盈2 (+20%)")
    
    def show_summary(self):
        """显示持仓汇总"""
        print("\n" + "="*70)
        print("持仓汇总")
        print("="*70)
        
        if not self.positions:
            print("\n空仓")
            print(f"现金: {self.cash:.2f}")
            print(f"总资产: {self.cash:.2f}")
            return
        
        total_value = self.cash
        
        print(f"\n{'股票':<10} {'持仓':<8} {'成本':<10} {'现价':<10} {'市值':<12} {'盈亏':<15}")
        print("-" * 70)
        
        for code, pos in self.positions.items():
            # 获取最新价格
            if code in self.stock_data:
                price = self.stock_data[code]['close'].iloc[-1]
            else:
                price = pos['buy_price']
            
            market_value = pos['shares'] * price
            profit = market_value - pos['cost']
            profit_pct = profit / pos['cost'] * 100
            
            total_value += market_value
            
            status = "✅" if profit > 0 else "❌"
            print(f"{pos['name']:<10} {pos['shares']:<8} {pos['buy_price']:<10.2f} {price:<10.2f} {market_value:<12.2f} {status} {profit:+.2f} ({profit_pct:+.1f}%)")
        
        total_return = (total_value - self.initial_capital) / self.initial_capital * 100
        
        print("-" * 70)
        print(f"现金: {self.cash:.2f}")
        print(f"持仓市值: {total_value - self.cash:.2f}")
        print(f"总资产: {total_value:.2f}")
        print(f"总收益: {total_return:+.2f}%")
        print("="*70)
    
    def run(self):
        """运行交易机器人"""
        print("="*70)
        print("自动化模拟交易机器人 V4")
        print("="*70)
        
        # 检查信号
        signals = self.check_signals()
        
        # 更新持仓
        self.update_positions(signals)
        
        # 执行交易
        self.execute_trades(signals)
        
        # 显示汇总
        self.show_summary()
        
        # 保存
        self.save_positions()
        
        print(f"\n✅ 运行完成！结果已保存到 data/auto_portfolio.json")

def main():
    """主函数"""
    bot = AutoTradingBot(initial_capital=100000)
    bot.run()

if __name__ == '__main__':
    main()
