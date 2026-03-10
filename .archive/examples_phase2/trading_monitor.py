"""
交易监控和管理工具
==================
用于模拟账户的交易记录、信号检测、收益分析
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json

# TuShare配置（用于获取实时价格）
TUSHARE_TOKEN = '33649d8db312befd2e253d93e9bd2860e9c5e819864c8a2078b3869b'

try:
    import tushare as ts
    ts.set_token(TUSHARE_TOKEN)
    pro = ts.pro_api()
    HAS_TUSHARE = True
except:
    HAS_TUSHARE = False

class TradingMonitor:
    """交易监控系统"""
    
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # 持仓
        self.trades = []  # 交易记录
        self.daily_values = []  # 每日净值
        
        # 策略参数
        self.params = {
            'ma_fast': 10,
            'ma_slow': 30,
            'atr_stop_mult': 2.5,
            'take_profit_1': 0.10,
            'take_profit_2': 0.20,
        }
        
        # 股票池
        self.stock_pool = {
            '300750': {'name': '宁德时代', 'target_weight': 0.60},
            '601318': {'name': '中国平安', 'target_weight': 0.40}
        }
        
        # 加载历史数据
        self.load_data()
    
    def load_data(self):
        """加载历史数据"""
        print("\n加载股票数据...")
        
        for code in self.stock_pool.keys():
            filepath = Path(f'data/real_{code}.csv')
            if filepath.exists():
                df = pd.read_csv(filepath)
                if 'trade_date' in df.columns:
                    df = df.rename(columns={'trade_date': 'datetime', 'vol': 'volume'})
                df = df.sort_values('datetime').reset_index(drop=True)
                self.stock_pool[code]['data'] = df
                print(f"  ✅ {self.stock_pool[code]['name']}: {len(df)}天")
            else:
                print(f"  ⚠️ {self.stock_pool[code]['name']}: 数据不存在")
    
    def get_current_price(self, code):
        """获取当前价格"""
        if HAS_TUSHARE:
            try:
                ts_code = code + ('.SH' if code.startswith('6') else '.SZ')
                df = pro.daily(ts_code=ts_code, start_date=datetime.now().strftime('%Y%m%d'))
                if df is not None and len(df) > 0:
                    return float(df['close'].iloc[0])
            except:
                pass
        
        # 使用最新历史数据
        if code in self.stock_pool and 'data' in self.stock_pool[code]:
            return float(self.stock_pool[code]['data']['close'].iloc[-1])
        
        return None
    
    def check_signal(self, code, date=None):
        """检查交易信号"""
        if code not in self.stock_pool or 'data' not in self.stock_pool[code]:
            return None
        
        df = self.stock_pool[code]['data'].copy()
        
        # 如果指定日期，取到该日期为止
        if date:
            df = df[df['datetime'] <= date]
        
        if len(df) < 60:
            return None
        
        # 计算指标
        df['ma_fast'] = df['close'].rolling(self.params['ma_fast']).mean()
        df['ma_slow'] = df['close'].rolling(self.params['ma_slow']).mean()
        
        # 最新数据
        latest = df.iloc[-1]
        ma_fast = latest['ma_fast']
        ma_slow = latest['ma_slow']
        close = latest['close']
        
        # 趋势判断
        trend = 'UP' if ma_fast > ma_slow else 'DOWN'
        
        # 交易信号
        signal = None
        if ma_fast > ma_slow:
            # 检查是否刚金叉
            prev = df.iloc[-2]
            if prev['ma_fast'] <= prev['ma_slow']:
                signal = 'BUY'
        else:
            # 检查是否刚死叉
            prev = df.iloc[-2]
            if prev['ma_fast'] >= prev['ma_slow']:
                signal = 'SELL'
        
        return {
            'code': code,
            'name': self.stock_pool[code]['name'],
            'date': latest['datetime'],
            'close': close,
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'trend': trend,
            'signal': signal
        }
    
    def buy(self, code, shares, price, date, reason=''):
        """买入"""
        cost = shares * price
        if cost > self.cash:
            print(f"❌ 资金不足 (需要{cost:.2f}, 可用{self.cash:.2f})")
            return False
        
        self.cash -= cost
        
        if code not in self.positions:
            self.positions[code] = {
                'name': self.stock_pool[code]['name'],
                'shares': 0,
                'avg_cost': 0,
                'stop_loss': 0,
                'highest': 0
            }
        
        pos = self.positions[code]
        total_shares = pos['shares'] + shares
        pos['avg_cost'] = (pos['avg_cost'] * pos['shares'] + cost) / total_shares
        pos['shares'] = total_shares
        pos['highest'] = max(pos['highest'], price)
        pos['stop_loss'] = pos['avg_cost'] * (1 - 0.08)  # 8%止损
        
        self.trades.append({
            'date': date,
            'code': code,
            'name': pos['name'],
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'amount': cost,
            'reason': reason
        })
        
        print(f"✅ 买入 {pos['name']} {shares}股 @ {price:.2f} (成本{cost:.2f})")
        return True
    
    def sell(self, code, shares, price, date, reason=''):
        """卖出"""
        if code not in self.positions or self.positions[code]['shares'] < shares:
            print(f"❌ 持仓不足")
            return False
        
        amount = shares * price
        self.cash += amount
        
        pos = self.positions[code]
        profit = (price - pos['avg_cost']) * shares
        profit_pct = (price / pos['avg_cost'] - 1) * 100
        
        pos['shares'] -= shares
        if pos['shares'] == 0:
            del self.positions[code]
        
        self.trades.append({
            'date': date,
            'code': code,
            'name': pos['name'],
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'amount': amount,
            'profit': profit,
            'profit_pct': profit_pct,
            'reason': reason
        })
        
        print(f"✅ 卖出 {pos['name']} {shares}股 @ {price:.2f} | 盈亏: {profit:+.2f} ({profit_pct:+.2f}%) [{reason}]")
        return True
    
    def summary(self):
        """持仓汇总"""
        total_value = self.cash
        
        print("\n" + "="*70)
        print("持仓汇总")
        print("="*70)
        
        if not self.positions:
            print("\n空仓")
            print(f"现金: {self.cash:.2f}")
            print(f"总资产: {total_value:.2f}")
            return
        
        print(f"\n{'股票':<10} {'持仓':<8} {'成本':<10} {'现价':<10} {'市值':<12} {'盈亏':<15}")
        print("-" * 70)
        
        for code, pos in self.positions.items():
            current_price = self.get_current_price(code) or pos['avg_cost']
            market_value = pos['shares'] * current_price
            profit = (current_price - pos['avg_cost']) * pos['shares']
            profit_pct = (current_price / pos['avg_cost'] - 1) * 100
            
            total_value += market_value
            
            status = "✅" if profit > 0 else "❌"
            print(f"{pos['name']:<10} {pos['shares']:<8} {pos['avg_cost']:<10.2f} {current_price:<10.2f} {market_value:<12.2f} {status} {profit:+.2f} ({profit_pct:+.2f}%)")
        
        total_return = (total_value / self.initial_capital - 1) * 100
        
        print("-" * 70)
        print(f"现金: {self.cash:.2f}")
        print(f"持仓市值: {total_value - self.cash:.2f}")
        print(f"总资产: {total_value:.2f}")
        print(f"总收益: {total_return:+.2f}%")
        print("="*70)
    
    def check_stop_loss(self):
        """检查止损"""
        print("\n止损检查:")
        
        for code, pos in list(self.positions.items()):
            current_price = self.get_current_price(code)
            if not current_price:
                continue
            
            # 更新最高价
            if current_price > pos['highest']:
                pos['highest'] = current_price
                # 更新止损位（移动止损）
                new_stop = pos['highest'] * 0.92  # 从最高价回落8%
                if new_stop > pos['stop_loss']:
                    pos['stop_loss'] = new_stop
            
            # 检查止损
            if current_price <= pos['stop_loss']:
                print(f"  ⚠️ {pos['name']} 触发止损: {current_price:.2f} <= {pos['stop_loss']:.2f}")
            else:
                print(f"  ✅ {pos['name']}: {current_price:.2f} (止损{pos['stop_loss']:.2f})")
    
    def check_take_profit(self):
        """检查止盈"""
        print("\n止盈检查:")
        
        for code, pos in list(self.positions.items()):
            current_price = self.get_current_price(code)
            if not current_price:
                continue
            
            profit_pct = (current_price / pos['avg_cost'] - 1)
            
            if profit_pct >= self.params['take_profit_2']:
                print(f"  💰 {pos['name']} 达到第二止盈目标: {profit_pct*100:+.2f}% >= 20%")
            elif profit_pct >= self.params['take_profit_1']:
                print(f"  💰 {pos['name']} 达到第一止盈目标: {profit_pct*100:+.2f}% >= 10%")
            else:
                print(f"  ✅ {pos['name']}: {profit_pct*100:+.2f}%")
    
    def daily_report(self):
        """每日报告"""
        print("\n" + "="*70)
        print(f"每日报告 - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
        # 1. 检查信号
        print("\n【交易信号】")
        for code in self.stock_pool.keys():
            signal = self.check_signal(code)
            if signal:
                status = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
                print(f"  {status} {signal['name']}: {signal['trend']} | MA{self.params['ma_fast']}={signal['ma_fast']:.2f} | MA{self.params['ma_slow']}={signal['ma_slow']:.2f} | 现价={signal['close']:.2f}")
                if signal['signal']:
                    print(f"      ⚡ 信号: {signal['signal']}")
        
        # 2. 持仓汇总
        self.summary()
        
        # 3. 止损检查
        self.check_stop_loss()
        
        # 4. 止盈检查
        self.check_take_profit()
        
        # 5. 交易统计
        if self.trades:
            print("\n【交易统计】")
            df_trades = pd.DataFrame(self.trades)
            buys = df_trades[df_trades['action'] == 'BUY']
            sells = df_trades[df_trades['action'] == 'SELL']
            
            print(f"  买入次数: {len(buys)}")
            print(f"  卖出次数: {len(sells)}")
            
            if len(sells) > 0 and 'profit' in sells.columns:
                total_profit = sells['profit'].sum()
                win_trades = len(sells[sells['profit'] > 0])
                win_rate = win_trades / len(sells) * 100 if len(sells) > 0 else 0
                
                print(f"  总盈亏: {total_profit:+.2f}")
                print(f"  胜率: {win_rate:.1f}% ({win_trades}/{len(sells)})")
    
    def export_trades(self, filename='data/trades_log.csv'):
        """导出交易记录"""
        if not self.trades:
            print("没有交易记录")
            return
        
        df = pd.DataFrame(self.trades)
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        print(f"\n✅ 交易记录已导出: {filename}")
    
    def save_portfolio(self, filename='data/portfolio.json'):
        """保存持仓"""
        data = {
            'initial_capital': self.initial_capital,
            'cash': self.cash,
            'positions': self.positions,
            'update_time': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 持仓已保存: {filename}")
    
    def load_portfolio(self, filename='data/portfolio.json'):
        """加载持仓"""
        if not Path(filename).exists():
            print("没有找到持仓文件")
            return
        
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.cash = data['cash']
        self.positions = data['positions']
        
        print(f"✅ 持仓已加载")
        self.summary()

def main():
    """主函数"""
    print("="*70)
    print("交易监控系统")
    print("="*70)
    
    # 创建监控器
    monitor = TradingMonitor(initial_capital=100000)
    
    # 生成每日报告
    monitor.daily_report()
    
    # 示例：模拟交易
    print("\n" + "="*70)
    print("模拟交易示例")
    print("="*70)
    
    # 检查是否需要加载已有持仓
    if Path('data/portfolio.json').exists():
        print("\n发现已有持仓，是否加载？(y/n): ", end='')
        # 这里自动加载
        monitor.load_portfolio()
    else:
        # 检查信号
        print("\n【建议操作】")
        for code in ['300750', '601318']:
            signal = monitor.check_signal(code)
            if signal:
                if signal['signal'] == 'BUY':
                    print(f"  💡 {signal['name']}: 建议买入")
                    # 计算建议买入量
                    target_value = monitor.initial_capital * monitor.stock_pool[code]['target_weight']
                    suggested_shares = int(target_value / signal['close'] / 100) * 100  # 整手
                    print(f"      建议数量: {suggested_shares}股")
                    print(f"      建议价格: {signal['close']:.2f}")
                elif signal['signal'] == 'SELL':
                    print(f"  💡 {signal['name']}: 建议卖出")
                else:
                    print(f"  ⚪ {signal['name']}: 持有/观望")
    
    # 保存持仓
    monitor.save_portfolio()
    
    # 导出交易记录
    if monitor.trades:
        monitor.export_trades()
    
    print("\n" + "="*70)
    print("使用说明")
    print("="*70)
    print("""
1. 买入: monitor.buy('300750', 200, 180.00, '2026-03-05', 'MA金叉')
2. 卖出: monitor.sell('300750', 100, 190.00, '2026-03-10', '止盈50%')
3. 汇总: monitor.summary()
4. 报告: monitor.daily_report()
5. 保存: monitor.save_portfolio()
6. 导出: monitor.export_trades()
    """)

if __name__ == '__main__':
    main()
