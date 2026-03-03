"""
实盘模拟交易系统
================
- 使用最优保守策略
- 支持多股票轮动
- 实时监控和报告
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import time

# ============ 配置 ============
INITIAL_CAPITAL = 100000  # 初始资金10万
MAX_POSITION = 0.25       # 最大25%仓位
MA_SHORT = 5
MA_LONG = 30
ATR_STOP = 2.5
ATR_TRAIL = 2.0

# ============ 工具函数 ============
def sma(data, period): 
    return data.rolling(window=period).mean()

def atr(high, low, close, period=14):
    tr = pd.concat([high-low, abs(high-close.shift()), abs(low-close.shift())], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def load_data(filepath):
    if not Path(filepath).exists():
        return None
    df = pd.read_csv(filepath)
    df = df.rename(columns={'日期': 'datetime', '开盘': 'open', '最高': 'high',
                            '最低': 'low', '收盘': 'close', '成交量': 'volume'})
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df.sort_values('datetime').reset_index(drop=True)

# ============ 交易引擎 ============
class PaperTrader:
    """模拟交易引擎"""
    
    def __init__(self, capital=INITIAL_CAPITAL):
        self.initial_capital = capital
        self.cash = capital
        self.positions = {}  # {symbol: {shares, entry, stop, highest}}
        self.trades = []
        self.equity_curve = [capital]
        
    def get_equity(self, prices):
        """计算总权益"""
        pos_value = sum(
            self.positions[s]['shares'] * prices.get(s, self.positions[s]['entry'])
            for s in self.positions
        )
        return self.cash + pos_value
    
    def process_bar(self, symbol, df, i):
        """处理单根K线"""
        price = float(df['close'].iloc[i])
        high = float(df['high'].iloc[i])
        low = float(df['low'].iloc[i])
        
        # 计算指标
        atr_val = float(atr(df['high'], df['low'], df['close'], 14).iloc[i])
        ma_s = float(sma(df['close'], MA_SHORT).iloc[i])
        ma_l = float(sma(df['close'], MA_LONG).iloc[i])
        
        if pd.isna(ma_s) or pd.isna(ma_l) or pd.isna(atr_val):
            return
        
        # 更新持仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            
            # 更新最高价和追踪止损
            if high > pos['highest']:
                pos['highest'] = high
                new_stop = pos['highest'] - atr_val * ATR_TRAIL
                if new_stop > pos['stop']:
                    pos['stop'] = new_stop
            
            # 止损
            if low <= pos['stop']:
                pnl = (price - pos['entry']) / pos['entry']
                self.trades.append({
                    'symbol': symbol, 'action': 'sell', 'type': 'stop',
                    'price': price, 'shares': pos['shares'], 'pnl': pnl
                })
                self.cash += pos['shares'] * price
                del self.positions[symbol]
                return
            
            # 信号卖出
            if ma_s < ma_l:
                pnl = (price - pos['entry']) / pos['entry']
                self.trades.append({
                    'symbol': symbol, 'action': 'sell', 'type': 'signal',
                    'price': price, 'shares': pos['shares'], 'pnl': pnl
                })
                self.cash += pos['shares'] * price
                del self.positions[symbol]
        
        # 买入
        elif ma_s > ma_l and symbol not in self.positions:
            shares = int(self.cash * MAX_POSITION / price)
            if shares > 0:
                cost = shares * price
                if cost <= self.cash:
                    self.cash -= cost
                    self.positions[symbol] = {
                        'shares': shares,
                        'entry': price,
                        'stop': price - atr_val * ATR_STOP,
                        'highest': high
                    }
                    self.trades.append({
                        'symbol': symbol, 'action': 'buy',
                        'price': price, 'shares': shares
                    })
    
    def run_backtest(self, data_files):
        """运行回测"""
        # 加载所有数据
        all_data = {}
        max_len = 0
        
        for symbol, filepath in data_files.items():
            df = load_data(filepath)
            if df is not None and len(df) > 50:
                all_data[symbol] = df
                max_len = max(max_len, len(df))
        
        if not all_data:
            print("没有有效数据")
            return None
        
        print(f"\n加载数据: {len(all_data)}只股票")
        
        # 回测
        for i in range(50, max_len):
            prices = {}
            
            for symbol, df in all_data.items():
                if i < len(df):
                    self.process_bar(symbol, df, i)
                    prices[symbol] = float(df['close'].iloc[i])
            
            self.equity_curve.append(self.get_equity(prices))
        
        # 平仓
        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            final_price = float(all_data[symbol]['close'].iloc[-1])
            pnl = (final_price - pos['entry']) / pos['entry']
            self.cash += pos['shares'] * final_price
            self.trades.append({
                'symbol': symbol, 'action': 'sell', 'type': 'final',
                'price': final_price, 'shares': pos['shares'], 'pnl': pnl
            })
        
        return self._calculate_performance()
    
    def _calculate_performance(self):
        """计算性能"""
        final = self.equity_curve[-1]
        ret = (final - self.initial_capital) / self.initial_capital
        
        eq = pd.Series(self.equity_curve)
        rets = eq.pct_change().dropna()
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
        
        peak = eq.expanding().max()
        dd = (eq - peak) / peak
        max_dd = dd.min()
        
        sells = [t for t in self.trades if t['action'] == 'sell']
        wins = [t for t in sells if t.get('pnl', 0) > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        return {
            'final_equity': final,
            'total_return': ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'total_trades': len([t for t in self.trades if t['action'] == 'buy']),
            'win_rate': win_rate,
            'trades': self.trades
        }

def print_report(result):
    """打印报告"""
    print("\n" + "="*60)
    print("📊 实盘模拟报告")
    print("="*60)
    print(f"初始资金: ¥100,000")
    print(f"最终权益: ¥{result['final_equity']:,.2f}")
    print(f"总收益: {result['total_return']*100:+.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"最大回撤: {result['max_drawdown']*100:.1f}%")
    print(f"交易次数: {result['total_trades']}")
    print(f"胜率: {result['win_rate']*100:.0f}%")
    
    # 评级
    if result['sharpe_ratio'] > 1.0 and result['max_drawdown'] > -0.10:
        grade = "A 🏆"
    elif result['sharpe_ratio'] > 0.5 and result['max_drawdown'] > -0.15:
        grade = "B ✅"
    elif result['sharpe_ratio'] > 0:
        grade = "C ⚠️"
    else:
        grade = "D ❌"
    
    print(f"\n策略评级: {grade}")
    print("="*60)
    
    # 交易明细
    print("\n交易记录:")
    for t in result['trades'][-10:]:
        if t['action'] == 'buy':
            print(f"  买入 {t['symbol']}: {t['shares']}股 @ ¥{t['price']:.2f}")
        else:
            print(f"  {t['type']} {t['symbol']}: {t['shares']}股 @ ¥{t['price']:.2f} | PnL: {t['pnl']*100:+.2f}%")

def main():
    """主函数"""
    print("="*60)
    print("🚀 实盘模拟交易系统")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n策略参数:")
    print(f"  均线: {MA_SHORT}/{MA_LONG}")
    print(f"  ATR止损: {ATR_STOP}x")
    print(f"  最大仓位: {MAX_POSITION*100:.0f}%")
    
    # 数据文件
    data_files = {
        '000858': 'data/real_000858.csv',
        '002594': 'data/real_002594.csv',
    }
    
    # 运行
    trader = PaperTrader()
    result = trader.run_backtest(data_files)
    
    if result:
        print_report(result)
        
        # 保存
        with open('paper_trading_result.json', 'w', encoding='utf-8') as f:
            output = {
                'timestamp': datetime.now().isoformat(),
                'summary': {k: v for k, v in result.items() if k != 'trades'},
                'trades': result['trades'][-20:]  # 最近20条
            }
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print("\n结果已保存: paper_trading_result.json")

if __name__ == "__main__":
    main()
