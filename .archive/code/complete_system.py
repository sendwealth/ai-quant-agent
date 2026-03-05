"""
完整量化交易系统 v5.0
=====================
- 使用模拟数据验证策略
- 支持切换到真实数据
- 完整的回测+实盘模拟框架
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ============== 数据模块 ==============

class DataSource(Enum):
    MOCK = "mock"
    AKSHARE = "akshare"


def generate_trending_data(n_days: int, trend: str = "up", volatility: float = 0.02) -> pd.DataFrame:
    """生成趋势性模拟数据"""
    dates = pd.date_range(end=datetime.now(), periods=n_days, freq='B')
    
    np.random.seed(42)
    base_price = 50.0
    
    if trend == "up":
        drift = 0.0008  # 正向漂移
    elif trend == "down":
        drift = -0.0005
    else:  # sideways
        drift = 0.0
    
    returns = np.random.normal(drift, volatility, len(dates))
    prices = base_price * np.exp(np.cumsum(returns))
    
    # 添加一些波动
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, len(dates)))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, len(dates)))
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return df


def fetch_data(symbol: str, start_date: str, end_date: str, 
               source: DataSource = DataSource.MOCK) -> pd.DataFrame:
    """获取数据"""
    if source == DataSource.MOCK:
        return generate_trending_data(500, trend=random.choice(["up", "sideways", "up"]))
    
    # AkShare
    try:
        import akshare as ak
        df = ak.stock_zh_a_hist(symbol=symbol, period='daily',
                                start_date=start_date, end_date=end_date, adjust='qfq')
        df = df.rename(columns={
            '日期': 'datetime', '开盘': 'open', '最高': 'high',
            '最低': 'low', '收盘': 'close', '成交量': 'volume'
        })
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime').reset_index(drop=True)
    except Exception as e:
        print(f"获取失败: {e}，使用模拟数据")
        return generate_trending_data(500, "up")


# ============== 指标模块 ==============

def sma(data, period): return data.rolling(window=period).mean()
def ema(data, period): return data.ewm(span=period, adjust=False).mean()

def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    return 100 - (100 / (1 + gain / loss))

def macd(data, fast=12, slow=26, signal=9):
    ema_f = ema(data, fast)
    ema_s = ema(data, slow)
    macd_line = ema_f - ema_s
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line, macd_line - signal_line

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def bollinger(data, period=20, std=2):
    mid = sma(data, period)
    std_val = data.rolling(window=period).std()
    return mid + std_val * std, mid, mid - std_val * std

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()
    
    plus_di = 100 * plus_dm.rolling(window=period).mean() / atr_val
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr_val
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    return dx.rolling(window=period).mean()


# ============== 策略模块 ==============

@dataclass
class StrategyConfig:
    """策略配置"""
    ma_short: int = 5
    ma_long: int = 20
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    atr_stop: float = 2.0
    atr_trail: float = 1.5
    adx_threshold: int = 20
    volume_mult: float = 1.2
    max_position: float = 0.3
    min_signal_strength: float = 0.5


class TradingStrategy:
    """交易策略"""
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """分析市场"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 计算指标
        ma_s = sma(close, self.config.ma_short)
        ma_l = sma(close, self.config.ma_long)
        rsi_val = rsi(close, 14)
        macd_line, signal_line, hist = macd(close)
        atr_val = atr(high, low, close, 14)
        bb_upper, bb_mid, bb_lower = bollinger(close)
        adx_val = adx(high, low, close, 14)
        vol_ma = volume.rolling(20).mean()
        
        price = close.iloc[-1]
        
        # 信号计算
        signals = {}
        
        # 1. 趋势信号
        if ma_s.iloc[-1] > ma_l.iloc[-1]:
            signals['trend'] = 1
        else:
            signals['trend'] = -1
        
        # 2. RSI信号
        if rsi_val.iloc[-1] < self.config.rsi_oversold:
            signals['rsi'] = 1
        elif rsi_val.iloc[-1] > self.config.rsi_overbought:
            signals['rsi'] = -1
        else:
            signals['rsi'] = 0
        
        # 3. MACD信号
        if hist.iloc[-1] > 0:
            signals['macd'] = 1
        else:
            signals['macd'] = -1
        
        # 4. 布林带位置
        bb_pos = (price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        if bb_pos < 0.2:
            signals['bollinger'] = 1
        elif bb_pos > 0.8:
            signals['bollinger'] = -1
        else:
            signals['bollinger'] = 0
        
        # 5. ADX趋势强度
        signals['adx'] = adx_val.iloc[-1]
        signals['adx_strong'] = adx_val.iloc[-1] > self.config.adx_threshold
        
        # 6. 成交量确认
        signals['volume_confirmed'] = volume.iloc[-1] > vol_ma.iloc[-1] * self.config.volume_mult
        
        # 综合信号
        buy_score = 0
        sell_score = 0
        
        if signals['trend'] == 1 and signals['adx_strong']:
            buy_score += 2
        if signals['rsi'] == 1:
            buy_score += 1
        if signals['macd'] == 1:
            buy_score += 1
        if signals['bollinger'] == 1:
            buy_score += 1
        if signals['volume_confirmed']:
            buy_score += 0.5
        
        if signals['trend'] == -1:
            sell_score += 2
        if signals['rsi'] == -1:
            sell_score += 1
        if signals['macd'] == -1:
            sell_score += 1
        if signals['bollinger'] == -1:
            sell_score += 1
        
        # 最终决策
        if buy_score >= 3:
            action = 'buy'
            strength = min(buy_score / 5.5, 1.0)
        elif sell_score >= 2:
            action = 'sell'
            strength = min(sell_score / 5, 1.0)
        else:
            action = 'hold'
            strength = 0
        
        # 止损止盈
        stop_loss = price - atr_val.iloc[-1] * self.config.atr_stop
        take_profit = price + atr_val.iloc[-1] * self.config.atr_stop * 2
        
        return {
            'action': action,
            'strength': strength,
            'signals': signals,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': atr_val.iloc[-1],
            'price': price
        }


# ============== 交易引擎 ==============

class TradingEngine:
    """交易引擎"""
    
    def __init__(self, initial_capital: float = 100000, config: StrategyConfig = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trailing_stop = 0
        self.highest_price = 0
        
        self.strategy = TradingStrategy(config)
        self.trades = []
        self.equity_curve = [initial_capital]
        
        # 风控
        self.max_daily_loss = 0.03
        self.daily_start_equity = initial_capital
        self.cooldown = 0
    
    def get_equity(self, price: float) -> float:
        return self.cash + self.position * price
    
    def run_backtest(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """回测"""
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 回测开始")
            print(f"数据: {len(df)} 条")
            print(f"时间: {df['datetime'].iloc[0]} - {df['datetime'].iloc[-1]}")
            print("="*60)
        
        for i in range(60, len(df)):
            current_df = df.iloc[:i+1]
            price = df['close'].iloc[i]
            date = str(df['datetime'].iloc[i])[:10]
            
            # 冷却期
            if self.cooldown > 0:
                self.cooldown -= 1
                self.equity_curve.append(self.get_equity(price))
                continue
            
            # 更新追踪止损
            if self.position > 0:
                if price > self.highest_price:
                    self.highest_price = price
                    new_trail = self.highest_price - self.strategy.config.atr_trail * \
                                atr(df['high'].iloc[:i+1], df['low'].iloc[:i+1], 
                                    df['close'].iloc[:i+1], 14).iloc[-1]
                    self.trailing_stop = max(self.trailing_stop, new_trail)
            
            # 止损检查
            if self.position > 0 and (price <= self.stop_loss or price <= self.trailing_stop):
                pnl = (price - self.entry_price) / self.entry_price
                self.trades.append({
                    'date': date, 'action': 'sell', 'type': 'stop',
                    'price': price, 'shares': self.position, 'pnl': pnl
                })
                if verbose:
                    print(f"[{date}] 止损卖出: {self.position}股 @ ¥{price:.2f} | PnL: {pnl*100:+.2f}%")
                self.cash += self.position * price
                self.position = 0
                self.cooldown = 3
                continue
            
            # 止盈检查
            if self.position > 0 and price >= self.take_profit:
                pnl = (price - self.entry_price) / self.entry_price
                self.trades.append({
                    'date': date, 'action': 'sell', 'type': 'profit',
                    'price': price, 'shares': self.position, 'pnl': pnl
                })
                if verbose:
                    print(f"[{date}] 止盈卖出: {self.position}股 @ ¥{price:.2f} | PnL: {pnl*100:+.2f}%")
                self.cash += self.position * price
                self.position = 0
                self.cooldown = 2
                continue
            
            # 生成信号
            signal = self.strategy.analyze(current_df)
            
            # 买入
            if signal['action'] == 'buy' and self.position == 0 and signal['strength'] >= 0.5:
                pos_value = self.cash * self.strategy.config.max_position * signal['strength']
                shares = int(pos_value / price)
                
                if shares > 0:
                    self.position = shares
                    self.cash -= shares * price
                    self.entry_price = price
                    self.stop_loss = signal['stop_loss']
                    self.take_profit = signal['take_profit']
                    self.trailing_stop = signal['stop_loss']
                    self.highest_price = price
                    
                    if verbose:
                        print(f"[{date}] 买入: {shares}股 @ ¥{price:.2f} | "
                              f"止损:¥{self.stop_loss:.2f} | 止盈:¥{self.take_profit:.2f}")
            
            # 卖出
            elif signal['action'] == 'sell' and self.position > 0:
                pnl = (price - self.entry_price) / self.entry_price
                self.trades.append({
                    'date': date, 'action': 'sell', 'type': 'signal',
                    'price': price, 'shares': self.position, 'pnl': pnl
                })
                if verbose:
                    print(f"[{date}] 信号卖出: {self.position}股 @ ¥{price:.2f} | PnL: {pnl*100:+.2f}%")
                self.cash += self.position * price
                self.position = 0
                self.cooldown = 3
            
            self.equity_curve.append(self.get_equity(price))
        
        # 最终平仓
        if self.position > 0:
            final_price = df['close'].iloc[-1]
            pnl = (final_price - self.entry_price) / self.entry_price
            self.trades.append({
                'date': str(df['datetime'].iloc[-1])[:10], 'action': 'sell', 'type': 'final',
                'price': final_price, 'shares': self.position, 'pnl': pnl
            })
            self.cash += self.position * final_price
            if verbose:
                print(f"[最终] 平仓: {self.position}股 @ ¥{final_price:.2f} | PnL: {pnl*100:+.2f}%")
        
        return self._calculate_performance()
    
    def _calculate_performance(self) -> Dict:
        """计算性能"""
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        
        annual_return = (1 + total_return) ** (252 / len(self.equity_curve)) - 1
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()
        
        sells = [t for t in self.trades if t['action'] == 'sell']
        wins = [t for t in sells if t['pnl'] > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        return {
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'volatility': volatility,
            'total_trades': len(self.trades),
            'win_rate': win_rate,
            'trades': self.trades
        }


# ============== 优化器 ==============

class SimpleOptimizer:
    """简单参数优化器"""
    
    def __init__(self):
        self.param_combinations = [
            # 保守型
            StrategyConfig(ma_short=5, ma_long=30, atr_stop=2.5, max_position=0.2),
            # 平衡型
            StrategyConfig(ma_short=5, ma_long=20, atr_stop=2.0, max_position=0.3),
            # 激进型
            StrategyConfig(ma_short=3, ma_long=15, atr_stop=1.5, max_position=0.4),
            # RSI偏重
            StrategyConfig(ma_short=10, ma_long=30, rsi_oversold=25, rsi_overbought=75, max_position=0.25),
            # 趋势偏重
            StrategyConfig(ma_short=5, ma_long=20, adx_threshold=25, max_position=0.35),
        ]
    
    def optimize(self, df: pd.DataFrame) -> Tuple[StrategyConfig, Dict]:
        """优化"""
        best_config = None
        best_result = None
        best_score = -float('inf')
        
        for config in self.param_combinations:
            engine = TradingEngine(100000, config)
            result = engine.run_backtest(df, verbose=False)
            
            # 综合评分
            score = (
                result['total_return'] * 100 +
                result['sharpe_ratio'] * 30 +
                result['max_drawdown'] * 50 +
                result['win_rate'] * 20
            )
            
            if score > best_score:
                best_score = score
                best_config = config
                best_result = result
        
        return best_config, best_result


def print_report(result: Dict, config: StrategyConfig = None):
    """打印报告"""
    print("\n" + "="*60)
    print("📊 回测报告")
    print("="*60)
    print(f"最终权益: ¥{result['final_equity']:,.2f}")
    print(f"总收益: {result['total_return']*100:+.2f}%")
    print(f"年化收益: {result['annual_return']*100:+.2f}%")
    print(f"夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
    print(f"波动率: {result['volatility']*100:.2f}%")
    print(f"交易次数: {result['total_trades']}")
    print(f"胜率: {result['win_rate']*100:.1f}%")
    
    if config:
        print(f"\n策略参数:")
        print(f"  MA: {config.ma_short}/{config.ma_long}")
        print(f"  RSI: {config.rsi_oversold}/{config.rsi_overbought}")
        print(f"  ATR止损: {config.atr_stop}x")
        print(f"  最大仓位: {config.max_position*100:.0f}%")
    
    # 评级
    if result['sharpe_ratio'] > 1.5 and result['max_drawdown'] > -0.10:
        grade = "A 🏆"
    elif result['sharpe_ratio'] > 1.0 and result['max_drawdown'] > -0.15:
        grade = "B ✅"
    elif result['sharpe_ratio'] > 0.5 and result['max_drawdown'] > -0.20:
        grade = "C ⚠️"
    else:
        grade = "D ❌"
    
    print(f"\n策略评级: {grade}")
    print("="*60)


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 完整量化交易系统 v5.0")
    print("="*60)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 生成测试数据
    print("\n生成测试数据...")
    df = generate_trending_data(500, trend="up")
    print(f"✓ 生成 {len(df)} 条数据")
    
    # 优化
    print("\n🔍 参数优化...")
    optimizer = SimpleOptimizer()
    best_config, best_result = optimizer.optimize(df)
    
    # 用最优参数回测
    print("\n📈 使用最优参数回测...")
    engine = TradingEngine(100000, best_config)
    result = engine.run_backtest(df, verbose=True)
    
    print_report(result, best_config)
    
    # 多次测试
    print("\n" + "="*60)
    print("📊 多轮测试验证")
    print("="*60)
    
    results = []
    for i in range(5):
        df_test = generate_trending_data(500, trend=random.choice(["up", "sideways", "up"]))
        engine = TradingEngine(100000, best_config)
        result = engine.run_backtest(df_test, verbose=False)
        results.append(result)
        status = "✅" if result['total_return'] > 0 else "❌"
        print(f"{status} 轮{i+1}: 收益{result['total_return']*100:+.2f}%, 夏普{result['sharpe_ratio']:.2f}")
    
    avg_return = np.mean([r['total_return'] for r in results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
    win_count = sum(1 for r in results if r['total_return'] > 0)
    
    print(f"\n汇总: 盈利{win_count}/5轮, 平均收益{avg_return*100:+.2f}%, 平均夏普{avg_sharpe:.2f}")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'best_config': asdict(best_config),
        'results': [{
            'total_return': r['total_return'],
            'sharpe_ratio': r['sharpe_ratio'],
            'max_drawdown': r['max_drawdown'],
            'win_rate': r['win_rate']
        } for r in results]
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n结果已保存: backtest_results.json")


if __name__ == "__main__":
    main()
