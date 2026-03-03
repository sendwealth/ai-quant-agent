"""
实时模拟交易系统 v2.0
====================
功能:
1. 实时数据获取 (AkShare)
2. 多策略组合 (趋势+均值回归+动量)
3. 智能仓位管理
4. 风险控制
5. 自动轮动
6. 性能监控

使用方法:
python examples/live_paper_trading.py --stock 600519 --capital 100000 --days 30
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.astock_fetcher import AStockDataFetcher
from utils.indicators import sma, ema, rsi, macd, bollinger_bands, atr, adx


class StrategyType(Enum):
    TREND = "trend"           # 趋势跟踪
    MEAN_REVERSION = "mr"     # 均值回归
    MOMENTUM = "momentum"     # 动量
    COMBO = "combo"           # 组合策略


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    shares: int
    entry_price: float
    entry_time: str
    stop_loss: float
    take_profit: float
    trailing_stop: float
    highest_price: float
    strategy: str
    capital_used: float


@dataclass
class Trade:
    """交易记录"""
    timestamp: str
    symbol: str
    action: str  # buy/sell
    shares: int
    price: float
    pnl: float
    reason: str
    strategy: str


class TrendStrategy:
    """趋势跟踪策略"""
    
    def __init__(self):
        self.name = "trend"
        self.params = {
            'ma_fast': 5,
            'ma_slow': 20,
            'adx_threshold': 20,
            'atr_stop': 2.0,
            'atr_trail': 1.5,
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """生成趋势信号"""
        close = df['close']
        
        # 计算指标
        ma_fast = sma(close, self.params['ma_fast'])
        ma_slow = sma(close, self.params['ma_slow'])
        adx_val = adx(df['high'], df['low'], close, 14)
        atr_val = atr(df['high'], df['low'], close, 14)
        
        # 趋势判断
        uptrend = ma_fast.iloc[-1] > ma_slow.iloc[-1]
        trend_strength = adx_val.iloc[-1]
        
        signal = 0
        reason = ""
        
        if uptrend and trend_strength > self.params['adx_threshold']:
            signal = 1
            reason = f"上升趋势(ADX={trend_strength:.1f})"
        elif not uptrend and trend_strength > self.params['adx_threshold']:
            signal = -1
            reason = f"下降趋势(ADX={trend_strength:.1f})"
        
        # 计算止损止盈
        price = close.iloc[-1]
        stop_loss = price - atr_val.iloc[-1] * self.params['atr_stop']
        take_profit = price + atr_val.iloc[-1] * self.params['atr_stop'] * 2
        
        return {
            'signal': signal,
            'strength': trend_strength / 50,  # 归一化
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }


class MeanReversionStrategy:
    """均值回归策略"""
    
    def __init__(self):
        self.name = "mean_reversion"
        self.params = {
            'rsi_period': 14,
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'bb_period': 20,
            'bb_std': 2.0,
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """生成均值回归信号"""
        close = df['close']
        
        # 计算指标
        rsi_val = rsi(close, self.params['rsi_period'])
        upper, middle, lower = bollinger_bands(close, self.params['bb_period'], self.params['bb_std'])
        
        current_rsi = rsi_val.iloc[-1]
        price = close.iloc[-1]
        
        signal = 0
        reason = ""
        
        # 超卖 + 接近布林下轨 = 买入
        if current_rsi < self.params['rsi_oversold'] and price < lower.iloc[-1] * 1.02:
            signal = 1
            reason = f"超卖(RSI={current_rsi:.1f}, 触及布林下轨)"
        # 超买 + 接近布林上轨 = 卖出
        elif current_rsi > self.params['rsi_overbought'] and price > upper.iloc[-1] * 0.98:
            signal = -1
            reason = f"超买(RSI={current_rsi:.1f}, 触及布林上轨)"
        
        # 止损止盈基于布林带
        stop_loss = lower.iloc[-1] * 0.95 if signal == 1 else upper.iloc[-1] * 1.05
        take_profit = middle.iloc[-1]  # 回归到中轨
        
        return {
            'signal': signal,
            'strength': abs(50 - current_rsi) / 50,
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }


class MomentumStrategy:
    """动量策略"""
    
    def __init__(self):
        self.name = "momentum"
        self.params = {
            'lookback': 10,
            'momentum_threshold': 0.03,  # 3%动量
            'volume_ma': 20,
            'volume_mult': 1.5,
        }
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """生成动量信号"""
        close = df['close']
        volume = df['volume']
        
        # 计算动量
        lookback = self.params['lookback']
        momentum = (close.iloc[-1] - close.iloc[-lookback]) / close.iloc[-lookback]
        
        # 成交量确认
        vol_ma = volume.rolling(self.params['volume_ma']).mean().iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_ma
        
        signal = 0
        reason = ""
        
        if momentum > self.params['momentum_threshold'] and vol_ratio > self.params['volume_mult']:
            signal = 1
            reason = f"上涨动量({momentum*100:.1f}%, 放量{vol_ratio:.1f}x)"
        elif momentum < -self.params['momentum_threshold'] and vol_ratio > self.params['volume_mult']:
            signal = -1
            reason = f"下跌动量({momentum*100:.1f}%, 放量{vol_ratio:.1f}x)"
        
        # ATR止损
        atr_val = atr(df['high'], df['low'], close, 14).iloc[-1]
        price = close.iloc[-1]
        stop_loss = price - atr_val * 2 if signal == 1 else price + atr_val * 2
        take_profit = price + atr_val * 3 if signal == 1 else price - atr_val * 3
        
        return {
            'signal': signal,
            'strength': abs(momentum) / 0.1,  # 归一化
            'reason': reason,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }


class ComboStrategy:
    """组合策略 - 多策略投票"""
    
    def __init__(self):
        self.name = "combo"
        self.strategies = [
            TrendStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy()
        ]
        self.weights = [0.4, 0.3, 0.3]  # 趋势40%, 均值回归30%, 动量30%
    
    def generate_signal(self, df: pd.DataFrame) -> Dict:
        """组合信号"""
        signals = []
        
        for strategy in self.strategies:
            sig = strategy.generate_signal(df)
            sig['strategy'] = strategy.name
            signals.append(sig)
        
        # 加权投票
        total_signal = sum(s['signal'] * self.weights[i] * s['strength'] 
                          for i, s in enumerate(signals))
        
        # 综合决策
        if total_signal > 0.2:
            final_signal = 1
        elif total_signal < -0.2:
            final_signal = -1
        else:
            final_signal = 0
        
        # 汇总原因
        reasons = [f"{s['strategy']}:{s['signal']:+d}" 
                  for s in signals if s['signal'] != 0]
        
        # 使用最强信号的止损止盈
        strongest = max(signals, key=lambda x: abs(x['signal']) * x['strength'])
        
        return {
            'signal': final_signal,
            'strength': abs(total_signal),
            'reason': " | ".join(reasons) if reasons else "无明确信号",
            'stop_loss': strongest['stop_loss'],
            'take_profit': strongest['take_profit'],
            'sub_signals': signals
        }


class PaperTradingEngine:
    """模拟交易引擎"""
    
    def __init__(self, initial_capital: float = 100000, strategy: str = "combo"):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve = [initial_capital]
        
        self.fetcher = AStockDataFetcher()
        
        # 选择策略
        if strategy == "trend":
            self.strategy = TrendStrategy()
        elif strategy == "mean_reversion":
            self.strategy = MeanReversionStrategy()
        elif strategy == "momentum":
            self.strategy = MomentumStrategy()
        else:
            self.strategy = ComboStrategy()
        
        # 风控参数
        self.max_position_ratio = 0.3  # 单只股票最大30%仓位
        self.max_total_position = 0.8  # 总仓位最大80%
        self.max_daily_loss = 0.05     # 单日最大亏损5%
        
        # 追踪变量
        self.daily_start_equity = initial_capital
        self.last_trade_date = None
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """计算总权益"""
        position_value = sum(
            pos.shares * prices.get(symbol, pos.entry_price)
            for symbol, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def update_stops(self, symbol: str, current_price: float, atr: float):
        """更新追踪止损"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        
        # 更新最高价
        if current_price > pos.highest_price:
            pos.highest_price = current_price
        
        # 更新追踪止损 (基于ATR)
        new_trailing_stop = pos.highest_price - atr * 1.5
        if new_trailing_stop > pos.trailing_stop:
            pos.trailing_stop = new_trailing_stop
            pos.stop_loss = max(pos.stop_loss, pos.trailing_stop)
    
    def check_stops(self, symbol: str, current_price: float) -> Optional[Trade]:
        """检查止损止盈"""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # 止损
        if current_price <= pos.stop_loss:
            pnl = (current_price - pos.entry_price) / pos.entry_price
            trade = Trade(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                action='sell',
                shares=pos.shares,
                price=current_price,
                pnl=pnl,
                reason=f"止损触发(¥{pos.stop_loss:.2f})",
                strategy=pos.strategy
            )
            self.cash += pos.shares * current_price
            del self.positions[symbol]
            self.trades.append(trade)
            return trade
        
        # 止盈
        if current_price >= pos.take_profit:
            pnl = (current_price - pos.entry_price) / pos.entry_price
            trade = Trade(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                action='sell',
                shares=pos.shares,
                price=current_price,
                pnl=pnl,
                reason=f"止盈触发(¥{pos.take_profit:.2f})",
                strategy=pos.strategy
            )
            self.cash += pos.shares * current_price
            del self.positions[symbol]
            self.trades.append(trade)
            return trade
        
        return None
    
    def execute_signal(self, symbol: str, signal: Dict, current_price: float, 
                       atr: float, df: pd.DataFrame):
        """执行交易信号"""
        
        # 检查日亏损限制
        current_equity = self.get_equity({symbol: current_price})
        daily_pnl = (current_equity - self.daily_start_equity) / self.daily_start_equity
        if daily_pnl < -self.max_daily_loss:
            return None
        
        # 买入信号
        if signal['signal'] == 1 and symbol not in self.positions:
            # 检查仓位限制
            total_position = sum(p.capital_used for p in self.positions.values())
            if total_position / self.initial_capital > self.max_total_position:
                return None
            
            # 计算仓位
            max_position_value = self.cash * self.max_position_ratio
            position_value = max_position_value * min(signal['strength'], 1.0)
            shares = int(position_value / current_price)
            
            if shares > 0 and self.cash >= shares * current_price:
                self.cash -= shares * current_price
                self.positions[symbol] = Position(
                    symbol=symbol,
                    shares=shares,
                    entry_price=current_price,
                    entry_time=datetime.now().isoformat(),
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    trailing_stop=signal['stop_loss'],
                    highest_price=current_price,
                    strategy=self.strategy.name,
                    capital_used=shares * current_price
                )
                
                trade = Trade(
                    timestamp=datetime.now().isoformat(),
                    symbol=symbol,
                    action='buy',
                    shares=shares,
                    price=current_price,
                    pnl=0,
                    reason=signal['reason'],
                    strategy=self.strategy.name
                )
                self.trades.append(trade)
                return trade
        
        # 卖出信号
        elif signal['signal'] == -1 and symbol in self.positions:
            pos = self.positions[symbol]
            pnl = (current_price - pos.entry_price) / pos.entry_price
            
            trade = Trade(
                timestamp=datetime.now().isoformat(),
                symbol=symbol,
                action='sell',
                shares=pos.shares,
                price=current_price,
                pnl=pnl,
                reason=signal['reason'],
                strategy=self.strategy.name
            )
            
            self.cash += pos.shares * current_price
            del self.positions[symbol]
            self.trades.append(trade)
            return trade
        
        return None
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str, 
                     verbose: bool = True) -> Dict:
        """运行回测"""
        
        # 获取数据
        df = self.fetcher.fetch_stock_daily(symbol, start_date, end_date)
        if df is None or len(df) < 100:
            print(f"⚠️ 数据不足: {symbol}")
            return None
        
        print(f"\n{'='*70}")
        print(f"📊 回测: {symbol}")
        print(f"策略: {self.strategy.name}")
        print(f"时间: {df['datetime'].iloc[0]} - {df['datetime'].iloc[-1]}")
        print(f"数据: {len(df)} 条")
        print("="*70)
        
        # 回测循环
        for i in range(100, len(df)):
            current_df = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            current_date = str(df['datetime'].iloc[i])[:10]
            
            # 重置日亏损追踪
            if self.last_trade_date != current_date:
                self.daily_start_equity = self.get_equity({symbol: current_price})
                self.last_trade_date = current_date
            
            # 计算ATR
            atr_val = atr(current_df['high'], current_df['low'], 
                         current_df['close'], 14).iloc[-1]
            
            # 更新追踪止损
            self.update_stops(symbol, current_price, atr_val)
            
            # 检查止损止盈
            stop_trade = self.check_stops(symbol, current_price)
            if stop_trade and verbose:
                print(f"[{current_date}] {stop_trade.action}: {stop_trade.shares}股 "
                      f"@ ¥{stop_trade.price:.2f} | {stop_trade.reason} | PnL: {stop_trade.pnl*100:+.2f}%")
            
            # 生成信号
            signal = self.strategy.generate_signal(current_df)
            
            # 执行交易
            trade = self.execute_signal(symbol, signal, current_price, atr_val, current_df)
            if trade and verbose:
                print(f"[{current_date}] {trade.action}: {trade.shares}股 "
                      f"@ ¥{trade.price:.2f} | {trade.reason}")
            
            # 记录权益
            equity = self.get_equity({symbol: current_price})
            self.equity_curve.append(equity)
        
        # 最终平仓
        if symbol in self.positions:
            pos = self.positions[symbol]
            final_price = df['close'].iloc[-1]
            pnl = (final_price - pos.entry_price) / pos.entry_price
            
            trade = Trade(
                timestamp=str(df['datetime'].iloc[-1]),
                symbol=symbol,
                action='sell',
                shares=pos.shares,
                price=final_price,
                pnl=pnl,
                reason="回测结束平仓",
                strategy=pos.strategy
            )
            self.cash += pos.shares * final_price
            del self.positions[symbol]
            self.trades.append(trade)
            
            if verbose:
                print(f"[最终] 平仓: {trade.shares}股 @ ¥{trade.price:.2f} | PnL: {trade.pnl*100:+.2f}%")
        
        # 计算性能
        return self._calculate_performance(symbol)
    
    def _calculate_performance(self, symbol: str) -> Dict:
        """计算性能指标"""
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        
        days = len(self.equity_curve)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        trades_list = [asdict(t) for t in self.trades]
        win_trades = [t for t in trades_list if t.get('pnl', 0) > 0]
        loss_trades = [t for t in trades_list if t.get('pnl', 0) <= 0 and t.get('action') == 'sell']
        
        win_rate = len(win_trades) / len([t for t in trades_list if t.get('action') == 'sell']) if trades_list else 0
        
        return {
            'symbol': symbol,
            'strategy': self.strategy.name,
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'total_trades': len([t for t in trades_list if t.get('action') == 'buy']),
            'win_rate': win_rate,
            'trades': trades_list,
            'equity_curve': self.equity_curve
        }
    
    def print_report(self, result: Dict):
        """打印报告"""
        print("\n" + "="*70)
        print("📊 回测报告")
        print("="*70)
        print(f"股票: {result['symbol']}")
        print(f"策略: {result['strategy']}")
        print(f"初始资金: ¥{result['initial_capital']:,.2f}")
        print(f"最终权益: ¥{result['final_equity']:,.2f}")
        print(f"总收益: {result['total_return']*100:+.2f}%")
        print(f"年化收益: {result['annual_return']*100:+.2f}%")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"波动率: {result['volatility']*100:.2f}%")
        print(f"交易次数: {result['total_trades']}")
        print(f"胜率: {result['win_rate']*100:.1f}%")
        
        # 评级
        if result['sharpe_ratio'] > 1.5 and result['max_drawdown'] > -0.1:
            grade = "A 🏆 优秀"
        elif result['sharpe_ratio'] > 1.0 and result['max_drawdown'] > -0.15:
            grade = "B ✅ 良好"
        elif result['sharpe_ratio'] > 0.5 and result['max_drawdown'] > -0.2:
            grade = "C ⚠️ 一般"
        else:
            grade = "D ❌ 需优化"
        
        print(f"\n策略评级: {grade}")
        print("="*70)


def run_multi_stock_backtest(symbols: List[str], strategy: str = "combo",
                             initial_capital: float = 100000):
    """多股票回测"""
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')
    
    results = []
    
    for symbol in symbols:
        engine = PaperTradingEngine(initial_capital, strategy)
        result = engine.run_backtest(symbol, start_date, end_date, verbose=False)
        
        if result:
            results.append(result)
            print(f"\n{symbol}: 收益{result['total_return']*100:+.2f}%, "
                  f"夏普{result['sharpe_ratio']:.2f}, "
                  f"胜率{result['win_rate']*100:.1f}%")
    
    if results:
        print("\n" + "="*70)
        print("📈 多股票汇总")
        print("="*70)
        
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        win_count = sum(1 for r in results if r['total_return'] > 0)
        
        print(f"测试股票: {len(results)}只")
        print(f"盈利股票: {win_count}只 ({win_count/len(results)*100:.0f}%)")
        print(f"平均收益: {avg_return*100:+.2f}%")
        print(f"平均夏普: {avg_sharpe:.2f}")
        print(f"平均回撤: {avg_drawdown*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='AI量化交易模拟系统')
    parser.add_argument('--stock', type=str, default='600519', help='股票代码')
    parser.add_argument('--capital', type=float, default=100000, help='初始资金')
    parser.add_argument('--strategy', type=str, default='combo', 
                        choices=['trend', 'mean_reversion', 'momentum', 'combo'],
                        help='策略类型')
    parser.add_argument('--days', type=int, default=365, help='回测天数')
    parser.add_argument('--multi', action='store_true', help='多股票测试')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🚀 AI量化交易模拟系统 v2.0")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.multi:
        # 多股票测试
        symbols = ['600519', '000858', '600036', '601318', '000001', 
                   '002594', '600276', '601012', '600900', '000725']
        run_multi_stock_backtest(symbols, args.strategy, args.capital)
    else:
        # 单股票测试
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=args.days)).strftime('%Y%m%d')
        
        engine = PaperTradingEngine(args.capital, args.strategy)
        result = engine.run_backtest(args.stock, start_date, end_date, verbose=True)
        
        if result:
            engine.print_report(result)
            
            # 保存结果
            output_file = f"backtest_{args.stock}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'symbol': result['symbol'],
                    'strategy': result['strategy'],
                    'total_return': result['total_return'],
                    'annual_return': result['annual_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown'],
                    'win_rate': result['win_rate'],
                    'total_trades': result['total_trades']
                }, f, ensure_ascii=False, indent=2)
            print(f"\n结果已保存: {output_file}")


if __name__ == "__main__":
    main()
