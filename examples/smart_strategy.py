"""
智能自适应策略 v3.0
===================
核心改进:
1. 市场环境识别 - 自动切换策略
2. 减少交易频率 - 避免震荡市频繁止损
3. 更严格的入场条件 - 提高胜率
4. 动态仓位管理 - 根据信号强度调整
5. 多重确认机制 - 减少假信号
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.astock_fetcher import AStockDataFetcher
from utils.indicators import sma, ema, rsi, macd, bollinger_bands, atr, adx


class MarketState(Enum):
    """市场状态"""
    STRONG_BULL = "strong_bull"      # 强势上涨
    WEAK_BULL = "weak_bull"          # 弱势上涨
    SIDEWAYS = "sideways"            # 横盘震荡
    WEAK_BEAR = "weak_bear"          # 弱势下跌
    STRONG_BEAR = "strong_bear"      # 强势下跌
    HIGH_VOLATILITY = "high_vol"     # 高波动


@dataclass
class Signal:
    """交易信号"""
    action: str          # buy, sell, hold
    strength: float      # 0-1
    confidence: float    # 0-1 信心度
    reason: str
    position_ratio: float  # 建议仓位
    stop_loss: float
    take_profit: float
    market_state: str
    risk_level: str      # low, medium, high


class MarketAnalyzer:
    """市场分析器"""
    
    def __init__(self):
        self.lookback_short = 10
        self.lookback_medium = 20
        self.lookback_long = 60
    
    def analyze(self, df: pd.DataFrame) -> MarketState:
        """分析市场状态"""
        close = df['close']
        high = df['high']
        low = df['low']
        
        # 1. 计算趋势强度 (ADX)
        adx_val = adx(high, low, close, 14).iloc[-1]
        
        # 2. 计算价格位置
        ma_20 = sma(close, 20).iloc[-1]
        ma_60 = sma(close, 60).iloc[-1]
        price = close.iloc[-1]
        
        # 3. 计算动量
        momentum_10 = (close.iloc[-1] / close.iloc[-self.lookback_short] - 1)
        momentum_20 = (close.iloc[-1] / close.iloc[-self.lookback_medium] - 1)
        
        # 4. 计算波动率
        returns = close.pct_change()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # 5. 计算RSI
        rsi_val = rsi(close, 14).iloc[-1]
        
        # 判断逻辑
        if volatility > 0.40:  # 年化波动率>40%
            return MarketState.HIGH_VOLATILITY
        
        # 趋势判断
        price_above_ma20 = price > ma_20
        price_above_ma60 = price > ma_60
        ma_bullish = ma_20 > ma_60
        
        # ADX趋势强度
        has_strong_trend = adx_val > 25
        has_moderate_trend = adx_val > 20
        
        # 综合判断
        if has_strong_trend:
            if price_above_ma20 and price_above_ma60 and ma_bullish and momentum_20 > 0.05:
                return MarketState.STRONG_BULL
            elif price_above_ma20 and ma_bullish and momentum_10 > 0:
                return MarketState.WEAK_BULL
            elif not price_above_ma20 and not price_above_ma60 and not ma_bullish and momentum_20 < -0.05:
                return MarketState.STRONG_BEAR
            elif not price_above_ma20 and not ma_bullish and momentum_10 < 0:
                return MarketState.WEAK_BEAR
        
        return MarketState.SIDEWAYS
    
    def get_state_info(self, state: MarketState) -> Dict:
        """获取状态信息"""
        info = {
            MarketState.STRONG_BULL: {
                'trend': 'up',
                'strength': 0.9,
                'strategy': 'trend_follow',
                'position_max': 0.4,
                'stop_atr': 2.5,
                'risk': 'medium'
            },
            MarketState.WEAK_BULL: {
                'trend': 'up',
                'strength': 0.6,
                'strategy': 'trend_follow',
                'position_max': 0.25,
                'stop_atr': 2.0,
                'risk': 'medium'
            },
            MarketState.SIDEWAYS: {
                'trend': 'none',
                'strength': 0.3,
                'strategy': 'mean_reversion',
                'position_max': 0.15,
                'stop_atr': 1.5,
                'risk': 'high'
            },
            MarketState.WEAK_BEAR: {
                'trend': 'down',
                'strength': -0.6,
                'strategy': 'avoid',
                'position_max': 0.1,
                'stop_atr': 2.0,
                'risk': 'high'
            },
            MarketState.STRONG_BEAR: {
                'trend': 'down',
                'strength': -0.9,
                'strategy': 'avoid',
                'position_max': 0,
                'stop_atr': 3.0,
                'risk': 'very_high'
            },
            MarketState.HIGH_VOLATILITY: {
                'trend': 'uncertain',
                'strength': 0,
                'strategy': 'reduce_size',
                'position_max': 0.15,
                'stop_atr': 2.5,
                'risk': 'high'
            }
        }
        return info.get(state, info[MarketState.SIDEWAYS])


class SignalGenerator:
    """信号生成器"""
    
    def __init__(self):
        self.analyzer = MarketAnalyzer()
        
        # 信号确认参数
        self.min_confidence = 0.6      # 最低信心度
        self.confirmation_count = 2    # 需要确认的指标数
        
    def generate(self, df: pd.DataFrame, trade_history: List = None) -> Signal:
        """生成交易信号"""
        
        # 1. 分析市场状态
        market_state = self.analyzer.analyze(df)
        state_info = self.analyzer.get_state_info(market_state)
        
        # 2. 计算技术指标
        indicators = self._calculate_indicators(df)
        
        # 3. 根据市场状态选择策略
        if state_info['strategy'] == 'avoid':
            return Signal(
                action='hold',
                strength=0,
                confidence=0.9,
                reason=f"市场环境不利({market_state.value})，建议观望",
                position_ratio=0,
                stop_loss=0,
                take_profit=0,
                market_state=market_state.value,
                risk_level=state_info['risk']
            )
        
        # 4. 生成信号
        if state_info['strategy'] == 'trend_follow':
            signal = self._trend_signal(df, indicators, state_info)
        else:  # mean_reversion
            signal = self._mean_reversion_signal(df, indicators, state_info)
        
        # 5. 多重确认
        confirmations = self._get_confirmations(df, indicators, signal.action)
        confidence = len(confirmations) / 5  # 5个确认指标
        
        # 6. 更新信号信心度
        signal.confidence = confidence
        signal.market_state = market_state.value
        
        # 7. 如果信心度不足，转为持有
        if confidence < self.min_confidence and signal.action != 'hold':
            signal.action = 'hold'
            signal.reason = f"确认不足({confidence:.0%})，暂时观望"
        
        return signal
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """计算技术指标"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 均线
        ma_5 = sma(close, 5)
        ma_10 = sma(close, 10)
        ma_20 = sma(close, 20)
        ma_60 = sma(close, 60)
        
        # RSI
        rsi_14 = rsi(close, 14)
        
        # MACD
        macd_line, signal_line, histogram = macd(close)
        
        # 布林带
        bb_upper, bb_mid, bb_lower = bollinger_bands(close, 20, 2)
        
        # ATR
        atr_14 = atr(high, low, close, 14)
        
        # ADX
        adx_14 = adx(high, low, close, 14)
        
        # 成交量
        vol_ma = volume.rolling(20).mean()
        
        return {
            'ma_5': ma_5.iloc[-1],
            'ma_10': ma_10.iloc[-1],
            'ma_20': ma_20.iloc[-1],
            'ma_60': ma_60.iloc[-1],
            'rsi': rsi_14.iloc[-1],
            'macd': macd_line.iloc[-1],
            'macd_signal': signal_line.iloc[-1],
            'macd_hist': histogram.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'bb_mid': bb_mid.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'atr': atr_14.iloc[-1],
            'adx': adx_14.iloc[-1],
            'vol_ma': vol_ma.iloc[-1],
            'volume': volume.iloc[-1],
            'close': close.iloc[-1]
        }
    
    def _trend_signal(self, df: pd.DataFrame, ind: Dict, state_info: Dict) -> Signal:
        """趋势跟踪信号"""
        price = ind['close']
        
        # 入场条件
        buy_conditions = [
            ind['ma_5'] > ind['ma_10'] > ind['ma_20'],  # 均线多头排列
            ind['macd_hist'] > 0,                        # MACD金叉
            ind['adx'] > 20,                             # 有趋势
            ind['volume'] > ind['vol_ma'] * 1.2,         # 放量
            ind['rsi'] < 70,                             # 未超买
        ]
        
        sell_conditions = [
            ind['ma_5'] < ind['ma_10'],                  # 短期死叉
            ind['macd_hist'] < 0,                        # MACD死叉
            ind['rsi'] > 80,                             # 超买
        ]
        
        buy_score = sum(buy_conditions)
        sell_score = sum(sell_conditions)
        
        if buy_score >= 3:
            action = 'buy'
            strength = buy_score / 5
            reason = f"趋势买入信号(得分{buy_score}/5)"
        elif sell_score >= 2:
            action = 'sell'
            strength = sell_score / 3
            reason = f"趋势卖出信号(得分{sell_score}/3)"
        else:
            action = 'hold'
            strength = 0
            reason = "等待趋势确认"
        
        # 计算止损止盈
        stop_loss = price - ind['atr'] * state_info['stop_atr']
        take_profit = price + ind['atr'] * state_info['stop_atr'] * 2
        
        return Signal(
            action=action,
            strength=strength,
            confidence=0,
            reason=reason,
            position_ratio=state_info['position_max'] * strength,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_state='',
            risk_level=state_info['risk']
        )
    
    def _mean_reversion_signal(self, df: pd.DataFrame, ind: Dict, state_info: Dict) -> Signal:
        """均值回归信号"""
        price = ind['close']
        
        # 布林带位置
        bb_position = (price - ind['bb_lower']) / (ind['bb_upper'] - ind['bb_lower'])
        
        # 入场条件（更严格）
        buy_conditions = [
            ind['rsi'] < 30,                            # 严重超卖
            price < ind['bb_lower'] * 1.02,             # 接近布林下轨
            ind['volume'] > ind['vol_ma'] * 1.5,        # 放量
            ind['macd_hist'] > ind['macd_hist'] * 0.8 if hasattr(ind['macd_hist'], '__gt__') else True,  # MACD改善
        ]
        
        sell_conditions = [
            ind['rsi'] > 70,                            # 严重超买
            price > ind['bb_upper'] * 0.98,             # 接近布林上轨
        ]
        
        # 震荡市更保守
        if bb_position < 0.2 and ind['rsi'] < 35:
            action = 'buy'
            strength = 0.7
            reason = f"超卖反弹(RSI={ind['rsi']:.0f}, BB位置={bb_position:.0%})"
        elif bb_position > 0.8 and ind['rsi'] > 65:
            action = 'sell'
            strength = 0.7
            reason = f"超买回落(RSI={ind['rsi']:.0f}, BB位置={bb_position:.0%})"
        else:
            action = 'hold'
            strength = 0
            reason = "震荡市观望"
        
        stop_loss = ind['bb_lower'] * 0.95 if action == 'buy' else price * 1.03
        take_profit = ind['bb_mid'] if action == 'buy' else ind['bb_mid']
        
        return Signal(
            action=action,
            strength=strength,
            confidence=0,
            reason=reason,
            position_ratio=state_info['position_max'] * strength if action == 'buy' else 0,
            stop_loss=stop_loss,
            take_profit=take_profit,
            market_state='',
            risk_level=state_info['risk']
        )
    
    def _get_confirmations(self, df: pd.DataFrame, ind: Dict, action: str) -> List[str]:
        """获取确认信号"""
        confirmations = []
        
        if action == 'buy':
            if ind['ma_5'] > ind['ma_10']:
                confirmations.append('MA金叉')
            if ind['macd_hist'] > 0:
                confirmations.append('MACD金叉')
            if ind['adx'] > 20:
                confirmations.append('趋势确认')
            if ind['volume'] > ind['vol_ma']:
                confirmations.append('放量')
            if ind['rsi'] < 70:
                confirmations.append('RSI未超买')
        
        elif action == 'sell':
            if ind['ma_5'] < ind['ma_10']:
                confirmations.append('MA死叉')
            if ind['macd_hist'] < 0:
                confirmations.append('MACD死叉')
            if ind['rsi'] > 30:
                confirmations.append('RSI超买')
            if ind['volume'] > ind['vol_ma']:
                confirmations.append('放量')
        
        return confirmations


class SmartTrader:
    """智能交易系统"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position = 0
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trailing_stop = 0
        self.highest_price = 0
        
        self.signal_gen = SignalGenerator()
        self.fetcher = AStockDataFetcher()
        
        self.trades = []
        self.equity_curve = [initial_capital]
        
        # 冷却期（避免频繁交易）
        self.last_trade_idx = -100
        self.cooldown_period = 5  # 5根K线冷却
        
        # 日亏损控制
        self.daily_start_equity = initial_capital
        self.max_daily_loss = 0.03  # 3%
        self.last_date = None
    
    def run_backtest(self, symbol: str, start_date: str, end_date: str,
                     verbose: bool = True) -> Dict:
        """运行回测"""
        
        df = self.fetcher.fetch_stock_daily(symbol, start_date, end_date)
        if df is None or len(df) < 100:
            print(f"⚠️ 数据不足: {symbol}")
            return None
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 智能策略回测 v3.0")
            print(f"股票: {symbol}")
            print(f"时间: {df['datetime'].iloc[0]} - {df['datetime'].iloc[-1]}")
            print(f"数据: {len(df)} 条")
            print("="*70)
        
        # 回测循环
        for i in range(100, len(df)):
            current_df = df.iloc[:i+1]
            price = df['close'].iloc[i]
            current_date = str(df['datetime'].iloc[i])[:10]
            
            # 日亏损重置
            if current_date != self.last_date:
                self.daily_start_equity = self.cash + self.position * price
                self.last_date = current_date
            
            # 检查日亏损
            current_equity = self.cash + self.position * price
            daily_pnl = (current_equity - self.daily_start_equity) / self.daily_start_equity
            
            # 更新追踪止损
            if self.position > 0:
                if price > self.highest_price:
                    self.highest_price = price
                    atr_val = atr(current_df['high'], current_df['low'], 
                                 current_df['close'], 14).iloc[-1]
                    self.trailing_stop = max(self.trailing_stop, 
                                            self.highest_price - atr_val * 1.5)
            
            # 止损检查
            if self.position > 0 and (price <= self.stop_loss or price <= self.trailing_stop):
                pnl = (price - self.entry_price) / self.entry_price
                self.trades.append({
                    'type': 'stop_loss',
                    'price': price,
                    'shares': self.position,
                    'pnl': pnl,
                    'date': current_date
                })
                if verbose:
                    print(f"[{current_date}] 止损卖出: {self.position}股 @ ¥{price:.2f} | PnL: {pnl*100:+.2f}%")
                self.cash += self.position * price
                self.position = 0
                self.last_trade_idx = i
                continue
            
            # 止盈检查
            if self.position > 0 and price >= self.take_profit:
                pnl = (price - self.entry_price) / self.entry_price
                self.trades.append({
                    'type': 'take_profit',
                    'price': price,
                    'shares': self.position,
                    'pnl': pnl,
                    'date': current_date
                })
                if verbose:
                    print(f"[{current_date}] 止盈卖出: {self.position}股 @ ¥{price:.2f} | PnL: {pnl*100:+.2f}%")
                self.cash += self.position * price
                self.position = 0
                self.last_trade_idx = i
                continue
            
            # 冷却期检查
            if i - self.last_trade_idx < self.cooldown_period:
                continue
            
            # 日亏损限制
            if daily_pnl < -self.max_daily_loss:
                continue
            
            # 生成信号
            signal = self.signal_gen.generate(current_df, self.trades)
            
            # 执行交易
            if signal.action == 'buy' and self.position == 0 and signal.confidence >= 0.6:
                position_value = self.cash * signal.position_ratio
                shares = int(position_value / price)
                
                if shares > 0:
                    self.position = shares
                    self.cash -= shares * price
                    self.entry_price = price
                    self.stop_loss = signal.stop_loss
                    self.take_profit = signal.take_profit
                    self.trailing_stop = signal.stop_loss
                    self.highest_price = price
                    self.last_trade_idx = i
                    
                    if verbose:
                        print(f"[{current_date}] 买入: {shares}股 @ ¥{price:.2f} | "
                              f"止损:¥{self.stop_loss:.2f} | 止盈:¥{self.take_profit:.2f}")
                        print(f"        原因: {signal.reason} | 信心度: {signal.confidence:.0%} | "
                              f"市场: {signal.market_state}")
            
            elif signal.action == 'sell' and self.position > 0:
                pnl = (price - self.entry_price) / self.entry_price
                self.trades.append({
                    'type': 'signal',
                    'price': price,
                    'shares': self.position,
                    'pnl': pnl,
                    'date': current_date,
                    'reason': signal.reason
                })
                if verbose:
                    print(f"[{current_date}] 信号卖出: {self.position}股 @ ¥{price:.2f} | "
                          f"PnL: {pnl*100:+.2f}% | {signal.reason}")
                self.cash += self.position * price
                self.position = 0
                self.last_trade_idx = i
            
            # 记录权益
            equity = self.cash + self.position * price
            self.equity_curve.append(equity)
        
        # 最终平仓
        if self.position > 0:
            final_price = df['close'].iloc[-1]
            pnl = (final_price - self.entry_price) / self.entry_price
            self.trades.append({
                'type': 'final',
                'price': final_price,
                'shares': self.position,
                'pnl': pnl,
                'date': str(df['datetime'].iloc[-1])[:10]
            })
            self.cash += self.position * final_price
            if verbose:
                print(f"[最终] 平仓: {self.position}股 @ ¥{final_price:.2f} | PnL: {pnl*100:+.2f}%")
        
        return self._calculate_performance(symbol)
    
    def _calculate_performance(self, symbol: str) -> Dict:
        """计算性能"""
        final_equity = self.equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        
        days = len(self.equity_curve)
        years = days / 252
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()
        
        sells = [t for t in self.trades if t.get('type') in ['signal', 'stop_loss', 'take_profit', 'final']]
        wins = [t for t in sells if t.get('pnl', 0) > 0]
        win_rate = len(wins) / len(sells) if sells else 0
        
        return {
            'symbol': symbol,
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
    
    def print_report(self, result: Dict):
        """打印报告"""
        print("\n" + "="*70)
        print("📊 回测报告")
        print("="*70)
        print(f"股票: {result['symbol']}")
        print(f"初始资金: ¥{self.initial_capital:,.2f}")
        print(f"最终权益: ¥{result['final_equity']:,.2f}")
        print(f"总收益: {result['total_return']*100:+.2f}%")
        print(f"年化收益: {result['annual_return']*100:+.2f}%")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"最大回撤: {result['max_drawdown']*100:.2f}%")
        print(f"波动率: {result['volatility']*100:.2f}%")
        print(f"交易次数: {result['total_trades']}")
        print(f"胜率: {result['win_rate']*100:.1f}%")
        
        # 评级
        if result['sharpe_ratio'] > 1.5 and result['max_drawdown'] > -0.10:
            grade = "A 🏆 优秀"
        elif result['sharpe_ratio'] > 1.0 and result['max_drawdown'] > -0.15:
            grade = "B ✅ 良好"
        elif result['sharpe_ratio'] > 0.5 and result['max_drawdown'] > -0.20:
            grade = "C ⚠️ 一般"
        else:
            grade = "D ❌ 需优化"
        
        print(f"\n策略评级: {grade}")
        print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 智能自适应策略 v3.0")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置
    INITIAL_CAPITAL = 100000
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    # 测试股票
    test_stocks = ['600519', '000858', '002594', '601318', '000001']
    
    results = []
    for symbol in test_stocks:
        trader = SmartTrader(INITIAL_CAPITAL)
        result = trader.run_backtest(symbol, start_date, end_date, verbose=True)
        
        if result:
            results.append(result)
            trader.print_report(result)
    
    # 汇总
    if results:
        print("\n" + "="*70)
        print("📈 汇总统计")
        print("="*70)
        
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        win_count = sum(1 for r in results if r['total_return'] > 0)
        
        print(f"测试股票: {len(results)}只")
        print(f"盈利股票: {win_count}只 ({win_count/len(results)*100:.0f}%)")
        print(f"平均收益: {avg_return*100:+.2f}%")
        print(f"平均夏普: {avg_sharpe:.2f}")


if __name__ == "__main__":
    main()
