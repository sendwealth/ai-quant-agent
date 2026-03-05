"""
盈利优化策略 v2.0
=================
改进点:
1. 多信号确认系统 (MA + RSI + MACD + 布林带 + 成交量)
2. 市场环境自动识别 (趋势/震荡)
3. 自适应参数调整 (基于波动率)
4. 凯利公式仓位管理
5. 智能止盈止损 (ATR + 追踪止损)
6. 多股票轮动策略
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.astock_fetcher import AStockDataFetcher, get_popular_astocks
from utils.indicators import sma, ema, rsi, macd, bollinger_bands, atr, adx


class MarketRegime(Enum):
    """市场环境"""
    BULL = "bull"        # 牛市 (强趋势上涨)
    BEAR = "bear"        # 熊市 (强趋势下跌)
    SIDEWAYS = "sideways"  # 震荡市 (无明确趋势)
    VOLATILE = "volatile"  # 高波动 (大起大落)


@dataclass
class TradingSignal:
    """交易信号"""
    action: str  # 'buy', 'sell', 'hold'
    strength: float  # 0-1 信号强度
    reason: str  # 信号原因
    position_size: float  # 建议仓位比例
    stop_loss: float  # 止损价
    take_profit: float  # 止盈价


class ProfitOptimizer:
    """盈利优化器"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.fetcher = AStockDataFetcher()
        
        # 自适应参数
        self.params = {
            # 趋势策略参数
            'ma_short': 5,
            'ma_long': 20,
            'ema_short': 8,
            'ema_long': 21,
            
            # 震荡策略参数
            'rsi_oversold': 25,
            'rsi_overbought': 75,
            'bb_std': 2.0,
            
            # 风控参数
            'atr_stop_multiplier': 2.0,
            'atr_trail_multiplier': 1.5,
            'max_position': 0.3,  # 最大30%仓位
            'max_daily_loss': 0.05,  # 单日最大亏损5%
            
            # 信号权重
            'trend_weight': 0.4,
            'momentum_weight': 0.3,
            'volume_weight': 0.2,
            'sentiment_weight': 0.1,
        }
    
    def detect_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """
        识别市场环境
        
        Returns:
            MarketRegime: 市场环境类型
        """
        close = df['close']
        
        # 计算指标
        ma_20 = sma(close, 20)
        ma_50 = sma(close, 50)
        ma_200 = sma(close, 200)
        
        # 波动率 (20日)
        returns = close.pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        current_vol = volatility.iloc[-1]
        
        # 趋势强度 (ADX)
        adx_val = adx(df['high'], df['low'], close, 14)
        current_adx = adx_val.iloc[-1]
        
        # 价格位置
        price = close.iloc[-1]
        ma_20_val = ma_20.iloc[-1]
        ma_50_val = ma_50.iloc[-1]
        ma_200_val = ma_200.iloc[-1] if not pd.isna(ma_200.iloc[-1]) else ma_50_val
        
        # 判断逻辑
        if current_vol > 0.35:  # 高波动 (>35%年化)
            return MarketRegime.VOLATILE
        
        # 趋势判断
        uptrend = price > ma_20_val > ma_50_val > ma_200_val
        downtrend = price < ma_20_val < ma_50_val < ma_200_val
        
        if current_adx > 25:
            if uptrend:
                return MarketRegime.BULL
            elif downtrend:
                return MarketRegime.BEAR
        
        return MarketRegime.SIDEWAYS
    
    def calculate_signal_strength(self, df: pd.DataFrame, regime: MarketRegime) -> Tuple[float, Dict]:
        """
        计算综合信号强度
        
        Returns:
            (signal_strength, signal_components)
        """
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        signals = {}
        
        # 1. 趋势信号 (MA交叉)
        ma_short = sma(close, self.params['ma_short'])
        ma_long = sma(close, self.params['ma_long'])
        ma_trend = 1 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1
        ma_cross_strength = abs((ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1])
        signals['trend'] = ma_trend * min(ma_cross_strength * 10, 1)
        
        # 2. 动量信号 (RSI)
        rsi_val = rsi(close, 14)
        current_rsi = rsi_val.iloc[-1]
        if current_rsi < 30:
            signals['momentum'] = 0.8  # 超卖，买入信号
        elif current_rsi > 70:
            signals['momentum'] = -0.8  # 超买，卖出信号
        elif current_rsi < 40:
            signals['momentum'] = 0.4
        elif current_rsi > 60:
            signals['momentum'] = -0.4
        else:
            signals['momentum'] = 0
        
        # 3. MACD信号
        macd_line, signal_line, histogram = macd(close)
        macd_strength = histogram.iloc[-1]
        signals['macd'] = np.clip(macd_strength * 100, -1, 1)
        
        # 4. 布林带信号
        upper, middle, lower = bollinger_bands(close, 20, self.params['bb_std'])
        price = close.iloc[-1]
        bb_position = (price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
        if bb_position < 0.2:
            signals['bollinger'] = 0.6  # 接近下轨，买入
        elif bb_position > 0.8:
            signals['bollinger'] = -0.6  # 接近上轨，卖出
        else:
            signals['bollinger'] = 0
        
        # 5. 成交量确认
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1]
        signals['volume'] = 0.5 if vol_ratio > 1.5 else 0
        
        # 根据市场环境调整权重
        if regime == MarketRegime.BULL:
            weights = {'trend': 0.5, 'momentum': 0.2, 'macd': 0.15, 'bollinger': 0.05, 'volume': 0.1}
        elif regime == MarketRegime.BEAR:
            weights = {'trend': 0.3, 'momentum': 0.3, 'macd': 0.2, 'bollinger': 0.1, 'volume': 0.1}
        elif regime == MarketRegime.SIDEWAYS:
            weights = {'trend': 0.2, 'momentum': 0.25, 'macd': 0.15, 'bollinger': 0.3, 'volume': 0.1}
        else:  # VOLATILE
            weights = {'trend': 0.25, 'momentum': 0.25, 'macd': 0.2, 'bollinger': 0.2, 'volume': 0.1}
        
        # 计算综合信号
        total_signal = sum(signals[k] * weights[k] for k in signals)
        
        return total_signal, signals
    
    def kelly_position_size(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        凯利公式计算最优仓位
        
        Kelly % = W - [(1-W) / R]
        W = 胜率
        R = 盈亏比 (平均盈利/平均亏损)
        """
        if avg_loss == 0:
            return self.params['max_position']
        
        r = avg_win / abs(avg_loss)
        kelly = win_rate - ((1 - win_rate) / r)
        
        # 保守起见，使用半凯利
        kelly = kelly * 0.5
        
        # 限制在最大仓位内
        return min(max(kelly, 0.05), self.params['max_position'])
    
    def calculate_stop_levels(self, df: pd.DataFrame, entry_price: float, 
                               signal: str) -> Tuple[float, float]:
        """
        计算止损止盈位
        
        Returns:
            (stop_loss, take_profit)
        """
        atr_val = atr(df['high'], df['low'], df['close'], 14).iloc[-1]
        
        if signal == 'buy':
            stop_loss = entry_price - atr_val * self.params['atr_stop_multiplier']
            take_profit = entry_price + atr_val * self.params['atr_stop_multiplier'] * 2
        else:
            stop_loss = entry_price + atr_val * self.params['atr_stop_multiplier']
            take_profit = entry_price - atr_val * self.params['atr_stop_multiplier'] * 2
        
        return stop_loss, take_profit
    
    def generate_signal(self, df: pd.DataFrame, 
                        trade_history: List[Dict] = None) -> TradingSignal:
        """
        生成交易信号
        
        Args:
            df: 历史数据
            trade_history: 历史交易记录 (用于计算胜率等)
        
        Returns:
            TradingSignal
        """
        # 1. 识别市场环境
        regime = self.detect_market_regime(df)
        
        # 2. 计算信号强度
        signal_strength, components = self.calculate_signal_strength(df, regime)
        
        # 3. 确定行动
        if signal_strength > 0.3:
            action = 'buy'
        elif signal_strength < -0.3:
            action = 'sell'
        else:
            action = 'hold'
        
        # 4. 计算仓位
        if trade_history and len(trade_history) > 10:
            wins = [t for t in trade_history if t.get('pnl', 0) > 0]
            losses = [t for t in trade_history if t.get('pnl', 0) <= 0]
            win_rate = len(wins) / len(trade_history)
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0.05
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else -0.03
            position_size = self.kelly_position_size(win_rate, avg_win, avg_loss)
        else:
            position_size = self.params['max_position'] * 0.5  # 初始保守仓位
        
        # 5. 计算止损止盈
        entry_price = df['close'].iloc[-1]
        stop_loss, take_profit = self.calculate_stop_levels(df, entry_price, action)
        
        # 6. 生成原因说明
        reason = f"市场:{regime.value} | 信号:{signal_strength:.2f} | "
        reason += " | ".join([f"{k}:{v:.2f}" for k, v in components.items()])
        
        return TradingSignal(
            action=action,
            strength=abs(signal_strength),
            reason=reason,
            position_size=position_size if action != 'hold' else 0,
            stop_loss=stop_loss,
            take_profit=take_profit
        )


class MultiStockStrategy:
    """多股票轮动策略"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.optimizer = ProfitOptimizer(initial_capital)
        self.fetcher = AStockDataFetcher()
        
    def rank_stocks(self, symbols: List[str], 
                    start_date: str, end_date: str) -> pd.DataFrame:
        """
        对股票进行评分排名
        
        Returns:
            排名后的DataFrame
        """
        results = []
        
        for symbol in symbols:
            try:
                df = self.fetcher.fetch_stock_daily(symbol, start_date, end_date)
                if df is None or len(df) < 100:
                    continue
                
                # 计算评分
                regime = self.optimizer.detect_market_regime(df)
                signal_strength, components = self.optimizer.calculate_signal_strength(df, regime)
                
                # 计算动量 (20日收益率)
                momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1)
                
                # 计算波动率
                volatility = df['close'].pct_change().std() * np.sqrt(252)
                
                results.append({
                    'symbol': symbol,
                    'signal': signal_strength,
                    'regime': regime.value,
                    'momentum': momentum,
                    'volatility': volatility,
                    'price': df['close'].iloc[-1],
                    **components
                })
                
            except Exception as e:
                print(f"⚠️ {symbol} 分析失败: {e}")
                continue
        
        df_results = pd.DataFrame(results)
        
        if len(df_results) == 0:
            return df_results
        
        # 综合评分
        df_results['score'] = (
            df_results['signal'] * 0.4 +
            df_results['momentum'] * 2 +
            (1 - df_results['volatility']) * 0.2
        )
        
        # 按评分排序
        df_results = df_results.sort_values('score', ascending=False)
        
        return df_results
    
    def select_top_stocks(self, ranked_df: pd.DataFrame, 
                          top_n: int = 3) -> List[str]:
        """
        选择得分最高的股票
        
        Args:
            ranked_df: 排名后的DataFrame
            top_n: 选择数量
        
        Returns:
            股票代码列表
        """
        if len(ranked_df) == 0:
            return []
        
        # 只选择买入信号的股票
        buy_signals = ranked_df[ranked_df['signal'] > 0.3]
        
        return buy_signals.head(top_n)['symbol'].tolist()


class BacktestEngine:
    """回测引擎"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.optimizer = ProfitOptimizer(initial_capital)
        self.fetcher = AStockDataFetcher()
    
    def run_backtest(self, symbol: str, 
                     start_date: str, end_date: str,
                     verbose: bool = True) -> Dict:
        """
        运行回测
        
        Returns:
            回测结果
        """
        # 获取数据
        df = self.fetcher.fetch_stock_daily(symbol, start_date, end_date)
        if df is None or len(df) < 100:
            print(f"⚠️ 数据不足: {symbol}")
            return None
        
        # 初始化状态
        cash = self.initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trailing_stop = 0
        highest_price = 0
        
        trades = []
        equity_curve = [self.initial_capital]
        
        # 回测循环
        for i in range(100, len(df)):
            current_df = df.iloc[:i+1]
            price = df['close'].iloc[i]
            
            # 更新追踪止损
            if position > 0:
                highest_price = max(highest_price, price)
                atr_val = atr(current_df['high'], current_df['low'], 
                            current_df['close'], 14).iloc[-1]
                trailing_stop = max(trailing_stop, 
                                   highest_price - atr_val * self.optimizer.params['atr_trail_multiplier'])
            
            # 检查止损
            if position > 0 and (price <= stop_loss or price <= trailing_stop):
                cash = position * price
                pnl = (price - entry_price) / entry_price
                trades.append({
                    'type': 'stop_loss',
                    'price': price,
                    'shares': position,
                    'pnl': pnl,
                    'date': df['datetime'].iloc[i]
                })
                position = 0
                entry_price = 0
                continue
            
            # 检查止盈
            if position > 0 and price >= take_profit:
                cash = position * price
                pnl = (price - entry_price) / entry_price
                trades.append({
                    'type': 'take_profit',
                    'price': price,
                    'shares': position,
                    'pnl': pnl,
                    'date': df['datetime'].iloc[i]
                })
                position = 0
                entry_price = 0
                continue
            
            # 生成信号
            signal = self.optimizer.generate_signal(current_df, trades)
            
            # 执行交易
            if signal.action == 'buy' and position == 0:
                position_value = cash * signal.position_size
                shares = int(position_value / price)
                if shares > 0:
                    position = shares
                    cash -= shares * price
                    entry_price = price
                    stop_loss = signal.stop_loss
                    take_profit = signal.take_profit
                    trailing_stop = stop_loss
                    highest_price = price
                    
                    if verbose:
                        print(f"买入: {shares}股 @ ¥{price:.2f} | 止损:¥{stop_loss:.2f} | 止盈:¥{take_profit:.2f}")
            
            elif signal.action == 'sell' and position > 0:
                cash = position * price
                pnl = (price - entry_price) / entry_price
                trades.append({
                    'type': 'signal',
                    'price': price,
                    'shares': position,
                    'pnl': pnl,
                    'date': df['datetime'].iloc[i]
                })
                
                if verbose:
                    print(f"卖出: {position}股 @ ¥{price:.2f} | 盈亏:{pnl*100:+.2f}%")
                
                position = 0
                entry_price = 0
            
            # 记录权益
            equity = cash + position * price
            equity_curve.append(equity)
        
        # 最终平仓
        if position > 0:
            final_price = df['close'].iloc[-1]
            cash = position * final_price
            pnl = (final_price - entry_price) / entry_price
            trades.append({
                'type': 'final',
                'price': final_price,
                'shares': position,
                'pnl': pnl,
                'date': df['datetime'].iloc[-1]
            })
        
        # 计算性能指标
        final_equity = equity_curve[-1]
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        equity_series = pd.Series(equity_curve)
        daily_returns = equity_series.pct_change().dropna()
        
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_drawdown = drawdown.min()
        
        win_trades = [t for t in trades if t.get('pnl', 0) > 0]
        loss_trades = [t for t in trades if t.get('pnl', 0) <= 0]
        win_rate = len(win_trades) / len(trades) if trades else 0
        
        return {
            'symbol': symbol,
            'final_equity': final_equity,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'trades': trades,
            'equity_curve': equity_curve
        }
    
    def print_report(self, result: Dict):
        """打印回测报告"""
        print("\n" + "="*70)
        print("📊 回测报告")
        print("="*70)
        print(f"股票: {result['symbol']}")
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
            grade = "A 🏆"
        elif result['sharpe_ratio'] > 1.0 and result['max_drawdown'] > -0.15:
            grade = "B ✅"
        elif result['sharpe_ratio'] > 0.5 and result['max_drawdown'] > -0.2:
            grade = "C ⚠️"
        else:
            grade = "D ❌"
        
        print(f"策略评级: {grade}")
        print("="*70)


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🚀 盈利优化策略 v2.0")
    print("="*70)
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 配置
    INITIAL_CAPITAL = 100000
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y%m%d')  # 2年数据
    
    # 创建回测引擎
    engine = BacktestEngine(INITIAL_CAPITAL)
    
    # 测试股票
    test_stocks = ['600519', '000858', '600036', '601318', '000001']
    
    print(f"\n📊 测试股票: {test_stocks}")
    print(f"📅 时间范围: {start_date} - {end_date}")
    
    results = []
    for symbol in test_stocks:
        print(f"\n{'='*70}")
        print(f"回测: {symbol}")
        print("="*70)
        
        result = engine.run_backtest(symbol, start_date, end_date, verbose=True)
        if result:
            results.append(result)
            engine.print_report(result)
    
    # 汇总
    if results:
        print("\n" + "="*70)
        print("📈 汇总报告")
        print("="*70)
        
        avg_return = np.mean([r['total_return'] for r in results])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in results])
        avg_drawdown = np.mean([r['max_drawdown'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])
        
        print(f"平均总收益: {avg_return*100:+.2f}%")
        print(f"平均夏普比率: {avg_sharpe:.2f}")
        print(f"平均最大回撤: {avg_drawdown*100:.2f}%")
        print(f"平均胜率: {avg_win_rate*100:.1f}%")
    
    # 多股票策略测试
    print("\n" + "="*70)
    print("🎯 多股票轮动策略")
    print("="*70)
    
    multi_strategy = MultiStockStrategy(INITIAL_CAPITAL)
    ranked = multi_strategy.rank_stocks(get_popular_astocks()[:10], start_date, end_date)
    
    if len(ranked) > 0:
        print("\n股票评分排名:")
        print(ranked[['symbol', 'score', 'signal', 'regime', 'momentum']].head(10).to_string())
        
        top_stocks = multi_strategy.select_top_stocks(ranked, top_n=3)
        print(f"\n推荐买入: {top_stocks}")


if __name__ == "__main__":
    main()
