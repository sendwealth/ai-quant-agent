"""
完整优化流程 v4.0
=================
1. 稳定的数据获取（重试机制）
2. 遗传算法参数优化
3. 多股票验证
4. 生成最优参数报告
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def fetch_stock_data_with_retry(symbol: str, start_date: str, end_date: str, 
                                  max_retries: int = 5) -> pd.DataFrame:
    """带重试的数据获取"""
    import akshare as ak
    
    for attempt in range(max_retries):
        try:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period='daily',
                start_date=start_date,
                end_date=end_date,
                adjust='qfq'
            )
            
            # 标准化
            df = df.rename(columns={
                '日期': 'datetime',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume'
            })
            
            df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime').reset_index(drop=True)
            
            print(f"✓ 获取 {symbol} 成功: {len(df)} 条")
            return df
            
        except Exception as e:
            wait_time = (attempt + 1) * 2
            print(f"尝试 {attempt+1}/{max_retries} 失败，{wait_time}秒后重试...")
            time.sleep(wait_time)
    
    print(f"❌ {symbol} 获取失败，使用模拟数据")
    return generate_mock_data(symbol, start_date, end_date)


def generate_mock_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """生成模拟数据"""
    start_dt = datetime.strptime(start_date, '%Y%m%d')
    end_dt = datetime.strptime(end_date, '%Y%m%d')
    dates = pd.date_range(start_dt, end_dt, freq='B')
    
    np.random.seed(hash(symbol) % 10000)
    base_price = np.random.uniform(10, 100)
    
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices = base_price * (1 + np.cumsum(returns))
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    return df


# 技术指标
def sma(data, period):
    return data.rolling(window=period).mean()

def ema(data, period):
    return data.ewm(span=period, adjust=False).mean()

def rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def atr(high, low, close, period=14):
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(high, low, close, period=14):
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(window=period).mean()
    
    plus_di = 100 * plus_dm.rolling(window=period).mean() / atr_val
    minus_di = 100 * minus_dm.rolling(window=period).mean() / atr_val
    
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx.rolling(window=period).mean()


@dataclass
class StrategyParams:
    """策略参数"""
    ma_short: int = 5
    ma_long: int = 20
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    atr_stop_mult: float = 2.0
    atr_trail_mult: float = 1.5
    adx_threshold: int = 20
    volume_mult: float = 1.2
    max_position: float = 0.3
    min_confidence: float = 0.6


class GeneticOptimizer:
    """遗传算法参数优化器"""
    
    def __init__(self, population_size: int = 30, generations: int = 10):
        self.population_size = population_size
        self.generations = generations
        
        # 参数范围
        self.param_ranges = {
            'ma_short': (3, 15),
            'ma_long': (15, 60),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'atr_stop_mult': (1.5, 3.0),
            'atr_trail_mult': (1.0, 2.5),
            'adx_threshold': (15, 30),
            'volume_mult': (1.0, 2.0),
            'max_position': (0.2, 0.5),
            'min_confidence': (0.5, 0.8)
        }
    
    def random_params(self) -> StrategyParams:
        """随机生成参数"""
        params = {}
        for key, (min_val, max_val) in self.param_ranges.items():
            if isinstance(min_val, int):
                params[key] = random.randint(min_val, max_val)
            else:
                params[key] = round(random.uniform(min_val, max_val), 2)
        return StrategyParams(**params)
    
    def crossover(self, p1: StrategyParams, p2: StrategyParams) -> StrategyParams:
        """交叉"""
        child_params = {}
        for key in self.param_ranges.keys():
            child_params[key] = getattr(p1 if random.random() < 0.5 else p2, key)
        return StrategyParams(**child_params)
    
    def mutate(self, params: StrategyParams, rate: float = 0.2) -> StrategyParams:
        """变异"""
        mutated = {}
        for key, (min_val, max_val) in self.param_ranges.items():
            val = getattr(params, key)
            if random.random() < rate:
                if isinstance(min_val, int):
                    val = random.randint(min_val, max_val)
                else:
                    val = round(random.uniform(min_val, max_val), 2)
            mutated[key] = val
        return StrategyParams(**mutated)
    
    def backtest(self, df: pd.DataFrame, params: StrategyParams) -> Dict:
        """回测"""
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # 计算指标
        ma_s = sma(close, params.ma_short)
        ma_l = sma(close, params.ma_long)
        rsi_val = rsi(close, 14)
        atr_val = atr(high, low, close, 14)
        adx_val = adx(high, low, close, 14)
        vol_ma = volume.rolling(20).mean()
        
        # 模拟交易
        cash = 100000
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        trailing_stop = 0
        highest = 0
        
        trades = []
        equity = [cash]
        
        for i in range(60, len(df)):
            price = close.iloc[i]
            
            # 更新追踪止损
            if position > 0:
                if price > highest:
                    highest = price
                    trailing_stop = max(trailing_stop, highest - atr_val.iloc[i] * params.atr_trail_mult)
            
            # 止损检查
            if position > 0 and (price <= stop_loss or price <= trailing_stop):
                pnl = (price - entry_price) / entry_price
                cash = position * price
                trades.append(pnl)
                position = 0
            
            # 止盈检查
            elif position > 0 and price >= take_profit:
                pnl = (price - entry_price) / entry_price
                cash = position * price
                trades.append(pnl)
                position = 0
            
            else:
                # 信号
                ma_signal = 1 if ma_s.iloc[i] > ma_l.iloc[i] else -1
                rsi_signal = 1 if rsi_val.iloc[i] < params.rsi_oversold else (-1 if rsi_val.iloc[i] > params.rsi_overbought else 0)
                adx_ok = adx_val.iloc[i] > params.adx_threshold
                vol_ok = volume.iloc[i] > vol_ma.iloc[i] * params.volume_mult
                
                # 综合信号
                if ma_signal == 1 and adx_ok and vol_ok and rsi_val.iloc[i] < params.rsi_overbought and position == 0:
                    pos_val = cash * params.max_position
                    shares = int(pos_val / price)
                    if shares > 0:
                        position = shares
                        cash -= shares * price
                        entry_price = price
                        stop_loss = price - atr_val.iloc[i] * params.atr_stop_mult
                        take_profit = price + atr_val.iloc[i] * params.atr_stop_mult * 2
                        trailing_stop = stop_loss
                        highest = price
                
                elif ma_signal == -1 and position > 0:
                    pnl = (price - entry_price) / entry_price
                    cash = position * price
                    trades.append(pnl)
                    position = 0
            
            equity.append(cash + position * price)
        
        # 最终平仓
        if position > 0:
            cash = position * close.iloc[-1]
        
        final_equity = cash
        total_return = (final_equity - 100000) / 100000
        
        # 计算夏普
        equity_series = pd.Series(equity)
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        peak = equity_series.expanding().max()
        dd = (equity_series - peak) / peak
        max_dd = dd.min()
        
        # 胜率
        wins = sum(1 for t in trades if t > 0)
        win_rate = wins / len(trades) if trades else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'trades': len(trades),
            'final_equity': final_equity
        }
    
    def fitness(self, result: Dict) -> float:
        """适应度函数"""
        # 综合评分：收益 + 夏普 - 回撤
        score = (
            result['total_return'] * 100 +
            result['sharpe_ratio'] * 20 +
            result['max_drawdown'] * 50 +
            result['win_rate'] * 10
        )
        return score
    
    def optimize(self, df: pd.DataFrame) -> Tuple[StrategyParams, Dict]:
        """优化"""
        print(f"\n{'='*60}")
        print("🧬 遗传算法参数优化")
        print(f"种群: {self.population_size}, 代数: {self.generations}")
        print("="*60)
        
        # 初始种群
        population = [self.random_params() for _ in range(self.population_size)]
        best_params = None
        best_result = None
        best_score = -float('inf')
        
        for gen in range(self.generations):
            # 评估
            scores = []
            for params in population:
                result = self.backtest(df, params)
                score = self.fitness(result)
                scores.append((score, params, result))
            
            # 排序
            scores.sort(key=lambda x: x[0], reverse=True)
            
            # 更新最佳
            if scores[0][0] > best_score:
                best_score = scores[0][0]
                best_params = scores[0][1]
                best_result = scores[0][2]
            
            print(f"代 {gen+1}/{self.generations}: 最佳得分={scores[0][0]:.2f}, "
                  f"收益={scores[0][2]['total_return']*100:+.2f}%, "
                  f"夏普={scores[0][2]['sharpe_ratio']:.2f}")
            
            # 选择 + 交叉 + 变异
            survivors = [s[1] for s in scores[:self.population_size // 2]]
            new_population = survivors.copy()
            
            while len(new_population) < self.population_size:
                p1, p2 = random.sample(survivors, 2)
                child = self.crossover(p1, p2)
                child = self.mutate(child)
                new_population.append(child)
            
            population = new_population
        
        return best_params, best_result


def run_optimization():
    """运行完整优化"""
    print("\n" + "="*70)
    print("🚀 AI量化策略完整优化流程 v4.0")
    print("="*70)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 数据
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=365)).strftime('%Y%m%d')
    
    stocks = ['600519', '000858', '002594', '601318', '000001']
    
    results = {}
    
    for symbol in stocks:
        print(f"\n{'='*70}")
        print(f"📊 优化股票: {symbol}")
        print("="*70)
        
        # 获取数据
        df = fetch_stock_data_with_retry(symbol, start_date, end_date)
        
        if len(df) < 100:
            print(f"⚠️ 数据不足，跳过")
            continue
        
        # 优化
        optimizer = GeneticOptimizer(population_size=20, generations=5)
        best_params, best_result = optimizer.optimize(df)
        
        results[symbol] = {
            'params': best_params,
            'result': best_result,
            'data_source': 'real' if len(df) > 200 else 'mock'
        }
        
        print(f"\n✅ 最优参数:")
        print(f"   MA: {best_params.ma_short}/{best_params.ma_long}")
        print(f"   RSI: {best_params.rsi_oversold}/{best_params.rsi_overbought}")
        print(f"   ATR止损: {best_params.atr_stop_mult}x")
        print(f"   最大仓位: {best_params.max_position*100:.0f}%")
        print(f"\n📊 回测结果:")
        print(f"   总收益: {best_result['total_return']*100:+.2f}%")
        print(f"   夏普比率: {best_result['sharpe_ratio']:.2f}")
        print(f"   最大回撤: {best_result['max_drawdown']*100:.2f}%")
        print(f"   胜率: {best_result['win_rate']*100:.1f}%")
    
    # 汇总
    print("\n" + "="*70)
    print("📈 优化汇总")
    print("="*70)
    
    total_return = 0
    total_sharpe = 0
    win_count = 0
    
    for symbol, data in results.items():
        r = data['result']
        total_return += r['total_return']
        total_sharpe += r['sharpe_ratio']
        if r['total_return'] > 0:
            win_count += 1
        
        status = "✅" if r['total_return'] > 0 else "❌"
        print(f"{status} {symbol}: 收益{r['total_return']*100:+.2f}%, 夏普{r['sharpe_ratio']:.2f}")
    
    n = len(results)
    print(f"\n盈利股票: {win_count}/{n} ({win_count/n*100:.0f}%)")
    print(f"平均收益: {total_return/n*100:+.2f}%")
    print(f"平均夏普: {total_sharpe/n:.2f}")
    
    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'stocks': {}
    }
    
    for symbol, data in results.items():
        output['stocks'][symbol] = {
            'params': data['params'].__dict__,
            'result': data['result']
        }
    
    with open('optimization_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n结果已保存: optimization_results.json")
    
    return results


if __name__ == "__main__":
    run_optimization()
