"""真实数据回测 — AkShare A股行情"""

import numpy as np
import pandas as pd
import logging
import akshare as ak

from quant_agent.backtest.engine import BacktestEngine, CommissionModel, SlippageModel
from quant_agent.strategy.indicators import rsi, macd, ema, bollinger_bands, adx, atr

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

STOCKS = {
    "300750": "宁德时代",
    "002475": "立讯精密",
    "601318": "中国平安",
    "600276": "恒瑞医药",
    "000858": "五粮液",
}


def fetch_real_data(stock_code: str, start: str = "20240101", end: str = "20260411") -> pd.DataFrame:
    """从 AkShare 获取真实行情"""
    logger.info(f"📥 获取 {stock_code} 行情...")
    df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", start_date=start, end_date=end)
    df = df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume",
        "成交额": "amount", "涨跌幅": "pct_change",
    })
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
    return df[["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]].reset_index(drop=True)


def strategy_dual_ma(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    ema_fast = ema(close, 12)
    ema_slow = ema(close, 26)
    signals = pd.Series(0, index=range(len(close)), dtype=float)
    # 金叉买入
    signals[(ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))] = 1
    # 死叉卖出
    signals[(ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))] = -1
    # 持仓保持
    signals = signals.replace(0, np.nan).ffill().fillna(0)
    return signals


def strategy_bollinger(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    upper, middle, lower = bollinger_bands(close, 20, 2.0)
    signals = pd.Series(0, index=range(len(close)), dtype=float)
    # 收盘价低于下轨买入
    signals[close < lower] = 1
    # 收盘价高于上轨卖出
    signals[close > upper] = -1
    # 持仓保持
    signals = signals.replace(0, np.nan).ffill().fillna(0)
    return signals


def strategy_rsi_macd(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    rsi_val = rsi(close, 14)
    macd_line, signal_line, histogram = macd(close)
    signals = pd.Series(0, index=range(len(close)), dtype=float)
    # RSI 超卖 + MACD 金叉
    signals[(rsi_val < 35) & (histogram > 0)] = 1
    # RSI 超买 + MACD 死叉
    signals[(rsi_val > 65) & (histogram < 0)] = -1
    signals = signals.replace(0, np.nan).ffill().fillna(0)
    return signals


def strategy_trend_adx(df: pd.DataFrame) -> pd.Series:
    close, high, low = df["close"], df["high"], df["low"]
    ema_fast = ema(close, 10)
    ema_slow = ema(close, 30)
    adx_val = adx(high, low, close, 14)
    signals = pd.Series(0, index=range(len(close)), dtype=float)
    # 强趋势 + EMA 方向
    signals[(ema_fast > ema_slow) & (adx_val > 25)] = 1
    signals[(ema_fast < ema_slow) & (adx_val > 25)] = -1
    signals = signals.replace(0, np.nan).ffill().fillna(0)
    return signals


STRATEGIES = {
    "双均线EMA": strategy_dual_ma,
    "布林带": strategy_bollinger,
    "RSI+MACD": strategy_rsi_macd,
    "ADX趋势": strategy_trend_adx,
}


def run_backtest(stock_code: str, df: pd.DataFrame, strategy_name: str, strategy_fn) -> dict:
    signals = strategy_fn(df)
    engine = BacktestEngine(
        initial_capital=100000,
        commission=CommissionModel(commission_rate=0.0003, stamp_tax_rate=0.001),
        slippage=SlippageModel(basis_points=1.0),
    )
    result = engine.run(df, signals)
    return result


def main():
    print("\n" + "=" * 70)
    print("🦞 AI Quant Agent v3.0 — 真实数据回测")
    print("=" * 70)

    all_results = {}

    for code, name in STOCKS.items():
        try:
            df = fetch_real_data(code)
            if df.empty or len(df) < 60:
                logger.warning(f"{code} 数据不足，跳过")
                continue

            print(f"\n{'━' * 70}")
            print(f"  📊 {name} ({code})  |  {len(df)} 个交易日  |  "
                  f"{df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
            print(f"  当前价: {df['close'].iloc[-1]:.2f}  |  "
                  f"期间涨跌: {df['pct_change'].sum():.2%}")
            print(f"{'━' * 70}")

            stock_results = {}
            for s_name, s_fn in STRATEGIES.items():
                result = run_backtest(code, df, s_name, s_fn)
                stock_results[s_name] = result

                emoji = "🟢" if result.total_return > 0 else "🔴"
                print(f"  {emoji} {s_name:<10}  收益:{result.total_return:>7.2%}  "
                      f"年化:{result.annual_return:>7.2%}  MaxDD:{result.max_drawdown:>7.2%}  "
                      f"Sharpe:{result.sharpe_ratio:>6.3f}  "
                      f"胜率:{result.win_rate:>5.0%}  "
                      f"交易:{result.total_trades:>3}次")

            all_results[code] = {"name": name, "results": stock_results, "df": df}

        except Exception as e:
            logger.error(f"{code} 回测失败: {e}")

    # 汇总对比
    print("\n\n" + "=" * 70)
    print("📋 全市场回测汇总")
    print("=" * 70)
    print(f"{'股票':<12} {'策略':<10} {'总收益':>8} {'年化':>8} {'最大回撤':>8} "
          f"{'Sharpe':>7} {'胜率':>6} {'交易':>4}")
    print("-" * 70)

    for code, data in all_results.items():
        for s_name, r in data["results"].items():
            emoji = "🟢" if r.total_return > 0 else "🔴"
            print(f"{emoji}{data['name']:<11} {s_name:<10} {r.total_return:>7.2%} "
                  f"{r.annual_return:>7.2%} {r.max_drawdown:>7.2%} "
                  f"{r.sharpe_ratio:>7.3f} {r.win_rate:>5.0%} {r.total_trades:>4}")

    # 最优组合
    print("\n" + "-" * 70)
    best_by_sharpe = None
    best_sharpe = -999
    for code, data in all_results.items():
        for s_name, r in data["results"].items():
            if r.sharpe_ratio > best_sharpe and r.total_trades > 0:
                best_sharpe = r.sharpe_ratio
                best_by_sharpe = (code, data["name"], s_name, r)

    if best_by_sharpe:
        code, name, s_name, r = best_by_sharpe
        print(f"🏆 最优 (Sharpe): {name}({code}) + {s_name}")
        print(f"   收益 {r.total_return:.2%} | 年化 {r.annual_return:.2%} | "
              f"MaxDD {r.max_drawdown:.2%} | Sharpe {r.sharpe_ratio:.3f} | "
              f"胜率 {r.win_rate:.0%} | 盈亏比 {r.profit_factor:.2f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
