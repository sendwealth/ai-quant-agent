"""真实数据回测 — 使用 BaoStock 获取行情，技术指标策略回测"""

from datetime import datetime

import numpy as np
import pandas as pd

from quant_agent.data.sources.baostock import BaoStockSource
from quant_agent.strategy.indicators import rsi, macd, ema, bollinger_bands, adx
from quant_agent.backtest.engine import BacktestEngine, CommissionModel, SlippageModel


# ── 策略信号生成 ──

def strategy_dual_ema(df: pd.DataFrame) -> pd.Series:
    """双均线策略: EMA12/EMA60 金叉死叉"""
    close = df["close"]
    e12 = ema(close, 12)
    e60 = ema(close, 60)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[(e12 > e60) & (e12.shift(1) <= e60.shift(1))] = 1
    sig[(e12 < e60) & (e12.shift(1) >= e60.shift(1))] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)


def strategy_rsi_macd(df: pd.DataFrame) -> pd.Series:
    """RSI + MACD 组合策略"""
    close = df["close"]
    r = rsi(close, 14)
    _, _, h = macd(close)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[(r < 35) & (h > 0)] = 1
    sig[(r > 65) & (h < 0)] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)


def strategy_bollinger(df: pd.DataFrame) -> pd.Series:
    """布林带策略: 触及下轨买入，触及上轨卖出"""
    close = df["close"]
    upper, _, lower = bollinger_bands(close, 20, 2.0)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[close < lower] = 1
    sig[close > upper] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)


STRATEGIES = {
    "双均线EMA12/60": strategy_dual_ema,
    "RSI+MACD": strategy_rsi_macd,
    "布林带": strategy_bollinger,
}

# ── 回测股票池 ──

STOCKS = {
    "300750": "宁德时代",
    "600519": "贵州茅台",
    "002594": "比亚迪",
    "601318": "中国平安",
    "000858": "五粮液",
    "600276": "恒瑞医药",
}


def run_backtest():
    print(f"\n{'=' * 90}")
    print(f"  AI Quant Agent v3.0 — 真实数据回测")
    print(f"  日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'=' * 90}")

    print(f"\n📊 获取行情数据 (BaoStock)...")
    bs = BaoStockSource()
    price_cache: dict[str, pd.DataFrame] = {}
    for code, name in STOCKS.items():
        df = bs.get_price_data(code, days=250, adjust="qfq")
        if df is not None and len(df) > 60:
            price_cache[code] = df
            print(f"  ✅ {code} {name}: {len(df)} 条")
        else:
            print(f"  ❌ {code} {name}: 获取失败")
    bs.logout()

    if not price_cache:
        print("\n❌ 无可用数据")
        return

    engine = BacktestEngine(
        initial_capital=100000,
        commission=CommissionModel(commission_rate=0.0003, stamp_tax_rate=0.001),
        slippage=SlippageModel(basis_points=1.0),
    )

    if not price_cache:
        print("\n❌ 无可用数据")
        return

    # ── 逐策略回测 ──
    all_results = []

    for strat_name, strat_fn in STRATEGIES.items():
        print(f"\n{'─' * 90}")
        print(f"📈 策略: {strat_name}")
        print(f"{'─' * 90}")
        print(f"{'代码':<8} {'名称':<8} {'数据量':>6} {'总收益':>8} {'年化':>8} "
              f"{'最大回撤':>8} {'Sharpe':>7} {'胜率':>6} {'盈亏比':>6} {'交易':>4}")
        print(f"{'─' * 90}")

        strat_results = []

        for code, name in STOCKS.items():
            df = price_cache.get(code)
            if df is None:
                continue

            try:
                signals = strat_fn(df)
                result = engine.run(df, signals)

                strat_results.append({
                    "code": code, "name": name, "strategy": strat_name,
                    "days": len(df),
                    **{k: getattr(result, k) for k in [
                        "total_return", "annual_return", "max_drawdown",
                        "sharpe_ratio", "win_rate", "profit_factor", "total_trades",
                    ]},
                })

                emoji = "🟢" if result.total_return > 0 else "🔴"
                print(f"{emoji}{code:<7} {name:<8} {len(df):>6} "
                      f"{result.total_return:>7.2%} {result.annual_return:>7.2%} "
                      f"{result.max_drawdown:>7.2%} {result.sharpe_ratio:>7.3f} "
                      f"{result.win_rate:>5.0%} {result.profit_factor:>6.2f} "
                      f"{result.total_trades:>4}")

            except Exception as e:
                print(f"  ⚠️ {code} {name}: {e}")

        if strat_results:
            rdf = pd.DataFrame(strat_results)
            avg_ret = rdf["total_return"].mean()
            avg_sharpe = rdf["sharpe_ratio"].mean()
            wins = (rdf["total_return"] > 0).sum()
            print(f"{'─' * 90}")
            print(f"  平均收益: {avg_ret:.2%}  平均Sharpe: {avg_sharpe:.3f}  "
                  f"盈利: {wins}/{len(rdf)}")

            all_results.extend(strat_results)

    # ── 全局汇总 ──
    if not all_results:
        print("\n❌ 无回测结果")
        return

    print(f"\n{'=' * 90}")
    print(f"📋 全局汇总")
    print(f"{'=' * 90}")

    ardf = pd.DataFrame(all_results)

    # 按策略汇总
    print(f"\n按策略:")
    for strat_name in STRATEGIES:
        sdf = ardf[ardf["strategy"] == strat_name]
        if sdf.empty:
            continue
        avg_ret = sdf["total_return"].mean()
        avg_sharpe = sdf["sharpe_ratio"].mean()
        avg_maxdd = sdf["max_drawdown"].mean()
        wins = (sdf["total_return"] > 0).sum()
        emoji = "🟢" if avg_ret > 0 else "🔴"
        print(f"  {emoji} {strat_name:<12} 平均收益: {avg_ret:>7.2%}  "
              f"Sharpe: {avg_sharpe:>6.3f}  MaxDD: {avg_maxdd:>7.2%}  "
              f"盈利: {wins}/{len(sdf)}")

    # 按股票汇总
    print(f"\n按股票:")
    for code, name in STOCKS.items():
        sdf = ardf[ardf["code"] == code]
        if sdf.empty:
            continue
        avg_ret = sdf["total_return"].mean()
        best_strat = sdf.loc[sdf["sharpe_ratio"].idxmax(), "strategy"]
        emoji = "🟢" if avg_ret > 0 else "🔴"
        print(f"  {emoji} {code} {name:<8} 平均收益: {avg_ret:>7.2%}  "
              f"最佳策略: {best_strat}")

    # 最优组合
    best = ardf.loc[ardf["sharpe_ratio"].idxmax()]
    print(f"\n🏆 最优组合: {best['code']} {best['name']} + {best['strategy']}")
    print(f"   收益: {best['total_return']:.2%}  Sharpe: {best['sharpe_ratio']:.3f}  "
          f"MaxDD: {best['max_drawdown']:.2%}  胜率: {best['win_rate']:.0%}")

    print(f"\n{'=' * 90}")
    print(f"✅ 回测完成 — {len(all_results)} 组结果")
    print(f"{'=' * 90}\n")


if __name__ == "__main__":
    run_backtest()
