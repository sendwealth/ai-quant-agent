"""选股 + 回测 一体化（多数据源自动降级: BaoStock → AkShare → Tushare）"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from datetime import datetime

from quant_agent.backtest.engine import BacktestEngine, CommissionModel, SlippageModel
from quant_agent.strategy.indicators import rsi, macd, ema, adx, bollinger_bands

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# === 多数据源降级 ===

def get_daily_multi(codes: list[str], start="20240101") -> dict[str, pd.DataFrame]:
    """批量获取日线（BaoStock一次登录）"""
    try:
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        return src.get_price_data_batch(codes, days=250)
    except Exception as e:
        logger.warning(f"BaoStock 批量失败: {e}")
    # 回退逐个获取
    results = {}
    for code in codes:
        df = get_daily(code, start)
        if df is not None:
            results[code] = df
    return results


def get_daily(code: str, start="20240101", end=None) -> pd.DataFrame | None:
    """单股获取（用于回测脚本直接调用）"""
    try:
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        return src.get_price_data(code, days=250)
    except:
        return None


def get_stock_list() -> pd.DataFrame:
    """获取A股列表"""
    try:
        from quant_agent.data.sources.baostock import BaoStockSource
        src = BaoStockSource()
        df = src.get_stock_list()
        if not df.empty:
            return df
    except Exception as e:
        logger.warning(f"BaoStock 列表失败: {e}")

    # 回退: 用内置沪深300池
    logger.info("使用内置股池")
    return pd.DataFrame({"symbol": POOL})


# === 内置股池 ===

POOL = [
    "000001","000002","000063","000333","000338","000425","000538","000568","000651","000661",
    "000725","000776","000858","000895","000938","001979","002007","002120","002129","002142",
    "002230","002236","002241","002304","002352","002415","002459","002460","002475","002493",
    "002555","002594","002601","002602","002709","002714","002812","002841","002916","003816",
    "300003","300014","300015","300033","300059","300124","300142","300223","300274","300347",
    "300394","300408","300413","300418","300433","300454","300457","300496","300529","300601",
    "300628","300676","300750","300760","300782","300832","300896","301269","600009","600010",
    "600016","600019","600025","600028","600029","600030","600031","600036","600048","600050",
    "600061","600085","600089","600104","600111","600115","600150","600153","600160","600176",
    "600177","600196","600208","600219","600230","600271","600276","600282","600299","600309",
    "600332","600346","600362","600369","600383","600390","600406","600436","600438","600486",
    "600489","600498","600519","600521","600570","600588","600589","600596","600600","600606",
    "600690","600703","600745","600809","600837","600845","600859","600887","600893","600900",
    "600905","600918","600919","600941","601006","601012","601066","601088","601111","601127",
    "601138","601166","601211","601225","601228","601236","601288","601318","601328","601336",
    "601390","601398","601601","601628","601633","601668","601669","601688","601728","601766",
    "601788","601799","601818","601857","601881","601899","601901","601919","601939","601985",
    "601989","603160","603259","603288","603369","603501","603799","603833","603986",
]


# === 策略 ===

STRATEGIES = {
    "双均线EMA12/60": lambda df: _sig_dual_ma(df),
    "布林带": lambda df: _sig_bollinger(df),
    "RSI+MACD": lambda df: _sig_rsi_macd(df),
    "ADX趋势": lambda df: _sig_adx(df),
}

def _sig_dual_ma(df):
    close = df["close"]
    e12, e60 = ema(close, 12), ema(close, 60)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[(e12 > e60) & (e12.shift(1) <= e60.shift(1))] = 1
    sig[(e12 < e60) & (e12.shift(1) >= e60.shift(1))] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)

def _sig_bollinger(df):
    close = df["close"]
    upper, _, lower = bollinger_bands(close, 20, 2.0)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[close < lower] = 1
    sig[close > upper] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)

def _sig_rsi_macd(df):
    close = df["close"]
    r = rsi(close, 14)
    _, _, h = macd(close)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[(r < 35) & (h > 0)] = 1
    sig[(r > 65) & (h < 0)] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)

def _sig_adx(df):
    close, high, low = df["close"], df["high"], df["low"]
    e10, e30 = ema(close, 10), ema(close, 30)
    a = adx(high, low, close, 14)
    sig = pd.Series(0.0, index=range(len(close)))
    sig[(e10 > e30) & (a > 25)] = 1
    sig[(e10 < e30) & (a > 25)] = -1
    return sig.replace(0, np.nan).ffill().fillna(0)


# === 选股评分 ===

def score(code: str, df: pd.DataFrame = None) -> dict | None:
    if df is None:
        df = get_daily(code)
    if df is None:
        return None

    close, high, low = df["close"], df["high"], df["low"]
    current = float(close.iloc[-1])
    avg_amt = float(df["amount"].iloc[-20:].mean())

    if avg_amt < 5000 or current < 5 or current > 300:
        return None

    # 技术面 (40)
    tech = 0
    e20, e60 = ema(close, 20), ema(close, 60)
    if len(e60) > 0 and float(e20.iloc[-1]) > float(e60.iloc[-1]):
        tech += 12
        if current > float(e20.iloc[-1]): tech += 5
    _, _, hist = macd(close)
    if float(hist.iloc[-1]) > 0:
        tech += 8
        if float(hist.iloc[-1]) > float(hist.iloc[-2]): tech += 5
    rsi_v = float(rsi(close, 14).iloc[-1])
    if 35 < rsi_v < 65: tech += 5
    elif rsi_v < 35: tech += 3
    if float(adx(high, low, close, 14).iloc[-1]) > 25: tech += 5

    # 动量 (35)
    mom = 0
    r20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
    r60 = float(close.iloc[-1] / close.iloc[-60] - 1) if len(close) >= 60 else 0
    mom += 15 if r20 > 0.1 else (10 if r20 > 0.05 else (5 if r20 > 0 else 0))
    mom += 15 if r60 > 0.2 else (10 if r60 > 0.1 else (5 if r60 > 0 else 0))
    vol = float(close.pct_change().dropna().std() * np.sqrt(252))
    if 0.15 < vol < 0.4: mom += 5
    elif vol < 0.15: mom += 2

    # 流动性 (25)
    liq = 0
    liq += 15 if avg_amt > 50000 else (10 if avg_amt > 20000 else (7 if avg_amt > 10000 else 4))
    avg_v = float(df["vol"].iloc[-20:].mean())
    recent_v = float(df["vol"].iloc[-5:].mean())
    if avg_v > 0:
        vr = recent_v / avg_v
        liq += 10 if 1.2 < vr < 2.0 else (5 if vr >= 2.0 else (5 if vr > 0.8 else 0))

    return {"code": code, "price": current, "score": tech + mom + liq,
            "tech": tech, "mom": mom, "liq": liq}


# === 主流程 ===

def main(top_n=10, strategy_name="双均线EMA12/60", use_all_stocks=False):
    print(f"\n{'='*80}")
    print(f"🦞 AI Quant Agent v3.0 — 自动选股 + 回测")
    print(f"{'='*80}")

    # Step 1: 选股
    if use_all_stocks:
        pool_df = get_stock_list()
        pool = pool_df["symbol"].tolist()
        print(f"\n📊 Step 1: 全市场选股 ({len(pool)} 只) → Top {top_n}")
    else:
        pool = POOL
        print(f"\n📊 Step 1: 沪深300核心池 ({len(pool)} 只) → Top {top_n}")

    daily_cache = get_daily_multi(pool)
    print(f"  获取数据: {len(daily_cache)}/{len(pool)} 只")
    scores = []
    for code in pool:
        df = daily_cache.get(code)
        if df is not None:
            s = score(code, df)
            if s:
                scores.append(s)

    scores.sort(key=lambda x: x["score"], reverse=True)
    selected = scores[:top_n]
    print(f"  入选 {len(selected)} 只:")
    for i, s in enumerate(selected):
        print(f"    {i+1:>2}. {s['code']}  评分:{s['score']:>3}  技术:{s['tech']:>2}  动量:{s['mom']:>2}  流动性:{s['liq']:>2}  价格:{s['price']:>8.2f}")

    if not selected:
        print("❌ 无选股结果"); return

    # Step 2: 回测
    print(f"\n📈 Step 2: 策略 [{strategy_name}] 回测...")
    engine = BacktestEngine(
        initial_capital=100000,
        commission=CommissionModel(commission_rate=0.0003, stamp_tax_rate=0.001),
        slippage=SlippageModel(basis_points=1.0),
    )
    strategy_fn = STRATEGIES[strategy_name]

    results = []
    for s in selected:
        code = s["code"]
        df = daily_cache.get(code)
        if df is None:
            continue
        signals = strategy_fn(df)
        r = engine.run(df, signals)
        results.append({"code": code, "price": s["price"], "score": s["score"],
                        **{k: getattr(r, k) for k in [
                            "total_return", "annual_return", "max_drawdown", "sharpe_ratio",
                            "sortino_ratio", "win_rate", "total_trades", "profit_factor",
                            "max_consecutive_losses"
                        ]}})
        emoji = "🟢" if r.total_return > 0 else "🔴"
        print(f"  {emoji} {code}  收益:{r.total_return:>7.2%}  年化:{r.annual_return:>7.2%}  "
              f"MaxDD:{r.max_drawdown:>7.2%}  Sharpe:{r.sharpe_ratio:>6.3f}  "
              f"胜率:{r.win_rate:>5.0%}  交易:{r.total_trades:>3}次")

    if not results:
        print("❌ 无回测结果"); return

    # Step 3: 汇总
    rdf = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print(f"📋 回测汇总 — {strategy_name}")
    print(f"{'='*80}")
    print(f"{'代码':<8} {'评分':>4} {'总收益':>8} {'年化':>8} {'最大回撤':>8} "
          f"{'Sharpe':>7} {'胜率':>6} {'盈亏比':>6} {'交易':>4}")
    print("-" * 80)
    for _, r in rdf.iterrows():
        emoji = "🟢" if r["total_return"] > 0 else "🔴"
        print(f"{emoji}{r['code']:<7} {r['score']:>4.0f} {r['total_return']:>7.2%} "
              f"{r['annual_return']:>7.2%} {r['max_drawdown']:>7.2%} "
              f"{r['sharpe_ratio']:>7.3f} {r['win_rate']:>5.0%} "
              f"{r['profit_factor']:>6.2f} {int(r['total_trades']):>4}")

    wins = (rdf["total_return"] > 0).sum()
    avg_ret = rdf["total_return"].mean()
    avg_sharpe = rdf["sharpe_ratio"].mean()
    print("-" * 80)
    print(f"  盈利: {wins}/{len(rdf)}  平均收益: {avg_ret:.2%}  平均Sharpe: {avg_sharpe:.3f}")

    best = rdf.loc[rdf["sharpe_ratio"].idxmax()]
    print(f"\n🏆 最优: {best['code']}  Sharpe={best['sharpe_ratio']:.3f}  "
          f"收益={best['total_return']:.2%}  MaxDD={best['max_drawdown']:.2%}")
    print(f"{'='*80}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="🦞 AI Quant Agent — 自动选股+回测")
    p.add_argument("--top", type=int, default=10, help="选股数量")
    p.add_argument("--strategy", default="双均线EMA12/60",
                   choices=list(STRATEGIES.keys()))
    p.add_argument("--all", action="store_true", help="全市场扫描(较慢)")
    p.add_argument("--strategies", action="store_true", help="运行所有策略对比")
    args = p.parse_args()

    if args.strategies:
        for name in STRATEGIES:
            main(top_n=args.top, strategy_name=name)
    else:
        main(top_n=args.top, strategy_name=args.strategy, use_all_stocks=args.all)
