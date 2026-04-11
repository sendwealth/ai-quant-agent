"""智能选股 — 多维度评分筛选（Tushare daily）"""

import os
import logging
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quant_agent.strategy.indicators import rsi, macd, ema, adx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# A股主要成分股池（覆盖沪深300核心）
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
    "601989","603160","601985","603259","603288","603369","603501","603799","603833","603986",
]


def get_daily(code: str) -> pd.DataFrame | None:
    import tushare as ts
    token = os.environ.get("QUANT_TUSHARE_TOKEN") or os.environ.get("TUSHARE_TOKEN")
    ts.set_token(token)
    pro = ts.pro_api()
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=200)).strftime("%Y%m%d")
    try:
        # code -> ts_code
        ts_code = f"{code}.SH" if code.startswith("6") else f"{code}.SZ"
        df = pro.daily(ts_code=ts_code, start_date=start, end_date=end)
        if df is None or len(df) < 60:
            return None
        df = df.sort_values("trade_date").reset_index(drop=True)
        for c in ["close", "high", "low", "vol", "amount"]:
            df[c] = df[c].astype(float)
        return df
    except Exception as e:
        return None


def score(code: str, name: str) -> dict | None:
    df = get_daily(code)
    if df is None:
        return None

    close, high, low = df["close"], df["high"], df["low"]
    current = float(close.iloc[-1])
    avg_amount = float(df["amount"].iloc[-20:].mean())

    # 预筛: 日均成交额<5000万 或 价格异常
    if avg_amount < 5000 or current < 5 or current > 300:
        return None

    # 技术面 (40分)
    tech = 0
    e20 = ema(close, 20)
    e60 = ema(close, 60)
    if len(e60) > 0 and float(e20.iloc[-1]) > float(e60.iloc[-1]):
        tech += 12
        if current > float(e20.iloc[-1]):
            tech += 5
    _, _, hist = macd(close)
    if float(hist.iloc[-1]) > 0:
        tech += 8
        if float(hist.iloc[-1]) > float(hist.iloc[-2]):
            tech += 5
    rsi_val = float(rsi(close, 14).iloc[-1])
    if 35 < rsi_val < 65:
        tech += 5
    elif rsi_val < 35:
        tech += 3
    if float(adx(high, low, close, 14).iloc[-1]) > 25:
        tech += 5

    # 动量 (35分)
    mom = 0
    r20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
    r60 = float(close.iloc[-1] / close.iloc[-60] - 1) if len(close) >= 60 else 0
    mom += 15 if r20 > 0.1 else (10 if r20 > 0.05 else (5 if r20 > 0 else 0))
    mom += 15 if r60 > 0.2 else (10 if r60 > 0.1 else (5 if r60 > 0 else 0))
    vol = float(close.pct_change().dropna().std() * np.sqrt(252))
    if 0.15 < vol < 0.4:
        mom += 5
    elif vol < 0.15:
        mom += 2

    # 流动性 (25分)
    liq = 0
    liq += 15 if avg_amount > 50000 else (10 if avg_amount > 20000 else (7 if avg_amount > 10000 else 4))
    avg_vol = float(df["vol"].iloc[-20:].mean())
    recent_vol = float(df["vol"].iloc[-5:].mean())
    if avg_vol > 0:
        vr = recent_vol / avg_vol
        liq += 10 if 1.2 < vr < 2.0 else (5 if vr >= 2.0 else (5 if vr > 0.8 else 0))

    return {
        "code": code, "name": name, "price": current,
        "score": tech + mom + liq,
        "tech": tech, "momentum": mom, "liquidity": liq,
        "amount_20d": avg_amount,
        "ret_20d": r20, "ret_60d": r60,
        "volatility": vol, "rsi": rsi_val,
    }


def main():
    logger.info(f"🔍 智能选股 — 股票池 {len(POOL)} 只")

    results = []
    for i, code in enumerate(POOL):
        logger.info(f"  [{i+1}/{len(POOL)}] {code}")
        s = score(code, code)
        if s:
            results.append(s)

    df = pd.DataFrame(results)
    if df.empty:
        print("无结果"); return
    df = df.sort_values("score", ascending=False).head(20).reset_index(drop=True)

    print(f"\n{'='*90}")
    print(f"🦞 AI Quant Agent v3.0 — 智能选股 Top {len(df)}")
    print(f"{'='*90}")
    print(f"{'#':>2} {'代码':<8} {'价格':>8} {'评分':>4} "
          f"{'技术':>4} {'动量':>4} {'流动性':>4} {'成交额亿':>8} "
          f"{'20日':>7} {'60日':>7} {'RSI':>5}")
    print("-" * 90)
    for i, r in df.iterrows():
        print(f"{i+1:>2} {r['code']:<8} {r['price']:>8.2f} "
              f"{r['score']:>4.0f} {r['tech']:>4.0f} {r['momentum']:>4.0f} "
              f"{r['liquidity']:>4.0f} {r['amount_20d']/10000:>8.1f} "
              f"{r['ret_20d']:>6.1%} {r['ret_60d']:>6.1%} {r['rsi']:>5.1f}")
    print("=" * 90)
    codes = ", ".join(f"{r['code']}" for _, r in df.iterrows())
    print(f"\n🎯 入选: {codes}")
    return df


if __name__ == "__main__":
    main()
