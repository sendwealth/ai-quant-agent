"""Tushare 数据源适配器 — 财务报表 + 行情"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Optional

import pandas as pd

from ..rate_limiter import RateLimiter
from .base import DataSource, FinancialSnapshot, StatementType

logger = logging.getLogger(__name__)


# Exceptions that are safe to retry (transient I/O errors)
_RETRYABLE = (ConnectionError, TimeoutError, OSError)


def retry(max_retries: int = 3, backoff_base: float = 2.0):
    """重试装饰器（指数退避，仅重试网络/IO 错误）"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_err = None
            for attempt in range(1, max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except _RETRYABLE as e:
                    last_err = e
                    if attempt < max_retries:
                        wait = backoff_base ** (attempt - 1)
                        logger.warning(f"{func.__name__} 第{attempt}次失败 (retryable): {e}, {wait:.0f}s后重试")
                        time.sleep(wait)
                    else:
                        raise
                except Exception:
                    # Non-retryable error — propagate immediately
                    raise
            raise last_err  # type: ignore  # pragma: no cover
        return wrapper
    return decorator


class TushareSource(DataSource):
    """Tushare Pro 数据源 — 主力财务数据来源"""

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ):
        self._token = token
        self._pro = None
        self._rate_limiter = rate_limiter or RateLimiter(
            max_calls=200, period=60.0,
        )

    def __repr__(self) -> str:
        token_status = "set" if self._token else "unset"
        return f"TushareSource(token={token_status})"

    def _init_api(self):
        """延迟初始化 Tushare API"""
        if self._pro is not None:
            return
        if not self._token:
            raise RuntimeError(
                "Tushare token 未配置。请通过 Settings.tushare_token 传入 "
                "或设置环境变量 QUANT_TUSHARE_TOKEN"
            )
        import tushare as ts
        self._pro = ts.pro_api(self._token)
        logger.info("Tushare Pro API initialized (per-instance token)")

    @property
    def name(self) -> str:
        return "tushare"

    @property
    def available(self) -> bool:
        try:
            self._init_api()
            return True
        except Exception:
            return False

    def _to_ts_code(self, stock_code: str) -> str:
        """转换股票代码: 300750 → 300750.SZ, 600519 → 600519.SH, 830799 → 830799.BJ"""
        if stock_code.startswith("6"):
            return f"{stock_code}.SH"
        if stock_code.startswith("8"):
            return f"{stock_code}.BJ"
        return f"{stock_code}.SZ"

    @retry(max_retries=3)
    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> Optional[pd.DataFrame]:
        """获取历史行情"""
        try:
            self._init_api()
            self._rate_limiter.block_until_ready()
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime("%Y%m%d")
            start_date = (datetime.now() - timedelta(days=int(days * 1.5))).strftime("%Y%m%d")

            df = self._pro.daily(
                ts_code=self._to_ts_code(stock_code),
                start_date=start_date,
                end_date=end_date,
            )
            if df is None or df.empty:
                return None

            # 标准化列名
            df = df.rename(columns={
                "trade_date": "date",
                "vol": "volume",
                "pct_chg": "pct_change",
            })
            df = df.sort_values("date").reset_index(drop=True)
            # 保留最后 days 条
            df = df.tail(days).reset_index(drop=True)

            # 类型转换
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            logger.info(f"✅ Tushare: 获取 {stock_code} {len(df)} 天行情")
            return df[["date", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logger.warning(f"Tushare 行情获取失败: {e}")
            return None

    @retry(max_retries=3)
    def get_realtime_price(self, stock_code: str) -> Optional[float]:
        """获取最新收盘价"""
        try:
            self._init_api()
            self._rate_limiter.block_until_ready()
            df = self._pro.daily(
                ts_code=self._to_ts_code(stock_code),
                limit=1,
            )
            if df is not None and not df.empty:
                return float(df["close"].iloc[0])
        except Exception as e:
            logger.warning(f"Tushare 实时价格获取失败: {e}")
        return None

    @retry(max_retries=3)
    def get_financial_statements(
        self, stock_code: str, statement_type: StatementType, periods: int = 4
    ) -> Optional[pd.DataFrame]:
        """获取财务报表（真实数据）"""
        try:
            self._init_api()
            self._rate_limiter.block_until_ready()
            ts_code = self._to_ts_code(stock_code)

            if statement_type == StatementType.INCOME:
                df = self._pro.income(ts_code=ts_code, limit=periods)
            elif statement_type == StatementType.BALANCE:
                df = self._pro.balancesheet(ts_code=ts_code, limit=periods)
            elif statement_type == StatementType.CASHFLOW:
                df = self._pro.cashflow(ts_code=ts_code, limit=periods)
            elif statement_type == StatementType.INDICATORS:
                df = self._pro.fina_indicator(ts_code=ts_code, limit=periods)
            else:
                return None

            if df is None or df.empty:
                return None

            # 按日期降序（最新在前）
            if "end_date" in df.columns:
                df = df.sort_values("end_date", ascending=False).reset_index(drop=True)

            logger.info(f"✅ Tushare: 获取 {stock_code} {statement_type.value} ({len(df)}期)")
            return df
        except Exception as e:
            logger.warning(f"Tushare 财务报表获取失败: {e}")
            return None

    @retry(max_retries=3)
    def get_financial_snapshot(self, stock_code: str) -> Optional[FinancialSnapshot]:
        """获取财务快照 — 从真实报表计算核心指标"""
        try:
            self._init_api()
            ts_code = self._to_ts_code(stock_code)

            # 1. 获取财务指标（tushare fina_indicator 包含 ROE、毛利率等）
            self._rate_limiter.block_until_ready()
            indicators = self._pro.fina_indicator(ts_code=ts_code, limit=1)
            if indicators is None or indicators.empty:
                logger.warning(f"Tushare: {stock_code} 无财务指标数据")
                return None

            row = indicators.iloc[0]

            # 2. 获取利润表计算真实 ROE
            self._rate_limiter.block_until_ready()
            income = self._pro.income(ts_code=ts_code, limit=2)
            self._rate_limiter.block_until_ready()
            balance = self._pro.balancesheet(ts_code=ts_code, limit=2)

            data: dict[str, Any] = {}

            # 从 fina_indicator 直接获取
            data["roe"] = self._safe_float(row.get("roe"))
            data["gross_margin"] = self._safe_float(row.get("grossprofit_margin"))
            data["net_margin"] = self._safe_float(row.get("netprofit_margin"))
            data["debt_ratio"] = self._safe_float(row.get("debt_to_assets"))
            data["current_ratio"] = self._safe_float(row.get("current_ratio"))

            # 营收增长和利润增长
            if income is not None and len(income) >= 2:
                latest_rev = self._safe_float(income.iloc[0].get("total_revenue"))
                prev_rev = self._safe_float(income.iloc[1].get("total_revenue"))
                latest_profit = self._safe_float(income.iloc[0].get("net_profit"))
                prev_profit = self._safe_float(income.iloc[1].get("net_profit"))

                data["revenue_growth"] = (
                    (latest_rev - prev_rev) / prev_rev if prev_rev > 0 else None
                )
                data["profit_growth"] = (
                    (latest_profit - prev_profit) / prev_profit if prev_profit > 0 else None
                )
            else:
                data["revenue_growth"] = None
                data["profit_growth"] = None

            # 从利润表和资产负债表交叉验证 ROE
            if income is not None and balance is not None and len(income) >= 1 and len(balance) >= 1:
                net_profit = self._safe_float(income.iloc[0].get("net_profit"))
                total_equity = self._safe_float(balance.iloc[0].get("total_equity"))
                prev_equity = (
                    self._safe_float(balance.iloc[1].get("total_equity"))
                    if len(balance) >= 2 else total_equity
                )
                avg_equity = (total_equity + prev_equity) / 2 if (total_equity + prev_equity) > 0 else None

                if net_profit and avg_equity and avg_equity > 0:
                    calc_roe = net_profit / avg_equity
                    # 如果 fina_indicator 的 ROE 为空或异常，使用计算值
                    if data["roe"] is None or abs(data["roe"]) > 1:
                        data["roe"] = calc_roe
                    data["roe_calc"] = calc_roe  # 保留计算值用于审计

            # 获取估值指标（每日行情接口）
            self._rate_limiter.block_until_ready()
            daily = self._pro.daily_basic(
                ts_code=ts_code, fields="pe_ttm, pb, ps_ttm, total_mv", limit=1
            )
            if daily is not None and not daily.empty:
                d = daily.iloc[0]
                data["pe_ttm"] = self._safe_float(d.get("pe_ttm"))
                data["pb"] = self._safe_float(d.get("pb"))
                data["ps_ttm"] = self._safe_float(d.get("ps_ttm"))
                data["total_mv"] = self._safe_float(d.get("total_mv"))  # 总市值（万元）

            # 获取实时价格
            price = self.get_realtime_price(stock_code)
            data["price"] = price

            # 报告期
            data["report_date"] = str(row.get("end_date", ""))

            logger.info(
                f"✅ Tushare: {stock_code} 财务快照 "
                f"ROE={data.get('roe')}, P/E={data.get('pe_ttm')}, P/B={data.get('pb')}"
            )
            return FinancialSnapshot(stock_code, data)

        except Exception as e:
            logger.error(f"Tushare 财务快照获取失败: {e}")
            return None

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """安全转换为 float，None/NaN/异常值返回 None"""
        if value is None:
            return None
        try:
            v = float(value)
            return None if pd.isna(v) else v
        except (ValueError, TypeError):
            return None
