"""WeStock 数据源适配器 — 腾讯自选股行情数据接口 (westock-data-clawhub)

通过 `npx westock-data-clawhub@1.0.4` CLI 获取数据，解析 Markdown 表格输出。

数据源能力（腾讯自选股）：
- K线 (kline)：个股/指数/板块/ETF，日/周/月/季/年，前/后复权
- 分时 (minute)：个股/指数/板块
- 财务报表 (finance)：利润表/资产负债表/现金流量表 (A股/港股/美股)
- 公司简况 (profile)
- 资金流向 (asfund/hkfund/usfund)
- 技术指标 (technical)：MACD/KDJ/RSI/BOLL/BIAS/WR/DMI 等
- 筹码成本 (chip)、股东结构 (shareholder)、分红 (dividend) 等

代码格式转换：
- 本项目内部使用 6 位纯数字 A 股代码 (如 "300750")
- WeStock 使用带前缀格式 ("sz300750" / "sh600000" / "bj430047")
- A股: 60/68/9 开头 -> sh, 00/30/20/30 开头 -> sz, 8/4 开头 -> bj

注意事项：
- 货币单位：港股返回港元/美元，美股返回美元；A股为人民币
- CLI 输出前会打印 NVS 帮助到 stdout，需用 `--version` 探测或过滤；
  实际数据输出为 Markdown 表格，表头行以 `|` 开头且含列名
- npx 首次运行会下载包，属中等供应链风险（已在技能说明中声明）
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from typing import Any

import pandas as pd

from ..rate_limiter import RateLimiter
from .base import DataSource, FinancialSnapshot

logger = logging.getLogger(__name__)

# WeStock CLI 包名与版本（固定版本以确保可复现）
_WESTOCK_PKG = "westock-data-clawhub@1.0.4"

# 重试仅针对瞬态错误
_RETRYABLE = (ConnectionError, TimeoutError, OSError)

# 复权类型映射：项目 adjust -> westock fq 参数
_FQ_MAP = {"qfq": "qfq", "hfq": "hfq", "": "bfq", None: "qfq"}


def to_westock_code(stock_code: str) -> str:
    """将项目内部 6 位代码转换为 WeStock 带前缀格式。

    Args:
        stock_code: 6 位 A 股代码 (如 "300750", "600000", "430047")

    Returns:
        WeStock 格式代码 (如 "sz300750", "sh600000", "bj430047")
    """
    code = stock_code.strip()
    if code[:2].lower() in ("sh", "sz", "bj", "hk"):
        return code  # 已经是带前缀格式
    if len(code) != 6 or not code.isdigit():
        raise ValueError(f"Invalid 6-digit A-share code: {stock_code!r}")
    prefix = code[:2]
    if prefix in ("60", "68", "90", "88", "9"):
        return f"sh{code}"
    if prefix in ("00", "30", "20", "02", "03"):
        return f"sz{code}"
    if prefix[0] in ("8", "4"):  # 北交所: 8/4 开头 (83/87/43/92 等)
        return f"bj{code}"
    # 默认按深交所处理
    return f"sz{code}"


def from_westock_code(westock_code: str) -> str:
    """从 WeStock 带前缀代码提取 6 位纯数字代码。

    仅对 A 股 (sh/sz/bj) 有效；港股/美股返回原样。
    """
    code = westock_code.strip().lower()
    for p in ("sh", "sz", "bj"):
        if code.startswith(p):
            return code[len(p) :]
    return westock_code


def _parse_markdown_table(text: str) -> pd.DataFrame | None:
    """解析 CLI 输出的 Markdown 表格。

    兼容多表输出（用空行/标题分隔），返回第一个有效数据表。

    Args:
        text: CLI stdout 文本

    Returns:
        解析后的 DataFrame，无表时返回 None
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    # 收集连续的表格行 (以 | 开头)
    tables: list[list[str]] = []
    current: list[str] = []
    for ln in lines:
        if ln.startswith("|") or (ln.startswith("|:") or set(ln) <= set("|-: ")):
            if ln.startswith("|"):
                current.append(ln)
            elif current:
                # 分隔行 (|---|---|)
                current.append(ln)
        else:
            if current:
                tables.append(current)
                current = []
    if current:
        tables.append(current)

    for tbl in tables:
        # 过滤掉分隔行，保留表头 + 数据行
        data_rows = [r for r in tbl if r.startswith("|")]
        if len(data_rows) < 2:
            continue
        header = [c.strip() for c in data_rows[0].strip("|").split("|")]
        # 分隔行通常是第二行
        if len(data_rows) >= 3 and set(data_rows[1]).issubset(set("|-: ")):
            body = data_rows[2:]
        else:
            body = data_rows[1:]
        records = []
        for row in body:
            cells = [c.strip() for c in row.strip("|").split("|")]
            if len(cells) != len(header):
                continue
            records.append(dict(zip(header, cells, strict=False)))
        if records:
            return pd.DataFrame(records)

    return None


class WeStockSource(DataSource):
    """WeStock 数据源 — 腾讯自选股接口，通过 npx CLI 调用。

    免费、无需 token，支持 A股/港股/美股行情与财务。
    作为降级链中的稳定免费源（优先级高于 AkShare）。
    """

    def __init__(
        self,
        enabled: bool = True,
        max_retries: int = 3,
        rate_limiter: RateLimiter | None = None,
        timeout: float = 60.0,
    ):
        self._enabled = enabled
        self.max_retries = max_retries
        self.timeout = timeout
        self._rate_limiter = rate_limiter or RateLimiter(
            max_calls=60,  # 保守限速，腾讯接口较敏感
            period=60.0,
        )
        self._npx_available = shutil.which("npx") is not None

    @property
    def name(self) -> str:
        return "westock"

    @property
    def available(self) -> bool:
        if not self._enabled:
            return False
        if not self._npx_available:
            logger.warning("WeStock: npx not found, source unavailable")
            return False
        return True

    # ── CLI 调用 ───────────────────────────────────────────────────────

    def _run_cli(self, *args: str) -> str | None:
        """调用 westock CLI 并返回 stdout 文本。

        Returns:
            CLI stdout 文本；失败返回 None
        """
        if not self.available:
            return None
        self._rate_limiter.block_until_ready()
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                proc = subprocess.run(
                    ["npx", "-y", _WESTOCK_PKG, *args],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if proc.returncode != 0:
                    err = proc.stderr.strip() or proc.stdout.strip()
                    raise RuntimeError(f"CLI exit {proc.returncode}: {err[:200]}")
                return proc.stdout
            except (_RETRYABLE, subprocess.TimeoutExpired) as e:
                last_err = e
                if attempt < self.max_retries:
                    wait = 2 ** (attempt - 1)
                    logger.warning(f"WeStock CLI attempt {attempt} failed: {e}, retry in {wait}s")
                    time.sleep(wait)
            except Exception as e:
                logger.warning(f"WeStock CLI non-transient error: {e}")
                return None
        logger.error(f"WeStock CLI exhausted retries: {last_err}")
        return None

    def _query_table(self, *args: str) -> pd.DataFrame | None:
        """运行 CLI 并解析为 DataFrame。"""
        out = self._run_cli(*args)
        if not out:
            return None
        return _parse_markdown_table(out)

    # ── 行情数据 ───────────────────────────────────────────────────────

    def get_price_data(
        self, stock_code: str, days: int = 250, adjust: str = "qfq"
    ) -> pd.DataFrame | None:
        """获取历史日线行情 (kline)。"""
        try:
            ws_code = to_westock_code(stock_code)
            fq = _FQ_MAP.get(adjust, "qfq")
            limit = max(days, 5)
            df = self._query_table(
                "kline", ws_code, "--period", "day", "--limit", str(limit), "--fq", fq
            )
            if df is None or df.empty:
                return None

            # 列名映射：westock 用 last 表示收盘
            rename = {
                "date": "date",
                "open": "open",
                "last": "close",
                "high": "high",
                "low": "low",
                "volume": "volume",
                "amount": "amount",
            }
            df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

            for col in ["open", "high", "low", "close", "volume", "amount"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")

            df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
            # 保留必要的标准列
            keep = [
                c
                for c in ["date", "open", "high", "low", "close", "volume", "amount"]
                if c in df.columns
            ]
            df = df[keep]
            if df.empty:
                return None
            logger.info(f"WeStock: fetched {stock_code} {len(df)} bars")
            return df
        except Exception as e:
            logger.warning(f"WeStock price data failed for {stock_code}: {e}")
            return None

    def get_realtime_price(self, stock_code: str) -> float | None:
        """获取实时价格 — 从最新日线 kline 的 last 列获取。"""
        try:
            df = self.get_price_data(stock_code, days=1, adjust="qfq")
            if df is not None and not df.empty and "close" in df.columns:
                price = float(df.iloc[-1]["close"])
                if price > 0:
                    return price
        except Exception as e:
            logger.warning(f"WeStock realtime price failed for {stock_code}: {e}")
        return None

    # ── 财务数据 ───────────────────────────────────────────────────────

    def get_financial_snapshot(self, stock_code: str) -> FinancialSnapshot | None:
        """获取财务快照 — 从 lrb(利润表) / zcfz(资产负债表) 提取核心指标。

        WeStock finance 返回的是原始报表，需计算衍生指标：
        - ROE = 归母净利润 / 净资产（近似）
        - 毛利率 = (营业收入 - 营业成本) / 营业收入
        - 净利率 = 净利润 / 营业收入
        """
        try:
            ws_code = to_westock_code(stock_code)
            lrb = self._query_table("finance", ws_code, "--type", "lrb", "--num", "4")
            zcfz = self._query_table("finance", ws_code, "--type", "zcfz", "--num", "4")
            if lrb is None or lrb.empty:
                return None

            data: dict[str, Any] = {}

            # 取最新一期 (第一行通常是最近期)
            latest_lrb = lrb.iloc[0]
            latest_zcfz = zcfz.iloc[0] if zcfz is not None and not zcfz.empty else None

            def _num(val: Any) -> float | None:
                try:
                    if val is None or val in ("", "-", "None"):
                        return None
                    return float(str(val).replace(",", ""))
                except (ValueError, TypeError):
                    return None

            # 利润表字段 (A股)
            revenue = _num(latest_lrb.get("OperatingRevenue")) or _num(
                latest_lrb.get("TotalOperatingRevenue")
            )
            profit = _num(latest_lrb.get("NPParentCompanyOwners")) or _num(
                latest_lrb.get("NetProfit")
            )
            total_cost = _num(latest_lrb.get("TotalOperatingCost"))

            if revenue and revenue > 0:
                if total_cost:
                    gm = (revenue - total_cost) / revenue
                    data["gross_margin"] = gm
                if profit is not None:
                    data["net_margin"] = profit / revenue

            if latest_zcfz is not None:
                equity = _num(latest_zcfz.get("TotalOwnerEquity")) or _num(
                    latest_zcfz.get("SEWithoutMI")
                )
                total_assets = _num(latest_zcfz.get("TotalAssets"))
                total_liab = _num(latest_zcfz.get("TotalLiabilities"))
                if equity and equity > 0 and profit is not None:
                    data["roe"] = profit / equity
                if total_assets and total_assets > 0 and total_liab is not None:
                    data["debt_ratio"] = total_liab / total_assets

            # 报告期
            for col in ("EndDate", "InfoPublDate", "_date"):
                if col in latest_lrb.index:
                    data["report_date"] = str(latest_lrb[col])[:10]
                    break

            # 当前价格
            price = self.get_realtime_price(stock_code)
            if price:
                data["price"] = price

            if not data:
                return None
            snapshot = FinancialSnapshot(stock_code, data)
            logger.info(
                f"WeStock: financial snapshot for {stock_code}, "
                f"{len([v for v in data.values() if v is not None])} fields"
            )
            return snapshot
        except Exception as e:
            logger.warning(f"WeStock financial snapshot failed for {stock_code}: {e}")
            return None

    def get_financial_statements(
        self, stock_code: str, statement_type: Any, periods: int = 4
    ) -> pd.DataFrame | None:
        """获取原始财务报表 (lrb/zcfz/xjll)。"""
        try:
            from .base import StatementType

            ws_code = to_westock_code(stock_code)
            type_map = {
                StatementType.INCOME: "lrb",
                StatementType.BALANCE: "zcfz",
                StatementType.CASHFLOW: "xjll",
            }
            wt = type_map.get(StatementType(statement_type), "lrb")
            return self._query_table("finance", ws_code, "--type", wt, "--num", str(periods))
        except Exception as e:
            logger.warning(f"WeStock financial statements failed for {stock_code}: {e}")
            return None

    # ── 技术指标 (增强) ─────────────────────────────────────────────────

    def get_technical_indicators(self, stock_code: str) -> pd.DataFrame | None:
        """获取技术指标 (MACD/RSI/KDJ/BOLL 等)。

        返回最新一期指标，可作为技术分析的补充数据源。
        """
        try:
            ws_code = to_westock_code(stock_code)
            return self._query_table("technical", ws_code, "--group", "all")
        except Exception as e:
            logger.warning(f"WeStock technical failed for {stock_code}: {e}")
            return None

    # ── 扩展数据 (平台特色) ───────────────────────────────────────────────────

    def search_stock(self, keyword: str) -> pd.DataFrame | None:
        """搜索股票/基金/板块 (search 命令，不支持批量)。

        Args:
            keyword: 股票名称或代码关键字 (如 "宁德时代", "银行")

        Returns:
            DataFrame with columns: code, name, type
        """
        try:
            return self._query_table("search", keyword)
        except Exception as e:
            logger.warning(f"WeStock search failed for {keyword}: {e}")
            return None

    def get_minute_data(self, stock_code: str, days: int = 1) -> pd.DataFrame | None:
        """获取分时数据 (minute 命令，不支持批量)。

        Args:
            stock_code: 6 位 A 股代码
            days: 1=当日分时, 5=近5日分时

        Returns:
            DataFrame with columns: code, time, price, volume, amount
        """
        try:
            ws_code = to_westock_code(stock_code)
            if days and days > 1:
                return self._query_table("minute", ws_code, "--days", str(days))
            return self._query_table("minute", ws_code)
        except Exception as e:
            logger.warning(f"WeStock minute failed for {stock_code}: {e}")
            return None

    def get_company_profile(self, stock_code: str) -> dict[str, Any] | None:
        """获取公司简况 (profile 命令)。

        Returns:
            公司基本信息 dict (name/industry/sector/listedDate/website 等)
        """
        try:
            ws_code = to_westock_code(stock_code)
            df = self._query_table("profile", ws_code)
            if df is None or df.empty:
                return None
            return df.iloc[0].to_dict()
        except Exception as e:
            logger.warning(f"WeStock profile failed for {stock_code}: {e}")
            return None

    def get_fund_flow(self, stock_code: str, date: str | None = None) -> pd.DataFrame | None:
        """获取 A 股资金流向 (asfund 命令)。

        Args:
            stock_code: 6 位 A 股代码
            date: 可选日期 YYYY-MM-DD

        Returns:
            DataFrame with 主力净流入/散户/大宗等字段
        """
        try:
            ws_code = to_westock_code(stock_code)
            args = ["asfund", ws_code]
            if date:
                args += ["--date", date]
            return self._query_table(*args)
        except Exception as e:
            logger.warning(f"WeStock fund flow failed for {stock_code}: {e}")
            return None

    def get_chip(self, stock_code: str) -> pd.DataFrame | None:
        """获取筹码成本分布 (chip 命令，仅支持沪深A股)。

        Returns:
            DataFrame with columns: closePrice, chipProfitRate, chipAvgCost,
            chipConcentration90, chipConcentration70
        """
        try:
            ws_code = to_westock_code(stock_code)
            return self._query_table("chip", ws_code)
        except Exception as e:
            logger.warning(f"WeStock chip failed for {stock_code}: {e}")
            return None

    def get_shareholders(self, stock_code: str) -> list[dict[str, Any]] | None:
        """获取股东结构 (shareholder 命令，支持 A股/港股)。

        返回多个段落 (十大股东/十大流通股东/股东户数) 的列表，
        每段为 {section, date, rows: [...]}.
        """
        try:
            ws_code = to_westock_code(stock_code)
            out = self._run_cli("shareholder", ws_code)
            if not out:
                return None
            return self._parse_shareholder(out)
        except Exception as e:
            logger.warning(f"WeStock shareholder failed for {stock_code}: {e}")
            return None

    def get_dividend(self, stock_code: str, years: int | None = None) -> pd.DataFrame | None:
        """获取分红数据 (dividend 命令)。

        Args:
            stock_code: 6 位 A 股代码
            years: 可选，返回近 N 年分红

        Returns:
            DataFrame with columns: reportEndDate, cashDiviRMB, dividendPlan 等
        """
        try:
            ws_code = to_westock_code(stock_code)
            args = ["dividend", ws_code]
            if years:
                args += ["--years", str(years)]
            return self._query_table(*args)
        except Exception as e:
            logger.warning(f"WeStock dividend failed for {stock_code}: {e}")
            return None

    @staticmethod
    def _parse_shareholder(text: str) -> list[dict[str, Any]]:
        """解析 shareholder 命令输出 (带 #### 标题的多段表格)。

        输出格式示例::

            #### sh600519 贵州茅台 (2026-03-31)

            **十大股东**

            | no | name | holdShares | holdPct | holdChange |
            | --- | --- | --- | --- | --- |
            ...
        """
        sections: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None
        section_title: str | None = None
        buf: list[str] = []

        def _save_section() -> None:
            """将当前累积的表格行存入 current['rows']，若已有内容则归档。"""
            nonlocal current, section_title, buf
            if current is None:
                buf = []
                return
            if buf:
                df = _parse_markdown_table("\n".join(buf))
                if df is not None and not df.empty:
                    current.setdefault("sections", []).append(
                        {
                            "section": section_title,
                            "rows": df.to_dict(orient="records"),
                        }
                    )
            buf = []

        for line in text.splitlines():
            if line.startswith("####"):
                # 新股票标题：先归档旧 current，再开新 current
                _save_section()
                if current is not None:
                    sections.append(current)
                header = line[4:].strip()
                import re as _re

                m = _re.search(r"\((\d{4}-\d{2}-\d{2})\)", header)
                current = {
                    "raw": header,
                    "date": m.group(1) if m else None,
                    "sections": [],
                }
                section_title = None
            elif line.startswith("**") and line.endswith("**"):
                # 段落小标题 (如 **十大股东**)：先保存上一段表格，再切换标题
                _save_section()
                section_title = line.strip("*")
            elif line.startswith("|"):
                buf.append(line)
            # 其它行 (空行/说明) 忽略，不重置 buf (由下一段落触发保存)
        # 收尾
        _save_section()
        if current is not None:
            sections.append(current)
        return sections
