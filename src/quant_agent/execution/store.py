"""SQLite 状态存储 (P2.3) — 替代 JSON，提供原子、线程安全的持久化。

单文件数据库（WAL 模式），每次成交后在一个事务内写入当前完整快照
（cash / commission / positions / trades / equity / orders）。进程重启时
从数据库重建 ``Portfolio`` 与订单列表。原子性由 SQLite 事务保证。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from ..portfolio import CommissionModel, Portfolio, Position, Trade
from .orders import Order

SCHEMA_VERSION = 1


class SqliteStateStore:
    """SQLite 持久化后端，替代原 JSON 快照。"""

    def __init__(self, data_dir: str, db_name: str = "paper_trading.db") -> None:
        self._db_path = Path(data_dir) / "paper_trading" / db_name
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    # -- 路径 / 状态 ------------------------------------------------------

    @property
    def db_path(self) -> Path:
        return self._db_path

    def has_state(self) -> bool:
        try:
            row = self._conn.execute("SELECT 1 FROM state WHERE id = 1").fetchone()
            return row is not None
        except sqlite3.Error:
            return False

    # -- schema -----------------------------------------------------------

    def _init_schema(self) -> None:
        with self._conn:
            self._conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS state ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "cash REAL, commission_rate REAL, stamp_tax_rate REAL, min_commission REAL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS positions ("
                "stock_code TEXT PRIMARY KEY, shares INTEGER, avg_price REAL, "
                "current_price REAL, entry_date TEXT, stop_loss REAL, take_profit REAL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS trades ("
                "id INTEGER PRIMARY KEY, stock_code TEXT, direction TEXT, entry_date TEXT, "
                "exit_date TEXT, entry_price REAL, exit_price REAL, shares INTEGER, "
                "pnl REAL, pnl_pct REAL, commission REAL, status TEXT)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS equity (idx INTEGER PRIMARY KEY, value REAL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS orders (order_id TEXT PRIMARY KEY, payload TEXT)"
            )
            self._conn.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )

    # -- 写入 -------------------------------------------------------------

    def save(self, portfolio: Portfolio, orders: list[Order]) -> None:
        """原子写入完整快照（含订单）。"""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM state")
            self._conn.execute(
                "INSERT INTO state (id, cash, commission_rate, stamp_tax_rate, min_commission) "
                "VALUES (1, ?, ?, ?, ?)",
                (
                    portfolio.cash,
                    portfolio.commission.commission_rate,
                    portfolio.commission.stamp_tax_rate,
                    portfolio.commission.min_commission,
                ),
            )
            self._conn.execute("DELETE FROM positions")
            for p in portfolio.positions.values():
                self._conn.execute(
                    "INSERT INTO positions (stock_code, shares, avg_price, current_price, "
                    "entry_date, stop_loss, take_profit) VALUES (?,?,?,?,?,?,?)",
                    (
                        p.stock_code,
                        p.shares,
                        p.avg_price,
                        p.current_price,
                        p.entry_date,
                        p.stop_loss,
                        p.take_profit,
                    ),
                )
            self._conn.execute("DELETE FROM trades")
            for i, t in enumerate(portfolio.trades):
                self._conn.execute(
                    "INSERT INTO trades (id, stock_code, direction, entry_date, exit_date, "
                    "entry_price, exit_price, shares, pnl, pnl_pct, commission, status) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        i,
                        t.stock_code,
                        t.direction,
                        t.entry_date,
                        t.exit_date,
                        t.entry_price,
                        t.exit_price,
                        t.shares,
                        t.pnl,
                        t.pnl_pct,
                        t.commission,
                        t.status,
                    ),
                )
            self._conn.execute("DELETE FROM equity")
            for i, v in enumerate(portfolio.equity_curve):
                self._conn.execute("INSERT INTO equity (idx, value) VALUES (?,?)", (i, v))
            self._conn.execute("DELETE FROM orders")
            for o in orders:
                self._conn.execute(
                    "INSERT OR REPLACE INTO orders (order_id, payload) VALUES (?,?)",
                    (o.order_id, json.dumps(o.to_dict(), ensure_ascii=False, default=str)),
                )

    # -- 读取 -------------------------------------------------------------

    def load(self, initial_capital: float) -> tuple[Portfolio, list[Order]] | None:
        """重建 Portfolio 与订单列表；无状态返回 None。"""
        if not self.has_state():
            return None
        with self._lock:
            s = self._conn.execute(
                "SELECT cash, commission_rate, stamp_tax_rate, min_commission FROM state WHERE id=1"
            ).fetchone()
            if s is None:
                return None
            cash, cr, sr, mc = s
            commission = CommissionModel(
                commission_rate=cr if cr is not None else 0.0003,
                stamp_tax_rate=sr if sr is not None else 0.001,
                min_commission=mc if mc is not None else 5.0,
            )

            cur = self._conn.execute("SELECT * FROM positions")
            pcols = [d[0] for d in cur.description]
            positions: dict[str, Position] = {}
            for r in cur.fetchall():
                d = dict(zip(pcols, r, strict=True))
                positions[d["stock_code"]] = Position(
                    stock_code=d["stock_code"],
                    shares=d["shares"],
                    avg_price=d["avg_price"],
                    current_price=d["current_price"],
                    entry_date=d.get("entry_date") or "",
                    stop_loss=d.get("stop_loss") or 0.0,
                    take_profit=d.get("take_profit") or 0.0,
                )

            tcur = self._conn.execute("SELECT * FROM trades")
            tcols = [d[0] for d in tcur.description]
            trades: list[Trade] = []
            for r in tcur.fetchall():
                d = dict(zip(tcols, r, strict=True))
                trades.append(
                    Trade(
                        stock_code=d["stock_code"],
                        direction=d["direction"],
                        entry_date=d.get("entry_date") or "",
                        exit_date=d.get("exit_date"),
                        entry_price=d["entry_price"],
                        exit_price=d["exit_price"],
                        shares=d["shares"],
                        pnl=d["pnl"],
                        pnl_pct=d["pnl_pct"],
                        commission=d["commission"],
                        status=d["status"],
                    )
                )

            equity = [
                v for (v,) in self._conn.execute("SELECT value FROM equity ORDER BY idx").fetchall()
            ]

            orders: list[Order] = []
            for (payload,) in self._conn.execute("SELECT payload FROM orders").fetchall():
                try:
                    orders.append(Order.from_dict(json.loads(payload)))
                except Exception:
                    continue

        pf = Portfolio(
            cash=cash if cash is not None else initial_capital,
            positions=positions,
            trades=trades,
            equity_curve=equity,
            commission=commission,
        )
        return pf, orders

    # -- 清理 -------------------------------------------------------------

    def close(self) -> None:
        try:
            self._conn.close()
        except sqlite3.Error:
            pass
