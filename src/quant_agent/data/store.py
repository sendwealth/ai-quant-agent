"""数据存储 — Parquet 文件存储（预留 PG 迁移接口）

特性:
- 原子写入: 写入临时文件后 os.replace()，避免并发写入导致文件损坏
- 文件锁: read-modify-write 周期全程加锁，防止并发追加丢失数据
"""

from __future__ import annotations

import fcntl
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class FileLock:
    """基于 fcntl.flock 的文件锁上下文管理器。

    使用独立的 .lock 文件避免与 Parquet 读写产生冲突。
    在同一文件系统上，多个进程竞争同一 lockfile 时，
    先拿到锁的进程执行 read-modify-write，其余进程阻塞等待。

    用法::

        with FileLock("/data/parquet/financial/300750.lock"):
            # 在此块内，其他进程对同一 lockfile 的 FileLock 会阻塞
            ...
    """

    def __init__(self, lock_path: str | Path) -> None:
        self._lock_path = Path(lock_path)
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR)
        fcntl.flock(self._fd, fcntl.LOCK_EX)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[override]
        self.release()


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """原子写入 Parquet: 先写临时文件再 os.replace()。

    os.replace() 在同一文件系统上是原子操作，保证读者不会看到
    写到一半的文件。临时文件创建在 target 所在目录，确保同一
    文件系统从而保证 rename 的原子性。
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        suffix=".parquet.tmp",
        prefix=target.stem + "_",
        dir=str(target.parent),
    )
    try:
        os.close(fd)
        df.to_parquet(tmp_path, index=False)
        os.replace(tmp_path, str(target))
    except BaseException:
        # 清理残留的临时文件
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class DataStore:
    """数据存储层 — 先用 Parquet，后续迁移到 PostgreSQL

    写操作使用原子写入（write-to-temp + os.replace），
    save_financial 的 read-modify-write 周期使用文件锁保护。
    """

    def __init__(self, base_dir: str = "data/parquet"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save_price(
        self, stock_code: str, df: pd.DataFrame, source: str = "unknown"
    ) -> Path:
        """保存行情数据到 Parquet（原子写入）"""
        if df is None or df.empty:
            return Path("")
        path = self.base_dir / "price" / f"{stock_code}.parquet"
        _atomic_write_parquet(df, path)
        logger.debug(f"保存行情: {path} ({len(df)}行, source={source})")
        return path

    def load_price(self, stock_code: str) -> Optional[pd.DataFrame]:
        """加载行情数据"""
        path = self.base_dir / "price" / f"{stock_code}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        logger.debug(f"加载行情: {path} ({len(df)}行)")
        return df

    def save_financial(
        self, stock_code: str, data: dict, source: str = "unknown"
    ) -> Path:
        """保存财务快照到 Parquet（追加模式，文件锁保护 read-modify-write）

        使用与目标 parquet 同目录的 .lock 文件，确保两个进程同时
        对同一股票追加数据时，不会因交错读写导致数据丢失或文件损坏。
        """
        path = self.base_dir / "financial" / f"{stock_code}.parquet"
        lock_path = path.with_suffix(".lock")

        with FileLock(lock_path):
            df_new = pd.DataFrame([data])
            if path.exists():
                df_existing = pd.read_parquet(path)
                df = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df = df_new

            _atomic_write_parquet(df, path)

        logger.debug(f"保存财务: {path} (source={source})")
        return path

    def load_financial(self, stock_code: str, latest: bool = True) -> Optional[pd.DataFrame]:
        """加载财务数据"""
        path = self.base_dir / "financial" / f"{stock_code}.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        if latest and "report_date" in df.columns:
            df = df.sort_values("report_date", ascending=False).head(1)
        return df

    def list_stocks(self) -> list[str]:
        """列出所有有行情数据的股票代码"""
        price_dir = self.base_dir / "price"
        if not price_dir.exists():
            return []
        return [p.stem for p in price_dir.glob("*.parquet")]

    def is_fresh(self, stock_code: str, max_age_hours: int = 4) -> bool:
        """检查数据是否新鲜"""
        path = self.base_dir / "price" / f"{stock_code}.parquet"
        if not path.exists():
            return False
        import time
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < max_age_hours
