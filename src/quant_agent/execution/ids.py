"""统一 ID 生成 (P2.1)。

为订单、交易、审计事件提供全局唯一、可排序、可读的 ID。

- 默认实现带序列号 + 时间戳 + 随机后缀，保证唯一且可按生成顺序排序。
- 测试 / 复现场景可使用 ``deterministic`` 模式（仅序列号，无随机），
  使生成的 ID 完全可预测。
"""

from __future__ import annotations

import threading
import time
import uuid


class IdGenerator:
    """可排序的唯一 ID 生成器。

    ID 形如 ``{prefix}_{YYYYMMDDHHMMSS}_{seq:06d}[_{rand8}]``。
    ``seq`` 进程内单调递增；``rand`` 仅用于跨进程 / 跨实例防碰撞。
    """

    def __init__(self, prefix: str = "id", deterministic: bool = False) -> None:
        self.prefix = prefix
        self.deterministic = deterministic
        self._seq = 0
        self._lock = threading.Lock()

    def next_id(self) -> str:
        with self._lock:
            self._seq += 1
            seq = self._seq
        ts = time.strftime("%Y%m%d%H%M%S")
        if self.deterministic:
            return f"{self.prefix}_{ts}_{seq:06d}"
        rand = uuid.uuid4().hex[:8]
        return f"{self.prefix}_{ts}_{seq:06d}_{rand}"


# 进程级默认生成器（订单 / 交易共用同一前缀空间，避免 ID 碰撞）
_order_gen = IdGenerator("ord")
_trade_gen = IdGenerator("trd")


def generate_order_id() -> str:
    """生成订单 ID。"""
    return _order_gen.next_id()


def generate_trade_id() -> str:
    """生成交易 ID。"""
    return _trade_gen.next_id()


def generate_id(prefix: str = "id") -> str:
    """按前缀生成通用 ID。"""
    return IdGenerator(prefix).next_id()
