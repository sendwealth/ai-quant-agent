"""回测可信度增强（推荐 #4）— 样本外 walk-forward 验证与 point-in-time 校验。

回测指标完整 ≠ 策略已验证。本模块补齐「可信回测」所需的两块拼图：

1. **样本外验证（walk-forward）**：把历史按时间顺序切成多段
   ``[训练] -> [测试]`` 折，每折只在训练集上确定信号逻辑、在紧随其后的测试集
   上评估，**测试区间严格在训练区间之后**（无前视泄漏），从而量化样本外
   （out-of-sample, OOS）绩效与样本内（in-sample）的衰减。

2. **point-in-time 校验**：检测信号是否存在长度越界、含 NaN 等典型前视/对齐
   错误，作为回测可信度的最低门槛。

所有切分与校验均为纯函数，便于单元测试；不触网、不依赖具体数据源。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .engine import BacktestEngine, BacktestResult


def walk_forward_splits(
    n: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[tuple[range, range]]:
    """生成无泄漏的 walk-forward 索引切分（按时间顺序，不重叠、不交叉）。

    Args:
        n: 总样本点数（如交易日数）。
        train_size: 每折训练窗口长度。
        test_size: 每折测试窗口长度。
        step: 折与折之间的滑动步长；默认等于 ``test_size``（不重叠测试）。

    Returns:
        列表，元素为 ``(train_range, test_range)``；保证
        ``test_range`` 严格位于 ``train_range`` 之后，杜绝前视泄漏。

    示例::

        >>> walk_forward_splits(10, train_size=4, test_size=2)
        [(range(0,4), range(4,6)), (range(2,6), range(6,8)), (range(4,8), range(8,10))]
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size 与 test_size 必须为正")
    step = test_size if step is None else step
    if step <= 0:
        raise ValueError("step 必须为正")
    splits: list[tuple[range, range]] = []
    start = 0
    while start + train_size + test_size <= n:
        train = range(start, start + train_size)
        test = range(start + train_size, start + train_size + test_size)
        splits.append((train, test))
        start += step
    return splits


def validate_point_in_time(
    price_data: pd.DataFrame,
    signals: pd.Series | None = None,
) -> list[str]:
    """point-in-time 校验：返回发现的问题列表（空列表表示通过）。

    检查项：
    - signals 长度不得越过 price_data（典型前视泄漏）。
    - signals 不得含 NaN（未对齐/缺失）。
    - price_data 须按时间升序（``date`` 列单调）。

    注意：本函数仅做结构层面的前视防护；真正「当日信号只能用当日及之前数据」
    由策略接口 ``StrategyContext(price, prev_price, ...)`` 在逐日模拟中结构性保证。
    """
    issues: list[str] = []
    if signals is not None:
        if len(signals) > len(price_data):
            issues.append(
                f"signals 长度({len(signals)})超过价格数据({len(price_data)})，可能存在前视泄漏"
            )
        if int(signals.isna().sum()) > 0:
            issues.append(f"signals 含 {int(signals.isna().sum())} 个 NaN（未对齐/缺失）")
    if "date" in price_data.columns and len(price_data) > 1:
        dates = pd.to_datetime(price_data["date"], errors="coerce", format="mixed")
        if dates.notna().all() and not dates.is_monotonic_increasing:
            issues.append("价格数据未按时间升序排列（point-in-time 前提被破坏）")
    return issues


@dataclass
class WalkForwardFold:
    """单折 walk-forward 结果。"""

    train_idx: range
    test_idx: range
    in_sample: BacktestResult
    out_of_sample: BacktestResult


@dataclass
class WalkForwardReport:
    """walk-forward 验证汇总。"""

    folds: list[WalkForwardFold] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def _oos_means(self, attr: str) -> float:
        vals = [getattr(f.out_of_sample, attr) for f in self.folds]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        return float(arr.mean()) if arr.size else 0.0

    @property
    def oos_total_return(self) -> float:
        return self._oos_means("total_return")

    @property
    def oos_sharpe(self) -> float:
        return self._oos_means("sharpe_ratio")

    @property
    def oos_max_drawdown(self) -> float:
        return self._oos_means("max_drawdown")

    @property
    def in_sample_sharpe(self) -> float:
        vals = [f.in_sample.sharpe_ratio for f in self.folds]
        arr = np.array([v for v in vals if np.isfinite(v)], dtype=float)
        return float(arr.mean()) if arr.size else 0.0

    @property
    def degradation(self) -> float:
        """样本内→样本外 Sharpe 衰减（>=0 表示衰减，越大越可能过拟合）。"""
        return self.in_sample_sharpe - self.oos_sharpe

    def summary(self) -> str:
        return (
            f"walk-forward: folds={self.n_folds} | "
            f"OOS total_return={self.oos_total_return:.2%} "
            f"OOS sharpe={self.oos_sharpe:.2f} "
            f"IS sharpe={self.in_sample_sharpe:.2f} "
            f"degradation={self.degradation:.2f}"
        )


class WalkForwardValidator:
    """样本外 walk-forward 验证器。

    每折：用 ``signal_fn`` 在训练窗口上生成信号并在训练集回测（样本内），
    再在紧随其后的测试窗口上用同样逻辑生成信号并回测（样本外）。测试窗口
    严格位于训练窗口之后，因此不存在前视泄漏。

    Args:
        engine: 回测引擎实例（含佣金/滑点/复权配置）。
        signal_fn: 接收一段 ``price_data`` 子集，返回与其对齐的
            ``signals`` 序列（1/0/-1）。代表「策略逻辑」，在每折独立调用，
            天然体现「用历史拟合、在未见数据验证」。
    """

    def __init__(
        self,
        engine: BacktestEngine,
        signal_fn: Callable[[pd.DataFrame], pd.Series],
    ) -> None:
        self.engine = engine
        self.signal_fn = signal_fn

    def run(
        self,
        price_data: pd.DataFrame,
        train_size: int,
        test_size: int,
        step: int | None = None,
        benchmark: pd.Series | None = None,
        provenance: list | None = None,
        research_mode: bool = False,
    ) -> WalkForwardReport:
        """执行 walk-forward 验证。

        Args:
            price_data: 完整历史 OHLCV（按时间升序）。
            train_size / test_size / step: 见 :func:`walk_forward_splits`。
            benchmark: 可选基准序列（按时间升序，长度需覆盖测试窗口）。
            provenance: 可选数据谱系；命中硬门禁时回测会被拦截（推荐 #2）。
            research_mode: 显式研究模式，透传给每折 ``engine.run``；缺谱系时
                仅此模式可豁免 fail closed（推荐 #2）。

        Returns:
            WalkForwardReport
        """
        splits = walk_forward_splits(len(price_data), train_size, test_size, step)
        folds: list[WalkForwardFold] = []
        for train_idx, test_idx in splits:
            train_pd = price_data.iloc[list(train_idx)]
            test_pd = price_data.iloc[list(test_idx)]
            sig_train = self.signal_fn(train_pd)
            sig_test = self.signal_fn(test_pd)

            bench_train = benchmark.iloc[list(train_idx)] if benchmark is not None else None
            bench_test = benchmark.iloc[list(test_idx)] if benchmark is not None else None

            in_sample = self.engine.run(
                train_pd,
                signals=sig_train,
                benchmark=bench_train,
                provenance=provenance,
                research_mode=research_mode,
            )
            out_of_sample = self.engine.run(
                test_pd,
                signals=sig_test,
                benchmark=bench_test,
                provenance=provenance,
                research_mode=research_mode,
            )
            folds.append(
                WalkForwardFold(
                    train_idx=train_idx,
                    test_idx=test_idx,
                    in_sample=in_sample,
                    out_of_sample=out_of_sample,
                )
            )
        return WalkForwardReport(folds=folds)
