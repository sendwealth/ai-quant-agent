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

    注意：本函数仅做结构层面的前视防护；逐日无前视由
    :func:`_strict_oos_signals` 在样本外生成信号时结构性保证。
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


def _strict_oos_signals(
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    train_pd: pd.DataFrame,
    test_pd: pd.DataFrame,
) -> pd.Series:
    """逐日无前视地生成样本外（OOS）信号。

    对测试窗口的每个 bar ``i``，仅用 ``train + test[:i+1]``（截至当日及之前的
    数据）调用 ``signal_fn``，取返回序列的最后一个值作为 bar ``i`` 的信号。

    这从结构上杜绝「策略函数内部读取测试窗口未来行」导致的前视泄漏：任何 bar
    的信号都只能依赖其自身及之前的数据，无法触及 ``test[i+1:]``。

    Args:
        signal_fn: 策略函数，接收一段 OHLCV 子集，返回与其对齐的 ``signals``
            （1/0/-1）；约定返回序列的**最后一个元素**对应当前（最新）bar 的决策。
        train_pd: 训练窗口（全部为历史数据）。
        test_pd: 测试窗口（按时间升序）。

    Returns:
        与 ``test_pd`` 对齐的信号序列。

    Raises:
        ValueError: signal_fn 返回长度与输入窗口不一致（对齐被破坏）。
    """
    if len(test_pd) == 0:
        return pd.Series(dtype="int64", index=test_pd.index)
    values: list[int] = []
    for i in range(len(test_pd)):
        window = pd.concat([train_pd, test_pd.iloc[: i + 1]], ignore_index=False)
        sig = signal_fn(window)
        if len(sig) != len(window):
            raise ValueError(
                f"signal_fn 返回长度({len(sig)})与输入窗口({len(window)})不一致，"
                "无法保证逐日无前视对齐"
            )
        last = sig.iloc[-1]
        values.append(int(last) if pd.notna(last) else 0)
    s = pd.Series(values, index=test_pd.index, dtype="int64")
    # 信号缺失（NaN / 前视warmup未就绪）视为持有(0)，并向后沿用上一有效信号。
    s = s.ffill().fillna(0).astype(int)
    return s


@dataclass
class WalkForwardFold:
    """单折 walk-forward 结果。"""

    train_idx: range
    test_idx: range
    in_sample: BacktestResult
    out_of_sample: BacktestResult
    oos_signals: pd.Series | None = None


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
    严格位于训练窗口之后，因此不存在跨折前视泄漏。

    关键加固（推荐 #3）：样本外信号默认以**逐日无前视**方式生成——每个测试
    bar 只把 ``train + test[:i+1]`` 喂给策略，结构性杜绝策略函数内部引用
    未来行导致的前视泄漏（``run(strict=True)``，默认开启）。

    Args:
        engine: 回测引擎实例（含佣金/滑点/复权配置）。
        signal_fn: 接收一段 ``price_data`` 子集，返回与其对齐的
            ``signals`` 序列（1/0/-1）。代表「策略逻辑」，在每折独立调用，
            天然体现「用历史拟合、在未见数据验证」。约定返回序列最后一个元素
            对应当前（最新）bar 的决策。
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
        strict: bool = True,
    ) -> WalkForwardReport:
        """执行 walk-forward 验证。

        Args:
            price_data: 完整历史 OHLCV（按时间升序）。
            train_size / test_size / step: 见 :func:`walk_forward_splits`。
            benchmark: 可选基准序列（按时间升序，长度需覆盖测试窗口）。
            provenance: 可选数据谱系；命中硬门禁时回测会被拦截（推荐 #2）。
            research_mode: 显式研究模式，透传给每折 ``engine.run``；缺谱系时
                仅此模式可豁免 fail closed（推荐 #2）。
            strict: 是否对样本外信号启用**逐日无前视**生成（默认 ``True``）。
                开启后，每个测试 bar 只基于 ``train + test[:i+1]`` 调用策略，
                结构性阻止策略内部引用未来行；关闭则沿用旧行为（整个测试窗口
                一次性传入，存在前视泄漏风险，**不建议**）。

        Returns:
            WalkForwardReport
        """
        splits = walk_forward_splits(len(price_data), train_size, test_size, step)
        folds: list[WalkForwardFold] = []
        for train_idx, test_idx in splits:
            train_pd = price_data.iloc[list(train_idx)]
            test_pd = price_data.iloc[list(test_idx)]
            # 样本内：整个训练窗口一次性生成（训练窗口全为历史，无 OOS 泄漏）。
            sig_train = self.signal_fn(train_pd)
            # 样本外：默认逐日无前视，结构性防止策略读取未来行（推荐 #3）。
            if strict:
                sig_test = _strict_oos_signals(self.signal_fn, train_pd, test_pd)
            else:
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
                    oos_signals=sig_test,
                )
            )
        return WalkForwardReport(folds=folds)
