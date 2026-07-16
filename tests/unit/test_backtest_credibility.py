"""推荐 #4 回测可信度增强测试：walk-forward 样本外验证、point-in-time 校验、
复权记录。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest

from quant_agent.backtest.engine import BacktestEngine
from quant_agent.backtest.walk_forward import (
    WalkForwardValidator,
    validate_point_in_time,
    walk_forward_splits,
)
from quant_agent.data.gate import DataTrustError


def _price_data(n=30, seed=42):
    rng = np.random.default_rng(seed)
    # 上升趋势 + 噪声，保证有可交易信号
    close = 100 + np.cumsum(rng.normal(0.5, 1.0, n))
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({"date": dates, "close": close})


def _signals(df: pd.DataFrame) -> pd.Series:
    """简单动量：前半段持有，后半段空仓（仅为验证器结构正确）。"""
    n = len(df)
    sig = np.zeros(n, dtype=int)
    sig[2 : n // 2] = 1
    sig[n // 2 : n // 2 + 2] = -1
    return pd.Series(sig)


class TestWalkForwardSplits:
    def test_basic_splits_no_leakage(self):
        splits = walk_forward_splits(10, train_size=4, test_size=2)
        # start=0: (0-4,4-6); start=2: (2-6,6-8); start=4: (4-8,8-10)
        assert splits == [
            (range(0, 4), range(4, 6)),
            (range(2, 6), range(6, 8)),
            (range(4, 8), range(8, 10)),
        ]
        # 无泄漏：每折 test 严格在 train 之后
        for tr, te in splits:
            assert te.start >= tr.stop

    def test_empty_when_too_short(self):
        assert walk_forward_splits(5, train_size=4, test_size=4) == []

    def test_invalid_sizes_raise(self):
        with pytest.raises(ValueError):
            walk_forward_splits(20, train_size=0, test_size=2)
        with pytest.raises(ValueError):
            walk_forward_splits(20, train_size=4, test_size=0)
        with pytest.raises(ValueError):
            walk_forward_splits(20, train_size=4, test_size=2, step=0)

    def test_step_overlap_allowed(self):
        # step < test_size 时测试窗口可重叠，但仍无前视（test 在 train 后）
        splits = walk_forward_splits(12, train_size=4, test_size=4, step=2)
        assert len(splits) >= 2
        for tr, te in splits:
            assert te.start >= tr.stop


class TestPointInTime:
    def test_no_issues_on_clean(self):
        df = _price_data(20)
        assert validate_point_in_time(df, _signals(df)) == []

    def test_detects_signal_length_overflow(self):
        df = _price_data(10)
        long_sig = pd.Series(np.ones(15, dtype=int))
        issues = validate_point_in_time(df, long_sig)
        assert any("前视泄漏" in i for i in issues)

    def test_detects_nan_signals(self):
        df = _price_data(10)
        sig = pd.Series(np.zeros(10, dtype=int))
        sig.iloc[3] = np.nan
        issues = validate_point_in_time(df, sig)
        assert any("NaN" in i for i in issues)

    def test_detects_unordered_dates(self):
        df = _price_data(10)
        df = df.iloc[::-1].reset_index(drop=True)  # 逆序
        issues = validate_point_in_time(df)
        assert any("升序" in i for i in issues)


class TestEngineCredibility:
    def test_adjust_recorded(self):
        eng = BacktestEngine(initial_capital=100000, adjust="hfq")
        res = eng.run(_price_data(20), signals=_signals(_price_data(20)), research_mode=True)
        assert res.adjust == "hfq"

    def test_adjust_override_in_run(self):
        eng = BacktestEngine(initial_capital=100000, adjust="qfq")
        res = eng.run(
            _price_data(20), signals=_signals(_price_data(20)), adjust="none", research_mode=True
        )
        assert res.adjust == "none"

    def test_pit_issues_recorded(self):
        eng = BacktestEngine(initial_capital=100000)
        df = _price_data(10)
        long_sig = pd.Series(np.ones(15, dtype=int))
        res = eng.run(df, signals=long_sig, research_mode=True)
        assert res.point_in_time_issues  # 记录了前视问题
        assert any("前视泄漏" in i for i in res.point_in_time_issues)


class TestWalkForwardValidator:
    def test_runs_and_oos_computed(self):
        eng = BacktestEngine(initial_capital=100000)
        df = _price_data(40)
        val = WalkForwardValidator(engine=eng, signal_fn=_signals)
        report = val.run(df, train_size=12, test_size=6, step=6, research_mode=True)
        assert report.n_folds >= 1
        # 每折测试区间都在训练区间之后（无泄漏）
        for fold in report.folds:
            assert fold.test_idx.start >= fold.train_idx.stop
        # OOS 指标为有限数（未崩溃）
        assert np.isfinite(report.oos_total_return)
        assert np.isfinite(report.oos_sharpe)
        # degradation 为数值（样本内减样本外）
        assert np.isfinite(report.degradation)

    def test_blocked_by_trust_gate(self):
        from quant_agent.data.sources.base import DataProvenance

        eng = BacktestEngine(initial_capital=100000)
        df = _price_data(40)
        val = WalkForwardValidator(engine=eng, signal_fn=_signals)
        prov = [
            DataProvenance(
                source="sample",
                identifier="600519:price",
                fetched_at="2024-01-01T00:00:00",
                data_type="price",
                confidence="low",
            )
        ]
        with pytest.raises(DataTrustError):
            val.run(df, train_size=12, test_size=6, provenance=prov)

    def test_walk_forward_blocks_without_provenance(self):
        """walk-forward 缺谱系且非研究模式时，fail closed 拦截。"""
        eng = BacktestEngine(initial_capital=100000)
        df = _price_data(40)
        val = WalkForwardValidator(engine=eng, signal_fn=_signals)
        with pytest.raises(DataTrustError):
            val.run(df, train_size=12, test_size=6)

    def test_summary_string(self):
        eng = BacktestEngine(initial_capital=100000)
        df = _price_data(40)
        val = WalkForwardValidator(engine=eng, signal_fn=_signals)
        report = val.run(df, train_size=12, test_size=6, step=6, research_mode=True)
        assert "walk-forward" in report.summary()

    def _leaky_wf_df(self) -> pd.DataFrame:
        """构造单折 walk-forward 数据：训练 12 根单调，测试 6 根末位人为抬升。

        测试窗口为 ``[10, 11, 12, 13, 14, 100]``，最后一 bar 远超前值。
        """
        train_close = np.linspace(50, 60, 12)
        test_close = np.array([10, 11, 12, 13, 14, 100], dtype=float)
        close = np.concatenate([train_close, test_close])
        dates = pd.date_range("2024-01-01", periods=len(close), freq="D")
        return pd.DataFrame({"date": dates, "close": close})

    def _peek_last_fn(self) -> Callable[[pd.DataFrame], pd.Series]:
        """泄漏型策略：把整个窗口最后一根 close 与首根比较，结果作用于所有 bar。

        若窗口末位 > 50 则全仓买入(1)，否则空仓(0)。这是典型的前视泄漏——
        决策偷看了窗口最后一 bar。
        """

        def fn(df: pd.DataFrame) -> pd.Series:
            last = df["close"].iloc[-1]
            val = 1 if last > 50 else 0
            return pd.Series(np.full(len(df), val, dtype=int))

        return fn

    def test_strict_oos_is_no_lookahead(self):
        """严格模式逐日生成，未来 bar 不影响早期信号（杜绝前视泄漏）。"""
        eng = BacktestEngine(initial_capital=100000)
        df = self._leaky_wf_df()
        val = WalkForwardValidator(engine=eng, signal_fn=self._peek_last_fn())

        rep_strict = val.run(df, train_size=12, test_size=6, step=6, research_mode=True)
        rep_leaky = val.run(
            df, train_size=12, test_size=6, step=6, research_mode=True, strict=False
        )

        # 严格：每个 bar 只看自身及之前。测试窗口 [10..14,100] 中前 5 根 <50 → 0，
        # 末根 100 → 1。未来 bar 不会污染早期决策。
        assert list(rep_strict.folds[0].oos_signals) == [0, 0, 0, 0, 0, 1]
        # 非严格：整个测试窗口末位 100 > 50 → 全部 bar 误判为买入(1)。
        assert list(rep_leaky.folds[0].oos_signals) == [1, 1, 1, 1, 1, 1]

    def test_strict_blocks_future_bar_leakage(self):
        """篡改未来 bar 时：严格模式早期信号不变，非严格模式被污染。"""
        eng = BacktestEngine(initial_capital=100000)
        df = self._leaky_wf_df()
        df_b = df.copy()
        # 仅把测试窗口最后一 bar 由 100 改为 10（消除末位尖峰）。
        df_b.iloc[len(df_b) - 1, df_b.columns.get_loc("close")] = 10
        val = WalkForwardValidator(engine=eng, signal_fn=self._peek_last_fn())

        rep_strict_a = val.run(df, train_size=12, test_size=6, step=6, research_mode=True)
        rep_strict_b = val.run(df_b, train_size=12, test_size=6, step=6, research_mode=True)
        rep_leaky_a = val.run(
            df, train_size=12, test_size=6, step=6, research_mode=True, strict=False
        )
        rep_leaky_b = val.run(
            df_b, train_size=12, test_size=6, step=6, research_mode=True, strict=False
        )

        # 严格：首根测试 bar 只依赖 train+test[0]，末位改动与它无关 → 不变。
        assert (
            rep_strict_a.folds[0].oos_signals.iloc[0] == rep_strict_b.folds[0].oos_signals.iloc[0]
        )
        # 非严格：末位改动改变了整个测试窗口的决策 → 首根 bar 被污染。
        assert rep_leaky_a.folds[0].oos_signals.iloc[0] != rep_leaky_b.folds[0].oos_signals.iloc[0]

    def test_strict_raises_on_misaligned_signal_fn(self):
        """signal_fn 返回长度与窗口不一致时，严格模式应报错（对齐被破坏）。"""

        def bad_fn(df: pd.DataFrame) -> pd.Series:
            return pd.Series([1, 0])  # 长度恒为 2，与窗口不符

        eng = BacktestEngine(initial_capital=100000)
        df = _price_data(40)
        val = WalkForwardValidator(engine=eng, signal_fn=bad_fn)
        with pytest.raises(ValueError):
            val.run(df, train_size=12, test_size=6, step=6, research_mode=True)
