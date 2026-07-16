"""P1.3 BacktestRunManifest + P1.4 假设守卫 单元测试"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_agent.backtest.engine import BacktestEngine, SlippageModel
from quant_agent.backtest.guards import (
    BacktestAssumptionReport,
    BacktestAssumptionViolation,
    check_assumptions,
)
from quant_agent.backtest.manifest import (
    MANIFEST_SCHEMA_VERSION,
    BacktestRunManifest,
    build_manifest,
    data_fingerprint,
)

# ── P1.3 manifest ──────────────────────────────────────────────────────────


def test_build_manifest_includes_env_fingerprint():
    m = build_manifest(
        strategy_name="dual_ema",
        params={"a": 1},
        seed=42,
        data_hash="deadbeef",
        benchmark="buy&hold",
        execution_assumptions={"t_plus_one_enforced": False},
        env={
            "git_sha": "abc",
            "uv_lock_fingerprint": "def",
            "python_version": "3.12",
            "package_version": "3.1.0",
        },
    )
    assert m.strategy_name == "dual_ema"
    assert m.seed == 42
    assert m.data_hash == "deadbeef"
    assert m.git_sha == "abc"
    assert m.uv_lock_fingerprint == "def"
    assert m.manifest_schema_version == MANIFEST_SCHEMA_VERSION


def test_manifest_roundtrip_json(tmp_path):
    m = build_manifest(strategy_name="x", params={"k": "v"}, env={})
    p = tmp_path / "manifest.json"
    m.save(p)
    loaded = BacktestRunManifest.from_dict(json.loads(p.read_text(encoding="utf-8")))
    assert loaded.strategy_name == m.strategy_name
    assert loaded.params == m.params


def test_data_fingerprint_stable_and_distinct():
    df1 = pd.DataFrame({"close": [1, 2, 3]})
    df2 = pd.DataFrame({"close": [1, 2, 4]})
    assert data_fingerprint(df1) == data_fingerprint(df1)
    assert data_fingerprint(df1) != data_fingerprint(df2)


def test_manifest_from_dict_ignores_unknown_keys():
    m = BacktestRunManifest.from_dict({"strategy_name": "z", "bogus": 1})
    assert m.strategy_name == "z"


# ── P1.4 assumption guards ───────────────────────────────────────────────────


def _df(dates, closes, highs=None, lows=None, volumes=None):
    data = {"date": dates, "close": closes}
    if highs is not None:
        data["high"] = highs
    if lows is not None:
        data["low"] = lows
    if volumes is not None:
        data["volume"] = volumes
    return pd.DataFrame(data)


def test_check_assumptions_detects_weekend():
    # 2025-01-04 是周六
    df = _df(["20250103", "20250104"], [10.0, 10.5], volumes=[10000, 10000])
    rep = check_assumptions(df, adjust="qfq")
    assert "2025-01-04" in rep.non_trading_days
    assert rep.trading_days == 2


def test_check_assumptions_detects_suspension():
    df = _df(["20250102", "20250103"], [10.0, 10.5], volumes=[10000, 0])
    rep = check_assumptions(df, adjust="qfq")
    assert "2025-01-03" in rep.suspended_dates


def test_check_assumptions_detects_limit_up():
    # 前日收盘 10，当日 high 11 (>= +9.5%) → 涨停
    df = _df(
        ["20250102", "20250103"],
        [10.0, 10.5],
        highs=[10.0, 11.0],
        lows=[10.0, 10.4],
        volumes=[10000, 10000],
    )
    rep = check_assumptions(df, adjust="qfq")
    assert "2025-01-03" in rep.limit_up_dates


def test_check_assumptions_low_liquidity():
    df = _df(["20250102", "20250103"], [10.0, 10.5], volumes=[10000, 100])
    rep = check_assumptions(df, adjust="qfq")
    assert "2025-01-03" in rep.low_liquidity_dates


def test_check_assumptions_undeclared_adjust_is_violation():
    df = _df(["20250102"], [10.0], volumes=[10000])
    rep = check_assumptions(df, adjust=None)
    assert rep.adjust_status is None
    assert any("复权" in v for v in rep.hard_violations)
    with pytest.raises(BacktestAssumptionViolation):
        check_assumptions(df, adjust=None, strict=True)


def test_check_assumptions_warnings_nonempty_on_issues():
    df = _df(["20250104"], [10.0], volumes=[0])  # 周末 + 停牌
    rep = check_assumptions(df, adjust="qfq")
    assert rep.warnings()  # 至少包含周末/停牌/前视警告


def test_report_serializes():
    rep = BacktestAssumptionReport(trading_days=1)
    d = rep.to_dict()
    assert d["trading_days"] == 1
    assert "warnings" in d


# ── P1.4 engine T+1 enforcement ──────────────────────────────────────────────


def _prices():
    # 买入后立刻卖出信号：bar0 买, bar1 卖
    return pd.DataFrame(
        {
            "date": ["20250102", "20250103", "20250106", "20250107"],
            "close": [100.0, 110.0, 110.0, 110.0],
        }
    )


def test_t_plus_one_flag_accepted_daily_noop():
    """日频回测下，T+1 是 no-op（当日买入最早次日卖出，本就允许）。

    该测试验证：引擎接受 ``enforce_t_plus_one`` 开关且不崩溃；日频分辨率下
    T+1 不改变结果（真实的 T+1 约束只作用于日内信号，已在假设中声明）。
    """
    df = _prices()
    signals = pd.Series([1, -1, 0, 0])
    eng_off = BacktestEngine(slippage=SlippageModel(basis_points=0.0))
    r_off = eng_off.run(df, signals, research_mode=True)
    eng_on = BacktestEngine(slippage=SlippageModel(basis_points=0.0), enforce_t_plus_one=True)
    r_on = eng_on.run(df, signals, research_mode=True)
    assert r_off.total_trades >= 1
    assert r_on.total_trades >= 1
    # 日频下 T+1 不应改变结果
    assert r_on.total_return == r_off.total_return


def test_manifest_records_t_plus_one_assumption():
    m = build_manifest(
        strategy_name="x",
        execution_assumptions={"t_plus_one_enforced": True, "look_ahead_same_bar": False},
        env={},
    )
    assert m.execution_assumptions["t_plus_one_enforced"] is True
    assert m.execution_assumptions["look_ahead_same_bar"] is False
