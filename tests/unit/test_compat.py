"""Tests for version / compatibility self-check (P4.3)."""

from __future__ import annotations

from quant_agent.compat import (
    MIN_PYTHON_VERSION,
    CompatReport,
    _version_in_range,
    check_compatibility,
    check_dependency_versions,
    check_python_version,
)


def test_min_python_version_floor():
    assert MIN_PYTHON_VERSION == (3, 10)


def test_check_python_version_true_on_modern():
    assert check_python_version() is True


def test_version_in_range_open_upper():
    assert _version_in_range("2.5.0", "2.0", None) is True
    assert _version_in_range("1.9.0", "2.0", None) is False


def test_version_in_range_closed_upper():
    assert _version_in_range("0.2.0", "0.2", "0.3") is True
    assert _version_in_range("0.3.0", "0.2", "0.3") is False  # 上界不含


def test_version_in_range_unparseable_passes():
    assert _version_in_range("not-a-version", "2.0", None) is True


def test_check_dependency_versions_injects_get_version():
    fake = {
        "pydantic": "2.7.0",
        "pandas": "1.5.0",  # 低于 [2.0, None) -> 应报错
        "missing-opt": "9.9.9",
    }

    def _get(name: str) -> str:
        return fake[name]

    issues, installed = check_dependency_versions(get_version=_get)
    assert installed["pydantic"] == "2.7.0"
    assert any("pandas" in i for i in issues)


def test_check_compatibility_report_shape():
    rep = check_compatibility()
    assert isinstance(rep, CompatReport)
    assert rep.python_ok is True
    assert rep.to_dict()["python_version"]
    assert "ok" in rep.to_dict()
