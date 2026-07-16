"""Tests for dependency license inventory (P4.4)."""

from __future__ import annotations

from scripts.license_check import (
    DIRECT_RUNTIME,
    find_restrictive,
    license_of,
)


def _fake_metadata(classifier=None, license_field=None):
    data = {}
    if classifier is not None:
        data["Classifier"] = [classifier]
    if license_field is not None:
        data["License"] = license_field

    class _Meta:
        def get_all(self, key):
            return data.get(key, [])

        def get(self, key, default=None):
            return data.get(key, default)

    return _Meta()


def test_license_from_classifier():
    meta = _fake_metadata(classifier="License :: OSI Approved :: MIT License")
    assert license_of(meta) == "MIT License"


def test_license_falls_back_to_license_field():
    meta = _fake_metadata(license_field="BSD-3-Clause")
    assert license_of(meta) == "BSD-3-Clause"


def test_license_unknown_when_missing():
    meta = _fake_metadata()
    assert license_of(meta) == "UNKNOWN"


def test_find_restrictive_detects_gpl():
    rows = [
        {"name": "okpkg", "version": "1.0", "license": "MIT", "declared": "yes"},
        {"name": "copyleft", "version": "2.0", "license": "GPL-3.0", "declared": "yes"},
        {"name": "commercial", "version": "1.0", "license": "Commercial", "declared": "no"},
    ]
    flagged = find_restrictive(rows)
    assert {r["name"] for r in flagged} == {"copyleft", "commercial"}


def test_find_restrictive_permissive_clean():
    rows = [
        {"name": "a", "version": "1.0", "license": "MIT", "declared": "yes"},
        {"name": "b", "version": "1.0", "license": "Apache-2.0", "declared": "yes"},
        {"name": "c", "version": "1.0", "license": "BSD-3-Clause", "declared": "yes"},
    ]
    assert find_restrictive(rows) == []


def test_direct_runtime_includes_core_deps():
    assert "pandas" in DIRECT_RUNTIME
    assert "langchain" in DIRECT_RUNTIME
    assert "baostock" not in DIRECT_RUNTIME  # 仅作为 optional extra
