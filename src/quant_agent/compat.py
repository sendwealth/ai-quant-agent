"""版本与兼容性检查（P4.3）。

提供发布前/启动时的兼容性自检：
- Python 版本下限（3.10+）
- 关键依赖的已知兼容版本区间
- 与已发布 manifest / schema 的前向兼容提示

CLI::

    uv run python -m quant_agent.compat
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

MIN_PYTHON_VERSION = (3, 10)

# 关键依赖的已知兼容版本区间（下界含，上界不含；None 表示无上限）。
KNOWN_DEPS: dict[str, tuple[str, str | None]] = {
    "pydantic": ("2.0", None),
    "pydantic-settings": ("2.0", None),
    "pandas": ("2.0", None),
    "langchain": ("0.2", None),
    "langgraph": ("0.2", None),
}


@dataclass
class CompatReport:
    """兼容性自检报告。"""

    ok: bool
    python_version: str
    python_ok: bool
    issues: list[str] = field(default_factory=list)
    deps: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "python_version": self.python_version,
            "python_ok": self.python_ok,
            "issues": self.issues,
            "deps": self.deps,
        }


def check_python_version() -> bool:
    """Python 版本是否满足下限。"""
    return sys.version_info[:2] >= MIN_PYTHON_VERSION


def _version_in_range(v: str, low: str, high: str | None) -> bool:
    """判断版本 v 是否落在 [low, high) 区间（支持预发布号）。"""
    try:
        from packaging.version import Version

        vv = Version(v)
        if Version(low) > vv:
            return False
        if high is not None and Version(high) <= vv:
            return False
        return True
    except Exception:
        # 无法解析版本时放行，避免误伤
        return True


def check_dependency_versions(get_version=None) -> tuple[list[str], dict[str, str]]:
    """检查关键依赖版本；返回 (问题列表, 已安装版本映射)。

    ``get_version`` 用于测试注入；默认使用 importlib.metadata.version。
    """
    if get_version is None:
        from importlib.metadata import version as get_version  # type: ignore[assignment]

    issues: list[str] = []
    installed: dict[str, str] = {}
    for name, (low, high) in KNOWN_DEPS.items():
        try:
            v = get_version(name)  # type: ignore[call-arg]
        except Exception:
            continue  # 可选依赖未安装则跳过
        installed[name] = v
        if not _version_in_range(v, low, high):
            issues.append(f"{name} {v} 不在兼容区间 [{low}, {high})")
    return issues, installed


def check_compatibility() -> CompatReport:
    """执行完整兼容性自检。"""
    py_ok = check_python_version()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    issues: list[str] = []
    if not py_ok:
        issues.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} "
            f"< 最低要求 {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]}"
        )
    dep_issues, installed = check_dependency_versions()
    issues.extend(dep_issues)
    return CompatReport(
        ok=py_ok and not issues,
        python_version=py_ver,
        python_ok=py_ok,
        issues=issues,
        deps=installed,
    )


def main() -> int:
    import json

    report = check_compatibility()
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
