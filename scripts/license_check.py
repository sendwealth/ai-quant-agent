"""依赖许可证清单（P4.4）。

列出当前环境已安装、且与本项目运行时依赖对应的包及其许可证，
用于开源发布前的许可合规自检（确认所有再分发许可可接受）。

用法::

    uv run python scripts/license_check.py            # 打印表格
    uv run python scripts/license_check.py --json     # JSON 输出

仅检查 ``pyproject.toml`` 中声明的运行时 + 可选依赖（以及它们
传递引入的包），不扫描无关的开发工具。
"""

from __future__ import annotations

import json
import sys
from typing import Any

# 与 pyproject.toml [project].dependencies 对齐（用于把“直接依赖”标记出来）
DIRECT_RUNTIME = {
    "akshare",
    "tushare",
    "pandas",
    "numpy",
    "pydantic-settings",
    "python-dotenv",
    "pyarrow",
    "pyyaml",
    "tenacity",
    "langchain",
    "langchain-openai",
    "langchain-community",
    "efinance",
    "typer",
    "matplotlib",
}

# 配置中声明的可选依赖 group（baostock extra）
OPTIONAL_GROUPS: dict[str, set[str]] = {
    "baostock": {"baostock"},
}

# 被认定为「非自由 / 高风险」需人工复核的许可证关键词
RESTRICTIVE_KEYWORDS = (
    "GPL",
    "AGPL",
    "LGPL",
    "SSPL",
    "Commons Clause",
    "Proprietary",
    "Commercial",
)


def _read_project_dependencies() -> set[str]:
    """从 pyproject.toml 解析声明的依赖名（小写）。"""
    names: set[str] = set()
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        try:
            import tomli as tomllib  # type: ignore
        except ModuleNotFoundError:
            return names
    pyproject = _repo_root() / "pyproject.toml"
    if not pyproject.exists():
        return names
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        # 形如 "akshare>=1.12.0" 或 "baostock[extra]"
        name = dep.split(">=")[0].split("==")[0].split("<")[0].split("[")[0]
        names.add(name.strip().lower())
    for group in OPTIONAL_GROUPS.values():
        names |= group
    return names


def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


def license_of(metadata: Any) -> str:
    """从 distribution 元数据提取许可证字符串。"""
    classifiers = metadata.get_all("Classifier") or []
    for c in classifiers:
        if c.startswith("License") and "::" in c:
            # 形如 "License :: OSI Approved :: MIT License" -> 取末段
            return c.split("::")[-1].strip()
    direct = metadata.get("License")
    if direct:
        return str(direct)
    return "UNKNOWN"


def build_inventory() -> list[dict[str, str]]:
    """构建依赖许可证清单。"""
    from importlib.metadata import distributions

    declared = _read_project_dependencies()
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    for dist in distributions():
        name = dist.metadata["Name"]
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        lic = license_of(dist.metadata)
        rows.append(
            {
                "name": name,
                "version": dist.version,
                "license": lic,
                "declared": "yes" if key in declared else "no",
            }
        )
    rows.sort(key=lambda r: r["name"].lower())
    return rows


def find_restrictive(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """返回需要人工复核的高风险许可证行。"""
    out: list[dict[str, str]] = []
    for r in rows:
        upper = r["license"].upper()
        if any(k.upper() in upper for k in RESTRICTIVE_KEYWORDS):
            out.append(r)
    return out


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    as_json = "--json" in argv

    rows = build_inventory()
    restrictive = find_restrictive(rows)

    if as_json:
        print(json.dumps({"packages": rows, "review": restrictive}, indent=2))
    else:
        print(f"{'Package':<32}{'Version':<14}{'License':<28}{'Declared'}")
        print("-" * 90)
        for r in rows:
            flag = "  <-- REVIEW" if r in restrictive else ""
            print(f"{r['name']:<32}{r['version']:<14}{r['license']:<28}{r['declared']}{flag}")
        if restrictive:
            print("\n⚠️  发现需人工复核的许可证（可能为 copyleft / 商业许可）：")
            for r in restrictive:
                print(f"  - {r['name']}: {r['license']}")

    # 退出码：存在高风险许可证且非宽松时返回 1，便于 CI 阻断
    return 1 if restrictive else 0


if __name__ == "__main__":
    raise SystemExit(main())
