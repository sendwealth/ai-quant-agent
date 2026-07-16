"""回测运行清单 (BacktestRunManifest) — 让回测结论可复现、可审计。

P1.3 目标：每次回测都产出一份结构化清单，记录
「用了什么策略/参数/随机种子/代码版本/依赖指纹/数据指纹/基准/执行假设」，
使任意回测结论都能被他人在相同环境下重跑、验证。

清单序列化版本 ``MANIFEST_SCHEMA_VERSION`` 应随字段变更递增，便于迁移与校验。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# 清单 schema 版本 — 字段变更时递增
MANIFEST_SCHEMA_VERSION = "1.0"

# 仓库根目录（用于获取 git SHA / uv.lock）
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git_sha() -> str:
    """获取当前 git HEAD commit SHA；非 git 仓库时返回 'unknown'。"""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _uv_lock_fingerprint() -> str:
    """计算 uv.lock 的 sha256 指纹；文件不存在时返回 'unknown'。"""
    lock = _REPO_ROOT / "uv.lock"
    if lock.exists():
        try:
            return hashlib.sha256(lock.read_bytes()).hexdigest()
        except Exception:
            pass
    return "unknown"


def _package_version() -> str:
    """读取已安装包版本；开发模式/未安装时回退 'unknown'。"""
    try:
        from importlib.metadata import version

        return version("ai-quant-agent")
    except Exception:
        return "unknown"


def collect_environment() -> dict[str, str]:
    """收集运行环境指纹（git SHA / uv.lock / Python / 包版本）。"""
    import sys

    return {
        "git_sha": _git_sha(),
        "uv_lock_fingerprint": _uv_lock_fingerprint(),
        "python_version": sys.version.split()[0],
        "package_version": _package_version(),
    }


@dataclass
class BacktestRunManifest:
    """单次回测运行的完整清单 — 复现与审计的依据。

    所有字段均有默认值（向后兼容 / 部分字段未知时仍可序列化）。
    """

    manifest_schema_version: str = MANIFEST_SCHEMA_VERSION
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    strategy_name: str = "unknown"
    strategy_version: str = "unknown"
    params: dict = field(default_factory=dict)
    seed: int | None = None
    data_hash: str | None = None  # 输入价格/信号数据指纹 (sha256 前 16 位)
    benchmark: str | None = None  # 基准描述（如 "沪深300" 或 "buy&hold"）
    # 执行假设 — 回测引擎采纳（或不采纳）的市场规则
    execution_assumptions: dict = field(default_factory=dict)
    # 运行环境指纹
    git_sha: str = "unknown"
    uv_lock_fingerprint: str = "unknown"
    python_version: str = "unknown"
    package_version: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_dict(cls, d: dict) -> BacktestRunManifest:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        """将清单持久化为 JSON 文件（原子写入）。"""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json(), encoding="utf-8")
        tmp.replace(path)


def build_manifest(
    strategy_name: str,
    params: dict | None = None,
    seed: int | None = None,
    data_hash: str | None = None,
    benchmark: str | None = None,
    execution_assumptions: dict | None = None,
    strategy_version: str = "unknown",
    env: dict | None = None,
) -> BacktestRunManifest:
    """构造一份回测清单，自动填充运行环境指纹。

    Args:
        strategy_name: 策略名（如 "dual_ema" / "RSI+MACD" / "consensus"）
        params: 策略 + 引擎参数（已解析为可序列化 dict）
        seed: 随机种子（若有）
        data_hash: 输入数据指纹（见 :func:`data_fingerprint`）
        benchmark: 基准描述
        execution_assumptions: 执行假设字典
        strategy_version: 策略版本标识
        env: 覆盖环境指纹（测试用）
    """
    env = env or collect_environment()
    return BacktestRunManifest(
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        params=params or {},
        seed=seed,
        data_hash=data_hash,
        benchmark=benchmark,
        execution_assumptions=execution_assumptions or {},
        git_sha=env.get("git_sha", "unknown"),
        uv_lock_fingerprint=env.get("uv_lock_fingerprint", "unknown"),
        python_version=env.get("python_version", "unknown"),
        package_version=env.get("package_version", "unknown"),
    )


def data_fingerprint(*objs: Any) -> str:
    """计算多个对象的联合数据指纹 (sha256 前 16 位)。

    用于把价格序列 + 信号序列压缩成一个可复现标识，写入清单 ``data_hash``。
    """
    import pandas as pd

    parts: list[str] = []
    for o in objs:
        try:
            if isinstance(o, pd.DataFrame):
                parts.append(o.to_json(orient="records", date_format="iso"))
            elif isinstance(o, pd.Series):
                parts.append(o.to_json(date_format="iso"))
            elif isinstance(o, dict):
                parts.append(json.dumps(o, default=str, sort_keys=True, ensure_ascii=False))
            else:
                parts.append(str(o))
        except Exception:
            parts.append(str(o))
    joined = "\n".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]
