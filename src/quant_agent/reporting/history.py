"""报告历史 — 将分析结果持久化，支持列出 / 对比 / 导出

存储位置：data/reports/，每份报告一个 JSON，并在 index.json 维护轻量索引。
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from ..orchestrator import AnalysisReport

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("data/reports")


def save_report(report: AnalysisReport, base_dir: str | Path = REPORTS_DIR) -> Path:
    """保存分析报告为 JSON，并更新索引。

    Returns:
        报告文件路径
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{report.stock_code}_{ts}.json"
    path = base_dir / filename

    payload = report.to_dict()
    payload["_file"] = filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    _update_index(base_dir, report, filename, ts)
    logger.info("报告已保存: %s", path)
    return path


def _update_index(base_dir: Path, report: AnalysisReport, filename: str, ts: str) -> None:
    index_path = base_dir / "index.json"
    index: list[dict] = []
    if index_path.exists():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            index = []

    signal = report.risk_result.signal if report.risk_result else "HOLD"
    conf = report.risk_result.confidence if report.risk_result else 0.0
    index.append(
        {
            "file": filename,
            "stock_code": report.stock_code,
            "timestamp": report.timestamp,
            "signal": signal,
            "confidence": conf,
            "saved_at": ts,
        }
    )
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)


def list_reports(base_dir: str | Path = REPORTS_DIR) -> list[dict]:
    """列出历史报告索引（按保存时间倒序）"""
    base_dir = Path(base_dir)
    index_path = base_dir / "index.json"
    if not index_path.exists():
        return []
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return sorted(index, key=lambda x: x.get("saved_at", ""), reverse=True)


def load_report(file_name: str, base_dir: str | Path = REPORTS_DIR) -> dict | None:
    """按文件名加载报告 JSON"""
    base_dir = Path(base_dir)
    path = base_dir / file_name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("加载报告失败 %s: %s", file_name, e)
        return None


def latest_for_stock(stock_code: str, base_dir: str | Path = REPORTS_DIR) -> dict | None:
    """获取某股票最近一次报告"""
    for entry in list_reports(base_dir):
        if entry.get("stock_code") == stock_code:
            return load_report(entry["file"], base_dir)
    return None
