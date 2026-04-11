"""Append-only audit trail for trade decisions.

Every trade decision is written to a JSON-lines file (one JSON object per
line) that is *append-only* -- the file is never overwritten.  One file is
created per calendar month (``audit_YYYYMM.jsonl``) inside the configured
log directory.

Thread-safety is guaranteed via a ``threading.Lock`` so that concurrent
agent threads can safely log decisions without corrupting the file.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


class AuditLogger:
    """Append-only audit logger for trade decisions.

    Parameters
    ----------
    log_dir:
        Directory where monthly audit files are stored.  Created
        automatically on first use if it does not exist.
    """

    def __init__(self, log_dir: str) -> None:
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_trade_decision(
        self,
        stock_code: str,
        signal: str,
        agent_results: list[dict[str, Any]],
        final_decision: dict[str, Any],
    ) -> None:
        """Append a single trade-decision record to the audit log.

        Parameters
        ----------
        stock_code:
            The stock being evaluated (e.g. ``"300750"``).
        signal:
            The consensus signal -- ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
        agent_results:
            A list of dicts, one per analyst agent, each containing at
            least ``agent_name``, ``signal``, ``confidence``, and
            ``reasoning``.
        final_decision:
            The execution-level decision dict with keys such as
            ``action``, ``quantity``, ``price``, ``stop_loss``,
            ``take_profit``.
        """
        entry: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "stock_code": stock_code,
            "signal": signal,
            "agent_votes": agent_results,
            "final_decision": final_decision,
        }

        line = json.dumps(entry, ensure_ascii=False, default=str) + "\n"

        filepath = self._current_filepath()
        with self._lock:
            with open(filepath, "a", encoding="utf-8") as fh:
                fh.write(line)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _current_filepath(self) -> Path:
        """Return the audit file path for the current month."""
        filename = f"audit_{datetime.now().strftime('%Y%m')}.jsonl"
        return self._log_dir / filename
