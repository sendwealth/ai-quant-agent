"""Repo-root pytest config.

Prepends the repository root to ``sys.path`` so that first-party helper
modules under ``scripts/`` (e.g. ``scripts.license_check``) are importable
from tests without being installed as a package.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
