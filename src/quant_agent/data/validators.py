"""Stock code validation for A-share market"""

from __future__ import annotations

import re

# Valid A-share prefixes: 60xxxx (Shanghai), 00xxxx (Shenzhen main),
# 30xxxx (ChiNext), 8xxxxx (Beijing)
_STOCK_CODE_RE = re.compile(r"^(60|00|30|8\d)\d{4}$")


def validate_stock_code(code: str) -> str:
    """Validate and normalise an A-share stock code.

    Args:
        code: Raw stock code string (e.g. ``"300750"``).

    Returns:
        The cleaned 6-digit stock code.

    Raises:
        ValueError: If the code is not a valid 6-digit A-share identifier.
    """
    if code is None:
        raise ValueError("Stock code must not be None")

    cleaned = str(code).strip()

    if not cleaned.isdigit():
        raise ValueError(
            f"Invalid stock code '{code}': must contain only digits"
        )

    if len(cleaned) != 6:
        raise ValueError(
            f"Invalid stock code '{code}': must be exactly 6 digits, got {len(cleaned)}"
        )

    if not _STOCK_CODE_RE.match(cleaned):
        raise ValueError(
            f"Invalid stock code '{code}': prefix must be one of 60 (Shanghai), "
            "00 (Shenzhen main board), 30 (ChiNext), or 8 (Beijing)"
        )

    return cleaned
