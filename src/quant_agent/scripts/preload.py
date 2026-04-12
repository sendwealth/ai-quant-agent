"""Pre-download stock data to local parquet cache.

Usage:
    python -m quant_agent.scripts.preload --stocks 300750,002475
    python -m quant_agent.scripts.preload --from-file stocks.txt --days 365
    python -m quant_agent.scripts.preload --financial-only --stocks 300750
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def preload_price_data(
    service,
    stock_codes: list[str],
    days: int = 250,
) -> tuple[int, list[str]]:
    """Download price data for all stocks with progress reporting.

    Returns (success_count, failed_codes).
    """
    total = len(stock_codes)
    success = 0
    failed: list[str] = []

    for i, code in enumerate(stock_codes, 1):
        try:
            df = service.get_price_data(code, days=days, use_cache=False)
            if df is not None:
                success += 1
                print(f"  [{i}/{total}] {code}: OK ({len(df)} bars)")
            else:
                failed.append(code)
                print(f"  [{i}/{total}] {code}: NO DATA")
        except Exception as e:
            failed.append(code)
            print(f"  [{i}/{total}] {code}: ERROR - {e}")

    return success, failed


def preload_financial_data(
    service,
    stock_codes: list[str],
) -> tuple[int, list[str]]:
    """Download financial snapshots for all stocks.

    Returns (success_count, failed_codes).
    """
    total = len(stock_codes)
    success = 0
    failed: list[str] = []

    for i, code in enumerate(stock_codes, 1):
        try:
            snapshot = service.get_financial_snapshot(code)
            if snapshot is not None:
                success += 1
                fields = len([v for v in snapshot.to_dict().values() if v is not None])
                print(f"  [{i}/{total}] {code}: OK ({fields} fields)")
            else:
                failed.append(code)
                print(f"  [{i}/{total}] {code}: NO DATA")
        except Exception as e:
            failed.append(code)
            print(f"  [{i}/{total}] {code}: ERROR - {e}")

    return success, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download stock data")
    parser.add_argument(
        "--stocks", help="Comma-separated stock codes (e.g. 300750,002475)"
    )
    parser.add_argument(
        "--from-file", help="File with one stock code per line"
    )
    parser.add_argument("--days", type=int, default=250, help="Price history days")
    parser.add_argument(
        "--price-only", action="store_true", help="Only download price data"
    )
    parser.add_argument(
        "--financial-only", action="store_true", help="Only download financial data"
    )
    args = parser.parse_args()

    # Build stock list
    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",") if c.strip()]
    elif args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            print(f"File not found: {path}")
            sys.exit(1)
        codes = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    else:
        # Use default from config
        from quant_agent.config import get_settings
        settings = get_settings()
        codes = [c.strip() for c in settings.preload_stocks.split(",") if c.strip()]

    if not codes:
        print("No stock codes provided. Use --stocks or --from-file.")
        sys.exit(1)

    print(f"Pre-downloading data for {len(codes)} stocks...")

    from quant_agent.config import get_settings
    from quant_agent.data.service import DataService

    settings = get_settings()
    service = DataService(settings)

    price_ok, price_failed = 0, []
    fin_ok, fin_failed = 0, []

    if not args.financial_only:
        print("\n--- Price Data ---")
        price_ok, price_failed = preload_price_data(service, codes, args.days)

    if not args.price_only:
        print("\n--- Financial Data ---")
        fin_ok, fin_failed = preload_financial_data(service, codes)

    print(f"\n{'=' * 50}")
    if not args.financial_only:
        print(f"Price:     {price_ok}/{len(codes)} successful")
    if not args.price_only:
        print(f"Financial: {fin_ok}/{len(codes)} successful")

    all_failed = set(price_failed) | set(fin_failed)
    if all_failed:
        print(f"Failed:    {', '.join(sorted(all_failed))}")


if __name__ == "__main__":
    main()
