"""One-time LOBSTER preprocessing: message files -> daily order-flow CSVs.

Run this ONCE before the main replication (it is the heavy, long-pole step).
For each deep-dive stock it streams the per-year LOBSTER message files, rebuilds
the top of book day-by-day, and writes data/orderflow/<ticker>_orderflow.csv with
daily ofi / rel_spread / signed_imb / trade_vol. The main run then merges these
as lagged predictors for the deep-dive panel automatically.

Usage:
  python preprocess_lobster.py                 # all deep-dive stocks
  python preprocess_lobster.py --only AAPL AMD # subset
  python preprocess_lobster.py --resume        # skip stocks whose CSV already exists
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import argparse
from pathlib import Path

from config import (STOCK_BAR_DEEPDIVE, LOBSTER_MSG_DIR, LOBSTER_FILE_PATTERN, LOBSTER_YEARS,
                    ORDERFLOW_DIR, ORDERFLOW_FILE_PATTERN)
from lobster_features import build_lobster_daily


def main(only=None, resume=False):
    ORDERFLOW_DIR.mkdir(parents=True, exist_ok=True)
    tickers = only if only else STOCK_BAR_DEEPDIVE
    for tkr in tickers:
        out_path = ORDERFLOW_DIR / ORDERFLOW_FILE_PATTERN.format(ticker=tkr)
        print(f'\n=== {tkr} ===')
        try:
            build_lobster_daily(tkr, LOBSTER_MSG_DIR, LOBSTER_YEARS,
                                file_pattern=LOBSTER_FILE_PATTERN, out_path=out_path, resume=resume)
        except Exception as e:
            print(f'[FAIL] {tkr}: {e}')
    print('\nOrder-flow preprocessing done. CSVs in', ORDERFLOW_DIR.resolve())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None, help='subset of deep-dive tickers')
    ap.add_argument('--resume', action='store_true', help='skip tickers whose CSV exists')
    args = ap.parse_args()
    main(only=args.only, resume=args.resume)
