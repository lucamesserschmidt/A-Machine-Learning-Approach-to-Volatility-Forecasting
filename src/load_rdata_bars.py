"""Tier A loader: supervisor's 5-minute stock bars (.RData) -> daily realized measures.

Each bar file holds one stock over the full sample (object 'data.<TICKER>') with
columns Datetime, Price, Size, Delta, Returns, where Returns are clean 5-minute
log returns. From these we build, per day:

    rv5_var : realized variance   = sum r_i^2                 (headline target)
    rv_pos  : upside semivariance  = sum r_i^2 [r_i > 0]       (SHAR)
    rv_neg  : downside semivariance= sum r_i^2 [r_i < 0]       (SHAR)
    rq      : realized quarticity   = (n/3) sum r_i^4          (HARQ)
    bv      : bipower variation     = (pi/2) sum |r_i||r_{i-1}|(HAR-CJ continuous part)

The resulting daily frame is fed through the existing _add_common_daily_predictors,
so all HAR-family and ML models work unchanged. Garman-Klass is retired for these
stocks: this is true 5-minute realized variance, the paper's actual target.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import MIN_VARIANCE
from data_utils import _add_common_daily_predictors

_BAR_MIN_PER_DAY = 50          # require a reasonably complete session
_PI_2 = np.pi / 2.0


def load_bar_rdata(path: Path, asset_name: str) -> pd.DataFrame:
    """Read one .RData bar file -> tidy intraday frame [datetime, asset, price, size, ret]."""
    import pyreadr
    if not Path(path).exists():
        raise FileNotFoundError(path)
    objs = pyreadr.read_r(str(path))
    if not objs:
        raise ValueError(f'No data.frame found in {path} (nested list? use the message loader).')
    # Prefer an object named like 'data.<TICKER>'; else take the first.
    key = next((k for k in objs if str(k).lower().startswith('data.')), list(objs)[0])
    df = objs[key].copy()
    df.columns = [str(c) for c in df.columns]
    rename = {'Datetime': 'datetime', 'Price': 'price', 'Size': 'size', 'Returns': 'ret'}
    df = df.rename(columns=rename)
    need = ['datetime', 'price', 'ret']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f'{path} missing columns {miss}; found {list(df.columns)}')
    df['datetime'] = pd.to_datetime(df['datetime'])
    if 'size' not in df.columns:
        df['size'] = np.nan
    for c in ['price', 'size', 'ret']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['datetime', 'price', 'ret']).sort_values('datetime')
    df['asset'] = asset_name
    return df[['datetime', 'asset', 'price', 'size', 'ret']].reset_index(drop=True)


def bars_to_daily_realized(intraday: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-minute bars to daily realized measures."""
    x = intraday.copy()
    x['date'] = x['datetime'].dt.normalize()
    asset = x['asset'].iloc[0] if len(x) else 'UNKNOWN'

    rows = []
    for day, g in x.groupby('date', sort=True):
        r = g['ret'].values
        n = len(r)
        if n < _BAR_MIN_PER_DAY:
            continue
        rv5 = float(np.sum(r ** 2))
        rv_pos = float(np.sum(r[r > 0] ** 2))
        rv_neg = float(np.sum(r[r < 0] ** 2))
        rq = float(n / 3.0 * np.sum(r ** 4))
        # bipower variation (jump-robust estimate of the continuous variation)
        bv = float(_PI_2 * np.sum(np.abs(r[1:]) * np.abs(r[:-1])))
        price = g['price'].values
        size = g['size'].values
        rows.append({
            'date': pd.Timestamp(day), 'asset': asset,
            'open': float(price[0]), 'close': float(price[-1]),
            'volume': float(np.nansum(size)),
            'dollar_volume': float(np.nansum(price * size)),
            'n_bars': int(n),
            'rv5_var': max(rv5, MIN_VARIANCE),
            'rv_pos': max(rv_pos, MIN_VARIANCE), 'rv_neg': max(rv_neg, MIN_VARIANCE),
            'rq': max(rq, MIN_VARIANCE), 'bv': max(bv, MIN_VARIANCE),
        })
    if not rows:
        raise ValueError(f'No daily RV produced for {asset}; check bar coverage.')
    daily = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    print(f'[STOCK RV5] {asset}: {len(daily):,} daily obs, '
          f'{daily["date"].min().date()} .. {daily["date"].max().date()}')
    return daily


def add_stock_bar_features(path: Path, asset_name: str, macro, start_date=None, end_date=None,
                           orderflow_csv: Path | None = None):
    """Full Tier-A pipeline for one stock: .RData bars -> model-ready feature frame.

    If an order-flow CSV is supplied (Tier B), its daily columns are merged and
    lagged (ofi_lag1, relspr_lag1, imb_lag1) to match the other predictors'
    shift(1) timing.
    """
    intraday = load_bar_rdata(path, asset_name)
    daily = bars_to_daily_realized(intraday)
    df = _add_common_daily_predictors(daily, macro, rv_col='rv5_var',
                                      start_date=start_date, end_date=end_date)
    if orderflow_csv is not None and Path(orderflow_csv).exists():
        of = pd.read_csv(orderflow_csv, parse_dates=['date'])
        keep = [c for c in ['date', 'ofi', 'rel_spread', 'signed_imb'] if c in of.columns]
        df = df.merge(of[keep], on='date', how='left')
        df['ofi_lag1'] = df['ofi'].shift(1)
        df['relspr_lag1'] = df['rel_spread'].shift(1)
        df['imb_lag1'] = df['signed_imb'].shift(1)
        n_of = int(df['ofi_lag1'].notna().sum())
        print(f'[ORDER-FLOW] {asset_name}: merged {n_of} days with order-flow predictors')
    return df


if __name__ == '__main__':
    import sys
    from config import MACRO_FILE
    from data_utils import load_fred_macro, add_horizon_targets, prepare_model_frame
    from models import har_forecasters
    macro = load_fred_macro(MACRO_FILE)
    path = sys.argv[1] if len(sys.argv) > 1 else '/mnt/user-data/uploads/1781968769180_AAPL_returns.RData'
    df = add_stock_bar_features(Path(path), 'AAPL_STOCK_RV5', macro)
    df = add_horizon_targets(df, 'rv5_var', [1, 5, 22])
    print('rows:', len(df), '| range', df['date'].min().date(), '->', df['date'].max().date())
    print('HAR-family available:', [f.name for f in har_forecasters(set(df.columns))])
    print('jump/continuous cols present:', [c for c in ['CD', 'CW', 'CM', 'JD', 'JW', 'JM'] if c in df.columns])
    frame = prepare_model_frame(df, ['RVD', 'RVW', 'RVM', 'VIX_lag1', 'EPU_lag1',
                                     'd_DTB3_lag1', 'd_log_dollar_volume_lag1', 'M1W_lag1'], 'y_h1_rv5_var')
    print('MALL complete-case rows:', len(frame))
