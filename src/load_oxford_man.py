"""Loader for the Oxford-Man Institute Realized Library CSV.

Turns the long-format OMI file into the daily-RV frame the rest of the package
already understands, so OMI's research-grade RV5 can be used as the target with
no other code changes.

Provides, per chosen symbol:
  date, close, volume(=NaN), rv5_var (=rv5), rv_pos, rv_neg (from rsv), rq(absent)
which _add_common_daily_predictors turns into HAR/LogHAR/LevHAR/SHAR features.
(HARQ needs realized quarticity, which OMI does not provide, so HARQ is simply
not generated for OMI data -- the model registry skips it automatically.)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from config import MIN_VARIANCE
from data_utils import _add_common_daily_predictors, add_horizon_targets


def load_oxford_man_symbol(path, symbol, macro, start_date=None, end_date=None):
    raw = pd.read_csv(path)
    raw = raw.rename(columns={raw.columns[0]: 'date'})
    raw = raw[raw['Symbol'] == symbol].copy()
    if raw.empty:
        avail = pd.read_csv(path, usecols=['Symbol'])['Symbol'].unique()
        raise ValueError(f'Symbol {symbol} not found. Available e.g.: {sorted(avail)[:8]} ...')
    raw['date'] = pd.to_datetime(raw['date'], utc=True).dt.tz_localize(None).dt.normalize()
    raw = raw.sort_values('date').drop_duplicates('date')

    rv5 = pd.to_numeric(raw['rv5'], errors='coerce').clip(lower=MIN_VARIANCE)
    rsv = pd.to_numeric(raw['rsv'], errors='coerce').clip(lower=0)        # downside semivariance
    daily = pd.DataFrame({
        'date': raw['date'].values,
        'asset': symbol,
        'close': pd.to_numeric(raw['close_price'], errors='coerce').values,
        'open': pd.to_numeric(raw['open_price'], errors='coerce').values,
        'high': np.nan, 'low': np.nan,
        'volume': np.nan,                       # OMI carries no volume
        'rv5_var': rv5.values,
        'rv_neg': rsv.values,
        'rv_pos': (rv5.values - rsv.values).clip(min=MIN_VARIANCE),
    })
    daily = daily.dropna(subset=['close']).reset_index(drop=True)
    return _add_common_daily_predictors(daily, macro, rv_col='rv5_var',
                                        start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    from config import MACRO_FILE, OXFORD_MAN_FILE
    from data_utils import load_fred_macro, prepare_model_frame
    from models import har_forecasters
    macro = load_fred_macro(MACRO_FILE)
    df = load_oxford_man_symbol(OXFORD_MAN_FILE, '.SPX', macro)
    df = add_horizon_targets(df, 'rv5_var', [1, 5, 22])
    print('SPX rows:', len(df), '| range', df['date'].min().date(), '->', df['date'].max().date())
    har_cols = ['RVD', 'RVW', 'RVM']
    avail = set(df.columns)
    print('HAR-family available:', [f.name for f in har_forecasters(avail)])
    # MALL minus the volume feature (OMI has no volume)
    mall_omi = ['RVD', 'RVW', 'RVM', 'VIX_lag1', 'EPU_lag1', 'd_DTB3_lag1', 'M1W_lag1']
    frame = prepare_model_frame(df, mall_omi, 'y_h1_rv5_var')
    print('MALL_OMI complete-case rows:', len(frame))
    print('Has semivariance (SHAR) cols:', all(c in df.columns for c in ['RVpos_lag1', 'RVneg_lag1']))
    print('Has RQ (HARQ) cols:', 'RQ_RVD_inter' in df.columns)
