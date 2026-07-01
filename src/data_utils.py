from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from config import (
    MIN_VARIANCE,
    FILTER_INTRADAY_REGULAR_HOURS, REGULAR_SESSION_START, REGULAR_SESSION_END,
    INTRADAY_RESAMPLE_RULE, MIN_INTRADAY_5MIN_BARS_PER_DAY,
)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_price_file(path: Path, asset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        out = df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high',
                                 'Low': 'low', 'Close': 'close', 'Volume': 'volume'})
        out['date'] = pd.to_datetime(out['date'])
    elif '<DATE>' in df.columns:
        out = df.rename(columns={'<DATE>': 'date', '<OPEN>': 'open', '<HIGH>': 'high',
                                 '<LOW>': 'low', '<CLOSE>': 'close', '<VOL>': 'volume'})
        out['date'] = pd.to_datetime(out['date'].astype(str), format='%Y%m%d')
    else:
        raise ValueError(f'Unknown price-file format: {path}. Columns: {df.columns.tolist()}')
    out = out[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out['asset'] = asset_name
    out = out.dropna(subset=['date', 'open', 'high', 'low', 'close'])
    out = out[(out['open'] > 0) & (out['high'] > 0) & (out['low'] > 0) & (out['close'] > 0)]
    return out.sort_values('date').drop_duplicates('date')


def load_fred_macro(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    macro = pd.read_csv(path)
    macro.columns = [c.strip() for c in macro.columns]
    if 'date' not in macro.columns:
        raise ValueError('Macro file must contain a date column.')
    macro['date'] = pd.to_datetime(macro['date'])
    for col in ['VIXCLS', 'USEPUINDXD', 'DTB3']:
        if col not in macro.columns:
            raise ValueError(f'Macro file missing required column: {col}')
        macro[col] = pd.to_numeric(macro[col].replace('.', np.nan), errors='coerce')
    macro = macro.sort_values('date').drop_duplicates('date')
    macro['d_DTB3'] = macro['DTB3'].diff()
    return macro[['date', 'VIXCLS', 'USEPUINDXD', 'DTB3', 'd_DTB3']]


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _add_common_daily_predictors(df, macro, rv_col, start_date=None, end_date=None):
    """HAR variables, log lags, leverage terms, macro lags, volume, momentum."""
    df = df.copy().sort_values('date')
    df['ret'] = np.log(df['close']).diff()

    df[rv_col] = pd.to_numeric(df[rv_col], errors='coerce').clip(lower=MIN_VARIANCE)

    # HAR lags (daily / weekly / monthly averages of lagged RV).
    df['RVD'] = df[rv_col].shift(1)
    df['RVW'] = df[rv_col].rolling(5).mean().shift(1)
    df['RVM'] = df[rv_col].rolling(22).mean().shift(1)

    # LogHAR lags.
    df['logRVD'] = np.log(df['RVD'].clip(lower=MIN_VARIANCE))
    df['logRVW'] = np.log(df['RVW'].clip(lower=MIN_VARIANCE))
    df['logRVM'] = np.log(df['RVM'].clip(lower=MIN_VARIANCE))

    # LevHAR: aggregated negative returns r^-_{t-1|t-h}.
    rneg = df['ret'].clip(upper=0.0)
    df['rneg_d'] = rneg.shift(1)
    df['rneg_w'] = rneg.rolling(5).mean().shift(1)
    df['rneg_m'] = rneg.rolling(22).mean().shift(1)

    # SHAR / HARQ inputs (only present if the source provides semivariance / RQ).
    if 'rv_pos' in df.columns:
        df['RVpos_lag1'] = df['rv_pos'].shift(1)
        df['RVneg_lag1'] = df['rv_neg'].shift(1)
    if 'rq' in df.columns:
        df['RQ_lag1'] = df['rq'].shift(1)
        df['RQ_RVD_inter'] = np.sqrt(df['RQ_lag1'].clip(lower=0)) * df['RVD']

    # HAR-CJ: continuous (C) + jump (J) decomposition (needs bipower variation 'bv').
    # J_t = max(RV_t - BV_t, 0); C_t = RV_t - J_t. Andersen-Bollerslev-Diebold (2007).
    if 'bv' in df.columns:
        rvv = df[rv_col]
        jump = (rvv - df['bv']).clip(lower=0.0)
        cont = (rvv - jump).clip(lower=MIN_VARIANCE)
        df['CD'] = cont.shift(1)
        df['CW'] = cont.rolling(5).mean().shift(1)
        df['CM'] = cont.rolling(22).mean().shift(1)
        df['JD'] = jump.shift(1)
        df['JW'] = jump.rolling(5).mean().shift(1)
        df['JM'] = jump.rolling(22).mean().shift(1)

    # Volume, momentum.
    if 'dollar_volume' not in df.columns:
        df['dollar_volume'] = df['close'] * df['volume']
    df['log_dollar_volume'] = np.log(df['dollar_volume'].replace(0, np.nan))
    df['d_log_dollar_volume'] = df['log_dollar_volume'].diff()
    df['d_log_dollar_volume_lag1'] = df['d_log_dollar_volume'].shift(1)
    df['M1W'] = np.log(df['close'] / df['close'].shift(5))
    df['M1W_lag1'] = df['M1W'].shift(1)

    # Macro (forward-filled then lagged to avoid look-ahead).
    df = df.merge(macro, on='date', how='left')
    for col in ['VIXCLS', 'USEPUINDXD', 'd_DTB3']:
        df[col] = df[col].ffill()
    df['VIX_lag1'] = df['VIXCLS'].shift(1)
    df['EPU_lag1'] = df['USEPUINDXD'].shift(1)
    df['d_DTB3_lag1'] = df['d_DTB3'].shift(1)

    if start_date:
        df = df[df['date'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['date'] <= pd.Timestamp(end_date)]
    return df.reset_index(drop=True)


def add_volatility_features(px, macro, start_date=None, end_date=None):
    """Daily OHLC: Garman-Klass variance target (robustness layer)."""
    df = px.copy().sort_values('date')
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])
    df['gk_var'] = (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).clip(lower=MIN_VARIANCE)
    return _add_common_daily_predictors(df, macro, rv_col='gk_var', start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# Intraday -> realized variance (headline target)
# ---------------------------------------------------------------------------
def load_intraday_file(path: Path, asset_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.loc[:, ~df.columns.str.match(r'^unnamed')]
    if '0' in df.columns and 'date' in df.columns:
        df = df.drop(columns=['0'])
    required = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Intraday file missing columns {missing}. Found: {df.columns.tolist()}')
    keep = required + [c for c in ['barcount', 'average'] if c in df.columns]
    out = df[keep].copy()
    date_clean = out['date'].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
    out['datetime'] = pd.to_datetime(date_clean, format='%Y%m%d %H:%M:%S', errors='coerce')
    if out['datetime'].isna().mean() > 0.5:
        out['datetime'] = pd.to_datetime(date_clean, errors='coerce')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        out[c] = pd.to_numeric(out[c], errors='coerce')
    out = out.dropna(subset=['datetime', 'open', 'high', 'low', 'close'])
    out = out[(out['open'] > 0) & (out['high'] > 0) & (out['low'] > 0) & (out['close'] > 0)]
    out = out.drop_duplicates().sort_values('datetime')
    out = out.groupby('datetime', as_index=False).agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
    print(f'[INTRADAY] {path.name}: {len(out):,} 1-min bars, '
          f'{out["datetime"].dt.normalize().nunique():,} trading days, '
          f'{out["datetime"].min().date()} to {out["datetime"].max().date()}')
    out['asset'] = asset_name
    return out[['datetime', 'asset', 'open', 'high', 'low', 'close', 'volume']]


def intraday_to_daily_realized_variance(intraday, filter_regular_hours=FILTER_INTRADAY_REGULAR_HOURS):
    """5-minute realized variance + positive/negative semivariance + realized quarticity."""
    x = intraday.copy().sort_values('datetime')
    if filter_regular_hours:
        x = x.set_index('datetime').between_time(REGULAR_SESSION_START, REGULAR_SESSION_END).reset_index()
    x['date'] = x['datetime'].dt.normalize()
    asset = x['asset'].iloc[0] if len(x) else 'UNKNOWN'

    rows = []
    for day, g in x.groupby('date', sort=True):
        g = g.set_index('datetime').sort_index()
        close5 = g['close'].resample(INTRADAY_RESAMPLE_RULE).last().dropna()
        if len(close5) < MIN_INTRADAY_5MIN_BARS_PER_DAY:
            continue
        r5 = np.log(close5).diff().dropna().values
        n = len(r5)
        rv5 = float(np.sum(r5 ** 2))
        rv_pos = float(np.sum(r5[r5 > 0] ** 2))
        rv_neg = float(np.sum(r5[r5 < 0] ** 2))
        rq = float(n / 3.0 * np.sum(r5 ** 4))         # realized quarticity
        rows.append({
            'date': pd.Timestamp(day), 'asset': asset,
            'open': float(g['open'].iloc[0]), 'high': float(g['high'].max()),
            'low': float(g['low'].min()), 'close': float(g['close'].iloc[-1]),
            'volume': float(g['volume'].sum()),
            'dollar_volume': float((g['close'] * g['volume']).sum()),
            'n_5min_bars': int(len(close5)),
            'rv5_var': max(rv5, MIN_VARIANCE),
            'rv_pos': max(rv_pos, MIN_VARIANCE), 'rv_neg': max(rv_neg, MIN_VARIANCE),
            'rq': max(rq, MIN_VARIANCE),
        })
    if not rows:
        raise ValueError('No daily RV produced from intraday data. Check session filter / time format.')
    daily = pd.DataFrame(rows).sort_values('date').reset_index(drop=True)
    print(f'[INTRADAY RV] {asset}: {len(daily):,} daily RV obs, '
          f'{daily["date"].min().date()} to {daily["date"].max().date()}')
    return daily


def add_intraday_realized_variance_features(intraday, macro, start_date=None, end_date=None):
    daily = intraday_to_daily_realized_variance(intraday)
    return _add_common_daily_predictors(daily, macro, rv_col='rv5_var', start_date=start_date, end_date=end_date)


# ---------------------------------------------------------------------------
# Targets / frame prep
# ---------------------------------------------------------------------------
def add_horizon_targets(df, target_col, horizons=None):
    """Forward average-volatility targets, leak-free.

    h=1:  y at t is RV at t+1.
    h>1:  y at t is mean RV over t+1..t+h.
    """
    if horizons is None:
        horizons = [1, 5, 22]
    out = df.copy().sort_values('date')
    out['ret_next'] = out['ret'].shift(-1)
    for h in horizons:
        out[f'y_h{h}_{target_col}'] = out[target_col].shift(-1).rolling(h, min_periods=h).mean().shift(-(h - 1))
    return out


def prepare_model_frame(df, feature_cols, y_col):
    ret_col = 'ret_next' if 'ret_next' in df.columns else 'ret'
    extra = [c for c in ['logRVD', 'logRVW', 'logRVM', 'rneg_d', 'rneg_w', 'rneg_m',
                         'RVpos_lag1', 'RVneg_lag1', 'RQ_RVD_inter',
                         'CD', 'CW', 'CM', 'JD', 'JW', 'JM'] if c in df.columns]
    model_cols = sorted(set(feature_cols + extra))
    cols = ['date', 'asset', ret_col] + model_cols + [y_col]
    out = df[cols].copy()
    if ret_col != 'ret':
        out = out.rename(columns={ret_col: 'ret'})
    # Require complete cases across every column any model might use, so HAR-family
    # variants (LevHAR/SHAR/HARQ) and log lags never receive NaN inputs.
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=model_cols + [y_col, 'ret'])
    return out.reset_index(drop=True)
