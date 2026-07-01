from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import mean_squared_error

from config import MIN_VARIANCE, VAR_ALPHA, VALID_FRACTION
from models import har_forecasters, ml_forecasters
from tuning import tune_ml_models
from stats_tests import diebold_mariano, kupiec_pof, christoffersen_independence, christoffersen_cc

BENCHMARK = 'HAR'


def make_forecasters(feature_cols, available_cols, tuned):
    return har_forecasters(available_cols) + ml_forecasters(feature_cols, tuned)


def rolling_expanding_forecast(frame, feature_cols, y_col, test_start, valid_start=None,
                               refit_every=21, mode='expanding', rolling_train_window_days=2500,
                               horizon=1, do_tuning=True):
    """Tune once on pre-test data, then roll forward refitting weights monthly."""
    df = frame.sort_values('date').reset_index(drop=True).copy()
    df['_target_'] = df[y_col]
    available_cols = set(df.columns)

    test_mask = df['date'] >= pd.Timestamp(test_start)
    test_positions = np.where(test_mask.values)[0]
    if len(test_positions) == 0:
        raise ValueError('No test observations. Check test_start vs data range.')

    pretest = df[df['date'] < pd.Timestamp(test_start)]
    tuned = {}
    if do_tuning and len(pretest) > 400:
        try:
            tuned = tune_ml_models(pretest, feature_cols, y_col, VALID_FRACTION)
        except Exception as e:
            print(f'  [tuning skipped: {e}]')

    records = []
    starts = list(range(0, len(test_positions), refit_every))
    for s in starts:
        block = test_positions[s:s + refit_every]
        b0 = int(block[0])
        if mode == 'rolling':
            tr0 = max(0, b0 - rolling_train_window_days)
        else:
            tr0 = 0
        train_df = df.iloc[tr0:b0]
        test_df = df.iloc[block]
        if len(train_df) < 300:
            continue
        for fc in make_forecasters(feature_cols, available_cols, tuned):
            try:
                fc.fit(train_df)
                pred = fc.predict(test_df)
            except Exception as e:
                print(f'  [model {fc.name} failed: {e}]')
                continue
            for d, a, yt, p, r in zip(test_df['date'], test_df['asset'], test_df[y_col],
                                      pred, test_df['ret']):
                records.append({'date': d, 'asset': a, 'model': fc.name, 'y_true': yt,
                                'prediction': p, 'ret_next': r})
    return pd.DataFrame(records), tuned


def _mean_qlike(y, f):
    eps = 1e-12
    y = np.clip(np.asarray(y, float), eps, None)
    f = np.clip(np.asarray(f, float), eps, None)
    r = y / f
    return float(np.mean(r - np.log(r) - 1.0))


def summarize_predictions(preds, horizon=1, benchmark=BENCHMARK):
    if preds.empty:
        return pd.DataFrame()
    if benchmark not in preds['model'].unique():
        benchmark = preds['model'].unique()[0]
    bench = preds[preds['model'] == benchmark][['date', 'asset', 'prediction']]
    bench = bench.rename(columns={'prediction': 'bench_pred'})
    rows = []
    for model, g in preds.groupby('model'):
        m = g.merge(bench, on=['date', 'asset'], how='inner')
        mse = mean_squared_error(m['y_true'], m['prediction'])
        bmse = mean_squared_error(m['y_true'], m['bench_pred'])
        ql = _mean_qlike(m['y_true'], m['prediction'])
        bql = _mean_qlike(m['y_true'], m['bench_pred'])
        dm, p = diebold_mariano(m['y_true'], m['bench_pred'], m['prediction'], horizon=horizon, loss='mse')
        dmq, pq = diebold_mariano(m['y_true'], m['bench_pred'], m['prediction'], horizon=horizon, loss='qlike')
        rows.append({'model': model, 'n_test': len(m), 'mse': mse, 'benchmark_mse': bmse,
                     'relative_mse_vs_har': mse / bmse if bmse > 0 else np.nan,
                     'qlike': ql, 'relative_qlike_vs_har': ql / bql if bql > 0 else np.nan,
                     'dm_stat': dm, 'dm_pvalue_model_beats_har': p,
                     'dm_qlike_pvalue_model_beats_har': pq})
    return pd.DataFrame(rows).sort_values('relative_qlike_vs_har').reset_index(drop=True)


def cross_sectional_summary(all_results: pd.DataFrame) -> pd.DataFrame:
    """Average relative MSE and DM-significance share across assets.

    Mirrors the paper's cross-sectional tables: for each (data_source,
    feature_set, horizon, model) report the mean relative MSE over assets and
    the fraction of assets where the model significantly beats HAR at 5%.
    """
    if all_results.empty:
        return pd.DataFrame()
    g = all_results.groupby(['data_source', 'feature_set', 'horizon', 'model'])
    out = g.agg(
        mean_rel_mse=('relative_mse_vs_har', 'mean'),
        median_rel_mse=('relative_mse_vs_har', 'median'),
        mean_rel_qlike=('relative_qlike_vs_har', 'mean'),
        median_rel_qlike=('relative_qlike_vs_har', 'median'),
        n_assets=('asset', 'nunique'),
        share_beats_har_mse_5pct=('dm_pvalue_model_beats_har', lambda s: float(np.mean(s < 0.05))),
        share_beats_har_qlike_5pct=('dm_qlike_pvalue_model_beats_har', lambda s: float(np.mean(s < 0.05))),
    ).reset_index()
    return out.sort_values(['data_source', 'feature_set', 'horizon', 'mean_rel_qlike']).reset_index(drop=True)


def _var_thresholds(sigma, ret, alpha, method, min_history):
    """Return the per-day VaR threshold (negative number) for a given method.

    gaussian : VaR_t = z_alpha * sigma_t  (conditionally normal).
    fhs      : filtered historical simulation. Standardize returns by the
               forecast volatility, then use the EXPANDING empirical alpha-
               quantile of past standardized residuals (no look-ahead). Before
               `min_history` points are available, fall back to the Gaussian
               quantile. This replaces the normality assumption with the
               empirical (fat-tailed, possibly skewed) residual distribution.
    """
    if method == 'gaussian':
        return norm.ppf(alpha) * sigma
    # FHS
    zstd = ret / sigma                          # standardized residuals
    thr = np.empty_like(sigma)
    z_gauss = norm.ppf(alpha)
    for t in range(len(sigma)):
        if t >= min_history:
            q = np.quantile(zstd[:t], alpha)     # past residuals only
        else:
            q = z_gauss
        thr[t] = q * sigma[t]
    return thr


def var_backtest(preds, alpha=VAR_ALPHA, benchmark=BENCHMARK,
                 method='gaussian', min_history=None):
    """One-day VaR from variance forecasts with Kupiec + Christoffersen tests.

    method='gaussian' : conditionally-normal VaR (z_alpha * sigma).
    method='fhs'      : filtered historical simulation (empirical residual
                        quantile), the paper's distribution-free approach.
    A violation occurs when the realized next-day return falls below VaR.
    Reported loss is the asymmetric quantile (tick) loss used in the paper.
    """
    from config import FHS_MIN_HISTORY
    if min_history is None:
        min_history = FHS_MIN_HISTORY
    rows = []
    for model, g in preds.groupby('model'):
        g = g.sort_values('date')
        sigma = np.sqrt(np.clip(g['prediction'].values, MIN_VARIANCE, None))
        r = g['ret_next'].values
        var_thr = _var_thresholds(sigma, r, alpha, method, min_history)
        hit = (r < var_thr).astype(int)
        tick = np.where(hit == 1, (alpha - 1) * (r - var_thr), alpha * (r - var_thr))
        n, k = len(hit), int(hit.sum())
        lr_uc, p_uc = kupiec_pof(hit, alpha)
        lr_ind, p_ind = christoffersen_independence(hit)
        lr_cc, p_cc = christoffersen_cc(hit, alpha)
        rows.append({'model': model, 'var_method': method, 'alpha': alpha, 'n': n,
                     'violations': k, 'violation_rate': k / n if n else np.nan,
                     'expected_rate': alpha, 'mean_tick_loss': float(np.mean(tick)),
                     'kupiec_uc_pvalue': p_uc, 'christoffersen_ind_pvalue': p_ind,
                     'christoffersen_cc_pvalue': p_cc})
    return pd.DataFrame(rows).sort_values('mean_tick_loss').reset_index(drop=True)
