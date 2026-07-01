"""Run the replication-plus analytics on a finished replication.

Reads results_enhanced/all_test_predictions.csv (written by run_replication.py)
and produces:
  * mcs_results.csv             -- per-cell Model Confidence Set membership/p-values
  * mcs_share_summary.csv       -- share of assets where each model is in the MCS
  * regime_relative_loss.csv    -- relative loss vs HAR within VIX-tercile regimes
  * friedman_h.csv              -- pairwise interaction strength for refit RF (deep-dive)
  * plots/ale2d_<asset>_<f1>_<f2>.png -- 2-D ALE for the strongest pair

Usage:
  python run_replication_plus.py                 # everything
  python run_replication_plus.py --skip-hstat    # skip the refit-based H-stat/2-D ALE
  python run_replication_plus.py --mcs-B 500     # faster bootstrap
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (RESULTS_DIR, PLOTS_DIR, MACRO_FILE, START_DATE, END_DATE,
                    STOCK_BAR_DIR, STOCK_BAR_FILE_PATTERN, STOCK_BAR_DEEPDIVE,
                    STOCK_TEST_START, ORDERFLOW_DIR, ORDERFLOW_FILE_PATTERN,
                    FEATURE_SETS_OF, RANDOM_STATE, RF_N_ESTIMATORS)
from data_utils import load_fred_macro, add_horizon_targets, prepare_model_frame
from load_rdata_bars import add_stock_bar_features
from replication_plus import mcs_across_cells, regime_split, friedman_h_pairwise
from ale import ale_2d

PRED_PATH = RESULTS_DIR / 'all_test_predictions.csv'
H_PAIRS = [('RVD', 'VIX_lag1'), ('RVD', 'RVW'), ('RVW', 'VIX_lag1'),
           ('RVD', 'M1W_lag1'), ('RVD', 'd_log_dollar_volume_lag1'),
           ('RVD', 'ofi_lag1'), ('RVD', 'relspr_lag1')]


def run_mcs(preds, alpha, B, block):
    print(f'[MCS] running across cells (alpha={alpha}, B={B}, block={block}) ...')
    mcs = mcs_across_cells(preds, loss='qlike', alpha=alpha, B=B, block=block)
    mcs.to_csv(RESULTS_DIR / 'mcs_results.csv', index=False)
    if not mcs.empty:
        share = (mcs.groupby(['data_source', 'feature_set', 'horizon', 'model'])
                 .agg(share_in_mcs=('in_mcs', 'mean'), n_assets=('in_mcs', 'size'),
                      mean_mcs_p=('mcs_pvalue', 'mean')).reset_index()
                 .sort_values(['data_source', 'feature_set', 'horizon', 'share_in_mcs'],
                              ascending=[True, True, True, False]))
        share.to_csv(RESULTS_DIR / 'mcs_share_summary.csv', index=False)
        print(f'[MCS] wrote mcs_results.csv ({len(mcs)} rows) + mcs_share_summary.csv')
    return mcs


def run_regime(preds, macro):
    print('[REGIME] splitting OOS by VIX tercile (calm/normal/turbulent) ...')
    reg = regime_split(preds, macro, loss='qlike', benchmark='HAR', n_regimes=3)
    if not reg.empty:
        reg.to_csv(RESULTS_DIR / 'regime_relative_loss.csv', index=False)
        print(f'[REGIME] wrote regime_relative_loss.csv ({len(reg)} rows)')
    else:
        print('[REGIME] no VIX overlap with test dates -- skipped')
    return reg


def run_hstat(macro, sample, seed):
    from sklearn.ensemble import RandomForestRegressor
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fcols = FEATURE_SETS_OF['MALL']
    all_h = []
    for tkr in STOCK_BAR_DEEPDIVE:
        path = STOCK_BAR_DIR / STOCK_BAR_FILE_PATTERN.format(ticker=tkr)
        if not path.exists():
            print(f'[H-STAT] {tkr}: bars not found, skip'); continue
        of_csv = ORDERFLOW_DIR / ORDERFLOW_FILE_PATTERN.format(ticker=tkr)
        of_csv = of_csv if of_csv.exists() else None
        asset = f'{tkr}_STOCK_RV5'
        try:
            df = add_stock_bar_features(path, asset, macro, START_DATE, END_DATE, orderflow_csv=of_csv)
            df = add_horizon_targets(df, 'rv5_var', [1])
            cols = [c for c in fcols if c in df.columns]
            frame = prepare_model_frame(df, cols, 'y_h1_rv5_var')
            train = frame[frame['date'] < pd.Timestamp(STOCK_TEST_START)]
            if len(train) < 200:
                print(f'[H-STAT] {tkr}: too few train rows, skip'); continue
            X = train[cols]; y = np.log(np.clip(train['y_h1_rv5_var'].values, 1e-12, None))
            rf = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=10,
                                       min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X, y)
            pairs = [(a, b) for a, b in H_PAIRS if a in cols and b in cols]
            h = friedman_h_pairwise(rf, X, pairs, sample=sample, seed=seed)
            h.insert(0, 'asset', asset)
            all_h.append(h)
            print(f'[H-STAT] {asset}: top interaction {h.iloc[0].feature_1}x{h.iloc[0].feature_2} '
                  f'H={h.iloc[0].H:.3f}')
            # 2-D ALE for the strongest pair
            f1, f2 = h.iloc[0]['feature_1'], h.iloc[0]['feature_2']
            e1, e2, A = ale_2d(rf, X, f1, f2, n_bins=20)
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            c1 = 0.5 * (e1[:-1] + e1[1:]); c2 = 0.5 * (e2[:-1] + e2[1:])
            im = ax.contourf(c1, c2, A.T, levels=14, cmap='RdBu_r')
            fig.colorbar(im, ax=ax, label='2-D ALE (interaction)')
            ax.set_xlabel(f1); ax.set_ylabel(f2)
            ax.set_title(f'{asset}: 2-D ALE  {f1} x {f2}  (H={h.iloc[0].H:.3f})')
            fig.tight_layout(); fig.savefig(PLOTS_DIR / f'ale2d_{asset}_{f1}_{f2}.png', dpi=120)
            plt.close(fig)
        except Exception as e:
            print(f'[H-STAT] {tkr}: {e}')
    if all_h:
        out = pd.concat(all_h, ignore_index=True)
        out.to_csv(RESULTS_DIR / 'friedman_h.csv', index=False)
        print(f'[H-STAT] wrote friedman_h.csv ({len(out)} rows) + 2-D ALE plots')
        return out
    return pd.DataFrame()


def main(alpha=0.10, B=1000, block=10, skip_hstat=False, hsample=150, seed=0):
    if not PRED_PATH.exists():
        raise SystemExit(f'{PRED_PATH} not found -- run run_replication.py first.')
    preds = pd.read_csv(PRED_PATH, parse_dates=['date'])
    print(f'loaded {len(preds):,} prediction rows | '
          f'{preds["asset"].nunique()} assets, models {sorted(preds["model"].unique())}')
    run_mcs(preds, alpha, B, block)
    macro = load_fred_macro(MACRO_FILE)
    run_regime(preds, macro)
    if not skip_hstat:
        run_hstat(macro, hsample, seed)
    print('\nReplication-plus analytics done. Outputs in', RESULTS_DIR.resolve())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.10, help='MCS confidence: keep models with p>=alpha')
    ap.add_argument('--mcs-B', type=int, default=1000, help='bootstrap replications')
    ap.add_argument('--block', type=int, default=10, help='moving-block length')
    ap.add_argument('--skip-hstat', action='store_true')
    ap.add_argument('--hsample', type=int, default=150, help='subsample for Friedman H / 2-D ALE')
    args = ap.parse_args()
    main(alpha=args.alpha, B=args.mcs_B, block=args.block, skip_hstat=args.skip_hstat, hsample=args.hsample)
