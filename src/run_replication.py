"""Main driver for the upgraded volatility-ML replication.

Pipeline per asset / feature set (MHAR, MALL) / horizon:
  * tune ML hyperparameters once on pre-test data,
  * expanding-window OOS forecasts for HAR/LogHAR/LevHAR/SHAR/HARQ + ML + NN,
  * relative MSE + HAC Diebold-Mariano vs HAR,
  * ALE variable importance (MALL, h=1) for RF and best NN,
  * VaR backtest (h=1): Gaussian AND filtered historical simulation (FHS),
    each with Kupiec + Christoffersen tests,
  * cross-sectional relative-MSE summary across assets.

Data sources:
  * Oxford-Man Realized Library RV5  (headline, cross-section of indices)
  * self-built intraday RV5 from 1-minute bars (robustness / "we can build RV")
  * daily Garman-Klass range variance (robustness, long modern sample)

Run:  python run_replication.py
Flags: --only A B ... | --skip-oxford | --skip-intraday | --skip-daily
Tip: set LIGHT_MODE = True in config.py for a fast smoke run.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import argparse
import pandas as pd
from pathlib import Path

from config import (RESULTS_DIR, PLOTS_DIR, DAILY_ASSET_FILES, INTRADAY_ASSET_FILES,
                    OXFORD_MAN_FILE, OXFORD_MAN_SYMBOLS, MACRO_FILE, START_DATE, END_DATE,
                    HORIZONS, FEATURE_SETS, FEATURE_SETS_OMI, EVALUATION_MODE,
                    REFIT_EVERY_N_DAYS, ROLLING_TRAIN_WINDOW_DAYS, VAR_ALPHA, VAR_METHODS,
                    DAILY_VALID_START, DAILY_TEST_START, INTRADAY_VALID_START, INTRADAY_TEST_START,
                    OXFORD_VALID_START, OXFORD_TEST_START, ALE_FEATURES, ALE_N_BINS,
                    STOCK_BAR_DIR, STOCK_BAR_FILE_PATTERN, STOCK_BAR_TICKERS,
                    STOCK_VALID_START, STOCK_TEST_START)
from data_utils import (load_price_file, load_intraday_file, load_fred_macro, add_volatility_features,
                        add_intraday_realized_variance_features, add_horizon_targets, prepare_model_frame)
from load_oxford_man import load_oxford_man_symbol
from load_rdata_bars import add_stock_bar_features
from evaluation import (rolling_expanding_forecast, summarize_predictions, cross_sectional_summary,
                        var_backtest, make_forecasters)
from ale import ale_importance
from plotting import (plot_relative_mse, plot_ale_panel, plot_ale_importance,
                      plot_cross_sectional_heatmap)


def build_jobs(macro, only_assets=None, skip_stock=False, skip_oxford=False, skip_intraday=False, skip_daily=False):
    """Each job = (asset, df, target_col, data_source, valid_start, test_start, feature_sets)."""
    jobs = []
    # 0) Supervisor's 5-minute stock bars -> real per-stock RV5 (HEADLINE).
    if not skip_stock:
        from config import (STOCK_BAR_DEEPDIVE, ORDERFLOW_DIR, ORDERFLOW_FILE_PATTERN,
                            FEATURE_SETS_OF)
        for tkr in STOCK_BAR_TICKERS:
            asset = f'{tkr}_STOCK_RV5'
            if only_assets and asset not in only_assets:
                continue
            path = STOCK_BAR_DIR / STOCK_BAR_FILE_PATTERN.format(ticker=tkr)
            if not path.exists():
                print(f'[SKIP] {asset}: {path} not found'); continue
            print(f'[LOAD] stock bars {tkr}')
            of_csv = ORDERFLOW_DIR / ORDERFLOW_FILE_PATTERN.format(ticker=tkr)
            of_csv = of_csv if (tkr in STOCK_BAR_DEEPDIVE and of_csv.exists()) else None
            try:
                df = add_stock_bar_features(path, asset, macro, START_DATE, END_DATE, orderflow_csv=of_csv)
            except Exception as e:
                print(f'[SKIP] {asset}: {e}'); continue
            df = add_horizon_targets(df, 'rv5_var', HORIZONS)
            fsets = FEATURE_SETS_OF if (of_csv is not None and 'ofi_lag1' in df.columns) else FEATURE_SETS
            jobs.append((asset, df, 'rv5_var', 'stock_rv5',
                         STOCK_VALID_START, STOCK_TEST_START, fsets))

    # 1) Oxford-Man Realized Library RV5 -- international-index robustness.
    if not skip_oxford and OXFORD_MAN_FILE.exists():
        for sym in OXFORD_MAN_SYMBOLS:
            asset = f'OMI_{sym.lstrip(".")}'
            if only_assets and asset not in only_assets:
                continue
            print(f'[LOAD] Oxford-Man {sym}')
            try:
                df = load_oxford_man_symbol(OXFORD_MAN_FILE, sym, macro, START_DATE, END_DATE)
            except Exception as e:
                print(f'[SKIP] {asset}: {e}'); continue
            df = add_horizon_targets(df, 'rv5_var', HORIZONS)
            jobs.append((asset, df, 'rv5_var', 'oxford_rv5',
                         OXFORD_VALID_START, OXFORD_TEST_START, FEATURE_SETS_OMI))
    elif not skip_oxford:
        print(f'[SKIP] Oxford-Man: {OXFORD_MAN_FILE} not found')

    # 2) Self-built intraday RV5 (robustness; demonstrates RV construction).
    if not skip_intraday:
        for asset, path in INTRADAY_ASSET_FILES.items():
            if only_assets and asset not in only_assets:
                continue
            if not path.exists():
                print(f'[SKIP] {asset}: {path} not found'); continue
            print(f'[LOAD] intraday {asset}')
            intr = load_intraday_file(path, asset)
            df = add_intraday_realized_variance_features(intr, macro, START_DATE, END_DATE)
            df = add_horizon_targets(df, 'rv5_var', HORIZONS)
            jobs.append((asset, df, 'rv5_var', 'intraday_rv5',
                         INTRADAY_VALID_START, INTRADAY_TEST_START, FEATURE_SETS))

    # 3) Daily Garman-Klass (robustness; long modern sample).
    if not skip_daily:
        for asset, path in DAILY_ASSET_FILES.items():
            if only_assets and asset not in only_assets:
                continue
            if not path.exists():
                print(f'[SKIP] {asset}: {path} not found'); continue
            print(f'[LOAD] daily {asset}')
            px = load_price_file(path, asset)
            df = add_volatility_features(px, macro, START_DATE, END_DATE)
            df = add_horizon_targets(df, 'gk_var', HORIZONS)
            jobs.append((asset, df, 'gk_var', 'daily_gk',
                         DAILY_VALID_START, DAILY_TEST_START, FEATURE_SETS))
    return jobs


MODEL_CSV = 'model_comparison_all.csv'
PRED_CSV = 'all_test_predictions.csv'
VAR_CSV = 'var_backtest_h1.csv'
ALE_CSV = 'ale_importance_mall_h1.csv'
_APPEND_FILES = [MODEL_CSV, PRED_CSV, VAR_CSV, ALE_CSV]


def _append_csv(df, path):
    """Append rows to a CSV, writing the header only if the file is new."""
    if df is None or len(df) == 0:
        return
    header = not path.exists()
    df.to_csv(path, mode='a', header=header, index=False)


def _completed_cells(path):
    """Set of (asset, feature_set, horizon) already present, for resuming."""
    if not path.exists():
        return set()
    try:
        done = pd.read_csv(path, usecols=['asset', 'feature_set', 'horizon'])
        return set(map(tuple, done.drop_duplicates().values.tolist()))
    except Exception:
        return set()


def run(only_assets=None, skip_stock=False, skip_oxford=False, skip_intraday=False, skip_daily=False,
        fresh=False, deepdive_only=False, results_dir=None):
    global RESULTS_DIR, PLOTS_DIR
    if results_dir is not None:
        RESULTS_DIR = Path(results_dir)
        PLOTS_DIR = RESULTS_DIR / 'plots'
    if deepdive_only:
        from config import STOCK_BAR_DEEPDIVE
        only_assets = [f'{t}_STOCK_RV5' for t in STOCK_BAR_DEEPDIVE]
        skip_oxford = skip_intraday = skip_daily = True
        print(f'[DEEP-DIVE ONLY] {only_assets} -> {RESULTS_DIR}')
    RESULTS_DIR.mkdir(exist_ok=True); PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    if fresh:
        for f in _APPEND_FILES:
            (RESULTS_DIR / f).unlink(missing_ok=True)
        print('[FRESH] cleared accumulating result CSVs')

    macro = load_fred_macro(MACRO_FILE)
    jobs = build_jobs(macro, only_assets, skip_stock, skip_oxford, skip_intraday, skip_daily)

    # Resume support: skip (asset, feature_set, horizon) cells already on disk.
    done = _completed_cells(RESULTS_DIR / MODEL_CSV)
    if done:
        print(f'[RESUME] {len(done)} cells already complete; they will be skipped')

    for asset, df, target_col, data_source, valid_start, test_start, feature_sets in jobs:
        print(f'\n=== {asset} [{data_source}] {df["date"].min().date()}..{df["date"].max().date()} n={len(df)} ===')
        for fs_name, fs_cols in feature_sets.items():
            if any(c not in df.columns for c in fs_cols):
                continue
            for h in HORIZONS:
                if (asset, fs_name, h) in done:
                    print(f'  [skip done] {fs_name} h={h}'); continue
                y_col = f'y_h{h}_{target_col}'
                frame = prepare_model_frame(df, fs_cols, y_col)
                if len(frame) < 800:
                    print(f'[SKIP] {asset} {fs_name} h={h}: n={len(frame)}'); continue
                n_test = int((frame['date'] >= pd.Timestamp(test_start)).sum())
                if n_test < 40:
                    print(f'[SKIP] {asset} {fs_name} h={h}: only {n_test} test obs'); continue
                print(f'  {fs_name} h={h}: n={len(frame)} test={n_test}')
                preds, tuned = rolling_expanding_forecast(
                    frame, fs_cols, y_col, test_start, valid_start,
                    REFIT_EVERY_N_DAYS, EVALUATION_MODE, ROLLING_TRAIN_WINDOW_DAYS, horizon=h)
                if preds.empty:
                    continue
                preds = preds.assign(feature_set=fs_name, horizon=h, target_col=target_col,
                                     data_source=data_source, asset=asset)
                summ = summarize_predictions(preds, horizon=h)
                summ = summ.assign(asset=asset, feature_set=fs_name, horizon=h,
                                   target_col=target_col, data_source=data_source)
                # --- append this cell's results immediately (crash-safe) ---
                _append_csv(summ[['asset', 'data_source', 'target_col', 'feature_set', 'horizon',
                                  'model', 'n_test', 'mse', 'benchmark_mse', 'relative_mse_vs_har',
                                  'qlike', 'relative_qlike_vs_har', 'dm_stat',
                                  'dm_pvalue_model_beats_har', 'dm_qlike_pvalue_model_beats_har']],
                            RESULTS_DIR / MODEL_CSV)
                _append_csv(preds, RESULTS_DIR / PRED_CSV)
                done.add((asset, fs_name, h))
                plot_relative_mse(summ, PLOTS_DIR / f'relmse_{asset}_{fs_name}_h{h}.png',
                                  f'{asset} {fs_name} h={h}: relative MSE vs HAR')

                if h == 1:
                    for vm in VAR_METHODS:
                        vt = var_backtest(preds, alpha=VAR_ALPHA, method=vm)
                        vt = vt.assign(asset=asset, feature_set=fs_name, data_source=data_source)
                        _append_csv(vt, RESULTS_DIR / VAR_CSV)

                # ALE on MALL, h=1, for the strongest ML model.
                if fs_name == 'MALL' and h == 1:
                    tr = frame[frame['date'] < pd.Timestamp(test_start)].copy()
                    te = frame[frame['date'] >= pd.Timestamp(test_start)].copy()
                    tr['_target_'] = tr[y_col]
                    if len(tr) > 400 and len(te) > 80:
                        ale_targets = ('RandomForest', 'NN2', 'NN3')
                        for fc in make_forecasters(fs_cols, set(frame.columns), tuned):
                            if fc.name in ale_targets:
                                fc.fit(tr)
                                feats = [f for f in ALE_FEATURES if f in fs_cols]
                                imp = ale_importance(fc, te[fs_cols], feats, ALE_N_BINS)
                                imp = imp.assign(asset=asset, model=fc.name, data_source=data_source)
                                _append_csv(imp, RESULTS_DIR / ALE_CSV)
                                plot_ale_importance(imp, PLOTS_DIR / f'ale_vi_{asset}_{fc.name}.png',
                                                    f'{asset} {fc.name}: ALE importance')
                                plot_ale_panel(fc, te[fs_cols], feats,
                                               PLOTS_DIR / f'ale_panel_{asset}_{fc.name}.png',
                                               f'{asset} {fc.name}: ALE')
                                break   # one ML model's ALE is enough per cell

    # --- final aggregation: recomputed from the accumulated CSV on disk, so it
    #     reflects every asset across this and any previous runs (no clobbering) ---
    model_path = RESULTS_DIR / MODEL_CSV
    if model_path.exists():
        res = pd.read_csv(model_path)
        best = res.sort_values('relative_qlike_vs_har').groupby(
            ['asset', 'data_source', 'feature_set', 'horizon'], as_index=False).first()
        best.to_csv(RESULTS_DIR / 'best_model_by_cell.csv', index=False)
        cs = cross_sectional_summary(res)
        cs.to_csv(RESULTS_DIR / 'cross_sectional_relative_mse.csv', index=False)
        for ds in cs['data_source'].unique():
            for fsn in cs['feature_set'].unique():
                plot_cross_sectional_heatmap(cs, PLOTS_DIR / f'heatmap_{ds}_{fsn}.png', ds, fsn)
        print('\n=== CROSS-SECTIONAL SUMMARY (QLIKE = lead metric; lower=better) ===')
        print(cs.to_string(index=False))
    print('\nDone. Outputs in', RESULTS_DIR.resolve())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', nargs='*', default=None, help='subset of asset keys (e.g. OMI_SPX SPY_DAILY_GK)')
    ap.add_argument('--skip-stock', action='store_true', help='skip the 5-min stock-bar headline')
    ap.add_argument('--skip-oxford', action='store_true')
    ap.add_argument('--skip-intraday', action='store_true')
    ap.add_argument('--skip-daily', action='store_true')
    ap.add_argument('--fresh', action='store_true',
                    help='wipe accumulating result CSVs and start clean (otherwise runs resume/accumulate)')
    ap.add_argument('--deepdive-only', action='store_true',
                    help='run only the order-flow deep-dive stocks (produces MALL vs MALL+OF)')
    ap.add_argument('--results-dir', default=None,
                    help='write outputs to this directory instead of results_enhanced')
    args = ap.parse_args()
    run(only_assets=args.only, skip_stock=args.skip_stock, skip_oxford=args.skip_oxford,
        skip_intraday=args.skip_intraday, skip_daily=args.skip_daily, fresh=args.fresh,
        deepdive_only=args.deepdive_only, results_dir=args.results_dir)
