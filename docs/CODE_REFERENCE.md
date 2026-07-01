# Volatility ML Replication — Source Code Documentation

Detailed technical documentation for the full codebase (~2,700 lines of Python across
16 modules). This explains exactly what every module, class, and function does, and how
data flows through the pipeline from raw files to final results.

---

## High-level pipeline

The entire project is a pipeline with four stages:

```
RAW DATA                    DAILY FEATURES              FORECASTS               ANALYSIS
─────────                   ──────────────              ─────────               ────────
5-min bars (.RData)    ──►  RV5, semivariance,    ──►   expanding-window   ──►  relative MSE/QLIKE
LOBSTER messages       ──►  RQ, bipower, OFI,           OOS forecasts           DM tests, MCS
Oxford-Man CSV         ──►  spread, HAR lags,           (HAR family + ML)       regime split
FRED macro CSV         ──►  macro lags, targets                                 H-stat, ALE, VaR
```

The first stage (loading + feature engineering) produces a single `DataFrame` per
asset with date-indexed rows. Each row contains the realized variance target,
all lagged predictors, and the forward-looking horizon targets. The second stage
runs expanding-window out-of-sample forecasts with periodic refitting. The third
stage evaluates and compares. Every stage is checkpointed to CSV so runs can be
interrupted and resumed.

---

## Module-by-module reference

### `config.py` (234 lines) — central configuration

Every tunable parameter lives here. Nothing is hardcoded elsewhere.

**Key settings:**

| Setting | What it controls |
|---|---|
| `STOCK_BAR_DIR`, `STOCK_BAR_TICKERS` (54) | paths to the supervisor's 5-min bar files |
| `STOCK_BAR_DEEPDIVE` (7 tickers) | which stocks also run with order-flow predictors |
| `OXFORD_MAN_SYMBOLS` (8 indices) | which OMI indices to include |
| `HORIZONS = [1, 5, 22]` | forecast horizons in trading days |
| `FEATURE_SETS` | M_HAR = `[RVD, RVW, RVM]`, M_ALL = M_HAR + VIX, EPU, T-bill, dollar volume, Monday |
| `FEATURE_SETS_OF` | same as above but adds `MALL_OF` with `ofi_lag1, relspr_lag1, imb_lag1` |
| `FEATURE_SETS_OMI` | M_ALL without dollar volume (OMI has no volume data) |
| `REFIT_EVERY_N_DAYS` | how often model weights are refit (21 = monthly, 63 = quarterly) |
| `EVALUATION_MODE` | `'expanding'` (paper's approach) or `'rolling'` |
| `ML_LOG_TARGET = True` | train ML on log(RV) with Jensen correction (critical for stability) |
| `NN_ARCHITECTURES` | geometric-pyramid MLP shapes: NN1=(2,), NN2=(4,2), NN3=(8,4,2) |
| `NN_N_SEEDS`, `NN_ENSEMBLE_TOP` | train N seeds, keep top K by validation loss |
| `RF_GRID`, `GB_GRID`, etc. | hyperparameter grids for one-shot tuning |
| `LIGHT_MODE` | when True, overrides everything to a fast smoke test (1 stock, 1 horizon, small grids) |

**Split dates per data source:**
- Stocks: valid 2016, test 2017 (data spans 2009–2019)
- Oxford-Man: valid 2015, test 2016 (data spans 2000–2018)

---

### `data_utils.py` (246 lines) — shared feature engineering

The feature-construction backbone. Every data source (stocks, OMI, daily GK) passes
through `_add_common_daily_predictors`, which builds the same predictor set from a
daily RV frame + macro data.

**`load_price_file(path, asset_name)`**
Reads a daily OHLCV CSV (Stooq `.us.txt` or standard format) → DataFrame with
`date, asset, open, high, low, close, volume`. Computes Garman-Klass variance
as the volatility target for the legacy daily panel.

**`load_fred_macro(path)`**
Reads the FRED macro CSV → DataFrame with `date, VIXCLS, USEPUINDXD, DTB3`.
These become the macro predictors in M_ALL (lagged, differenced where needed).

**`_add_common_daily_predictors(df, macro, rv_col, start_date, end_date)`**
The core feature builder. Takes any daily frame with a realized-variance column
and constructs ALL the predictors used by every model:

1. **HAR lags** (all shifted by 1 to avoid look-ahead):
   - `RVD` = yesterday's RV (daily component)
   - `RVW` = mean RV over the past 5 days (weekly component)
   - `RVM` = mean RV over the past 22 days (monthly component)
2. **Log-HAR lags**: `logRVD, logRVW, logRVM` = log of the above (for LogHAR)
3. **Leverage lags**: `rneg_d, rneg_w, rneg_m` = mean of min(return, 0) (for LevHAR)
4. **Semivariance lags** (if `rv_pos`/`rv_neg` columns exist): `RVpos_lag1, RVneg_lag1` (for SHAR)
5. **Realized quarticity interaction** (if `rq` exists): `RQ_RVD_inter = sqrt(RQ) × RVD` (for HARQ)
6. **Jump/continuous decomposition** (if bipower `bv` exists):
   - Jump: `J = max(RV − BV, 0)`; Continuous: `C = RV − J`
   - Lags: `CD, CW, CM` (continuous daily/weekly/monthly), `JD, JW, JM` (jump lags)
   - These are the HAR-CJ inputs (Andersen-Bollerslev-Diebold 2007)
7. **Macro predictors**: VIX (lagged), EPU (lagged), Δ T-bill (differenced, lagged)
8. **Market microstructure**: dollar volume (log, differenced, lagged), Monday dummy

**`add_horizon_targets(df, target_col, horizons)`**
Creates forward-looking target columns: `y_h1_<col>` = next-day RV,
`y_h5_<col>` = mean RV over next 5 days, `y_h22_<col>` = mean over next 22 days.
Uses `.shift(-h)` with rolling mean — no data leakage because the features are
all lagged by at least 1 day.

**`prepare_model_frame(df, feature_cols, y_col)`**
Selects the feature columns + target + date/asset, drops rows with any NaN
(complete-case analysis), and returns the clean frame ready for the forecast loop.

---

### `load_rdata_bars.py` (133 lines) — Tier A: stock bars → RV5

Reads the supervisor's `.RData` 5-minute bar files and produces daily realized
measures. This is the **headline data source** — genuine intraday equity RV5,
matching the paper's target.

**`load_bar_rdata(path, asset_name)`**
Reads one `.RData` file (object `data.<TICKER>`) with columns
`Datetime, Price, Size, Delta, Returns` → tidy intraday frame.
`Returns` are clean 5-min log returns (pre-computed by the data provider).

**`bars_to_daily_realized(intraday)`**
Aggregates 5-min bars to daily realized measures:
- `rv5_var = Σ r²` (realized variance — the target)
- `rv_pos = Σ r² [r > 0]` (upside semivariance, for SHAR)
- `rv_neg = Σ r² [r < 0]` (downside semivariance, for SHAR)
- `rq = (n/3) · Σ r⁴` (realized quarticity, for HARQ)
- `bv = (π/2) · Σ |rᵢ| · |rᵢ₋₁|` (bipower variation, for HAR-CJ jump/continuous split)

Days with fewer than 50 bars are dropped (incomplete sessions).

**`add_stock_bar_features(path, asset_name, macro, ..., orderflow_csv=None)`**
Full Tier-A pipeline: load bars → daily RV → `_add_common_daily_predictors`.
If an order-flow CSV is provided (Tier B), merges it and creates lagged
predictors `ofi_lag1, relspr_lag1, imb_lag1`.

---

### `lobster_features.py` (277 lines) — Tier B: LOBSTER messages → order-flow

Reconstructs the top of book from raw LOBSTER message data and computes three
daily microstructure predictors.

**LOBSTER event types used:**
- Type 1 = new limit order (add to book)
- Type 2 = partial cancellation (reduce size)
- Type 3 = full deletion (remove from book)
- Type 4 = visible execution (trade, removes from book)
- Type 5 = hidden execution (trade, not in visible book)
- Type 6 = cross/auction (skipped)

**`_process_day(time, etype, oid, size, price, direction)`**
The core reconstructor. For one trading day:
1. Maintains a **bid book** (SortedDict, price → total size) and **ask book**.
2. Replays every message in time order: adds insert to the correct side,
   reduces/removes on cancels and executions, tracks the order registry.
3. **Uncross-on-insert fix:** when a new limit order's price crosses the
   opposite best quote, clears the stale opposite levels. This is necessary
   because without the paired LOBSTER orderbook file, stale quotes accumulate
   and the book comes out crossed (negative spreads). This was the main bug
   we diagnosed and fixed.
4. After each book-changing event, if both sides have quotes:
   - Records the spread: `(best_ask − best_bid) / midpoint`
   - Computes the **OFI increment** (Cont-Kukanov-Stoikov 2014): measures net
     changes in best-quote depth; positive = buying pressure
5. For executions: accumulates signed volume (`−Direction × Size`) for the
   **signed trade imbalance**.

Returns: `(ofi, rel_spread, signed_imb, trade_vol)` for the day.

**`read_message_df(path)`**
Reads `.parquet` (preferred, memory-efficient) or `.RData` (nested list) message files.

**`_year_rows_streaming(path, asset, batch_size)`**
Memory-flat streaming for large `.parquet` files (e.g. AAPL at ~2 GB/year):
reads in row-group batches via PyArrow, buffers incomplete days across batch
boundaries, processes each complete day, then discards the raw data. Memory
stays constant regardless of file size.

**`_year_rows_inmemory(path, asset)`**
Full-load for small `.RData` files (e.g. AGEN at ~50 MB/year).

**`build_lobster_daily(ticker, msg_dir, years, ...)`**
Orchestrates the full preprocessing for one stock across all years.
Per-year incremental CSV append + per-year resume: each completed year is
written immediately, so a crash loses at most the in-progress year.
Prefers `.parquet` files (streamed), falls back to `.RData` (full-load).

---

### `load_oxford_man.py` (65 lines) — OMI realized library loader

**`load_oxford_man_symbol(path, symbol, macro, ...)`**
Reads the long-format OMI CSV, filters to one symbol (e.g. `.SPX`), extracts
`rv5` (5-min realized variance) and `bv` (bipower variation) columns, then
feeds through `_add_common_daily_predictors`. OMI uses `FEATURE_SETS_OMI`
(no dollar volume column available).

---

### `models.py` (205 lines) — forecaster definitions

A unified interface: every forecaster has `.name`, `.fit(train_df)`, `.predict(test_df)`.
The forecast loop treats all models identically.

**`class MLForecaster`**
Wraps any sklearn estimator. Key features:
- **Log-target option:** when `log_target=True`, fits on `log(RV)` and back-transforms
  with the Jensen bias correction: `E[exp(f)] = exp(f + 0.5 · σ²_resid)`.
- **Log-forecast clip:** for log-target models, clips the log-prediction to
  `[log_min − margin, log_max + margin]` before exponentiating. This prevents
  a linear model from extrapolating to an absurd variance on an extreme feature
  day (which would blow up MSE). This was the "pristine-MSE fix."
- **Level-model winsorization:** for level-target models (HAR, SHAR, HARQ, HARCJ),
  forecasts are clipped to `[1st percentile of training RV, 3× training max]`.
  This was the "LevHAR fix" — without it, LevHAR's OLS extrapolation produced
  explosive forecasts on a handful of stocks, blowing up mean QLIKE to ~3×.

**`class EnsembleNN`**
Geometric-pyramid MLP seed ensemble. Trains `N` networks (different random seeds),
ranks by internal validation MSE, averages the best `K`. All NNs use log-target
with standardized features and targets. Architectures mirror the paper's NN1–NN4
(widths halve per layer: e.g. NN3 = 8→4→2).

**`har_forecasters(available_cols)`**
Returns the HAR-family models whose required columns are present:

| Model | Inputs | Log-target? | What it tests |
|---|---|---|---|
| HAR | RVD, RVW, RVM | No | The baseline (Corsi 2009) |
| LogHAR | logRVD, logRVW, logRVM | Yes | Log transform advantage |
| LevHAR | HAR + leverage lags | No | Asymmetric/leverage effect |
| SHAR | semivariance lags + RVW, RVM | No | Up/down decomposition |
| HARQ | HAR + √RQ × RVD interaction | No | Time-varying parameters |
| HARCJ | CD, CW, CM, JD, JW, JM | No | Jump/continuous split |

**`ml_forecasters(feature_cols, tuned)`**
Returns Ridge, Lasso, Elastic Net (all log-target), Random Forest, Gradient Boosting
(both log-target), and all configured NN architectures. Hyperparameters come from
the tuning results if available.

---

### `tuning.py` (92 lines) — hyperparameter tuning

**`tune_ml_models(pretest_frame, feature_cols, y_col, valid_fraction)`**
One-shot, time-aware tuning on the validation tail of the pre-test data.
Splits chronologically (last `valid_fraction` of pre-test = validation set),
fits each model variant, scores on validation QLIKE (log-scale), picks the best.
Returns a dict of tuned parameters keyed by model name. This runs once before the
forecast loop starts; during the loop, only weights are refit (not hyperparameters).

---

### `evaluation.py` (177 lines) — forecast loop + evaluation

**`rolling_expanding_forecast(frame, feature_cols, y_col, test_start, ...)`**
The main out-of-sample engine:
1. Tunes hyperparameters once on pre-test data (calls `tune_ml_models`).
2. Walks forward through the test period in blocks of `refit_every` days.
3. At each block: selects the training window (expanding from the start, or
   rolling with a fixed width), fits all models, predicts the block.
4. Returns a DataFrame of `(date, asset, model, y_true, prediction)` rows.

**`summarize_predictions(preds, horizon, benchmark='HAR')`**
Computes relative MSE and QLIKE vs the benchmark for each model, plus the
Diebold-Mariano test (HAC Newey-West + Harvey-Leybourne-Newbold small-sample
correction) p-value for "model beats HAR."

**`cross_sectional_summary(all_results)`**
Aggregates per-model results across assets: mean/median relative loss, and
the share of assets where the model significantly beats HAR at 5%.

**`var_backtest(preds, ...)`**
Value-at-Risk backtest:
- **Gaussian VaR:** assumes normal returns, scales by √(forecast variance).
- **FHS (Filtered Historical Simulation):** standardizes past returns by
  model-fitted volatility, takes the empirical quantile of the standardized
  residuals. More accurate for fat-tailed returns.
- Tests: Kupiec (unconditional coverage) + Christoffersen (independence + joint).

---

### `stats_tests.py` (144 lines) — econometric tests

**`diebold_mariano(y_true, pred_benchmark, pred_model, horizon, loss)`**
Diebold-Mariano test for equal predictive ability. Computes the loss differential
series `d_t = L(benchmark) − L(model)`, estimates its long-run variance with
Newey-West HAC (bandwidth = horizon), and applies the Harvey-Leybourne-Newbold
finite-sample correction. Supports both MSE and QLIKE loss.

**`kupiec_pof(violations, alpha)`**
Kupiec (1995) proportion-of-failures test: likelihood ratio test that the
observed VaR violation rate equals the nominal α.

**`christoffersen_independence(violations)`**
Christoffersen (1998) independence test: are violations clustered (today's
violation predicts tomorrow's)? Uses a Markov-chain likelihood ratio.

**`christoffersen_cc(violations, alpha)`**
Joint conditional coverage test = Kupiec + independence combined.

---

### `ale.py` (143 lines) — Accumulated Local Effects

**`ale_1d(model, X, feature, n_bins)`**
First-order ALE (Apley & Zhu 2020). Partitions the feature into quantile bins,
computes the average local effect (prediction change when moving from the bin's
lower to upper edge), accumulates, and centers. Returns `(bin_edges, ale_values)`.
Unlike PDP, ALE is unbiased even when features are correlated — this is why the
paper uses it.

**`ale_importance(model, X, features, n_bins)`**
ALE-based variable importance: the standard deviation of each feature's 1-D ALE
curve. Higher std = the model's prediction varies more with that feature = more
important. Returns a ranked DataFrame.

**`ale_2d(model, X, f1, f2, n_bins)`**
Second-order (interaction) ALE. Computes the pure interaction surface: the part
of the joint effect that is NOT explained by the two marginal 1-D effects.
A flat surface (~0) means no interaction; structure means the model's response
to f1 depends on the value of f2. This directly visualizes the interactions the
paper claims ML exploits but never measured.

---

### `replication_plus.py` (195 lines) — critical-assessment analytics

**`model_confidence_set(loss, model_names, alpha, B, block, seed)`**
Model Confidence Set (Hansen, Lunde & Nason 2011). Given a (T × m) matrix of
per-observation losses, identifies the subset of models that are statistically
indistinguishable from the best at confidence level (1 − α).

Algorithm: iteratively eliminates the worst model (largest range statistic) if
the bootstrap p-value < α; stops when the null "all surviving models are equal"
cannot be rejected. Uses a moving-block bootstrap to handle serial correlation
in the loss differentials.

Returns each model's MCS p-value and whether it's in the confidence set.

**`mcs_across_cells(preds, loss, alpha, B, block)`**
Runs an MCS per (data_source, asset, feature_set, horizon) cell from the
stored predictions. This is the proper joint-inference replacement for the
paper's pairwise Diebold-Mariano comparisons.

**`regime_split(preds, macro, loss, benchmark, n_regimes)`**
Splits the out-of-sample period into VIX-tercile regimes (calm / normal /
turbulent) and computes relative loss vs HAR within each regime. Tests whether
the ML edge is concentrated in particular market conditions.

**`friedman_h_pairwise(model, X, pairs, sample, seed)`**
Friedman's H-statistic for feature interactions. For each pair (f1, f2):
1. Compute centered partial dependence for f1 alone, f2 alone, and (f1, f2) jointly.
2. H² = Σ(PD_joint − PD_f1 − PD_f2)² / Σ PD_joint²
3. H ∈ [0, 1]; 0 = no interaction, 1 = pure interaction.

Uses a subsample for speed (the full-data PD computation would be O(n²) per pair).

---

### `simulation.py` (311 lines) — controlled simulation study

Three experiments on synthetic data where the truth is known.

**`simulate_log_har(n, burn, nonlinear_strength, df_innov, leverage, noise_dim, seed)`**
Generates daily log-RV from a HAR recursion with:
- Persistence coefficients (0.36 + 0.33 + 0.27 = 0.96)
- A nonlinear exogenous driver: `strength × X² × sign(X)` (VIX-like)
- Student-t innovations with tunable degrees of freedom
- A leverage effect (negative returns → higher next-day vol)
- Pure-noise AR(1) regressors (to test overfitting resistance)

**`experiment_A()`** — fits LogHAR and RF on SIM_HAR (lags only) vs SIM_ALL
(lags + driver + noise). Shows that RF only beats LogHAR when the nonlinear
driver is in the information set — the paper's M_HAR→M_ALL mechanism.

**`experiment_B()`** — varies tail heaviness from Gaussian to t(3). Shows
the log-target advantage grows with heavier tails — motivates LogHAR + QLIKE.

**`experiment_C()`** — compares the ACF of HAR's vs RF's fitted values against
the true long-memory ACF at h=1 and h=22. RF tracks the slow decay better at
the monthly horizon — explains why ML gains grow with horizon.

---

### `run_replication.py` (268 lines) — main driver

**`build_jobs(macro, ...)`**
Constructs the job list: each job = (asset, prepared DataFrame, target column,
data source label, valid/test dates, feature set dict). Jobs are built from
four sources in priority order:
1. **Stock bars** (Tier A, headline): 54 stocks, `FEATURE_SETS` or `FEATURE_SETS_OF`
2. **Oxford-Man** (robustness): 8 indices, `FEATURE_SETS_OMI`
3. **Intraday** (self-built SPY): `FEATURE_SETS`
4. **Daily GK** (legacy): `FEATURE_SETS`

Each source can be skipped with `--skip-stock`, `--skip-oxford`, etc.

**`run(...)`**
For each job, for each feature set × horizon:
1. Checks the resume cache (skips already-done cells).
2. Calls `rolling_expanding_forecast` → predictions.
3. Calls `summarize_predictions` → relative loss + DM tests.
4. Appends results to accumulating CSVs (model_comparison_all.csv,
   all_test_predictions.csv).
5. Runs VaR backtest (h=1 only) and ALE importance (MALL only).
6. After all cells: computes cross-sectional summary.

**Checkpointing:** every completed (asset, feature_set, horizon) cell is
immediately appended to CSV. If the process is killed, restarting picks up
where it left off. Use `--fresh` to wipe and restart.

---

### `run_replication_plus.py` (140 lines) — replication-plus driver

Reads `all_test_predictions.csv` (written by the main driver) and runs the
three critical-assessment analytics:
1. MCS across all cells → `mcs_results.csv` + `mcs_share_summary.csv`
2. Regime split → `regime_relative_loss.csv`
3. Friedman H + 2-D ALE for each deep-dive stock → `friedman_h.csv` + plots

**Flags:** `--skip-hstat` (skip the refit-based H-stat/ALE), `--mcs-B 500`
(fewer bootstrap reps for speed).

---

### `preprocess_lobster.py` (43 lines) — LOBSTER preprocessing driver

One-time script that calls `build_lobster_daily` for each deep-dive stock.
Supports `--resume` (continue from where it stopped) and `--only AAPL AMD`
(process a subset). Outputs go to `data/orderflow/<ticker>_orderflow.csv`.

---

### `plotting.py` (69 lines) — visualization helpers

- `plot_relative_mse`: bar chart of relative QLIKE/MSE vs HAR for all models.
- `plot_ale_panel`: grid of 1-D ALE curves for the top features.
- `plot_ale_importance`: horizontal bar chart of ALE-based variable importance.
- `plot_cross_sectional_heatmap`: heatmap of median relative QLIKE across
  (model × horizon), aggregated over all assets. The main summary figure.

---

## Data flow diagram

```
AAPL_returns.RData
    │
    ▼
load_bar_rdata()          ──►  intraday frame (datetime, price, ret)
    │
    ▼
bars_to_daily_realized()  ──►  daily (rv5_var, rv_pos, rv_neg, rq, bv)
    │
    ▼
_add_common_daily_predictors()
    │   merges FRED macro
    │   builds RVD/RVW/RVM, log lags, leverage, semivar, RQ, jump/cont
    │   if orderflow_csv provided: merges ofi_lag1, relspr_lag1, imb_lag1
    ▼
add_horizon_targets()     ──►  adds y_h1_rv5_var, y_h5_rv5_var, y_h22_rv5_var
    │
    ▼
prepare_model_frame()     ──►  complete-case frame (features + target + date/asset)
    │
    ▼
rolling_expanding_forecast()
    │   tunes hyperparameters once (tune_ml_models)
    │   walks forward in refit blocks
    │   fits all HAR + ML models, predicts each block
    ▼
summarize_predictions()   ──►  relative MSE/QLIKE, DM p-values
    │
    ▼
cross_sectional_summary() ──►  mean/median across assets, share beating HAR
    │
    ▼
[CSV outputs]             ──►  model_comparison_all.csv, all_test_predictions.csv,
                               cross_sectional_relative_mse.csv, var_backtest_h1.csv,
                               ale_importance_mall_h1.csv
```

---

## Key design decisions explained

**Why log-target?** On raw RV levels, the right-skewed crisis spikes dominate the
MSE loss and make least-squares OLS-HAR nearly unbeatable (the ML models' flexibility
becomes a liability). Training on log(RV) stabilizes the scale, and the Jensen
correction (`exp(f + 0.5·σ²)`) ensures unbiased back-transformation. The simulation
study (Experiment B) demonstrates this: the log advantage grows monotonically with
tail heaviness.

**Why QLIKE over MSE?** QLIKE (Patton 2011) is the only loss function that is both
robust to the variance proxy (consistent ranking regardless of which RV estimator
you use) and less dominated by tail events than MSE. MSE can make a model look bad
because of one crisis day; QLIKE cannot. We report both but lead with QLIKE.

**Why expanding window, not rolling?** The paper uses expanding (all history up to
today). Rolling discards old data, which hurts in a short sample. We support both
via config but default to expanding.

**Why the log-forecast clip?** A log-target linear model (Ridge, Lasso) can
extrapolate to log-predictions far outside the training range on an extreme
feature day. Exponentiating that gives an absurd variance forecast (we saw 60×
blow-ups). The clip bounds the log-prediction to `[train_min − 0.5·range,
train_max + 0.5·range]` before exponentiating — generous enough to never bind
in normal conditions, tight enough to prevent pathological MSE.

**Why the level-model winsorization?** Same idea for the non-log models. LevHAR's
OLS regression on raw RV can produce negative or explosive forecasts on extreme
days. Floor at the 1st training percentile, cap at 3× training max.

**Why per-year resume in LOBSTER preprocessing?** AAPL's message files are ~2 GB
per year (tens of millions of messages). The pure-Python `rdata` reader expands
that several-fold in memory, causing OOM kills on a laptop. The fix: (1) stream
from Parquet in row-group batches (memory stays flat), (2) write each completed
year immediately and skip done years on restart.

---

## How to run

```bash
# install dependencies
pip install -r requirements.txt

# 1. Place data (see data/README.md for expected layout)

# 2. Smoke test (LIGHT_MODE = True in config.py)
python src/run_replication.py --only AAPL_STOCK_RV5 --skip-oxford --skip-intraday --skip-daily --fresh

# 3. LOBSTER preprocessing (one-time; large stocks need Parquet export first)
python src/preprocess_lobster.py --resume

# 4. Full headline run (set LIGHT_MODE = False, REFIT_EVERY_N_DAYS = 63)
caffeinate -i python src/run_replication.py --skip-oxford --skip-intraday --skip-daily

# 5. Order-flow deep-dive comparison (separate output dir)
python src/run_replication.py --deepdive-only --results-dir results_orderflow --fresh

# 6. Replication-plus analytics (reads stored predictions, fast)
python src/run_replication_plus.py

# 7. Simulation study (no data needed)
python src/simulation.py
```
