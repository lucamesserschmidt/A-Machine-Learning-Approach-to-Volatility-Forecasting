# Light vs full runs — how to configure

Everything is controlled from `config.py`. There is a single switch at the
bottom of that file:

```python
LIGHT_MODE = True     # fast smoke run
LIGHT_MODE = False    # full fidelity
```

`LIGHT_MODE` changes **only run volume, never methodology** — same models, same
tests, same metrics. Use it to confirm the pipeline runs and produces sensible
numbers in a few minutes, then flip it off for the real run.

## What LIGHT_MODE = True does

| Knob | Light | Full (LIGHT_MODE = False) |
|------|-------|---------------------------|
| NN architectures | NN2 only | NN1, NN2, NN3 |
| NN seeds / ensemble | 3 / best 2 | 8 / best 4 (raise toward 100 / 10 for the paper) |
| Refit cadence | every 63 days | every 21 days (monthly) |
| Horizons | h = 1 | h = 1, 5, 22 |
| Tuning grids | coarse | full grids |
| Trees | 150 estimators | 300 (RF) / 400 (GB) |
| Assets | `OMI_SPX` + `SPY_DAILY_GK` | all OMI symbols + intraday + 12 daily |

## Recommended workflow

1. **Smoke test (light).** Leave `LIGHT_MODE = True` and run one asset:
   ```bash
   python run_replication.py --only OMI_SPX --skip-daily --skip-intraday
   ```
   A couple of minutes. Confirms loaders, models, tests, and outputs all work.

2. **Overnight headline run (recommended for a first real result).**
   Set `LIGHT_MODE = False`, then apply this overnight preset in `config.py`
   (quarterly refit + a slightly smaller NN ensemble -- same methodology, ~1/4
   the compute of the paper-faithful defaults):
   ```python
   LIGHT_MODE = False
   REFIT_EVERY_N_DAYS = 63                       # quarterly, not monthly
   NN_ARCHITECTURES = {'NN2': (4, 2), 'NN3': (8, 4, 2)}
   NN_N_SEEDS = 6
   NN_ENSEMBLE_TOP = 3
   ```
   Then launch with sleep disabled (macOS):
   ```bash
   caffeinate -i python run_replication.py
   ```
   With the default panel (8 OMI indices + 4 daily-GK names + intraday SPY) this
   is roughly an overnight job. To prioritise the headline, run OMI first:
   ```bash
   caffeinate -i python run_replication.py --skip-daily --skip-intraday   # ~3-4 h
   ```

3. **Full fidelity / full panel (later, multi-night).** Set
   `NN_N_SEEDS = 8`, `REFIT_EVERY_N_DAYS = 21`, swap `DAILY_ASSET_FILES =
   DAILY_ASSET_FILES_ALL` (all 12), widen `OXFORD_MAN_SYMBOLS`, and run the
   three data sources on separate nights. Thanks to checkpointing (below) they
   accumulate into one set of CSVs.

## Crash-safe checkpointing and resume (important for overnight)

Results are now written **incrementally, one cell at a time**, to the
accumulating CSVs (`model_comparison_all.csv`, `all_test_predictions.csv`,
`var_backtest_h1.csv`, `ale_importance_mall_h1.csv`). This means:

* **A crash or a sleeping Mac is not fatal.** Whatever finished is saved.
* **Re-running the same command resumes** -- completed `(asset, feature_set,
  horizon)` cells are detected and skipped, so you just relaunch to continue.
* **Multiple runs accumulate** rather than overwrite, so you can do OMI one
  night and the daily panel another; `cross_sectional_relative_mse.csv` and
  `best_model_by_cell.csv` are recomputed from the union on every run.
* Use `--fresh` to wipe the accumulating CSVs and start clean.

So even if the overnight run does not fully finish by morning, you will have
complete results for every asset that did finish, and can resume the rest later.

## Daily panel size

`DAILY_ASSET_FILES` defaults to 4 names (SPY, AAPL, JPM, XOM -- market + tech +
financials + energy), which is plenty to show the RV5 results are robust across
assets. `DAILY_ASSET_FILES_ALL` holds the full 12-name panel for later.


## Other useful switches (in `config.py`)

* `OXFORD_MAN_SYMBOLS` — the cross-section of indices. Light mode forces just
  `['.SPX']`; in full mode it is the 8-index default. Add any OMI ticker
  (`.HSI`, `.STOXX50E`, …) to widen the panel.
* `HORIZONS` — `[1, 5, 22]` in full mode. Drop to `[1]` for speed.
* `REFIT_EVERY_N_DAYS` — raise (e.g. 126) to refit less often and run faster.
* `NN_N_SEEDS`, `NN_ENSEMBLE_TOP` — push toward 100 / 10 for paper-faithful NNs.
* `ML_LOG_TARGET` — keep `True` (ML models train on log-RV; see note below).
* `VAR_METHODS` — `('gaussian', 'fhs')`; both are reported.

## Quick timing intuition

Runtime ≈ (#assets) × (#feature sets) × (#horizons) × (#refit blocks) ×
(cost of fitting all models, dominated by the NN ensemble and the tuning pass).
The two biggest levers are `NN_N_SEEDS` and `REFIT_EVERY_N_DAYS`; the daily GK
series is the slowest single asset because its test window is the longest
(2019→present).

## Note on the log-target / QLIKE change

Realized variance has a heavy right tail (crisis spikes). On raw levels, OLS-HAR
is nearly unbeatable on MSE and the NN is numerically unstable. The package now
(a) trains the ML models on **log-RV** (`ML_LOG_TARGET = True`, inverted with a
Jensen correction), (b) applies an **insanity filter** (floors/caps forecasts to
the training range) so a stray non-positive forecast cannot blow up, and
(c) reports the robust **QLIKE** loss alongside MSE. Lead with QLIKE in the
write-up; it is the standard loss in the realized-volatility literature.
