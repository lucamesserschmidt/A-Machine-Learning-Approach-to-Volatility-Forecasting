# A Machine Learning Approach to Volatility Forecasting — Replication & Extension

A from-scratch replication and critical extension of Christensen, Siggaard & Veliyev (2023),
*"A Machine Learning Approach to Volatility Forecasting,"* Journal of Financial Econometrics,
for a seminar in Financial Risk Modelling (University of Göttingen).

**Skills demonstrated:** time-series forecasting, machine learning (tree ensembles, neural
networks, regularized linear models), statistical hypothesis testing, backtesting and risk
model validation, interpretable ML, reproducible research.

The paper's thesis: machine learning (tree ensembles, neural networks) beats the HAR benchmark
for realized-variance forecasting, but **only** when given a rich predictor set (M_ALL vs M_HAR),
because ML exploits nonlinearities and interactions the linear HAR model cannot.

## What this project does

1. **Faithful replication** on genuine intraday equity data: real 5-minute realized variance
   (RV5) for 54 individual US stocks (2009–2019), matching the paper's target and asset class —
   not a daily-volatility proxy.
2. **Microstructure extension:** order-flow predictors (OFI, effective spread, signed trade
   imbalance), reconstructed from raw LOBSTER-style limit-order messages via top-of-book
   reconstruction for 7 stocks, tested as additions to the feature set.
3. **Simulation study** (3 controlled experiments) demonstrating *why* the paper's results
   arise — nonlinearity, tail-heaviness, and long-memory persistence — independent of any
   real-data results. See [`docs/simulation_study.md`](docs/simulation_study.md).
4. **A critical assessment ("replication-plus")** addressing three specific weaknesses in the
   paper's methodology:
   - **Multiple-testing in model comparison** → Model Confidence Set (Hansen, Lunde & Nason 2011)
     instead of pairwise Diebold–Mariano tests.
   - **A single full-sample ranking** → out-of-sample performance split by volatility regime
     (calm / normal / turbulent via VIX terciles).
   - **An unmeasured interaction claim** → Friedman's H-statistic and 2-D Accumulated Local
     Effects, directly quantifying the interactions the paper only asserts.

## Key findings

- **The paper's qualitative result replicates.** LogHAR and the ML models (RF, GB, NNs) beat HAR
  on QLIKE, most clearly at the 5-day horizon. Under a Model Confidence Set, ML models are among
  the statistically-best set on ~90% of stocks vs ~70% for HAR.
- **The paper's central *mechanism* does not fully replicate on individual stocks.** Enriching
  M_HAR to M_ALL helps the penalized-linear models but hurts the tree/NN models at longer
  horizons, and the strongest single model (LogHAR) uses none of the extra predictors.
- **Microstructure order-flow adds no forecasting value** — consistent with near-zero measured
  interaction strength (Friedman H ≈ 0.06 for order-flow pairs vs ≈ 0.18 for RV × VIX).
- **The ML edge is regime- and horizon-dependent.** It concentrates in calm/normal volatility
  regimes and at the weekly horizon, and does not clearly survive into turbulent periods.
- **Interactions exist but are narrower than claimed.** The strongest interactions are among
  the HAR-type variables themselves (RVD × VIX, H ≈ 0.30), not the broader M_ALL set — which
  explains why adding more predictors does not uniformly help the nonlinear models.

## Repository structure

```
src/                        16 modules: loaders, models, evaluation, replication-plus analytics
results/
  tables/
    cross_sectional_headline.csv   54-stock headline (rel MSE + QLIKE, 3 horizons, 13 models)
    orderflow_comparison.csv       7-stock MALL vs MALL+OF comparison
    mcs_share_summary.csv          Model Confidence Set: share of stocks each model is in the MCS
    regime_relative_loss.csv       relative loss vs HAR within VIX-tercile regimes
    friedman_h.csv                 pairwise interaction strength (Friedman H, 7 deep-dive stocks)
    ale_importance_mall_h1.csv     ALE variable importance (1-D)
    var_backtest_h1.csv            VaR violations (Gaussian + FHS, Kupiec + Christoffersen tests)
    best_model_by_cell.csv         winning model per (asset, feature_set, horizon) cell
    sim_exp{A,B,C}_*.csv           simulation study results
  plots/
    heatmap_stock_rv5_{MALL,MHAR}.png   cross-sectional heatmaps (54 stocks × 3 horizons)
    relmse_AAPL_STOCK_RV5_*.png         AAPL relative loss bar charts (h = 1, 5, 22)
    ale_*_AAPL_STOCK_RV5_*.png          ALE panel + variable importance for AAPL RF
    sim_exp{A,B,C}.png                  simulation study figures
docs/
    simulation_study.md             simulation study write-up (3 experiments, DGP, interpretation)
    DATA_RECOMMENDATIONS.md         data-sourcing rationale and alternatives
    CONFIG_LIGHT_VS_FULL.md         light vs full-fidelity configuration guide
    IMPROVEMENTS_SUMMARY.md         methodology notes and upgrade log
data/README.md                     data sources + expected layout (data not included)
requirements.txt
```

## Data

Not included — see [`data/README.md`](data/README.md). The headline stock and order-message data
are proprietary (supervisor-provided / LOBSTER-licensed); the Oxford-Man and FRED data are public
but fetched separately. The code runs against any data placed in the layout described there.

## Method summary

- **Target:** realized variance (RV5) from 5-minute returns; forecast horizons 1, 5, and 22 trading days.
- **HAR benchmarks:** HAR, LogHAR, SHAR, HARQ, HAR-CJ (jump/continuous decomposition via bipower variation).
- **ML models:** Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting, seed-ensembled
  MLPs (geometric-pyramid NN1–NN3).
- **Training:** log-RV target with Jensen-correction back-transform; expanding-window
  out-of-sample evaluation with periodic refitting; level-model forecasts winsorized to the
  training support to prevent pathological extrapolation.
- **Evaluation:** relative MSE and QLIKE vs HAR, HAC Diebold–Mariano with HLN correction,
  Model Confidence Sets (Hansen–Lunde–Nason), cross-sectional summaries, VaR backtests
  (Gaussian + filtered-historical-simulation with Kupiec and Christoffersen tests).
- **Interpretation:** 1-D and 2-D Accumulated Local Effects (Apley & Zhu 2020) and Friedman's
  H-statistic for interaction quantification.
- **Simulation:** 3 controlled experiments validating the log-target, QLIKE, and long-memory
  design choices. See [`docs/simulation_study.md`](docs/simulation_study.md).

## Usage

```bash
pip install -r requirements.txt

# 1. Headline: 54-stock RV5 cross-section
python src/run_replication.py --skip-oxford --skip-intraday --skip-daily

# 2. Order-flow preprocessing (one-time, resumable; large files need Parquet export first)
python src/preprocess_lobster.py --resume

# 3. Order-flow deep-dive comparison (MALL vs MALL+OF, 7 stocks)
python src/run_replication.py --deepdive-only --results-dir results_orderflow --fresh

# 4. Replication-plus analytics (MCS, regime split, Friedman H / 2-D ALE)
python src/run_replication_plus.py

# 5. Simulation study (no data needed, ~2 min)
python src/simulation.py
```

See `src/config.py` for the full set of options (feature sets, horizons, model grids, NN seeds,
refit cadence, light vs full-fidelity mode).

## Acknowledgements

Seminar project for Financial Risk Modelling, University of Göttingen, supervised by
Prof. J. Hambuckers. Replicates and extends Christensen, K., Siggaard, M., & Veliyev, B. (2023),
*A Machine Learning Approach to Volatility Forecasting*, Journal of Financial Econometrics.
