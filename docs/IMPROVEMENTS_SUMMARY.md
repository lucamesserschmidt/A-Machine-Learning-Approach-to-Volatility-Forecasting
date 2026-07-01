# What changed and why — upgrade summary

This maps every flaw from the original audit and every request you made to the
concrete change, with the evidence that it works. Use it as a checklist when
writing the seminar paper.

## Original audit → fix

| # | Audit finding (original project) | Fix in this version | Evidence |
|---|----------------------------------|---------------------|----------|
| 1 | Neural network **commented out** — omitted the paper's headline model | Re-enabled as a **seed-ensemble** of geometric-pyramid MLPs (NN1–NN3), `models.EnsembleNN` | Fires in the SPY validation run; configurable seeds/architectures |
| 2 | Lasso/EN `alpha=1e-5` **collapsed to OLS**; validation set defined but never used | **One-shot, time-aware tuning** on a validation tail (`tuning.tune_ml_models`), reused across OOS | Ridge tunes to **α=100** on SPY (real shrinkage), not the degenerate 1e-5 |
| 3 | No log target / **LogHAR** | LogHAR added with **Jensen bias correction** `exp(f + ½·var_resid)`; log lags built in `data_utils` | LogHAR runs and beats HAR at h=1 on SPY (rel-MSE 0.95) |
| 4 | DM test used **i.i.d. SE** — invalid for h=5,22 | **HAC (Newey–West) DM + Harvey–Leybourne–Newbold** correction, Student-t reference (`stats_tests.diebold_mariano`) | Produces sensible one-sided p-values (RF beats HAR p≈0.04) |
| 5 | Main target = **Garman–Klass**, not RV; intraday path gave only ~86 test days | **Intraday RV5 is the headline**; earlier intraday split (test from 2018) for a real OOS window; GK kept as robustness | Intraday split configured; GK demoted to robustness layer |
| 6 | Used **PDP + permutation** importance (PDP fails under correlation) | **ALE** curves + ALE variable importance (Apley & Zhu 2020), `ale.py` | Computes on RF/NN; robust to RV/VIX correlation |

## Requests → delivery

| Request | Delivered |
|---------|-----------|
| Re-enable the NN | `EnsembleNN`, NN1–NN3, multi-seed best-of averaging |
| Add tuning | `tuning.py`, validation-tail grid search, no look-ahead |
| Fix the DM test | HAC + HLN one-sided test in `stats_tests.py` |
| Add log-RV target / LogHAR | LogHAR with Jensen correction; log lags in features |
| Make intraday RV5 the headline | `config` headline = `SPY_INTRADAY_RV5`; GK = robustness |
| Add an implementation [ALE] | `ale.py` (curves + importance), plotted in `plotting.py` |
| Cross-sectional relative-MSE table | `evaluation.cross_sectional_summary` + heatmap |
| "Everything you'd optimise" | LevHAR / SHAR / HARQ; semivariance + realized quarticity from intraday; Christoffersen (ind + CC) added to VaR; tidy prediction export; argparse driver |
| Free-data recommendation | `DATA_RECOMMENDATIONS.md` — adopt the archived **Oxford-Man Realized Library** (ready-made RV5 for ~30 indices, loadable in R via `bvhar`) |
| Simulation hook | `simulation.py` — three controlled experiments (below) |

## Additional HAR-family models (beyond LogHAR)

* **LevHAR** — HAR + aggregated negative returns `r⁻_{t-1|t-h}` (leverage).
  Available on all assets (uses daily signed returns).
* **SHAR** — semivariance HAR using positive/negative 5-minute realized
  semivariance. Intraday only (needs signed high-frequency returns).
* **HARQ** — HAR with the `√RQ · RVD` interaction (realized quarticity correction
  for attenuation bias). Intraday only.

On the SPY daily validation (GK), **LevHAR** was among the strongest HAR
variants (rel-MSE 0.81 at h=1), consistent with a real leverage effect.

## Validation evidence (SPY daily, h=1, MALL, smoke configuration)

All models fire; tuning produces genuine regularization; HAC-DM and the VaR
tests run. Representative numbers (reduced NN/refit for speed):

```
model            relative_mse_vs_har   dm_pvalue_model_beats_har
RandomForest             0.768                  0.038
LevHAR                   0.813                  0.035
Ridge (α=100)            0.872                  0.021
Lasso                    0.876                  0.012
GradientBoosting         0.893                  0.295
ElasticNet               0.938                  0.045
LogHAR                   0.953                  0.016
HAR  (benchmark)         1.000                   —
```

Random Forest wins at h=1, matching the paper's data-rich finding. (NN is
undertrained in this *smoke* configuration; raise `NN_N_SEEDS` and lower
`REFIT_EVERY_N_DAYS` for the real run.)

> VaR note: Gaussian VaR over-rejects here (violation rates 6–19% vs 5%) because
> returns are fat-tailed and the variance target is a proxy. This is expected
> and worth stating; filtered historical simulation is the proper fix.

## Simulation study results (QLIKE loss)

* **A (nonlinearity):** RF rel-QLIKE vs LogHAR = **1.10** with HAR lags only →
  **0.91** once the nonlinear driver is added; linear LogHAR+driver stays ≈1.0.
  ML's edge is *caused by* exploitable nonlinearity, not by flexibility per se.
* **B (tails):** LogHAR's advantage over level-HAR is large throughout and
  **largest at the heaviest tail** (t with 3 dof). Log-modeling tames outliers.
* **C (long memory):** mean |ACF − truth| — HAR **0.012 at h=1** but **0.166 at
  h=22**; RF **0.077 at h=22**. RF captures long-memory persistence better at the
  monthly horizon (paper's Fig 8).

## Suggested narrative for the paper

1. Headline: RV5 forecasting, MHAR vs MALL, HAR-family vs ML, valid HAC-DM
   inference, cross-sectional table.
2. Mechanism: the simulation shows the empirical pattern is *structural* — ML
   helps when (and because) there is exploitable nonlinearity in a richer
   information set; LogHAR helps because RV is right-skewed/heavy-tailed; trees
   approximate long memory better at long horizons.
3. Interpretation: ALE importances on MALL (which predictors the ML models use).
4. Risk: VaR backtest, with the fat-tail caveat and FHS as the stated extension.
5. Data ambition: self-built RV5 from intraday data *and* a path to the
   Oxford-Man library for a full cross-section.

---

# Round 2 — Oxford-Man data, FHS VaR, robust evaluation

Follow-up changes after wiring in the real Oxford-Man Realized Library, which
surfaced (and fixed) a methodological issue.

| Change | What / why | Files |
|--------|-----------|-------|
| **Oxford-Man loader** | `load_oxford_man_symbol` turns the long-format OMI CSV into the daily-RV frame the pipeline already understands. RV5 headline, `rsv` → SHAR; HARQ auto-skips (no realized quarticity in OMI). | `load_oxford_man.py` |
| **OMI wired into the driver** | A cross-section of indices (`OXFORD_MAN_SYMBOLS`, default 8) runs as the headline `oxford_rv5` source, with an OMI feature set (`FEATURE_SETS_OMI`, no volume) and its own 2016 OOS split. | `config.py`, `run_replication.py` |
| **FHS VaR added** | Filtered historical simulation: standardize returns by the forecast vol, then use the **expanding empirical quantile** of past standardized residuals (no look-ahead). Reported alongside Gaussian. | `evaluation.py`, `config.py` |
| **ML on log-RV** | RV5 is heavy-tailed; on raw levels OLS-HAR is unbeatable on MSE and the NN exploded (rel-MSE ~1200). ML models + NN now train on **log-RV** with a Jensen correction (`ML_LOG_TARGET=True`). | `models.py`, `tuning.py`, `config.py` |
| **Insanity filter** | Forecasts are floored/capped to the training range, so a stray non-positive forecast (e.g. level-OLS LevHAR) cannot blow up QLIKE. | `models.py` |
| **QLIKE reporting** | Robust QLIKE loss (Patton 2011) reported next to MSE, with a QLIKE-based Diebold–Mariano test and cross-sectional shares. QLIKE is the lead metric. | `stats_tests.py`, `evaluation.py` |
| **LIGHT_MODE switch** | One flag in `config.py` shrinks NN/refit/grids/panel for a fast smoke run without touching methodology. | `config.py`, `CONFIG_LIGHT_VS_FULL.md` |

## OMI .SPX result after the fixes (h=1, QLIKE = lead metric)

```
feature_set  model             rel_QLIKE   rel_MSE   beats_HAR@5%(QLIKE)
MALL         RandomForest        0.775      0.927          yes
MALL         GradientBoosting    0.781      0.955          yes
MALL         LogHAR              0.783      0.962          yes
MALL         HAR (benchmark)     1.000      1.000           —
...          Ridge/Lasso/EN      ~1.12      ~1.20          no  (linear models
                                                                 on level features
                                                                 gain little from a
                                                                 log target)
```

Nonlinear ML (RF, GB) beats HAR significantly on QLIKE, the gain grows from
MHAR→MALL, and LogHAR is the strongest linear HAR — the paper's pattern.

## VaR calibration on OMI RV5 (now sensible)

Both methods are near the 5% target (e.g. RF Gaussian 5.9%, FHS 5.9%; NN2 4.8%),
with Kupiec/Christoffersen mostly non-rejecting. The earlier 6–19% over-rejection
was specific to the Garman–Klass proxy, not true realized variance.
