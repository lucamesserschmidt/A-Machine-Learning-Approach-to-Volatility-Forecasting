# Simulation Study

A controlled simulation study that demonstrates **why** the paper's empirical
results arise, not merely reproduces them. Because we specify the data-generating
process (DGP), we know the truth and can isolate each mechanism.

## Data-Generating Process

Daily log-realized-variance follows a HAR-type recursion with four ingredients:

- **HAR persistence:** `log RV_t = c + 0.36·log RV_{t-1} + 0.33·RV_W + 0.27·RV_M + ...`
  (coefficients sum to 0.96, matching the long-memory persistence of real RV).
- **Nonlinear exogenous driver (X):** a persistent AR(1) process that enters as
  `0.5 · X² · sign(X)` — think of it as a VIX-like variable whose impact on
  volatility is nonlinear and asymmetric. This is the signal that separates M_HAR
  from M_ALL.
- **Heavy-tailed innovations:** drawn from a Student-t distribution with tunable
  degrees of freedom (from Gaussian to very heavy t(3)), producing the crisis
  spikes that challenge level-scale models.
- **Leverage effect:** negative returns amplify next-day volatility (coefficient 0.20).
- **Pure-noise regressors:** four irrelevant AR(1) series, testing whether models
  are fooled by junk predictors.

The sample size is 6,000 days (~24 years) with a 70/30 train-test split, and
results are averaged over multiple random seeds for stability.

## Experiment A — Nonlinearity and Data Richness

**Question:** does ML's edge over HAR appear only with the nonlinear driver in the
information set (the paper's M_HAR → M_ALL mechanism)?

**Design:** fit LogHAR and Random Forest on two feature sets — SIM_HAR (RV lags
only) and SIM_ALL (RV lags + driver X + noise variables) — and compare relative
QLIKE vs LogHAR.

**Result** (`sim_expA_nonlinearity.csv`, `sim_expA.png`): on SIM_HAR, RF ≈ LogHAR
(~1.10 relative QLIKE — slightly worse, because it cannot improve on LogHAR's
correct functional form with only three inputs). On SIM_ALL, RF drops to ~0.91
(9% better), because it captures the nonlinear X²·sign(X) structure that the
linear LogHAR cannot. This cleanly confirms the paper's central thesis under
controlled conditions: **ML gains come from nonlinear predictors, not from a
generic flexibility advantage.**

## Experiment B — Log Transform vs Tail-Heaviness

**Question:** why does LogHAR (log-target) consistently beat level-HAR, and does
the advantage grow with tail-heaviness?

**Design:** vary the innovation distribution from Gaussian (df = ∞) to very heavy
tails (df = 3), fit LogHAR and RF (both on log-RV) vs a level-HAR benchmark, and
track relative QLIKE.

**Result** (`sim_expB_tails.csv`, `sim_expB.png`): under Gaussian innovations the
log advantage is modest. As tails get heavier (df = 10 → 6 → 4 → 3), LogHAR's
advantage grows steadily — from ~0.98 to ~0.80 relative QLIKE. RF follows the
same pattern. This demonstrates that the log transform works by **taming the
influence of crisis spikes on the loss function**, not just by stabilizing the
regression. It also motivates QLIKE (which is robust to the proxy's scale) as the
lead metric over MSE (which is dominated by tail events).

## Experiment C — Long-Memory Persistence at Longer Horizons

**Question:** can RF approximate the long-memory autocorrelation structure of
realized variance better than the three-component HAR at the monthly horizon?

**Design:** fit HAR and RF on h = 1 (daily) and h = 22 (monthly) targets, then
compare the autocorrelation function (ACF) of each model's in-sample fitted values
against the ACF of the true simulated RV.

**Result** (`sim_expC_acf_gap.csv`, `sim_expC.png`): at h = 1 both models track the
true ACF well (mean absolute gap ~0.03 for RF vs ~0.04 for HAR). At h = 22 the gap
widens: **RF tracks the slow decay much more closely** (gap ~0.08 vs ~0.17 for HAR).
This mirrors the paper's Figure 8 finding and explains why the empirical ML advantage
grows with the forecast horizon: the HAR's rigid three-component (daily/weekly/monthly)
structure cannot represent the smooth hyperbolic decay of long memory, while the
tree ensemble adapts its lag weighting flexibly.

## Reproducing

```bash
python src/simulation.py
```

Outputs go to `results_enhanced/` (CSVs + plots). The simulation runs in ~2 minutes
and requires no external data.
