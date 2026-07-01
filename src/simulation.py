"""Simulation study -- the 'simulation hook'.

We simulate realized variance from a KNOWN data-generating process (a log-HAR
recursion with a leverage term, an exogenous nonlinear driver, heavy-tailed
innovations, and pure-noise regressors). Because we control the truth, we can
demonstrate *why* the paper's empirical results arise, not merely reproduce them:

  Experiment A (nonlinearity / data-richness):
     ML's edge over HAR appears only once we add the nonlinear exogenous driver
     to the information set -- mirroring the empirical MHAR -> MALL jump.

  Experiment B (log transform vs tail-heaviness):
     The advantage of modeling log-RV (LogHAR) over level-RV grows as the
     innovation tails get heavier -- the paper's explanation for LogHAR's
     success (it tames outliers).

  Experiment C (long memory at longer horizons):
     The in-sample fitted-RV autocorrelation of RF tracks the true long-memory
     decay better than HAR at the one-month horizon (paper's Figure 8 story).

Run:  python simulation.py
Outputs: results_enhanced/sim_*.csv and plots in results_enhanced/plots/.
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from config import RESULTS_DIR, PLOTS_DIR, MIN_VARIANCE

RNG_BASE = 20260616


def simulate_log_har(n=6000, burn=300, nonlinear_strength=0.5, df_innov=np.inf,
                     leverage=0.20, noise_dim=4, seed=0):
    """Simulate daily log-RV with HAR persistence + nonlinear exogenous driver.

    Returns a dataframe with RV (levels), log returns, the exogenous driver X
    (informative, nonlinear), and `noise_dim` irrelevant AR(1) regressors.
    """
    rng = np.random.default_rng(RNG_BASE + seed)
    N = n + burn
    # True HAR coefficients on log scale (persistent, summing < 1).
    persist = 0.36 + 0.33 + 0.27
    b1, b5, b22 = 0.36, 0.33, 0.27
    target_mean = -9.0                      # steady-state mean of log-RV
    c = target_mean * (1.0 - persist)       # intercept consistent with that mean

    # Exogenous persistent driver (think VIX-like), affects vol nonlinearly.
    X = np.zeros(N)
    for t in range(1, N):
        X[t] = 0.95 * X[t - 1] + rng.normal(0, 0.3)

    # Heavy-tailed innovations: scaled Student-t (df_innov=inf -> normal).
    if np.isinf(df_innov):
        eta = rng.normal(0, 0.30, N)
    else:
        s = np.sqrt((df_innov - 2) / df_innov)            # unit-variance scaling
        eta = rng.standard_t(df_innov, N) * 0.30 * s

    z = rng.standard_normal(N)               # scale-free return shocks
    # Demeaned drivers so they add variation, not a level shift that the
    # persistence would otherwise amplify into an explosive trend.
    # V-shaped (|X|) effect: non-monotonic, so a model linear in X cannot
    # capture it, but RF/NN can -- this is what isolates the ML edge.
    nl_series = nonlinear_strength * np.abs(X)
    nl_series = nl_series - nl_series.mean()
    lev_series = leverage * np.maximum(0.0, -z)
    lev_series = lev_series - lev_series.mean()

    logrv = np.full(N, target_mean)
    ret = np.zeros(N)
    for t in range(22, N):
        lag1 = logrv[t - 1]
        lag5 = logrv[t - 5:t].mean()
        lag22 = logrv[t - 22:t].mean()
        logrv[t] = (c + b1 * lag1 + b5 * lag5 + b22 * lag22
                    + nl_series[t - 1] + lev_series[t - 1] + eta[t])
        rv_t = np.exp(logrv[t])
        ret[t] = np.sqrt(max(rv_t, MIN_VARIANCE)) * z[t]

    rv = np.exp(logrv)
    noise = np.zeros((N, noise_dim))
    for j in range(noise_dim):
        for t in range(1, N):
            noise[t, j] = 0.9 * noise[t - 1, j] + rng.normal()

    df = pd.DataFrame({'rv': rv, 'ret': ret, 'X': X})
    for j in range(noise_dim):
        df[f'noise{j}'] = noise[:, j]
    df = df.iloc[burn:].reset_index(drop=True)
    df['t'] = np.arange(len(df))
    return df


def _features(df):
    d = df.copy()
    d['RVD'] = d['rv'].shift(1)
    d['RVW'] = d['rv'].rolling(5).mean().shift(1)
    d['RVM'] = d['rv'].rolling(22).mean().shift(1)
    d['logRVD'] = np.log(d['RVD'].clip(lower=MIN_VARIANCE))
    d['logRVW'] = np.log(d['RVW'].clip(lower=MIN_VARIANCE))
    d['logRVM'] = np.log(d['RVM'].clip(lower=MIN_VARIANCE))
    d['X_lag'] = d['X'].shift(1)
    for c in [c for c in d.columns if c.startswith('noise')]:
        d[c + '_lag'] = d[c].shift(1)
    d['y'] = d['rv'].shift(-1)               # one-day-ahead RV target
    return d.dropna().reset_index(drop=True)


def _fit_predict(model_kind, Xtr, ytr, Xte, log_target=False):
    if log_target:
        ytr_ = np.log(np.clip(ytr, MIN_VARIANCE, None))
    else:
        ytr_ = ytr
    if model_kind == 'OLS':
        m = LinearRegression().fit(Xtr, ytr_)
        pred = m.predict(Xte)
        resid_var = np.var(ytr_ - m.predict(Xtr))
    elif model_kind == 'Ridge':
        sc = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr_)
        pred = m.predict(sc.transform(Xte)); resid_var = np.var(ytr_ - m.predict(sc.transform(Xtr)))
    elif model_kind == 'RF':
        m = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                  n_jobs=-1, random_state=0).fit(Xtr, ytr_)
        pred = m.predict(Xte); resid_var = np.var(ytr_ - m.predict(Xtr))
    elif model_kind == 'NN':
        # Log-target seed ensemble: positive, stable forecasts on wide-range variance.
        yl = np.log(np.clip(ytr, MIN_VARIANCE, None))
        ymu, ysd = yl.mean(), yl.std() + 1e-12
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
        te_preds, fit_resids = [], []
        for s in range(3):
            m = MLPRegressor(hidden_layer_sizes=(8, 4, 2), activation='relu', alpha=1e-3,
                             early_stopping=True, max_iter=800, random_state=s).fit(Xtr_s, (yl - ymu) / ysd)
            fit_log = m.predict(Xtr_s) * ysd + ymu
            fit_resids.append(np.var(yl - fit_log))
            te_preds.append(m.predict(Xte_s) * ysd + ymu)
        pred_log = np.mean(te_preds, axis=0)
        lo, hi = yl.min(), yl.max()
        pred_log = np.clip(pred_log, lo, hi)           # guard against NN extrapolation
        pred = np.exp(pred_log + 0.5 * np.mean(fit_resids))
        return np.clip(pred, MIN_VARIANCE, None)
    if log_target:
        pred = np.exp(pred + 0.5 * resid_var)
    return np.clip(pred, MIN_VARIANCE, None)


def _qlike(y, f):
    """Patton (2011) QLIKE loss, robust to the variance proxy's fat tail."""
    y = np.clip(np.asarray(y, float), MIN_VARIANCE, None)
    f = np.clip(np.asarray(f, float), MIN_VARIANCE, None)
    r = y / f
    return float(np.mean(r - np.log(r) - 1.0))


def _relative_loss(d, feature_cols, benchmark='HAR'):
    """Relative QLIKE vs a benchmark (lower than 1 = better than benchmark).

    benchmark='HAR'    -> level OLS-HAR (shows the log-transform advantage).
    benchmark='LogHAR' -> LogHAR on HAR lags (nets out the log effect, so the
                          remaining differences isolate the nonlinearity edge).
    """
    cut = int(len(d) * 0.7)
    tr, te = d.iloc[:cut], d.iloc[cut:]
    Xtr, Xte = tr[feature_cols].values, te[feature_cols].values
    ytr, yte = tr['y'].values, te['y'].values
    har_cols = ['RVD', 'RVW', 'RVM']
    log_har_cols = ['logRVD', 'logRVW', 'logRVM']
    if benchmark == 'LogHAR':
        bench = _fit_predict('OLS', tr[log_har_cols].values, ytr, te[log_har_cols].values, log_target=True)
    else:
        bench = _fit_predict('OLS', tr[har_cols].values, ytr, te[har_cols].values, log_target=False)
    bq = _qlike(yte, bench)
    out = {benchmark: 1.0}
    log_cols = log_har_cols + [c for c in feature_cols if c not in har_cols]
    out['LogHAR'] = _qlike(yte, _fit_predict('OLS', tr[log_cols].values, ytr, te[log_cols].values, log_target=True)) / bq
    out['RF'] = _qlike(yte, _fit_predict('RF', Xtr, ytr, Xte, log_target=True)) / bq
    if benchmark != 'LogHAR':
        out['HAR'] = 1.0
    return out


def experiment_A():
    """ML edge appears only with the nonlinear exogenous driver (MHAR -> MALL)."""
    rows = []
    for seed in range(6):
        d = _features(simulate_log_har(seed=seed, df_innov=8))
        har_cols = ['RVD', 'RVW', 'RVM']
        all_cols = har_cols + ['X_lag'] + [c for c in d.columns if c.startswith('noise') and c.endswith('_lag')]
        for setname, cols in [('SIM_HAR', har_cols), ('SIM_ALL', all_cols)]:
            r = _relative_loss(d, cols, benchmark='LogHAR')
            for model, rel in r.items():
                rows.append({'seed': seed, 'feature_set': setname, 'model': model, 'rel_qlike': rel})
    res = pd.DataFrame(rows)
    summ = res.groupby(['feature_set', 'model'])['rel_qlike'].mean().reset_index()
    return summ.sort_values(['feature_set', 'rel_qlike'])


def experiment_B():
    """LogHAR advantage grows with tail-heaviness (smaller df = heavier tails)."""
    rows = []
    for df_innov in [np.inf, 10, 6, 4, 3]:
        for seed in range(12):
            d = _features(simulate_log_har(seed=seed, df_innov=df_innov))
            all_cols = ['RVD', 'RVW', 'RVM', 'X_lag']
            r = _relative_loss(d, all_cols)
            label = 'normal' if np.isinf(df_innov) else f't({df_innov})'
            rows.append({'innov': label, 'df': (999 if np.isinf(df_innov) else df_innov),
                         'LogHAR_rel_qlike': r['LogHAR'], 'RF_rel_qlike': r['RF']})
    res = pd.DataFrame(rows)
    return res.groupby(['innov', 'df'], as_index=False)[['LogHAR_rel_qlike', 'RF_rel_qlike']].mean().sort_values('df', ascending=False)


def experiment_C():
    """RF tracks long-memory persistence better than HAR at the monthly horizon."""
    d = _features(simulate_log_har(seed=0, df_innov=8, n=6000))
    har_cols = ['RVD', 'RVW', 'RVM']
    all_cols = har_cols + ['X_lag']
    out = {}
    for h, ycol in [(1, 'y'), (22, 'y22')]:
        dd = d.copy()
        if h == 22:
            dd['y22'] = dd['rv'].shift(-1).rolling(22).mean().shift(-(22 - 1))
            dd = dd.dropna().reset_index(drop=True)
        cut = int(len(dd) * 0.7)
        tr = dd.iloc[:cut]
        har = LinearRegression().fit(tr[har_cols], tr[ycol])
        rf = RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=10,
                                   n_jobs=-1, random_state=0).fit(tr[all_cols], tr[ycol])
        fitted_har = har.predict(tr[har_cols])
        fitted_rf = rf.predict(tr[all_cols])
        true = tr[ycol].values
        out[h] = {'true': true, 'HAR': fitted_har, 'RF': fitted_rf}
    return out


def _acf(x, nlags):
    x = np.asarray(x, float); x = x - x.mean()
    denom = np.dot(x, x)
    return np.array([1.0] + [np.dot(x[k:], x[:-k]) / denom for k in range(1, nlags + 1)])


def main():
    RESULTS_DIR.mkdir(exist_ok=True); PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    print('Experiment A: nonlinearity / data-richness (mean relative QLIKE vs LogHAR over seeds)')
    A = experiment_A(); print(A.to_string(index=False)); A.to_csv(RESULTS_DIR / 'sim_expA_nonlinearity.csv', index=False)

    print('\nExperiment B: log advantage vs tail-heaviness (mean relative QLIKE vs HAR)')
    B = experiment_B(); print(B.to_string(index=False)); B.to_csv(RESULTS_DIR / 'sim_expB_tails.csv', index=False)

    print('\nExperiment C: long-memory ACF of fitted RV (HAR vs RF) at h=1 and h=22')
    C = experiment_C()
    c_rows = []
    for h in [1, 22]:
        at = _acf(C[h]['true'], 60); ah = _acf(C[h]['HAR'], 60); ar = _acf(C[h]['RF'], 60)
        c_rows.append({'horizon': h,
                       'HAR_mean_abs_acf_gap': float(np.mean(np.abs(ah[1:] - at[1:]))),
                       'RF_mean_abs_acf_gap': float(np.mean(np.abs(ar[1:] - at[1:])))})
    Cdf = pd.DataFrame(c_rows); print(Cdf.to_string(index=False))
    Cdf.to_csv(RESULTS_DIR / 'sim_expC_acf_gap.csv', index=False)

    # ---- Plots ----
    # A: grouped bars
    fig, ax = plt.subplots(figsize=(9, 5))
    piv = A.pivot(index='model', columns='feature_set', values='rel_qlike')
    piv = piv.reindex(['LogHAR', 'RF'])
    x = np.arange(len(piv)); w = 0.38
    ax.bar(x - w / 2, piv['SIM_HAR'], w, label='SIM_HAR (lags only)', color='#9ab')
    ax.bar(x + w / 2, piv['SIM_ALL'], w, label='SIM_ALL (+ nonlinear driver)', color='#244d90')
    ax.axhline(1.0, ls='--', color='k', lw=1)
    ax.set_xticks(x); ax.set_xticklabels(piv.index); ax.set_ylabel('Relative QLIKE vs LogHAR')
    ax.set_title('Experiment A: only the nonlinear learners gain from the driver'); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS_DIR / 'sim_expA.png', dpi=150); plt.close(fig)

    # B: line vs tail heaviness
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(B['innov'], B['LogHAR_rel_qlike'], 'o-', label='LogHAR', color='#244d90')
    ax.plot(B['innov'], B['RF_rel_qlike'], 's-', label='RF', color='#cc7a00')
    ax.axhline(1.0, ls='--', color='k', lw=1)
    ax.set_xlabel('innovation tails (heavier ->)'); ax.set_ylabel('Relative QLIKE vs HAR')
    ax.set_title('Experiment B: LogHAR advantage grows as tails get heavier'); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS_DIR / 'sim_expB.png', dpi=150); plt.close(fig)

    # C: ACF panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, h in zip(axes, [1, 22]):
        nlags = 120
        ax.plot(_acf(C[h]['true'], nlags), label='true', color='k', lw=1.5)
        ax.plot(_acf(C[h]['HAR'], nlags), label='HAR', color='#cc4444')
        ax.plot(_acf(C[h]['RF'], nlags), label='RF', color='#244d90')
        ax.set_title(f'Fitted-RV ACF, horizon h={h}'); ax.set_xlabel('lag'); ax.set_ylabel('ACF')
        ax.legend()
    fig.suptitle('Experiment C: RF approximates long-memory persistence better at h=22')
    fig.tight_layout(); fig.savefig(PLOTS_DIR / 'sim_expC.png', dpi=150); plt.close(fig)

    print('\nSimulation done. CSVs + sim_expA/B/C.png in', RESULTS_DIR.resolve())


if __name__ == '__main__':
    main()
