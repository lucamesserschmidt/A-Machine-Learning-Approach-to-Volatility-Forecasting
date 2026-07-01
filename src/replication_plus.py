"""Replication-plus analytics -- the critical-assessment extensions.

Three additions that probe weaknesses in the paper's analysis:

1. Model Confidence Set (Hansen, Lunde & Nason 2011). The paper compares models
   with pairwise Diebold-Mariano tests, which suffer from multiple comparisons.
   The MCS identifies the set of models that are statistically indistinguishable
   from the best at a given confidence level, with the right joint inference.

2. Regime / crisis split. The paper reports a single full-sample ranking. We
   split the out-of-sample period into calm vs turbulent regimes (by VIX tercile)
   and ask whether ML's edge is concentrated in turbulent markets.

3. Friedman's H-statistic. The paper *claims* ML wins through nonlinear
   interactions but only ever shows 1-D ALE. The H-statistic directly quantifies
   interaction strength between predictor pairs for a fitted model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------
def qlike_loss(y, f):
    eps = 1e-12
    y = np.clip(np.asarray(y, float), eps, None)
    f = np.clip(np.asarray(f, float), eps, None)
    r = y / f
    return r - np.log(r) - 1.0


def se_loss(y, f):
    return (np.asarray(y, float) - np.asarray(f, float)) ** 2


# ---------------------------------------------------------------------------
# 1. Model Confidence Set
# ---------------------------------------------------------------------------
def _moving_block_indices(T, B, block, rng):
    """Moving-block bootstrap row indices: (B, T)."""
    n_blocks = int(np.ceil(T / block))
    out = np.empty((B, n_blocks * block), dtype=int)
    starts = rng.integers(0, T - block + 1, size=(B, n_blocks))
    for j in range(block):
        out[:, j::block] = starts + j
    return out[:, :T]


def model_confidence_set(loss, model_names, alpha=0.10, B=1000, block=10, seed=0):
    """MCS via the range statistic with a moving-block bootstrap.

    loss: (T, m) array of per-observation losses, columns aligned to model_names.
    Returns DataFrame: model, mcs_pvalue, in_mcs (pvalue >= alpha).
    """
    loss = np.asarray(loss, float)
    T, m = loss.shape
    rng = np.random.default_rng(seed)
    idx = _moving_block_indices(T, B, block, rng)             # (B, T)
    # bootstrap means of each column, precomputed once (resampling rows jointly)
    boot_means_full = np.stack([loss[idx[b]].mean(0) for b in range(B)])  # (B, m)
    means_full = loss.mean(0)

    alive = list(range(m))
    pvals = {}
    running = 0.0
    while len(alive) > 1:
        a = np.array(alive)
        meanL = means_full[a]                                 # (k,)
        bmeanL = boot_means_full[:, a]                        # (B, k)
        dbar = meanL[:, None] - meanL[None, :]               # (k, k)
        bd = bmeanL[:, :, None] - bmeanL[:, None, :]          # (B, k, k)
        var_d = bd.var(0) + 1e-15
        sd = np.sqrt(var_d)
        t_ij = dbar / sd
        TR = np.max(np.abs(t_ij))
        t_boot = (bd - dbar[None]) / sd[None]
        TR_boot = np.max(np.abs(t_boot.reshape(B, -1)), axis=1)
        pval = float(np.mean(TR_boot >= TR))
        running = max(running, pval)
        if pval < alpha:
            t_i = t_ij.max(axis=1)                            # worst model = largest excess loss
            worst_local = int(np.argmax(t_i))
            pvals[model_names[a[worst_local]]] = running
            alive.remove(a[worst_local])
        else:
            for j in alive:
                pvals[model_names[j]] = max(pvals.get(model_names[j], 0.0), running)
            break
    if len(alive) == 1:
        pvals[model_names[alive[0]]] = 1.0
    rows = [{'model': nm, 'mcs_pvalue': pvals.get(nm, np.nan),
             'in_mcs': pvals.get(nm, 0.0) >= alpha} for nm in model_names]
    return pd.DataFrame(rows).sort_values('mcs_pvalue', ascending=False).reset_index(drop=True)


def mcs_across_cells(preds, loss='qlike', alpha=0.10, B=1000, block=10):
    """Run an MCS per (data_source, asset, feature_set, horizon) cell."""
    lf = qlike_loss if loss == 'qlike' else se_loss
    keys = ['data_source', 'asset', 'feature_set', 'horizon']
    out = []
    for key, g in preds.groupby(keys):
        wide = g.pivot_table(index='date', columns='model', values='prediction')
        yt = g.pivot_table(index='date', columns='model', values='y_true').mean(axis=1)
        wide = wide.dropna()
        yt = yt.loc[wide.index]
        if len(wide) < 50 or wide.shape[1] < 2:
            continue
        L = np.column_stack([lf(yt.values, wide[c].values) for c in wide.columns])
        res = model_confidence_set(L, list(wide.columns), alpha=alpha, B=B, block=block)
        for _, r in res.iterrows():
            out.append({**dict(zip(keys, key)), 'model': r['model'],
                        'mcs_pvalue': r['mcs_pvalue'], 'in_mcs': r['in_mcs']})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 2. Regime / crisis split
# ---------------------------------------------------------------------------
def regime_split(preds, macro, loss='qlike', benchmark='HAR', n_regimes=3):
    """Relative loss vs benchmark within VIX-tercile regimes (calm/normal/turbulent)."""
    lf = qlike_loss if loss == 'qlike' else se_loss
    vix = macro[['date', 'VIXCLS']].dropna().copy()
    vix['date'] = pd.to_datetime(vix['date'])
    p = preds.copy()
    p['date'] = pd.to_datetime(p['date'])
    p = p.merge(vix, on='date', how='left').dropna(subset=['VIXCLS'])
    labels = ['calm', 'normal', 'turbulent'][:n_regimes]
    rows = []
    for (ds, fs, h), g in p.groupby(['data_source', 'feature_set', 'horizon']):
        # regime by VIX tercile over this cell's test dates
        q = pd.qcut(g.drop_duplicates('date')['VIXCLS'], n_regimes, labels=labels)
        regime_map = dict(zip(g.drop_duplicates('date')['date'], q))
        g = g.assign(regime=g['date'].map(regime_map))
        for regime, gr in g.groupby('regime'):
            bench = gr[gr['model'] == benchmark][['date', 'asset', 'prediction']].rename(
                columns={'prediction': 'bp'})
            for model, gm in gr.groupby('model'):
                mm = gm.merge(bench, on=['date', 'asset'], how='inner')
                if len(mm) < 20:
                    continue
                lm = lf(mm['y_true'].values, mm['prediction'].values).mean()
                lb = lf(mm['y_true'].values, mm['bp'].values).mean()
                rows.append({'data_source': ds, 'feature_set': fs, 'horizon': h,
                             'regime': regime, 'model': model,
                             'rel_loss_vs_har': lm / lb if lb > 0 else np.nan, 'n': len(mm)})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    return (res.groupby(['data_source', 'feature_set', 'horizon', 'regime', 'model'])
            .agg(mean_rel_loss=('rel_loss_vs_har', 'mean'), n_assets=('n', 'size'))
            .reset_index())


# ---------------------------------------------------------------------------
# 3. Friedman's H-statistic (pairwise interaction strength)
# ---------------------------------------------------------------------------
def _centered_pd(model, X, cols, grid_rows, bg_rows):
    """Centered partial dependence of `cols` evaluated at grid_rows, averaging
    over background rows bg_rows. Returns vector aligned to grid_rows."""
    Xbg = X.iloc[bg_rows]
    vals = np.empty(len(grid_rows))
    base = X.copy()
    for i, gi in enumerate(grid_rows):
        tmp = Xbg.copy()
        for c in cols:
            tmp[c] = X.iloc[gi][c]
        vals[i] = model.predict(tmp).mean()
    return vals - vals.mean()


def friedman_h_pairwise(model, X, pairs, sample=150, seed=0):
    """Friedman's H-statistic for given feature pairs on a fitted model.

    H2_jk = sum (PDjk - PDj - PDk)^2 / sum PDjk^2, on a subsample (for speed).
    Returns DataFrame: feature_1, feature_2, H (sqrt of H2, in [0,1]).
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    s = min(sample, n)
    grid = rng.choice(n, size=s, replace=False)
    bg = rng.choice(n, size=min(sample, n), replace=False)
    Xr = X.reset_index(drop=True)
    rows = []
    for f1, f2 in pairs:
        if f1 not in Xr.columns or f2 not in Xr.columns:
            continue
        pd1 = _centered_pd(model, Xr, [f1], grid, bg)
        pd2 = _centered_pd(model, Xr, [f2], grid, bg)
        pd12 = _centered_pd(model, Xr, [f1, f2], grid, bg)
        num = np.sum((pd12 - pd1 - pd2) ** 2)
        den = np.sum(pd12 ** 2) + 1e-15
        rows.append({'feature_1': f1, 'feature_2': f2, 'H': float(np.sqrt(max(num / den, 0.0)))})
    return pd.DataFrame(rows).sort_values('H', ascending=False).reset_index(drop=True)
