"""Accumulated Local Effects (ALE) for 1-D features.

Implements the centered ALE of Apley & Zhu (2020), exactly as described in
the paper (their eqs. 28-31). ALE is preferred over Partial Dependence
because it conditions on the local distribution of the feature, so it is
not distorted by correlation among predictors -- the precise weakness the
paper flags in PD plots, and a real concern here since RVD/RVW/RVM/VIX are
strongly correlated.

We also derive the paper's ALE-based variable-importance measure:
    I(Z_j) = sd over the data of the centered ALE evaluated at observations,
    VI(Z_j) = I(Z_j) / sum_k I(Z_k).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def ale_1d(model, X: pd.DataFrame, feature: str, n_bins: int = 40):
    """Compute the centered 1-D ALE of `feature` for a fitted `model`.

    model.predict must accept a DataFrame with the same columns as X.
    Returns (bin_centers, ale_values).
    """
    X = X.reset_index(drop=True)
    z = X[feature].values
    # Quantile bin edges so each interval holds ~equal mass (paper uses 100).
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(z, qs))
    if len(edges) < 3:
        # Degenerate (near-constant) feature.
        return np.array([z.mean()]), np.array([0.0])
    K = len(edges) - 1

    # Assign each observation to an interval (1..K).
    idx = np.clip(np.searchsorted(edges, z, side='left'), 1, K)

    local_effects = np.zeros(K)
    counts = np.zeros(K)
    for k in range(1, K + 1):
        mask = idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        lo, hi = edges[k - 1], edges[k]
        X_lo = X.loc[mask].copy()
        X_hi = X.loc[mask].copy()
        X_lo[feature] = lo
        X_hi[feature] = hi
        diff = model.predict(X_hi) - model.predict(X_lo)
        local_effects[k - 1] = np.mean(diff)
        counts[k - 1] = nk

    # Accumulate, then center so the data-weighted mean is zero.
    accumulated = np.concatenate([[0.0], np.cumsum(local_effects)])
    centers = edges
    # Map accumulated effect to each observation to compute the centering const.
    acc_at_obs = accumulated[idx]
    const = np.average(acc_at_obs)
    ale_centered = accumulated - const
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    # Return ALE at bin upper edges interpolated to bin centers.
    ale_at_centers = 0.5 * (ale_centered[:-1] + ale_centered[1:])
    return bin_centers, ale_at_centers


def ale_importance(model, X: pd.DataFrame, features: list[str], n_bins: int = 40) -> pd.DataFrame:
    """Paper's ALE-based variable importance: sd of centered ALE, normalized."""
    X = X.reset_index(drop=True)
    imps = {}
    for f in features:
        z = X[f].values
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(z, qs))
        if len(edges) < 3:
            imps[f] = 0.0
            continue
        K = len(edges) - 1
        idx = np.clip(np.searchsorted(edges, z, side='left'), 1, K)
        local_effects = np.zeros(K)
        for k in range(1, K + 1):
            mask = idx == k
            if not mask.any():
                continue
            lo, hi = edges[k - 1], edges[k]
            X_lo = X.loc[mask].copy(); X_lo[f] = lo
            X_hi = X.loc[mask].copy(); X_hi[f] = hi
            local_effects[k - 1] = np.mean(model.predict(X_hi) - model.predict(X_lo))
        accumulated = np.concatenate([[0.0], np.cumsum(local_effects)])
        acc_at_obs = accumulated[idx]
        ale_obs = acc_at_obs - acc_at_obs.mean()
        imps[f] = float(np.sqrt(np.mean(ale_obs ** 2)))
    total = sum(imps.values())
    rows = [{'feature': f, 'ale_importance': v,
             'ale_vi': (v / total if total > 0 else np.nan)} for f, v in imps.items()]
    return pd.DataFrame(rows).sort_values('ale_vi', ascending=False).reset_index(drop=True)


def ale_2d(model, X: pd.DataFrame, f1: str, f2: str, n_bins: int = 20):
    """Second-order (interaction) ALE of the pair (f1, f2), Apley & Zhu (2020).

    Returns (edges1, edges2, ALE_2d) where ALE_2d[i, j] is the centered pure
    interaction effect on the (f1, f2) grid -- the second-order effect with the
    two main effects removed. A flat surface (~0) means no interaction; structure
    means the model's response to f1 depends on f2. This is the 2-D analogue of
    the paper's 1-D ALE and directly visualizes the interactions ML is claimed
    to exploit.
    """
    X = X.reset_index(drop=True)
    z1, z2 = X[f1].values, X[f2].values
    e1 = np.unique(np.quantile(z1, np.linspace(0, 1, n_bins + 1)))
    e2 = np.unique(np.quantile(z2, np.linspace(0, 1, n_bins + 1)))
    if len(e1) < 3 or len(e2) < 3:
        return e1, e2, np.zeros((max(len(e1) - 1, 1), max(len(e2) - 1, 1)))
    K1, K2 = len(e1) - 1, len(e2) - 1
    i1 = np.clip(np.searchsorted(e1, z1, side='left'), 1, K1)
    i2 = np.clip(np.searchsorted(e2, z2, side='left'), 1, K2)

    # Second-order local differences in each cell: f(hi,hi)-f(hi,lo)-f(lo,hi)+f(lo,lo)
    D = np.zeros((K1, K2))
    cnt = np.zeros((K1, K2))
    for a in range(1, K1 + 1):
        for b in range(1, K2 + 1):
            mask = (i1 == a) & (i2 == b)
            nk = int(mask.sum())
            if nk == 0:
                continue
            Xc = X.loc[mask]
            lo1, hi1, lo2, hi2 = e1[a - 1], e1[a], e2[b - 1], e2[b]
            def pred(v1, v2):
                t = Xc.copy(); t[f1] = v1; t[f2] = v2
                return model.predict(t)
            D[a - 1, b - 1] = np.mean(pred(hi1, hi2) - pred(hi1, lo2) - pred(lo1, hi2) + pred(lo1, lo2))
            cnt[a - 1, b - 1] = nk

    # Accumulate in both directions, then remove main effects (center per row/col).
    acc = np.cumsum(np.cumsum(D, axis=0), axis=1)
    acc = np.pad(acc, ((1, 0), (1, 0)))
    w1 = np.concatenate([[0], cnt.sum(1)]); w1 = w1 / (w1.sum() + 1e-12)
    w2 = np.concatenate([[0], cnt.sum(0)]); w2 = w2 / (w2.sum() + 1e-12)
    acc = acc - (acc * w1[:, None]).sum(0)[None, :]
    acc = acc - (acc * w2[None, :]).sum(1)[:, None]
    return e1, e2, acc
