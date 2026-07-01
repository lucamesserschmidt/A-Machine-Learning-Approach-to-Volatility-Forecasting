"""Econometric tests used in the replication.

Diebold-Mariano with a HAC (Newey-West) long-run variance and the
Harvey-Leybourne-Newbold (1997) small-sample correction. This is the
*correct* version for multi-step (h>1) forecasts, where the loss
differential is serially correlated by construction and a plain i.i.d.
standard error is mis-sized.

Also: Kupiec (1995) unconditional-coverage LR test and Christoffersen
(1998) conditional-coverage / independence LR tests for VaR backtesting.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm, t as student_t, chi2


def _newey_west_lrv(d: np.ndarray, lag: int) -> float:
    """Newey-West HAC estimate of the long-run variance of series d."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / n
    lrv = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - k / (lag + 1.0)            # Bartlett kernel
        gamma_k = np.dot(dc[k:], dc[:-k]) / n
        lrv += 2.0 * w * gamma_k
    return lrv


def diebold_mariano(y_true, pred_benchmark, pred_model, horizon: int = 1,
                    power: int = 2, alternative: str = 'model_better', loss: str = 'mse'):
    """One-sided Diebold-Mariano test with HAC variance + HLN correction.

    loss='mse'   : squared-error loss |error|**power.
    loss='qlike' : Patton (2011) QLIKE = y/f - log(y/f) - 1 (robust to the
                   variance proxy's heavy tail; standard in the RV literature).
    Positive mean loss differential d = L_bench - L_model  =>  model is better.

    alternative='model_better' returns p-value for H1: model beats benchmark.
    Returns (dm_stat, p_value).
    """
    y_true = np.asarray(y_true, dtype=float)
    if loss == 'qlike':
        eps = 1e-12
        f_b = np.clip(np.asarray(pred_benchmark, float), eps, None)
        f_m = np.clip(np.asarray(pred_model, float), eps, None)
        yv = np.clip(y_true, eps, None)
        rb = yv / f_b; rm = yv / f_m
        d = (rb - np.log(rb) - 1.0) - (rm - np.log(rm) - 1.0)
    else:
        eb = y_true - np.asarray(pred_benchmark, dtype=float)
        em = y_true - np.asarray(pred_model, dtype=float)
        d = np.abs(eb) ** power - np.abs(em) ** power
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return np.nan, np.nan

    lag = max(0, horizon - 1)
    lrv = _newey_west_lrv(d, lag)
    if lrv <= 0:
        return np.nan, np.nan

    dbar = d.mean()
    dm = dbar / np.sqrt(lrv / n)

    # Harvey, Leybourne & Newbold (1997) small-sample correction.
    h = horizon
    corr = np.sqrt(max((n + 1 - 2 * h + h * (h - 1) / n) / n, 1e-12))
    dm_hln = dm * corr

    # Reference: Student-t with n-1 dof (HLN recommendation).
    dof = n - 1
    if alternative == 'model_better':
        p = 1.0 - student_t.cdf(dm_hln, df=dof)
    elif alternative == 'benchmark_better':
        p = student_t.cdf(dm_hln, df=dof)
    else:  # two-sided
        p = 2.0 * (1.0 - student_t.cdf(abs(dm_hln), df=dof))
    return float(dm_hln), float(p)


def kupiec_pof(violations: np.ndarray, alpha: float):
    """Kupiec (1995) proportion-of-failures unconditional-coverage LR test.

    H0: violation probability == alpha. Returns (LR_stat, p_value).
    """
    x = np.asarray(violations, dtype=int)
    n = len(x)
    k = int(x.sum())
    if n == 0:
        return np.nan, np.nan
    pi_hat = k / n
    eps = 1e-12
    ll_null = k * np.log(alpha + eps) + (n - k) * np.log(1 - alpha + eps)
    ll_alt = k * np.log(pi_hat + eps) + (n - k) * np.log(1 - pi_hat + eps)
    lr = -2.0 * (ll_null - ll_alt)
    p = 1.0 - chi2.cdf(lr, df=1)
    return float(lr), float(p)


def christoffersen_independence(violations: np.ndarray):
    """Christoffersen (1998) independence LR test (Markov, against clustering).

    Returns (LR_stat, p_value).
    """
    x = np.asarray(violations, dtype=int)
    if len(x) < 2:
        return np.nan, np.nan
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(x[:-1], x[1:]):
        if a == 0 and b == 0:
            n00 += 1
        elif a == 0 and b == 1:
            n01 += 1
        elif a == 1 and b == 0:
            n10 += 1
        else:
            n11 += 1
    eps = 1e-12
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_null = (n00 + n10) * np.log(1 - pi + eps) + (n01 + n11) * np.log(pi + eps)
    ll_alt = (n00 * np.log(1 - pi01 + eps) + n01 * np.log(pi01 + eps)
              + n10 * np.log(1 - pi11 + eps) + n11 * np.log(pi11 + eps))
    lr = -2.0 * (ll_null - ll_alt)
    p = 1.0 - chi2.cdf(lr, df=1)
    return float(lr), float(p)


def christoffersen_cc(violations: np.ndarray, alpha: float):
    """Christoffersen conditional-coverage LR test = Kupiec + independence.

    Returns (LR_stat, p_value) with 2 dof.
    """
    lr_uc, _ = kupiec_pof(violations, alpha)
    lr_ind, _ = christoffersen_independence(violations)
    if np.isnan(lr_uc) or np.isnan(lr_ind):
        return np.nan, np.nan
    lr_cc = lr_uc + lr_ind
    p = 1.0 - chi2.cdf(lr_cc, df=2)
    return float(lr_cc), float(p)
