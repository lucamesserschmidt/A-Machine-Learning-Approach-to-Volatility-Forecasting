"""One-shot, time-aware hyperparameter tuning on a validation set.

The paper tunes hyperparameters on a validation set and (for NNs) fixes them
across the out-of-sample window. We do the same: split the pre-test data into
an inner training part and a validation tail, grid-search each model, pick the
configuration with the lowest validation MSE, and reuse it across the rolling
OOS. This is computationally honest and avoids look-ahead because only
pre-test data is touched.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from config import (RANDOM_STATE, RF_N_ESTIMATORS, GB_N_ESTIMATORS, RIDGE_ALPHAS,
                    LASSO_ALPHAS, ENET_ALPHAS, ENET_L1_RATIOS, RF_GRID, GB_GRID,
                    VALID_FRACTION)


def _split(frame, feature_cols, y_col, valid_fraction):
    frame = frame.sort_values('date').reset_index(drop=True)
    n = len(frame)
    cut = int(n * (1 - valid_fraction))
    tr, va = frame.iloc[:cut], frame.iloc[cut:]
    return (tr[feature_cols].values, tr[y_col].values,
            va[feature_cols].values, va[y_col].values)


def tune_ml_models(pretest_frame, feature_cols, y_col, valid_fraction=VALID_FRACTION) -> dict:
    """Grid-search Ridge/Lasso/EN/RF/GB on a validation tail. Returns best params."""
    from config import ML_LOG_TARGET, MIN_VARIANCE
    Xtr, ytr, Xva, yva = _split(pretest_frame, feature_cols, y_col, valid_fraction)
    if ML_LOG_TARGET:
        # Tune on the same (log) scale the models are actually trained on.
        ytr = np.log(np.clip(ytr, MIN_VARIANCE, None))
        yva = np.log(np.clip(yva, MIN_VARIANCE, None))
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    rs = RANDOM_STATE
    tuned = {}

    best = (np.inf, None)
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a, random_state=rs).fit(Xtr_s, ytr)
        e = mean_squared_error(yva, m.predict(Xva_s))
        if e < best[0]:
            best = (e, a)
    tuned['Ridge'] = {'alpha': best[1]}

    best = (np.inf, None)
    for a in LASSO_ALPHAS:
        m = Lasso(alpha=a, max_iter=50000, random_state=rs).fit(Xtr_s, ytr)
        e = mean_squared_error(yva, m.predict(Xva_s))
        if e < best[0]:
            best = (e, a)
    tuned['Lasso'] = {'alpha': best[1]}

    best = (np.inf, None)
    for a in ENET_ALPHAS:
        for l1 in ENET_L1_RATIOS:
            m = ElasticNet(alpha=a, l1_ratio=l1, max_iter=50000, random_state=rs).fit(Xtr_s, ytr)
            e = mean_squared_error(yva, m.predict(Xva_s))
            if e < best[0]:
                best = (e, (a, l1))
    tuned['ElasticNet'] = {'alpha': best[1][0], 'l1_ratio': best[1][1]}

    # Trees use unscaled features.
    best = (np.inf, None)
    for md in RF_GRID['max_depth']:
        for ml in RF_GRID['min_samples_leaf']:
            m = RandomForestRegressor(n_estimators=RF_N_ESTIMATORS, max_depth=md,
                                      min_samples_leaf=ml, n_jobs=-1, random_state=rs).fit(Xtr, ytr)
            e = mean_squared_error(yva, m.predict(Xva))
            if e < best[0]:
                best = (e, {'max_depth': md, 'min_samples_leaf': ml})
    tuned['RandomForest'] = best[1]

    best = (np.inf, None)
    for lr in GB_GRID['learning_rate']:
        for md in GB_GRID['max_depth']:
            m = GradientBoostingRegressor(n_estimators=GB_N_ESTIMATORS, learning_rate=lr,
                                          max_depth=md, min_samples_leaf=10, random_state=rs).fit(Xtr, ytr)
            e = mean_squared_error(yva, m.predict(Xva))
            if e < best[0]:
                best = (e, {'learning_rate': lr, 'max_depth': md})
    tuned['GradientBoosting'] = best[1]

    tuned['NN'] = {'alpha': 1e-4}     # NN weight-decay kept fixed (paper streamlines NN tuning)
    return tuned
