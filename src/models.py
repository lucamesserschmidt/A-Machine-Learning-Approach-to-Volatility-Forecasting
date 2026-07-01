"""Forecaster definitions: a unified interface over HAR-family and ML models.

Every forecaster exposes:
    .name
    .fit(train_df)          -> fits on a dataframe (selects its own columns)
    .predict(test_df)       -> predictions on the RV (levels) scale

This lets the rolling evaluator treat OLS-HAR, LogHAR, LevHAR, SHAR, HARQ,
Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting and the NN ensemble
identically, while each handles its own feature columns and target transform.

Log-target models (LogHAR) predict log-RV and invert with the Jensen
bias-correction used in the paper (footnote 11):
    E[exp(f)] = exp(f + 0.5 * var_resid).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from config import (RANDOM_STATE, MIN_VARIANCE, RF_N_ESTIMATORS, GB_N_ESTIMATORS,
                    NN_ARCHITECTURES, NN_N_SEEDS, NN_ENSEMBLE_TOP, NN_MAX_ITER, NN_PATIENCE)


def _clip(p):
    return np.asarray(p, dtype=float).clip(min=MIN_VARIANCE)


class MLForecaster:
    """sklearn estimator on a fixed feature matrix, optional log target."""

    def __init__(self, name, estimator, feature_cols, log_target=False, scale=True):
        self.name = name
        self.estimator = estimator
        self.feature_cols = list(feature_cols)
        self.log_target = log_target
        self.scale = scale
        self._fitted = None
        self._resid_var = 0.0

    def _pipe(self):
        if self.scale:
            return Pipeline([('scale', StandardScaler()), ('model', self.estimator)])
        return self.estimator

    def fit(self, train_df):
        X = train_df[self.feature_cols].values
        y = train_df['_target_'].values
        ypos = np.clip(y, MIN_VARIANCE, None)
        if self.log_target:
            # Log models are protected by the log-range clip in predict(); keep
            # a loose outer floor/cap.
            self._floor = max(MIN_VARIANCE, 0.05 * float(np.median(ypos)))
            self._cap = 50.0 * float(np.max(ypos))
            y = np.log(ypos)
            self._logmin, self._logmax = float(y.min()), float(y.max())
        else:
            # Level models (HAR / LevHAR / SHAR / HARQ / HARCJ): a linear fit on raw
            # RV can extrapolate to a negative or explosive forecast on an extreme
            # feature day, which blows up QLIKE (this is what made LevHAR's mean
            # relative loss ~3x). Winsorize forecasts to the training support:
            # floor at the 1st percentile, cap at 3x the training max.
            self._floor = max(MIN_VARIANCE, float(np.quantile(ypos, 0.01)))
            self._cap = 3.0 * float(np.max(ypos))
        pipe = self._pipe()
        pipe.fit(X, y)
        if self.log_target:
            resid = y - pipe.predict(X)
            self._resid_var = float(np.var(resid))
        self._fitted = pipe
        return self

    def predict(self, test_df):
        X = test_df[self.feature_cols].values
        pred = self._fitted.predict(X)
        if self.log_target:
            # Clip the log-forecast to the training range (+margin) before exp,
            # so a linear model cannot extrapolate to an absurd variance on an
            # extreme feature day (which would blow up squared-error loss).
            margin = 0.5 * (self._logmax - self._logmin)
            pred = np.clip(pred, self._logmin - margin, self._logmax + margin)
            pred = np.exp(pred + 0.5 * self._resid_var)
        return np.clip(pred, getattr(self, '_floor', MIN_VARIANCE), getattr(self, '_cap', np.inf))


class EnsembleNN:
    """Geometric-pyramid MLP ensemble.

    Trains NN_N_SEEDS networks of a given architecture, ranks them by an
    internal validation MSE, and averages the best NN_ENSEMBLE_TOP. This
    mirrors the paper's seed-ensembling idea (best 10 of 100), scaled down
    for tractability. sklearn lacks Leaky-ReLU, so 'relu' is used; the paper
    (footnote 7) reports activation choice is immaterial.
    """

    def __init__(self, name, hidden_layer_sizes, feature_cols, alpha=1e-4,
                 n_seeds=NN_N_SEEDS, top=NN_ENSEMBLE_TOP, log_target=True):
        self.name = name
        self.hidden_layer_sizes = hidden_layer_sizes
        self.feature_cols = list(feature_cols)
        self.alpha = alpha
        self.n_seeds = n_seeds
        self.top = top
        self.log_target = log_target
        self._members = []
        self._scaler = None
        self._ymu = 0.0
        self._ysd = 1.0
        self._resid_var = 0.0

    def fit(self, train_df):
        X = train_df[self.feature_cols].values
        y = train_df['_target_'].values
        ypos = np.clip(y, MIN_VARIANCE, None)
        self._floor = max(MIN_VARIANCE, 0.05 * float(np.median(ypos)))
        self._cap = 50.0 * float(np.max(ypos))
        if self.log_target:
            y = np.log(ypos)
        # Standardize the (log) target for stable, comparable training.
        self._ymu, self._ysd = float(y.mean()), float(y.std() + 1e-12)
        y_s = (y - self._ymu) / self._ysd
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        scored = []
        for s in range(self.n_seeds):
            net = MLPRegressor(
                hidden_layer_sizes=self.hidden_layer_sizes,
                activation='relu',
                alpha=self.alpha,
                learning_rate_init=1e-3,
                solver='adam',
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=NN_PATIENCE,
                max_iter=NN_MAX_ITER,
                random_state=RANDOM_STATE + s,
            )
            net.fit(Xs, y_s)
            val = getattr(net, 'best_validation_score_', None)
            score = -val if val is not None else net.loss_
            scored.append((score, net))
        scored.sort(key=lambda t: t[0])
        self._members = [net for _, net in scored[:max(1, self.top)]]
        if self.log_target:
            fit_log = np.column_stack([m.predict(Xs) for m in self._members]).mean(axis=1) * self._ysd + self._ymu
            self._resid_var = float(np.var(y - fit_log))
        return self

    def predict(self, test_df):
        Xs = self._scaler.transform(test_df[self.feature_cols].values)
        preds_s = np.column_stack([m.predict(Xs) for m in self._members]).mean(axis=1)
        pred = preds_s * self._ysd + self._ymu
        if self.log_target:
            lo, hi = self._ymu - 6 * self._ysd, self._ymu + 6 * self._ysd
            pred = np.exp(np.clip(pred, lo, hi) + 0.5 * self._resid_var)
        return np.clip(pred, getattr(self, '_floor', MIN_VARIANCE), getattr(self, '_cap', np.inf))


def har_forecasters(available_cols) -> list:
    """HAR-family OLS models. Each is included only if its columns exist."""
    specs = {
        'HAR':     (['RVD', 'RVW', 'RVM'], False),
        'LogHAR':  (['logRVD', 'logRVW', 'logRVM'], True),
        'LevHAR':  (['RVD', 'RVW', 'RVM', 'rneg_d', 'rneg_w', 'rneg_m'], False),
        'SHAR':    (['RVneg_lag1', 'RVpos_lag1', 'RVW', 'RVM'], False),
        'HARQ':    (['RVD', 'RVW', 'RVM', 'RQ_RVD_inter'], False),
        'HARCJ':   (['CD', 'CW', 'CM', 'JD', 'JW', 'JM'], False),
    }
    out = []
    for name, (cols, logt) in specs.items():
        if all(c in available_cols for c in cols):
            out.append(MLForecaster(name, LinearRegression(), cols, log_target=logt, scale=False))
    return out


def ml_forecasters(feature_cols, tuned: dict | None = None) -> list:
    """ML models on a given feature set, using tuned hyperparameters if supplied."""
    from config import ML_LOG_TARGET
    tuned = tuned or {}
    lt = ML_LOG_TARGET
    rs = RANDOM_STATE
    ridge_a = tuned.get('Ridge', {}).get('alpha', 1.0)
    lasso_a = tuned.get('Lasso', {}).get('alpha', 1e-3)
    enet_a = tuned.get('ElasticNet', {}).get('alpha', 1e-3)
    enet_l1 = tuned.get('ElasticNet', {}).get('l1_ratio', 0.5)
    rf_p = tuned.get('RandomForest', {'max_depth': 6, 'min_samples_leaf': 10})
    gb_p = tuned.get('GradientBoosting', {'learning_rate': 0.05, 'max_depth': 2})
    nn_alpha = tuned.get('NN', {}).get('alpha', 1e-4)

    models = [
        MLForecaster('Ridge', Ridge(alpha=ridge_a, random_state=rs), feature_cols, log_target=lt),
        MLForecaster('Lasso', Lasso(alpha=lasso_a, max_iter=50000, random_state=rs), feature_cols, log_target=lt),
        MLForecaster('ElasticNet', ElasticNet(alpha=enet_a, l1_ratio=enet_l1, max_iter=50000, random_state=rs), feature_cols, log_target=lt),
        MLForecaster('RandomForest', RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, n_jobs=-1, random_state=rs, **rf_p), feature_cols, log_target=lt, scale=False),
        MLForecaster('GradientBoosting', GradientBoostingRegressor(
            n_estimators=GB_N_ESTIMATORS, random_state=rs, min_samples_leaf=10, **gb_p), feature_cols, log_target=lt, scale=False),
    ]
    for nn_name, arch in NN_ARCHITECTURES.items():
        models.append(EnsembleNN(nn_name, arch, feature_cols, alpha=nn_alpha, log_target=lt))
    return models
