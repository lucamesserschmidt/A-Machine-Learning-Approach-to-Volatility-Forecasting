"""Central configuration for the volatility-ML replication package.

This is a from-scratch upgrade of the original conceptual replication of
Christensen, Siggaard & Veliyev (2023), "A Machine Learning Approach to
Volatility Forecasting", Journal of Financial Econometrics.

Key upgrades over the first version:
  * Intraday 5-minute realized variance (RV5) is the HEADLINE target.
  * Daily Garman-Klass variance is kept as a robustness layer only.
  * Neural networks are re-enabled (geometric-pyramid MLP + seed ensemble).
  * Regularized / tree / NN hyperparameters are tuned on a validation set.
  * LogHAR (paper's best HAR) plus LevHAR / SHAR / HARQ horse-race models.
  * Diebold-Mariano test uses HAC (Newey-West) + Harvey-Leybourne-Newbold.
  * Accumulated Local Effects (ALE) variable importance (Apley & Zhu 2020).
  * Cross-sectional relative-MSE summary tables across assets.
  * VaR backtest with Kupiec + Christoffersen tests.
"""
from pathlib import Path

DATA_DIR = Path('data')
RESULTS_DIR = Path('results_enhanced')
PLOTS_DIR = RESULTS_DIR / 'plots'

START_DATE = '2005-03-01'
END_DATE = None
MIN_VARIANCE = 1e-12
RANDOM_STATE = 42

# ----------------------------------------------------------------------------
# Assets
# ----------------------------------------------------------------------------
# Intraday RV5 is the headline (closest to the paper). Daily GK is robustness.
INTRADAY_ASSET_FILES = {
    'SPY_INTRADAY_RV5': DATA_DIR / 'intraday' / 'spy_1min.csv',
}

# Daily Garman-Klass robustness panel. Default = 4 names across sectors
# (market ETF, big-tech, financials, energy) -- enough to show the RV5 results
# are not specific to one asset/measure. Swap in DAILY_ASSET_FILES_ALL for the
# full 12-name panel once you have time for a multi-night run.
DAILY_ASSET_FILES_ALL = {
    'SPY_DAILY_GK': DATA_DIR / 'spy_us_d.csv',
    'SPX_DAILY_GK': DATA_DIR / '^spx_d.csv',
    'AAPL_DAILY_GK': DATA_DIR / 'stock_data' / 'aapl.us.txt',
    'MSFT_DAILY_GK': DATA_DIR / 'stock_data' / 'msft.us.txt',
    'INTC_DAILY_GK': DATA_DIR / 'stock_data' / 'intc.us.txt',
    'WMT_DAILY_GK': DATA_DIR / 'stock_data' / 'wmt.us.txt',
    'JPM_DAILY_GK': DATA_DIR / 'stock_data' / 'jpm.us.txt',
    'XOM_DAILY_GK': DATA_DIR / 'stock_data' / 'xom.us.txt',
    'JNJ_DAILY_GK': DATA_DIR / 'stock_data' / 'jnj.us.txt',
    'KO_DAILY_GK': DATA_DIR / 'stock_data' / 'ko.us.txt',
    'NVDA_DAILY_GK': DATA_DIR / 'stock_data' / 'nvda.us.txt',
    'AMZN_DAILY_GK': DATA_DIR / 'stock_data' / 'amzn.us.txt',
}
DAILY_ASSET_FILES = {
    'SPY_DAILY_GK': DATA_DIR / 'spy_us_d.csv',
    'AAPL_DAILY_GK': DATA_DIR / 'stock_data' / 'aapl.us.txt',
    'JPM_DAILY_GK': DATA_DIR / 'stock_data' / 'jpm.us.txt',
    'XOM_DAILY_GK': DATA_DIR / 'stock_data' / 'xom.us.txt',
}

MACRO_FILE = DATA_DIR / 'fred_macro_predictors_1990_today.csv'

# ----------------------------------------------------------------------------
# Supervisor's 5-minute stock bars (.RData) -- the NEW HEADLINE (real per-stock
# RV5, the paper's actual target and asset class, 2009-2019).
# Each file is one stock: <ticker>_returns.RData with object data.<TICKER>.
# ----------------------------------------------------------------------------
STOCK_BAR_DIR = DATA_DIR / 'stock_bars'
STOCK_BAR_FILE_PATTERN = '{ticker}_returns.RData'
STOCK_BAR_TICKERS = [
    'AAPL', 'ADI', 'AGEN', 'AGNC', 'AMAT', 'AMD', 'AMZN', 'APA', 'ARCC', 'ATVI',
    'AZN', 'CMCSA', 'CPRT', 'CSCO', 'CSX', 'DISH', 'DXCM', 'EBAY', 'EGHT', 'ERIC',
    'EXC', 'FAST', 'FCEL', 'FITB', 'FLEX', 'FOLD', 'GILD', 'GOOG', 'HBAN', 'INTC',
    'JBLU', 'MCHP', 'MRVL', 'MSFT', 'MU', 'NFLX', 'NVAX', 'NVDA', 'NWL', 'OSTK',
    'PAA', 'PENN', 'PEP', 'PLUG', 'PTEN', 'QCOM', 'SBUX', 'SIRI', 'TXN', 'VOD',
    'WDC', 'WEN', 'XEL', 'ZION',
]
# Stocks that also have LOBSTER messages -> order-flow deep-dive (Tier B).
STOCK_BAR_DEEPDIVE = ['AAPL', 'AMD', 'AMZN', 'ATVI', 'AZN', 'APA', 'ARCC']
# OOS split for the 2009-2019 stock sample.
STOCK_VALID_START = '2016-01-01'
STOCK_TEST_START = '2017-01-01'

# LOBSTER message files (one per stock-year) and the daily order-flow CSVs the
# preprocessing pass writes. Run preprocess_lobster.py once before the main run.
LOBSTER_MSG_DIR = DATA_DIR / 'lobster_messages'
LOBSTER_FILE_PATTERN = '{ticker}_raw_{year}.RData'
LOBSTER_YEARS = list(range(2009, 2020))
ORDERFLOW_DIR = DATA_DIR / 'orderflow'
ORDERFLOW_FILE_PATTERN = '{ticker}_orderflow.csv'

# Oxford-Man Institute Realized Library (long-format CSV: one row per day/symbol).
# Headline RV5 source for a faithful cross-section. This snapshot spans
# 2000-01 .. 2018-06 for ~31 indices. Add/remove symbols to widen the panel.
OXFORD_MAN_FILE = DATA_DIR / 'oxfordmanrealizedvolatilityindices.csv'
OXFORD_MAN_SYMBOLS = ['.SPX', '.DJI', '.IXIC', '.RUT', '.FTSE', '.GDAXI', '.N225', '.FCHI']

# ----------------------------------------------------------------------------
# Forecast design
# ----------------------------------------------------------------------------
HORIZONS = [1, 5, 22]               # one-day, one-week, one-month (paper's horizons)

# Feature sets, mirroring the paper's M_HAR and M_ALL.
FEATURE_SETS = {
    'MHAR': ['RVD', 'RVW', 'RVM'],
    'MALL': ['RVD', 'RVW', 'RVM', 'VIX_lag1', 'EPU_lag1',
             'd_DTB3_lag1', 'd_log_dollar_volume_lag1', 'M1W_lag1'],
}

# Oxford-Man carries no volume, so M_ALL drops the dollar-volume predictor.
# (Realized quarticity is also absent, so HARQ is skipped automatically.)
FEATURE_SETS_OMI = {
    'MHAR': ['RVD', 'RVW', 'RVM'],
    'MALL': ['RVD', 'RVW', 'RVM', 'VIX_lag1', 'EPU_lag1', 'd_DTB3_lag1', 'M1W_lag1'],
}

# Deep-dive stocks with LOBSTER order-flow predictors. We keep a CLEAN M_ALL
# (paper's predictors only) and a separate M_ALL+OF (augmented with lagged
# order-flow imbalance, relative spread, signed trade imbalance) so the marginal
# value of order-flow can be measured directly (M_ALL vs M_ALL+OF).
FEATURE_SETS_OF = {
    'MHAR': ['RVD', 'RVW', 'RVM'],
    'MALL': ['RVD', 'RVW', 'RVM', 'VIX_lag1', 'EPU_lag1', 'd_DTB3_lag1',
             'd_log_dollar_volume_lag1', 'M1W_lag1'],
    'MALL_OF': ['RVD', 'RVW', 'RVM', 'VIX_lag1', 'EPU_lag1', 'd_DTB3_lag1',
                'd_log_dollar_volume_lag1', 'M1W_lag1', 'ofi_lag1', 'relspr_lag1', 'imb_lag1'],
}

# Out-of-sample windows.
#  * Intraday RV5 spans ~2008-2021, so it gets its own earlier test split.
#  * Daily GK spans 2005-today, so it gets a long modern test window.
EVALUATION_MODE = 'expanding'        # 'expanding' or 'rolling'
ROLLING_TRAIN_WINDOW_DAYS = 2500
REFIT_EVERY_N_DAYS = 21              # refit (weights) about monthly

# Daily GK split
DAILY_VALID_START = '2017-01-01'
DAILY_TEST_START = '2019-01-01'
# Intraday RV5 split (data ends 2021-05) -> meaningful OOS window
INTRADAY_VALID_START = '2017-01-01'
INTRADAY_TEST_START = '2018-01-01'
# Oxford-Man RV5 split (data spans 2000..2018) -> ~2.5y OOS to mid-2018
OXFORD_VALID_START = '2015-01-01'
OXFORD_TEST_START = '2016-01-01'

# Validation fraction used for one-shot hyperparameter tuning (tail of pre-test data)
VALID_FRACTION = 0.15

# ----------------------------------------------------------------------------
# Intraday cleaning / RV construction
# ----------------------------------------------------------------------------
FILTER_INTRADAY_REGULAR_HOURS = True
REGULAR_SESSION_START = '07:30:00'
REGULAR_SESSION_END = '14:00:00'
INTRADAY_RESAMPLE_RULE = '5min'
MIN_INTRADAY_5MIN_BARS_PER_DAY = 60   # full regular day ~ 78 five-minute bars

# ----------------------------------------------------------------------------
# Models / tuning
# ----------------------------------------------------------------------------
# Neural network: geometric-pyramid architectures (paper NN1..NN4) + seed ensemble.
# The paper trains 100 seeds and ensembles the best 10. Defaults below are lighter
# for tractability; raise NN_N_SEEDS / NN_ENSEMBLE_TOP toward 100 / 10 for fidelity.
NN_ARCHITECTURES = {
    'NN1': (2,),
    'NN2': (4, 2),
    'NN3': (8, 4, 2),
}
NN_N_SEEDS = 8
NN_ENSEMBLE_TOP = 4
NN_MAX_ITER = 500
NN_PATIENCE = 25

# Tuning grids (kept modest; tuned once on the initial train/validation split).
RIDGE_ALPHAS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
LASSO_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
ENET_ALPHAS = [1e-4, 1e-3, 1e-2, 1e-1]
ENET_L1_RATIOS = [0.1, 0.5, 0.9]
RF_GRID = {'max_depth': [4, 6, 10], 'min_samples_leaf': [5, 10, 20]}
RF_N_ESTIMATORS = 300
GB_GRID = {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [1, 2, 3]}
GB_N_ESTIMATORS = 400

# Realized variance is right-skewed with a heavy tail (crisis spikes). Training
# the ML models on log-RV (inverted with a Jensen correction) stabilizes them
# and matches standard practice; on raw levels, least-squares OLS-HAR is nearly
# unbeatable on MSE and the NN is numerically unstable. Set False to force levels.
ML_LOG_TARGET = True

# ----------------------------------------------------------------------------
# VaR
# ----------------------------------------------------------------------------
VAR_ALPHA = 0.05      # paper's headline level (1-alpha = 95%); 0.01 also discussed
VAR_METHODS = ('gaussian', 'fhs')   # both are computed and reported
FHS_MIN_HISTORY = 250               # burn-in before the empirical residual quantile is used

# ----------------------------------------------------------------------------
# ALE
# ----------------------------------------------------------------------------
ALE_N_BINS = 40
ALE_FEATURES = ['RVD', 'RVW', 'VIX_lag1', 'M1W_lag1']

# ----------------------------------------------------------------------------
# LIGHT MODE  --  flip to True for a fast smoke run, False for full fidelity.
# ----------------------------------------------------------------------------
# Light mode shrinks the NN ensemble, coarsens the refit cadence, trims the
# tuning grids, and restricts the asset panel so a full pass finishes in
# minutes instead of hours. It does NOT change any methodology -- only volume.
LIGHT_MODE = True

if LIGHT_MODE:
    # Smaller NN ensemble (paper-faithful is 100 seeds / best 10).
    NN_ARCHITECTURES = {'NN2': (4, 2)}
    NN_N_SEEDS = 3
    NN_ENSEMBLE_TOP = 2
    NN_MAX_ITER = 300
    # Refit ~quarterly instead of monthly, and only one horizon.
    REFIT_EVERY_N_DAYS = 63
    HORIZONS = [1]
    # Trim tuning grids and tree sizes.
    RIDGE_ALPHAS = [1e-2, 1.0, 100.0]
    LASSO_ALPHAS = [1e-4, 1e-2]
    ENET_ALPHAS = [1e-3, 1e-1]
    ENET_L1_RATIOS = [0.5]
    RF_GRID = {'max_depth': [6], 'min_samples_leaf': [10]}
    GB_GRID = {'learning_rate': [0.05], 'max_depth': [2]}
    RF_N_ESTIMATORS = 150
    GB_N_ESTIMATORS = 150
    # Restrict the panel: headline OMI index + one daily asset, skip slow intraday.
    OXFORD_MAN_SYMBOLS = ['.SPX']
    DAILY_ASSET_FILES = {'SPY_DAILY_GK': DATA_DIR / 'spy_us_d.csv'}
    # Stock headline: one ticker only for a fast smoke run.
    STOCK_BAR_TICKERS = ['AAPL']
