"""
Central configuration for the ML forecasting pipeline.

All magic numbers and hyperparameters are defined here with justification.
"""

import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

GOLD_CSV = os.path.join(DATA_DIR, "gold.csv")
SILVER_CSV = os.path.join(DATA_DIR, "silver_new.csv")  # Use silver_new per instructions

PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")

# ─── Forecast ────────────────────────────────────────────────────────────────
# 3 months ≈ 63 trading days (21 trading days/month × 3)
FORECAST_HORIZON = 63

# ─── Train / Validation / Test Split ─────────────────────────────────────────
# Chronological split to prevent look-ahead bias.
# 70% train, 15% validation (for hyperparameter tuning & ensemble weights), 15% test.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ─── Feature Engineering ─────────────────────────────────────────────────────
# Return look-back periods (trading days)
RETURN_PERIODS = [1, 5, 21, 63]

# Rolling volatility windows (annualised via sqrt(252))
VOLATILITY_WINDOWS = [21, 63, 252]

# Simple Moving Average windows
SMA_WINDOWS = [20, 50, 200]

# RSI period
RSI_PERIOD = 14

# MACD parameters (fast, slow, signal)
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Rolling z-score window
ZSCORE_WINDOW = 200

# ─── HMM ─────────────────────────────────────────────────────────────────────
# Number of hidden states for Gaussian HMM.
# 3 states: (1) low-vol trending, (2) high-vol trending, (3) crisis/correction.
# Chosen via BIC comparison in hmm_regime.py; 3 is the empirical sweet spot for
# commodity price series — 2 merges distinct regimes, 4+ leads to unstable states.
HMM_N_STATES = 3
HMM_N_ITER = 100
HMM_COVARIANCE_TYPE = "full"  # Full covariance captures cross-feature correlations

# ─── Ridge Regression ────────────────────────────────────────────────────────
# alpha=1.0: Default L2 penalty. With ~30+ correlated features (returns at
# multiple horizons, overlapping MAs), moderate regularisation prevents
# multicollinearity-driven instability. Cross-validated in practice.
RIDGE_ALPHA = 1.0

# ─── LightGBM ───────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "n_estimators": 2000,       # More trees since learning rate is low; early stopping prevents waste
    "learning_rate": 0.005,     # Returns are ~0.02 magnitude; 0.05 was too fast (stopped at 3 iters)
    "max_depth": 4,             # Shallower trees for ~2k samples; reduces overfitting
    "num_leaves": 15,           # 2^4 - 1; compatible with max_depth=4
    "subsample": 0.8,           # Row sampling to reduce variance
    "colsample_bytree": 0.7,   # Feature sampling per tree; slightly more aggressive
    "min_child_samples": 30,    # Larger min leaf for stability on noisy return targets
    "reg_alpha": 0.1,           # L1 regularisation for feature selection
    "reg_lambda": 1.0,          # L2 regularisation for smoother predictions
    "random_state": 42,
    "verbose": -1,
}
LGBM_EARLY_STOPPING = 100  # More patience since learning rate is slower

# ─── k-NN ────────────────────────────────────────────────────────────────────
# k=10: With ~2000 training samples, sqrt(N) ≈ 45 is too large and over-smooths.
# k=10 provides a balance: enough neighbours for stability while retaining
# sensitivity to local structure. Distance weighting ensures closer analogues
# dominate the prediction.
KNN_K = 10
KNN_WEIGHTS = "distance"

# ─── SVR ─────────────────────────────────────────────────────────────────────
# RBF kernel: Universal approximator; captures nonlinear patterns without
# specifying polynomial degree. Good default for unknown functional forms.
# C=1.0: Lower regularisation for return-scale targets (~0.02 magnitude);
#        C=10 was designed for price-scale targets in the thousands.
# epsilon=0.001: Return targets have std ~0.06; the old 0.1 epsilon was larger
#        than most targets, so SVR treated nearly everything as within the
#        insensitive tube and learned almost nothing.
# gamma='scale': 1/(n_features * X.var()), adapts automatically to data scale.
SVR_KERNEL = "rbf"
SVR_C = 1.0
SVR_EPSILON = 0.001
SVR_GAMMA = "scale"
