"""
Support Vector Regression (SVR) for gold price forecasting.

Why SVR?
- Kernel trick maps features into high-dimensional space to capture
  non-linear relationships without explicitly computing the transformation
- Epsilon-insensitive loss ignores small errors (robust to noise in financial data)
- Complementary to tree-based models: SVR finds a global smooth function
  while trees partition the space

Configuration justification:
- kernel='rbf': Radial Basis Function is a universal approximator.
  It captures non-linear patterns without requiring domain knowledge
  about the polynomial degree. RBF works well as a default when the
  true functional form is unknown.
- C=10: The regularisation parameter. C=10 provides moderate penalty
  for margin violations — loose enough to tolerate financial noise,
  tight enough to fit real signal. C=1 underfits; C=1000 overfits.
- epsilon=0.1: Width of the insensitive tube. Points within ±ε of
  the prediction are not penalised. 0.1 is a standard default.
- gamma='scale': Computed as 1/(n_features × X.var()), automatically
  adapts to feature scale and dimensionality.

Limitations:
- O(n²) to O(n³) training complexity — slower than Ridge/LightGBM
- Requires feature scaling (StandardScaler)
- Single hyperparameter set may not be optimal across all regimes
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from ml_models.config import SVR_KERNEL, SVR_C, SVR_EPSILON, SVR_GAMMA


def train_svr(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train an SVR model with feature scaling.

    Returns:
        (model, scaler) — both needed for prediction
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # Also scale the target for SVR (helps convergence)
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_scaled = (y_train - y_mean) / y_std

    model = SVR(kernel=SVR_KERNEL, C=SVR_C, epsilon=SVR_EPSILON, gamma=SVR_GAMMA)
    model.fit(X_scaled, y_scaled)

    train_pred = model.predict(X_scaled) * y_std + y_mean
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    print(f"[SVR] Train RMSE: {train_rmse:.2f}, kernel='{SVR_KERNEL}', C={SVR_C}")

    return model, scaler, y_mean, y_std


def predict_svr(
    model: SVR,
    scaler: StandardScaler,
    X: np.ndarray,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """Generate predictions using trained SVR model, reversing target scaling."""
    X_scaled = scaler.transform(X)
    y_scaled = model.predict(X_scaled)
    return y_scaled * y_std + y_mean
