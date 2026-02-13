"""
Ridge Regression model for gold price forecasting.

Why Ridge?
- Serves as the LINEAR BASELINE. If tree/kernel models can't beat Ridge,
  the problem is dominated by linear relationships and feature quality
  matters more than model complexity.
- L2 regularisation handles multicollinearity (correlated returns, MAs).
- Fast to train, interpretable coefficients.

Assumptions:
- Linear relationship between features and forward price
- Features are scaled (StandardScaler applied before fitting)

Limitations:
- Cannot capture non-linear patterns or interactions
- Assumes homoscedastic residuals
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from ml_models.config import RIDGE_ALPHA


def train_ridge(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train a Ridge Regression model with feature scaling.

    Returns:
        (model, scaler) — both needed for prediction
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = Ridge(alpha=RIDGE_ALPHA, random_state=42)
    model.fit(X_scaled, y_train)

    train_score = model.score(X_scaled, y_train)
    print(f"[Ridge] Train R²: {train_score:.4f}, alpha={RIDGE_ALPHA}")

    return model, scaler


def predict_ridge(model: Ridge, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Generate predictions using trained Ridge model."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
