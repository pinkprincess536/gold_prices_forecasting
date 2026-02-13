"""
k-Nearest Neighbors Regressor for gold price forecasting.

Why k-NN?
- Instance-based / non-parametric: makes no assumptions about functional form
- Captures local patterns: "what happened when features looked like this before?"
- Complementary to parametric models (Ridge) and tree-based models (LightGBM)

Configuration:
- k=10: With ~2000 training samples, sqrt(N)≈45 is too smooth.
  k=10 balances bias/variance — enough neighbours for stability,
  few enough to retain sensitivity to local structure.
- weights='distance': closer historical analogues contribute more,
  reducing the influence of distant (less relevant) neighbours.
- Features must be scaled (StandardScaler) because k-NN uses distance.

Limitations:
- Curse of dimensionality: with 30+ features, Euclidean distance becomes
  less meaningful. Mitigation: feature importance from LightGBM could inform
  feature selection (not implemented here for simplicity).
- No extrapolation: cannot predict beyond the range of training data.
  If gold reaches all-time highs, k-NN finds no analogues.
- Computationally expensive at prediction time (must search all training points).
"""

import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from ml_models.config import KNN_K, KNN_WEIGHTS


def train_knn(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    """
    Train a k-NN Regressor with feature scaling.

    Returns:
        (model, scaler) — both needed for prediction
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KNeighborsRegressor(n_neighbors=KNN_K, weights=KNN_WEIGHTS)
    model.fit(X_scaled, y_train)

    train_score = model.score(X_scaled, y_train)
    print(f"[k-NN] Train R²: {train_score:.4f}, k={KNN_K}, weights='{KNN_WEIGHTS}'")

    return model, scaler


def predict_knn(
    model: KNeighborsRegressor, scaler: StandardScaler, X: np.ndarray
) -> np.ndarray:
    """Generate predictions using trained k-NN model."""
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled)
