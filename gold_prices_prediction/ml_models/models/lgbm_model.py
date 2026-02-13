"""
LightGBM Regressor for gold price forecasting.

Why LightGBM?
- Gradient-boosted trees capture non-linear feature interactions
- Handles mixed feature types (numeric + categorical regime labels)
- Built-in feature importance for interpretability
- Fast training even with hundreds of boosting rounds

Time-series safety:
- NO random shuffling of training data
- Early stopping on a chronologically later validation set
- Subsample + colsample_bytree to reduce overfitting on sequential data

Overfitting prevention:
- max_depth=6 limits tree complexity
- min_child_samples=20 prevents noisy leaf nodes
- Early stopping monitors validation RMSE
"""

import numpy as np
import lightgbm as lgb
from ml_models.config import LGBM_PARAMS, LGBM_EARLY_STOPPING


def train_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
) -> lgb.LGBMRegressor:
    """
    Train a LightGBM Regressor with early stopping on validation set.

    Args:
        X_train, y_train: Training data (chronologically first)
        X_val, y_val: Validation data (chronologically after train)
        feature_names: Optional list of feature names for importance output

    Returns:
        Fitted LGBMRegressor
    """
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=True),
            lgb.log_evaluation(period=100),
        ],
    )

    print(f"[LightGBM] Best iteration: {model.best_iteration_}")

    # Feature importance
    if feature_names is not None:
        importances = model.feature_importances_
        feat_imp = sorted(zip(feature_names, importances), key=lambda x: -x[1])
        print("[LightGBM] Top 10 features:")
        for name, imp in feat_imp[:10]:
            print(f"  {name}: {imp}")

    return model


def predict_lgbm(model: lgb.LGBMRegressor, X: np.ndarray) -> np.ndarray:
    """Generate predictions using trained LightGBM model."""
    return model.predict(X)
