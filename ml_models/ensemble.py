"""
Ensemble / blending module.

Two blending strategies:
1. Simple average — equal-weight baseline
2. Inverse-RMSE weighted — models with lower validation RMSE get higher weight

Why blend?
- Diverse model types (linear, tree, instance-based, kernel) make
  different errors. Averaging reduces correlated prediction errors.
- Ensembles are more stable than individual models: a single model
  may overfit to a particular regime while others compensate.
- In finance, model disagreement itself is informative — high
  disagreement signals uncertainty / regime transition.
"""

import numpy as np
import pandas as pd


def simple_average(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """Equal-weight average of all model predictions."""
    preds = np.column_stack(list(predictions.values()))
    return preds.mean(axis=1)


def inverse_rmse_weighted(
    predictions: dict[str, np.ndarray],
    val_rmse: dict[str, float],
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Weighted average where weight ∝ 1/RMSE on validation set.

    Returns:
        (blended_predictions, weights_dict)
    """
    names = list(predictions.keys())
    preds = np.column_stack([predictions[n] for n in names])
    rmses = np.array([val_rmse[n] for n in names])

    # Inverse RMSE weights, normalised to sum to 1
    inv_rmse = 1.0 / rmses
    weights = inv_rmse / inv_rmse.sum()

    weight_dict = {n: float(w) for n, w in zip(names, weights)}
    print(f"[Ensemble] Inverse-RMSE weights: {weight_dict}")

    blended = preds @ weights
    return blended, weight_dict


def model_disagreement(predictions: dict[str, np.ndarray]) -> np.ndarray:
    """
    Standard deviation across model predictions at each time step.

    High disagreement → low confidence / regime uncertainty.
    """
    preds = np.column_stack(list(predictions.values()))
    return preds.std(axis=1)
