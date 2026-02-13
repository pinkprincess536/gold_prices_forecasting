"""
Evaluation metrics for the ML forecasting pipeline.

Metrics:
- MAE:  Mean Absolute Error — interpretable in price units (₹)
- RMSE: Root Mean Squared Error — penalises large errors more
- MAPE: Mean Absolute Percentage Error — scale-independent comparison
- Directional Accuracy: % of times model correctly predicts price direction
"""

import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAPE — excludes zero actuals to avoid division by zero."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray, current_prices: np.ndarray) -> float:
    """
    Percentage of times the model correctly predicts the direction of price change.

    Direction = sign(future_price - current_price).
    """
    actual_dir = np.sign(y_true - current_prices)
    pred_dir = np.sign(y_pred - current_prices)
    return float(np.mean(actual_dir == pred_dir) * 100)


def evaluate_all(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    current_prices: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Evaluate all models and return a summary DataFrame.

    Args:
        y_true: Actual target values
        predictions: dict of {model_name: predicted_values}
        current_prices: Current gold prices (for directional accuracy)

    Returns:
        DataFrame with columns [Model, MAE, RMSE, MAPE, Dir_Accuracy]
    """
    results = []
    for name, y_pred in predictions.items():
        row = {
            "Model": name,
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        }
        if current_prices is not None:
            row["Dir_Accuracy"] = directional_accuracy(y_true, y_pred, current_prices)
        results.append(row)

    df = pd.DataFrame(results)
    print("\n[Evaluation] Metrics Summary:")
    print(df.to_string(index=False))
    return df
