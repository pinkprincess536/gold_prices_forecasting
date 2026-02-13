"""
End-to-end ML pipeline orchestrator for gold price forecasting.

Run with:  python -m ml_models.pipeline

KEY DESIGN: Models predict 63-day forward LOG RETURNS (not raw prices).
This solves the extrapolation problem — returns are stationary and bounded,
so tree/instance-based models work correctly even when prices break new highs.
Predictions are converted back to price: price_pred = current_price × exp(return_pred).

Steps:
1. Load & clean data
2. Engineer features
3. Detect regimes (HMM) and add as feature
4. Construct target (63-day forward log return)
5. Time-aware train/val/test split
6. Train all models on return target
7. Convert return predictions → price predictions
8. Evaluate on test set
9. Blend predictions
10. Save predictions.csv and plots
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for plot saving
import matplotlib.pyplot as plt

from ml_models.config import (
    FORECAST_HORIZON,
    TRAIN_RATIO,
    VAL_RATIO,
    OUTPUT_DIR,
    PLOT_DIR,
    PREDICTIONS_CSV,
)
from ml_models.data_loader import load_data
from ml_models.feature_engineering import build_features
from ml_models.models.hmm_regime import fit_hmm
from ml_models.models.ridge_model import train_ridge, predict_ridge
from ml_models.models.lgbm_model import train_lgbm, predict_lgbm
from ml_models.models.knn_model import train_knn, predict_knn
from ml_models.models.svr_model import train_svr, predict_svr
from ml_models.evaluation import evaluate_all, rmse
from ml_models.ensemble import inverse_rmse_weighted, model_disagreement

warnings.filterwarnings("ignore")


def construct_target(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the target variable: 63-day forward LOG RETURN.

    target[t] = log(price[t+63] / price[t])

    Why log returns instead of raw prices?
    - Returns are stationary: a 5% return looks the same at ₹30k or ₹70k
    - Tree/instance-based models can't extrapolate beyond training range
    - Predicting raw prices fails when test prices exceed all training values
    - Log returns have better statistical properties (additivity, approximate normality)

    The forward price is reconstructed as: price[t+63] = price[t] × exp(target[t])
    """
    df = features_df.copy()

    # Forward price (for reference / evaluation)
    df["forward_price"] = df["gold_close"].shift(-FORECAST_HORIZON)

    # Target: log return over next 63 days
    df["target"] = np.log(df["forward_price"] / df["gold_close"])

    before = len(df)
    df.dropna(subset=["target"], inplace=True)

    target_stats = df["target"].describe()
    print(f"[Target] Forward {FORECAST_HORIZON}d LOG RETURN: {before} -> {len(df)} rows")
    print(f"[Target] Return stats: mean={target_stats['mean']:.4f}, std={target_stats['std']:.4f}, "
          f"min={target_stats['min']:.4f}, max={target_stats['max']:.4f}")

    return df


def time_split(df: pd.DataFrame) -> tuple:
    """
    Chronological train/val/test split.
    No shuffling — prevents look-ahead bias.
    """
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    print(f"[Split] Train: {len(train)} ({train.index.min().date()} -> {train.index.max().date()})")
    print(f"[Split] Val:   {len(val)} ({val.index.min().date()} -> {val.index.max().date()})")
    print(f"[Split] Test:  {len(test)} ({test.index.min().date()} -> {test.index.max().date()})")

    return train, val, test


def get_xy(df: pd.DataFrame, feature_cols: list[str]):
    """Split DataFrame into feature matrix X and target vector y."""
    return df[feature_cols].values, df["target"].values


def returns_to_prices(log_return_preds: np.ndarray, current_prices: np.ndarray) -> np.ndarray:
    """Convert predicted log returns back to price predictions."""
    return current_prices * np.exp(log_return_preds)


# ─── Plotting helpers ────────────────────────────────────────────────────────

def plot_predictions(dates, actual, preds_dict, title, filename):
    """Plot actual vs predicted for one or more models."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, actual, color="black", linewidth=1.5, label="Actual", alpha=0.8)

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    for i, (name, pred) in enumerate(preds_dict.items()):
        ax.plot(dates, pred, linewidth=1.2, label=name, color=colors[i % len(colors)], alpha=0.75)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (₹/10g)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filename}")


def plot_regimes(dates, prices, regimes, filename):
    """Visualise HMM-detected regimes as background colors."""
    fig, ax = plt.subplots(figsize=(14, 6))
    regime_colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    regime_labels = {0: "Low-Vol Trend", 1: "High-Vol Trend", 2: "Crisis/Correction"}

    ax.plot(dates, prices, color="black", linewidth=1, alpha=0.8)

    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        ax.fill_between(
            dates, prices.min(), prices.max(),
            where=mask, alpha=0.15,
            color=regime_colors.get(regime, "gray"),
            label=regime_labels.get(regime, f"Regime {regime}"),
        )

    ax.set_title("Gold Price with HMM-Detected Market Regimes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Gold Price (₹/10g)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150)
    plt.close(fig)
    print(f"[Plot] Saved: {filename}")


def plot_metrics_bar(metrics_df, filename):
    """Bar chart comparing model metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, metric in enumerate(["MAE", "RMSE", "MAPE"]):
        ax = axes[i]
        bars = ax.bar(metrics_df["Model"], metrics_df[metric],
                       color=["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"])
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Model Performance Comparison (Test Set)", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved: {filename}")


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_pipeline():
    """Execute the full ML forecasting pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    print("=" * 70)
    print("  GOLD PRICE 3-MONTH FORWARD FORECASTING PIPELINE")
    print("  (Return-based prediction for regime-robust performance)")
    print("=" * 70)

    # Step 1: Load data
    # Note: Use ASCII-only text for Windows console compatibility
    print("\n-- Step 1: Loading data --")
    gold, silver = load_data()

    # Step 2: Feature engineering
    print("\n-- Step 2: Feature engineering --")
    features = build_features(gold, silver)

    # Step 3: HMM Regime detection
    print("\n-- Step 3: HMM Regime detection --")
    hmm_model, regime_labels, bic_scores = fit_hmm(features)
    features["hmm_regime"] = regime_labels

    # Save regime plot on full feature set
    plot_regimes(features.index, features["gold_close"], features["hmm_regime"], "regime_visualization.png")

    # Step 4: Construct target (LOG RETURN, not price)
    print("\n-- Step 4: Constructing target (forward log return) --")
    df = construct_target(features)

    # Step 5: Time split
    print("\n-- Step 5: Train/Val/Test split --")
    train, val, test = time_split(df)

    # Feature columns = everything except gold_close, target, forward_price
    feature_cols = [c for c in df.columns if c not in ["gold_close", "target", "forward_price"]]
    print(f"[Features] {len(feature_cols)} features used: {feature_cols[:5]}... ")

    X_train, y_train = get_xy(train, feature_cols)
    X_val, y_val = get_xy(val, feature_cols)
    X_test, y_test = get_xy(test, feature_cols)

    # Current prices and actual forward prices for conversion
    current_prices_train = train["gold_close"].values
    current_prices_val = val["gold_close"].values
    current_prices_test = test["gold_close"].values
    actual_forward_prices = test["forward_price"].values

    # Step 6: Train models (on log return targets)
    print("\n-- Step 6: Training models (predicting log returns) --")

    print("\n[6a] Ridge Regression")
    ridge_model, ridge_scaler = train_ridge(X_train, y_train)
    ridge_val_ret = predict_ridge(ridge_model, ridge_scaler, X_val)
    ridge_test_ret = predict_ridge(ridge_model, ridge_scaler, X_test)

    print("\n[6b] LightGBM")
    lgbm_model = train_lgbm(X_train, y_train, X_val, y_val, feature_names=feature_cols)
    lgbm_val_ret = predict_lgbm(lgbm_model, X_val)
    lgbm_test_ret = predict_lgbm(lgbm_model, X_test)

    print("\n[6c] k-NN")
    knn_model, knn_scaler = train_knn(X_train, y_train)
    knn_val_ret = predict_knn(knn_model, knn_scaler, X_val)
    knn_test_ret = predict_knn(knn_model, knn_scaler, X_test)

    print("\n[6d] SVR")
    svr_model, svr_scaler, svr_y_mean, svr_y_std = train_svr(X_train, y_train)
    svr_val_ret = predict_svr(svr_model, svr_scaler, X_val, svr_y_mean, svr_y_std)
    svr_test_ret = predict_svr(svr_model, svr_scaler, X_test, svr_y_mean, svr_y_std)

    # Step 7: Convert return predictions to price predictions
    print("\n-- Step 7: Converting returns to prices --")

    # Validation set (for ensemble weight calibration)
    val_price_preds = {
        "Ridge": returns_to_prices(ridge_val_ret, current_prices_val),
        "LightGBM": returns_to_prices(lgbm_val_ret, current_prices_val),
        "k-NN": returns_to_prices(knn_val_ret, current_prices_val),
        "SVR": returns_to_prices(svr_val_ret, current_prices_val),
    }
    val_actual_prices = val["forward_price"].values

    # Test set
    test_price_preds = {
        "Ridge": returns_to_prices(ridge_test_ret, current_prices_test),
        "LightGBM": returns_to_prices(lgbm_test_ret, current_prices_test),
        "k-NN": returns_to_prices(knn_test_ret, current_prices_test),
        "SVR": returns_to_prices(svr_test_ret, current_prices_test),
    }

    for name, pred in test_price_preds.items():
        print(f"  {name}: first pred = Rs {pred[0]:,.0f}, actual = Rs {actual_forward_prices[0]:,.0f}")

    # Step 8: Ensemble blending (on price predictions)
    print("\n-- Step 8: Ensemble blending --")
    val_rmses = {name: rmse(val_actual_prices, pred) for name, pred in val_price_preds.items()}
    print(f"[Ensemble] Validation RMSEs: { {k: f'{v:.0f}' for k, v in val_rmses.items()} }")

    ensemble_test, weights = inverse_rmse_weighted(test_price_preds, val_rmses)
    test_price_preds["Ensemble"] = ensemble_test

    # Step 9: Evaluate on test set (in price space)
    print("\n-- Step 9: Evaluation --")
    metrics_df = evaluate_all(actual_forward_prices, test_price_preds, current_prices_test)

    # Disagreement
    disagree = model_disagreement({k: v for k, v in test_price_preds.items() if k != "Ensemble"})
    print(f"[Ensemble] Mean model disagreement: Rs {disagree.mean():,.0f}, Max: Rs {disagree.max():,.0f}")

    # Step 10: Save outputs
    print("\n-- Step 10: Saving outputs --")

    # 10a: predictions.csv
    results_df = pd.DataFrame({
        "Date": test.index,
        "Actual": actual_forward_prices,
        "Current_Price": current_prices_test,
        "Ridge": test_price_preds["Ridge"],
        "LightGBM": test_price_preds["LightGBM"],
        "KNN": test_price_preds["k-NN"],
        "SVR": test_price_preds["SVR"],
        "Ensemble": ensemble_test,
        "Model_Disagreement": disagree,
    })
    results_df.to_csv(PREDICTIONS_CSV, index=False)
    print(f"[Output] Saved: {PREDICTIONS_CSV} ({len(results_df)} rows)")

    # 10b: Save metrics
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)

    # 10c: Save ensemble weights
    weights_df = pd.DataFrame([weights])
    weights_df.to_csv(os.path.join(OUTPUT_DIR, "ensemble_weights.csv"), index=False)

    # 10d: Plots
    test_dates = test.index

    # Individual model plots
    for name in ["Ridge", "LightGBM", "k-NN", "SVR"]:
        plot_predictions(
            test_dates, actual_forward_prices,
            {name: test_price_preds[name]},
            f"{name}: Predicted vs Actual (3-Month Forward)",
            f"{name.lower().replace('-', '')}_predictions.png",
        )

    # Ensemble comparison plot
    plot_predictions(
        test_dates, actual_forward_prices, test_price_preds,
        "All Models + Ensemble: Predicted vs Actual (3-Month Forward)",
        "ensemble_comparison.png",
    )

    # Metrics bar chart
    plot_metrics_bar(metrics_df, "metrics_comparison.png")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE (Return-based prediction)")
    print(f"  Predictions: {PREDICTIONS_CSV}")
    print(f"  Plots:       {PLOT_DIR}/")
    print("=" * 70)

    return results_df, metrics_df, weights


if __name__ == "__main__":
    run_pipeline()
