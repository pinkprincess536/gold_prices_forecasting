# Gold Price 3-Month Forward ML Forecasting — Documentation

## 1. Overview

This pipeline forecasts Indian gold prices (₹/10g) **63 trading days (~3 months) ahead** using five complementary machine learning approaches. It integrates with the existing Gold-Silver price band analysis repository without modifying any original code.



---

## 2. Data Description

| Dataset | File | Rows | Period | Price Unit | Key Column |
|---------|------|------|--------|------------|------------|
| Gold | `data/gold.csv` | ~3,104 | Jan 2014 – Feb 2026 | ₹/10g | `Price` |
| Silver | `data/silver_new.csv` | ~3,120 | Jan 2014 – Feb 2026 | ₹/kg | `Close` |

### Cleaning Steps
1. **Date parsing:** Gold uses `YYYY-MM-DD`; Silver uses `DD-MM-YYYY` — both converted to DatetimeIndex.
2. **Sort ascending:** Raw CSVs are reverse-chronological (most recent first); sorted to chronological order.
3. **Numeric cleaning:** Silver's `Close` and `Volume` contain commas and "K" suffixes — parsed programmatically.
4. **Alignment:** Both datasets trimmed to overlapping date range.
5. **Missing values:** Minor gaps (different holiday calendars) forward-filled. Rows with NaN `Close` dropped.

---

## 3. Feature Engineering Rationale

All features are computed from **past data only** to prevent look-ahead bias. Each feature group serves a specific analytical purpose:

### Returns (Momentum at Multiple Horizons)
- **1d, 5d, 21d, 63d log returns** — capture momentum from daily to quarterly.
- Log returns are used instead of simple returns for time-additivity and better normality.

### Rolling Volatility (Regime Detection)
- **21d, 63d, 252d annualised volatility** — captures volatility clustering.
- Gold exhibits clear volatility regimes: calm trending vs. crisis spikes.
- Multiple windows distinguish transient spikes from structural regime shifts.

### Moving Averages (Trend)
- **20d, 50d, 200d SMA** — short, medium, long-term trend.
- **Price-to-SMA ratios** normalise the signal across different price levels — a ratio of 1.05 means "5% above trend" regardless of whether gold is at ₹30,000 or ₹130,000.

### Momentum Indicators
- **RSI-14:** Measures trend exhaustion. >70 overbought, <30 oversold. Useful for mean-reversion timing.
- **MACD (12/26/9):** Captures momentum direction changes. The histogram (MACD - Signal) is the most actionable component.
- **Rate of Change (21d, 63d):** Simpler momentum measure — complements RSI's bounded [0, 100] range.

### Z-Score (Mean Reversion)
- **200d rolling z-score:** How many standard deviations price is from its long-term mean.
- Directly relevant to the existing price band analysis in this repo.

### Gold-Silver Ratio (Cross-Asset)
- **Gold/Silver price ratio + its 50d MA + 200d z-score**
- High ratio = risk-off (gold outperforming) → signals defensive positioning.
- Low ratio = risk-on → silver outperforming.
- This ratio is a well-known macro indicator in commodity markets.

### Silver Features (Cross-Asset Volatility)
- **Silver returns (1d, 5d, 21d, 63d)** — silver often leads gold in both directions.
- **Silver volatility (21d, 63d, 252d)** — silver volatility spikes sometimes precede gold moves.
- **Silver RSI** — cross-asset momentum divergence is informative.

### Calendar Features
- **Day of week, month** — captures settlement effects and seasonal patterns.

### Volume Features
- **Volume-to-MA ratio** — unusual volume relative to 20d average signals institutional activity.

### HMM Regime Labels
- **Integer regime label (0, 1, 2)** from Gaussian HMM — captures latent market state.
- Used as a categorical feature by downstream models.

---

## 4. Target Variable

**Target:** Gold closing price **63 trading days forward** (≈ 3 calendar months).

```
target[t] = gold_close[t + 63]
```

The last 63 rows of the dataset have no target and are excluded. This is a direct price prediction (not a return prediction) because the downstream application is "what will gold cost in 3 months?"

---

## 5. Train / Validation / Test Split

**Chronological split** — no shuffling, no random CV (these would cause data leakage in time series):

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model fitting |
| Validation | 15% | Ensemble weight calibration, early stopping (LightGBM) |
| Test | 15% | Final evaluation — models never see this data |

Approximate date ranges depend on available data after feature warmup (~252 rows lost to 252d rolling windows).

---

## 6. Model Descriptions

### 6.1 Gaussian HMM (Regime Detection)

**Role:** NOT a direct price predictor. Detects latent market regimes from observed returns and volatility. Regime labels become features for other models.

| Parameter | Value | Justification |
|-----------|-------|---------------|
| States | 3 | low-vol trend / high-vol trend / crisis. Validated by BIC comparison (2–5 states). |
| Covariance | Full | Captures correlations between return and volatility features. |
| Iterations | 100 | Sufficient for EM convergence. |
| Input features | 1d return, 21d volatility | Minimal, robust inputs. |

**Pros:** Captures regime-switching behaviour that other models miss. Transition matrix reveals regime .
Example: “What environment are we in right now?”

**Cons:** 
-Assumes Gaussian emissions (financial returns have fat tails). 
-State assignments can be unstable near regime boundaries.
-HMM underestimates rare shocks.

### 6.2 Ridge Regression (Linear Baseline)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Alpha | 1.0 | Moderate L2 penalty. With 30+ correlated features, regularisation prevents multicollinearity-driven coefficient instability. |
| Scaling | StandardScaler | Required — Ridge is sensitive to feature magnitudes. |

**Pros:** Fast, interpretable coefficients, serves as the baseline. If non-linear models can't beat Ridge, the problem is dominated by linear relationships.

**Cons:** Cannot capture interactions or non-linear patterns. Ridge can only draw a straight line.Ridge assumes each feature affects price independently.

### 6.3 LightGBM (Gradient-Boosted Trees)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| n_estimators | 500 | Upper bound; actual trees determined by early stopping. |
| learning_rate | 0.05 | Slower learning → better generalisation. |
| max_depth | 6 | Moderate depth for ~2,000 training samples. Deeper → overfitting. |
| subsample | 0.8 | Row sampling reduces variance. |
| colsample_bytree | 0.8 | Feature sampling decorrelates trees. |
| min_child_samples | 20 | Prevents noisy leaf nodes with few observations. |
| Early stopping | 50 rounds | Stops training when validation RMSE plateaus. |

**Pros:** Captures non-linear feature interactions.
Doesn’t assume the relationship is straight-line.
It learns situations. Fast training. Handles mixed feature types natively (numeric + regime labels).

**Cons:** Can overfit on small financial datasets if not carefully regularised. Does not extrapolate beyond training range.This is a real issue during market crashes.

**Time-series safety:** No random shuffling. Validation set is chronologically after training set.

### 6.4 k-Nearest Neighbors (Instance-Based)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| k | 10 | sqrt(N) ≈ 45 over-smooths. k=10 balances bias/variance for ~2,000 samples. |
| weights | distance | Closer historical analogues contribute more to the prediction. |
| Scaling | StandardScaler | Essential — k-NN uses Euclidean distance. |

**Pros:** Non-parametric — no assumptions about functional form. Captures "what happened when markets looked like this before?" and historical analogy prediction.

**Cons:** Curse of dimensionality — with 30+ features, Euclidean distance becomes less meaningful. Cannot extrapolate (if gold breaks all-time highs, there are no historical analogues). Slow at prediction time (must scan all training points).

### 6.5 Support Vector Regression (Kernel Method)

| Parameter | Value | Justification |
|-----------|-------|---------------|
| C | 10.0 | Moderate regularisation. C=1 underfits; C=1000 overfits financial noise. |
| epsilon | 0.1 | Standard insensitive tube width — small errors are tolerated (noise-robust). |
| Target scaling | Yes | SVR convergence improves with standardised targets. |

**Pros:** Finds a global smooth function (unlike trees which partition). Epsilon-insensitive loss is naturally robust to financial noise.SVR’s entire purpose is to not get distracted by noise.

**Cons:**  slowest model in the pipeline.It compares many points with each other.Single hyperparameter set across all regimes.

---

## 7. Ensemble / Blending

### Strategy: Inverse-RMSE Weighted Average

Each model's weight is proportional to `1/RMSE` on the **validation set**:

```
weight_i = (1/RMSE_i) / Σ(1/RMSE_j)
```

### Why Blend?

1. **Error diversity:** Linear (Ridge), tree (LightGBM), instance-based (k-NN), and kernel (SVR) models make different errors. Averaging reduces correlated prediction errors.

2. **Stability:** Individual models may overfit to specific regimes. The ensemble provides more stable predictions across regime transitions.

3. **Model disagreement is informative:** When models strongly disagree (high standard deviation across predictions), it signals uncertainty — possibly a regime transition. The dashboard displays this metric.

### When Ensembles Improve Stability

Ensembles work best when constituent models are:
- Individually competent (each model has reasonable accuracy)
- Diverse (they make different errors)

In our case, model diversity is high by construction: we use fundamentally different learning paradigms.

---

## 8. Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **MAE** | Average absolute error in ₹. Interpretable: "on average, the prediction is ₹X off." |
| **RMSE** | Root mean squared error. Penalises large errors more than MAE. Used for ensemble weights. |
| **MAPE** | Percentage error. Scale-independent — allows comparison across different price levels. |
| **Directional Accuracy** | % of time the model correctly predicts whether price goes up or down. Important for trading decisions. |

---

## 9. Limitations & Assumptions

### Assumptions

- **Stationarity of relationships:** Feature-target relationships learned from history will persist. This is often violated during structural breaks (pandemics, policy changes).

- **Linear time structure:** We use a fixed 63-day horizon. In reality, "3 months forward" depends on market microstructure.

- **Gaussian HMM emissions:** Financial returns have fat tails; Gaussian assumption underestimates extreme regime transitions.

### Limitations

- **No exogenous features:** The pipeline uses only price/volume data. Macro variables (interest rates, USD/INR, crude oil, geopolitical risk) are excluded for simplicity but would improve forecasts.

- **No transaction cost modeling:** Predictions don't account for trading friction.

- **Small dataset:** ~2,000 usable training samples (after feature warmup) is small for ML. Overfitting risk is real, especially for LightGBM and k-NN.

- **No walk-forward validation:** A single train/val/test split is used. Production systems should use expanding or sliding window cross-validation.

- **No uncertainty quantification:** The ensemble provides model disagreement as a proxy, but formal prediction intervals are not implemented.

- **Extrapolation risk:** If gold prices enter unprecedented territory, all models (especially k-NN and LightGBM) will struggle.

---

## 10. How to Run

```bash

python -m venv .venv
.\.venv\Scripts\activate (windows)
or
source venv/bin/activate (macOS / linux)

pip install -r requirements.txt

python -m ml_models.pipeline

streamlit run dashboard/ml_dashboard.py
```

---

## 11. Output Files

| File | Description |
|------|-------------|
| `outputs/predictions.csv` | Date, Actual, per-model predictions, Ensemble, Disagreement |
| `outputs/metrics.csv` | MAE, RMSE, MAPE, Directional Accuracy per model |
| `outputs/ensemble_weights.csv` | Inverse-RMSE ensemble weights |
| `outputs/plots/regime_visualization.png` | HMM regime overlay on gold price |
| `outputs/plots/*_predictions.png` | Individual model prediction plots |
| `outputs/plots/ensemble_comparison.png` | All models + ensemble comparison |
| `outputs/plots/metrics_comparison.png` | Bar chart of model metrics |
