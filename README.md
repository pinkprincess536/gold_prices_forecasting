
# Gold ML Forecasting

---

## Overview

This project presents a structured quantitative analysis of **Gold and Silver price behaviour** using historical time-series data.

It combines:

* Statistical price band analysis (Z-score & rolling volatility)
*  3-month forward ML forecasting (regime-aware)
* Interactive Streamlit dashboard
*  Risk-aware allocation insights

The objective is to move beyond narrative opinions and **quantify trend, volatility, deviation regimes, and forward risk** using statistically grounded methods.

The analysis is aligned with the article *“Noble Metals in Action”* and focuses on identifying:

* Long-term trends
* Volatility regimes
* Overheated vs corrected price zones
* Forward-looking risk structure (ML-based)

---

## Objectives

* Build a **regime-aware 3-month forward ML forecasting pipeline**

* Translate statistical outputs into **risk-aware investment observations**

---

## Data

* **Frequency:** Daily
* **Metals:** Gold, Silver
* **Currency:** INR
* **Time Period:** ~10+ years (2014–2026)
* **Source:** Publicly available historical data (Kaggle )

> All data handling, cleaning, feature engineering, and modeling steps are fully reproducible via the provided Python modules.

---

## Methodology

---

### 1. Return & Volatility Analysis

* Computed **daily returns**
* Calculated **rolling volatility (252-day window)** to represent annualised risk
* Compared volatility behaviour between Gold and Silver
* Identified high-volatility regimes and linked them to macro events (e.g., pandemic, liquidity stress, policy uncertainty)

**Deliverables:**

* Volatility comparison charts
* Data-backed commentary on regime behaviour

---

### 2. Trend & Price Band Construction

For each metal:

#### Trend Measure

* 200-day Moving Average (long-term structural trend proxy)

#### Price Bands

* Rolling standard deviation
* Z-score based deviation bands

[
Z = \frac{Price - MA_{200}}{STD_{200}}
]

#### Why Z-Score Bands?

* Normalises price deviation relative to volatility
* Enables meaningful cross-asset comparison
* Adjusts for regime shifts better than raw standard deviation
* Avoids scale distortion (₹30,000 vs ₹70,000 price levels)

#### Interpretation

* Positive Z-score → Price above trend
* Negative Z-score → Price below trend
* Higher |Z| → Stronger statistical deviation

---

### 3. Price Band Visualisation

Separate charts created for:

* **Gold**
* **Silver**

Each chart includes:

* Actual price
* 200-day moving average
* Upper & lower volatility bands
* Z-score interpretation
* Clear investor-readable labeling

The dashboard visually highlights whether price is:

* Above upper band
* Near upper band
* Within bands
* Below lower band

---

### 4. Regime-Aware ML Forecasting (3-Month Forward)

A modular ML pipeline forecasts **63 trading days (~3 months) ahead**.

#### Forecast Design

* Target: Forward log return
* Converted back to price after prediction
* Chronological train/validation/test split (70/15/15)
* No shuffling (prevents look-ahead bias)

#### Models Used

* Gaussian Hidden Markov Model (Regime detection)
* Ridge Regression (Linear baseline)
* LightGBM (Gradient boosting)
* k-Nearest Neighbors (Instance-based learning)
* Support Vector Regression (Kernel method)

#### Ensemble Blending

Models are combined using **Inverse-RMSE weighting**:

[
w_i = \frac{1/RMSE_i}{\sum (1/RMSE_j)}
]

This reduces correlated prediction errors and improves stability across volatility regimes.

Model disagreement is also tracked as a proxy for forecast uncertainty.

---

### 5. Interpretation & Insights

Using the latest available prices, the analysis answers:


* Does current behaviour align with elevated volatility regimes?
* Which metal is structurally more volatile?
* Which shows wider historical deviations from equilibrium?
* What does the 3-month ML forecast suggest about forward risk?

All insights are **data-backed and qualitative**, avoiding speculative claims.

---

### 6. Allocation Insight (Risk-Aware Observation)

Based on:

* Current Z-score positioning
* Volatility regime
* ML forecast structure
* Model disagreement

The analysis discusses whether the current environment suggests:

* Gradual accumulation
* Staggered allocation
* Waiting for mean reversion

> This is framed strictly as **risk-aware observation**, not investment advice.

---

## Interactive Dashboard

Built using Streamlit, the dashboard includes:

* Price band charts (Gold & Silver)
* Rolling volatility comparison
* Latest regime status
* SIP vs Lump Sum simulator
* Risk (drawdown) comparison
* Allocation insight panel


---

## Tools & Libraries

* Python
* pandas
* numpy
* scikit-learn
* lightgbm
* hmmlearn
* matplotlib
* plotly
* streamlit

---

## Key Takeaways


* Gold demonstrates smoother trend adherence with episodic spikes
* Z-score bands provide a disciplined deviation framework
* Regime-aware ML improves forward risk interpretation
* Ensemble modeling enhances forecast stability
* Price bands reduce emotional bias in allocation decisions

---

## Limitations

* Backward-looking analysis
* Limited macro variables included
* Fixed forecast horizon (63 days)
* No walk-forward cross-validation
* Extreme structural breaks may reduce model reliability

---



**Aswathi Pillai**

GitHub:
[https://github.com/pinkprincess536]
---

