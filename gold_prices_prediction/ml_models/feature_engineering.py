"""
Feature engineering for gold price forecasting.

All features are computed from PAST data only to avoid look-ahead bias.
Features capture: momentum, volatility regimes, trend, mean-reversion,
cross-asset dynamics, and seasonality.
"""

import pandas as pd
import numpy as np
from ml_models.config import (
    RETURN_PERIODS,
    VOLATILITY_WINDOWS,
    SMA_WINDOWS,
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    ZSCORE_WINDOW,
    FORECAST_HORIZON,
)


def compute_returns(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold") -> pd.DataFrame:
    """
    Log returns at multiple horizons.

    Log returns are preferred over simple returns for:
    - Additivity over time (log(P_t/P_0) = sum of log returns)
    - Better normality approximation
    - Symmetric treatment of gains/losses
    """
    for period in RETURN_PERIODS:
        df[f"{prefix}_ret_{period}d"] = np.log(df[price_col] / df[price_col].shift(period))
    return df


def compute_rolling_volatility(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold") -> pd.DataFrame:
    """
    Annualised rolling volatility (standard deviation of daily log returns × √252).

    Multiple windows capture:
    - 21d: Short-term/intramonth volatility regime
    - 63d: Quarterly volatility
    - 252d: Annual baseline volatility
    """
    daily_ret = np.log(df[price_col] / df[price_col].shift(1))
    for window in VOLATILITY_WINDOWS:
        df[f"{prefix}_vol_{window}d"] = daily_ret.rolling(window).std() * np.sqrt(252)
    return df


def compute_sma(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold") -> pd.DataFrame:
    """
    Simple Moving Averages and price-to-MA ratios.

    Ratios normalise the relationship, making it comparable across price levels.
    """
    for window in SMA_WINDOWS:
        ma_col = f"{prefix}_sma_{window}d"
        df[ma_col] = df[price_col].rolling(window).mean()
        # Price relative to its MA (>1 = above trend, <1 = below)
        df[f"{prefix}_price_to_sma_{window}d"] = df[price_col] / df[ma_col]
    return df


def compute_rsi(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold", period: int = RSI_PERIOD) -> pd.DataFrame:
    """
    Relative Strength Index (RSI).

    RSI = 100 - 100/(1 + RS), where RS = avg_gain / avg_loss over `period` days.
    Measures momentum exhaustion: >70 overbought, <30 oversold.
    """
    delta = df[price_col].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"{prefix}_rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def compute_macd(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold") -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence).

    MACD line = EMA(fast) - EMA(slow)
    Signal line = EMA of MACD line
    Histogram = MACD - Signal (momentum strength)
    """
    ema_fast = df[price_col].ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = df[price_col].ewm(span=MACD_SLOW, adjust=False).mean()

    df[f"{prefix}_macd"] = ema_fast - ema_slow
    df[f"{prefix}_macd_signal"] = df[f"{prefix}_macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df[f"{prefix}_macd_hist"] = df[f"{prefix}_macd"] - df[f"{prefix}_macd_signal"]
    return df


def compute_zscore(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold") -> pd.DataFrame:
    """
    Rolling Z-score: how many standard deviations price is from its rolling mean.

    Positive = above trend, negative = below. Useful for mean-reversion signals.
    """
    rolling_mean = df[price_col].rolling(ZSCORE_WINDOW).mean()
    rolling_std = df[price_col].rolling(ZSCORE_WINDOW).std()
    df[f"{prefix}_zscore_{ZSCORE_WINDOW}d"] = (df[price_col] - rolling_mean) / rolling_std.replace(0, np.nan)
    return df


def compute_rate_of_change(df: pd.DataFrame, price_col: str = "Close", prefix: str = "gold") -> pd.DataFrame:
    """
    Rate of Change (ROC): percentage change over a look-back period.
    21d ROC captures monthly momentum.
    """
    for period in [21, 63]:
        df[f"{prefix}_roc_{period}d"] = (df[price_col] / df[price_col].shift(period) - 1) * 100
    return df


def compute_volume_features(df: pd.DataFrame, prefix: str = "gold") -> pd.DataFrame:
    """Volume features: normalised volume and volume-to-MA ratio."""
    if "Volume" not in df.columns or df["Volume"].isna().all():
        return df

    df[f"{prefix}_vol_ma_20"] = df["Volume"].rolling(20).mean()
    df[f"{prefix}_vol_ratio"] = df["Volume"] / df[f"{prefix}_vol_ma_20"].replace(0, np.nan)
    return df


def compute_gold_silver_ratio(gold: pd.DataFrame, silver: pd.DataFrame) -> pd.Series:
    """
    Gold/Silver price ratio — a classic cross-asset relative value indicator.

    High ratio → gold expensive relative to silver (risk-off regime)
    Low ratio → silver expensive relative to gold (risk-on regime)
    """
    # Align on common dates
    common_idx = gold.index.intersection(silver.index)
    ratio = gold.loc[common_idx, "Close"] / silver.loc[common_idx, "Close"]
    return ratio


def build_features(gold: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from gold and silver price data.

    Returns a DataFrame indexed by Date with all engineered features.
    The target variable (forward price) is NOT added here — see pipeline.py.
    """
    # Work on copies
    gf = gold[["Close"]].copy()
    if "Volume" in gold.columns:
        gf["Volume"] = gold["Volume"]

    sf = silver[["Close"]].copy()

    # ── Gold features ─────────────────────────────────────────────────────
    gf = compute_returns(gf, "Close", "gold")
    gf = compute_rolling_volatility(gf, "Close", "gold")
    gf = compute_sma(gf, "Close", "gold")
    gf = compute_rsi(gf, "Close", "gold")
    gf = compute_macd(gf, "Close", "gold")
    gf = compute_zscore(gf, "Close", "gold")
    gf = compute_rate_of_change(gf, "Close", "gold")
    gf = compute_volume_features(gf, "gold")

    # ── Silver features (subset — cross-asset signal) ─────────────────────
    sf = compute_returns(sf, "Close", "silver")
    sf = compute_rolling_volatility(sf, "Close", "silver")
    sf = compute_rsi(sf, "Close", "silver")

    # ── Gold-Silver ratio ─────────────────────────────────────────────────
    ratio = compute_gold_silver_ratio(gold, silver)
    gf["gs_ratio"] = ratio
    gf["gs_ratio_ma_50"] = gf["gs_ratio"].rolling(50).mean()
    gf["gs_ratio_zscore"] = (
        (gf["gs_ratio"] - gf["gs_ratio"].rolling(200).mean())
        / gf["gs_ratio"].rolling(200).std().replace(0, np.nan)
    )

    # ── Merge silver features into gold frame ────────────────────────────
    silver_feat_cols = [c for c in sf.columns if c.startswith("silver_")]
    gf = gf.join(sf[silver_feat_cols], how="left")

    # ── Calendar features ────────────────────────────────────────────────
    gf["day_of_week"] = gf.index.dayofweek
    gf["month"] = gf.index.month

    # ── Drop intermediate columns not needed as features ─────────────────
    drop_cols = ["Volume"]
    gf.drop(columns=[c for c in drop_cols if c in gf.columns], inplace=True)

    # Keep gold Close for target construction (will be dropped before training)
    gf.rename(columns={"Close": "gold_close"}, inplace=True)

    # ── Drop rows with NaN from rolling windows ──────────────────────────
    initial_len = len(gf)
    gf.dropna(inplace=True)
    print(f"[Features] Built {len(gf.columns)} features, {initial_len} → {len(gf)} rows after dropping NaN warmup period")

    return gf
