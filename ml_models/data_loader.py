"""
Data loader for the ML forecasting pipeline.

Responsibilities:
- Load gold.csv and silver_new.csv
- Parse dates, sort chronologically, clean numeric values
- Align date ranges between gold and silver
- Handle missing values (forward-fill for minor gaps)
"""

import pandas as pd
import numpy as np
from ml_models.config import GOLD_CSV, SILVER_CSV


def load_gold(path: str = GOLD_CSV) -> pd.DataFrame:
    """Load and clean gold price data."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Rename 'Price' → 'Close' for consistency
    df.rename(columns={"Price": "Close"}, inplace=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # Clean OHLV columns
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(
            df["Volume"].astype(str).str.replace(",", ""), errors="coerce"
        )

    return df


def load_silver(path: str = SILVER_CSV) -> pd.DataFrame:
    """Load and clean silver_new price data."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y")
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # Clean numeric columns (silver has commas and K suffixes in Volume)
    df["Close"] = pd.to_numeric(
        df["Close"].astype(str).str.replace(",", ""), errors="coerce"
    )
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(",", ""), errors="coerce"
            )

    if "Volume" in df.columns:
        vol = df["Volume"].astype(str)
        # Handle K (thousands) suffix
        mask_k = vol.str.upper().str.endswith("K")
        vol_clean = vol.str.upper().str.replace("K", "").str.replace(",", "")
        vol_numeric = pd.to_numeric(vol_clean, errors="coerce")
        vol_numeric[mask_k] = vol_numeric[mask_k] * 1000
        df["Volume"] = vol_numeric

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load, clean, and align gold and silver data.

    Returns:
        (gold_df, silver_df) — both with DatetimeIndex, sorted ascending,
        aligned to overlapping date range, minor gaps forward-filled.
    """
    gold = load_gold()
    silver = load_silver()

    # Align to common date range
    start = max(gold.index.min(), silver.index.min())
    end = min(gold.index.max(), silver.index.max())
    gold = gold.loc[start:end].copy()
    silver = silver.loc[start:end].copy()

    # Forward-fill minor gaps (e.g. different holiday calendars)
    gold["Close"] = gold["Close"].ffill()
    silver["Close"] = silver["Close"].ffill()

    # Drop rows where Close is still NaN (beginning of series)
    gold.dropna(subset=["Close"], inplace=True)
    silver.dropna(subset=["Close"], inplace=True)

    print(f"[DataLoader] Gold:   {gold.shape[0]} rows, {gold.index.min().date()} → {gold.index.max().date()}")
    print(f"[DataLoader] Silver: {silver.shape[0]} rows, {silver.index.min().date()} → {silver.index.max().date()}")

    return gold, silver
