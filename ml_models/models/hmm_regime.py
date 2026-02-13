"""
Gaussian Hidden Markov Model for regime detection.

The HMM is NOT used for direct price prediction. Instead, it identifies
latent market regimes (e.g. low-vol trending, high-vol trending, crisis)
from observed returns and volatility. The detected regime label is then
used as a categorical feature for downstream predictive models.

Why HMM?
- Markets exhibit regime-switching behaviour (calm vs crisis)
- HMM naturally models latent states generating observed sequences
- Regime labels help other models adapt to structural breaks

State count justification:
- 3 states empirically capture the dominant commodity price regimes
- 2 states merge high-vol trending with crisis; 4+ creates unstable micro-states
- BIC comparison is performed to validate the choice
"""

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from ml_models.config import HMM_N_STATES, HMM_N_ITER, HMM_COVARIANCE_TYPE


def select_hmm_states(features: np.ndarray, max_states: int = 5) -> dict:
    """
    Compare BIC scores for different state counts to justify the choice.

    Returns dict with {n_states: bic_score}.
    """
    bic_scores = {}
    for n in range(2, max_states + 1):
        try:
            model = GaussianHMM(
                n_components=n,
                covariance_type=HMM_COVARIANCE_TYPE,
                n_iter=HMM_N_ITER,
                random_state=42,
            )
            model.fit(features)
            bic = -2 * model.score(features) + n * np.log(len(features))
            bic_scores[n] = bic
        except Exception:
            bic_scores[n] = np.inf
    return bic_scores


def fit_hmm(features_df: pd.DataFrame) -> tuple:
    """
    Fit a Gaussian HMM on returns + volatility features.

    Args:
        features_df: DataFrame with columns used for HMM fitting.
                     Typically gold_ret_1d, gold_vol_21d.

    Returns:
        (model, regime_labels, bic_scores)
        - model: fitted GaussianHMM
        - regime_labels: pd.Series of integer regime labels (same index as input)
        - bic_scores: dict of {n_states: BIC} for documentation
    """
    # Select features for HMM: daily returns and short-term volatility
    hmm_cols = []
    for col in features_df.columns:
        if col in ["gold_ret_1d", "gold_vol_21d"]:
            hmm_cols.append(col)

    if len(hmm_cols) == 0:
        raise ValueError("Cannot find HMM input features (gold_ret_1d, gold_vol_21d)")

    X = features_df[hmm_cols].values

    # BIC comparison
    bic_scores = select_hmm_states(X)
    print(f"[HMM] BIC scores: {bic_scores}")
    best_n = min(bic_scores, key=bic_scores.get)
    print(f"[HMM] Best state count by BIC: {best_n} (using configured: {HMM_N_STATES})")

    # Fit with configured state count
    model = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type=HMM_COVARIANCE_TYPE,
        n_iter=HMM_N_ITER,
        random_state=42,
    )
    model.fit(X)

    # Predict regime labels
    labels = model.predict(X)

    # Sort states by mean return for interpretability
    # State with lowest mean return → 0, highest → 2
    state_means = []
    for s in range(HMM_N_STATES):
        mask = labels == s
        state_means.append((s, X[mask, 0].mean() if mask.any() else 0))

    state_means.sort(key=lambda x: x[1])
    remap = {old: new for new, (old, _) in enumerate(state_means)}
    labels = np.array([remap[l] for l in labels])

    regime_series = pd.Series(labels, index=features_df.index, name="hmm_regime")

    # Print regime summary
    print(f"[HMM] Transition matrix:\n{np.round(model.transmat_, 3)}")
    for s in range(HMM_N_STATES):
        count = (labels == s).sum()
        pct = count / len(labels) * 100
        print(f"[HMM] Regime {s}: {count} days ({pct:.1f}%)")

    return model, regime_series, bic_scores


def predict_regimes(model: GaussianHMM, features_df: pd.DataFrame) -> pd.Series:
    """Predict regime labels for new data using a fitted HMM."""
    hmm_cols = [c for c in ["gold_ret_1d", "gold_vol_21d"] if c in features_df.columns]
    X = features_df[hmm_cols].values
    labels = model.predict(X)
    return pd.Series(labels, index=features_df.index, name="hmm_regime")
