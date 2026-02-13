"""
Streamlit dashboard for ML Gold Price Forecasting.

Run with:  streamlit run dashboard/ml_dashboard.py

Features:
- Model selection (multi-select)
- Predicted vs actual interactive chart
- Performance metrics cards with directional accuracy
- Date range filtering
- Model disagreement visualization
- Prediction error distribution
- Methodology explanation

"""

import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
METRICS_CSV = os.path.join(OUTPUT_DIR, "metrics.csv")
WEIGHTS_CSV = os.path.join(OUTPUT_DIR, "ensemble_weights.csv")

MODEL_COLORS = {
    "Ridge": "#e74c3c",
    "LightGBM": "#2ecc71",
    "KNN": "#3498db",
    "SVR": "#9b59b6",
    "Ensemble": "#f39c12",
}

MODEL_DISPLAY_NAMES = {
    "Ridge": "Ridge Regression",
    "LightGBM": "LightGBM",
    "KNN": "k-NN",
    "SVR": "SVR",
    "Ensemble": "Ensemble (Weighted)",
}

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gold Price ML Forecast — FinSharpe",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        color: white;
        border: 1px solid rgba(255,255,255,0.05);
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-card .model-name {
        font-size: 0.8em;
        color: #8892b0;
        margin: 0 0 6px 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .primary-metric {
        font-size: 1.6em;
        font-weight: 700;
        margin: 4px 0;
        background: linear-gradient(90deg, #f39c12, #e74c3c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card .secondary-metrics {
        font-size: 0.78em;
        color: #8892b0;
        margin: 6px 0 0 0;
        line-height: 1.6;
    }
    .best-badge {
        display: inline-block;
        background: rgba(46, 204, 113, 0.15);
        color: #2ecc71;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.7em;
        font-weight: 600;
        margin-left: 4px;
    }
    .methodology-box {
        background: rgba(26, 26, 46, 0.7);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Data Loading ────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    if not os.path.exists(PREDICTIONS_CSV):
        return None, None, None
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=["Date"])
    metrics = pd.read_csv(METRICS_CSV) if os.path.exists(METRICS_CSV) else None
    weights = pd.read_csv(WEIGHTS_CSV) if os.path.exists(WEIGHTS_CSV) else None
    return df, metrics, weights


df, metrics_df, weights_df = load_predictions()

if df is None:
    st.error("⚠️ No predictions found. Run the pipeline first: `python -m ml_models.pipeline`")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
st.sidebar.title("🔧 Controls")

model_options = [c for c in df.columns if c not in ["Date", "Actual", "Current_Price", "Model_Disagreement"]]
selected_models = st.sidebar.multiselect(
    "Select Models",
    model_options,
    default=model_options,  # Show all by default
)

# Date range
min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if len(date_range) == 2:
    mask = (df["Date"].dt.date >= date_range[0]) & (df["Date"].dt.date <= date_range[1])
    df_filtered = df[mask].copy()
else:
    df_filtered = df.copy()

# Ensemble weights in sidebar
st.sidebar.divider()
st.sidebar.markdown("**Ensemble Weights (Inverse-RMSE)**")
if weights_df is not None:
    for col in weights_df.columns:
        val = weights_df[col].iloc[0]
        st.sidebar.progress(val, text=f"{col}: {val:.1%}")

st.sidebar.divider()
st.sidebar.markdown("**Methodology**")
st.sidebar.info(
    "Models predict **63-day forward log returns**, "
    "then convert to prices via:\n\n"
    "`price_pred = current × exp(return_pred)`\n\n"
    "Returns are stationary — models generalise across all price regimes."
)

# ─── Header ──────────────────────────────────────────────────────────────────
st.title(" Gold Price 3-Month Forward ML Forecast")
st.markdown(
    f"**Test Period:** {min_date} → {max_date}  ·  "
    f"**Forecast Horizon:** 63 trading days (~3 months)  ·  "
    f"**Prediction Points:** {len(df_filtered)}"
)
st.divider()

# ─── Metrics Cards ───────────────────────────────────────────────────────────
if metrics_df is not None:
    st.subheader("Model Performance")

    # Find best values for highlighting
    best_mape = metrics_df["MAPE"].min()
    best_rmse = metrics_df["RMSE"].min()
    best_dir = metrics_df["Dir_Accuracy"].max()

    cols = st.columns(len(metrics_df))
    for i, row in metrics_df.iterrows():
        with cols[i]:
            is_best_mape = row["MAPE"] == best_mape
            is_best_dir = row["Dir_Accuracy"] == best_dir
            badge = ' <span class="best-badge">★ BEST</span>' if is_best_mape else ""
            dir_badge = ' <span class="best-badge">★ BEST</span>' if is_best_dir else ""

            display_name = MODEL_DISPLAY_NAMES.get(row["Model"], row["Model"])
            st.markdown(f"""
            <div class="metric-card">
                <div class="model-name">{display_name}</div>
                <div class="primary-metric">MAPE: {row['MAPE']:.2f}%{badge}</div>
                <div class="secondary-metrics">
                    RMSE: ₹{row['RMSE']:,.0f}<br>
                    MAE: ₹{row['MAE']:,.0f}<br>
                    Direction: {row['Dir_Accuracy']:.1f}%{dir_badge}
                </div>
            </div>
            """, unsafe_allow_html=True)
    st.divider()

# ─── Main Prediction Chart ───────────────────────────────────────────────────
st.subheader("Predictions vs Actual")

fig = go.Figure()

# Actual price line
fig.add_trace(go.Scatter(
    x=df_filtered["Date"], y=df_filtered["Actual"],
    name="Actual (3M Forward)",
    line=dict(color="white", width=2.5),
    hovertemplate="Actual: ₹%{y:,.0f}<extra></extra>",
))

# Model predictions
for model in selected_models:
    if model in df_filtered.columns:
        display_name = MODEL_DISPLAY_NAMES.get(model, model)
        fig.add_trace(go.Scatter(
            x=df_filtered["Date"], y=df_filtered[model],
            name=display_name,
            line=dict(color=MODEL_COLORS.get(model, "#888"), width=1.5),
            hovertemplate=f"{display_name}: " + "₹%{y:,.0f}<extra></extra>",
        ))

# Current price for context
fig.add_trace(go.Scatter(
    x=df_filtered["Date"], y=df_filtered["Current_Price"],
    name="Current Price (at prediction time)",
    line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
    hovertemplate="Current: ₹%{y:,.0f}<extra></extra>",
))

fig.update_layout(
    template="plotly_dark",
    height=550,
    yaxis_title="Gold Price (₹/10g)",
    xaxis_title="Date",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=30, b=40),
    plot_bgcolor="rgba(10,10,26,0.8)",
)
st.plotly_chart(fig, use_container_width=True)

# ─── Two-column layout: Disagreement + Error Distribution ───────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    # Model Disagreement
    if "Model_Disagreement" in df_filtered.columns:
        st.subheader("Model Disagreement (Uncertainty)")
        fig_dis = go.Figure()

        # Color gradient based on disagreement level
        fig_dis.add_trace(go.Scatter(
            x=df_filtered["Date"], y=df_filtered["Model_Disagreement"],
            fill="tozeroy", fillcolor="rgba(231, 76, 60, 0.15)",
            line=dict(color="#e74c3c", width=1.5),
            name="Std Dev Across Models",
            hovertemplate="Disagreement: ₹%{y:,.0f}<extra></extra>",
        ))

        avg_disagree = df_filtered["Model_Disagreement"].mean()
        fig_dis.add_hline(
            y=avg_disagree,
            line_dash="dash", line_color="rgba(255,255,255,0.3)",
            annotation_text=f"Avg: ₹{avg_disagree:,.0f}",
            annotation_position="top right",
            annotation_font_color="rgba(255,255,255,0.5)",
        )

        fig_dis.update_layout(
            template="plotly_dark",
            height=350,
            yaxis_title="Disagreement (₹)",
            xaxis_title="Date",
            margin=dict(l=40, r=40, t=20, b=40),
            plot_bgcolor="rgba(10,10,26,0.8)",
        )
        st.plotly_chart(fig_dis, use_container_width=True)
        st.caption("⚠️ High disagreement = regime transitions or structural breaks. Use predictions with lower confidence in these periods.")

with col_right:
    # Prediction Error Boxplot
    st.subheader("Error Distribution")
    error_data = []
    for model in selected_models:
        if model in df_filtered.columns:
            errors = df_filtered["Actual"] - df_filtered[model]
            display_name = MODEL_DISPLAY_NAMES.get(model, model)
            for e in errors:
                error_data.append({"Model": display_name, "Error (₹)": e})

    if error_data:
        err_df = pd.DataFrame(error_data)
        fig_box = px.box(
            err_df, x="Model", y="Error (₹)",
            color="Model",
            color_discrete_sequence=["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12"],
        )
        fig_box.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig_box.update_layout(
            template="plotly_dark",
            height=350,
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
            plot_bgcolor="rgba(10,10,26,0.8)",
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Centered on 0 = no bias. Tight boxes = consistent accuracy.")

# ─── Detailed Metrics Table ──────────────────────────────────────────────────
if metrics_df is not None:
    st.divider()
    st.subheader("📊 Detailed Metrics Table")
    display_metrics = metrics_df.copy()
    display_metrics["Model"] = display_metrics["Model"].map(
        lambda x: MODEL_DISPLAY_NAMES.get(x, x)
    )
    styled = display_metrics.style.format({
        "MAE": "₹{:,.0f}",
        "RMSE": "₹{:,.0f}",
        "MAPE": "{:.2f}%",
        "Dir_Accuracy": "{:.1f}%",
    }).highlight_min(
        subset=["MAE", "RMSE", "MAPE"], color="#1a472a"
    ).highlight_max(
        subset=["Dir_Accuracy"], color="#1a472a"
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

# ─── Methodology Section ────────────────────────────────────────────────────
st.divider()
with st.expander("📖 Methodology & Model Details", expanded=False):
    st.markdown("""
    ### Return-Based Prediction Framework

    Models predict **63-day forward log returns** rather than raw prices:
    - `target = log(price[t+63] / price[t])`
    - `predicted_price = current_price × exp(predicted_return)`

    **Why?** Returns are stationary — a 5% return looks the same whether gold is at ₹30k or ₹70k.
    Tree-based and instance-based models **cannot extrapolate** beyond training prices, but they can
    predict returns that translate to any price level.

    ---

    ### Models

    | Model | Type | Key Strength |
    |-------|------|--------------|
    | **Ridge Regression** | Linear (L2-regularised) | Extrapolation, interpretability |
    | **LightGBM** | Gradient-boosted trees | Non-linear interactions, feature importance |
    | **k-NN** | Instance-based | No assumptions, local pattern matching |
    | **SVR** | Kernel-based regression | Non-linear with regularisation |
    | **Ensemble** | Inverse-RMSE weighted blend | Combines strengths, reduces variance |

    ### Ensemble Weighting

    Each model's ensemble weight is proportional to `1 / validation_RMSE`. Models with lower
    validation error get higher weights. This is a soft selection that keeps all models in play
    while favouring the most accurate ones.

    ### 36 Engineered Features

    Returns (1/5/21/63d), rolling volatility (21/63/252d), SMAs (20/50/200d), RSI, MACD,
    z-scores, price-to-SMA ratios, gold-silver ratio, calendar features, and HMM regime labels.
    """)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
