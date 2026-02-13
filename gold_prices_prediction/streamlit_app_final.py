import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from babel.numbers import format_currency


def format_inr(amount, decimals: int = 2) -> str:
    """
    Format a numeric value using the Indian numbering system (lakhs, crores).
    Example: 1234567.89 -> ₹12,34,567.89
    """
    if pd.isna(amount):
        return "N/A"
    pattern = "¤#,##,##0" + (("." + ("0" * decimals)) if decimals > 0 else "")
    return format_currency(amount, "INR", locale="en_IN", format=pattern)


# -----------------------------------------------------------------------------
# Configuration & Style
# -----------------------------------------------------------------------------



st.set_page_config(
    page_title="Gold & Silver Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a cleaner look
st.markdown("""
<style>
    .metric-card {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .metric-label {
        font-size: 0.9em;
        color: #666;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.6em;
        font-weight: bold;
        color: #333;
    }
    .status-text {
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
        display: inline-block;
        margin-top: 10px;
    }
    .status-Above { background-color: #ffebee; color: #c62828; }
    .status-Near { background-color: #fff8e1; color: #f57f17; }
    .status-Within { background-color: #e8f5e9; color: #2e7d32; }
    .status-Below { background-color: #e3f2fd; color: #1565c0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data():
    """
    Loads Gold and Silver data from CSVs.
    Returns:
        gold (full), silver (full) -> For Price Bands (Task 2/3)
        gold_aligned, silver_aligned -> For Volatility Analysis (Task 1)
    """
    # Load Data
    try:
        gold = pd.read_csv("data/gold.csv")
        silver = pd.read_csv("data/silver_new.csv")
    except FileNotFoundError:
        st.error("⚠️ Data files not found! Please ensure 'data/gold.csv' and 'data/silver.csv' exist.")
        st.stop()

    # --- Processing Steps ---
    
    # Date Parsing & Indexing
    gold['Date'] = pd.to_datetime(gold['Date'])
    silver['Date'] = pd.to_datetime(silver['Date'], format='%d-%m-%Y')

    gold.set_index('Date', inplace=True)
    silver.set_index('Date', inplace=True)

    gold = gold.sort_index()
    silver = silver.sort_index()

    # Rename
    gold.rename(columns={'Price': 'Close'}, inplace=True)
    silver.rename(columns={'Price': 'Close'}, inplace=True)

    # Numeric Conversion
    silver['Close'] = pd.to_numeric(silver['Close'].astype(str).str.replace(',', ''), errors='coerce')
    gold['Close'] = pd.to_numeric(gold['Close'], errors='coerce')

    # ----------------------------------------
    # Task 1: Volatility Analysis (Requires Alignment)
    # ----------------------------------------
    start_date = max(gold.index.min(), silver.index.min())
    end_date = min(gold.index.max(), silver.index.max())
    
    gold_aligned = gold.loc[start_date:end_date].copy()
    silver_aligned = silver.loc[start_date:end_date].copy()

    gold_aligned['returns'] = gold_aligned['Close'].pct_change()
    silver_aligned['returns'] = silver_aligned['Close'].pct_change()

    window_vol = 252
    gold_aligned['volatility'] = gold_aligned['returns'].rolling(window_vol).std() * np.sqrt(252)
    silver_aligned['volatility'] = silver_aligned['returns'].rolling(window_vol).std() * np.sqrt(252)

    # ----------------------------------------
    # Task 2 & 3: Price Band Construction (Uses FULL Data)
    # ----------------------------------------
    window_ma = 200
    
    # Calculate bands on the FULL datasets (gold, silver) not the aligned ones
    for df in [gold, silver]:
        df['MA_200'] = df['Close'].rolling(window_ma).mean()
        df['STD_200'] = df['Close'].rolling(window_ma).std()
        
        df['Upper_Band'] = df['MA_200'] + (2 * df['STD_200'])
        df['Lower_Band'] = df['MA_200'] - (2 * df['STD_200'])
        
        df['z_score'] = (df['Close'] - df['MA_200']) / df['STD_200']
        
        # Also calculate volatility on full df for the metrics to be available if needed
        df['returns'] = df['Close'].pct_change()
        df['volatility'] = df['returns'].rolling(window_vol).std() * np.sqrt(252)

    return gold, silver, gold_aligned, silver_aligned

def determine_band_position(current_price, upper, lower):
    """Determines the textual description of price position relative to bands."""
    if pd.isna(current_price) or pd.isna(upper) or pd.isna(lower):
        return "N/A", "Within"
    
    if current_price > upper:
        return "Above Upper Band", "Above"
    elif current_price >= (upper * 0.98): # Within 2% of upper band
        return "Near Upper Band", "Near"
    elif current_price < lower:
        return "Below Lower Band", "Below"
    else:
        return "Within Bands", "Within"

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def plot_price_bands(df, asset_name, color_line):
    """Generates an interactive Plotly chart for Price vs Bands."""
    fig = go.Figure()

    # Bands Area (Filled)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper_Band'],
        line=dict(width=0),
        mode='lines',
        showlegend=False,
        name='Upper Band'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower_Band'],
        line=dict(width=0),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(200, 200, 200, 0.2)', # Light gray fill
        showlegend=False,
        name='Lower Band'
    ))

    # Upper/Lower Band Lines (Dashed)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Upper_Band'],
        line=dict(color='gray', width=1, dash='dash'),
        name='Upper Band (+2σ)'
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Lower_Band'],
        line=dict(color='gray', width=1, dash='dash'),
        name='Lower Band (-2σ)'
    ))

    # Moving Average
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MA_200'],
        line=dict(color='#FF8C00', width=2), # Dark Orange
        name='200-Day MA'
    ))

    # Price Line
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Close'],
        line=dict(color=color_line, width=2),
        name=f'{asset_name} Price'
    ))

    fig.update_layout(
        title=f"<b>{asset_name} Price Trends & Volatility Bands</b>",
        yaxis_title="Price",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=600
    )
    return fig

def plot_volatility_comparison(gold, silver):
    """Plots comparative rolling volatility."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=gold.index, y=gold['volatility'],
        line=dict(color='#B59410', width=2),
        name='Gold Volatility'
    ))
    
    fig.add_trace(go.Scatter(
        x=silver.index, y=silver['volatility'],
        line=dict(color='#7d8597', width=2), # Silver-ish blue
        name='Silver Volatility'
    ))

    fig.update_layout(
        title="<b>Rolling Volatility Comparison (252-Day)</b>",
        yaxis_title="Annualized Volatility",
        xaxis_title="Date",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig

# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
gold_full, silver_full, gold_aligned, silver_aligned = load_and_process_data()

# 1. Header
st.title("Gold & Silver: Trend, Volatility, and Price Band Analysis")
st.markdown("**Data:** 2014–2026 | **Method:** 200-Day Moving Average & Rolling Volatility")
st.divider()

# 2. Snapshots (Use FULL data for latest prices)S
col1, col2 = st.columns(2)

def render_snapshot(col, title, df, unit):
    last = df.iloc[-1]
    pos_text, pos_status = determine_band_position(last['Close'], last['Upper_Band'], last['Lower_Band'])
    
    with col:
        st.subheader(f"{title} Snapshot")
        # Custom HTML card
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Latest Price ({unit})</div>
            <div class="metric-value">{format_inr(last['Close'], 2)}</div>
            <div class="metric-label" style="margin-top:10px;">200-Day MA</div>
            <div style="font-size: 1.1em; color: #555;">{format_inr(last['MA_200'], 2)}</div>
            <div class="status-text status-{pos_status}">{pos_text}</div>
            <div style="margin-top:15px; font-size:0.9em; color:#888;">
                Current Volatility: <b>{last['volatility']:.2%}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

render_snapshot(col1, "Gold", gold_full, "₹/10g")
render_snapshot(col2, "Silver", silver_full, "₹/1kg")

st.divider()

# 3. Price Charts (Use FULL data)
st.plotly_chart(plot_price_bands(gold_full, "Gold", "#B59410"), use_container_width=True)
st.plotly_chart(plot_price_bands(silver_full, "Silver", "#2E86C1"), use_container_width=True)

# 4. Volatility Section (Use ALIGNED data for valid comparison)
st.divider()
st.subheader("Volatility Comparison")
st.plotly_chart(plot_volatility_comparison(gold_aligned, silver_aligned), use_container_width=True)
st.caption("Silver exhibits consistently higher rolling volatility and sharper spikes compared to gold, particularly during macro stress periods.")

# 5. Insights & Macro
st.divider()
c_insight, c_macro = st.columns([1, 1])

with c_insight:
    st.info("""
    **Allocation Insight**
    
    **Recommendation:** Cautious / Staggered Allocation
    
    **Rationale:**
    *   Prices may be trading near or above upper statistical bands.
    *   Elevated volatility regimes suggest higher risk of short-term mean reversion.
    *   Historical data indicates entering during band extensions requires disciplined sizing.
    """)

with c_macro:
    st.markdown("""
    ### Macro Context
    *   **2020–2021:** Extreme volatility spike driven by the global pandemic and subsequent liquidity shocks.
    *   **Post-2022:** Volatility remains elevated due to persistent inflation, central bank rate tightening, and geopolitical instability.
    *   **Silver Specifics:** Shows amplified price movements compared to Gold, reflecting its dual nature as both a precious metal and an industrial commodity (demand sensitivity).
    """)

# 6. Interactive Simulator: SIP vs Lump Sum
st.divider()
st.subheader("Interactive Strategy Simulator: SIP vs. Lump Sum")
st.markdown("Simulate how a **Staggered (SIP)** approach compares to a **Lump Sum** investment over a specific period.")

# --- Inputs Section ---
st.markdown("#### Settings")
c_in1, c_in2, c_in3, c_in4 = st.columns([1, 1, 1, 1])

with c_in1:
    sim_asset = st.selectbox("Select Asset", ["Gold", "Silver"])

# Context for date picker
df_sim = gold_full if sim_asset == "Gold" else silver_full
min_date = df_sim.index.min().date()
max_date = df_sim.index.max().date()

with c_in2:
    sim_start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"), min_value=min_date, max_value=max_date)

with c_in3:
    monthly_inv = st.number_input("Monthly Inv. (₹)", value=10000, step=1000)

with c_in4:
    # Align button with inputs
    st.write("") 
    st.write("")
    run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

# --- Simulation Execution ---
if run_btn:
    # Filter Data
    mask = df_sim.index >= pd.to_datetime(sim_start_date)
    sim_data = df_sim.loc[mask].copy()
    
    if len(sim_data) == 0:
        st.error("No data available for selected date.")
    else:
        # --- SIP Calculation ---
        sip_data = sim_data.resample('MS').first()
        
        sip_data['units_bought'] = monthly_inv / sip_data['Close']
        sip_data['cum_units'] = sip_data['units_bought'].cumsum()
        sip_data['total_invested'] = np.arange(1, len(sip_data) + 1) * monthly_inv
        
        # Daily Portfolio Value reconstruction
        daily_units = sip_data['cum_units'].reindex(sim_data.index, method='ffill')
        sim_data['sip_value'] = daily_units * sim_data['Close']
        sim_data['sip_invested'] = sip_data['total_invested'].reindex(sim_data.index, method='ffill')
        
        # --- Lump Sum Calculation ---
        total_months = len(sip_data)
        total_capital = total_months * monthly_inv
        
        lump_units = total_capital / sim_data['Close'].iloc[0]
        sim_data['lump_value'] = lump_units * sim_data['Close']
        
        # --- Results Extraction ---
        final_sip_val = sim_data['sip_value'].iloc[-1]
        final_lump_val = sim_data['lump_value'].iloc[-1]
        final_invested = total_capital
        
        sip_return = (final_sip_val - final_invested) / final_invested
        lump_return = (final_lump_val - final_invested) / final_invested
        
        # --- Risk / Drawdown ---
        sip_peak = sim_data['sip_value'].cummax()
        sip_mdd = ((sim_data['sip_value'] - sip_peak) / sip_peak).min()

        lump_peak = sim_data['lump_value'].cummax()
        lump_mdd = ((sim_data['lump_value'] - lump_peak) / lump_peak).min()

        # --- Display Results (Full Width) ---
        st.divider()
        st.markdown(
            f"<h3 style='text-align: center; color: #4CAF50;'>Total Invested: {format_inr(final_invested, 0)}</h3>",
            unsafe_allow_html=True,
        )
        st.write("")

        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SIP Final Value", format_inr(final_sip_val, 0), f"{sip_return:.1%}")
        m2.metric("SIP Max Risk (DD)", f"{sip_mdd:.1%}", delta_color="inverse")
        m3.metric("Lump Sum Value", format_inr(final_lump_val, 0), f"{lump_return:.1%}")
        m4.metric("Lump Sum Risk (DD)", f"{lump_mdd:.1%}", delta_color="inverse")

        st.write("")
        
        # Insight Logic
        if abs(lump_mdd) > abs(sip_mdd) + 0.05: 
            risk_msg = "💡 **Insight:** While Lump Sum may have higher returns in strong bull runs, **SIP significantly reduced portfolio risk** (Drawdown), protecting capital during corrections."
        else:
            risk_msg = "💡 **Insight:** In this specific sustained uptrend, both strategies performed well, but Lump Sum capitalized on early low prices."
        st.info(risk_msg)

        # Plot
        st.subheader("Portfolio Growth Comparison")
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=sim_data.index, y=sim_data['sip_value'], name='SIP Portfolio Value', line=dict(color='#2ecc71', width=3)))
        fig_sim.add_trace(go.Scatter(x=sim_data.index, y=sim_data['lump_value'], name='Lump Sum Portfolio Value', line=dict(color='#e74c3c', width=3)))
        
        fig_sim.update_layout(
            yaxis_title="Portfolio Value (₹)", 
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"), 
            hovermode="x unified",
            height=500, # Taller graph as requested
            margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig_sim, use_container_width=True)
