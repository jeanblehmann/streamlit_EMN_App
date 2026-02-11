import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page config
st.set_page_config(page_title="EMN AI Portfolio Builder", layout="wide")

st.title("🚀 Equity Market Neutral (EMN) AI Portfolio Builder")

# --- 1. DATA FETCHING (Same as before) ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    data_list = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            beta = stock.info.get('beta', 1.0)
            hist = stock.history(period="6mo")
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            np.random.seed(sum(ord(c) for c in t)) 
            data_list.append({
                "Ticker": t, "Beta": round(beta, 2), 
                "Momentum": round(momentum, 2),
                "AI_Sentiment": round(np.random.uniform(-1, 1), 2),
                "XGB_Conviction": round(np.random.uniform(0.3, 0.95), 2),
                "Cluster": f"Group {np.random.choice([1, 2, 3, 4])}"
            })
        except: continue
    return pd.DataFrame(data_list)

universe = ['AAPL', 'TSLA', 'XOM', 'JPM', 'MSFT', 'META', 'NVDA', 'AMZN', 'GOOGL', 'BRK-B']
df_stocks = get_market_data(universe)

# --- 2. UNIVERSE DASHBOARD ---
st.subheader("📊 Stock Universe & AI Insights")
styled_df = df_stocks.style.background_gradient(subset=['AI_Sentiment'], cmap='RdYlGn') \
    .background_gradient(subset=['XGB_Conviction'], cmap='Greens') \
    .format({'Momentum': "{:.2f}%", 'XGB_Conviction': "{:.0%}"})
st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.divider()

# --- 3. SELECTION ---
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    longs = st.multiselect("🟢 Select Long Positions", options=df_stocks['Ticker'], key="ls")
with col_sel2:
    shorts = st.multiselect("🔴 Select Short Positions", options=df_stocks['Ticker'], key="ss")

# --- 4. REACTIVE WEIGHTING ENGINE (THE FIX) ---

def init_weights(tickers, side):
    """Ensures weights exist in state and sum to 1.0."""
    key = f"w_{side}"
    if key not in st.session_state or set(st.session_state[key].keys()) != set(tickers):
        if tickers:
            st.session_state[key] = {t: 1.0/len(tickers) for t in tickers}
        else:
            st.session_state[key] = {}

init_weights(longs, "long")
init_weights(shorts, "short")

def update_weights(changed_ticker, tickers, side):
    """Force-balances other sliders when one is moved."""
    state_key = f"w_{side}"
    # Get the value the user just set on the slider
    new_val = st.session_state[f"s_{side}_{changed_ticker}"]
    st.session_state[state_key][changed_ticker] = new_val
    
    others = [t for t in tickers if t != changed_ticker]
    if not others:
        st.session_state[state_key][changed_ticker] = 1.0
        return

    # Calculate remaining pool
    remaining = 1.0 - new_val
    current_other_sum = sum(st.session_state[state_key][t] for t in others)
    
    if current_other_sum > 0:
        for t in others:
            st.session_state[state_key][t] = (st.session_state[state_key][t] / current_other_sum) * remaining
    else:
        for t in others:
            st.session_state[state_key][t] = remaining / len(others)

# Reset Button
if st.button("🔄 Reset Weights"):
    for k in ["w_long", "w_short"]: 
        if k in st.session_state: del st.session_state[k]
    st.rerun()

st.subheader("⚖️ Reactive Weight Balancing")

def render_side(tickers, side):
    if not tickers:
        st.info(f"Select {side} stocks to adjust weights.")
        return
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            # The 'value' comes from session_state, but the slider updates via 'on_change'
            st.slider(
                t, 0.0, 1.0, 
                value=float(st.session_state[f"w_{side}"][t]),
                key=f"s_{side}_{t}",
                on_change=update_weights,
                args=(t, tickers, side),
                format="%.2f"
            )

col_l, col_r = st.columns(2)
with col_l:
    st.write("🟢 Long Weights (Auto-balanced to 1.0)")
    render_side(longs, "long")
with col_r:
    st.write("🔴 Short Weights (Auto-balanced to 1.0)")
    render_side(shorts, "short")

# --- 5. CALCULATIONS & RESULTS ---
def calc_beta(weights_dict, data):
    if not weights_dict: return 0.0
    return sum(w * data.set_index("Ticker").loc[t, "Beta"] for t, w in weights_dict.items())

l_beta = calc_beta(st.session_state.get("w_long", {}), df_stocks)
s_beta = calc_beta(st.session_state.get("w_short", {}), df_stocks)
net_beta = l_beta - s_beta

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Long Side Beta", round(l_beta, 2))
c2.metric("Short Side Beta", round(s_beta, 2))
c3.metric("NET PORTFOLIO BETA", round(net_beta, 3))

if -0.05 <= net_beta <= 0.05:
    st.success("✅ **Beta Neutral!**")
else:
    st.error("⚠️ **Market Exposed**")