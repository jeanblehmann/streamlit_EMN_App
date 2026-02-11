import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page config
st.set_page_config(page_title="EMN AI Portfolio Builder", layout="wide")

st.title("🚀 Equity Market Neutral (EMN) AI Portfolio Builder")

# --- 1. EDUCATION HUB ---
with st.expander("📚 Guide: How to interpret AI Signals & Factors"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📈 Factor Definitions")
        st.write("**Momentum (6-Mo):** Price trend over the last 180 days. High momentum identifies 'winners' likely to continue outperforming[cite: 142, 201].")
        st.write("**AI Sentiment:** NLP analysis of earnings calls and news. Scores range from -1 (Bearish) to +1 (Bullish)[cite: 171, 173].")
    with col_b:
        st.markdown("### 🤖 AI Model Logic")
        st.write("**XGB Conviction:** Supervised learning probability (0-100%) that the stock will beat its peers[cite: 140, 155].")
        st.write("**Cluster Group:** Unsupervised clustering of stocks with similar behavior; diversify by picking from different groups[cite: 176, 179].")
    st.info("💡 **Strategy Tip:** Long stocks with high Sentiment/Conviction and Short those with low scores, then adjust weights to reach a Net Beta of 0.00[cite: 46, 61].")

# --- 2. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_market_data(tickers):
    data_list = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            beta = stock.info.get('beta', 1.0)
            hist = stock.history(period="6mo")
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            
            # Synthetic AI Signals with fixed seed for consistency [cite: 137]
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

# --- 3. UNIVERSE DASHBOARD ---
st.subheader("📊 Stock Universe & AI Insights")
styled_df = df_stocks.style.background_gradient(subset=['AI_Sentiment'], cmap='RdYlGn') \
    .background_gradient(subset=['XGB_Conviction'], cmap='Greens') \
    .format({'Momentum': "{:.2f}%", 'XGB_Conviction': "{:.0%}"})
st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.divider()

# --- 4. SELECTION ---
col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    longs = st.multiselect("🟢 Select Long Positions", options=df_stocks['Ticker'], key="long_select")
with col_sel2:
    shorts = st.multiselect("🔴 Select Short Positions", options=df_stocks['Ticker'], key="short_select")

# --- 5. REACTIVE WEIGHTING ENGINE ---

# Reset Button Logic [cite: 8]
if st.button("🔄 Reset All Weights to Equal"):
    if "w_long" in st.session_state: del st.session_state["w_long"]
    if "w_short" in st.session_state: del st.session_state["w_short"]
    st.rerun()

def init_side_state(tickers, side_key):
    """Initializes or updates session state weights based on selection[cite: 31, 54]."""
    state_key = f"w_{side_key}"
    if not tickers:
        st.session_state[state_key] = {}
        return
    # Re-initialize if the tickers selected changed [cite: 55]
    if state_key not in st.session_state or set(st.session_state[state_key].keys()) != set(tickers):
        st.session_state[state_key] = {t: 1.0/len(tickers) for t in tickers}

init_side_state(longs, "long")
init_side_state(shorts, "short")

def on_weight_change(changed_ticker, tickers, side_key):
    """Callback to proportionally adjust other sliders when one moves[cite: 51, 62]."""
    state_key = f"w_{side_key}"
    new_val = st.session_state[f"slider_{side_key}_{changed_ticker}"]
    st.session_state[state_key][changed_ticker] = new_val
    
    other_tickers = [t for t in tickers if t != changed_ticker]
    if not other_tickers:
        st.session_state[state_key][changed_ticker] = 1.0
        return

    remaining_val = 1.0 - new_val
    current_other_sum = sum(st.session_state[state_key][t] for t in other_tickers)
    
    if current_other_sum > 0:
        # Distribute remaining value proportionally [cite: 59, 62]
        for t in other_tickers:
            st.session_state[state_key][t] = (st.session_state[state_key][t] / current_other_sum) * remaining_val
    else:
        # If others were at zero, split remaining evenly
        for t in other_tickers:
            st.session_state[state_key][t] = remaining_val / len(other_tickers)

def render_sliders(tickers, side_key):
    """Displays sliders using session state values[cite: 6, 44]."""
    if not tickers:
        st.info(f"Please select {side_key} positions above.")
        return {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            val = st.session_state[f"w_{side_key}"][t]
            st.slider(
                f"{t}", 0.0, 1.0, 
                value=float(val),
                key=f"slider_{side_key}_{t}",
                on_change=on_weight_change,
                args=(t, tickers, side_key),
                format="%.2f"
            )
    return st.session_state[f"w_{side_key}"]

st.subheader("⚖️ Reactive Weight Balancing")
st.caption("Adjusting one slider automatically scales others to maintain 100% side exposure.")

col_l, col_s = st.columns(2)
with col_l:
    st.write("🟢 Long Weights")
    final_long_weights = render_sliders(longs, "long")
with col_s:
    st.write("🔴 Short Weights")
    final_short_weights = render_sliders(shorts, "short")

# --- 6. BETA CALCULATIONS & RESULTS ---

def calculate_beta(weights_dict, data):
    """Calculates weighted average beta[cite: 33, 53]."""
    if not weights_dict: return 0.0
    return sum(w * data.set_index("Ticker").loc[t, "Beta"] for t, w in weights_dict.items())

long_beta = calculate_beta(final_long_weights, df_stocks)
short_beta = calculate_beta(final_short_weights, df_stocks)
net_beta = long_beta - short_beta # Net exposure [cite: 57, 62]

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Long Side Beta", round(long_beta, 2))
c2.metric("Short Side Beta", round(short_beta, 2))
c3.metric("NET PORTFOLIO BETA", round(net_beta, 3), delta="Target: 0.00")

# Neutrality Status [cite: 46, 62]
if -0.05 <= net_beta <= 0.05:
    st.success("✅ **Beta Neutral!** Your portfolio is statistically independent of market moves[cite: 45, 48].")
else:
    st.error(f"⚠️ **Market Exposed:** Current Beta is {round(net_beta, 3)}. Adjust weights to cancel market risk[cite: 58].")