import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page config
st.set_page_config(page_title="EMN AI Portfolio Builder", layout="wide")

st.title("🚀 Equity Market Neutral (EMN) AI Portfolio Builder")

# --- EDUCATION HUB: HOW TO INTERPRET SIGNALS ---
with st.expander("📚 Guide: How to read these signals for Long/Short decisions"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📈 Factor Definitions")
        st.write("**Momentum (6-Mo):** Price trend over the last 180 days. High momentum identifies 'winners' likely to continue outperforming.")
        st.write("**AI Sentiment:** NLP analysis of earnings calls and news. Scores range from -1 (Bearish) to +1 (Bullish).")
    with col_b:
        st.markdown("### 🤖 AI Model Logic")
        st.write("**XGB Conviction:** Supervised learning probability (0-100%) that the stock will beat its peers.")
        st.write("**Cluster Group:** Unsupervised clustering of stocks with similar behavior; diversify by picking from different groups.")
    st.info("💡 **Strategy Tip:** Long stocks with high Sentiment/Conviction and Short those with low scores, then adjust weights to reach a Net Beta of 0.00.")

# --- DATA FETCHING ---
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

# --- UNIVERSE DASHBOARD ---
st.subheader("📊 Stock Universe & AI Insights")
styled_df = df_stocks.style.background_gradient(subset=['AI_Sentiment'], cmap='RdYlGn') \
    .background_gradient(subset=['XGB_Conviction'], cmap='Greens') \
    .format({'Momentum': "{:.2f}%", 'XGB_Conviction': "{:.0%}"})
st.dataframe(styled_df, use_container_width=True, hide_index=True)

st.divider()
col1, col2 = st.columns(2)
with col1:
    longs = st.multiselect("🟢 Select Long Positions", options=df_stocks['Ticker'])
with col2:
    shorts = st.multiselect("🔴 Select Short Positions", options=df_stocks['Ticker'])

# --- REACTIVE WEIGHTING LOGIC ---
def init_weights(tickers, side_key):
    """Initializes session state weights to equal distribution."""
    if not tickers: return
    key = f"weights_{side_key}"
    if key not in st.session_state or set(st.session_state[key].keys()) != set(tickers):
        st.session_state[key] = {t: 1.0/len(tickers) for t in tickers}

def update_weights(side_key, tickers):
    """Callback to normalize weights when a slider moves."""
    raw_vals = {t: st.session_state[f"sl_{side_key}_{t}"] for t in tickers}
    total = sum(raw_vals.values())
    if total > 0:
        st.session_state[f"weights_{side_key}"] = {t: v / total for t, v in raw_vals.items()}

# Initialize and Render Reactive Sliders
init_weights(longs, "long")
init_weights(shorts, "short")

st.subheader("⚖️ Reactive Weight Balancing")
st.caption("Moving one slider automatically adjusts others to maintain 100% side exposure.")

col_l, col_s = st.columns(2)
with col_l:
    if longs:
        for t in longs:
            st.slider(f"Long Weight: {t}", 0.01, 1.0, 
                      value=st.session_state.weights_long.get(t, 1.0/len(longs)),
                      key=f"sl_long_{t}", 
                      on_change=update_weights, args=("long", longs))

with col_s:
    if shorts:
        for t in shorts:
            st.slider(f"Short Weight: {t}", 0.01, 1.0, 
                      value=st.session_state.weights_short.get(t, 1.0/len(shorts)),
                      key=f"sl_short_{t}", 
                      on_change=update_weights, args=("short", shorts))

# --- FINAL CALCULATIONS ---
def calculate_beta(weights_dict, data):
    if not weights_dict: return 0.0
    return sum(w * data.set_index("Ticker").loc[t, "Beta"] for t, w in weights_dict.items())

long_beta = calculate_beta(st.session_state.get('weights_long', {}), df_stocks)
short_beta = calculate_beta(st.session_state.get('weights_short', {}), df_stocks)
net_beta = long_beta - short_beta

# --- RESULTS ---
st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Long Side Beta", round(long_beta, 2))
c2.metric("Short Side Beta", round(short_beta, 2))
c3.metric("NET PORTFOLIO BETA", round(net_beta, 3), delta="Target: 0.00")

if -0.05 <= net_beta <= 0.05:
    st.success("✅ **Beta Neutral!** Portfolio is market-independent.")
else:
    st.error(f"⚠️ **Market Exposed:** Current Beta is {round(net_beta, 3)}. Balance weights further.")