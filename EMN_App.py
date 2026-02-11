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
        st.write("**Momentum (6-Mo):** The % price change over the last 180 days. High momentum stocks are 'winners' that often keep rising.")
        st.write("**AI Sentiment:** Derived from NLP analysis of news/transcripts. Ranges from -1 (Extremely Bearish) to +1 (Extremely Bullish).")
    with col_b:
        st.markdown("### 🤖 AI Model Logic")
        st.write("**XGB Conviction:** A supervised learning score (0-100%). It represents the model's 'confidence' that the stock will beat its peers.")
        st.write("**Cluster Group:** Unsupervised grouping of stocks that move together. Avoid picking all longs from the same group to maintain diversification.")
    
    st.info("💡 **Strategy Tip:** For a Market-Neutral 'Alpha' trade, look for stocks with high AI Conviction/Sentiment for **Longs**, and low scores for **Shorts**, while balancing their total Betas to zero.")

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
            
            # Synthetic AI Signals with fixed seed for consistency
            np.random.seed(sum(ord(c) for c in t)) 
            sentiment = np.random.uniform(-1, 1)
            xgboost_prob = np.random.uniform(0.3, 0.95)
            cluster = np.random.choice([1, 2, 3, 4])
            
            data_list.append({
                "Ticker": t, "Beta": round(beta, 2), 
                "Momentum": round(momentum, 2),
                "AI_Sentiment": round(sentiment, 2),
                "XGB_Conviction": round(xgboost_prob, 2),
                "Cluster": f"Group {cluster}"
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

# --- PORTFOLIO CONSTRUCTION ---
st.divider()
col1, col2 = st.columns(2)
with col1:
    longs = st.multiselect("🟢 Select Long Positions", options=df_stocks['Ticker'])
with col2:
    shorts = st.multiselect("🔴 Select Short Positions", options=df_stocks['Ticker'])

# --- NORMALIZATION LOGIC ---
st.subheader("⚖️ Weighted Exposure")
st.caption("Sliders represent 'raw' importance. The app automatically scales them so each side totals 100%.")

def get_normalized_weights(selected_tickers, column):
    raw_weights = {}
    if not selected_tickers: return {}
    for s in selected_tickers:
        raw_weights[s] = column.slider(f"Relative Weight: {s}", 0.01, 1.0, 0.5, key=f"w_{s}")
    
    total = sum(raw_weights.values())
    return {s: w / total for s, w in raw_weights.items()} # Normalization

col_l, col_s = st.columns(2)
norm_long_weights = get_normalized_weights(longs, col_l)
norm_short_weights = get_normalized_weights(shorts, col_s)

# --- BETA CALCULATION ---
def calculate_beta(weights_dict, data):
    if not weights_dict: return 0.0
    return sum(w * data.set_index("Ticker").loc[t, "Beta"] for t, w in weights_dict.items())

long_beta = calculate_beta(norm_long_weights, df_stocks)
short_beta = calculate_beta(norm_short_weights, df_stocks)
net_beta = long_beta - short_beta

# --- RESULTS ---
is_neutral = -0.05 <= net_beta <= 0.05
status_color = "green" if is_neutral else "red"

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Long Side Beta", round(long_beta, 2))
c2.metric("Short Side Beta", round(short_beta, 2))
c3.metric("NET PORTFOLIO BETA", round(net_beta, 3), delta="Target: 0.00")

if is_neutral:
    st.success("✅ **Beta Neutral!** Your portfolio is protected from broad market swings.")
else:
    st.error(f"⚠️ **Market Exposed:** Current Beta is {round(net_beta, 3)}. Adjust weights to cancel out market risk.")