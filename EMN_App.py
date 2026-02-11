import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf

# Page config
st.set_page_config(page_title="EMN AI Portfolio Builder", layout="wide")

st.title("Equity Market Neutral (EMN) AI Portfolio Builder")

# 1. DATA SOURCE: Live Beta & Momentum via yfinance
@st.cache_data(ttl=3600)  # Cache data for 1 hour to stay fast
def get_market_data(tickers):
    data_list = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            # Retrieve 5Y Monthly Beta from Yahoo Finance
            beta = stock.info.get('beta', 1.0)
            
            # Calculate 6-Month Momentum
            hist = stock.history(period="6mo")
            momentum = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            
            # 2. AI SIGNALS: Structured Synthetic Data
            # Seeding ensures consistent behavior for your session demo
            np.random.seed(sum(ord(c) for c in t)) 
            sentiment = np.random.uniform(-1, 1)
            xgboost_prob = np.random.uniform(0.3, 0.95)
            cluster = np.random.choice([1, 2, 3, 4])
            
            data_list.append({
                "Ticker": t, "Beta": round(beta, 2), 
                "Momentum (%)": round(momentum, 2),
                "AI_Sentiment": round(sentiment, 2),
                "XGB_Conviction": round(xgboost_prob, 2),
                "Cluster": f"Group {cluster}"
            })
        except Exception:
            continue
    return pd.DataFrame(data_list)

# Define curated universe
universe = ['AAPL', 'TSLA', 'XOM', 'JPM', 'MSFT', 'META', 'NVDA', 'AMZN', 'GOOGL', 'BRK-B']
df_stocks = get_market_data(universe)

# 3. STOCK UNIVERSE TABLE (Sortable with Heatmaps)
st.subheader("Stock Universe & AI Insights")
st.write("Analyze the factors below to identify your Long/Short pairs.")

# Create heatmap styling
styled_df = df_stocks.style.background_gradient(subset=['AI_Sentiment'], cmap='RdYlGn') \
    .background_gradient(subset=['XGB_Conviction'], cmap='Greens') \
    .format({'Momentum (%)': "{:.2f}%", 'XGB_Conviction': "{:.0%}"})

st.dataframe(styled_df, use_container_width=True, hide_index=True)

# 4. PORTFOLIO CONSTRUCTION
st.divider()
col1, col2 = st.columns(2)

with col1:
    longs = st.multiselect("🟢 Select Long Positions", options=df_stocks['Ticker'])
with col2:
    shorts = st.multiselect("🔴 Select Short Positions", options=df_stocks['Ticker'])

# Assign Weights
col_l, col_s = st.columns(2)
l_weights = {s: col_l.slider(f"Weight: {s}", 0.0, 1.0, 0.2, key=f"l_{s}") for s in longs}
s_weights = {s: col_s.slider(f"Weight: {s}", 0.0, 1.0, 0.2, key=f"s_{s}") for s in shorts}

# 5. METRIC & DYNAMIC BETA DIAL
def calculate_beta(weights, tickers, data):
    if not tickers: return 0.0
    subset = data.set_index("Ticker").loc[tickers]
    return sum(weights[t] * subset.loc[t, "Beta"] for t in tickers)

total_beta = calculate_beta(l_weights, longs, df_stocks) - calculate_beta(s_weights, shorts, df_stocks)

# Dynamic color logic for Beta Neutrality
is_neutral = -0.05 <= total_beta <= 0.05
status_color = "green" if is_neutral else "red"
status_msg = "✅ NEUTRAL" if is_neutral else "⚠️ MARKET EXPOSED"

st.subheader("Portfolio Risk Balance")
c1, c2 = st.columns(2)
with c1:
    # Use Markdown for custom color logic as standard st.metric color is limited
    st.metric("Net Portfolio Beta", round(total_beta, 3), 
              delta="Target: 0.00", delta_color="off")
with c2:
    st.markdown(f"### Status: :{status_color}[{status_msg}]")

if is_neutral:
    st.success("Great job! You have successfully neutralized the market factor.")
else:
    st.info("Adjust your sliders to bring the Net Beta closer to 0.00.")