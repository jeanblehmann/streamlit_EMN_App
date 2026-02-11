import streamlit as st
import pandas as pd

# Sample stock data
stocks = pd.DataFrame({
    'Ticker': ['AAPL', 'TSLA', 'XOM', 'JPM'],
    'Beta': [1.2, 1.8, 0.9, 1.1],
    'Momentum': ['Strong', 'Weak', 'Neutral', 'Strong'],
    'AI_Signal': ['Positive', 'Negative', 'Neutral', 'Positive']
})

st.title("Equity Market Neutral Portfolio Builder")

# Choose long and short positions
longs = st.multiselect("Select Long Positions", options=stocks['Ticker'])
shorts = st.multiselect("Select Short Positions", options=stocks['Ticker'])

# Display table
selected = stocks[stocks['Ticker'].isin(longs + shorts)]
st.write("Selected Stocks", selected)

# Assign weights
st.subheader("Assign Weights (Total Long = Total Short)")
long_weights = {stock: st.slider(f"{stock} Long Weight", 0.0, 1.0, 0.1) for stock in longs}
short_weights = {stock: st.slider(f"{stock} Short Weight", 0.0, 1.0, 0.1) for stock in shorts}

# Calculate portfolio beta
def calc_portfolio_beta(weights, tickers):
    betas = stocks.set_index("Ticker").loc[tickers]["Beta"]
    return sum(weights[t] * betas[t] for t in tickers)

beta = calc_portfolio_beta(long_weights, longs) - calc_portfolio_beta(short_weights, shorts)
st.metric("Net Portfolio Beta", round(beta, 3))
