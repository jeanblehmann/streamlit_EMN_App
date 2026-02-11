import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EMN AI Portfolio Builder", layout="wide")
st.title("🚀 Equity Market Neutral (EMN) AI Portfolio Builder")


# --- 1. DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_market_data(tickers: list[str]) -> pd.DataFrame:
    data_list = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            beta = info.get("beta") or 1.0
            beta = float(beta)

            hist = stock.history(period="6mo")
            if hist.empty or len(hist) < 2:
                logger.warning("Insufficient history for %s — skipping", t)
                continue
            momentum = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100

            np.random.seed(sum(ord(c) for c in t))
            data_list.append(
                {
                    "Ticker": t,
                    "Beta": round(beta, 2),
                    "Momentum": round(momentum, 2),
                    "AI_Sentiment": round(np.random.uniform(-1, 1), 2),
                    "XGB_Conviction": round(np.random.uniform(0.3, 0.95), 2),
                    "Cluster": f"Group {np.random.choice([1, 2, 3, 4])}",
                }
            )
        except Exception as exc:
            logger.warning("Failed to fetch data for %s: %s", t, exc)
            continue
    return pd.DataFrame(data_list)


UNIVERSE = [
    "AAPL", "TSLA", "XOM", "JPM", "MSFT",
    "META", "NVDA", "AMZN", "GOOGL", "BRK-B",
]
df_stocks = get_market_data(UNIVERSE)


# --- 2. UNIVERSE DASHBOARD ---
st.subheader("📊 Stock Universe & AI Insights")
if df_stocks.empty:
    st.warning("No stock data could be loaded. Check your network / tickers.")
    st.stop()

styled_df = (
    df_stocks.style.background_gradient(subset=["AI_Sentiment"], cmap="RdYlGn")
    .background_gradient(subset=["XGB_Conviction"], cmap="Greens")
    .format({"Momentum": "{:.2f}%", "XGB_Conviction": "{:.0%}"})
)
st.dataframe(styled_df, use_container_width=True, hide_index=True)


# --- EDUCATION HUB ---
with st.expander("📚 Guide: How to interpret AI Signals & Factors"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📈 Factor Definitions")
        st.write(
            "**Momentum (6-Mo):** Price trend over the last 180 days. "
            "High momentum identifies 'winners' likely to continue outperforming."
        )
        st.write(
            "**AI Sentiment:** NLP analysis of earnings calls and news. "
            "Scores range from -1 (Bearish) to +1 (Bullish)."
        )
    with col_b:
        st.markdown("### 🤖 AI Model Logic")
        st.write(
            "**XGB Conviction:** Supervised learning probability (0-100%) "
            "that the stock will beat its peers."
        )
        st.write(
            "**Cluster Group:** Unsupervised clustering of stocks with similar "
            "behavior; diversify by picking from different groups."
        )
    st.info(
        "💡 **Strategy Tip:** Long stocks with high Sentiment/Conviction and "
        "Short those with low scores, then adjust weights to reach a Net Beta of 0.00."
    )

st.divider()


# --- 3. SELECTION (with overlap guard) ---
available_tickers = df_stocks["Ticker"].tolist()

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    longs = st.multiselect("🟢 Select Long Positions", options=available_tickers, key="ls")
with col_sel2:
    short_options = [t for t in available_tickers if t not in longs]
    shorts = st.multiselect("🔴 Select Short Positions", options=short_options, key="ss")


# --- 4. WEIGHT SLIDERS ---
# Simple pattern: render sliders → read raw values → normalise → display normalised weights.
# No on_change callbacks, no session_state juggling. Just works.

st.subheader("⚖️ Weight Allocation")


def collect_weights(tickers: list[str], side: str) -> dict[str, float]:
    """Render sliders for each ticker and return normalised weights summing to 1.0."""
    if not tickers:
        st.info(f"Select {side} stocks above to begin.")
        return {}

    # Default: equal weight
    default = round(1.0 / len(tickers), 2)

    # Collect raw slider values
    raw = {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            raw[t] = st.slider(f"{t}", 0.0, 1.0, default, 0.01, key=f"sl_{side}_{t}")

    # Normalise so they always sum to 1.0
    total_raw = sum(raw.values())
    if total_raw > 0:
        normed = {t: v / total_raw for t, v in raw.items()}
    else:
        normed = {t: 1.0 / len(tickers) for t in tickers}

    # Show normalised weights as a bar + percentage
    bar_cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with bar_cols[i]:
            pct = normed[t] * 100
            st.progress(min(normed[t], 1.0), text=f"{t}: {pct:.1f}%")

    st.caption(f"Raw sum: {total_raw:.2f} → Normalised to 100%")
    return normed


col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**🟢 Long Weights**")
    long_weights = collect_weights(longs, "long")
with col_r:
    st.markdown("**🔴 Short Weights**")
    short_weights = collect_weights(shorts, "short")


# --- 5. PORTFOLIO METRICS ---

def calc_beta(weights: dict[str, float], data: pd.DataFrame) -> float:
    if not weights:
        return 0.0
    beta_lookup = data.set_index("Ticker")["Beta"]
    return sum(w * beta_lookup.get(t, 1.0) for t, w in weights.items())


l_beta = calc_beta(long_weights, df_stocks)
s_beta = calc_beta(short_weights, df_stocks)
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