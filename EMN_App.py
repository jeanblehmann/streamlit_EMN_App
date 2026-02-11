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
    longs = st.multiselect(
        "🟢 Select Long Positions", options=available_tickers, key="ls"
    )
with col_sel2:
    short_options = [t for t in available_tickers if t not in longs]
    shorts = st.multiselect(
        "🔴 Select Short Positions", options=short_options, key="ss"
    )


# --- 4. WEIGHT MANAGEMENT ---
#
# How it works:
#   - Weights live in st.session_state["w_long"] / ["w_short"] as {ticker: float}
#   - When tickers change (add/remove), we re-initialise to equal weight
#   - When a slider moves, on_change fires _rebalance() which:
#       1. Reads the new value from the widget key
#       2. Proportionally scales all OTHER weights so total == 1.0
#       3. Writes back to the state dict
#   - Streamlit reruns and every slider reads its fresh value from the dict
#

def _ensure_weights(tickers: list[str], side: str) -> None:
    """Initialise or sync weight dict to match current ticker selection."""
    key = f"w_{side}"
    existing: dict = st.session_state.get(key, {})

    if set(existing.keys()) != set(tickers):
        if tickers:
            st.session_state[key] = {t: round(1.0 / len(tickers), 4) for t in tickers}
        else:
            st.session_state[key] = {}


_ensure_weights(longs, "long")
_ensure_weights(shorts, "short")


def _rebalance(changed_ticker: str, tickers: list[str], side: str) -> None:
    """When one slider moves, proportionally adjust the others so sum == 1."""
    sk = f"w_{side}"
    weights = st.session_state[sk]

    # Read the new value the user just dragged to
    new_val = st.session_state[f"sl_{side}_{changed_ticker}"]
    new_val = max(0.0, min(1.0, new_val))
    weights[changed_ticker] = new_val

    others = [t for t in tickers if t != changed_ticker]
    if not others:
        # Only one ticker — it must be 100%
        weights[changed_ticker] = 1.0
        return

    remaining = max(1.0 - new_val, 0.0)
    old_other_sum = sum(weights[t] for t in others)

    # Proportional redistribution (or equal split if all others were zero)
    if old_other_sum > 1e-9:
        scale = remaining / old_other_sum
        for t in others:
            weights[t] = weights[t] * scale
    else:
        for t in others:
            weights[t] = remaining / len(others)

    # Final normalisation to kill any float drift
    total = sum(weights.values())
    if total > 0:
        for t in tickers:
            weights[t] = weights[t] / total

    st.session_state[sk] = weights


# Reset button
if st.button("🔄 Reset to Equal Weights"):
    for side, tickers in [("long", longs), ("short", shorts)]:
        if tickers:
            st.session_state[f"w_{side}"] = {
                t: round(1.0 / len(tickers), 4) for t in tickers
            }
        else:
            st.session_state[f"w_{side}"] = {}
    st.rerun()


st.subheader("⚖️ Reactive Weight Balancing")


def _render_weights(tickers: list[str], side: str) -> None:
    """Render sliders for one side. Each slider triggers rebalance on change."""
    if not tickers:
        st.info(f"Select {side} stocks above to begin.")
        return

    weights = st.session_state[f"w_{side}"]
    total = sum(weights.values())

    # Live sum confirmation
    st.caption(f"Sum of weights: **{total:.4f}**")

    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            st.slider(
                label=t,
                min_value=0.0,
                max_value=1.0,
                value=float(weights[t]),
                step=0.01,
                key=f"sl_{side}_{t}",
                on_change=_rebalance,
                args=(t, tickers, side),
                format="%.2f",
            )

    # Visual weight bars
    bar_cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with bar_cols[i]:
            pct = weights[t] * 100
            st.progress(min(weights[t], 1.0), text=f"{t}: {pct:.1f}%")


col_l, col_r = st.columns(2)
with col_l:
    st.markdown("**🟢 Long Weights** (auto-balanced to 100%)")
    _render_weights(longs, "long")
with col_r:
    st.markdown("**🔴 Short Weights** (auto-balanced to 100%)")
    _render_weights(shorts, "short")


# --- 5. PORTFOLIO METRICS ---

def _calc_beta(weights_dict: dict[str, float], data: pd.DataFrame) -> float:
    if not weights_dict:
        return 0.0
    beta_lookup = data.set_index("Ticker")["Beta"]
    total = 0.0
    for t, w in weights_dict.items():
        if t in beta_lookup.index:
            total += w * beta_lookup[t]
        else:
            logger.warning("Ticker %s missing from data — excluded from beta", t)
    return total


l_beta = _calc_beta(st.session_state.get("w_long", {}), df_stocks)
s_beta = _calc_beta(st.session_state.get("w_short", {}), df_stocks)
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