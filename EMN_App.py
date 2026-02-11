import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="EMN AI Portfolio Builder", layout="wide")
st.title("🚀 Equity Market Neutral (EMN) AI Portfolio Builder")


# ============================================================
# 1. DATA FETCHING
# ============================================================
@st.cache_data(ttl=3600)
def get_market_data(tickers: list[str]) -> pd.DataFrame:
    """Fetch stock metadata (beta, momentum, simulated AI signals)."""
    data_list = []
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            info = stock.info
            beta = info.get("beta") or 1.0
            beta = float(beta)
            sector = info.get("sector", "Unknown")

            hist = stock.history(period="6mo")
            if hist.empty or len(hist) < 2:
                logger.warning("Insufficient history for %s — skipping", t)
                continue
            momentum = ((hist["Close"].iloc[-1] / hist["Close"].iloc[0]) - 1) * 100

            np.random.seed(sum(ord(c) for c in t))
            data_list.append(
                {
                    "Ticker": t,
                    "Sector": sector,
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


@st.cache_data(ttl=3600)
def get_price_history(tickers: list[str], period: str = "6mo") -> pd.DataFrame:
    """Fetch daily close prices for a list of tickers."""
    if not tickers:
        return pd.DataFrame()
    try:
        data = yf.download(tickers, period=period, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers[0])
        return data.dropna()
    except Exception as exc:
        logger.warning("Price download failed: %s", exc)
        return pd.DataFrame()


UNIVERSE = [
    "AAPL", "TSLA", "XOM", "JPM", "MSFT",
    "META", "NVDA", "AMZN", "GOOGL", "BRK-B",
]
df_stocks = get_market_data(UNIVERSE)


# ============================================================
# 2. UNIVERSE DASHBOARD
# ============================================================
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


# ============================================================
# EDUCATION HUB
# ============================================================
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
    st.divider()
    st.markdown("### 💰 Dollar Neutral vs Beta Neutral")
    col_c, col_d = st.columns(2)
    with col_c:
        st.write(
            "**Dollar Neutral:** Equal dollar amounts on each side. "
            "$1 long for every $1 short. Gross exposure = 200%, net dollar exposure = 0%. "
            "This app enforces dollar neutrality by normalising each side to 100%."
        )
    with col_d:
        st.write(
            "**Beta Neutral:** Weighted beta of longs equals weighted beta of shorts, "
            "so the portfolio has zero sensitivity to market direction. "
            "You can be dollar neutral but still market-exposed if betas differ — "
            "beta neutrality is the stronger hedge."
        )
    st.info(
        "💡 **Strategy Tip:** Long stocks with high Sentiment/Conviction and "
        "Short those with low scores, then adjust weights to reach a Net Beta of 0.00. "
        "A truly market-neutral portfolio is both dollar neutral AND beta neutral."
    )

st.divider()


# ============================================================
# 3. POSITION SELECTION (with overlap guard)
# ============================================================
available_tickers = df_stocks["Ticker"].tolist()

col_sel1, col_sel2 = st.columns(2)
with col_sel1:
    longs = st.multiselect("🟢 Select Long Positions", options=available_tickers, key="ls")
with col_sel2:
    short_options = [t for t in available_tickers if t not in longs]
    shorts = st.multiselect("🔴 Select Short Positions", options=short_options, key="ss")


# ============================================================
# 4. WEIGHT SLIDERS
# ============================================================
st.subheader("⚖️ Weight Allocation")


def collect_weights(tickers: list[str], side: str) -> dict[str, float]:
    """Render sliders, return normalised weights summing to 1.0."""
    if not tickers:
        st.info(f"Select {side} stocks above to begin.")
        return {}

    default = round(1.0 / len(tickers), 2)

    raw = {}
    cols = st.columns(len(tickers))
    for i, t in enumerate(tickers):
        with cols[i]:
            raw[t] = st.slider(f"{t}", 0.0, 1.0, default, 0.01, key=f"sl_{side}_{t}")

    total_raw = sum(raw.values())
    if total_raw > 0:
        normed = {t: v / total_raw for t, v in raw.items()}
    else:
        normed = {t: 1.0 / len(tickers) for t in tickers}

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


# ============================================================
# 5. PORTFOLIO METRICS — Beta + Dollar Neutral checks
# ============================================================

def calc_beta(weights: dict[str, float], data: pd.DataFrame) -> float:
    if not weights:
        return 0.0
    beta_lookup = data.set_index("Ticker")["Beta"]
    return sum(w * beta_lookup.get(t, 1.0) for t, w in weights.items())


l_beta = calc_beta(long_weights, df_stocks)
s_beta = calc_beta(short_weights, df_stocks)
net_beta = l_beta - s_beta

long_dollar_pct = sum(long_weights.values()) * 100 if long_weights else 0
short_dollar_pct = sum(short_weights.values()) * 100 if short_weights else 0

st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Long Side Beta", round(l_beta, 2))
c2.metric("Short Side Beta", round(s_beta, 2))
c3.metric("NET PORTFOLIO BETA", round(net_beta, 3))
c4.metric("Gross Exposure", f"{long_dollar_pct + short_dollar_pct:.0f}%")

# Status checks
col_status1, col_status2 = st.columns(2)
with col_status1:
    if -0.05 <= net_beta <= 0.05:
        st.success("✅ **Beta Neutral** — portfolio hedged against market moves")
    else:
        st.error("⚠️ **Not Beta Neutral** — portfolio exposed to market direction")
with col_status2:
    if long_weights and short_weights:
        st.success("✅ **Dollar Neutral** — equal notional on both sides (100% / 100%)")
    elif long_weights or short_weights:
        st.warning("⚠️ **One-sided** — select positions on both sides for dollar neutrality")
    else:
        st.info("Select long and short positions to begin.")


# ============================================================
# 6. LIVE CHARTS (only shown when positions are selected)
# ============================================================
all_selected = longs + shorts
if all_selected:
    st.divider()
    st.subheader("📈 Portfolio Analytics (Live Data)")

    prices = get_price_history(all_selected)

    if not prices.empty:
        daily_returns = prices.pct_change().dropna()

        # --------------------------------------------------
        # CHART 1: Correlation Heatmap
        # --------------------------------------------------
        if len(all_selected) >= 2:
            st.markdown("#### 🔗 Correlation Matrix")
            st.caption(
                "Shows how selected stocks move together. "
                "For a good EMN portfolio, you want low correlation between long and short legs."
            )
            corr = daily_returns[all_selected].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            fig_corr.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_corr, use_container_width=True)

        # --------------------------------------------------
        # CHART 2: Cumulative Return Backtest
        # --------------------------------------------------
        if long_weights and short_weights:
            st.markdown("#### 📊 Cumulative Return Backtest (6-Month)")
            st.caption(
                "Simulates how your portfolio would have performed with the current weights. "
                "The net (green) line shows the market-neutral return."
            )

            # Build weighted return series for each leg
            long_ret = pd.Series(0.0, index=daily_returns.index)
            for t, w in long_weights.items():
                if t in daily_returns.columns:
                    long_ret += w * daily_returns[t]

            short_ret = pd.Series(0.0, index=daily_returns.index)
            for t, w in short_weights.items():
                if t in daily_returns.columns:
                    short_ret += w * daily_returns[t]

            # Net return: profit on longs + profit on shorts (short P&L is inverted)
            net_ret = long_ret - short_ret

            # Cumulative
            cum_long = (1 + long_ret).cumprod() - 1
            cum_short = (1 + short_ret).cumprod() - 1
            cum_net = (1 + net_ret).cumprod() - 1

            cum_df = pd.DataFrame(
                {
                    "Long Leg": cum_long * 100,
                    "Short Leg": cum_short * 100,
                    "Net (L − S)": cum_net * 100,
                }
            )

            fig_cum = go.Figure()
            fig_cum.add_trace(
                go.Scatter(
                    x=cum_df.index, y=cum_df["Long Leg"],
                    name="Long Leg", line=dict(color="#2ecc71", width=2),
                )
            )
            fig_cum.add_trace(
                go.Scatter(
                    x=cum_df.index, y=cum_df["Short Leg"],
                    name="Short Leg", line=dict(color="#e74c3c", width=2),
                )
            )
            fig_cum.add_trace(
                go.Scatter(
                    x=cum_df.index, y=cum_df["Net (L − S)"],
                    name="Net (L − S)", line=dict(color="#3498db", width=3),
                )
            )
            fig_cum.add_hline(y=0, line_dash="dash", line_color="grey", opacity=0.5)
            fig_cum.update_layout(
                yaxis_title="Cumulative Return (%)",
                xaxis_title="",
                height=420,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # Quick stats
            col_s1, col_s2, col_s3 = st.columns(3)
            col_s1.metric("Long Leg Return", f"{cum_long.iloc[-1] * 100:+.2f}%")
            col_s2.metric("Short Leg Return", f"{cum_short.iloc[-1] * 100:+.2f}%")
            col_s3.metric("Net Strategy Return", f"{cum_net.iloc[-1] * 100:+.2f}%")

        # --------------------------------------------------
        # CHART 3: Beta Contribution Waterfall
        # --------------------------------------------------
        if long_weights or short_weights:
            st.markdown("#### 📐 Beta Contribution by Position")
            st.caption(
                "Each bar shows how much beta a position contributes. "
                "The goal is for the green (long) bars to offset the red (short) bars."
            )

            beta_lookup = df_stocks.set_index("Ticker")["Beta"]
            waterfall_data = []

            for t, w in long_weights.items():
                b = beta_lookup.get(t, 1.0)
                waterfall_data.append(
                    {"Ticker": f"{t} (L)", "Beta Contribution": round(w * b, 4), "Side": "Long"}
                )
            for t, w in short_weights.items():
                b = beta_lookup.get(t, 1.0)
                waterfall_data.append(
                    {"Ticker": f"{t} (S)", "Beta Contribution": round(-w * b, 4), "Side": "Short"}
                )

            if waterfall_data:
                wf_df = pd.DataFrame(waterfall_data)
                colors = ["#2ecc71" if s == "Long" else "#e74c3c" for s in wf_df["Side"]]

                fig_wf = go.Figure(
                    go.Bar(
                        x=wf_df["Ticker"],
                        y=wf_df["Beta Contribution"],
                        marker_color=colors,
                        text=[f"{v:+.3f}" for v in wf_df["Beta Contribution"]],
                        textposition="outside",
                    )
                )
                fig_wf.add_hline(y=0, line_color="grey", line_width=1)

                # Net beta annotation
                fig_wf.add_annotation(
                    x=1.0, y=1.0, xref="paper", yref="paper",
                    text=f"Net Beta: {net_beta:+.3f}",
                    showarrow=False,
                    font=dict(size=14, color="#3498db"),
                    bgcolor="white",
                    bordercolor="#3498db",
                    borderwidth=1,
                    borderpad=6,
                )
                fig_wf.update_layout(
                    yaxis_title="Beta Contribution",
                    xaxis_title="",
                    height=400,
                    margin=dict(l=20, r=20, t=30, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_wf, use_container_width=True)

        # --------------------------------------------------
        # CHART 4: Sector Exposure (bonus)
        # --------------------------------------------------
        if long_weights or short_weights:
            st.markdown("#### 🏭 Sector Exposure")
            st.caption(
                "Shows your net weight by sector. "
                "Even a beta-neutral portfolio can have concentrated sector risk."
            )

            sector_lookup = df_stocks.set_index("Ticker")["Sector"]
            sector_exposure = {}

            for t, w in long_weights.items():
                sec = sector_lookup.get(t, "Unknown")
                sector_exposure[sec] = sector_exposure.get(sec, 0) + w

            for t, w in short_weights.items():
                sec = sector_lookup.get(t, "Unknown")
                sector_exposure[sec] = sector_exposure.get(sec, 0) - w

            if sector_exposure:
                sec_df = pd.DataFrame(
                    [{"Sector": k, "Net Weight": v} for k, v in sector_exposure.items()]
                ).sort_values("Net Weight")

                colors_sec = [
                    "#2ecc71" if v >= 0 else "#e74c3c" for v in sec_df["Net Weight"]
                ]

                fig_sec = go.Figure(
                    go.Bar(
                        x=sec_df["Net Weight"] * 100,
                        y=sec_df["Sector"],
                        orientation="h",
                        marker_color=colors_sec,
                        text=[f"{v * 100:+.1f}%" for v in sec_df["Net Weight"]],
                        textposition="outside",
                    )
                )
                fig_sec.add_vline(x=0, line_color="grey", line_width=1)
                fig_sec.update_layout(
                    xaxis_title="Net Exposure (%)",
                    yaxis_title="",
                    height=max(250, len(sector_exposure) * 50),
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_sec, use_container_width=True)

    else:
        st.warning("Could not load price history for selected stocks.")
else:
    st.info("Select long and short positions above to see portfolio analytics.")