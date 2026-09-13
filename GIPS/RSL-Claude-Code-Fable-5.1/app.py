"""Return Sensitivity Lab - GIPS Standards for Firms, Session 4.

Run locally:      streamlit run app.py
Community Cloud:  point the deployment at this file (see README.md).
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

import engine as E
from composite import (annualise_quarter, asset_vs_equal, composite_routes, compound,
                       default_card6_book, default_card7_book)

# ------------------------------------------------------------------ identity

NAVY, NAVY2, GOLD, ICE = "#0F1A3C", "#161E3D", "#D4A24C", "#CADCFC"
ICE_DIM = "rgba(202,220,252,0.55)"
GRID = "rgba(202,220,252,0.10)"
FONT = "Calibri, 'Segoe UI', 'Helvetica Neue', sans-serif"

st.set_page_config(page_title="Return Sensitivity Lab", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  .block-container {{ padding-top: 1.4rem; }}
  h1, h2, h3 {{ font-family: {FONT}; letter-spacing: 0.01em; }}
  .rsl-kicker {{ color: {GOLD}; font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase; }}
  .rsl-card {{ border-left: 3px solid {GOLD}; background: {NAVY2}; padding: 0.7rem 1rem; margin: 0.4rem 0 0.8rem 0; border-radius: 4px; }}
  .rsl-flag {{ border-left: 3px solid {GOLD}; background: rgba(212,162,76,0.10); padding: 0.6rem 0.9rem; margin-bottom: 0.55rem; border-radius: 4px; }}
  .rsl-note {{ border-left: 3px solid {ICE}; background: rgba(202,220,252,0.06); padding: 0.6rem 0.9rem; margin-bottom: 0.55rem; border-radius: 4px; }}
  .rsl-quiet {{ border-left: 3px solid rgba(202,220,252,0.25); color: {ICE_DIM}; padding: 0.5rem 0.9rem; margin-bottom: 0.55rem; border-radius: 4px; }}
  .rsl-chip {{ display: inline-block; font-size: 0.72rem; padding: 0.05rem 0.45rem; margin: 0.15rem 0.25rem 0 0;
               border: 1px solid {GOLD}; color: {GOLD}; border-radius: 10px; }}
  .rsl-lbl {{ font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: {ICE_DIM}; }}
  div[data-testid="stMetricValue"] {{ font-family: {FONT}; }}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent


@st.cache_data
def load_yaml(name: str):
    with open(ROOT / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


SCENARIOS = load_yaml("scenarios.yaml")
SCEN_BY_KEY = {s["key"]: s for s in SCENARIOS}
PROVISIONS = {p["id"]: p for p in load_yaml("provisions.yaml")}


# ------------------------------------------------------- compatibility helpers

def _st_version() -> tuple[int, int]:
    try:
        parts = st.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return 1, 0


NEW_WIDTH_API = _st_version() >= (1, 49)   # width="stretch" replaced use_container_width in 1.49


def plot(fig):
    if NEW_WIDTH_API:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def table(df, **kw):
    if NEW_WIDTH_API:
        st.dataframe(df, width="stretch", **kw)
    else:
        st.dataframe(df, use_container_width=True, **kw)


def theme(fig, height=520):
    fig.update_layout(paper_bgcolor=NAVY, plot_bgcolor=NAVY2, height=height,
                      font=dict(color=ICE, family=FONT, size=13),
                      colorway=[GOLD, ICE, "#8FA3D6", "#E8C983", "#7A8BB8"],
                      legend=dict(orientation="h", y=1.04, x=0, bgcolor="rgba(0,0,0,0)"),
                      margin=dict(l=40, r=20, t=50, b=30), hovermode="x unified")
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor="rgba(202,220,252,0.25)", linecolor=GRID)
    return fig


# ------------------------------------------------------------- session state

WIDGET_DEFAULTS = {
    "preset": "equity", "event_shift": 0, "seg1": 8.0, "seg2": 1.35135, "v0": 100.0, "fee": 0.0,
    "basis": "gross", "frequency": "monthly", "method": "dietz", "threshold": 10.0, "revalue_large": True,
    "window_start": date(2020, 12, 31), "window_end": date(2025, 12, 31), "annualise": False,
    "lead_with": "TWR", "vehicle": "segregated", "controls_flows": False,
    "shift_months": 0, "shift_days": 0,
}


def flows_df_from_cfg(flows: list[dict]) -> pd.DataFrame:
    rows = [{"Date": pd.Timestamp(f["date"]), "Direction": f.get("direction", "Subscription"),
             "Amount": float(f["amount"]), "Label": f.get("label", "")} for f in (flows or [])]
    if not rows:
        return pd.DataFrame({"Date": pd.Series([], dtype="datetime64[ns]"), "Direction": pd.Series([], dtype="object"),
                             "Amount": pd.Series([], dtype="float"), "Label": pd.Series([], dtype="object")})
    return pd.DataFrame(rows)


def apply_config(cfg: dict):
    for k, v in WIDGET_DEFAULTS.items():
        st.session_state[k] = cfg.get(k, v)
    st.session_state["threshold"] = float(cfg.get("threshold", 0) or 0)
    st.session_state["fee"] = float(cfg.get("fee", 0) or 0)
    st.session_state["v0"] = float(cfg.get("v0", 100))
    st.session_state["seg1"] = float(cfg.get("seg1", 8.0))
    st.session_state["seg2"] = float(cfg.get("seg2", 1.35135))
    st.session_state["shift_months"] = 0
    st.session_state["shift_days"] = 0
    st.session_state["flows_df"] = flows_df_from_cfg(cfg.get("flows", []))
    st.session_state["flows_version"] = st.session_state.get("flows_version", 0) + 1


def load_selected_scenario():
    apply_config(SCEN_BY_KEY[st.session_state["scenario_key"]]["config"])


def swap_segments():
    a, b = st.session_state["seg1"], st.session_state["seg2"]
    st.session_state["seg1"], st.session_state["seg2"] = b, a


def reset_segments():
    st.session_state["seg1"], st.session_state["seg2"] = 8.0, 1.35135


def on_preset_change():
    p = st.session_state["preset"]
    if p == "card_base":
        st.session_state["window_start"] = date(2025, 5, 31)
        st.session_state["window_end"] = date(2025, 6, 30)
        st.session_state["flows_df"] = flows_df_from_cfg([{"date": date(2025, 6, 15), "direction": "Subscription",
                                                           "amount": 40, "label": "Flow at day 15"}])
        st.session_state["flows_version"] += 1
    else:
        st.session_state["window_start"] = date(2020, 12, 31)
        st.session_state["window_end"] = date(2025, 12, 31)
        if st.session_state.get("_prev_preset") == "card_base":
            st.session_state["flows_df"] = flows_df_from_cfg([{"date": date(2022, 3, 31), "direction": "Subscription",
                                                               "amount": 20, "label": "Top-up"}])
            st.session_state["flows_version"] += 1
    st.session_state["shift_months"] = 0
    st.session_state["shift_days"] = 0
    st.session_state["_prev_preset"] = p


if "initialised" not in st.session_state:
    st.session_state["initialised"] = True
    st.session_state["scenario_key"] = "free_play"
    st.session_state["flows_version"] = 0
    apply_config(SCEN_BY_KEY["free_play"]["config"])
    st.session_state["_prev_preset"] = "equity"

# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.markdown('<div class="rsl-kicker">GIPS Standards for Firms · Session 4</div>', unsafe_allow_html=True)
    st.markdown("## Return Sensitivity Lab")

    st.selectbox("Scenario", options=[s["key"] for s in SCENARIOS],
                 format_func=lambda k: SCEN_BY_KEY[k]["name"], key="scenario_key", on_change=load_selected_scenario)
    st.button("Reset this scenario", on_click=load_selected_scenario)

    st.markdown("---")
    st.markdown("**Market path**")
    preset_keys = list(E.PRESETS.keys())
    st.radio("Return path", preset_keys, format_func=lambda k: E.PRESETS[k]["label"], key="preset",
             on_change=on_preset_change)
    if st.session_state["preset"] == "card_base":
        st.number_input("Return up to the flow date (%)", min_value=-50.0, max_value=50.0, step=0.5, key="seg1",
                        format="%.5f", help="Base case: +8.0% over days 1-15.")
        st.number_input("Return after the flow date (%)", min_value=-50.0, max_value=50.0, step=0.5, key="seg2",
                        format="%.5f", help="Base case: +1.35135% over days 16-30, so 108 + 40 grows to 150.")
        cs1, cs2 = st.columns(2)
        with cs1:
            st.button("Swap segments", on_click=swap_segments)
        with cs2:
            st.button("Base case", on_click=reset_segments)
    else:
        st.slider("Move the drawdown (months)", -12, 12, key="event_shift",
                  help="Shifts the record's drawdown and recovery earlier or later. Private markets: whole quarters.")

    st.markdown("**Portfolio**")
    st.number_input("Opening value (m)", min_value=0.0, step=5.0, key="v0")
    st.number_input("Management fee (% p.a.)", min_value=0.0, max_value=5.0, step=0.05, key="fee")
    st.radio("Basis", ["gross", "net"], key="basis", horizontal=True,
             format_func=lambda x: {"gross": "Gross of fees", "net": "Net of fees"}[x])

    st.markdown("**Calculation**")
    st.radio("Time-weighted method", ["dietz", "exact"], key="method", horizontal=True,
             format_func=lambda x: {"dietz": "Modified Dietz, linked", "exact": "Revalue at every flow"}[x])
    st.radio("Valuation frequency", ["daily", "monthly", "quarterly"], key="frequency", horizontal=True)
    st.number_input("Large cash flow threshold (% of portfolio; 0 = no policy)", min_value=0.0, max_value=100.0,
                    step=1.0, key="threshold")
    st.checkbox("Apply the policy: revalue at flows above the threshold", key="revalue_large")

    st.markdown("**Presentation**")
    st.checkbox("Annualise a sub-year window (demonstration)", key="annualise")
    st.radio("Figure the manager wants to lead with", ["TWR", "MWR"], key="lead_with", horizontal=True)
    st.radio("Vehicle", ["segregated", "closed_end"], key="vehicle",
             format_func=lambda x: {"segregated": "Segregated / open-ended", "closed_end": "Closed-end, fixed life or fixed commitment"}[x])
    st.checkbox("Firm controls the timing of external cash flows", key="controls_flows")

# ------------------------------------------------------------------ compute

spec = E.PathSpec(preset=st.session_state["preset"],
                  start=date(2025, 5, 31) if st.session_state["preset"] == "card_base" else date(2020, 12, 31),
                  event_shift=int(st.session_state.get("event_shift", 0)),
                  seg1=st.session_state["seg1"] / 100.0, seg2=st.session_state["seg2"] / 100.0)
index_gross = E.build_path(spec)
fee = float(st.session_state["fee"]) / 100.0
index_used = E.apply_fee(index_gross, fee) if st.session_state["basis"] == "net" else index_gross
path_start, path_end = index_used.index[0], index_used.index[-1]

# clamp the window to the path before the date widgets are drawn
ws, we = st.session_state["window_start"], st.session_state["window_end"]
ws = min(max(ws, path_start), path_end - timedelta(days=1))
we = min(max(we, ws + timedelta(days=1)), path_end)
st.session_state["window_start"], st.session_state["window_end"] = ws, we

scen = SCEN_BY_KEY[st.session_state["scenario_key"]]

# ------------------------------------------------------------------- header

st.markdown('<div class="rsl-kicker">Return Sensitivity Lab</div>', unsafe_allow_html=True)
st.markdown(f"## {scen['name']}")
if scen.get("card"):
    st.caption(scen["card"])
st.markdown(f'<div class="rsl-card">{scen["brief"]}</div>', unsafe_allow_html=True)

tab_lab, tab_comp, tab_prov, tab_notes = st.tabs(["Lab", "Composite & calculators", "Provisions", "Notes"])

# ================================================================== LAB TAB
with tab_lab:
    c_flows, c_window = st.columns([3, 2])
    with c_flows:
        st.markdown('<div class="rsl-lbl">External cash flows</div>', unsafe_allow_html=True)
        col_cfg = {
            "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD", min_value=path_start, max_value=path_end),
            "Direction": st.column_config.SelectboxColumn("Direction", options=["Subscription", "Redemption"], required=True),
            "Amount": st.column_config.NumberColumn("Amount (m)", min_value=0.0, format="%.2f"),
            "Label": st.column_config.TextColumn("Label"),
        }
        edited = st.data_editor(st.session_state["flows_df"], num_rows="dynamic", column_config=col_cfg,
                                key=f"flows_editor_{st.session_state['flows_version']}", hide_index=True)
        s1, s2 = st.columns(2)
        with s1:
            if st.session_state["preset"] == "card_base":
                st.slider("Shift all flows (days)", -14, 15, key="shift_days",
                          help="The base flow is at day 15; -14 puts it at day 1, +15 at day 30.")
            else:
                st.slider("Shift all flows (months)", -12, 12, key="shift_months")
        with s2:
            if st.session_state["preset"] == "card_base":
                st.caption("Day 1 to day 30 of the card's month. The window and the market path stay where they are.")
            else:
                st.slider("Shift all flows (days)", -30, 30, key="shift_days")
    with c_window:
        st.markdown('<div class="rsl-lbl">Presentation period</div>', unsafe_allow_html=True)
        st.date_input("Window start", min_value=path_start, max_value=path_end - timedelta(days=1), key="window_start")
        st.date_input("Window end", min_value=path_start + timedelta(days=1), max_value=path_end, key="window_end")
        st.caption(f"Record available: {path_start.isoformat()} to {path_end.isoformat()}. "
                   f"A flow dated on the window start is inside the opening value.")

    flows = E.flows_from_df(edited, int(st.session_state.get("shift_months", 0)), int(st.session_state.get("shift_days", 0)))
    values = E.portfolio_values(index_used, float(st.session_state["v0"]), flows)
    ws, we = st.session_state["window_start"], st.session_state["window_end"]
    if we <= ws:
        we = ws + timedelta(days=1)
    thr = float(st.session_state["threshold"]) / 100.0 or None
    res = E.evaluate_window(index_used, values, flows, ws, we, frequency=st.session_state["frequency"],
                            method=st.session_state["method"], threshold=thr,
                            revalue_large=bool(st.session_state["revalue_large"]))
    res_other_basis = None
    if fee > 0:
        other_index = index_gross if st.session_state["basis"] == "net" else E.apply_fee(index_gross, fee)
        other_values = E.portfolio_values(other_index, float(st.session_state["v0"]), flows)
        res_other_basis = E.evaluate_window(other_index, other_values, flows, ws, we, frequency=st.session_state["frequency"],
                                            method=st.session_state["method"], threshold=thr,
                                            revalue_large=bool(st.session_state["revalue_large"]))

    cfg = {k: st.session_state[k] for k in WIDGET_DEFAULTS}
    cfg["fee"] = fee
    cfg["preset"] = st.session_state["preset"]

    # ---------------------------------------------------------- headline
    show_ann = res.is_annualisable or bool(st.session_state["annualise"])

    def fmt_pair(r):
        if not np.isfinite(r):
            return "n/a", ""
        if show_ann:
            return f"{res.annualised(r) * 100:.2f}% p.a.", f"{r * 100:.2f}% over the window"
        return f"{r * 100:.2f}%", f"{res.days} days, not annualised"

    basis_lbl = "gross" if st.session_state["basis"] == "gross" else "net"
    m1, m2, m3, m4 = st.columns(4)
    v, cap = fmt_pair(res.twr_exact)
    m1.metric(f"Exact time-weighted ({basis_lbl})", v)
    m1.caption(cap + (f" · from first capital on {res.first_capital.isoformat()}" if res.first_capital else ""))
    v, cap = fmt_pair(res.twr_reported)
    lbl = "TWR as calculated - revalued at each flow" if res.twr_method == "exact" else f"TWR as calculated - Modified Dietz, {st.session_state['frequency']}"
    err_bp = (res.twr_reported - res.twr_exact) * 1e4 if np.isfinite(res.twr_reported) else float("nan")
    m2.metric(lbl, v, delta=(f"{err_bp:+.0f} bp vs exact" if np.isfinite(err_bp) else None), delta_color="off")
    m2.caption(cap)
    v, cap = fmt_pair(res.mwr)
    m3.metric(f"Money-weighted ({basis_lbl})", v)
    m3.caption(cap + (f" · solver: {res.mwr_method}" if res.mwr_method != "newton" else ""))
    gap = res.gap
    g_show = (res.annualised(res.mwr) - res.annualised(res.twr_exact)) if (show_ann and np.isfinite(res.mwr)) else gap
    m4.metric("MWR minus exact TWR", f"{g_show * 100:+.2f} pp" if np.isfinite(g_show) else "n/a",
              delta=("timing flatters the manager" if gap > 0.0005 else "timing hurt the client" if gap < -0.0005 else "no material timing effect"),
              delta_color="off")
    m4.caption(f"Opening {res.v_start:,.2f} · closing {res.v_end:,.2f} · {len(res.flows)} flow(s) in window")
    if res_other_basis is not None:
        other = "net" if basis_lbl == "gross" else "gross"
        st.caption(f"Same window {other} of the {fee * 100:.2f}% fee: exact TWR {E.pct(res_other_basis.twr_exact)}, "
                   f"MWR {E.pct(res_other_basis.mwr)}"
                   + (f" ({E.pct(res_other_basis.annualised(res_other_basis.twr_exact))} and "
                      f"{E.pct(res_other_basis.annualised(res_other_basis.mwr))} annualised)." if res.is_annualisable else "."))
    if not res.is_annualisable and st.session_state["annualise"]:
        st.markdown('<div class="rsl-flag">Annualised figures on a window of less than one year are shown for demonstration only. '
                    '2.A.12: returns for periods of less than one year must not be annualised.</div>', unsafe_allow_html=True)

    # ------------------------------------------------------------- chart
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.06)
    x = list(res.twr_path.index)
    fig.add_trace(go.Scatter(x=x, y=(res.twr_path.values - 1) * 100, name="Time-weighted cumulative (the market path)",
                             line=dict(color=GOLD, width=2.4), hovertemplate="%{y:.2f}%<extra>TWR</extra>"), row=1, col=1)
    if np.isfinite(res.mwr):
        fig.add_trace(go.Scatter(x=x, y=(res.mwr_path.values - 1) * 100, name="Money-weighted implied (the investor's rate)",
                                 line=dict(color=ICE, width=2.0, dash="dot"), hovertemplate="%{y:.2f}%<extra>MWR implied</extra>"), row=1, col=1)
    if len(res.flows):
        subs = res.flows[res.flows["Amount"] > 0]
        reds = res.flows[res.flows["Amount"] < 0]
        for df_, name, sym in ((subs, "Subscription", "triangle-up"), (reds, "Redemption", "triangle-down")):
            if len(df_):
                fig.add_trace(go.Scatter(x=list(df_["Date"]), y=[(res.twr_path.loc[d] - 1) * 100 for d in df_["Date"]],
                                         mode="markers", name=name,
                                         marker=dict(symbol=sym, size=13, color=GOLD if name == "Subscription" else ICE,
                                                     line=dict(color=NAVY, width=1)),
                                         text=[f"{a:+,.1f}" for a in df_["Amount"]],
                                         hovertemplate="%{text} on %{x}<extra>" + name + "</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=res.values.values, name="Portfolio value", fill="tozeroy",
                             line=dict(color=ICE, width=1.4), fillcolor="rgba(202,220,252,0.10)",
                             hovertemplate="%{y:,.1f}<extra>Value</extra>"), row=2, col=1)
    if len(res.flows):
        fig.add_trace(go.Bar(x=list(res.flows["Date"]), y=list(res.flows["Amount"]), name="External cash flow",
                             marker_color=GOLD, width=(res.days / 90) * 86400000, hovertemplate="%{y:+,.1f}<extra>Flow</extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Cumulative %", row=1, col=1, ticksuffix="%")
    fig.update_yaxes(title_text="Value (m)", row=2, col=1)
    theme(fig, height=560)
    plot(fig)

    # ------------------------------------------ narrative and disclosures
    items = E.narrative(res, index_used, cfg)
    discl = E.disclosure_prompt(res, cfg)
    cited = []
    for it in items + discl:
        for p in it.get("provisions", []):
            if p not in cited:
                cited.append(p)

    cn, cd = st.columns([3, 2])
    with cn:
        st.markdown('<div class="rsl-lbl">What the configuration says</div>', unsafe_allow_html=True)
        for it in items:
            chips = "".join(f'<span class="rsl-chip">{p}</span>' for p in it.get("provisions", []))
            st.markdown(f'<div class="rsl-{it["level"]}">{it["text"]}<br>{chips}</div>', unsafe_allow_html=True)
    with cd:
        st.markdown('<div class="rsl-lbl">Disclosures the reader would need</div>', unsafe_allow_html=True)
        for it in discl:
            chips = "".join(f'<span class="rsl-chip">{p}</span>' for p in it.get("provisions", []))
            st.markdown(f'<div class="rsl-note">{it["text"]}<br>{chips}</div>', unsafe_allow_html=True)

    with st.expander("Provisions cited on this screen"):
        for pid in cited:
            p = PROVISIONS.get(pid)
            if p:
                st.markdown(f"**{pid}** · {p['type']} · <span style='color:{ICE_DIM}'>{p['section']}</span>", unsafe_allow_html=True)
                st.markdown(p["text"])

    # ------------------------------------------------------------ detail
    with st.expander("Working: sub-periods, flows and the flow-by-flow decomposition"):
        st.markdown('<div class="rsl-lbl">Sub-periods used for the calculated TWR</div>', unsafe_allow_html=True)
        sub = res.subperiods.copy()
        for c in ("Sub-period return", "Exact sub-period"):
            sub[c] = sub[c] * 100
        table(sub.style.format({"Opening": "{:,.3f}", "Closing": "{:,.3f}", "Flows": "{:+,.2f}", "Weighted flows": "{:+,.3f}",
                                "Sub-period return": "{:.4f}%", "Exact sub-period": "{:.4f}%"}), hide_index=True)
        if len(res.flows):
            st.markdown('<div class="rsl-lbl">Flows in the window</div>', unsafe_allow_html=True)
            fl = res.flows.copy()
            fl["Elapsed"] = fl["Elapsed"] * 100
            fl["Dietz weight"] = fl["Dietz weight"]
            fl["Size vs value"] = fl["Size vs value"] * 100
            fl["Return earned after flow"] = fl["Return earned after flow"] * 100
            table(fl.style.format({"Amount": "{:+,.2f}", "Elapsed": "{:.1f}% of window", "Dietz weight": "{:.4f}",
                                   "Size vs value": "{:.1f}%", "Return earned after flow": "{:.2f}%"}), hide_index=True)
            st.markdown('<div class="rsl-lbl">Contribution of each flow to the MWR minus TWR gap (added in date order)</div>',
                        unsafe_allow_html=True)
            cdf = res.contributions
            wf = go.Figure(go.Waterfall(x=["Exact TWR"] + list(cdf["Flow"]) + ["MWR"],
                                        measure=["absolute"] + ["relative"] * len(cdf) + ["total"],
                                        y=[res.twr_exact * 100] + list(cdf["Contribution"] * 100) + [0],
                                        increasing=dict(marker=dict(color=GOLD)), decreasing=dict(marker=dict(color=ICE)),
                                        totals=dict(marker=dict(color="#8FA3D6")), connector=dict(line=dict(color=GRID)),
                                        texttemplate="%{y:+.2f}", textposition="outside"))
            wf.update_yaxes(ticksuffix="%")
            theme(wf, height=380)
            wf.update_layout(showlegend=False)
            plot(wf)
            st.caption("Each bar is the change in the money-weighted return when that flow is added, holding the market path fixed. "
                       "The bars sum exactly to the gap; the order is the date order, so an early flow's bar also carries some of "
                       "the interaction with later flows.")

    st.download_button("Download this configuration's series (CSV)",
                       data=pd.DataFrame({"date": x, "index": index_used.loc[ws:we].values, "twr_cumulative": res.twr_path.values,
                                          "mwr_implied": res.mwr_path.values, "portfolio_value": res.values.values}).to_csv(index=False),
                       file_name="return_sensitivity_lab_series.csv", mime="text/csv")

# ============================================================ COMPOSITE TAB
with tab_comp:
    st.markdown("### Asset-weighted against equal-weighted (card 6)")
    st.caption("One composite. Edit the book; the large account's return is the lever on the card (6.0% to 11.5%).")
    if "card6_df" not in st.session_state:
        st.session_state["card6_df"] = default_card6_book()
    c6a, c6b = st.columns([2, 3])
    with c6a:
        book6 = st.data_editor(st.session_state["card6_df"], num_rows="dynamic", hide_index=True, key="card6_editor",
                               column_config={"Beginning value": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
                                              "Return %": st.column_config.NumberColumn(format="%.2f")})
    r6 = asset_vs_equal(book6)
    with c6b:
        if r6.get("n", 0) > 0:
            k1, k2, k3 = st.columns(3)
            k1.metric("Asset-weighted composite return", f"{r6['aw'] * 100:.2f}%")
            k2.metric("Equal-weighted composite return", f"{r6['ew'] * 100:.2f}%")
            k3.metric("Equal minus asset-weighted", f"{r6['gap'] * 100:+.2f} pp")
            k1.metric("Asset-weighted std dev", f"{r6['aw_sd'] * 100:.2f}%")
            k2.metric("Equal-weighted std dev", f"{r6['ew_sd'] * 100:.2f}%")
            k3.metric("Largest account's share", f"{r6['share_largest'] * 100:.0f}%")
            bars = go.Figure()
            d6 = book6.dropna(subset=["Beginning value", "Return %"])
            bars.add_trace(go.Bar(x=list(d6["Portfolio"]), y=list(d6["Return %"]), name="Portfolio return",
                                  marker_color=[GOLD if v == d6["Beginning value"].max() else ICE for v in d6["Beginning value"]],
                                  hovertemplate="%{y:.2f}%<extra></extra>"))
            bars.add_hline(y=r6["aw"] * 100, line=dict(color=GOLD, dash="dot"), annotation_text="asset-weighted", annotation_font_color=GOLD)
            bars.add_hline(y=r6["ew"] * 100, line=dict(color=ICE, dash="dot"), annotation_text="equal-weighted", annotation_font_color=ICE)
            bars.update_yaxes(ticksuffix="%")
            theme(bars, height=320)
            bars.update_layout(showlegend=False)
            plot(bars)
        st.markdown('<div class="rsl-note">Asset-weighted says what the strategy produced for the money; equal-weighted says '
                    'what it produced for the typical client. 2.A.36 requires the asset-weighted figure; 4.B.2(b) recommends '
                    'presenting the equal-weighted one alongside it.'
                    '<br><span class="rsl-chip">2.A.36</span><span class="rsl-chip">4.B.2</span></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### The three composite routes in 2.A.36 (card 7)")
    st.caption("One period of 30 days. 'Portfolio return %' is each portfolio's own calculated return (revalued at its flow); "
               "routes (a) and (b) weight those. Route (c) uses only the values. P4 joins mid-period with no beginning value.")
    if "card7_df" not in st.session_state:
        st.session_state["card7_df"] = default_card7_book()
    book7 = st.data_editor(st.session_state["card7_df"], num_rows="dynamic", hide_index=True, key="card7_editor",
                           column_config={"Beginning value": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
                                          "Flow": st.column_config.NumberColumn(format="%.1f", help="+ in / - out"),
                                          "Flow day": st.column_config.NumberColumn(format="%d", min_value=0, max_value=30),
                                          "Ending value": st.column_config.NumberColumn(format="%.1f", min_value=0.0),
                                          "Portfolio return %": st.column_config.NumberColumn(format="%.3f")})
    r7 = composite_routes(book7, days=30)
    q1, q2, q3 = st.columns(3)
    q1.metric("(a) Beginning-value weighted", f"{r7['a'] * 100:.4f}%" if np.isfinite(r7["a"]) else "n/a")
    q2.metric("(b) Beginning value plus weighted flows", f"{r7['b'] * 100:.4f}%" if np.isfinite(r7["b"]) else "n/a")
    q3.metric("(c) Aggregate method", f"{r7['c'] * 100:.4f}%" if np.isfinite(r7["c"]) else "n/a")
    t7 = r7["table"][["Portfolio", "Beginning value", "Flow", "Flow day", "w", "Weighted flow", "Dietz denominator",
                      "Ending value", "Gain", "Own Modified Dietz %", "Portfolio return %"]]
    table(t7.style.format({"Beginning value": "{:,.1f}", "Flow": "{:+,.1f}", "Flow day": "{:.0f}", "w": "{:.3f}",
                           "Weighted flow": "{:+,.2f}", "Dietz denominator": "{:,.2f}", "Ending value": "{:,.1f}",
                           "Gain": "{:+,.2f}", "Own Modified Dietz %": "{:.4f}", "Portfolio return %": "{:.3f}"}), hide_index=True)
    excl = ", ".join(r7["excluded_from_a"]) if r7["excluded_from_a"] else "none"
    st.markdown(f'<div class="rsl-note">Route (a) gives no weight to a portfolio without a beginning value - excluded here: {excl}. '
                f'Route (b) weights each portfolio by beginning value plus day-weighted flows. Route (c) treats the composite as one '
                f'portfolio: gain {r7["agg_num"]:+,.2f} over a Modified Dietz denominator of {r7["agg_den"]:,.2f}. If every '
                f'portfolio return were itself the Modified Dietz figure over the same period, route (b) would equal route (c) '
                f'exactly ({r7["b_if_dietz"] * 100:.4f}%); the difference on screen is the revaluation inside the portfolio returns. '
                f'Whichever route the firm has documented is applied consistently (1.A.5); the choice is the firm\'s, and a '
                f'change is a policy change with a date, not a recalculation of what was already reported.'
                f'<br><span class="rsl-chip">2.A.36</span><span class="rsl-chip">1.A.5</span><span class="rsl-chip">4.C.16</span></div>',
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Annualising a quarter (card 5)")
    a1, a2 = st.columns([1, 2])
    with a1:
        q = st.number_input("Quarterly return (%)", value=6.2, step=0.1, key="ann_q")
        q_list = st.text_input("Four consecutive quarters (%)", value="6.2, 1.1, -3.4, 2.0", key="ann_list")
    with a2:
        try:
            qs = [float(v.strip()) / 100 for v in q_list.split(",") if v.strip()]
        except ValueError:
            qs = []
        b1, b2, b3 = st.columns(3)
        b1.metric("Single quarter annualised", f"{annualise_quarter(q / 100) * 100:.2f}%")
        b2.metric("Four quarters compounded", f"{compound(qs) * 100:.2f}%" if qs else "n/a")
        b3.metric("Simple sum of the four", f"{sum(qs) * 100:.2f}%" if qs else "n/a")
        st.markdown('<div class="rsl-note">The annualised single quarter asserts that the quarter repeats four times. '
                    'The compounded figure reports what happened. Returns for periods of less than one year must not be '
                    'annualised (2.A.12), and not in an advertisement either (8.A.4).'
                    '<br><span class="rsl-chip">2.A.12</span><span class="rsl-chip">8.A.4</span></div>', unsafe_allow_html=True)

# =========================================================== PROVISIONS TAB
with tab_prov:
    st.markdown("### Provisions referenced by the lab")
    st.caption("Text from the 2020 GIPS Standards for Firms, as extracted in the course workbook. Working reference only.")
    q = st.text_input("Filter (provision number or a word in the text)", value="", key="prov_filter")
    for pid, p in PROVISIONS.items():
        if q and q.lower() not in pid.lower() and q.lower() not in p["text"].lower():
            continue
        st.markdown(f"**{pid}** · {p['type']} · <span style='color:{ICE_DIM}'>{p['section']}</span>", unsafe_allow_html=True)
        st.markdown(p["text"])
        st.markdown("")

# ================================================================ NOTES TAB
with tab_notes:
    st.markdown("### How the lab computes")
    st.markdown("""
- **Grid.** Calendar-daily valuation points. The opening date carries an index of 100.
- **Flows.** A flow dated D is applied at the close of D, is inside the value at D, and earns the market return from D+1. A flow dated on the window start is inside the opening value.
- **Market path.** Fixed by the preset (a seeded, deterministic series; the drawdown and its recovery are placed at a chosen month and can be moved). Moving a flow changes portfolio values, Modified Dietz and the money-weighted return. It does not change the exact time-weighted return.
- **Exact time-weighted return.** Revalued at every flow and geometrically linked - equal to the index ratio over the window.
- **Modified Dietz.** Inside each valuation sub-period (daily, monthly or quarterly), flows are day-weighted by the fraction of the sub-period remaining; sub-periods are linked. If a large cash flow policy is set and applied, a sub-period boundary is added at each flow at or above the threshold. Modified Dietz is a money-weighted approximation used as the building block; it is not itself a time-weighted measure.
- **Money-weighted return.** The internal rate of return on daily-dated flows over the window, solved by Newton-Raphson with a bisection fallback. Shown for the period where the window is under one year, annualised where it is one year or more. A sub-year figure can be annualised for demonstration and is flagged.
- **Fees.** A model management fee accrued daily against the gross path.
- **Card base case.** Thirty days, opening 100, +8.0% over days 1-15, +1.35135% over days 16-30, so that 108 + 40 closes at 150: exact 9.4595%, Modified Dietz 8.3333%, money-weighted 8.3613%.
- **Composite tab.** Card 6: asset- and equal-weighted returns and both dispersion lenses. Card 7: the three 2.A.36 routes on a 30-day period. Card 5: annualisation arithmetic.

**Not modelled today.** Benchmark comparison; temporary new accounts or removal for significant cash flows (3.A.12, 3.A.13); performance-based fees; multi-currency effects.
""")
    st.markdown("### Scenario presets")
    st.markdown("Presets live in `scenarios.yaml` and can be edited without touching the code. The provision text is in `provisions.yaml`.")
