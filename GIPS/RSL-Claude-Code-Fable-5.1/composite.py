"""Composite-level calculations: asset- versus equal-weighting (card 6), the three
composite return routes in 2.A.36 (card 7), and the sub-year annualisation
arithmetic (card 5)."""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- card 6


def default_card6_book() -> pd.DataFrame:
    rows = [{"Portfolio": "Large account", "Beginning value": 400.0, "Return %": 6.00}]
    rows += [{"Portfolio": f"Account {i}", "Beginning value": 20.0, "Return %": 11.29} for i in range(1, 12)]
    return pd.DataFrame(rows)


def asset_vs_equal(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["Beginning value", "Return %"])
    d = d[d["Beginning value"] > 0]
    if len(d) == 0:
        return {"n": 0}
    w = d["Beginning value"].values.astype(float)
    r = d["Return %"].values.astype(float) / 100.0
    aw = float(np.sum(w * r) / np.sum(w))
    ew = float(np.mean(r))
    # internal dispersion, both lenses (population form, as in the Session 5 material)
    ew_sd = float(np.sqrt(np.mean((r - ew) ** 2)))
    aw_sd = float(np.sqrt(np.sum(w * (r - aw) ** 2) / np.sum(w)))
    return {"n": int(len(d)), "aw": aw, "ew": ew, "gap": ew - aw, "ew_sd": ew_sd, "aw_sd": aw_sd,
            "high": float(r.max()), "low": float(r.min()), "share_largest": float(w.max() / w.sum())}


# ----------------------------------------------------------------- card 7


def default_card7_book() -> pd.DataFrame:
    # One period of 30 days. 'Portfolio return %' is the portfolio's own calculated return
    # (revalued at its flow) - the input to routes (a) and (b). Route (c) uses values only.
    return pd.DataFrame([
        {"Portfolio": "P1", "Beginning value": 500.0, "Flow": 0.0, "Flow day": 0, "Ending value": 530.0, "Portfolio return %": 6.000},
        {"Portfolio": "P2", "Beginning value": 300.0, "Flow": 30.0, "Flow day": 10, "Ending value": 345.0, "Portfolio return %": 4.900},
        {"Portfolio": "P3", "Beginning value": 200.0, "Flow": -40.0, "Flow day": 20, "Ending value": 170.0, "Portfolio return %": 5.250},
        {"Portfolio": "P4 (joins mid-period)", "Beginning value": 0.0, "Flow": 150.0, "Flow day": 15, "Ending value": 156.0, "Portfolio return %": 4.000},
    ])


def composite_routes(df: pd.DataFrame, days: int = 30) -> dict:
    d = df.dropna(subset=["Beginning value", "Ending value"]).copy()
    d["Flow"] = d["Flow"].fillna(0.0).astype(float)
    d["Flow day"] = d["Flow day"].fillna(0).astype(float)
    d["w"] = np.where(d["Flow"] != 0, (days - d["Flow day"]) / days, 0.0)
    d["Weighted flow"] = d["w"] * d["Flow"]
    d["Dietz denominator"] = d["Beginning value"] + d["Weighted flow"]
    d["Gain"] = d["Ending value"] - d["Beginning value"] - d["Flow"]
    with np.errstate(divide="ignore", invalid="ignore"):
        d["Own Modified Dietz %"] = np.where(d["Dietz denominator"] > 0, d["Gain"] / d["Dietz denominator"] * 100, np.nan)
    r = d["Portfolio return %"].astype(float).values / 100.0
    vb = d["Beginning value"].astype(float).values
    wb = d["Dietz denominator"].values

    route_a = float(np.sum(vb * r) / np.sum(vb)) if np.sum(vb) > 0 else float("nan")
    route_b = float(np.sum(wb * r) / np.sum(wb)) if np.sum(wb) > 0 else float("nan")
    agg_num = float(d["Ending value"].sum() - d["Beginning value"].sum() - d["Flow"].sum())
    agg_den = float(d["Beginning value"].sum() + d["Weighted flow"].sum())
    route_c = agg_num / agg_den if agg_den > 0 else float("nan")
    # route (b) with the portfolios' own Modified Dietz returns equals route (c) exactly
    r_dietz = d["Own Modified Dietz %"].fillna(0).values / 100.0
    route_b_dietz = float(np.sum(wb * r_dietz) / np.sum(wb)) if np.sum(wb) > 0 else float("nan")
    excluded = d[d["Beginning value"] <= 0]["Portfolio"].tolist()
    return {"table": d, "a": route_a, "b": route_b, "c": route_c, "b_if_dietz": route_b_dietz,
            "excluded_from_a": excluded, "agg_num": agg_num, "agg_den": agg_den}


# ----------------------------------------------------------------- card 5


def annualise_quarter(q: float) -> float:
    return (1 + q) ** 4 - 1


def compound(returns: list[float]) -> float:
    out = 1.0
    for r in returns:
        out *= (1 + r)
    return out - 1
