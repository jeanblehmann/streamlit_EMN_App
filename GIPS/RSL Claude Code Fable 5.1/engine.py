"""Return Sensitivity Lab - calculation engine.

Conventions
-----------
* The valuation grid is calendar-daily. Point 0 is the opening valuation date
  (index = 100). Every later calendar day is a valuation point.
* An external cash flow dated D is applied at the close of D: it is included in
  the value at D and earns the market return from D+1 onward. A flow dated on the
  window start is therefore part of the opening value, not a flow in the window.
* The market path is fixed. Moving a flow changes portfolio values, Modified Dietz
  and the money-weighted return; it does not change the exact time-weighted return.
* Modified Dietz is a money-weighted approximation used here as the sub-period
  building block inside a linked time-weighted return. It is not itself a
  time-weighted measure.
"""
from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- dates


def month_end(y: int, m: int) -> date:
    return date(y, m, calendar.monthrange(y, m)[1])


def add_months(d: date, n: int) -> date:
    """Shift a date by n months, clamping the day to the target month length."""
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    day = min(d.day, calendar.monthrange(y, m)[1])
    return date(y, m, day)


def is_month_end(d: date) -> bool:
    return d.day == calendar.monthrange(d.year, d.month)[1]


def is_quarter_end(d: date) -> bool:
    return is_month_end(d) and d.month in (3, 6, 9, 12)


# ------------------------------------------------------------------ path presets

PRESETS = {
    "equity": {
        "label": "Equity-like",
        # monthly drift by calendar year of the record (2021..2025), noise, mild mean reversion
        "drift": [0.018, 0.009, 0.009, 0.003, 0.008], "sigma": 0.024, "phi": -0.20,
        # the drawdown and its recovery, in consecutive months, replacing the base months
        "event": [-0.080, -0.120, -0.060, 0.025, 0.060, 0.045, 0.030],
        "event_month": 15,   # 0-based: month 15 of a Jan-2021 record is April 2022
        "quarterly": False,
    },
    "bond": {
        "label": "Bond-like",
        "drift": [0.003, 0.002, 0.004, 0.004, 0.004], "sigma": 0.006, "phi": 0.10,
        "event": [-0.025, -0.035, -0.020, 0.004, 0.012, 0.010],
        "event_month": 15,
        "quarterly": False,
    },
    "private_markets": {
        "label": "Private-markets-like (quarterly appraisal marks, lagged)",
        "drift": [0.035, 0.030, 0.025, 0.030, 0.028], "sigma": 0.015, "phi": 0.50,   # per quarter
        "event": [-0.040, -0.060, -0.025, 0.020, 0.035],                            # per quarter
        "event_month": 21,   # marks start falling at the Sep-2022 quarter end, two quarters after the listed market
        "quarterly": False,
    },
    "card_base": {
        "label": "Card base case (30 days: 100 -> 108 -> 150 with +40 at day 15)",
    },
}
PRESETS["private_markets"]["quarterly"] = True


@dataclass
class PathSpec:
    preset: str = "equity"
    start: date = date(2020, 12, 31)       # opening valuation date, index = 100
    months: int = 60
    event_shift: int = 0                   # months (private markets: rounded to whole quarters)
    seed: int = 42
    seg1: float = 0.08                     # card base case only: return to the flow date
    seg2: float = 150.0 / 148.0 - 1.0      # card base case only: return after the flow date
    card_days: int = 30
    card_split: int = 15


def monthly_returns(spec: PathSpec) -> np.ndarray:
    p = PRESETS[spec.preset]
    rng = np.random.default_rng(spec.seed)
    n = spec.months
    drift = p["drift"]

    def mu_at(month_i: int, per_quarter: bool) -> float:
        y = min((month_i // 12), len(drift) - 1)
        return drift[y]

    if p["quarterly"]:
        nq = n // 3
        eps = rng.normal(0.0, p["sigma"], nq)
        q = np.empty(nq)
        prev = None
        for i in range(nq):
            mu = mu_at(3 * i, True)
            prev = mu if prev is None else prev
            # appraisal smoothing: the mark carries part of the previous mark
            q[i] = mu + p["phi"] * (prev - mu) + (1 - p["phi"]) * eps[i]
            prev = q[i]
        ev_q = (p["event_month"] + spec.event_shift) // 3
        ev_q = max(0, min(nq - len(p["event"]), ev_q))
        for k, r in enumerate(p["event"]):
            q[ev_q + k] = r
        out = np.zeros(n)
        for i in range(nq):
            out[3 * i + 2] = q[i]
        return out

    eps = rng.normal(0.0, p["sigma"], n)
    r = np.empty(n)
    prev_dev = 0.0
    for i in range(n):
        mu = mu_at(i, False)
        dev = p["phi"] * prev_dev + eps[i]
        r[i] = mu + dev
        prev_dev = dev
    ev = p["event_month"] + spec.event_shift
    ev = max(0, min(n - len(p["event"]), ev))
    for k, x in enumerate(p["event"]):
        r[ev + k] = x
    return r


def build_path(spec: PathSpec) -> pd.Series:
    """Daily index (opening = 100) as a pandas Series indexed by date."""
    if spec.preset == "card_base":
        n, k = spec.card_days, spec.card_split
        f1 = (1 + spec.seg1) ** (1.0 / k)
        f2 = (1 + spec.seg2) ** (1.0 / (n - k))
        dates = [spec.start + timedelta(days=i) for i in range(n + 1)]
        idx = [100.0]
        for i in range(1, n + 1):
            idx.append(idx[-1] * (f1 if i <= k else f2))
        return pd.Series(idx, index=pd.Index(dates, name="date"), name="index")
    rets = monthly_returns(spec)
    dates = [spec.start]
    idx = [100.0]
    cur = spec.start
    for r in rets:
        nxt = add_months(month_end(cur.year, cur.month), 1)
        nxt = month_end(nxt.year, nxt.month)
        days = (nxt - cur).days
        f = (1 + r) ** (1.0 / days)
        for _ in range(days):
            cur = cur + timedelta(days=1)
            dates.append(cur)
            idx.append(idx[-1] * f)
    return pd.Series(idx, index=pd.Index(dates, name="date"), name="index")


def largest_drawdown(index: pd.Series) -> dict:
    """Peak, trough, depth and recovery date of the largest peak-to-trough fall."""
    vals = index.values
    dates = list(index.index)
    peak_i = 0
    best = {"depth": 0.0, "peak": dates[0], "trough": dates[0], "recovery": None}
    run_peak = 0
    for i in range(1, len(vals)):
        if vals[i] > vals[run_peak]:
            run_peak = i
        dd = vals[i] / vals[run_peak] - 1
        if dd < best["depth"]:
            best = {"depth": dd, "peak": dates[run_peak], "trough": dates[i], "recovery": None}
            peak_i = run_peak
    if best["depth"] < 0:
        peak_val = vals[peak_i]
        t_i = dates.index(best["trough"])
        for j in range(t_i, len(vals)):
            if vals[j] >= peak_val:
                best["recovery"] = dates[j]
                break
    return best


# ------------------------------------------------------------ portfolio values


@dataclass
class Flow:
    date: date
    amount: float          # + subscription / contribution, - redemption / withdrawal
    label: str = ""


def flows_from_df(df: pd.DataFrame, shift_months: int = 0, shift_days: int = 0) -> list[Flow]:
    out = []
    if df is None or len(df) == 0:
        return out
    for _, row in df.iterrows():
        d = row.get("Date")
        if d is None or pd.isna(d):
            continue
        if isinstance(d, pd.Timestamp):
            d = d.date()
        amt = row.get("Amount")
        if amt is None or pd.isna(amt):
            continue
        direction = str(row.get("Direction", "Subscription"))
        sign = -1.0 if direction.lower().startswith(("red", "with", "dist")) else 1.0
        d = add_months(d, shift_months) + timedelta(days=shift_days)
        out.append(Flow(d, sign * abs(float(amt)), str(row.get("Label", "") or "")))
    return sorted(out, key=lambda f: f.date)


def portfolio_values(index: pd.Series, v0: float, flows: list[Flow]) -> pd.Series:
    dates = list(index.index)
    pos = {d: i for i, d in enumerate(dates)}
    cf = np.zeros(len(dates))
    for f in flows:
        if f.date in pos:
            cf[pos[f.date]] += f.amount
    ratio = index.values[1:] / index.values[:-1]
    v = np.empty(len(dates))
    v[0] = v0 + cf[0]
    for i in range(1, len(dates)):
        v[i] = v[i - 1] * ratio[i - 1] + cf[i]
    return pd.Series(v, index=index.index, name="value")


def apply_fee(index: pd.Series, fee_pa: float) -> pd.Series:
    """Net index: the gross path reduced by a management fee accrued daily."""
    if fee_pa <= 0:
        return index
    days = np.array([(d - index.index[0]).days for d in index.index], dtype=float)
    return index * (1 - fee_pa) ** (days / 365.25)


# ------------------------------------------------------------------- returns


def modified_dietz(v_start: float, v_end: float, flows: list[tuple[float, float]]) -> tuple[float, float]:
    """flows: list of (weight, amount). Returns (return, denominator)."""
    cf = sum(a for _, a in flows)
    denom = v_start + sum(w * a for w, a in flows)
    if denom <= 0:
        # no capital at risk in the sub-period (for example a fund before its first call)
        return (0.0 if abs(v_end - v_start - cf) < 1e-9 else float("nan")), denom
    return (v_end - v_start - cf) / denom, denom


def solve_irr(v_start: float, v_end: float, flows: list[tuple[float, float]], guess: float = 0.05) -> tuple[float, str]:
    """Period IRR r solving v_start(1+r) + sum a_i (1+r)^(1-tau_i) = v_end.

    flows: list of (tau, amount) with tau in [0, 1] as the fraction of the period elapsed.
    Newton-Raphson from the guess, bisection fallback. Returns (r, method).
    """
    def f(r):
        g = 1.0 + r
        return v_start * g + sum(a * g ** (1 - t) for t, a in flows) - v_end

    def df(r):
        g = 1.0 + r
        return v_start + sum(a * (1 - t) * g ** (-t) for t, a in flows)

    r = guess
    for _ in range(60):
        try:
            fr, dr = f(r), df(r)
        except (OverflowError, ZeroDivisionError, ValueError):
            break
        if abs(dr) < 1e-14 or not np.isfinite(fr):
            break
        step = fr / dr
        r_new = r - step
        if r_new <= -0.999:
            r_new = (r - 0.999) / 2
        if abs(r_new - r) < 1e-13:
            return r_new, "newton"
        r = r_new
    # bisection fallback on a bracket that widens until the sign changes
    lo, hi = -0.999, 1.0
    flo = f(lo)
    for _ in range(40):
        if f(hi) * flo <= 0:
            break
        hi = hi * 2 + 1
    else:
        return float("nan"), "failed"
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-12 or (hi - lo) < 1e-14:
            return mid, "bisection"
        if fm * flo < 0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi), "bisection"


def annualise(r_period: float, days: int) -> float:
    return (1 + r_period) ** (365.25 / days) - 1


@dataclass
class WindowResult:
    start: date
    end: date
    days: int
    v_start: float
    v_end: float
    twr_exact: float
    twr_reported: float
    twr_method: str
    mwr: float
    mwr_method: str
    subperiods: pd.DataFrame
    flows: pd.DataFrame            # in-window flows with tau, weight, size vs value
    twr_path: pd.Series            # cumulative index, window start = 1
    mwr_path: pd.Series            # (1+mwr)^(t/T)
    values: pd.Series              # portfolio value across the window
    contributions: pd.DataFrame    # sequential MWR-TWR decomposition by flow
    unrevalued_large: list = field(default_factory=list)
    first_capital: date | None = None   # set when the portfolio opened at zero; TWR runs from here

    @property
    def gap(self) -> float:
        return self.mwr - self.twr_exact

    @property
    def is_annualisable(self) -> bool:
        return self.days >= 365

    def annualised(self, r: float) -> float:
        return annualise(r, self.days)


def _boundaries(dates: list[date], start: date, end: date, frequency: str,
                flows: list[Flow], values: pd.Series, threshold: float | None) -> list[date]:
    """Valuation sub-period boundaries: the window ends, the period ends inside it and,
    where a threshold is given, the date of every flow at or above it."""
    b = {start, end}
    for d in dates:
        if start < d < end:
            if frequency == "daily":
                b.add(d)
            elif frequency == "monthly" and is_month_end(d):
                b.add(d)
            elif frequency == "quarterly" and is_quarter_end(d):
                b.add(d)
    if threshold is not None and threshold > 0:
        for f in flows:
            if start < f.date <= end:
                prev = values.loc[f.date] - f.amount   # value just before the flow (same close)
                if prev > 0 and abs(f.amount) / prev >= threshold:
                    b.add(f.date)
    return sorted(b)


def evaluate_window(index: pd.Series, values: pd.Series, flows: list[Flow], start: date, end: date,
                    frequency: str = "monthly", method: str = "dietz",
                    threshold: float | None = None, revalue_large: bool = True) -> WindowResult:
    """Measures over [start, end].

    method      'exact'  -> revalue at every flow and link (the exact time-weighted return)
                'dietz'  -> Modified Dietz inside each valuation sub-period, linked
    frequency   'daily' | 'monthly' | 'quarterly' - the valuation sub-periods used by 'dietz'
    threshold   the firm's large cash flow threshold as a fraction of portfolio value (None = no policy)
    revalue_large  whether the policy is actually applied (False reproduces a flow left inside a sub-period)
    """
    dates = list(index.index)
    assert start in values.index and end in values.index and start < end
    T = (end - start).days
    v_s, v_e = float(values.loc[start]), float(values.loc[end])
    twr_base = start
    first_capital = None
    if v_s <= 0:
        funded = values.loc[start:end]
        funded = funded[funded > 0]
        if len(funded):
            first_capital = funded.index[0]
            twr_base = first_capital
    twr_exact = float(index.loc[end] / index.loc[twr_base]) - 1

    in_win = [f for f in flows if start < f.date <= end]

    # --- Modified Dietz linked across valuation sub-periods (with optional revaluation at large flows)
    if method == "exact":
        bounds = sorted({start, end} | {f.date for f in in_win})
    else:
        bounds = _boundaries(dates, start, end, frequency, in_win, values,
                             threshold if revalue_large else None)
    rows = []
    linked = 1.0
    for a, b in zip(bounds[:-1], bounds[1:]):
        va, vb = float(values.loc[a]), float(values.loc[b])
        fl = [((b - f.date).days / (b - a).days, f.amount) for f in in_win if a < f.date <= b]
        r, denom = modified_dietz(va, vb, fl)
        linked *= (1 + r)
        rows.append({"From": a, "To": b, "Days": (b - a).days, "Opening": va, "Closing": vb,
                     "Flows": sum(x for _, x in fl), "Weighted flows": sum(w * x for w, x in fl),
                     "Sub-period return": r, "Exact sub-period": float(index.loc[b] / index.loc[a]) - 1})
    sub = pd.DataFrame(rows)
    twr_reported = linked - 1

    # flows that sit inside a sub-period without revaluation and exceed the threshold
    unrevalued = []
    if method != "exact" and threshold:
        for f in in_win:
            prev = values.loc[f.date] - f.amount
            if prev > 0 and abs(f.amount) / prev >= threshold and f.date not in bounds:
                unrevalued.append(f)

    # --- Money-weighted (IRR on daily-dated flows)
    tau_flows = [((f.date - start).days / T, f.amount) for f in in_win]
    mwr, how = solve_irr(v_s, v_e, tau_flows, guess=twr_reported if np.isfinite(twr_reported) else 0.05)

    # --- flow table
    frows = []
    for f in in_win:
        prev = float(values.loc[f.date] - f.amount)
        frows.append({"Date": f.date, "Amount": f.amount, "Label": f.label,
                      "Elapsed": (f.date - start).days / T, "Dietz weight": (end - f.date).days / T,
                      "Size vs value": abs(f.amount) / prev if prev > 0 else float("nan"),
                      "Return earned after flow": float(index.loc[end] / index.loc[f.date]) - 1})
    fdf = pd.DataFrame(frows)

    # --- sequential decomposition of the MWR-TWR gap by flow (date order)
    crows = []
    running = twr_exact
    acc = []
    for f in in_win:
        acc.append(f)
        sl = index.loc[start:end]
        vals_k = portfolio_values(sl, v_s, [Flow(x.date, x.amount) for x in acc])
        tf = [((x.date - start).days / T, x.amount) for x in acc]
        mk, _ = solve_irr(v_s, float(vals_k.loc[end]), tf, guess=running)
        crows.append({"Flow": f"{f.date.isoformat()} {f.amount:+,.1f}", "Date": f.date,
                      "Amount": f.amount, "Contribution": mk - running})
        running = mk
    cdf = pd.DataFrame(crows)

    sl = index.loc[start:end]
    twr_path = sl / sl.loc[twr_base]
    if first_capital is not None:
        twr_path = twr_path.where(twr_path.index >= first_capital, 1.0)
    tt = np.array([(d - start).days / T for d in sl.index])
    mwr_path = pd.Series((1 + mwr) ** tt, index=sl.index) if np.isfinite(mwr) else twr_path * np.nan

    return WindowResult(start, end, T, v_s, v_e, twr_exact, twr_reported,
                        "exact" if method == "exact" else f"dietz-{frequency}",
                        mwr, how, sub, fdf, twr_path, mwr_path, values.loc[start:end], cdf, unrevalued, first_capital)


# ------------------------------------------------------------ narrative rules


def pct(x: float, dp: int = 2) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x * 100:.{dp}f}%"


def narrative(res: WindowResult, index_full: pd.Series, cfg: dict) -> list[dict]:
    """Rule-based reading of the configuration. Each item: level (flag/note), text, provisions."""
    out = []
    T = res.days
    gap = res.gap
    ann_ok = res.is_annualisable

    # 1. money-weighted versus time-weighted
    if res.flows is not None and len(res.flows) > 0 and np.isfinite(gap):
        big = res.contributions.iloc[res.contributions["Contribution"].abs().idxmax()] if len(res.contributions) else None
        if ann_ok:
            g_show = res.annualised(res.mwr) - res.annualised(res.twr_exact)
            unit = " pp p.a."
        else:
            g_show = gap
            unit = " pp"
        if abs(g_show) >= 0.015:
            lvl = "flag"
        elif abs(g_show) >= 0.003:
            lvl = "note"
        else:
            lvl = "quiet"
        direction = "above" if gap > 0 else "below"
        txt = (f"The money-weighted return sits {abs(g_show) * 100:.2f}{unit} {direction} the exact time-weighted return "
               f"over this window. The market path is identical in both; the difference is the timing and size "
               f"of the external cash flows.")
        if big is not None:
            txt += (f" The largest single contribution comes from the flow of {big['Amount']:+,.1f} on "
                    f"{big['Date'].isoformat()} ({big['Contribution'] * 100:+.2f} pp).")
        if gap > 0:
            txt += (" The client's money was in the market for the better part of the window. A manager leading "
                    "with the money-weighted figure would be taking credit for the timing of flows"
                    + (" it did not control." if not cfg.get("controls_flows") else " that it did control - which is the case 1.A.35 contemplates."))
        elif gap < 0:
            txt += (" The client's money was in the market for the weaker part of the window. The time-weighted "
                    "figure is the one that describes the manager's decisions; the money-weighted figure is closer "
                    "to what this client experienced.")
        out.append({"level": lvl, "text": txt, "provisions": ["1.A.35", "2.A.24", "2.A.29"]})
    elif res.flows is not None and len(res.flows) == 0:
        out.append({"level": "quiet", "text": "No external cash flows inside the window: the money-weighted and "
                    "time-weighted returns coincide. Add or move a flow to open the gap.", "provisions": []})

    # 2. presentation type versus vehicle eligibility
    if cfg.get("lead_with") == "MWR":
        eligible = cfg.get("controls_flows") and cfg.get("vehicle") == "closed_end"
        if eligible:
            out.append({"level": "note", "text": "Money-weighted return as the headline: the firm controls the flows and the "
                        "vehicle is closed-end / fixed life / fixed commitment, so 1.A.35 permits it. The choice then "
                        "has to be applied consistently for this composite or fund (1.A.36).", "provisions": ["1.A.35", "1.A.36"]})
        else:
            why = []
            if not cfg.get("controls_flows"):
                why.append("the firm does not control the external cash flows")
            if cfg.get("vehicle") != "closed_end":
                why.append("the vehicle is not closed-end, fixed life, fixed commitment or illiquid")
            out.append({"level": "flag", "text": "Money-weighted return as the headline is not available here: "
                        + " and ".join(why) + ". 1.A.35 requires a time-weighted return unless both limbs are met.",
                        "provisions": ["1.A.35"]})

    # 3. window excludes the record's largest drawdown
    dd = largest_drawdown(index_full)
    if dd["depth"] < -0.05:
        tr, pk = dd["trough"], dd["peak"]
        if not (res.start <= tr <= res.end):
            if tr < res.start:
                months_after = (res.start.year - tr.year) * 12 + res.start.month - tr.month
                if months_after <= 18:
                    out.append({"level": "flag", "text": f"The window opens {months_after} month(s) after the trough of the "
                                f"record's largest drawdown ({pct(dd['depth'], 1)} from {pk.isoformat()} to {tr.isoformat()}). "
                                f"Every figure shown is arithmetically correct and excludes it. In a GIPS advertisement the "
                                f"periods presented must follow one of the 8.C.1 patterns; a bespoke window sits under 8.A.13 "
                                f"as other information with equal or lesser prominence.", "provisions": ["8.C.1", "8.A.13", "4.A.1"]})
            elif pk > res.end:
                months_before = (pk.year - res.end.year) * 12 + pk.month - res.end.month
                if months_before <= 12:
                    out.append({"level": "flag", "text": f"The window closes {months_before} month(s) before the record's largest "
                                f"drawdown begins ({pct(dd['depth'], 1)} from {pk.isoformat()}). The period shown is not the "
                                f"most recent one available.", "provisions": ["8.C.1", "4.A.1"]})
        else:
            out.append({"level": "quiet", "text": f"The window contains the record's largest drawdown "
                        f"({pct(dd['depth'], 1)}, trough {tr.isoformat()}).", "provisions": []})

    # 4. annualising a sub-year period
    if not ann_ok and cfg.get("annualise"):
        out.append({"level": "flag", "text": f"The window is {T} days. Showing an annualised figure asserts that the "
                    f"period's return would repeat for a full year. Returns for periods of less than one year must "
                    f"not be annualised (2.A.12; 8.A.4 in an advertisement).", "provisions": ["2.A.12", "8.A.4"]})

    # 5. approximation and valuation frequency
    if res.twr_method != "exact":
        err = res.twr_reported - res.twr_exact
        if res.unrevalued_large:
            names = ", ".join(f"{f.amount:+,.1f} on {f.date.isoformat()}" for f in res.unrevalued_large)
            out.append({"level": "flag", "text": f"Flow(s) at or above the firm's large cash flow threshold sit inside a "
                        f"single Modified Dietz sub-period without revaluation: {names}. 2.A.23(c) requires a valuation on "
                        f"the date of all large cash flows and 2.A.24(c) a sub-period return at that point, where daily "
                        f"returns are not calculated. The approximation error against the exact figure is "
                        f"{err * 1e4:+.0f} bp.", "provisions": ["2.A.23", "2.A.24"]})
        elif abs(err) >= 0.0005:
            out.append({"level": "note", "text": f"Modified Dietz at {cfg.get('frequency')} frequency differs from the exact "
                        f"linked figure by {err * 1e4:+.0f} bp. Modified Dietz is a money-weighted approximation inside each "
                        f"sub-period; the residual is the money-weighting that revaluation at the flow would remove.",
                        "provisions": ["2.A.24", "2.B.1"]})
        else:
            out.append({"level": "quiet", "text": f"Modified Dietz and the exact linked figure differ by "
                        f"{err * 1e4:+.1f} bp on this configuration.", "provisions": ["2.A.24"]})
    if cfg.get("frequency") == "quarterly" and cfg.get("preset") != "private_markets":
        out.append({"level": "flag", "text": "Quarterly valuation on a portfolio that is not a private market investment "
                    "portfolio: 2.A.23 requires valuation at least monthly and 2.A.24(a) returns at least monthly.",
                    "provisions": ["2.A.23", "2.A.24"]})
    if cfg.get("preset") == "private_markets" and cfg.get("frequency") == "quarterly":
        out.append({"level": "quiet", "text": "Quarterly valuation is the permitted floor for private market investment "
                    "portfolios (2.A.40, 2.A.41).", "provisions": ["2.A.40", "2.A.41"]})

    # 6. fees
    if cfg.get("fee", 0) > 0:
        if cfg.get("basis") == "gross":
            out.append({"level": "note", "text": f"Returns shown gross of a {cfg['fee'] * 100:.2f}% p.a. management fee. "
                        f"Gross-of-fees must be labelled as such (4.A.3, 8.C.3) and the fee schedule disclosed (4.C.11).",
                        "provisions": ["4.A.3", "4.C.11", "8.C.3"]})
        else:
            out.append({"level": "note", "text": f"Returns shown net of a {cfg['fee'] * 100:.2f}% p.a. model fee accrued "
                        f"daily. A model fee must produce returns equal to or lower than actual fees would (2.A.31) and "
                        f"the model must be disclosed (4.C.7).", "provisions": ["2.A.30", "2.A.31", "4.C.7"]})
    return out


def disclosure_prompt(res: WindowResult, cfg: dict) -> list[dict]:
    """Disclosures a reader would need to interpret the figure on screen, given the configuration."""
    items = [
        {"text": "Reporting currency.", "provisions": ["4.C.9"]},
        {"text": "Whether the returns are gross-of-fees or net-of-fees, and the periods presented.", "provisions": ["4.A.3", "8.C.3"]},
        {"text": "That policies for valuing investments, calculating performance and preparing GIPS Reports are available on request.", "provisions": ["4.C.16"]},
    ]
    if cfg.get("basis") == "gross":
        items.append({"text": "Whether any fees other than transaction costs are deducted from the gross figure.", "provisions": ["4.C.6"]})
    else:
        items.append({"text": "Whether model or actual fees are used, and the model fee and method if model.", "provisions": ["4.C.7"]})
    items.append({"text": "The current fee schedule appropriate to the prospective client.", "provisions": ["4.C.11"]})
    if cfg.get("lead_with") == "MWR":
        items.append({"text": "Annualised since-inception money-weighted return through the most recent annual period end.", "provisions": ["5.A.1", "2.A.29"]})
        items.append({"text": "The frequency of external cash flows used in the money-weighted calculation, if not daily.", "provisions": ["5.C.35"]})
        if cfg.get("vehicle") == "closed_end":
            items.append({"text": "Paid-in capital, distributions, committed capital and the TVPI, DPI, PIC and RVPI multiples.", "provisions": ["5.A.4"]})
        items.append({"text": "Any change in the type of return presented, with its date.", "provisions": ["4.C.42"]})
    if res.twr_method != "exact":
        items.append({"text": "The composite-specific large cash flow definition that governs when portfolios are revalued.", "provisions": ["2.A.23", "2.A.24"]})
    if cfg.get("significant_cf_policy"):
        items.append({"text": "How the firm defines a significant cash flow for the composite and for which periods.", "provisions": ["4.C.35", "3.A.12"]})
    if not res.is_annualisable:
        items.append({"text": "That the figure is for a period of less than one year and is not annualised.", "provisions": ["2.A.12", "8.A.4"]})
    items.append({"text": "Any significant event a prospective client would need to interpret the record (for example a large flow timed against a market move, or a window that excludes a drawdown).", "provisions": ["4.C.19"]})
    return items
