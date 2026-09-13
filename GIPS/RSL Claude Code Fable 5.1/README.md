# Return Sensitivity Lab

Streamlit app for Session 4 of *GIPS Standards for Firms* (Euromoney Learning, 14–16 September 2026).
It replaces the static TWR/MWR calculation with an interactive one: a fixed market path, cash flows
the participant places on it, a valuation frequency, a calculation method and a presentation window.
Headline TWR and MWR, a cumulative chart, a rule-based reading of the configuration and the disclosures
a reader would need, with the provision text quoted from the 2020 Standards.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit front end (Lab, Composite & calculators, Provisions, Notes tabs) |
| `engine.py` | Path presets, portfolio valuation, exact TWR, Modified Dietz, IRR-based MWR, flow decomposition, narrative rules |
| `composite.py` | Asset- v equal-weighted (card 6), the three 2.A.36 routes (card 7), annualisation arithmetic (card 5) |
| `scenarios.yaml` | Scenario presets — edit without touching code |
| `provisions.yaml` | Provision text, extracted from the 485-provision workbook |
| `.streamlit/config.toml` | Midnight Executive theme |

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository (public or private) and push this folder to it — `app.py` at the repository root.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. **New app** → pick the repository and branch → main file path `app.py` → **Deploy**.
   Python version is read from `.python-version`; dependencies from `requirements.txt`; the theme from `.streamlit/config.toml`.
4. The app URL is `https://<app-name>.streamlit.app`. Put it on the participant materials.

Updating: push to the branch and the app redeploys. A free-tier app sleeps after inactivity and wakes on the first visit,
so open it before the session starts.

## Conventions

See the **Notes** tab in the app. In short: calendar-daily grid; a flow dated D is applied at the close of D and earns the
market return from D+1; the market path is fixed by the preset so moving a flow changes Modified Dietz and MWR but not
the exact TWR; MWR is an IRR on daily-dated flows, annualised only where the window is one year or more.

Card base case: 30 days, 100 → 108 → 150 with +40 at day 15 — exact 9.4595%, Modified Dietz 8.3333%, MWR 8.3613%.
