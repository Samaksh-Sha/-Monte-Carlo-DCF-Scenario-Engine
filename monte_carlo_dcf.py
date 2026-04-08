"""
monte_carlo_dcf.py  ·  Model 4 of 6
Monte Carlo DCF + Scenario Engine
─────────────────────────────────
10,000-iteration simulation with AR(1) mean-reverting growth paths,
correlated WACC/terminal-growth draws, tornado sensitivity, and
Bull/Base/Bear scenario comparison.

Usage:
    streamlit run monte_carlo_dcf.py

Dependencies:
    pip install streamlit yfinance plotly pandas numpy scipy
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Monte Carlo DCF Engine",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DARK THEME CSS ─────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
  [data-testid="stSidebar"]          { background:#0a0a0a; border-right:1px solid #1a1a1a; }
  [data-testid="stSidebar"] label    { color:#94a3b8 !important; font-size:12px; }
  .main .block-container             { padding-top:1.5rem; }
  .stMetric label                    { color:#64748b !important; font-size:11px !important; text-transform:uppercase; letter-spacing:.05em; }
  .stMetric [data-testid="stMetricValue"] { color:#f1f5f9 !important; font-weight:600; }
  .stMetric [data-testid="stMetricDelta"] { font-size:12px; }
  h1  { color:#f1f5f9 !important; font-size:22px !important; font-weight:600; }
  h2  { color:#e2e8f0 !important; font-size:16px !important; }
  h3  { color:#cbd5e1 !important; font-size:14px !important; }
  .stTabs [data-baseweb="tab"]        { color:#64748b; font-size:13px; }
  .stTabs [aria-selected="true"]      { color:#f1f5f9 !important; border-bottom:2px solid #6366f1 !important; }
  .stDataFrame                        { border:1px solid #1e1e1e; border-radius:8px; }
  footer                              { visibility:hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
COMPANIES: dict[str, dict] = {
    "Infosys":          {"ticker": "INFY.NS",       "sector": "IT"},
    "TCS":              {"ticker": "TCS.NS",         "sector": "IT"},
    "Reliance":         {"ticker": "RELIANCE.NS",    "sector": "Conglomerate"},
    "HDFC Bank":        {"ticker": "HDFCBANK.NS",    "sector": "Banking"},
    "Bajaj Finance":    {"ticker": "BAJFINANCE.NS",  "sector": "NBFC"},
    "Asian Paints":     {"ticker": "ASIANPAINT.NS",  "sector": "Consumer"},
    "Titan":            {"ticker": "TITAN.NS",        "sector": "Consumer"},
    "Maruti Suzuki":    {"ticker": "MARUTI.NS",       "sector": "Auto"},
    "Divi's Labs":      {"ticker": "DIVISLAB.NS",     "sector": "Pharma"},
    "Pidilite":         {"ticker": "PIDILITIND.NS",   "sector": "Specialty Chem"},
    "Manual Input":     {"ticker": None,              "sector": "Custom"},
}

SCENARIOS: dict[str, dict] = {
    "Bear": {"wacc": +0.020, "growth": -0.025, "tvg": -0.005, "color": "#ef4444", "emoji": "🔴"},
    "Base": {"wacc":  0.000, "growth":  0.000, "tvg":  0.000, "color": "#f59e0b", "emoji": "🟡"},
    "Bull": {"wacc": -0.020, "growth": +0.025, "tvg": +0.005, "color": "#22c55e", "emoji": "🟢"},
}

DARK_LAYOUT = dict(
    paper_bgcolor="#0d0d0d",
    plot_bgcolor="#111111",
    font=dict(family="Inter,-apple-system,sans-serif", color="#94a3b8", size=12),
    title_font=dict(color="#e2e8f0", size=14, family="Inter,-apple-system,sans-serif"),
    margin=dict(l=48, r=32, t=52, b=40),
)

DARK_AXIS = dict(gridcolor="#1a1a1a", linecolor="#2a2a2a", zerolinecolor="#2a2a2a")
DARK_LEGEND = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#2a2a2a", font=dict(color="#94a3b8"))

# ── DATA LAYER ─────────────────────────────────────────────────────────────────also
@st.cache_data(ttl=900, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """Pull key fundamentals from yfinance. Returns dict with raw values in INR."""
    try:
        stk  = yf.Ticker(ticker)
        info = stk.info or {}

        # ── BASIC INFO: Price & Shares (usually reliable) ───────────────────
        price = info.get("currentPrice") or info.get("regularMarketPrice") or 0.0
        shares = info.get("sharesOutstanding") or 1e9
        mktcap = info.get("marketCap", 0)
        beta = info.get("beta", 1.0) or 1.0
        sector = info.get("sector", "N/A")
        name = info.get("longName", ticker)

        # ── FETCH LATEST ANNUAL STATEMENT ──────────────────────────────────
        cf = stk.cashflow      # Annual cash flow
        bs = stk.balance_sheet # Annual balance sheet
        inc = stk.income_stmt  # Annual income statement

        # Initialize
        fcf_ttm = None
        total_debt = None
        total_cash = None
        ebitda = None
        op_cf = None
        capex = None

        # ── FCF: Operating Cash Flow - Capital Expenditure (latest annual) ──
        if cf is not None and not cf.empty:
            try:
                # Get most recent annual period (first column)
                if "Operating Cash Flow" in cf.index:
                    op_cf = cf.loc["Operating Cash Flow"].iloc[0]
                if "Capital Expenditure" in cf.index:
                    capex = cf.loc["Capital Expenditure"].iloc[0]
                
                if op_cf is not None and capex is not None:
                    # capex is typically negative, so: FCF = Op CF - CapEx (where CapEx < 0)
                    fcf_ttm = op_cf - capex
                elif op_cf is not None:
                    fcf_ttm = op_cf
            except Exception:
                pass

        # ── DEBT & CASH: Latest annual balance sheet ───────────────────────
        if bs is not None and not bs.empty:
            try:
                # Search for total debt
                for idx in bs.index:
                    idx_str = str(idx)
                    if "Total Debt" in idx_str:
                        total_debt = bs.loc[idx].iloc[0]
                        break
                
                # If not found, try Long Term Debt
                if total_debt is None:
                    for idx in bs.index:
                        idx_str = str(idx)
                        if "Long Term Debt" in idx_str:
                            total_debt = bs.loc[idx].iloc[0]
                            break

                # Search for cash and equivalents
                for idx in bs.index:
                    idx_str = str(idx)
                    if "Cash" in idx_str and "Equivalents" in idx_str:
                        total_cash = bs.loc[idx].iloc[0]
                        break
                
                # If not found, try just cash
                if total_cash is None:
                    for idx in bs.index:
                        idx_str = str(idx)
                        if idx_str == "Cash" or idx_str == "Cash And Cash Equivalents":
                            total_cash = bs.loc[idx].iloc[0]
                            break
            except Exception:
                pass

        # Fall back if balance sheet extraction failed
        if total_debt is None:
            total_debt = info.get("totalDebt", 0) or 0
        if total_cash is None:
            total_cash = info.get("totalCash", 0) or 0

        # ── EBITDA: Latest annual income statement ──────────────────────────
        if inc is not None and not inc.empty:
            try:
                for idx in inc.index:
                    if "EBITDA" in str(idx):
                        ebitda = inc.loc[idx].iloc[0]
                        break
            except Exception:
                pass

        if ebitda is None:
            ebitda = info.get("ebitda", None)

        # ── CALCULATE NET DEBT ─────────────────────────────────────────────
        net_debt = (total_debt or 0) - (total_cash or 0)

        # ── CALCULATE RATIOS ───────────────────────────────────────────────
        fcf_per_share = (fcf_ttm / shares) if fcf_ttm and shares > 0 else None
        fcf_yield_pct = ((fcf_ttm / shares / price) * 100) if fcf_ttm and shares > 0 and price > 0 else None

        return {
            "name": name,
            "price": price,
            "mktcap": mktcap,
            "shares": shares,
            "beta": beta,
            "sector": sector,
            "fcf_ttm": fcf_ttm,
            "op_cf": op_cf,
            "capex": capex,
            "total_debt": total_debt or 0,
            "total_cash": total_cash or 0,
            "net_debt": net_debt,
            "ebitda": ebitda,
            "fcf_per_share": fcf_per_share,
            "fcf_yield_pct": fcf_yield_pct,
            "error": None,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── SIMULATION ENGINE ──────────────────────────────────────────────────────────
def ar1_paths(
    mu: float, phi: float, sigma: float, n_years: int, n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate AR(1) growth-rate paths.
        g_t = phi * g_{t-1} + (1 - phi) * mu + sigma * eps_t
    Shape: (n_sims, n_years)
    """
    paths = np.empty((n_sims, n_years))
    eps   = rng.standard_normal((n_sims, n_years))
    paths[:, 0] = mu + sigma * eps[:, 0]
    for t in range(1, n_years):
        paths[:, t] = phi * paths[:, t - 1] + (1 - phi) * mu + sigma * eps[:, t]
    return paths


def monte_carlo_dcf(
    base_fcf:      float,
    n_years:       int,
    wacc_mu:       float,
    wacc_sig:      float,
    growth_mu:     float,
    growth_sig:    float,
    tvg_mu:        float,
    tvg_sig:       float,
    net_debt:      float,
    shares:        float,
    n_sims:        int,
    phi:           float,
    seed:          int = 42,
) -> dict:
    """
    Full Monte Carlo DCF.
    Returns intrinsic-value-per-share distribution + diagnostics.
    All monetary inputs in base currency units (₹).
    """
    rng = np.random.default_rng(seed)

    # --- Correlated WACC & terminal growth (ρ = 0.35) ----------------------
    rho   = 0.35
    cov   = [
        [wacc_sig**2,                rho * wacc_sig * tvg_sig],
        [rho * wacc_sig * tvg_sig,   tvg_sig**2              ],
    ]
    draws        = rng.multivariate_normal([wacc_mu, tvg_mu], cov, n_sims)
    wacc_s       = np.clip(draws[:, 0], 0.06, 0.40)
    tvg_s        = np.clip(draws[:, 1], 0.00, 0.08)

    # Gordon Growth validity: tvg must be < wacc
    bad          = tvg_s >= wacc_s - 0.005
    wacc_s[bad]  = wacc_mu
    tvg_s[bad]   = min(tvg_mu, wacc_mu - 0.01)

    # --- AR(1) FCF growth paths --------------------------------------------
    g_paths = ar1_paths(growth_mu, phi, growth_sig, n_years, n_sims, rng)

    # --- Discount FCF paths ------------------------------------------------
    fcf_mat    = np.zeros((n_sims, n_years))
    pv_fcf_sum = np.zeros(n_sims)

    for t in range(n_years):
        if t == 0:
            fcf_t = base_fcf * (1.0 + g_paths[:, 0])
        else:
            fcf_t = fcf_mat[:, t - 1] * (1.0 + g_paths[:, t])
        fcf_t        = np.maximum(fcf_t, 0.0)
        fcf_mat[:, t] = fcf_t
        pv_fcf_sum   += fcf_t / (1.0 + wacc_s) ** (t + 1)

    # --- Terminal value (Gordon Growth) ------------------------------------
    tv_fcf = fcf_mat[:, -1] * (1.0 + tvg_s)
    tv     = tv_fcf / (wacc_s - tvg_s)
    pv_tv  = tv    / (1.0 + wacc_s) ** n_years

    # --- Equity value per share -------------------------------------------
    ev      = pv_fcf_sum + pv_tv
    equity  = ev - net_debt
    vps_raw = np.where(shares > 0, equity / shares, np.nan)
    vps     = vps_raw[np.isfinite(vps_raw)]

    # Trim extreme 1% tails
    lo, hi  = np.percentile(vps, [1, 99])
    vps     = vps[(vps >= lo) & (vps <= hi)]

    pcts = {
        k: float(np.percentile(vps, p))
        for k, p in [("P5",5),("P10",10),("P25",25),("P50",50),("P75",75),("P90",90),("P95",95)]
    }
    pcts["Mean"] = float(np.mean(vps))
    pcts["Std"]  = float(np.std(vps))

    return {
        "vps":        vps,
        "pcts":       pcts,
        "g_paths":    g_paths,
        "wacc_s":     wacc_s,
        "tvg_s":      tvg_s,
        "fcf_mat":    fcf_mat,
        "pv_tv_frac": float(np.mean(pv_tv / ev)),  # TV as % of EV
    }


def scenario_dcf(
    base_fcf: float, n_years: int,
    wacc: float, growth: float, tvg: float,
    net_debt: float, shares: float,
) -> float:
    """Deterministic single-path DCF. Returns IV per share."""
    pv_sum = 0.0
    fcf    = base_fcf
    for t in range(1, n_years + 1):
        fcf    *= 1.0 + growth
        pv_sum += fcf / (1.0 + wacc) ** t

    last_fcf = base_fcf * (1.0 + growth) ** n_years
    tv       = last_fcf * (1.0 + tvg) / max(wacc - tvg, 0.001)
    pv_tv    = tv / (1.0 + wacc) ** n_years

    ev     = pv_sum + pv_tv
    equity = ev - net_debt
    return (equity / shares) if shares > 0 else 0.0


def tornado(
    base_fcf: float, n_years: int,
    wacc: float, growth: float, tvg: float,
    net_debt: float, shares: float,
) -> tuple[pd.DataFrame, float]:
    """Compute ±shock sensitivity for each key driver."""
    base = scenario_dcf(base_fcf, n_years, wacc, growth, tvg, net_debt, shares)

    shocks = {
        "WACC (±200 bps)":          (wacc - 0.02, wacc + 0.02, "wacc"),
        "FCF Growth (±3 pp)":       (growth + 0.03, growth - 0.03, "growth"),
        "Terminal Growth (±1 pp)":  (tvg + 0.01, tvg - 0.01, "tvg"),
        "Base FCF (±20%)":          (base_fcf * 1.20, base_fcf * 0.80, "base_fcf"),
        "Net Debt (±20%)":          (net_debt * 0.80, net_debt * 1.20, "net_debt"),
        "Horizon (±2 yrs)":         (n_years + 2, n_years - 2, "n_years"),
    }

    rows = []
    for label, (hi_val, lo_val, param) in shocks.items():
        kwargs = dict(base_fcf=base_fcf, n_years=n_years, wacc=wacc,
                      growth=growth, tvg=tvg, net_debt=net_debt, shares=shares)
        if param == "wacc":
            hi = scenario_dcf(**{**kwargs, "wacc": hi_val})
            lo = scenario_dcf(**{**kwargs, "wacc": lo_val})
        elif param == "growth":
            hi = scenario_dcf(**{**kwargs, "growth": hi_val})
            lo = scenario_dcf(**{**kwargs, "growth": lo_val})
        elif param == "tvg":
            hi = scenario_dcf(**{**kwargs, "tvg": hi_val})
            lo = scenario_dcf(**{**kwargs, "tvg": lo_val})
        elif param == "base_fcf":
            hi = scenario_dcf(**{**kwargs, "base_fcf": hi_val})
            lo = scenario_dcf(**{**kwargs, "base_fcf": lo_val})
        elif param == "net_debt":
            hi = scenario_dcf(**{**kwargs, "net_debt": hi_val})
            lo = scenario_dcf(**{**kwargs, "net_debt": lo_val})
        else:  # n_years
            hi_yr = max(int(hi_val), 3)
            lo_yr = max(int(lo_val), 3)
            hi = scenario_dcf(**{**kwargs, "n_years": hi_yr})
            lo = scenario_dcf(**{**kwargs, "n_years": lo_yr})

        rows.append({"Factor": label, "High": hi, "Low": lo,
                     "Range": abs(hi - lo),
                     "hi_delta": hi - base, "lo_delta": lo - base})

    df = pd.DataFrame(rows).sort_values("Range", ascending=True).reset_index(drop=True)
    return df, base


# ── MAIN ───────────────────────────────────────────────────────────────────────
def main() -> None:
    st.title("🎲 Monte Carlo DCF + Scenario Engine")
    st.caption("AR(1) growth paths · 10,000 iterations · correlated WACC/TVG · tornado sensitivity")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🏢 Company")
        co_name = st.selectbox("Company", list(COMPANIES.keys()))
        co      = COMPANIES[co_name]
        live_ok = (co["ticker"] is not None) and st.toggle("Fetch Live Data", value=True)

        live: Optional[dict] = None
        if live_ok:
            with st.spinner("Fetching..."):
                live = fetch_fundamentals(co["ticker"])
                if live.get("error"):
                    st.warning(f"Live data failed: {live['error']}")
                    live = None
                else:
                    st.success(f"✓ {live['name']}")
                    
                    # ── Data Quality Warnings ────────────────────────────────
                    fcf = live.get("fcf_ttm")
                    fcf_ps = live.get("fcf_per_share")
                    fcf_yield = live.get("fcf_yield_pct")
                    price = live.get("price")
                    
                    if fcf_yield is not None and fcf_yield < 0.5:
                        st.warning(
                            f"⚠️ **Low FCF Yield ({fcf_yield:.2f}%)**\n\n"
                            f"FCF per share (₹{fcf_ps:.2f}) seems very low relative to price (₹{price:.2f}). "
                            f"This may indicate incomplete or stale yfinance data. Consider overriding manually."
                        )
                    elif fcf_yield is not None and fcf_yield > 20:
                        st.warning(
                            f"⚠️ **High FCF Yield ({fcf_yield:.2f}%)**\n\n"
                            f"FCF data may be inflated or anomalous. Verify against latest financials."
                        )
                    
                    # ── Calculation Methodology ───────────────────────────────
                    st.info(
                        "📊 **Calculation Methodology**\n\n"
                        "All financial metrics are calculated from the latest annual financial statement:\n\n"
                        f"• **FCF** = Operating Cash Flow - Capital Expenditure\n"
                        f"• **Net Debt** = Total Debt - Cash & Equivalents\n\n"
                        "These calculations **may not match official audited values exactly**. "
                        "For investment decisions, verify against the latest official financial statements."
                    )
                    
                    if st.checkbox("Show raw financial data", value=False):
                        st.json({
                            "Operating CF": live.get("op_cf"),
                            "Capital Expenditure": live.get("capex"),
                            "FCF (Calculated)": live.get("fcf_ttm"),
                            "Total Debt": live.get("total_debt"),
                            "Cash & Equivalents": live.get("total_cash"),
                            "Net Debt (Calculated)": live.get("net_debt"),
                        })

        # Derive defaults from live data
        CR = 1e7  # 1 Crore
        def_fcf   = ((live["fcf_ttm"] or 5_000 * CR) / CR) if live else 1_500.0
        def_ndbt  = ((live["net_debt"] or 0) / CR) if live else 500.0
        def_shr   = ((live["shares"] or 1e9) / CR) if live else 400.0   # shares in Crore
        def_price = (live["price"] or 0.0) if live else 0.0

        st.markdown("---")
        st.markdown("### 💰 Financials (₹ Cr / ₹)")
        base_fcf    = st.number_input("Base FCF (₹ Cr)",             value=max(float(def_fcf), 100.0),  step=100.0)
        net_debt    = st.number_input("Net Debt (₹ Cr)",              value=float(def_ndbt),             step=100.0)
        shares_cr   = st.number_input("Shares Outstanding (Cr)",      value=max(float(def_shr), 1.0),    step=10.0)
        curr_price  = st.number_input("Current Market Price (₹)",     value=float(def_price),            step=10.0)

        st.markdown("---")
        st.markdown("### 📐 DCF Parameters")
        n_years     = st.slider("Projection Period (yrs)",   3,  10,  5)
        wacc_mu     = st.slider("WACC — Mean (%)",           6.0, 25.0, 12.0, 0.5) / 100
        wacc_sig    = st.slider("WACC — Std Dev (%)",        0.5,  5.0,  1.5, 0.25) / 100
        growth_mu   = st.slider("FCF Growth — Mean (%)",     0.0, 30.0, 12.0, 0.5) / 100
        growth_sig  = st.slider("FCF Growth — Std Dev (%)", 0.5, 10.0,  3.0, 0.5) / 100
        tvg_mu      = st.slider("Terminal Growth — Mean (%)", 2.0, 8.0, 4.5, 0.25) / 100
        tvg_sig     = st.slider("Terminal Growth — Std Dev (%)", 0.25, 2.0, 0.75, 0.25) / 100

        st.markdown("---")
        st.markdown("### 🔁 AR(1) Process")
        phi = st.slider(
            "Mean-Reversion Coefficient (φ)", 0.0, 0.95, 0.70, 0.05,
            help="φ=0 → i.i.d. shocks; φ=0.9 → high growth persistence",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Simulation")
        n_sims = st.selectbox("Iterations", [1_000, 5_000, 10_000, 25_000], index=2)
        seed   = int(st.number_input("Random Seed", value=42, step=1))
        run    = st.button("▶  Run Simulation", use_container_width=True, type="primary")

    # ── GATE ─────────────────────────────────────────────────────────────────
    if not run and "mc" not in st.session_state:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("**Inputs**\n\nSet company fundamentals and distribution parameters in the sidebar, then hit **Run Simulation**.")
        with c2:
            st.info("**Outputs**\n\nP5–P95 IV distribution · tornado sensitivity · scenario comparison · AR(1) growth path viewer")
        with c3:
            st.info("**Tip**\n\nUse sector WACC from Damodaran's Indian dataset. Set growth std dev ≈ 2–4% for stable large-caps.")
        return

    if run:
        with st.spinner(f"Running {n_sims:,} simulations…"):
            result = monte_carlo_dcf(
                base_fcf   = base_fcf   * CR,
                n_years    = n_years,
                wacc_mu    = wacc_mu,
                wacc_sig   = wacc_sig,
                growth_mu  = growth_mu,
                growth_sig = growth_sig,
                tvg_mu     = tvg_mu,
                tvg_sig    = tvg_sig,
                net_debt   = net_debt   * CR,
                shares     = shares_cr  * CR,
                n_sims     = n_sims,
                phi        = phi,
                seed       = seed,
            )
            st.session_state["mc"]     = result
            st.session_state["params"] = {
                "base_fcf": base_fcf, "n_years": n_years,
                "wacc": wacc_mu, "growth": growth_mu, "tvg": tvg_mu,
                "net_debt": net_debt, "shares_cr": shares_cr,
                "curr_price": curr_price, "phi": phi,
            }

    r  = st.session_state["mc"]
    p  = st.session_state["params"]
    pt = r["pcts"]
    vps = r["vps"]
    cmp = p["curr_price"]

    prob_up = float(np.mean(vps > cmp) * 100) if cmp > 0 else None

    # ── KPI STRIP ─────────────────────────────────────────────────────────────
    cols = st.columns(6)
    cols[0].metric("Median IV",    f"₹{pt['P50']:,.1f}")
    cols[1].metric("Mean IV",      f"₹{pt['Mean']:,.1f}")
    cols[2].metric("P10 (Bear)",   f"₹{pt['P10']:,.1f}")
    cols[3].metric("P90 (Bull)",   f"₹{pt['P90']:,.1f}")
    cols[4].metric("TV / EV",      f"{r['pv_tv_frac']*100:.0f}%")
    if prob_up is not None:
        delta = f"{(pt['P50'] - cmp) / cmp * 100:+.1f}% vs CMP"
        cols[5].metric("P(Upside vs CMP)", f"{prob_up:.1f}%", delta)
    else:
        cols[5].metric("Std Dev", f"₹{pt['Std']:,.1f}")

    st.divider()

    # ── TABS ──────────────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs(["📊 Distribution", "🌪️ Tornado", "📐 Scenarios", "📈 Growth Paths", "📋 Insights"])

    # ── TAB 1: DISTRIBUTION ──────────────────────────────────────────────────
    with t1:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=vps, nbinsx=90,
            marker_color="#6366f1", marker_line_width=0, opacity=0.75,
            name="Simulated IV",
        ))
        if cmp > 0:
            fig.add_vline(x=cmp, line_color="#f59e0b", line_width=2, line_dash="dash",
                          annotation_text=f"CMP ₹{cmp:,.0f}", annotation_font_color="#f59e0b",
                          annotation_position="top right")
        for key, color in [("P10","#ef4444"), ("P50","#22c55e"), ("P90","#3b82f6")]:
            fig.add_vline(x=pt[key], line_color=color, line_width=1.5, line_dash="dot",
                          annotation_text=f"{key} ₹{pt[key]:,.0f}", annotation_font_color=color)
        fig.update_layout(**DARK_LAYOUT, title="Intrinsic Value Distribution (per share)",
                          xaxis_title="Intrinsic Value (₹)", yaxis_title="Frequency",
                          xaxis=DARK_AXIS, yaxis=DARK_AXIS,
                          legend=DARK_LEGEND, showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Percentile table
        pct_keys = ["P5","P10","P25","P50","P75","P90","P95"]
        rows = []
        for k in pct_keys:
            row = {"Percentile": k, "IV per Share (₹)": f"₹{pt[k]:,.1f}"}
            if cmp > 0:
                row["vs CMP"] = f"{(pt[k] - cmp) / cmp * 100:+.1f}%"
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── TAB 2: TORNADO ───────────────────────────────────────────────────────
    with t2:
        CR = 1e7
        tdf, base_iv = tornado(
            p["base_fcf"] * CR, p["n_years"], p["wacc"], p["growth"],
            p["tvg"], p["net_debt"] * CR, p["shares_cr"] * CR,
        )

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(y=tdf["Factor"], x=tdf["hi_delta"], orientation="h",
                              name="Upside",   marker_color="#22c55e", opacity=0.85))
        fig2.add_trace(go.Bar(y=tdf["Factor"], x=tdf["lo_delta"], orientation="h",
                              name="Downside", marker_color="#ef4444", opacity=0.85))
        fig2.add_vline(x=0, line_color="#555", line_width=1)
        fig2.update_layout(**DARK_LAYOUT,
                           title=f"Tornado — Sensitivity to ±1σ Shocks  (Base IV ₹{base_iv:,.1f})",
                           xaxis_title="Δ Intrinsic Value per Share (₹)", yaxis_title="",
                           xaxis=DARK_AXIS, yaxis=DARK_AXIS,
                           barmode="overlay", height=400,
                           legend=dict(**DARK_LEGEND, orientation="h", y=1.08, x=0.5, xanchor="center"))
        st.plotly_chart(fig2, use_container_width=True)

        st.dataframe(
            tdf[["Factor","High","Low","Range"]].rename(columns={
                "High":"High Case (₹)", "Low":"Low Case (₹)", "Range":"Swing (₹)"}
            ).style.format({"High Case (₹)":"₹{:.1f}","Low Case (₹)":"₹{:.1f}","Swing (₹)":"₹{:.1f}"}),
            use_container_width=True, hide_index=True,
        )

    # ── TAB 3: SCENARIOS ─────────────────────────────────────────────────────
    with t3:
        CR = 1e7
        scen_vals = {
            s: scenario_dcf(
                p["base_fcf"] * CR, p["n_years"],
                p["wacc"] + adj["wacc"], p["growth"] + adj["growth"],
                p["tvg"]  + adj["tvg"],
                p["net_debt"] * CR, p["shares_cr"] * CR,
            )
            for s, adj in SCENARIOS.items()
        }

        sc_cols = st.columns(3)
        for i, (s, v) in enumerate(scen_vals.items()):
            lbl  = f"{SCENARIOS[s]['emoji']} {s}"
            delt = f"{(v - cmp) / cmp * 100:+.1f}% vs CMP" if cmp > 0 else None
            sc_cols[i].metric(lbl, f"₹{v:,.1f}", delt)

        fig3 = go.Figure(go.Bar(
            x=[f"{SCENARIOS[s]['emoji']} {s}" for s in scen_vals],
            y=list(scen_vals.values()),
            marker_color=[SCENARIOS[s]["color"] for s in scen_vals],
            text=[f"₹{v:,.1f}" for v in scen_vals.values()],
            textposition="outside", textfont=dict(color="#e2e8f0"),
        ))
        if cmp > 0:
            fig3.add_hline(y=cmp, line_dash="dash", line_color="#f59e0b", line_width=1.5,
                           annotation_text=f"CMP ₹{cmp:,.0f}", annotation_font_color="#f59e0b")
        fig3.update_layout(**DARK_LAYOUT, title="Scenario Intrinsic Value vs. CMP",
                           yaxis_title="IV per Share (₹)",
                           xaxis=DARK_AXIS, yaxis=DARK_AXIS,
                           legend=DARK_LEGEND, showlegend=False, height=380)
        st.plotly_chart(fig3, use_container_width=True)

        # Assumption table
        tbl = pd.DataFrame([
            {
                "Scenario":        f"{SCENARIOS[s]['emoji']} {s}",
                "WACC":            f"{(p['wacc'] + adj['wacc']) * 100:.1f}%",
                "FCF Growth":      f"{(p['growth'] + adj['growth']) * 100:.1f}%",
                "Terminal Growth": f"{(p['tvg'] + adj['tvg']) * 100:.1f}%",
                "IV per Share":    f"₹{scen_vals[s]:,.1f}",
            }
            for s, adj in SCENARIOS.items()
        ])
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── TAB 4: AR(1) PATHS ───────────────────────────────────────────────────
    with t4:
        gp    = r["g_paths"]          # shape (n_sims, n_years)
        years = list(range(1, p["n_years"] + 1))
        yr_labels = [f"Yr {y}" for y in years]

        fig4 = go.Figure()
        sample = min(60, len(gp))
        for i in range(sample):
            fig4.add_trace(go.Scatter(
                x=years, y=gp[i] * 100, mode="lines",
                line=dict(width=0.5, color="#6366f1"), opacity=0.25, showlegend=False,
            ))
        fig4.add_trace(go.Scatter(
            x=years, y=gp.mean(axis=0) * 100, mode="lines+markers",
            name="Mean Path", line=dict(width=2.5, color="#f59e0b"), marker_size=6,
        ))
        fig4.add_hline(y=p["growth"] * 100, line_dash="dot", line_color="#555", line_width=1,
                       annotation_text=f"μ = {p['growth']*100:.1f}%", annotation_font_color="#555")
        fig4.update_layout(**DARK_LAYOUT,
                           title=f"AR(1) FCF Growth Paths — φ={p['phi']:.2f}  |  {sample} of {len(gp):,} shown",
                           xaxis_title="Year", yaxis_title="FCF Growth Rate (%)",
                           xaxis=dict(**DARK_AXIS, tickvals=years, ticktext=yr_labels),
                           yaxis=DARK_AXIS,
                           legend=DARK_LEGEND, height=420)
        st.plotly_chart(fig4, use_container_width=True)

        phi_val = p["phi"]
        st.caption(
            f"φ = {phi_val:.2f} → "
            + ("High persistence: past growth shocks have a long memory."
               if phi_val >= 0.7 else
               "Moderate persistence: growth reverts to the mean relatively quickly."
               if phi_val >= 0.4 else
               "Low persistence: each year's growth is nearly independent.")
        )

    # ── TAB 5: INSIGHTS & INTERPRETATION ───────────────────────────────────
    with t5:
        st.markdown("### 📊 DCF Simulation Results Explained")
        
        # ── SECTION 1: Distribution ────────────────────────────────────────
        st.markdown("#### 1️⃣ Intrinsic Value Distribution")
        st.markdown(
            f"""
**What it shows:** The range of fair values the stock could have based on 10,000 different scenarios.

**Key numbers from your analysis:**
- **Median (P50):** ₹{pt['P50']:,.1f} — The middle value. If the stock is worth anything, it's likely around this.
- **Mean:** ₹{pt['Mean']:,.1f} — Average across all scenarios.
- **P10 (Bear case):** ₹{pt['P10']:,.1f} — 10% odds the stock is worth THIS MUCH OR LESS.
- **P90 (Bull case):** ₹{pt['P90']:,.1f} — 10% odds the stock is worth THIS MUCH OR MORE.

**What you should do:**
- If current price < P10: Stock is likely undervalued even in pessimistic scenarios.
- If current price > P90: Stock is likely overvalued.
            """)
        
        st.markdown("---")
        
        # ── SECTION 2: Terminal Value ──────────────────────────────────────
        st.markdown("#### 2️⃣ Terminal Value as % of Enterprise Value")
        tv_pct = r['pv_tv_frac'] * 100
        tv_interpretation = "🟢 Reasonable" if tv_pct < 70 else "🟡 Moderate concern" if tv_pct < 80 else "🔴 High model risk"
        st.markdown(
            f"""
**Current:** {tv_pct:.0f}% {tv_interpretation}

**What it means:**
- **TV > 80%**: Model is highly sensitive to terminal assumptions. Small errors compound significantly.
- **TV 50–75%**: Balanced between near-term and long-term value.
- **TV < 50%**: Near-term cash flows are reliable drivers of value.

**Key insight:** If TV is very high, your valuation depends heavily on assumptions about a company's performance 10+ years out. Be conservative!
            """)
        
        st.markdown("---")
        
        # ── SECTION 3: Valuation vs. CMP ───────────────────────────────────
        st.markdown("#### 3️⃣ Valuation vs. Current Market Price")
        if cmp > 0:
            median_delta_pct = (pt['P50'] - cmp) / cmp * 100
            p90_delta_pct = (pt['P90'] - cmp) / cmp * 100
            p10_delta_pct = (pt['P10'] - cmp) / cmp * 100
            
            if prob_up > 70:
                cmp_verdict = "🔴 **Likely Overvalued** — Majority of scenarios suggest downside"
            elif prob_up > 50:
                cmp_verdict = "🟡 **Fairly Valued with Downside Risk** — Slight bias to overvaluation"
            elif prob_up > 30:
                cmp_verdict = "🟡 **Fairly Valued with Upside Risk** — Slight bias to undervaluation"
            else:
                cmp_verdict = "🟢 **Likely Undervalued** — Majority of scenarios suggest upside"
            
            st.markdown(
                f"""
**Current Price:** ₹{cmp:,.2f}

**Valuation Verdict:** {cmp_verdict}

- **Median fair value:** ₹{pt['P50']:,.1f} ({median_delta_pct:+.1f}% vs. CMP)
- **Bear case (P10):** ₹{pt['P10']:,.1f} ({p10_delta_pct:+.1f}% vs. CMP)
- **Bull case (P90):** ₹{pt['P90']:,.1f} ({p90_delta_pct:+.1f}% vs. CMP)

**Probability Analysis:**
- **{prob_up:.1f}%** chance stock is undervalued (IV > CMP)
- **{100-prob_up:.1f}%** chance stock is overvalued (IV < CMP)

**Investment Implication:**
- If upside prob > 70%: Strong buy signal
- If upside prob 50–70%: Buy, but with caution
- If upside prob 30–50%: Hold or avoid
- If upside prob < 30%: Likely overvalued; avoid or wait for pullback
                """)
        else:
            st.info("📌 Enter current market price in the sidebar to see valuation verdict.")
        
        st.markdown("---")
        
        # ── SECTION 4: Sensitivity ─────────────────────────────────────────
        st.markdown("#### 4️⃣ Tornado Sensitivity Analysis")
        st.markdown(
            """
**What it shows:** Which assumptions have the BIGGEST impact on your valuation.

**How to read it:**
- **Largest bars** = Most sensitive drivers (focus research here)
- **Smallest bars** = Less critical (small errors don't matter much)

**Typical ranking (most to least critical):**
1. **WACC** — Changes of ±2% can swing valuation by 30%+
2. **Base FCF** — Historical FCF accuracy is foundational
3. **FCF Growth** — Near-term growth matters more than terminal
4. **Horizon** — Adding/removing 2 years has moderate impact
5. **Terminal Growth** — Less moved due to discounting

**What you should do:**
- **Spend 80% of research effort** validating top 2 factors (WACC & FCF)
- **Cross-check WACC** against Damodaran's sector benchmarks
- **Verify FCF** against annual report cash flow statement
- **Less critical to perfect:** Terminal growth (heavily discounted)
            """)
        
        st.markdown("---")
        
        # ── SECTION 5: Scenarios ───────────────────────────────────────────
        st.markdown("#### 5️⃣ Bull / Base / Bear Scenarios")
        st.markdown(
            """
**Three possible worlds:**

🔴 **Bear Case:** Economy slows, lower growth, higher discount rates (WACC up by 200 bps)
🟡 **Base Case:** Consensus expectations (your "most likely" outcome)
🟢 **Bull Case:** Strong growth in favorable environment (WACC down, growth up)

**How to use it:**
- If **CMP > Bull valuation**: Stock is overpriced even in best case → Avoid
- If **Base < CMP < Bull**: Risky; requires bull case to play out
- If **Bear < CMP < Base**: Good risk/reward; balanced upside/downside
- If **CMP < Bear**: Huge margin of safety; stock cheap in worst case
            """)
        
        st.markdown("---")
        
        # ── SECTION 6: Growth Paths ────────────────────────────────────────
        st.markdown("#### 6️⃣ AR(1) Growth Path Assumptions")
        phi_val = p['phi']
        phi_interpretation = (
            "**Strong growth momentum:** Past performance influences future" if phi_val >= 0.7
            else "**Moderate mean reversion:** Growth tends toward baseline" if phi_val >= 0.4
            else "**Random year-to-year:** Little to no autocorrelation"
        )
        st.markdown(
            f"""
**Your setting:** φ = {phi_val:.2f} — {phi_interpretation}

**What this means for your model:**
- **High φ (0.7–0.95):** If a company had 15% growth last year, expect 10%+ this year
- **Medium φ (0.4–0.6):** Growth fluctuates but trends back to baseline
- **Low φ (0–0.3):** Each year's growth is independent; no predictable pattern

**Best practice for different companies:**
- Mature tech companies: φ = 0.60–0.75 (some momentum persistence)
- Startups/growth: φ = 0.40–0.60 (volatile, mean-reverting)
- Turnarounds: φ = 0.20–0.40 (highly unpredictable)
            """)
        
        st.markdown("---")
        
        # ── SECTION 7: Confidence ──────────────────────────────────────────
        st.markdown("#### 7️⃣ How Confident Should You Be?")
        st.markdown(
            f"""
**Confidence Checklist:**

✓ **Terminal Value < 70%?** {("✅ YES — Near-term FCF is the main driver" if tv_pct < 70 else "⚠️  NO — Highly dependent on long-term assumptions")}

✓ **Valuation is clear (>30% diff from CMP)?** {("✅ YES — Clear signal" if abs(median_delta_pct) > 30 else "❌ NO — Ambiguous")}

✓ **Distribution is tight (P90/P10 < 1.5x)?** {("✅ YES — Relatively confident" if (pt['P90'] - pt['P10']) / max(pt['P10'], 0.1) < 1.5 else "❌ NO — Wide range, high uncertainty")}

**Overall Assessment:** This DCF is a **thought framework**, not a crystal ball.

**Before you invest:**
1. ✓ Verify your assumptions vs. analyst consensus & sector benchmarks
2. ✓ Compare with relative valuation (P/E, EV/EBITDA) 
3. ✓ Demand a margin of safety (20–30% below fair value before buying)
4. ✓ Monitor quarterly earnings; update model if FCF trends diverge
5. ✓ Recognize model risk increases beyond 5-year horizon
            """)


if __name__ == "__main__":
    main()
