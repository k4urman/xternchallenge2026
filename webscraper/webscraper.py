"""
EIA x datacenter.fyi Correlation Pipeline (XTERN / TechPoint - MISO Prompt 2)
=============================================================================
Part 1 - EIA API v2 client: pulls state-level electricity demand/sales and
         MISO RTO load from https://api.eia.gov/v2  (free key: eia.gov/opendata)
Part 2 - datacenter.fyi scraper: facility-level data center capacity by state
Part 3 - Merge on state, test correlation:
           "Do states where datacenter.fyi shows big data-center buildouts
            also show the load growth EIA measures?"
         -> Validates your PowerBI 'At Risk Load' story with PUBLIC data.

Usage:
    export EIA_API_KEY=your_key_here
    python webscraper.py
"""

import os, re, time, json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

EIA_KEY   = os.environ.get("EIA_API_KEY", "YOUR_EIA_API_KEY")
EIA_BASE  = "https://api.eia.gov/v2"
HEADERS   = {"User-Agent": "XTERN-MISO-research/1.0 (educational)"}

# MISO footprint states (from your dashboards)
MISO_STATES = ["IN","WI","LA","IA","MO","MI","MN","MS","KY","ND","AR","IL","SD","TX","MT"]

# ============================ PART 1: EIA API ============================

def eia_get(api_path, params=None, retries=3):
    """Generic EIA v2 GET with pagination (5,000-row API limit)."""
    params = dict(params or {})
    params["api_key"] = EIA_KEY
    rows, offset = [], 0
    while True:
        params.update({"offset": offset, "length": 5000})
        for attempt in range(retries):
            try:
                r = requests.get(f"{EIA_BASE}/{api_path}", params=params,
                                 headers=HEADERS, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        payload = r.json().get("response", {})
        batch = payload.get("data", [])
        rows.extend(batch)
        total = int(payload.get("total", len(rows)))
        offset += len(batch)
        if offset >= total or not batch:
            break
    return pd.DataFrame(rows)

def pull_eia_state_sales(start="2019-01", end="2026-08"):
    """
    Monthly retail electricity sales (MWh) by state = observed load growth.
    Route: electricity/retail-sales (facets: stateid, sectorid).
    """
    df = eia_get("electricity/retail-sales/data/", {
        "frequency": "monthly",
        "data[]": "sales",
        "facets[stateid][]": MISO_STATES,
        "facets[sectorid][]": "ALL",          # all sectors
        "start": start, "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    })
    df["sales_mwh"] = pd.to_numeric(df["sales"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df[["period", "stateid", "sales_mwh"]]

def pull_eia_miso_peak(start="2019-01-01", end="2026-08-01"):
    """
    MISO daily peak demand from the hourly grid monitor.
    Route: electricity/rto/region-data (facet respondent=MISO).
    """
    df = eia_get("electricity/rto/region-data/data/", {
        "frequency": "daily",
        "data[]": "value",
        "facets[respondent][]": "MISO",
        "start": start, "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    })
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df[["period", "value"]].rename(columns={"value": "miso_daily_mwh"})

def pull_eia_state_generation(start="2019-01", end="2026-08"):
    """Monthly net generation by state (all fuels) - supply side of the equation."""
    df = eia_get("electricity/electric-power-operational-data/data/", {
        "frequency": "monthly",
        "data[]": "generation",
        "facets[location][]": MISO_STATES,
        "facets[fueltypeid][]": "ALL",
        "start": start, "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
    })
    df["generation_mwh"] = pd.to_numeric(df["generation"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df[["period", "location", "generation_mwh"]]

# ======================= PART 2: datacenter.fyi scrape =======================

DC_BASE = "https://www.datacenter.fyi"

def scrape_datacenter_fyi(max_pages=25, delay=1.5):
    """
    Scrapes the datacenter.fyi facilities directory into a tidy DataFrame.
    The site renders facility cards/tables with fields like facility name,
    state, capacity (MW), developer, and status. Selectors are centralized
    in SELECTORS below -- tweak if the site redesigns.
    """
    SELECTORS = {
        "row":   "table tbody tr, .facility-card, .facility-row",
        "name":  "td:nth-child(1), .facility-name, h3",
        "state": "td:nth-child(2), .facility-state",
        "mw":    "td:nth-child(3), .facility-capacity",
        "dev":   "td:nth-child(4), .facility-developer",
        "status": "td:nth-child(5), .facility-status",
    }
    records = []
    with requests.Session() as s:
        s.headers.update(HEADERS)
        for page in range(1, max_pages + 1):
            for url in (f"{DC_BASE}/facilities?page={page}",
                        f"{DC_BASE}/data-centers?page={page}"):
                try:
                    r = s.get(url, timeout=30)
                    if r.status_code != 200:
                        continue
                except requests.RequestException:
                    continue
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(r.text, "html.parser")
                rows = soup.select(SELECTORS["row"])
                if not rows:
                    continue
                for row in rows:
                    def grab(key):
                        el = row.select_one(SELECTORS[key])
                        return el.get_text(strip=True) if el else None
                    mw_txt = grab("mw") or ""
                    mw = float(re.sub(r"[^\d.]", "", mw_txt)) if re.search(r"\d", mw_txt) else np.nan
                    records.append({
                        "name": grab("name"), "state": grab("state"),
                        "capacity_mw": mw, "developer": grab("dev"),
                        "status": grab("status"),
                    })
                break  # first URL pattern that worked
            time.sleep(delay)                     # be polite
            if records and page >= 3 and not rows:
                break                             # exhausted pagination
    dc = pd.DataFrame(records).dropna(subset=["state"])
    dc["state"] = dc["state"].str.upper().str.extract(r"\b([A-Z]{2})\b", expand=False)
    return dc.dropna(subset=["state"])

def datacenter_capacity_by_state(dc):
    """Aggregate scraped facility capacity -> state-level MW (operating vs planned)."""
    dc["is_planned"] = dc["status"].str.contains(
        "plan|proposed|under construction|announced", case=False, na=False)
    g = dc.groupby("state").agg(
        total_dc_mw=("capacity_mw", "sum"),
        n_facilities=("name", "count"),
        planned_dc_mw=("capacity_mw", lambda x: x[dc.loc[x.index, "is_planned"]].sum()),
    ).reset_index()
    return g

# ======================= PART 3: CORRELATION ENGINE =======================

def build_state_panel(eia_sales, dc_states):
    """
    Panel: one row per state with
      - datacenter.fyi capacity (MW)         <- independent variable
      - EIA load CAGR 2019->2026 (%)         <- dependent variable
      - EIA latest annual sales (MWh)
    """
    annual = (eia_sales.assign(year=eia_sales["period"].dt.year)
                       .groupby(["stateid", "year"])["sales_mwh"].sum().unstack())
    recent_years = [c for c in annual.columns if c >= 2019]
    panel = pd.DataFrame({
        "state": annual.index,
        "sales_2019_mwh": annual[2019] if 2019 in annual else np.nan,
        "sales_latest_mwh": annual[recent_years[-1]],
        "load_cagr_pct": ((annual[recent_years[-1]] / annual[2019]) ** (1 / (recent_years[-1] - 2019)) - 1) * 100,
    }).merge(dc_states, left_on="state", right_on="state", how="inner")
    return panel.dropna()

def correlate(panel):
    """Pearson + Spearman between data-center capacity and EIA load growth."""
    out = {}
    for x_col, label in [("total_dc_mw", "all DC capacity"),
                         ("planned_dc_mw", "planned DC capacity")]:
        sub = panel[(panel[x_col] > 0) & panel["load_cagr_pct"].notna()]
        if len(sub) >= 4:
            pr, pp = stats.pearsonr(sub[x_col], sub["load_cagr_pct"])
            sr, sp = stats.spearmanr(sub[x_col], sub["load_cagr_pct"])
            out[label] = {"n_states": len(sub),
                          "pearson_r": round(pr, 3), "pearson_p": round(pp, 4),
                          "spearman_r": round(sr, 3), "spearman_p": round(sp, 4)}
            print(f"\n[{label}]  n={len(sub)} states")
            print(f"  Pearson  r={pr:.3f}  p={pp:.4f}")
            print(f"  Spearman r={sr:.3f}  p={sp:.4f}")
            if pp < 0.05:
                print("  -> STATISTICALLY SIGNIFICANT: data-center buildouts track EIA-measured load growth.")
            else:
                print("  -> not significant at 5% (small n is expected -- 15 MISO states).")
    return out

def plot(panel, corr):
    fig, ax = plt.subplots(figsize=(9, 6))
    x, y = panel["total_dc_mw"], panel["load_cagr_pct"]
    ax.scatter(x, y, s=90, alpha=0.75, edgecolor="k")
    for _, r in panel.iterrows():
        ax.annotate(r["state"], (r["total_dc_mw"], r["load_cagr_pct"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    m, b = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, m * xs + b, "--", color="crimson", alpha=0.7)
    ax.set_xlabel("datacenter.fyi capacity in state (MW)")
    ax.set_ylabel("EIA retail sales CAGR 2019-latest (%)")
    ax.set_title("Data-Center Buildout vs. EIA-Measured Load Growth (MISO states)\n"
                 f"slope={m:.4f} %/MW -- each planned MW shows up as measured load")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("eia_datacenter_correlation.png", dpi=150)
    print("\nChart saved -> eia_datacenter_correlation.png")

# ================================== MAIN ==================================

if __name__ == "__main__":
    print(">>> Pulling EIA retail sales by state...")
    eia_sales = pull_eia_state_sales()
    print(f"    {len(eia_sales):,} rows")

    print(">>> Pulling MISO daily demand...")
    miso = pull_eia_miso_peak()
    print(f"    {len(miso):,} days; mean={miso['miso_daily_mwh'].mean():,.0f} MWh")

    print(">>> Scraping datacenter.fyi...")
    dc = scrape_datacenter_fyi()
    dc_states = datacenter_capacity_by_state(dc)
    print(f"    {dc_states['n_facilities'].sum():,.0f} facilities, "
          f"{dc_states['total_dc_mw'].sum():,.0f} MW tracked")

    print(">>> Building state panel + testing correlation...")
    panel = build_state_panel(eia_sales, dc_states)
    print(panel.sort_values("total_dc_mw", ascending=False).to_string(index=False))
    corr = correlate(panel)

    panel.to_csv("state_panel.csv", index=False)
    with open("correlation_results.json", "w") as f:
        json.dump(corr, f, indent=2)
    plot(panel, corr)

    print("Done.")
