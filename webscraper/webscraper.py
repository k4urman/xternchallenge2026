"""
Usage:
    $env:EIA_API_KEY="your_key"          # PowerShell
    python webscraper.py
"""

import os, re, time, json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from bs4 import BeautifulSoup

EIA_KEY   = os.environ.get("EIA_API_KEY", "YOUR_EIA_API_KEY")
EIA_BASE  = "https://api.eia.gov/v2"
HEADERS   = {"User-Agent": "XTERN-MISO-research/1.0 (educational)"}
MISO_STATES = ["IN","WI","LA","IA","MO","MI","MN","MS","KY","ND","AR","IL","SD","TX","MT"]

# ============================ PART 1: EIA API ============================

def eia_get(api_path, params=None, retries=3):
    params = dict(params or {})
    params["api_key"] = EIA_KEY
    rows, offset = [], 0
    while True:
        params.update({"offset": offset, "length": 5000})
        for attempt in range(retries):
            try:
                r = requests.get(f"{EIA_BASE}/{api_path}", params=params,
                                 headers=HEADERS, timeout=60)
                if r.status_code == 400:
                    try:
                        detail = r.json().get("error", r.text[:400])
                    except Exception:
                        detail = r.text[:400]
                    raise ValueError(f"EIA 400 on {api_path}: {detail}")
                r.raise_for_status()
                break
            except (requests.RequestException, ValueError):
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        payload = r.json().get("response", {})
        batch = payload.get("data", [])
        rows.extend(batch)
        offset += len(batch)
        if offset >= int(payload.get("total", len(rows))) or not batch:
            break
    return pd.DataFrame(rows)

def eia_inspect_route(api_path):
    r = requests.get(f"{EIA_BASE}/{api_path}", params={"api_key": EIA_KEY}, timeout=30)
    meta = r.json().get("response", {})
    print(f"Route: {api_path}")
    print("  frequencies:", [f.get("id") for f in meta.get("frequency", [])])
    for f in meta.get("facets", []):
        print(f"  facet '{f.get('id')}': {len(f.get('values', []))} values,",
              "e.g.", [v.get("id") for v in f.get("values", [])[:8]])

def pull_eia_state_sales(start="2019-01", end="2026-08"):
    df = eia_get("electricity/retail-sales/data/", {
        "frequency": "monthly",
        "data[]": "sales",
        "facets[stateid][]": MISO_STATES,
        "facets[sectorid][]": "ALL",
        "start": start, "end": end,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
    })
    df["sales_mwh"] = pd.to_numeric(df["sales"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df[["period", "stateid", "sales_mwh"]]

def pull_eia_miso_daily(start="2019-01-01", end="2026-08-01"):
    """
    FIX v3: EIA removed 'daily' from region-data - only 'hourly' remains.
    Pull hourly MISO demand and resample to daily here instead.
    (Hourly = ~61k rows for 2019-2026, paginates fine at 5k/call.)
    """
    df = eia_get("electricity/rto/region-data/data/", {
        "frequency": "hourly",
        "data[]": "value",
        "facets[respondent][]": "MISO",
        "facets[type][]": "D",
        "start": start, "end": end,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
    })
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    daily = (df.set_index("period")["value"]
               .resample("D").agg(miso_daily_mwh="mean", miso_daily_peak_mwh="max")
               .reset_index())
    return daily

def pull_eia_state_generation(start="2019-01", end="2026-08"):
    df = eia_get("electricity/electric-power-operational-data/data/", {
        "frequency": "monthly",
        "data[]": "generation",
        "facets[location][]": MISO_STATES,
        "facets[fueltypeid][]": "ALL",
        "start": start, "end": end,
        "sort[0][column]": "period", "sort[0][direction]": "asc",
    })
    df["generation_mwh"] = pd.to_numeric(df["generation"], errors="coerce")
    df["period"] = pd.to_datetime(df["period"])
    return df[["period", "location", "generation_mwh"]]

# ======================= PART 2: datacenter.fyi scrape =======================

DC_BASE = "https://www.datacenter.fyi"

# --- SELECTORS: if auto-diagnosis prints different markup, paste it here ---
SELECTORS = {
    "row":    "table tbody tr, .facility-card, .facility-row",
    "name":   "td:nth-child(1), .facility-name, h3",
    "state":  "td:nth-child(2), .facility-state",
    "mw":     "td:nth-child(3), .facility-capacity",
    "dev":    "td:nth-child(4), .facility-developer",
    "status": "td:nth-child(5), .facility-status",
}

def diagnose_page(html, url):
    """Print what the page actually contains so you can fix SELECTORS fast."""
    soup = BeautifulSoup(html, "html.parser")
    print("\n" + "=" * 70)
    print(f"SCRAPE DIAGNOSTIC - no facility rows matched on {url}")
    title = soup.title
    print(f"  page title: {title.get_text(strip=True) if title else 'NONE'}")
    tables = soup.find_all("table")
    print(f"  <table> count: {len(tables)}")
    for i, t in enumerate(tables[:3]):
        print(f"    table[{i}] classes: {t.get('class')}, rows: {len(t.find_all('tr'))}")
    # any element that looks list/card-like
    for cls in ["facility", "card", "row", "item", "listing"]:
        hits = soup.select(f"[class*='{cls}']")
        if hits:
            print(f"  elements with class containing '{cls}': {len(hits)} "
                  f"-> e.g. class='{hits[0].get('class')}'")
    # if the page is a JS shell, there will be almost no text
    body_text = soup.get_text(strip=True)
    print(f"  visible text length: {len(body_text)} chars "
          f"({'LIKELY JS-RENDERED - needs Selenium or an API endpoint' if len(body_text) < 500 else 'server-rendered, selectors just wrong'})")
    print("  -> open the page in Chrome DevTools, right-click a facility row,")
    print("     Copy > Copy selector, and update SELECTORS at the top of the script.")
    print("=" * 70 + "\n")

def scrape_datacenter_fyi(max_pages=25, delay=1.5):
    records, working_pattern = [], None
    with requests.Session() as s:
        s.headers.update(HEADERS)
        pages_tried = 0
        for page in range(1, max_pages + 1):
            if working_pattern is None and page > 3 and not records:
                break
            url = (f"{DC_BASE}/facilities?page={page}" if working_pattern is None
                   else working_pattern.format(page=page))
            try:
                r = s.get(url, timeout=30)
                if r.status_code != 200:
                    # try alternate URL once on page 1
                    if page == 1 and working_pattern is None:
                        alt = f"{DC_BASE}/data-centers?page={page}"
                        r = s.get(alt, timeout=30)
                        if r.status_code == 200:
                            working_pattern = f"{DC_BASE}/data-centers?page={{page}}"
                    if r.status_code != 200:
                        continue
                elif working_pattern is None:
                    working_pattern = f"{DC_BASE}/facilities?page={{page}}"
            except requests.RequestException:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select(SELECTORS["row"])
            if not rows:
                if page == 1 and not records:
                    diagnose_page(r.text, url)
                    raise SystemExit("Scraper stopped - fix SELECTORS using the diagnostic above.")
                break                      # pagination exhausted
            for row in rows:
                def grab(key):
                    el = row.select_one(SELECTORS[key])
                    return el.get_text(strip=True) if el else None
                mw_txt = grab("mw") or ""
                mw = float(re.sub(r"[^\d.]", "", mw_txt)) if re.search(r"\d", mw_txt) else np.nan
                records.append({"name": grab("name"), "state": grab("state"),
                                "capacity_mw": mw, "developer": grab("dev"),
                                "status": grab("status")})
            pages_tried = page
            time.sleep(delay)
    dc = pd.DataFrame(records)
    if dc.empty or "state" not in dc.columns:
        raise SystemExit("Scraper returned no usable rows - fix SELECTORS (see diagnostic).")
    dc["state"] = (dc["state"].astype(str).str.upper()
                   .str.extract(r"\b([A-Z]{2})\b", expand=False))
    return dc.dropna(subset=["state"])

def datacenter_capacity_by_state(dc):
    dc["is_planned"] = dc["status"].str.contains(
        "plan|proposed|under construction|announced", case=False, na=False)
    return dc.groupby("state").agg(
        total_dc_mw=("capacity_mw", "sum"),
        n_facilities=("name", "count"),
        planned_dc_mw=("capacity_mw", lambda x: x[dc.loc[x.index, "is_planned"]].sum()),
    ).reset_index()

# ======================= PART 3: CORRELATION ENGINE =======================

def build_state_panel(eia_sales, dc_states):
    annual = (eia_sales.assign(year=eia_sales["period"].dt.year)
                       .groupby(["stateid", "year"])["sales_mwh"].sum().unstack())
    recent_years = [c for c in annual.columns if c >= 2019]
    panel = pd.DataFrame({
        "state": annual.index,
        "sales_2019_mwh": annual[2019] if 2019 in annual else np.nan,
        "sales_latest_mwh": annual[recent_years[-1]],
        "load_cagr_pct": ((annual[recent_years[-1]] / annual[2019])
                          ** (1 / (recent_years[-1] - 2019)) - 1) * 100,
    }).merge(dc_states, on="state", how="inner")
    return panel.dropna()

def correlate(panel):
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
            print("  -> SIGNIFICANT" if pp < 0.05 else "  -> not significant at 5% (small n expected)")
    return out

def plot(panel):
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
    ax.set_title("Data-Center Buildout vs. EIA-Measured Load Growth (MISO states)")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("eia_datacenter_correlation.png", dpi=150)
    print("\nChart saved -> eia_datacenter_correlation.png")


# ================================== MAIN ==================================

if __name__ == "__main__":
    print(">>> Pulling EIA retail sales by state...")
    eia_sales = pull_eia_state_sales()
    print(f"    {len(eia_sales):,} rows")
    eia_sales.to_csv("eia_sales_raw.csv", index=False)

    print(">>> Pulling EIA generation by state...")
    eia_gen = pull_eia_state_generation()
    print(f"    {len(eia_gen):,} rows")
    eia_gen.to_csv("eia_gen_raw.csv", index=False)

    try:
        miso = pull_eia_miso_daily()
        print(f"    MISO: {len(miso):,} days resampled from hourly; "
              f"mean={miso['miso_daily_mwh'].mean():,.0f} MWh")
        miso.to_csv("miso_daily_demand.csv", index=False)
    except Exception as e:
        print(f"    !! MISO pull failed (non-fatal): {e}")

    # ---------- Build pure EIA state characterization ----------
    print(">>> Building EIA state characterization (no DC data)...")

    # 1. Annual sales per state
    sales_annual = (eia_sales
                    .assign(year=eia_sales["period"].dt.year)
                    .groupby(["stateid", "year"])["sales_mwh"]
                    .sum()
                    .reset_index())

    # 2. Annual generation per state
    gen_annual = (eia_gen
                  .assign(year=eia_gen["period"].dt.year)
                  .groupby(["location", "year"])["generation_mwh"]
                  .sum()
                  .reset_index())

    # 3. Merge sales + generation on (state, year)
    panel = pd.merge(sales_annual, gen_annual,
                     left_on=["stateid", "year"],
                     right_on=["location", "year"],
                     how="outer")
    panel.rename(columns={"stateid": "state", "sales_mwh": "sales", "generation_mwh": "gen"}, inplace=True)
    panel = panel[panel["state"].notna()]

    # 4. Keep only years >= 2019
    panel = panel[panel["year"] >= 2019]

    # 5. Summarize per state
    def summarize(df):
        years = sorted(df["year"].unique())
        if len(years) < 2:
            return None
        first, last = years[0], years[-1]
        sales_first = df[df["year"] == first]["sales"].sum()
        sales_last  = df[df["year"] == last]["sales"].sum()
        gen_first   = df[df["year"] == first]["gen"].sum()
        gen_last    = df[df["year"] == last]["gen"].sum()

        # CAGR
        n_years = last - first
        sales_cagr = (sales_last / sales_first) ** (1 / n_years) - 1 if sales_first > 0 else np.nan
        gen_cagr   = (gen_last / gen_first) ** (1 / n_years) - 1 if gen_first > 0 else np.nan

        # Peak sales year
        peak_year = df.loc[df["sales"].idxmax(), "year"] if df["sales"].notna().any() else np.nan

        # Number of years with positive growth (year‑over‑year)
        sales_sorted = df.sort_values("year")["sales"].dropna()
        pos_growth = (sales_sorted.pct_change() > 0).sum() if len(sales_sorted) > 1 else 0

        return pd.Series({
            "sales_2019_mwh": sales_first,
            "sales_latest_mwh": sales_last,
            "sales_cagr_pct": sales_cagr * 100,
            "gen_2019_mwh": gen_first,
            "gen_latest_mwh": gen_last,
            "gen_cagr_pct": gen_cagr * 100,
            "load_gen_ratio_latest": sales_last / gen_last if gen_last > 0 else np.nan,
            "sales_peak_year": peak_year,
            "num_years_growth": pos_growth,
            "years_span": n_years,
        })

    eia_characterization = panel.groupby("state").apply(summarize).dropna().reset_index()

    # 6. Save the clean characterization
    eia_characterization.to_csv("eia_characterization.csv", index=False)
    print(f"    EIA characterization saved → eia_characterization.csv")
    print(eia_characterization.to_string(index=False))

    # 7. (Optional) Produce a quick correlation – just to show internal EIA trends
    #    You can skip this if you want, but it shows you have a working dataset.
    if len(eia_characterization) > 3:
        corr = eia_characterization[["sales_cagr_pct", "gen_cagr_pct"]].corr()
        print("\nInternal EIA correlation (sales CAGR vs. gen CAGR):")
        print(corr)
