"""
predicts risk-adjusted profitability of MISO aligning with data-center
projects by state, 2026-2032.

Logic chain:
  1. LOAD     - MISO Large Load Readiness      (50.7 GW by state & year)
  2. QUEUE    - Generation Queue Readiness     (194.73 GW active/pending)
  3. RAW      - Load-Generation Alignment      (raw queue vs load - overstates supply)
  4. ADJUSTED - Reliability-Adjusted Readiness (ELC haircut + margin by year)
  5. TIERS    - Large Load Response Framework  (READY/WATCH/AT RISK/CRITICAL)

Core idea: raw queue says "plenty of power," but reliability-adjusted queue
says what actually shows up on a hot summer afternoon. Profit = (revenue from
served load) x (probability it can be served) - (cost of shortfall mitigation).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------------- 1. DATA transcribed from the dashboards (GW) ----------------
states = ["IN","WI","LA","IA","MO","MI","MN","MS","KY","ND","AR","IL"]

load_gw      = dict(zip(states, [12.0, 6.9, 5.9, 5.5, 4.0, 3.6, 3.1, 2.5, 2.3, 2.1, 2.0, 0.7]))
raw_queue_gw = dict(zip(states, [16.1, 10.9, 12.8, 11.8, 6.9, 32.7, 15.7, 4.0, 0.4, 1.9, 14.5, 30.5]))
adj_queue_gw = dict(zip(states, [9.36, 7.51, 6.46, 5.70, 4.13, 17.95, 6.11, 1.65, 0.15, 0.72, 14.77, 11.94]))
adj_coverage = dict(zip(states, [0.78, 1.10, 1.09, 1.03, 1.02, 4.98, 2.00, 0.65, 0.07, 0.34, 3.42, 1.61]))
tiers        = dict(zip(states, ["AT RISK","WATCH","WATCH","WATCH","WATCH","READY",
                                 "READY","CRITICAL","CRITICAL","CRITICAL","READY","READY"]))

# Reliability-adjusted margin GW by year (image 4 heatmap) -> serviceability gate
margin_by_year = {
 "AR": [1.78, 2.69, 5.01, 5.38, 4.77, 4.77, 4.77],
 "IL": [2.52, 5.30, 8.67, 9.16, 11.10, 11.20, 11.20],
 "IN": [1.38, -1.30, 0.04, 0.17, -1.40, -1.44, -2.64],
 "IA": [0.39, 0.93, 0.96, 1.59, 0.15, 0.15, 0.15],
 "KY": [0.15, -0.09, -0.33, -0.33, -0.33, -0.33, -2.13],
 "LA": [1.65, 1.93, 1.11, 0.32, 2.52, 2.52, 0.52],
 "MI": [1.03, 0.95, 4.48, 6.07, 14.34, 14.34, 14.34],
 "MN": [0.34, 0.62, 2.56, 2.74, 3.02, 3.05, 3.05],
 "MS": [-0.33, -0.53, -0.36, -0.26, -0.89, -0.89, -0.89],
 "MO": [0.71, 0.54, -0.22, -0.75, 0.09, 0.09, 0.09],
 "ND": [-0.18, -0.86, -0.26, -0.17, -1.40, -1.40, -1.40],
 "WI": [0.29, 1.73, 0.64, 0.76, 0.66, 0.66, 0.66],
}

annual_growth = {2026: 3.1, 2027: 11.4, 2028: 11.6, 2029: 2.9, 2030: 16.5, 2031: 0.0, 2032: 5.0}
years = list(annual_growth)
load_share = {s: load_gw[s] / sum(load_gw.values()) for s in states}

# ------------- 2. ECONOMIC ASSUMPTIONS (your "idea" levers) -------------
ASSUMPTIONS = {
    "trans_rev_per_mw_yr":   120_000,   # $/MW/yr network integration transmission service
    "energy_margin_per_mwh": 12.0,      # $/MWh energy + capacity margin on served load
    "capacity_factor":       0.80,      # data centers run hot
    "load_materialization":  0.75,      # P(load actually builds & ramps) - attrition
    "mitigation_cost_per_gw": 180_000_000,  # $/GW one-time gen-pairing + transmission fix
    "curtailment_contract_rev_per_gw_yr": 25_000_000,  # NEW revenue: DR/curtailment contracts
    "discount_rate":         0.08,
    "queue_survival":        {"READY": 0.95, "WATCH": 0.75, "AT RISK": 0.45, "CRITICAL": 0.20},
}

# ------------- 3. MONTE CARLO SIMULATION (10,000 scenarios/state) -------------
N = 10_000
disc = {y: 1 / (1 + ASSUMPTIONS["discount_rate"]) ** (y - 2026) for y in years}

def simulate_state(s):
    p_queue = ASSUMPTIONS["queue_survival"][tiers[s]]
    growth = np.array([annual_growth[y] * load_share[s] * ASSUMPTIONS["load_materialization"] for y in years])
    margin = np.array(margin_by_year[s])

    p_draw = np.clip(rng.normal(p_queue, 0.10, N), 0.05, 0.99)   # queue delivery risk
    m_draw = np.clip(rng.normal(ASSUMPTIONS["energy_margin_per_mwh"], 4.0, N), 2, 30)
    g_draw = np.clip(rng.normal(1.0, 0.15, N), 0.4, 1.6)          # ramp-timing risk

    npv_all = np.zeros(N)
    for n in range(N):
        cum_load, cash = 0.0, []
        for i, y in enumerate(years):
            cum_load += growth[i] * g_draw[n]
            serveable = max(0.0, min(cum_load, margin[i] * p_draw[n]))  # adj. margin gates firm service
            shortfall = cum_load - serveable                            # what needs mitigation
            hours = 8760 * ASSUMPTIONS["capacity_factor"]
            trans_rev = serveable * 1000 * ASSUMPTIONS["trans_rev_per_mw_yr"] / 1e6
            energy_rev = serveable * hours * m_draw[n] / 1000
            curt_rev   = shortfall * ASSUMPTIONS["curtailment_contract_rev_per_gw_yr"] / 1e6
            mitig_cost = shortfall * ASSUMPTIONS["mitigation_cost_per_gw"] / 1e6 * (1 if i == 0 else 0)
            cash.append((trans_rev + energy_rev + curt_rev - mitig_cost) * disc[y])
        npv_all[n] = sum(cash)
    return npv_all

results = {}
for s in states:
    sim = simulate_state(s)
    results[s] = {"expected_npv_$M": sim.mean(),
                  "p5_npv_$M": np.percentile(sim, 5),
                  "p95_npv_$M": np.percentile(sim, 95),
                  "p_loss_%": 100 * (sim < 0).mean(),
                  "tier": tiers[s], "load_gw": load_gw[s],
                  "adj_coverage_%": adj_coverage[s] * 100}

df = pd.DataFrame(results).T.sort_values("expected_npv_$M", ascending=False)
print(df.to_string())
print(f"Portfolio NPV: ${df['expected_npv_$M'].sum():,.0f}M "
      f"(range ${df['p5_npv_$M'].sum():,.0f}M - ${df['p95_npv_$M'].sum():,.0f}M)")

# ---------------- 4. CHART ----------------
fig, ax = plt.subplots(figsize=(11, 6))
colors = {"READY": "#2ca02c", "WATCH": "#ffbf00", "AT RISK": "#ff7f0e", "CRITICAL": "#d62728"}
x = np.arange(len(df))
ax.bar(x, df["expected_npv_$M"], color=[colors[t] for t in df["tier"]], alpha=0.85)
ax.errorbar(x, df["expected_npv_$M"],
            yerr=[df["expected_npv_$M"] - df["p5_npv_$M"], df["p95_npv_$M"] - df["expected_npv_$M"]],
            fmt="none", ecolor="black", capsize=3)
ax.set_xticks(x); ax.set_xticklabels(df.index)
ax.set_ylabel("Expected NPV 2026-2032 ($M)")
ax.set_title("MISO x Data-Center Alignment: Risk-Adjusted Profitability by State")
ax.legend([plt.Rectangle((0,0),1,1,color=c) for c in colors.values()], colors.keys())
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("miso_profitability.png", dpi=150)
plt.show()
