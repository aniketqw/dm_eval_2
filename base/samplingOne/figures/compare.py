# --------------------------------------------------------------
#  chronos_visual_comparison.py
# --------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict

# ------------------------------------------------------------------
# 0. CONFIG – UPDATE THESE TWO PATHS ONLY
# ------------------------------------------------------------------
RESULTS_CSV   = Path("/home/h20250169/study/modelTraining/dm_eval_2/base/results/zz_summary_report.csv")
FIGURES_DIR   = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# 1. PAPER REFERENCE VALUES (exact numbers from the arXiv v3 PDF)
# ------------------------------------------------------------------
PAPER_REF: Dict[str, Dict[str, float]] = {
    # dataset_key : { "wql": ..., "mase": ... }
    "dominick":                {"wql": 0.332, "mase": 0.820},
    "electricity_15min":       {"wql": 0.077, "mase": 0.391},
    "ercot":                   {"wql": 0.019, "mase": None},
    "exchange_rate":           {"wql": 0.013, "mase": 0.900},
    "m4_daily":                {"wql": 0.022, "mase": 3.144},
    "m4_hourly":               {"wql": 0.022, "mase": 0.682},
    "m4_monthly":              {"wql": 0.101, "mase": 0.960},
    "mexico_city_bikes":       {"wql": 0.602, "mase": 0.718},
    "monash_car_parts":        {"wql": 1.060, "mase": 0.780},
    "monash_covid_deaths":     {"wql": 0.045, "mase": 0.720},
    "monash_electricity_weekly":{"wql":0.059, "mase":1.230},
    "monash_fred_md":          {"wql": 0.020, "mase": 2.146},
    "monash_hospital":         {"wql": 0.144, "mase": 0.841},
    "monash_london_smart_meters":{"wql":0.700,"mase":2.020},
    "monash_m1_monthly":       {"wql": 0.109, "mase": 1.195},
    "monash_m1_quarterly":     {"wql": 0.126, "mase": 2.910},
    "monash_m1_yearly":        {"wql": 0.299, "mase": 6.034},
    "monash_m3_monthly":       {"wql": 0.059, "mase": 1.078},
    "monash_m3_quarterly":     {"wql": 0.076, "mase": 1.988},
    "monash_m3_yearly":        {"wql": 0.237, "mase": 4.408},
    "monash_nn5_weekly":       {"wql": 0.102, "mase": 0.853},
    "monash_pedestrian_counts":{"wql":0.162,"mase":0.160},
    "monash_saugeenday":       {"wql": 0.935, "mase":11.481},
    "monash_temperature_rain": {"wql": 0.076, "mase": 0.932},
    "monash_tourism_monthly":  {"wql": 0.144, "mase": 0.834},
    "monash_tourism_quarterly":{"wql":0.165,"mase":0.957},
    "monash_tourism_yearly":   {"wql":0.399,"mase":5.463},
    "monash_traffic":          {"wql":0.078,"mase":0.353},
    "solar_1h":                {"wql":0.380,"mase":0.297},
    "taxi_1h":                 {"wql":0.221,"mase":0.716},
    "taxi_30min":              {"wql":0.292,"mase":0.964},
    "uber_tlc_daily":          {"wql":0.141,"mase":0.752},
    "uber_tlc_hourly":         {"wql":0.364,"mase":0.844},
    "wind_farms_daily":        {"wql":1.027,"mase":0.703},
    "wind_farms_hourly":       {"wql":0.580,"mase":2.302},
}

# ------------------------------------------------------------------
# 2. LOAD YOUR RUN
# ------------------------------------------------------------------
df_run = pd.read_csv(RESULTS_CSV)
df_run = df_run.rename(columns=lambda c: c.strip())
df_run["dataset_name"] = df_run["dataset_name"].astype(str)

# ------------------------------------------------------------------
# 3. BUILD PAPER TABLE
# ------------------------------------------------------------------
paper_rows = []
for ds, vals in PAPER_REF.items():
    paper_rows.append({
        "dataset_name": ds,
        "paper_wql": vals["wql"],
        "paper_mase": vals.get("mase"),
    })
df_paper = pd.DataFrame(paper_rows)

# ------------------------------------------------------------------
# 4. MERGE
# ------------------------------------------------------------------
df = df_run.merge(df_paper, on="dataset_name", how="left")

# ------------------------------------------------------------------
# 5. COMPUTE DIFFERENCES & % CHANGES
# ------------------------------------------------------------------
df["wql_diff"]      = df["wql_geometric_mean"] - df["paper_wql"]
df["wql_pct"]       = df["wql_diff"] / df["paper_wql"] * 100

df["mase_diff"]     = df["mase_geometric_mean"] - df["paper_mase"]
df["mase_pct"]      = df["mase_diff"] / df["paper_mase"] * 100

# keep only rows that have a paper reference
df = df.dropna(subset=["paper_wql"])

# ------------------------------------------------------------------
# 6. SAVE FULL COMPARISON TABLE
# ------------------------------------------------------------------
out_table = FIGURES_DIR / "chronos_comparison_full.csv"
df.to_csv(out_table, index=False)
print(f"Full comparison table → {out_table}")

# ------------------------------------------------------------------
# 7. PLOT 1 – BAR CHART OF WQL (paper order)
# ------------------------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(12, 7))
order = df.sort_values("paper_wql")["dataset_name"]

ax = sns.barplot(
    data=df, x="dataset_name", y="paper_wql",
    color="#1f77b4", label="Paper", order=order
)
sns.barplot(
    data=df, x="dataset_name", y="wql_geometric_mean",
    color="#ff7f0e", alpha=0.7, label="Your run", order=order, ax=ax
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_ylabel("WQL (geometric mean)")
ax.set_xlabel("")
ax.legend()
plt.tight_layout()
for fmt in ["png", "pdf"]:
    plt.savefig(FIGURES_DIR / f"01_wql_bar_chart.{fmt}", dpi=300)
plt.close()
print("Figure 1 → 01_wql_bar_chart.*")

# ------------------------------------------------------------------
# 8. PLOT 2 – SCATTER WQL vs MASE
# ------------------------------------------------------------------
plt.figure(figsize=(8, 8))
sns.scatterplot(
    data=df, x="paper_wql", y="wql_geometric_mean",
    hue="dataset_name", s=80, legend=False, edgecolor="k"
)
sns.scatterplot(
    data=df, x="paper_mase", y="mase_geometric_mean",
    hue="dataset_name", s=80, legend=False, marker="^", edgecolor="k"
)

lims = [
    np.min([df["paper_wql"].min(), df["wql_geometric_mean"].min()]),
    np.max([df["paper_wql"].max(), df["wql_geometric_mean"].max()])
]
plt.plot(lims, lims, "--", color="gray", label="y=x")
plt.xlabel("Paper metric")
plt.ylabel("Your run metric")
plt.title("WQL (circles) & MASE (triangles) – exact reproduction line")
plt.legend(["y=x"], loc="upper left")
plt.tight_layout()
for fmt in ["png", "pdf"]:
    plt.savefig(FIGURES_DIR / f"02_scatter_wql_mase.{fmt}", dpi=300)
plt.close()
print("Figure 2 → 02_scatter_wql_mase.*")

# ------------------------------------------------------------------
# 9. PLOT 3 – HEATMAP OF RELATIVE IMPROVEMENT (%)
# ------------------------------------------------------------------
heatmap_data = df.pivot_table(
    index="dataset_name",
    values=["wql_pct", "mase_pct"],
    aggfunc="first"
).fillna(0)

plt.figure(figsize=(6, 10))
sns.heatmap(
    heatmap_data[["wql_pct", "mase_pct"]],
    annot=True, fmt=".1f", cmap="RdYlGn",
    center=0, cbar_kws={"label": "% change vs. paper"}
)
plt.title("Relative change (your run – paper) / paper × 100")
plt.ylabel("")
plt.tight_layout()
for fmt in ["png", "pdf"]:
    plt.savefig(FIGURES_DIR / f"03_heatmap_pct_change.{fmt}", dpi=300)
plt.close()
print("Figure 3 → 03_heatmap_pct_change.*")

# ------------------------------------------------------------------
# 10. PLOT 4 – BOX-PLOT OF SERIES COUNTS
# ------------------------------------------------------------------
plt.figure(figsize=(10, 5))
sns.boxplot(data=df, y="num_processed_series", color="#2ca02c")
plt.ylabel("# series evaluated")
plt.title("Distribution of evaluated series per dataset")
plt.tight_layout()
for fmt in ["png", "pdf"]:
    plt.savefig(FIGURES_DIR / f"04_series_count_boxplot.{fmt}", dpi=300)
plt.close()
print("Figure 4 → 04_series_count_boxplot.*")

# ------------------------------------------------------------------
# DONE
# ------------------------------------------------------------------
print("\nAll done! Check the `figures/` folder.")