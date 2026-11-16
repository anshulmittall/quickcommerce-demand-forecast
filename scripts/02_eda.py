"""
02_eda.py

Phase 1 EDA for the TSA synopsis: generates the three visuals required
by the guidelines — line plot, missing-value plot, and (optional)
seasonal decomposition — from the daily aggregate sales series.

Run this AFTER 01_load_subset.py, from inside your project folder.

Inputs:
  data/processed/daily_aggregate_sales.csv

Outputs (saved to outputs/figures/):
  01_line_plot.png
  02_missing_values.png
  03_seasonal_decomposition.png
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import os

PROCESSED_DIR = "data/processed"
FIG_DIR = "outputs/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ---- Load ----
df = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_aggregate_sales.csv"))
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print(f"Loaded {len(df)} days: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Sales range: min={df['sales'].min()}, max={df['sales'].max()}, mean={df['sales'].mean():.1f}")

# ============================================================
# 1. LINE PLOT — daily total sales over time
# ============================================================
plt.figure(figsize=(14, 5))
plt.plot(df["date"], df["sales"], linewidth=0.8, color="#2c7fb8")
plt.title("Daily Total Sales — FOODS Category, Store CA_1 (2011–2016)")
plt.xlabel("Date")
plt.ylabel("Units Sold")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "01_line_plot.png"), dpi=150)
plt.close()
print("Saved: 01_line_plot.png")

# ============================================================
# 2. MISSING / ZERO-SALES PLOT
# ============================================================
# M5 data has continuous dates by construction (no missing timestamps),
# but zero-sales days are common and worth visualizing as a data-quality check.
zero_days = df["sales"] == 0
print(f"Zero-sales days: {zero_days.sum()} out of {len(df)} "
      f"({100 * zero_days.mean():.2f}%)")

plt.figure(figsize=(14, 2.5))
plt.scatter(df["date"], zero_days.astype(int), s=4, color="#d95f0e")
plt.yticks([0, 1], ["Sales > 0", "Zero sales"])
plt.title("Zero-Sales Days Over Time (Data Quality Check)")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "02_missing_values.png"), dpi=150)
plt.close()
print("Saved: 02_missing_values.png")

# ============================================================
# 3. SEASONAL DECOMPOSITION — trend, seasonality, residual
# ============================================================
# Using weekly seasonality (period=7) since retail sales typically
# show strong day-of-week patterns (weekend spikes etc.)
series = df.set_index("date")["sales"]
decomposition = seasonal_decompose(series, model="additive", period=7)

fig = decomposition.plot()
fig.set_size_inches(14, 8)
fig.suptitle("Seasonal Decomposition (Weekly Period) — Daily Sales", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "03_seasonal_decomposition.png"), dpi=150)
plt.close()
print("Saved: 03_seasonal_decomposition.png")

print("\nAll EDA figures saved to outputs/figures/")
