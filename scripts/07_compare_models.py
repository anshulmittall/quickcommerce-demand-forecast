"""
07_compare_models.py

Final step: put every forecast side by side and answer the project's
main question — how much better are proper forecasting models than the
simple rules used today, and WHERE exactly do the simple rules fail?

This script trains nothing. It only reads the forecasts saved by
steps 04, 05, 05b and 06, so the comparison cannot change a model.

Run from the project root, after 04, 05, 05b and 06.

Inputs:
  data/processed/daily_aggregate_sales.csv
  outputs/forecasts/*.csv

Outputs:
  outputs/metrics/model_comparison.csv     (the main results table)
  outputs/metrics/error_by_weekday.csv
  outputs/metrics/closure_fix_effect.csv
  outputs/metrics/exogenous_effect.csv
  outputs/figures/05_model_comparison.png
  outputs/figures/06_forecast_vs_actual.png
  outputs/figures/07_error_by_weekday.png
  outputs/figures/08_closure_fix.png
  outputs/figures/09_exogenous_effect.png
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (COLOR_ACTUAL, COLOR_EXISTING, COLOR_OURS, FIG_DIR, HORIZON,
                    METRIC_DIR, load_daily, load_forecast, score,
                    test_start_position, use_project_style)

os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
use_project_style()

# ---- CONFIG: file name -> (name shown in tables/figures, group) ----
MODELS = {
    "naive":            ("Naive (repeat last day)", "Existing practice"),
    "moving_average_7": ("7-day moving average", "Existing practice"),
    "seasonal_naive":   ("Seasonal naive (same day last week)", "Existing practice"),
    "arima":            ("ARIMA (no seasonality)", "Classical"),
    "holt_winters":     ("Holt-Winters", "Classical"),
    "sarima":           ("SARIMA", "Classical"),
    "arimax":           ("ARIMAX (ARIMA + calendar flags)", "Classical"),
    "sarimax":          ("SARIMAX (SARIMA + calendar flags)", "Classical"),
    "varimax":          ("VARIMAX (3 categories + calendar flags)", "Classical"),
    "prophet":          ("Prophet", "ML / DL"),
    "lstm":             ("LSTM", "ML / DL"),
    "xgboost":          ("XGBoost", "ML / DL"),
}
# The rule the synopsis named as current practice ("past-week averages").
REFERENCE = "moving_average_7"
# ---------------------------------------------------------------------

series = load_daily()
actual = series.iloc[test_start_position(series):]
print(f"Test period: {actual.index[0].date()} to {actual.index[-1].date()} "
      f"({len(actual)} days, {len(actual) // HORIZON} weekly forecasts)")

forecasts = {name: load_forecast(name) for name in MODELS}


def weekly_mae(predicted):
    """Average error of each of the 52 forecast weeks (7 days per week)."""
    absolute_error = np.abs(actual - predicted).values
    return absolute_error.reshape(-1, HORIZON).mean(axis=1)


# ============================================================
# 1. MAIN RESULTS TABLE
# ============================================================
reference_wape = score(actual, forecasts[REFERENCE])["WAPE_%"]
seasonal_naive_wape = score(actual, forecasts["seasonal_naive"])["WAPE_%"]
reference_weekly = weekly_mae(forecasts[REFERENCE])

rows = []
for name, (label, group) in MODELS.items():
    predicted = forecasts[name]
    s = score(actual, predicted)
    error = predicted - actual      # positive = forecast too high
    rows.append({
        "model": label,
        "group": group,
        "MAE": round(s["MAE"], 1),
        "RMSE": round(s["RMSE"], 1),
        "WAPE_%": round(s["WAPE_%"], 2),
        # How much lower the error is than the existing rules (positive = better).
        "error_cut_vs_moving_avg_%": round(100 * (1 - s["WAPE_%"] / reference_wape), 1),
        "error_cut_vs_seasonal_naive_%": round(100 * (1 - s["WAPE_%"] / seasonal_naive_wape), 1),
        # In how many of the 52 weeks did it beat the moving average?
        "weeks_better_than_moving_avg": int((weekly_mae(predicted) < reference_weekly).sum()),
        # Forecast too LOW  -> shelf runs empty (stockout risk).
        # Forecast too HIGH -> food left over (wastage risk).
        "units_under_forecast": int(round(-error[error < 0].sum())),
        "units_over_forecast": int(round(error[error > 0].sum())),
    })

table = pd.DataFrame(rows).sort_values("WAPE_%").reset_index(drop=True)
table.to_csv(os.path.join(METRIC_DIR, "model_comparison.csv"), index=False)
print("\nModel comparison (best first):")
print(table.to_string(index=False))

# Best model that is not one of the simple rules.
ours = table[table["group"] != "Existing practice"]
best_label = ours.iloc[0]["model"]
best_name = [name for name, (label, _) in MODELS.items() if label == best_label][0]
reference_label = MODELS[REFERENCE][0]
print(f"\nBest model: {best_label}")

# ============================================================
# 2. FIGURE — all models ranked by WAPE
# ============================================================
plot_table = table.iloc[::-1]     # reversed so the best bar is at the top
colors = [COLOR_EXISTING if g == "Existing practice" else COLOR_OURS
          for g in plot_table["group"]]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(plot_table["model"], plot_table["WAPE_%"], color=colors, height=0.55)
ax.bar_label(bars, labels=[f"{v:.1f}%" for v in plot_table["WAPE_%"]],
             padding=4, color="#0b0b0b")
ax.set_xlabel("WAPE on the 52-week test period (lower is better)")
ax.set_title("Forecast error of every method — FOODS, store CA_1")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR_EXISTING),
                   plt.Rectangle((0, 0), 1, 1, color=COLOR_OURS)],
          labels=["Existing practice (simple rules)", "Models built in this project"],
          loc="center right")
plt.savefig(os.path.join(FIG_DIR, "05_model_comparison.png"))
plt.close()
print("Saved: 05_model_comparison.png")

# ============================================================
# 3. FIGURE — forecast vs actual, last 8 weeks
# ============================================================
last = actual.index[-8 * HORIZON:]

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(last, actual.loc[last], color=COLOR_ACTUAL, linewidth=2, label="Actual sales")
ax.plot(last, forecasts[REFERENCE].loc[last], color=COLOR_EXISTING, linewidth=2,
        label=f"{reference_label} (existing practice)")
ax.plot(last, forecasts[best_name].loc[last], color=COLOR_OURS, linewidth=2,
        label=f"{best_label} (this project)")
ax.set_ylabel("Units sold")
ax.set_title("Forecast vs actual — last 8 weeks of the test period")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))   # e.g. "01 Mar"
ax.legend(loc="upper left", ncol=3)
plt.savefig(os.path.join(FIG_DIR, "06_forecast_vs_actual.png"))
plt.close()
print("Saved: 06_forecast_vs_actual.png")

# ============================================================
# 4. WHERE DOES THE EXISTING RULE FAIL? — error by weekday
# ============================================================
# Mean of (forecast - actual) per weekday.
# Above zero = too much stock ordered; below zero = too little.
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
weekday_error = pd.DataFrame({
    reference_label: (forecasts[REFERENCE] - actual).groupby(actual.index.day_name()).mean(),
    best_label: (forecasts[best_name] - actual).groupby(actual.index.day_name()).mean(),
}).loc[weekday_order].round(1)
weekday_error.index.name = "weekday"
weekday_error.to_csv(os.path.join(METRIC_DIR, "error_by_weekday.csv"))
print("\nAverage (forecast - actual) by weekday, in units:")
print(weekday_error.to_string())

positions = np.arange(len(weekday_order))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(positions - 0.2, weekday_error[reference_label], width=0.36,
       color=COLOR_EXISTING, label=f"{reference_label} (existing practice)")
ax.bar(positions + 0.2, weekday_error[best_label], width=0.36,
       color=COLOR_OURS, label=f"{best_label} (this project)")
ax.axhline(0, color="#0b0b0b", linewidth=1)
ax.set_xticks(positions)
ax.set_xticklabels(weekday_order)
ax.set_ylabel("Average forecast − actual (units)")
ax.set_title("Where the existing rule fails: too high on weekdays, too low on weekends")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left")
plt.savefig(os.path.join(FIG_DIR, "07_error_by_weekday.png"))
plt.close()
print("Saved: 07_error_by_weekday.png")

# ============================================================
# 5. WHAT IS THE CLOSURE FIX WORTH?
# ============================================================
# Compare each simple rule with and without the 25-December fix,
# over the whole year and over the two weeks around Christmas.
christmas_weeks = actual.loc["2015-12-21":"2016-01-03"].index

closure_rows = []
for name in ["naive", "seasonal_naive", "moving_average_7"]:
    raw = load_forecast(f"{name}_raw")
    fixed = forecasts[name]
    closure_rows.append({
        "rule": MODELS[name][0],
        "MAE_year_raw": round(score(actual, raw)["MAE"], 1),
        "MAE_year_with_fix": round(score(actual, fixed)["MAE"], 1),
        "MAE_christmas_weeks_raw":
            round(np.abs(actual - raw).loc[christmas_weeks].mean(), 1),
        "MAE_christmas_weeks_with_fix":
            round(np.abs(actual - fixed).loc[christmas_weeks].mean(), 1),
    })

closure_table = pd.DataFrame(closure_rows)
closure_table.to_csv(os.path.join(METRIC_DIR, "closure_fix_effect.csv"), index=False)
print("\nEffect of the closure fix on the simple rules:")
print(closure_table.to_string(index=False))

zoom = actual.loc["2015-12-14":"2016-01-10"].index
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(zoom, actual.loc[zoom], color=COLOR_ACTUAL, linewidth=2, label="Actual sales")
ax.plot(zoom, load_forecast("seasonal_naive_raw").loc[zoom], color=COLOR_EXISTING,
        linewidth=2, label="Seasonal naive, raw (existing practice)")
ax.plot(zoom, forecasts["seasonal_naive"].loc[zoom], color=COLOR_OURS,
        linewidth=2, linestyle="--", label="Seasonal naive + closure fix (this project)")
ax.set_ylabel("Units sold")
ax.set_title("The Christmas closure: the raw rule misses 25 Dec, then copies the zero to 1 Jan")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.legend(loc="lower left")
plt.savefig(os.path.join(FIG_DIR, "08_closure_fix.png"))
plt.close()
print("Saved: 08_closure_fix.png")

# ============================================================
# 6. WHAT ARE THE EXOGENOUS VARIABLES WORTH?
# ============================================================
# ARIMAX and SARIMAX are ARIMA and SARIMA with the same orders plus the
# SNAP and event flags, so each pair differs ONLY in the extra inputs.
exog_rows = []
for without, with_x in [("arima", "arimax"), ("sarima", "sarimax")]:
    wape_without = score(actual, forecasts[without])["WAPE_%"]
    wape_with = score(actual, forecasts[with_x])["WAPE_%"]
    exog_rows.append({
        "pair": f"{without.upper()} -> {with_x.upper()}",
        "WAPE_without_exog_%": round(wape_without, 2),
        "WAPE_with_exog_%": round(wape_with, 2),
        "error_cut_%": round(100 * (1 - wape_with / wape_without), 1),
    })

exog_table = pd.DataFrame(exog_rows)
exog_table.to_csv(os.path.join(METRIC_DIR, "exogenous_effect.csv"), index=False)
print("\nEffect of the exogenous variables (SNAP and event flags):")
print(exog_table.to_string(index=False))

positions = np.arange(len(exog_table))
fig, ax = plt.subplots(figsize=(8, 4.5))
bars_without = ax.bar(positions - 0.2, exog_table["WAPE_without_exog_%"], width=0.36,
                      color=COLOR_ACTUAL, label="Sales history only")
bars_with = ax.bar(positions + 0.2, exog_table["WAPE_with_exog_%"], width=0.36,
                   color=COLOR_OURS, label="Sales history + SNAP and event flags")
ax.bar_label(bars_without, fmt="%.1f%%", padding=3, color="#0b0b0b")
ax.bar_label(bars_with, fmt="%.1f%%", padding=3, color="#0b0b0b")
ax.set_xticks(positions)
ax.set_xticklabels(["ARIMA  →  ARIMAX", "SARIMA  →  SARIMAX"])
ax.set_ylabel("WAPE on the 52-week test (lower is better)")
ax.set_title("What the exogenous variables are worth")
ax.set_ylim(0, 15)      # head-room so the legend does not sit on the bars
ax.grid(axis="x", visible=False)
ax.legend(loc="upper right")
plt.savefig(os.path.join(FIG_DIR, "09_exogenous_effect.png"))
plt.close()
print("Saved: 09_exogenous_effect.png")

print("\nAll comparison outputs saved to outputs/metrics/ and outputs/figures/")
