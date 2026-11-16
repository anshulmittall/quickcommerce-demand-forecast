"""
08_report_figures.py

Draws the figures used in the final report. The report is in two-column
(IEEE) format, so every figure here is one column wide (3.4 inches) with
small text. The numbers are the same as in steps 01 to 07; nothing is
trained here.

Run from the project root, after 07_compare_models.py.

Inputs:
  data/processed/daily_aggregate_sales.csv
  outputs/forecasts/*.csv
  outputs/metrics/model_comparison.csv, error_by_weekday.csv

Outputs (in outputs/figures/report/):
  fig1_sales_series.png    the full daily series with the three periods
  fig2_workflow.png        the workflow of the project
  fig3_acf.png             ACF before and after seasonal differencing
  fig4_comparison.png      MAPE of every method
  fig5_forecast.png        forecast against actual, last 8 weeks
  fig6_weekday_error.png   average error on each weekday
"""

import os

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from statsmodels.graphics.tsaplots import plot_acf

from common import (COLOR_ACTUAL, COLOR_EXISTING, COLOR_OURS, FIG_DIR, HORIZON,
                    METRIC_DIR, SEASON, fill_closures, load_daily,
                    load_forecast, test_start_position, use_project_style)

REPORT_DIR = os.path.join(FIG_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)

COLUMN_WIDTH = 3.4   # inches: one column of a two-column A4 page

use_project_style()
plt.rcParams.update({
    "font.family": "Times New Roman",   # same font as the report text
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300,
})


def save(name):
    plt.savefig(os.path.join(REPORT_DIR, name))
    plt.close()
    print("Saved:", name)


series = load_daily()
test_start = series.index[test_start_position(series)]
validation_start = test_start - pd.Timedelta(days=364)
actual = series.loc[test_start:]

table = pd.read_csv(os.path.join(METRIC_DIR, "model_comparison.csv"))
best_label = table[table["group"] != "Existing practice"].iloc[0]["model"]
# File name of the best model's forecast, e.g. "XGBoost" -> "xgboost".
best_name = best_label.split(" ")[0].lower().replace("-", "_")

# ============================================================
# FIG 1 - the series and how it is split
# ============================================================
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 1.9))
ax.plot(series.index, series.values, color=COLOR_ACTUAL, linewidth=0.4)
ax.axvspan(validation_start, test_start, color=COLOR_EXISTING, alpha=0.15, linewidth=0)
ax.axvspan(test_start, series.index[-1], color=COLOR_OURS, alpha=0.18, linewidth=0)
top = series.max() * 1.12
ax.set_ylim(0, top)
ax.text(series.index[0] + (validation_start - series.index[0]) / 2, top * 0.95,
        "training", ha="center", va="top")
ax.text(validation_start + pd.Timedelta(days=182), top * 0.95, "validation",
        ha="center", va="top")
ax.text(test_start + pd.Timedelta(days=182), top * 0.95, "test", ha="center", va="top")
ax.set_ylabel("Units sold per day")
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
save("fig1_sales_series.png")

# ============================================================
# FIG 2 - workflow diagram
# ============================================================
steps = [
    "M5 raw data (Kaggle): sales and calendar files",
    "Subset and aggregate: FOODS, store CA_1, daily total",
    "Exploration: trend, weekly cycle, 25 Dec closures",
    "Stationarity: ADF test, ACF / PACF, orders by AIC",
    "Models: 3 simple rules, 6 classical, 3 advanced",
    "Tuning of the advanced models on the validation year",
    "Walk-forward test: 52 weekly forecasts of 7 days",
    "Metrics: MAE, RMSE, MAPE, WAPE",
    "Comparison, insights and limitations",
]
box_height, gap = 0.62, 0.3
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 4.3))
ax.set_xlim(0, 10)
ax.set_ylim(0, len(steps) * (box_height + gap))
ax.axis("off")
for i, text in enumerate(steps):
    y = (len(steps) - 1 - i) * (box_height + gap) + gap / 2
    advanced = "advanced" in text.lower() or "tuning" in text.lower()
    ax.add_patch(FancyBboxPatch((0.4, y), 9.2, box_height,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor="#eaf1fb" if advanced else "white",
                                edgecolor=COLOR_OURS if advanced else COLOR_ACTUAL,
                                linewidth=0.8))
    ax.text(5, y + box_height / 2, text, ha="center", va="center")
    if i < len(steps) - 1:      # arrow to the next step
        ax.annotate("", xy=(5, y - gap + 0.02), xytext=(5, y - 0.02),
                    arrowprops={"arrowstyle": "-|>", "color": COLOR_ACTUAL, "linewidth": 0.8})
save("fig2_workflow.png")

# ============================================================
# FIG 3 - ACF before and after the seasonal difference
# ============================================================
train = fill_closures(series.iloc[:test_start_position(series)])
fig, axes = plt.subplots(2, 1, figsize=(COLUMN_WIDTH, 2.9), sharex=True)
plot_acf(train, lags=35, ax=axes[0], color=COLOR_OURS, markersize=2,
         vlines_kwargs={"colors": COLOR_OURS, "linewidth": 0.8})
axes[0].set_title("(a) Original series")
plot_acf(train.diff(SEASON).dropna(), lags=35, ax=axes[1], color=COLOR_OURS, markersize=2,
         vlines_kwargs={"colors": COLOR_OURS, "linewidth": 0.8})
axes[1].set_title("(b) After a seasonal difference of 7 days")
axes[1].set_xlabel("Lag (days)")
for ax in axes:
    ax.set_ylabel("ACF")
plt.tight_layout()
save("fig3_acf.png")

# ============================================================
# FIG 4 - MAPE of every method
# ============================================================
plot_table = table.sort_values("MAPE_%", ascending=False)
# Short names: drop the text in brackets, e.g. "ARIMAX (ARIMA + ...)" -> "ARIMAX".
names = [m.split(" (")[0] for m in plot_table["model"]]
colors = [COLOR_EXISTING if g == "Existing practice" else COLOR_OURS
          for g in plot_table["group"]]
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.7))
bars = ax.barh(names, plot_table["MAPE_%"], color=colors, height=0.6)
ax.bar_label(bars, labels=[f"{v:.1f}" for v in plot_table["MAPE_%"]], padding=2)
ax.set_xlabel("MAPE on the test year (%)")
ax.set_xlim(0, plot_table["MAPE_%"].max() * 1.12)
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR_EXISTING),
                   plt.Rectangle((0, 0), 1, 1, color=COLOR_OURS)],
          labels=["Simple rules", "Models of this project"], loc="upper right")
save("fig4_comparison.png")

# ============================================================
# FIG 5 - forecast against actual, last 8 weeks
# ============================================================
last = actual.index[-8 * HORIZON:]
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.0))
ax.plot(last, actual.loc[last], color=COLOR_ACTUAL, linewidth=1.2, label="Actual")
ax.plot(last, load_forecast("moving_average_7").loc[last], color=COLOR_EXISTING,
        label="7-day moving average")
ax.plot(last, load_forecast(best_name).loc[last], color=COLOR_OURS,
        label=best_label.split(" (")[0])
ax.set_ylabel("Units sold per day")
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.set_ylim(top=actual.loc[last].max() * 1.22)     # room for the legend
ax.legend(loc="upper center", ncol=3, columnspacing=1.0, handlelength=1.5)
save("fig5_forecast.png")

# ============================================================
# FIG 6 - average error on each weekday
# ============================================================
weekday = pd.read_csv(os.path.join(METRIC_DIR, "error_by_weekday.csv"), index_col="weekday")
positions = np.arange(len(weekday))
fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, 2.1))
ax.bar(positions - 0.2, weekday.iloc[:, 0], width=0.38, color=COLOR_EXISTING,
       label=weekday.columns[0])
ax.bar(positions + 0.2, weekday.iloc[:, 1], width=0.38, color=COLOR_OURS,
       label=weekday.columns[1].split(" (")[0])
ax.axhline(0, color="#0b0b0b", linewidth=0.6)
ax.set_xticks(positions)
ax.set_xticklabels([day[:3] for day in weekday.index])
ax.set_ylabel("Forecast − actual (units)")
ax.grid(axis="x", visible=False)
ax.legend(loc="lower left")
save("fig6_weekday_error.png")

print("\nReport figures saved to", REPORT_DIR)
