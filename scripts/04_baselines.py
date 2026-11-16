"""
04_baselines.py

Phase 2, step 2: the EXISTING practice we want to beat.

The synopsis (slide 4) said replenishment today is mostly rule-based:
"past-week sales averages" and simple moving averages. This script
turns those rules into forecasts so they can be measured. Any model
built later must beat these numbers, otherwise it adds nothing.

Each rule is run twice:
  - raw            : exactly as the rule is used today (no closure handling)
  - closure-aware  : with our 25-December fix from common.py
The closure-aware version goes into the main comparison so that every
method gets the same treatment. The raw version is kept to show, in
step 07, what the fix alone is worth.

Run from the project root, after 01_load_subset.py.

Inputs:
  data/processed/daily_aggregate_sales.csv

Outputs (in outputs/forecasts/):
  naive.csv, seasonal_naive.csv, moving_average_7.csv        (closure-aware)
  naive_raw.csv, seasonal_naive_raw.csv, moving_average_7_raw.csv
"""

import time

import numpy as np

from common import (HORIZON, SEASON, load_daily, report, save_forecast,
                    walk_forward)


def naive(history, future_dates):
    """Tomorrow = today. Repeat the last known value for all 7 days."""
    return np.repeat(history.iloc[-1], HORIZON)


def seasonal_naive(history, future_dates):
    """Next Monday = last Monday, next Tuesday = last Tuesday, ...
    i.e. copy the last 7 days."""
    return history.iloc[-SEASON:].values


def moving_average_7(history, future_dates):
    """The 'past-week average' rule: one number (the mean of the last
    7 days) used for every one of the next 7 days."""
    return np.repeat(history.iloc[-7:].mean(), HORIZON)


baselines = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "moving_average_7": moving_average_7,
}

series = load_daily()

for closure_aware in [True, False]:
    label = "closure-aware" if closure_aware else "raw (as used today)"
    print(f"\n--- Baselines, {label} ---")

    for name, forecast_fn in baselines.items():
        start = time.time()
        predicted = walk_forward(series, forecast_fn, closure_aware=closure_aware)
        actual = series.loc[predicted.index]

        file_name = name if closure_aware else f"{name}_raw"
        save_forecast(file_name, predicted)
        report(file_name, actual, predicted, time.time() - start)

print("\nSaved baseline forecasts to outputs/forecasts/")
