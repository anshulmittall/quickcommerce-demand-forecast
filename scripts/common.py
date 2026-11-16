"""
common.py

Things that every modelling script (03 to 07) must agree on:
the test period, the forecast horizon, the closure rule, the
walk-forward evaluation loop and the error metrics.

Keeping them in ONE file means every model is tested in exactly
the same way, so the final comparison is fair.

This file is imported by the other scripts. It is not run directly.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---- CONFIG ----------------------------------------------------
HORIZON = 7        # we forecast the next 7 days (one week of replenishment)
TEST_WEEKS = 52    # the last 52 weeks (364 days) are the test period
SEASON = 7         # weekly seasonality (found in the Phase 1 decomposition)
# ----------------------------------------------------------------

PROCESSED_DIR = "data/processed"
FORECAST_DIR = "outputs/forecasts"
METRIC_DIR = "outputs/metrics"
FIG_DIR = "outputs/figures"

# Colours used in every figure, so the same thing always has the same colour.
COLOR_ACTUAL = "#52514e"     # grey   = what really happened
COLOR_EXISTING = "#eb6834"   # orange = existing practice (simple rules)
COLOR_OURS = "#2a78d6"       # blue   = the models built in this project


def load_daily():
    """Load the daily FOODS / CA_1 sales as a Series indexed by date."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_aggregate_sales.csv"))
    df["date"] = pd.to_datetime(df["date"])
    series = df.set_index("date")["sales"].astype(float)
    # "D" = one row per calendar day. This also proves no date is missing.
    series = series.asfreq("D")
    return series


def load_categories():
    """Daily sales of all three categories of the store (FOODS, HOBBIES,
    HOUSEHOLD), one column each. Used by the VARIMAX model."""
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "daily_sales_by_category.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").asfreq("D").astype(float)


def load_calendar_flags():
    """The exogenous variables (the "X" in ARIMAX / SARIMAX / VARIMAX):
      snap_CA = 1 on days when SNAP food-benefit money can be spent in California
      event   = 1 on days with a festival, holiday or sports event
    The M5 calendar is published in advance (it runs past the end of the
    sales data), so using a FUTURE day's flags is not leakage."""
    calendar = pd.read_csv("data/raw/calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar.set_index("date")
    calendar["event"] = calendar["event_name_1"].notna().astype(int)
    return calendar[["snap_CA", "event"]].astype(float)


def test_start_position(series):
    """Row number where the test period begins (364 days before the end)."""
    return len(series) - TEST_WEEKS * HORIZON


def is_closure(dates):
    """True for 25 December. Phase 1 showed the store is closed that day
    every year (the only zero-sales days), so it is a PLANNED closure
    that is known in advance."""
    return (dates.month == 12) & (dates.day == 25)


def fill_closures(history):
    """Replace each closure day in the PAST data with the sales of the
    same weekday one week earlier.

    Why: a zero on 25 December is not real demand, it is a locked door.
    If we leave it in, a model learns a fake dip and the simple rules
    copy that zero into the next week's forecast.
    Only an older value is used, so there is no look-ahead.
    """
    filled = history.copy()
    for day in filled.index[is_closure(filled.index)]:
        week_before = day - pd.Timedelta(days=7)
        if week_before in filled.index:
            filled[day] = filled[week_before]
    return filled


def walk_forward(series, forecast_fn, closure_aware=True):
    """Test a model the way it would really be used.

    Every Monday of the test year:
      1. take ONLY the data before that Monday (the history),
      2. ask the model for the next 7 days,
      3. move one week ahead and repeat (52 times).

    forecast_fn(history, future_dates) must return 7 numbers.
    The model never sees the days it is asked to predict.

    closure_aware=True  -> closure days are cleaned in the history and
                           the forecast for a closure day is set to 0.
    closure_aware=False -> raw data, no special handling.
    """
    all_predictions = []
    for origin in range(test_start_position(series), len(series), HORIZON):
        history = series.iloc[:origin]
        future_dates = series.index[origin:origin + HORIZON]

        if closure_aware:
            history = fill_closures(history)

        prediction = np.asarray(forecast_fn(history, future_dates), dtype=float)
        prediction = np.clip(prediction, 0, None)   # sales cannot be negative

        if closure_aware:
            prediction[is_closure(future_dates)] = 0   # store is shut

        all_predictions.append(pd.Series(prediction, index=future_dates))

    return pd.concat(all_predictions)


def score(actual, predicted):
    """Four error measures, all 'lower is better'.

    MAE  = average size of the miss, in units.
    RMSE = like MAE but punishes big misses more.
    MAPE = average miss as a % of that day's sales. It divides by each
           day's sales, so a day with 0 sales (25 December, store closed)
           cannot be included and is left out.
    WAPE = total miss as a % of total sales. It divides only once, by the
           total, so it also works with the zero-sales day.
    """
    error = actual - predicted
    open_days = actual > 0      # MAPE cannot divide by zero
    return {
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "MAPE_%": 100 * np.mean(np.abs(error[open_days]) / actual[open_days]),
        "WAPE_%": 100 * np.sum(np.abs(error)) / np.sum(actual),
    }


def save_forecast(name, predicted):
    """Save one model's test-period forecast so step 07 can compare them."""
    os.makedirs(FORECAST_DIR, exist_ok=True)
    out = predicted.rename("forecast").rename_axis("date").reset_index()
    out.to_csv(os.path.join(FORECAST_DIR, f"{name}.csv"), index=False)


def load_forecast(name):
    """Read back a forecast saved by save_forecast()."""
    df = pd.read_csv(os.path.join(FORECAST_DIR, f"{name}.csv"))
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["forecast"]


def report(name, actual, predicted, seconds):
    """Print one line of results for a model."""
    s = score(actual, predicted)
    print(f"{name:<28} MAE={s['MAE']:7.1f}  RMSE={s['RMSE']:7.1f}  "
          f"MAPE={s['MAPE_%']:5.2f}%  WAPE={s['WAPE_%']:5.2f}%   ({seconds:.1f} s)")


def use_project_style():
    """One look for every figure: light background, faint grid, no box."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelcolor": "#52514e",
        "xtick.color": "#52514e",
        "ytick.color": "#52514e",
        "font.size": 10.5,
        "legend.frameon": False,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })
