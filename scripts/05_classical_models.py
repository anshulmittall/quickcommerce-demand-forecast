"""
05_classical_models.py

Phase 2, step 3: the classical statistical models promised in the
synopsis — Holt-Winters, ARIMA and SARIMA.

Each model is tested with the same walk-forward loop as the baselines
(common.walk_forward): refit on all past data every Monday, forecast
the next 7 days, repeat for 52 weeks.

Run from the project root, after 03_stationarity_acf.py.

Inputs:
  data/processed/daily_aggregate_sales.csv

Outputs (in outputs/forecasts/):
  holt_winters.csv, arima.csv, sarima.csv
"""

import time
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (HORIZON, SEASON, load_daily, report, save_forecast,
                    walk_forward)

# statsmodels warns when an optimiser stops slightly early on one of the
# 52 refits. The forecasts are still valid, so the noise is hidden.
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

# ---- CONFIG: orders with the lowest AIC in 03_stationarity_acf.py ----
ARIMA_ORDER = (2, 1, 2)             # (p, d, q)
SARIMA_ORDER = (2, 0, 1)            # (p, d, q)
SARIMA_SEASONAL = (0, 1, 1, SEASON)  # (P, D, Q, season length)
# ----------------------------------------------------------------------


def holt_winters(history, future_dates):
    """Triple exponential smoothing: a level, a trend and 7 weekday
    effects, each updated with more weight on recent days.
    Additive, to match the additive decomposition of Phase 1."""
    model = ExponentialSmoothing(history, trend="add", seasonal="add",
                                 seasonal_periods=SEASON)
    return model.fit().forecast(HORIZON)


def arima(history, future_dates):
    """ARIMA with NO seasonal part. Included on purpose: it shows what
    happens when the weekly pattern is ignored."""
    model = ARIMA(history, order=ARIMA_ORDER)
    return model.fit().forecast(HORIZON)


def sarima(history, future_dates):
    """ARIMA plus a seasonal part of length 7, so it compares each day
    with the same weekday in earlier weeks."""
    model = SARIMAX(history, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL)
    return model.fit(disp=False).forecast(HORIZON)


models = {
    "holt_winters": holt_winters,
    "arima": arima,
    "sarima": sarima,
}

series = load_daily()

print("--- Classical models (walk-forward, 52 weeks) ---")
for name, forecast_fn in models.items():
    start = time.time()
    predicted = walk_forward(series, forecast_fn)
    actual = series.loc[predicted.index]

    save_forecast(name, predicted)
    report(name, actual, predicted, time.time() - start)

print("\nSaved classical forecasts to outputs/forecasts/")
