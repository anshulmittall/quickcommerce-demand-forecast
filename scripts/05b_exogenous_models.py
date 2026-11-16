"""
05b_exogenous_models.py

Phase 2, step 4: the "X" versions of the classical models —
ARIMAX, SARIMAX and VARIMAX.

  X = eXogenous variables: extra inputs that are not sales, but help to
      explain sales. Here they are two calendar flags:
        snap_CA = 1 on days when SNAP food-benefit money can be spent
                  in California
        event   = 1 on days with a festival, holiday or sports event
  V = Vector: several series are forecast TOGETHER, so each one can
      learn from the past of the others.

ARIMAX and SARIMAX use exactly the same orders as ARIMA and SARIMA in
05_classical_models.py. The ONLY change is the two extra inputs, so the
difference in error shows what the exogenous variables are worth.

Each model is tested with the same walk-forward loop as everything else
(common.walk_forward).

Run from the project root, after 03_stationarity_acf.py.

Inputs:
  data/processed/daily_aggregate_sales.csv
  data/processed/daily_sales_by_category.csv   (FOODS, HOBBIES, HOUSEHOLD)
  data/raw/calendar.csv        (SNAP days and events, published in advance)

Outputs (in outputs/forecasts/):
  arimax.csv, sarimax.csv, varimax.csv
"""

import time
import warnings

from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from common import (HORIZON, SEASON, fill_closures, load_calendar_flags,
                    load_categories, load_daily, report, save_forecast,
                    walk_forward)

# statsmodels warns when an optimiser stops slightly early on one of the
# 52 refits. The forecasts are still valid, so the noise is hidden.
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

# ---- CONFIG ----------------------------------------------------
# Same orders as 05_classical_models.py (lowest AIC in step 03).
ARIMA_ORDER = (2, 1, 2)              # (p, d, q)
SARIMA_ORDER = (2, 0, 1)             # (p, d, q)
SARIMA_SEASONAL = (0, 1, 1, SEASON)  # (P, D, Q, season length)
VAR_LAGS = 28                        # VARIMAX looks back 4 weeks (lowest AIC in step 03)
# ----------------------------------------------------------------

exog = load_calendar_flags()      # SNAP and event flag for every day
categories = load_categories()    # FOODS, HOBBIES, HOUSEHOLD daily sales


# ============================================================
# 1. ARIMAX = ARIMA + exogenous variables
# ============================================================
def arimax(history, future_dates):
    """The same non-seasonal ARIMA as in step 05, plus the two calendar
    flags. The model learns how many extra units a SNAP day or an event
    day is worth, and ARIMA handles what is left over."""
    model = ARIMA(history, exog=exog.loc[history.index], order=ARIMA_ORDER)
    # To forecast, the model must be told next week's flags.
    return model.fit().forecast(HORIZON, exog=exog.loc[future_dates])


# ============================================================
# 2. SARIMAX = SARIMA + exogenous variables
# ============================================================
def sarimax(history, future_dates):
    """The same SARIMA as in step 05 (weekly seasonal part), plus the
    two calendar flags."""
    model = SARIMAX(history, exog=exog.loc[history.index],
                    order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL)
    return model.fit(disp=False).forecast(HORIZON, exog=exog.loc[future_dates])


# ============================================================
# 3. VARIMAX = Vector AR + Integration (differencing) + exogenous
# ============================================================
def varimax(history, future_dates):
    """Forecast FOODS together with HOBBIES and HOUSEHOLD of the same store.

    Idea: the same shoppers buy from all three categories, so a busy day
    for HOUSEHOLD may tell us something about FOODS in the coming days.

    V  = the three series are modelled together (a vector).
    AR = each series depends on the last VAR_LAGS days of ALL three.
    I  = seasonal differencing (today minus the same day last week),
         the same D = 1 that the ADF test chose for SARIMA.
    MA = not used (q = 0).
    X  = the SNAP and event flags (differenced the same way).

    With q = 0 this is the same model as statsmodels' VARMAX class with
    exog. The VAR class is used because it fits by least squares in
    milliseconds; VARMAX needs minutes for this many lags.
    """
    # The same past days as the FOODS history, for all three categories,
    # with the 25 December closure cleaned in each one.
    past = categories.loc[history.index].apply(fill_closures)

    # I: seasonal differencing. The first 7 days have no "last week".
    past_change = past.diff(SEASON).dropna()
    exog_change = exog.diff(SEASON)

    model = VAR(past_change, exog=exog_change.loc[past_change.index])
    fitted = model.fit(VAR_LAGS, trend="n")    # "n" = no constant (differenced data)

    # Forecast the CHANGE against last week for all three categories.
    # The model needs the last VAR_LAGS days and next week's flags.
    change = fitted.forecast(past_change.values[-VAR_LAGS:], steps=HORIZON,
                             exog_future=exog_change.loc[future_dates].values)
    foods_change = change[:, list(past.columns).index("FOODS")]

    # Add last week's sales back to return to units. Because the horizon
    # is exactly one week, "last week" is the last 7 known days.
    return past["FOODS"].values[-SEASON:] + foods_change


models = {
    "arimax": arimax,
    "sarimax": sarimax,
    "varimax": varimax,
}

series = load_daily()

print("--- Models with exogenous variables (walk-forward, 52 weeks) ---")
for name, forecast_fn in models.items():
    start = time.time()
    predicted = walk_forward(series, forecast_fn)
    actual = series.loc[predicted.index]

    save_forecast(name, predicted)
    report(name, actual, predicted, time.time() - start)

print("\nSaved exogenous-model forecasts to outputs/forecasts/")
