"""
03_stationarity_acf.py

Phase 2, step 1: check the series before fitting ARIMA / SARIMA.
ARIMA-type models need a STATIONARY series (constant mean and variance),
so this script decides how much differencing is needed and which
(p, q) orders to use. This is the Box-Jenkins method.

Only the TRAINING part is used here. The test year is never looked at,
so the model orders are not tuned on the data we later score on.

Run from the project root, after 01_load_subset.py.

Inputs:
  data/processed/daily_aggregate_sales.csv
  data/processed/daily_sales_by_category.csv   (for the VARIMAX lag choice)
  data/raw/calendar.csv                        (SNAP and event flags)

Outputs:
  outputs/metrics/adf_tests.csv          (stationarity test results)
  outputs/metrics/order_selection.csv    (AIC of every candidate order)
  outputs/metrics/var_lag_selection.csv  (AIC of every candidate VARIMAX lag length)
  outputs/figures/04_acf_pacf.png
"""

import itertools
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from common import (COLOR_OURS, FIG_DIR, METRIC_DIR, SEASON, fill_closures,
                    load_calendar_flags, load_categories, load_daily,
                    test_start_position, use_project_style)

# statsmodels prints a warning whenever one candidate order struggles to
# converge. Those candidates simply get a worse AIC, so the noise is hidden.
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", UserWarning)

os.makedirs(METRIC_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)
use_project_style()

# ---- Training data only ----
series = load_daily()
train = fill_closures(series.iloc[:test_start_position(series)])
print(f"Training data: {len(train)} days "
      f"({train.index[0].date()} to {train.index[-1].date()})")

# ============================================================
# 1. ADF TEST — is the series stationary?
# ============================================================
# Null hypothesis of ADF: "the series is NOT stationary".
# p-value < 0.05  -> reject the null -> stationary.
versions = {
    "Original series": train,
    "First difference (today - yesterday)": train.diff().dropna(),
    "Seasonal difference (today - same day last week)": train.diff(SEASON).dropna(),
}

adf_rows = []
for name, values in versions.items():
    result = adfuller(values, result_object=True)
    adf_rows.append({
        "series": name,
        "adf_statistic": round(result.statistic, 3),
        "p_value": round(result.pvalue, 4),
        "stationary_at_5%": "yes" if result.pvalue < 0.05 else "no",
    })

adf_table = pd.DataFrame(adf_rows)
adf_table.to_csv(os.path.join(METRIC_DIR, "adf_tests.csv"), index=False)
print("\nADF test results:")
print(adf_table.to_string(index=False))

# What the results mean for the model orders:
#   Original series is NOT stationary (p > 0.05)  -> plain ARIMA needs d = 1.
#   One seasonal difference makes it stationary   -> SARIMA uses D = 1, d = 0.

# ============================================================
# 2. ACF / PACF PLOTS
# ============================================================
# ACF  = correlation of the series with itself k days ago.
# PACF = the same, after removing the effect of the days in between.
# Spikes at lags 7, 14, 21... in the top row are the weekly pattern.
fig, axes = plt.subplots(2, 2, figsize=(14, 7))
seasonal_diff = train.diff(SEASON).dropna()

plot_acf(train, lags=35, ax=axes[0, 0], color=COLOR_OURS, vlines_kwargs={"colors": COLOR_OURS})
axes[0, 0].set_title("ACF — original series (spikes every 7 days)")
plot_pacf(train, lags=35, ax=axes[0, 1], color=COLOR_OURS, vlines_kwargs={"colors": COLOR_OURS})
axes[0, 1].set_title("PACF — original series")

plot_acf(seasonal_diff, lags=35, ax=axes[1, 0], color=COLOR_OURS, vlines_kwargs={"colors": COLOR_OURS})
axes[1, 0].set_title("ACF — after seasonal differencing (lag 7)")
plot_pacf(seasonal_diff, lags=35, ax=axes[1, 1], color=COLOR_OURS, vlines_kwargs={"colors": COLOR_OURS})
axes[1, 1].set_title("PACF — after seasonal differencing (lag 7)")

for ax in axes.flat:
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Correlation")

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "04_acf_pacf.png"))
plt.close()
print("\nSaved: 04_acf_pacf.png")

# ============================================================
# 3. ORDER SELECTION BY AIC
# ============================================================
# AIC rewards a good fit and penalises extra parameters; lower is better.
# AIC values are only comparable when the differencing is the same,
# so d and D are fixed from the ADF test and only p, q, P, Q are searched.
order_rows = []

print("\nFitting ARIMA candidates (d = 1)...")
for p, q in itertools.product([0, 1, 2], [0, 1, 2]):
    fitted = ARIMA(train, order=(p, 1, q)).fit()
    order_rows.append({"model": "ARIMA", "order": f"({p},1,{q})",
                       "seasonal_order": "-", "AIC": round(fitted.aic, 1)})

print("Fitting SARIMA candidates (d = 0, D = 1, season = 7)...")
for p, q, P, Q in itertools.product([0, 1, 2], [0, 1, 2], [0, 1], [0, 1]):
    fitted = SARIMAX(train, order=(p, 0, q),
                     seasonal_order=(P, 1, Q, SEASON)).fit(disp=False)
    order_rows.append({"model": "SARIMA", "order": f"({p},0,{q})",
                       "seasonal_order": f"({P},1,{Q},{SEASON})",
                       "AIC": round(fitted.aic, 1)})

order_table = pd.DataFrame(order_rows).sort_values(["model", "AIC"])
order_table.to_csv(os.path.join(METRIC_DIR, "order_selection.csv"), index=False)

print("\nBest 3 orders per model (lowest AIC first):")
print(order_table.groupby("model").head(3).to_string(index=False))
print("\nThe top order of each model is the one used in 05_classical_models.py")
print("(ARIMAX and SARIMAX in 05b_exogenous_models.py reuse the same orders.)")

# ============================================================
# 4. HOW MANY LAGS FOR VARIMAX?
# ============================================================
# VARIMAX forecasts FOODS together with HOBBIES and HOUSEHOLD. Its one
# setting is how many past days each series may look at.
# Candidates are whole weeks (1 to 4 weeks); the lowest AIC wins.
VAR_LAG_CANDIDATES = [7, 14, 21, 28]

# Same preparation as the model itself: training days only, closures
# cleaned, then seasonal differencing (D = 1, as chosen by the ADF test).
categories = load_categories().loc[train.index].apply(fill_closures)
category_change = categories.diff(SEASON).dropna()
exog_change = load_calendar_flags().diff(SEASON).loc[category_change.index]

# select_order fits a VAR for every lag count from 0 to maxlags on the
# same days, so the AIC values can be compared with each other.
selection = VAR(category_change, exog=exog_change).select_order(
    maxlags=max(VAR_LAG_CANDIDATES), trend="n")
aic_by_lag = selection.ics["aic"]      # position k = AIC of the model with k lags

var_table = pd.DataFrame({
    "lags_days": VAR_LAG_CANDIDATES,
    "AIC": [round(aic_by_lag[lags], 3) for lags in VAR_LAG_CANDIDATES],
}).sort_values("AIC")
var_table.to_csv(os.path.join(METRIC_DIR, "var_lag_selection.csv"), index=False)

print("\nVARIMAX lag candidates (lowest AIC first):")
print(var_table.to_string(index=False))
print("\nThe top row is VAR_LAGS in 05b_exogenous_models.py")
