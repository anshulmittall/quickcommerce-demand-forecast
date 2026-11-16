# Demand Forecasting for Quick-Commerce Retail (M5 Walmart data)

Time Series Analysis mini-project. We forecast the next 7 days of daily
FOODS sales for one Walmart store (CA_1) and measure how much better proper
forecasting models are than the simple rules used for replenishment today.

**Result:** XGBoost has a WAPE of 6.72%, against 16.98% for the
past-week-average rule. That is 60% less error, and XGBoost wins in 50 of the
52 test weeks. The best classical models are the ones with exogenous inputs:
VARIMAX (7.50%) and SARIMAX (7.64%).

## Why this project: what existing approaches lack

| Existing approach | How it works | What it lacks |
| --- | --- | --- |
| Past-week average rule (industry practice named in our synopsis) | One number, the mean of the last 7 days, is used for every day of next week | Ignores the weekday pattern: on average 412–474 units too high on Tue–Thu and 660–830 units too low on Sat–Sun. WAPE 16.98% |
| Seasonal naive (copy the same day last week) | Next Saturday = last Saturday | Trusts one noisy week and copies its anomalies: after the 25 Dec closure it forecasts 0 units for 1 Jan. WAPE 10.54% |
| M5 competition winners (Makridakis et al., 2022) | The winner averaged 220 LightGBM models over 30,490 item series, 28 days ahead, scored by WRMSSE | Built to win a leaderboard: heavy, hard to explain to a store planner, a 28-day horizon, and error that is not in units |

The M5 paper also reports that of 5,507 teams only 7.5% beat a simple
exponential-smoothing benchmark, and 51.6% did not even beat Naive. A complex
model is not automatically better, so here every model is tested against the
simple rules first.

What this project adds:

1. **Baselines first.** The existing rules are coded and scored
   (`04_baselines.py`), so every model has a number it must beat.
2. **One fair test.** All twelve methods go through the same walk-forward test:
   52 weekly forecasts, each 7 days ahead, made from past data only.
3. **Closure-aware forecasting.** The planned 25 December closure is cleaned
   from the history and forecast as zero. This alone cuts the Christmas-fortnight
   error of the simple rules by 38–40%.
4. **Planner-ready output.** A 7-day horizon at category level, error in units,
   split into under-forecast (stockout risk) and over-forecast (wastage risk).

We do **not** claim to beat the M5 winners on their own task (30,490 item
series, 28 days). The gap we fill is a light, explainable weekly forecast.

## How to run

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/01_load_subset.py
.venv/bin/python scripts/02_eda.py
.venv/bin/python scripts/03_stationarity_acf.py
.venv/bin/python scripts/04_baselines.py
.venv/bin/python scripts/05_classical_models.py
.venv/bin/python scripts/05b_exogenous_models.py
.venv/bin/python scripts/06_ml_models.py
.venv/bin/python scripts/07_compare_models.py
```

Run every script from the project root, in order. The whole pipeline takes
about 9 minutes on a laptop (05 takes about 1.5 minutes, 05b about 2.5 minutes,
06 about 4.5 minutes because the LSTM is retrained 52 times). `data/raw/` must contain the M5 files
`sales_train_validation.csv` and `calendar.csv` from Kaggle.

| Script | Phase | What it does |
| --- | --- | --- |
| `01_load_subset.py` | 1 | Cuts the M5 data down to FOODS / CA_1 and builds the daily series (plus the daily totals of all three categories, for VARIMAX) |
| `02_eda.py` | 1 | Line plot, zero-sales check, weekly decomposition |
| `03_stationarity_acf.py` | 2 | ADF test, ACF/PACF plots, ARIMA/SARIMA order and VARIMAX lag selection by AIC |
| `04_baselines.py` | 2 | The existing rules: naive, seasonal naive, 7-day moving average |
| `05_classical_models.py` | 2 | Holt-Winters, ARIMA, SARIMA |
| `05b_exogenous_models.py` | 2 | ARIMAX, SARIMAX, VARIMAX (models with exogenous inputs) |
| `06_ml_models.py` | 3 | Prophet, XGBoost, LSTM |
| `07_compare_models.py` | – | Results table and comparison figures (trains nothing) |
| `common.py` | – | Shared test period, walk-forward loop, closure rule, metrics, data loaders |

## How the models are tested

- **Data:** 1,913 days, 29 Jan 2011 to 24 Apr 2016.
- **Training:** the first 1,549 days. **Test:** the last 364 days
  (27 Apr 2015 to 24 Apr 2016).
- **Walk-forward:** every Monday of the test year the model gets only the data
  before that Monday, forecasts the next 7 days, then moves one week ahead.
  That gives 52 forecasts per model, and the model never sees the days it
  predicts.
- **Metrics:** MAE and RMSE in units, and WAPE (total error ÷ total sales).
  MAPE is not used because it divides by each day's sales and 25 December has
  zero sales.

## Results (52-week test)

| Method | Family | MAE | RMSE | WAPE | Error cut vs moving average | Weeks better (of 52) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | ML / DL | **194.5** | **254.7** | **6.72%** | **60.4%** | **50** |
| VARIMAX (3 categories + SNAP, events) | Classical | 217.1 | 291.3 | 7.50% | 55.8% | 49 |
| SARIMAX (SARIMA + SNAP, events) | Classical | 221.1 | 299.2 | 7.64% | 55.0% | 48 |
| LSTM | ML / DL | 235.8 | 318.2 | 8.15% | 52.0% | 48 |
| Prophet | ML / DL | 244.6 | 323.5 | 8.45% | 50.2% | 49 |
| SARIMA(2,0,1)(0,1,1)₇ | Classical | 254.3 | 334.2 | 8.79% | 48.3% | 49 |
| Holt-Winters | Classical | 267.4 | 350.5 | 9.24% | 45.6% | 47 |
| Seasonal naive | Existing practice | 305.0 | 392.9 | 10.54% | 38.0% | 44 |
| ARIMAX (ARIMA + SNAP, events) | Classical | 311.5 | 394.3 | 10.76% | 36.6% | 47 |
| ARIMA(2,1,2), no seasonal part | Classical | 347.6 | 435.9 | 12.01% | 29.3% | 43 |
| 7-day moving average | Existing practice | 491.5 | 594.7 | 16.98% | reference | – |
| Naive | Existing practice | 913.1 | 1,061.6 | 31.55% | −85.8% | 2 |

Every number here comes from `outputs/metrics/model_comparison.csv`.

What the numbers say:

- **Where the existing rule fails.** A flat weekly average is too high midweek
  and too low on the weekend (`outputs/figures/07_error_by_weekday.png`). Over
  the test year XGBoost cuts under-forecast units from 89,927 to 29,660 (−67%)
  and over-forecast units from 88,990 to 41,147 (−54%).
- **Seasonality beats complexity.** ARIMA without a seasonal part (12.01%) is
  worse than simply copying last week (10.54%).
- **Exogenous inputs help.** Adding two calendar inputs (SNAP days and event
  days) with the model orders unchanged cuts the error of ARIMA by 10.4%
  (12.01% → 10.76%, ARIMAX) and of SARIMA by 13.0% (8.79% → 7.64%, SARIMAX);
  see `outputs/metrics/exogenous_effect.csv`. ARIMAX is still behind seasonal
  naive: the inputs cannot replace the weekly seasonal part.
- **VARIMAX.** Forecasting FOODS together with HOBBIES and HOUSEHOLD of the
  same store, with the same two inputs, gives 7.50%, about the same as SARIMAX
  and second overall.
- **Deep learning did not win.** LSTM (8.15%) trails XGBoost and takes about
  4 minutes to test, against about 15 seconds.
- **Closure fix.** For seasonal naive, the average daily error in the Christmas
  fortnight (21 Dec to 3 Jan) falls from 766 to 458 units
  (`outputs/metrics/closure_fix_effect.csv`). In the main table every method,
  including the simple rules, already has this fix, so the 60% gain does not
  come from Christmas.

## Stationarity checks (Phase 2)

| Series | ADF p-value | Stationary at 5%? |
| --- | ---: | --- |
| Original | 0.081 | no |
| First difference | < 0.001 | yes |
| Seasonal difference (lag 7) | < 0.001 | yes |

So plain ARIMA uses d = 1, and SARIMA uses D = 1 with d = 0. The p and q
orders are the ones with the lowest AIC **on the training data only**
(`outputs/metrics/order_selection.csv`). ARIMAX and SARIMAX reuse those orders.

VARIMAX is a vector autoregression on the seasonally differenced series of the
three categories (the same D = 1), with the SNAP and event flags differenced
the same way and no moving-average term. It looks back 28 days: of 7, 14, 21
and 28 lags, 28 has the lowest AIC on the training data
(`outputs/metrics/var_lag_selection.csv`). With no MA term it is the same model
as statsmodels' `VARMAX(order=(28, 0))` with `exog`, but it is fitted with the
`VAR` class (least squares), which takes milliseconds instead of minutes.

## No-leakage rules

- Models only ever receive the history before the forecast week
  (`common.walk_forward`).
- XGBoost's sales features all look back at least 7 days, so all 7 forecast
  days can be predicted without any unknown value.
- SNAP days and event days come from the M5 calendar, which is published in
  advance. They are the exogenous inputs of ARIMAX, SARIMAX and VARIMAX.
- VARIMAX only uses the other two categories' sales from before the forecast
  week, never their sales in the week being forecast.
- The closure fix replaces a closure day with the same weekday one week
  **earlier**, never a later value.
- No model setting was tuned on the test year.

## Limits

- One store, one category and one test year. Other stores or categories may
  rank the models differently.
- M5 is US Walmart data used as a proxy; no public Indian quick-commerce
  dataset exists.
- Category-level totals only. Item-level (SKU) forecasts, prices and
  promotions are not modelled.
- VARIMAX was not tested without the other two categories, so the results do
  not show how much of its accuracy comes from them.
- XGBoost was the best of nine models on the same test year, so its lead is
  slightly optimistic; the 50-of-52-weeks result shows it is not one lucky week.

## References

1. Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy
   competition: Results, findings, and conclusions. *International Journal of
   Forecasting*, 38(4), 1346–1364.
2. Hyndman, R. J., & Athanasopoulos, G. (2021). *Forecasting: Principles and
   Practice* (3rd ed.). OTexts.
3. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time
   Series Analysis: Forecasting and Control* (5th ed.). Wiley.
4. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. *The American
   Statistician*, 72(1), 37–45.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
   *Proceedings of KDD '16*, 785–794.
6. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
   Computation*, 9(8), 1735–1780.
7. Kaggle. M5 Forecasting – Accuracy. https://www.kaggle.com/c/m5-forecasting-accuracy
