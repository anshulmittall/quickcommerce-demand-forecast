# Demand Forecasting for Quick-Commerce Retail (M5 Walmart data)

Time Series Analysis mini-project. We forecast the next 7 days of daily
FOODS sales for one Walmart store (CA_1) and measure how much better proper
forecasting models are than the simple rules used for replenishment today.

**Result:** XGBoost has a MAPE of 6.88% (RMSE 254.7 units), against 17.08%
(RMSE 594.7) for the past-week-average rule. That is 60% less error, and
XGBoost wins in 50 of the 52 test weeks. The best classical models are the ones
with exogenous inputs: VARIMAX (7.67%) and SARIMAX (7.84%).

Deliverables: the report (`report/`, IEEE format, PDF and Word), the slides
(`slides/`) and the code (`scripts/`).

## Why this project: what existing approaches lack

| Existing approach | How it works | What it lacks |
| --- | --- | --- |
| Past-week average rule (industry practice named in our synopsis) | One number, the mean of the last 7 days, is used for every day of next week | Ignores the weekday pattern: on average 412–474 units too high on Tue–Thu and 660–830 units too low on Sat–Sun. MAPE 17.08% |
| Seasonal naive (copy the same day last week) | Next Saturday = last Saturday | Trusts one noisy week and copies its anomalies: after the 25 Dec closure it forecasts 0 units for 1 Jan. MAPE 10.70% |
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
.venv/bin/python scripts/08_report_figures.py
```

Run every script from the project root, in order. The whole pipeline takes
about 15 minutes on a laptop (05 takes about 1.5 minutes, 05b about 2.5 minutes,
06 about 8 minutes: tuning on the validation year, then the LSTM is retrained
52 times). `data/raw/` must contain the M5 files
`sales_train_validation.csv` and `calendar.csv` from Kaggle.

| Script | Phase | What it does |
| --- | --- | --- |
| `01_load_subset.py` | 1 | Cuts the M5 data down to FOODS / CA_1 and builds the daily series (plus the daily totals of all three categories, for VARIMAX) |
| `02_eda.py` | 1 | Line plot, zero-sales check, weekly decomposition |
| `03_stationarity_acf.py` | 2 | ADF test, ACF/PACF plots, ARIMA/SARIMA order and VARIMAX lag selection by AIC |
| `04_baselines.py` | 2 | The existing rules: naive, seasonal naive, 7-day moving average |
| `05_classical_models.py` | 2 | Holt-Winters, ARIMA, SARIMA |
| `05b_exogenous_models.py` | 2 | ARIMAX, SARIMAX, VARIMAX (models with exogenous inputs) |
| `06_ml_models.py` | 3 | Prophet, XGBoost, LSTM: tuning on the validation year, then the final test |
| `07_compare_models.py` | – | Results table and comparison figures (trains nothing) |
| `08_report_figures.py` | – | One-column figures for the IEEE report (trains nothing) |
| `common.py` | – | Shared test period, walk-forward loop, closure rule, metrics, data loaders |

## How the models are tested

- **Data:** 1,913 days, 29 Jan 2011 to 24 Apr 2016.
- **Training:** the first 1,185 days. **Validation:** the next 364 days
  (28 Apr 2014 to 26 Apr 2015), used only to tune the advanced models.
  **Test:** the last 364 days (27 Apr 2015 to 24 Apr 2016), used only for the
  final scores. (The classical orders are chosen by AIC on training + validation.)
- **Walk-forward:** every Monday of the test year the model gets only the data
  before that Monday, forecasts the next 7 days, then moves one week ahead.
  That gives 52 forecasts per model, and the model never sees the days it
  predicts.
- **Metrics:** RMSE and MAE in units, MAPE in %, and WAPE (total error ÷
  total sales). MAPE divides by each day's sales, so the one zero-sales day of
  the test year (25 December 2015) is left out of it; WAPE includes every day.

### Hyperparameter tuning (advanced models)

Each setting forecast the 52 weeks of the validation year; the lowest error won
(`outputs/metrics/tuning_results.csv`). The test year was never used.

| Model | Settings tried | Chosen | Validation WAPE |
| --- | --- | --- | ---: |
| Prophet | changepoint prior scale 0.01, 0.05, 0.5 | 0.05 | 8.53% |
| XGBoost | max depth 2, 4, 6 × learning rate 0.05, 0.1 | depth 4, rate 0.05 | 6.85% |
| LSTM | units 16, 32 × window 14, 28 days | 32 units, 28 days | 8.58% |

## Results (52-week test)

| Method | Family | RMSE | MAE | MAPE | WAPE | Weeks better than moving avg (of 52) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **XGBoost** | Advanced | **254.7** | **194.5** | **6.88%** | **6.72%** | **50** |
| VARIMAX (3 categories + SNAP, events) | Classical | 291.3 | 217.1 | 7.67% | 7.50% | 49 |
| SARIMAX (SARIMA + SNAP, events) | Classical | 299.2 | 221.1 | 7.84% | 7.64% | 48 |
| LSTM | Advanced | 318.2 | 235.8 | 8.37% | 8.15% | 48 |
| Prophet | Advanced | 323.5 | 244.6 | 8.54% | 8.45% | 49 |
| SARIMA(2,0,1)(0,1,1)₇ | Classical | 334.2 | 254.3 | 9.02% | 8.79% | 49 |
| Holt-Winters | Classical | 350.5 | 267.4 | 9.62% | 9.24% | 47 |
| Seasonal naive | Existing practice | 392.9 | 305.0 | 10.70% | 10.54% | 44 |
| ARIMAX (ARIMA + SNAP, events) | Classical | 394.3 | 311.5 | 11.04% | 10.76% | 47 |
| ARIMA(2,1,2), no seasonal part | Classical | 435.9 | 347.6 | 12.64% | 12.01% | 43 |
| 7-day moving average | Existing practice | 594.7 | 491.5 | 17.08% | 16.98% | – |
| Naive | Existing practice | 1,061.6 | 913.1 | 35.67% | 31.55% | 2 |

Every number here comes from `outputs/metrics/model_comparison.csv`.

What the numbers say:

- **Where the existing rule fails.** A flat weekly average is too high midweek
  and too low on the weekend (`outputs/figures/07_error_by_weekday.png`). Over
  the test year XGBoost cuts under-forecast units from 89,927 to 29,660 (−67%)
  and over-forecast units from 88,990 to 41,147 (−54%).
- **Seasonality beats complexity.** ARIMA without a seasonal part (MAPE 12.64%)
  is worse than simply copying last week (10.70%).
- **Exogenous inputs help.** Adding two calendar inputs (SNAP days and event
  days) with the model orders unchanged cuts the MAPE of ARIMA from 12.64% to
  11.04% (ARIMAX) and of SARIMA from 9.02% to 7.84% (SARIMAX), about 13% each. ARIMAX is still behind seasonal
  naive: the inputs cannot replace the weekly seasonal part.
- **VARIMAX.** Forecasting FOODS together with HOBBIES and HOUSEHOLD of the
  same store, with the same two inputs, gives a MAPE of 7.67%, about the same as SARIMAX
  and second overall.
- **Deep learning did not win.** LSTM (MAPE 8.37%) trails XGBoost, VARIMAX and
  SARIMAX: about 1,900 daily points is little data, and it sees no calendar
  inputs. It is also the slowest model to retrain.
- **Closure fix.** For seasonal naive, the average daily error in the Christmas
  fortnight (21 Dec to 3 Jan) falls from 766 to 458 units
  (`outputs/metrics/closure_fix_effect.csv`). In the main table every method,
  including the simple rules, already has this fix, so the gain of XGBoost does not
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
- No model setting was tuned on the test year: the advanced models were tuned
  on the validation year.

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

The report uses these ten references in IEEE format:

1. S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "M5 accuracy competition: Results, findings, and conclusions," *Int. J. Forecast.*, vol. 38, no. 4, pp. 1346–1364, 2022.
2. "M5 Forecasting – Accuracy," Kaggle, 2020. https://www.kaggle.com/c/m5-forecasting-accuracy
3. S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods," *Int. J. Forecast.*, vol. 36, no. 1, pp. 54–74, 2020.
4. R. Fildes, S. Ma, and S. Kolassa, "Retail forecasting: Research and practice," *Int. J. Forecast.*, vol. 38, no. 4, pp. 1283–1318, 2022.
5. G. E. P. Box, G. M. Jenkins, G. C. Reinsel, and G. M. Ljung, *Time Series Analysis: Forecasting and Control*, 5th ed. Wiley, 2015.
6. R. J. Hyndman and G. Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. OTexts, 2021.
7. S. J. Taylor and B. Letham, "Forecasting at scale," *Amer. Statist.*, vol. 72, no. 1, pp. 37–45, 2018.
8. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD*, 2016, pp. 785–794.
9. S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Comput.*, vol. 9, no. 8, pp. 1735–1780, 1997.
10. R. J. Hyndman and A. B. Koehler, "Another look at measures of forecast accuracy," *Int. J. Forecast.*, vol. 22, no. 4, pp. 679–688, 2006.
