"""
06_ml_models.py

Phase 3: the advanced machine-learning / deep-learning models —
Prophet, XGBoost and LSTM.

The script has two steps:

  STEP 1 - TUNING. A few settings of each model are tried on a
           VALIDATION year (the 52 weeks just before the test year).
           The setting with the lowest validation error is kept.
  STEP 2 - FINAL TEST. Each model, with its best setting, is tested on
           the test year with the same walk-forward loop as every other
           method (common.walk_forward).

The data is therefore used like this:

  | ---------- training ---------- | validation year | test year |
                                     choose settings   final score

The test year is never used to choose a setting.

Run from the project root, after 01_load_subset.py. Takes about 8 minutes
(tuning about 3 minutes, and the LSTM is retrained 52 times in step 2).

Inputs:
  data/processed/daily_aggregate_sales.csv
  data/raw/calendar.csv        (SNAP days and events, published in advance)

Outputs:
  outputs/metrics/tuning_results.csv           (every setting that was tried)
  outputs/forecasts/prophet.csv, xgboost.csv, lstm.csv
"""

import logging
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow start-up messages

import keras
import numpy as np
import pandas as pd
from keras import layers
from prophet import Prophet
from xgboost import XGBRegressor

from common import (HORIZON, METRIC_DIR, fill_closures, load_calendar_flags,
                    load_daily, report, save_forecast, score,
                    test_start_position, walk_forward)

# Prophet and TensorFlow print a status line for every refit.
# They are progress messages, not problems, so they are switched off.
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---- CONFIG ----------------------------------------------------
LAGS = [7, 14, 21, 28]   # XGBoost: same weekday 1, 2, 3 and 4 weeks ago
LSTM_EPOCHS = 30         # LSTM: passes over the training data
SEED = 42                # fixed seed so the results can be reproduced

# The settings tried in step 1 (a small grid for each model).
PROPHET_GRID = [{"changepoint_prior_scale": c} for c in [0.01, 0.05, 0.5]]
XGBOOST_GRID = [{"max_depth": d, "learning_rate": r}
                for d in [2, 4, 6] for r in [0.05, 0.1]]
LSTM_GRID = [{"units": u, "window": w} for u in [16, 32] for w in [14, 28]]
# ----------------------------------------------------------------

# The calendar is known in advance (it runs past the end of the sales data),
# so using a FUTURE day's SNAP flag or event flag is not leakage.
calendar = load_calendar_flags()


# ============================================================
# MODEL 1 - PROPHET
# ============================================================
def prophet(history, future_dates, changepoint_prior_scale):
    """Prophet = trend + weekly pattern + yearly pattern + holiday effects,
    added together. US holidays are included because Walmart is a US store.

    changepoint_prior_scale = how easily the trend may bend.
    Small = stiff trend, large = flexible trend.
    """
    train = pd.DataFrame({"ds": history.index, "y": history.values})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    daily_seasonality=False,
                    changepoint_prior_scale=changepoint_prior_scale)
    model.add_country_holidays(country_name="US")
    model.fit(train)
    future = pd.DataFrame({"ds": future_dates})
    return model.predict(future)["yhat"].values


# ============================================================
# MODEL 2 - XGBOOST
# ============================================================
def make_features(sales):
    """Turn the series into a table: one row per day, one column per clue.

    Every sales-based clue looks back AT LEAST 7 days. That is what makes
    it legal to predict all 7 days of next week at once: even for the 7th
    day, a 7-day lag is still a day we have already seen.
    """
    features = pd.DataFrame(index=sales.index)
    for lag in LAGS:
        features[f"lag_{lag}"] = sales.shift(lag)
    # shift(7) first, THEN average, so the window ends 7 days ago.
    features["mean_last_week"] = sales.shift(7).rolling(7).mean()
    features["mean_last_4_weeks"] = sales.shift(7).rolling(28).mean()
    # Calendar clues (known in advance).
    features["day_of_week"] = features.index.dayofweek
    features["month"] = features.index.month
    features["day_of_month"] = features.index.day
    features["snap"] = calendar.loc[features.index, "snap_CA"].values
    features["event"] = calendar.loc[features.index, "event"].values
    return features


def xgboost(history, future_dates, max_depth, learning_rate):
    """Gradient-boosted decision trees on the feature table above.

    max_depth     = how many questions one tree may ask (bigger = more complex).
    learning_rate = how much each new tree is allowed to correct.
    """
    # Put empty (NaN) rows for next week under the history, so the same
    # function builds features for both. The future sales stay unknown.
    unknown_future = pd.Series(np.nan, index=future_dates)
    features = make_features(pd.concat([history, unknown_future]))

    # The first 34 days have no 28-day history yet, so they are dropped.
    train_x = features.loc[history.index].dropna()
    train_y = history.loc[train_x.index]

    model = XGBRegressor(n_estimators=300, learning_rate=learning_rate,
                         max_depth=max_depth, subsample=0.8, random_state=SEED)
    model.fit(train_x, train_y)
    return model.predict(features.loc[future_dates])


# ============================================================
# MODEL 3 - LSTM
# ============================================================
def lstm_fit(history, units, window):
    """Train a small recurrent neural network that reads the last `window`
    days in order and outputs the next 7 days in one go.

    units  = number of memory cells.
    window = how many past days the network reads.
    """
    keras.utils.set_random_seed(SEED)

    # Neural networks train badly on raw values in the thousands, so the
    # series is rescaled to mean 0 / std 1 using the HISTORY only.
    mean, std = history.mean(), history.std()
    scaled = ((history - mean) / std).values

    # Slide a window over the history to make training examples:
    # input = `window` days, target = the 7 days that followed.
    inputs, targets = [], []
    for start in range(len(scaled) - window - HORIZON + 1):
        inputs.append(scaled[start:start + window])
        targets.append(scaled[start + window:start + window + HORIZON])
    inputs = np.array(inputs).reshape(-1, window, 1)   # (examples, days, 1 value per day)
    targets = np.array(targets)

    model = keras.Sequential([
        layers.Input(shape=(window, 1)),
        layers.LSTM(units),
        layers.Dense(HORIZON),          # 7 outputs = 7 forecast days
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(inputs, targets, epochs=LSTM_EPOCHS, batch_size=32, verbose=0)
    return model, mean, std


def lstm_predict(model, mean, std, history, window):
    """Forecast the next 7 days from the last `window` days of the history."""
    scaled = ((history - mean) / std).values
    last_window = scaled[-window:].reshape(1, window, 1)
    prediction = model.predict(last_window, verbose=0)[0]
    return prediction * std + mean      # undo the scaling -> back to units


def lstm(history, future_dates, units, window):
    """Train on the history, then forecast the next 7 days."""
    model, mean, std = lstm_fit(history, units, window)
    return lstm_predict(model, mean, std, history, window)


series = load_daily()

# ============================================================
# STEP 1 - TUNING ON THE VALIDATION YEAR
# ============================================================
# Cut the test year off. walk_forward() always tests on the last 364 days
# of what it is given, so on this shorter series it tests on the
# validation year (the 52 weeks just before the test year).
before_test = series.iloc[:test_start_position(series)]
validation_actual = before_test.iloc[test_start_position(before_test):]
print(f"Validation year: {validation_actual.index[0].date()} to "
      f"{validation_actual.index[-1].date()}")

tuning_rows = []    # one row per setting tried
best = {}           # model name -> its best setting


def try_setting(model_name, setting, predicted):
    """Score one setting on the validation year and remember it."""
    wape = score(validation_actual, predicted)["WAPE_%"]
    tuning_rows.append({"model": model_name, "setting": str(setting),
                        "validation_WAPE_%": round(wape, 2)})
    print(f"  {model_name:<8} {str(setting):<45} validation WAPE = {wape:5.2f}%")
    return wape


def keep_best(model_name, grid, wapes):
    """Keep the setting with the lowest validation error."""
    best[model_name] = grid[int(np.argmin(wapes))]


print("\n--- Step 1: tuning (validation year) ---")
start = time.time()

# Prophet and XGBoost are fast, so each setting gets the full walk-forward
# test on the validation year: 52 refits, exactly like the final test.
wapes = []
for setting in PROPHET_GRID:
    predicted = walk_forward(before_test, lambda h, f: prophet(h, f, **setting))
    wapes.append(try_setting("prophet", setting, predicted))
keep_best("prophet", PROPHET_GRID, wapes)

wapes = []
for setting in XGBOOST_GRID:
    predicted = walk_forward(before_test, lambda h, f: xgboost(h, f, **setting))
    wapes.append(try_setting("xgboost", setting, predicted))
keep_best("xgboost", XGBOOST_GRID, wapes)

# The LSTM is slow to train, so here each setting is trained ONCE, on the
# data before the validation year, and then forecasts all 52 validation
# weeks without retraining. That is enough to compare settings.
training_part = fill_closures(before_test.iloc[:test_start_position(before_test)])
wapes = []
for setting in LSTM_GRID:
    model, mean, std = lstm_fit(training_part, **setting)
    predicted = walk_forward(
        before_test,
        lambda h, f: lstm_predict(model, mean, std, h, setting["window"]))
    wapes.append(try_setting("lstm", setting, predicted))
keep_best("lstm", LSTM_GRID, wapes)

# Save every setting that was tried, and mark the chosen one.
tuning_table = pd.DataFrame(tuning_rows)
tuning_table["chosen"] = [
    "yes" if row["setting"] == str(best[row["model"]]) else ""
    for row in tuning_rows]
os.makedirs(METRIC_DIR, exist_ok=True)
tuning_table.to_csv(os.path.join(METRIC_DIR, "tuning_results.csv"), index=False)
print(f"Tuning took {time.time() - start:.0f} s. Chosen settings:")
for model_name, setting in best.items():
    print(f"  {model_name:<8} {setting}")

# ============================================================
# STEP 2 - FINAL TEST ON THE TEST YEAR
# ============================================================
models = {
    "prophet": lambda h, f: prophet(h, f, **best["prophet"]),
    "xgboost": lambda h, f: xgboost(h, f, **best["xgboost"]),
    "lstm": lambda h, f: lstm(h, f, **best["lstm"]),
}

print("\n--- Step 2: ML / DL models, best settings (walk-forward, 52 weeks) ---")
for name, forecast_fn in models.items():
    start = time.time()
    predicted = walk_forward(series, forecast_fn)
    actual = series.loc[predicted.index]

    save_forecast(name, predicted)
    report(name, actual, predicted, time.time() - start)

print("\nSaved ML forecasts to outputs/forecasts/")
