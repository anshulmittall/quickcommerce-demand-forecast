"""
06_ml_models.py

Phase 3: the machine-learning / deep-learning models promised in the
synopsis — Prophet, XGBoost and LSTM.

Each model is tested with the same walk-forward loop as everything else
(common.walk_forward): retrain on all past data every Monday, forecast
the next 7 days, repeat for 52 weeks.

No setting in this file was tuned on the test year. The values in the
CONFIG block are ordinary defaults chosen before looking at test scores.

Run from the project root, after 01_load_subset.py. Takes about 4 minutes
(the LSTM is retrained 52 times).

Inputs:
  data/processed/daily_aggregate_sales.csv
  data/raw/calendar.csv        (SNAP days and events, published in advance)

Outputs (in outputs/forecasts/):
  prophet.csv, xgboost.csv, lstm.csv
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

from common import HORIZON, load_daily, report, save_forecast, walk_forward

# Prophet and TensorFlow print a status line for every one of the 52 refits.
# They are progress messages, not problems, so they are switched off.
logging.getLogger("cmdstanpy").disabled = True
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ---- CONFIG ----------------------------------------------------
LAGS = [7, 14, 21, 28]   # same weekday 1, 2, 3 and 4 weeks ago
LSTM_WINDOW = 28         # the LSTM reads the last 28 days...
LSTM_UNITS = 32          # ...with 32 memory cells...
LSTM_EPOCHS = 30         # ...and passes over the training data 30 times
SEED = 42                # fixed seed so the results can be reproduced
# ----------------------------------------------------------------

# The calendar is known in advance (it runs past the end of the sales data),
# so using a FUTURE day's SNAP flag or event flag is not leakage.
calendar = pd.read_csv("data/raw/calendar.csv")
calendar["date"] = pd.to_datetime(calendar["date"])
calendar = calendar.set_index("date")
calendar["event"] = calendar["event_name_1"].notna().astype(int)


# ============================================================
# 1. PROPHET
# ============================================================
def prophet(history, future_dates):
    """Prophet = trend + weekly pattern + yearly pattern + holiday effects,
    added together. US holidays are included because Walmart is a US store."""
    train = pd.DataFrame({"ds": history.index, "y": history.values})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    daily_seasonality=False)
    model.add_country_holidays(country_name="US")
    model.fit(train)
    future = pd.DataFrame({"ds": future_dates})
    return model.predict(future)["yhat"].values


# ============================================================
# 2. XGBOOST
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


def xgboost(history, future_dates):
    """Gradient-boosted decision trees on the feature table above."""
    # Put empty (NaN) rows for next week under the history, so the same
    # function builds features for both. The future sales stay unknown.
    unknown_future = pd.Series(np.nan, index=future_dates)
    features = make_features(pd.concat([history, unknown_future]))

    # The first 34 days have no 28-day history yet, so they are dropped.
    train_x = features.loc[history.index].dropna()
    train_y = history.loc[train_x.index]

    model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                         subsample=0.8, random_state=SEED)
    model.fit(train_x, train_y)
    return model.predict(features.loc[future_dates])


# ============================================================
# 3. LSTM
# ============================================================
def lstm(history, future_dates):
    """A small recurrent neural network: reads the last 28 days in order
    and outputs the next 7 days in one go."""
    keras.utils.set_random_seed(SEED)

    # Neural networks train badly on raw values in the thousands, so the
    # series is rescaled to mean 0 / std 1 using the HISTORY only.
    mean, std = history.mean(), history.std()
    scaled = ((history - mean) / std).values

    # Slide a window over the history to make training examples:
    # input = 28 days, target = the 7 days that followed.
    inputs, targets = [], []
    for start in range(len(scaled) - LSTM_WINDOW - HORIZON + 1):
        inputs.append(scaled[start:start + LSTM_WINDOW])
        targets.append(scaled[start + LSTM_WINDOW:start + LSTM_WINDOW + HORIZON])
    inputs = np.array(inputs).reshape(-1, LSTM_WINDOW, 1)   # (examples, days, 1 value per day)
    targets = np.array(targets)

    model = keras.Sequential([
        layers.Input(shape=(LSTM_WINDOW, 1)),
        layers.LSTM(LSTM_UNITS),
        layers.Dense(HORIZON),          # 7 outputs = 7 forecast days
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(inputs, targets, epochs=LSTM_EPOCHS, batch_size=32, verbose=0)

    last_window = scaled[-LSTM_WINDOW:].reshape(1, LSTM_WINDOW, 1)
    prediction = model.predict(last_window, verbose=0)[0]
    return prediction * std + mean      # undo the scaling -> back to units


models = {
    "prophet": prophet,
    "xgboost": xgboost,
    "lstm": lstm,
}

series = load_daily()

print("--- ML / DL models (walk-forward, 52 weeks) ---")
for name, forecast_fn in models.items():
    start = time.time()
    predicted = walk_forward(series, forecast_fn)
    actual = series.loc[predicted.index]

    save_forecast(name, predicted)
    report(name, actual, predicted, time.time() - start)

print("\nSaved ML forecasts to outputs/forecasts/")
