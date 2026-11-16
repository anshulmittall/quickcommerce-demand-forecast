"""
01_load_subset.py

Loads the full M5 dataset and extracts a manageable subset
(one category + one store) for the TSA Phase 1 synopsis EDA.

Run this from inside your project folder, with data/raw/ containing:
  sales_train_validation.csv, calendar.csv, sell_prices.csv

Output:
  data/processed/item_level_sales.csv   (item-level, long format — for later phases)
  data/processed/daily_aggregate_sales.csv  (daily total — for EDA line plot)
  data/processed/daily_sales_by_category.csv  (daily total of each category in the store — for VARIMAX)
"""

import pandas as pd
import os

# ---- CONFIG: change these if you want a different slice ----
CATEGORY = "FOODS"       # options: FOODS, HOBBIES, HOUSEHOLD
STORE_ID = "CA_1"        # options: e.g. CA_1, CA_2, TX_1, WI_1, etc.
# --------------------------------------------------------------

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

print("Loading raw files...")
sales = pd.read_csv(os.path.join(RAW_DIR, "sales_train_validation.csv"))
calendar = pd.read_csv(os.path.join(RAW_DIR, "calendar.csv"))

print(f"Full sales shape: {sales.shape}")

# Filter to chosen category + store
subset = sales[(sales["cat_id"] == CATEGORY) & (sales["store_id"] == STORE_ID)].copy()
print(f"Subset shape (category={CATEGORY}, store={STORE_ID}): {subset.shape}")

if subset.empty:
    raise ValueError(
        f"No rows found for category='{CATEGORY}' and store='{STORE_ID}'. "
        "Check the values against sales['cat_id'].unique() and sales['store_id'].unique()."
    )

# Identify the day columns (d_1, d_2, ... d_1913)
id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
day_cols = [c for c in subset.columns if c.startswith("d_")]

# Melt wide -> long: one row per (item_id, day)
long_df = subset.melt(
    id_vars=id_cols,
    value_vars=day_cols,
    var_name="d",
    value_name="sales"
)

# Merge with calendar to get actual dates
long_df = long_df.merge(calendar[["d", "date"]], on="d", how="left")
long_df["date"] = pd.to_datetime(long_df["date"])

# Aggregate to daily total sales across all items in this category+store
daily = long_df.groupby("date", as_index=False)["sales"].sum()
daily = daily.sort_values("date").reset_index(drop=True)

print(f"Daily aggregated series: {daily.shape[0]} days "
      f"({daily['date'].min().date()} to {daily['date'].max().date()})")

# Save both: full item-level long table (for later phases) and the daily aggregate (for EDA)
long_df[["date", "item_id", "sales"]].to_csv(
    os.path.join(PROCESSED_DIR, "item_level_sales.csv"), index=False
)
daily.to_csv(os.path.join(PROCESSED_DIR, "daily_aggregate_sales.csv"), index=False)

# Daily totals of ALL three categories in this store (FOODS, HOBBIES, HOUSEHOLD).
# Needed by the VARIMAX model, which forecasts several series together.
store = sales[sales["store_id"] == STORE_ID]
by_category = store.groupby("cat_id")[day_cols].sum().T      # rows = days, columns = categories
by_category.index = pd.to_datetime(calendar.set_index("d").loc[by_category.index, "date"].values)
by_category.index.name = "date"
by_category.to_csv(os.path.join(PROCESSED_DIR, "daily_sales_by_category.csv"))

print("Saved:")
print(f"  - {PROCESSED_DIR}/item_level_sales.csv (item-level, long format)")
print(f"  - {PROCESSED_DIR}/daily_aggregate_sales.csv (daily total, for EDA line plot)")
print(f"  - {PROCESSED_DIR}/daily_sales_by_category.csv (daily total per category, for VARIMAX)")
