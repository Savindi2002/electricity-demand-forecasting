
import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "electricity_weather.csv"

OUTPUT_DIR = BASE_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df["DATE"] = pd.to_datetime(df["DATE"])

df = df.sort_values("DATE").reset_index(drop=True)

print("Original dataset:", df.shape)

# ==========================================
# 3. CHECK DUPLICATE DATES
# ==========================================

duplicate_mask = df["DATE"].duplicated(keep=False)

duplicate_records = df[duplicate_mask].copy()

duplicate_records.to_csv(
    OUTPUT_DIR / "excluded_duplicate_dates.csv",
    index=False
)

print("\nDuplicate records:", len(duplicate_records))

# Exclude ALL records belonging to ambiguous dates.
df = df[~duplicate_mask].copy()

df = df.sort_values("DATE").reset_index(drop=True)

print("After duplicate-date exclusion:", df.shape)

# ==========================================
# 4. CREATE DATE FEATURES
# ==========================================

df["year"] = df["DATE"].dt.year

df["month"] = df["DATE"].dt.month

df["day"] = df["DATE"].dt.day

df["day_of_week"] = df["DATE"].dt.dayofweek

df["day_of_year"] = df["DATE"].dt.dayofyear

df["is_weekend"] = (
    df["day_of_week"] >= 5
)

# Cyclical month encoding
df["month_sin"] = np.sin(
    2 * np.pi * df["month"] / 12
)

df["month_cos"] = np.cos(
    2 * np.pi * df["month"] / 12
)

# Cyclical weekday encoding
df["weekday_sin"] = np.sin(
    2 * np.pi * df["day_of_week"] / 7
)

df["weekday_cos"] = np.cos(
    2 * np.pi * df["day_of_week"] / 7
)

# ==========================================
# 5. REMOVE UNSAFE OLD FEATURES
# ==========================================

unsafe_features = [
    "lagging_average",
    "previous_difference",
    "month_cyclic"
]

df = df.drop(
    columns=unsafe_features,
    errors="ignore"
)

# ==========================================
# 6. SAVE CLEANED DATASET
# ==========================================

df.to_csv(
    OUTPUT_DIR / "cleaned_unique_dates.csv",
    index=False
)

print("\nCleaned dataset saved!")

# ==========================================
# 7. CREATE DAILY HISTORICAL DATA
# ==========================================

daily_index = pd.date_range(
    start=df["DATE"].min(),
    end=df["DATE"].max(),
    freq="D"
)

daily_demand = (
    df.set_index("DATE")["peak_demand"]
    .reindex(daily_index)
)

# ==========================================
# 8. CREATE SAFE LAG FEATURES
# ==========================================

# Demand from exactly one calendar day earlier
df["lag_1_day"] = (
    df["DATE"].map(daily_demand.shift(1))
)

# Demand from exactly seven calendar days earlier
df["lag_7_days"] = (
    df["DATE"].map(daily_demand.shift(7))
)

# Average demand during the previous seven
# calendar days, excluding the current date
rolling_average = (
    daily_demand.shift(1)
    .rolling(window=7, min_periods=7)
    .mean()
)

df["rolling_mean_7_days"] = (
    df["DATE"].map(rolling_average)
)

# ==========================================
# 9. CREATE MODELING DATASET
# ==========================================

required_lags = [
    "lag_1_day",
    "lag_7_days",
    "rolling_mean_7_days"
]

model_df = df.dropna(
    subset=required_lags
).copy()

# ==========================================
# 10. SAVE FINAL DATASETS
# ==========================================

model_df.to_csv(
    OUTPUT_DIR / "modeling_dataset.csv",
    index=False
)

# ==========================================
# 11. FINAL REPORT
# ==========================================

print("\n========== PREPROCESSING REPORT ==========")

print("Original records:", 1957)

print("Records after duplicate exclusion:", len(df))

print("Modeling records:", len(model_df))

print(
    "Records removed due to unavailable "
    "7-day history:",
    len(df) - len(model_df)
)

print("\nMissing values in modeling dataset:")

print(model_df.isnull().sum())

print("\nFinal dataset shape:", model_df.shape)

print("\nAvailable features:")

print(model_df.columns.tolist())

print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")
