
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "electricity_weather.csv"
REPORT_DIR = BASE_DIR / "reports" / "quality"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["DATE"] = pd.to_datetime(df["DATE"])

# Sort data
df = df.sort_values("DATE").reset_index(drop=True)

# 1. Save all duplicate-date records
duplicates = df[
    df["DATE"].duplicated(keep=False)
].copy()

duplicates.to_csv(
    REPORT_DIR / "duplicate_date_details.csv",
    index=False
)

print("\n===== DUPLICATE DATE DETAILS =====")
print(duplicates.to_string(index=False))

# 2. Investigate missing dates
all_dates = pd.date_range(
    df["DATE"].min(),
    df["DATE"].max(),
    freq="D"
)

available_dates = pd.DatetimeIndex(
    df["DATE"].unique()
)

missing_dates = all_dates.difference(available_dates)

print("\nTotal missing dates:", len(missing_dates))

# Group consecutive missing dates
missing_series = pd.Series(missing_dates)

if not missing_series.empty:
    groups = (
        missing_series.diff().dt.days.ne(1)
    ).cumsum()

    missing_groups = (
        missing_series.groupby(groups)
        .agg(["min", "max", "count"])
    )

    print("\n===== MISSING DATE PERIODS =====")
    print(missing_groups)

    missing_groups.to_csv(
        REPORT_DIR / "missing_date_periods.csv"
    )

# 3. Check previous_difference
# This checks whether it uses the current target.

df["calculated_difference"] = (
    df["peak_demand"].diff()
)

df["difference_match"] = np.isclose(
    df["previous_difference"],
    df["calculated_difference"],
    equal_nan=False
)

print("\n===== PREVIOUS DIFFERENCE CHECK =====")
print(
    "Matching rows:",
    df["difference_match"].sum()
)

print(
    "Total rows:",
    len(df)
)

# 4. Investigate lagging_average
df["calculated_lag_average"] = (
    df["peak_demand"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

print("\n===== LAGGING AVERAGE CHECK =====")

print(
    "Matching rows:",
    np.isclose(
        df["lagging_average"],
        df["calculated_lag_average"],
        equal_nan=False
    ).sum()
)

# 5. Save investigation report
df.to_csv(
    REPORT_DIR / "feature_investigation.csv",
    index=False
)

print("\nInvestigation completed!")
print("Reports saved in:", REPORT_DIR)
