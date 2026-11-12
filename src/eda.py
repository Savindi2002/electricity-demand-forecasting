
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------
# 1. PROJECT PATHS
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "electricity_weather.csv"
REPORT_DIR = BASE_DIR / "reports" / "eda"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# 2. LOAD DATA
# ---------------------------------

df = pd.read_csv(DATA_PATH)

df["DATE"] = pd.to_datetime(df["DATE"])

# Sort chronologically
df = df.sort_values("DATE").reset_index(drop=True)

print("\n===== DATASET OVERVIEW =====")
print("Shape:", df.shape)
print(df.head())

# ---------------------------------
# 3. DUPLICATE DATE ANALYSIS
# ---------------------------------

print("\n===== DUPLICATE DATE ANALYSIS =====")

duplicates = df[df["DATE"].duplicated(keep=False)]

print(duplicates[["DATE", "peak_demand"]])

duplicates.to_csv(
    REPORT_DIR / "duplicate_dates.csv",
    index=False
)

# ---------------------------------
# 4. CHECK MISSING DATES
# ---------------------------------

print("\n===== MISSING DATE ANALYSIS =====")

expected_dates = pd.date_range(
    start=df["DATE"].min(),
    end=df["DATE"].max(),
    freq="D"
)

existing_dates = pd.DatetimeIndex(df["DATE"].unique())

missing_dates = expected_dates.difference(existing_dates)

print("Expected dates:", len(expected_dates))
print("Available unique dates:", len(existing_dates))
print("Missing dates:", len(missing_dates))

pd.DataFrame({
    "missing_date": missing_dates
}).to_csv(REPORT_DIR / "missing_dates.csv", index=False)

# ---------------------------------
# 5. DESCRIPTIVE STATISTICS
# ---------------------------------

print("\n===== DESCRIPTIVE STATISTICS =====")

print(df.describe().round(2))

df.describe().to_csv(
    REPORT_DIR / "descriptive_statistics.csv"
)

# ---------------------------------
# 6. PEAK DEMAND DISTRIBUTION
# ---------------------------------

plt.figure(figsize=(10, 5))

sns.histplot(
    df["peak_demand"],
    kde=True,
    bins=30
)

plt.title("Distribution of Peak Electricity Demand")
plt.xlabel("Peak Demand")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(REPORT_DIR / "demand_distribution.png")
plt.show()

# ---------------------------------
# 7. ELECTRICITY DEMAND OVER TIME
# ---------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["DATE"],
    df["peak_demand"]
)

plt.title("Daily Peak Electricity Demand (2018-2023)")
plt.xlabel("Date")
plt.ylabel("Peak Demand")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(REPORT_DIR / "demand_over_time.png")
plt.show()

# ---------------------------------
# 8. MONTHLY DEMAND ANALYSIS
# ---------------------------------

df["month"] = df["DATE"].dt.month

monthly_demand = df.groupby("month")["peak_demand"].mean()

plt.figure(figsize=(10, 5))

sns.barplot(
    x=monthly_demand.index,
    y=monthly_demand.values
)

plt.title("Average Electricity Demand by Month")
plt.xlabel("Month")
plt.ylabel("Average Peak Demand")

plt.tight_layout()
plt.savefig(REPORT_DIR / "monthly_demand.png")
plt.show()

# ---------------------------------
# 9. WEEKDAY VS WEEKEND
# ---------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="is_weekend",
    y="peak_demand"
)

plt.title("Electricity Demand: Weekdays vs Weekends")
plt.xlabel("Weekend")
plt.ylabel("Peak Demand")

plt.tight_layout()
plt.savefig(REPORT_DIR / "weekday_weekend.png")
plt.show()

# ---------------------------------
# 10. WEATHER IMPACT ANALYSIS
# ---------------------------------

weather_features = [
    "feelslike",
    "precip",
    "windspeed",
    "relative_humidity",
    "solarenergy",
    "windchill"
]

for feature in weather_features:

    plt.figure(figsize=(8, 5))

    sns.regplot(
        data=df,
        x=feature,
        y="peak_demand",
        scatter_kws={"alpha": 0.4}
    )

    plt.title(f"{feature} vs Peak Electricity Demand")
    plt.tight_layout()

    plt.savefig(
        REPORT_DIR / f"{feature}_vs_demand.png"
    )

    plt.show()

# ---------------------------------
# 11. CORRELATION ANALYSIS
# ---------------------------------

numeric_df = df.select_dtypes(include=np.number)

correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(REPORT_DIR / "correlation_heatmap.png")
plt.show()

print("\n===== CORRELATION WITH PEAK DEMAND =====")

print(
    correlation["peak_demand"]
    .sort_values(ascending=False)
)

print("\nEDA COMPLETED SUCCESSFULLY!")

print("Reports saved in:", REPORT_DIR)
