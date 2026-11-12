
import pandas as pd

# Load dataset
df = pd.read_csv("data/electricity_weather.csv")

# Basic information
print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns.tolist())

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print(df.duplicated().sum())

# Check duplicate dates
df["DATE"] = pd.to_datetime(df["DATE"])

print("\n===== DUPLICATE DATES =====")
print(df["DATE"].duplicated().sum())

print("\n===== DATE RANGE =====")
print(df["DATE"].min(), "to", df["DATE"].max())

print("\n===== TARGET VARIABLE =====")
print(df["peak_demand"].describe())

print("\n===== DATASET INFORMATION =====")
df.info()
