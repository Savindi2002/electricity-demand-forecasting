
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTION_PATH = (
    BASE_DIR / "reports" / "models"
    / "best_model_predictions.csv"
)

COMPARISON_PATH = (
    BASE_DIR / "reports" / "models"
    / "model_comparison.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions and model comparison
df = pd.read_csv(PREDICTION_PATH)
comparison = pd.read_csv(COMPARISON_PATH)

df["DATE"] = pd.to_datetime(df["DATE"])

actual = df["Actual"]
predicted = df["Predicted"]

# Calculate evaluation metrics
mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
r2 = r2_score(actual, predicted)

print("\nBEST MODEL EVALUATION")
print("---------------------")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

# 1. Actual vs Predicted
plt.figure(figsize=(14, 6))
plt.plot(df["DATE"], actual, label="Actual Demand")
plt.plot(df["DATE"], predicted, label="Predicted Demand")
plt.title("Actual vs Predicted Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Peak Demand")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_vs_predicted.png", dpi=300)
plt.close()

# 2. Actual vs Predicted Scatter Plot
plt.figure(figsize=(7, 7))
sns.scatterplot(x=actual, y=predicted)
minimum = min(actual.min(), predicted.min())
maximum = max(actual.max(), predicted.max())
plt.plot([minimum, maximum], [minimum, maximum], "--")
plt.title("Actual vs Predicted Demand")
plt.xlabel("Actual Demand")
plt.ylabel("Predicted Demand")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "actual_predicted_scatter.png", dpi=300)
plt.close()

# 3. Prediction Errors
errors = actual - predicted

plt.figure(figsize=(10, 5))
sns.histplot(errors, kde=True)
plt.axvline(0, linestyle="--")
plt.title("Prediction Error Distribution")
plt.xlabel("Actual - Predicted")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "error_distribution.png", dpi=300)
plt.close()

# 4. Model Comparison
comparison = comparison.sort_values("RMSE")

plt.figure(figsize=(10, 6))
sns.barplot(
    data=comparison,
    x="RMSE",
    y="Model"
)
plt.title("Model Performance Comparison (RMSE)")
plt.xlabel("RMSE - Lower is Better")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=300)
plt.close()

# 5. Residuals over time
plt.figure(figsize=(14, 5))
plt.scatter(df["DATE"], errors, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.title("Prediction Errors Over Time")
plt.xlabel("Date")
plt.ylabel("Residual")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "residuals_over_time.png", dpi=300)
plt.close()

print("\nEvaluation graphs saved in:", OUTPUT_DIR)
