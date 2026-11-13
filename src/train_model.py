
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.dummy import DummyRegressor

from xgboost import XGBRegressor

# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================

df = pd.read_csv(DATA_PATH)

df["DATE"] = pd.to_datetime(df["DATE"])

df = df.sort_values("DATE").reset_index(drop=True)

print("\nDATASET SHAPE:", df.shape)

# ==========================================
# 3. SELECT FEATURES
# ==========================================

target = "peak_demand"

excluded_columns = [
    "DATE",
    target,
    "previous_difference",
    "lagging_average",
    "month_cyclic"
]

features = [
    column for column in df.select_dtypes(
        include=["number", "bool"]
    ).columns
    if column not in excluded_columns
]

X = df[features].copy()
y = df[target].copy()

# ==========================================
# 4. CHRONOLOGICAL TRAIN TEST SPLIT
# ==========================================

split_index = int(len(df) * 0.80)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\nTRAINING RECORDS:", len(X_train))
print("TESTING RECORDS:", len(X_test))

print(
    "\nTraining period:",
    df["DATE"].iloc[0],
    "to",
    df["DATE"].iloc[split_index - 1]
)

print(
    "Testing period:",
    df["DATE"].iloc[split_index],
    "to",
    df["DATE"].iloc[-1]
)

# ==========================================
# 5. DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )
}

# ==========================================
# 6. EVALUATION FUNCTION
# ==========================================

def evaluate_model(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

# ==========================================
# 7. BASELINE MODEL
# ==========================================

results = []

baseline_prediction = X_test["lag_1_day"]

results.append(
    evaluate_model(
        "Persistence Baseline",
        y_test,
        baseline_prediction
    )
)

# ==========================================
# 8. TRAIN AND TEST MODELS
# ==========================================

trained_models = {}
predictions = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predicted = model.predict(X_test)

    trained_models[name] = model
    predictions[name] = predicted

    result = evaluate_model(
        name,
        y_test,
        predicted
    )

    results.append(result)

    print("MAE:", round(result["MAE"], 3))
    print("RMSE:", round(result["RMSE"], 3))
    print("R2:", round(result["R2"], 4))

# ==========================================
# 9. COMPARE MODEL PERFORMANCE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="RMSE",
    ascending=True
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

# ==========================================
# 10. SELECT BEST ML MODEL
# ==========================================

best_model_name = min(
    predictions,
    key=lambda name: next(
        result["RMSE"]
        for result in results
        if result["Model"] == name
    )
)

best_model = trained_models[best_model_name]

print("\nBEST ML MODEL:", best_model_name)

# ==========================================
# 11. SAVE BEST MODEL
# ==========================================

joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

# Save feature names
joblib.dump(
    features,
    MODEL_DIR / "feature_names.pkl"
)

# ==========================================
# 12. SAVE TEST PREDICTIONS
# ==========================================

best_predictions = predictions[best_model_name]

prediction_results = pd.DataFrame({
    "DATE": df["DATE"].iloc[split_index:].values,
    "Actual_Demand": y_test.values,
    "Predicted_Demand": best_predictions,
    "Error": y_test.values - best_predictions
})

prediction_results.to_csv(
    REPORT_DIR / "best_model_predictions.csv",
    index=False
)

print("\nMODEL SAVED SUCCESSFULLY!")

print("Model:", MODEL_DIR / "best_model.pkl")

print("\nTRAINING COMPLETED!")
