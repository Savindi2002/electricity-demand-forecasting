
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"
PREDICTION_PATH = (
    BASE_DIR / "reports" / "models"
    / "best_model_predictions.csv"
)
COMPARISON_PATH = (
    BASE_DIR / "reports" / "models"
    / "model_comparison.csv"
)

st.set_page_config(
    page_title="Electricity Demand Prediction",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Demand Prediction Dashboard")
st.caption("Electricity Demand Prediction Using Weather Data")

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTION_PATH)
    comparison = pd.read_csv(COMPARISON_PATH)

    data["DATE"] = pd.to_datetime(data["DATE"])
    predictions["DATE"] = pd.to_datetime(predictions["DATE"])

    return data, predictions, comparison

try:
    data, predictions, comparison = load_data()

    # Sidebar
    st.sidebar.header("Dashboard Filters")

    years = sorted(data["DATE"].dt.year.unique())

    selected_year = st.sidebar.selectbox(
        "Select Year",
        years,
        index=len(years) - 1
    )

    filtered_data = data[
        data["DATE"].dt.year == selected_year
    ]

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Records",
        f"{len(filtered_data):,}"
    )

    col2.metric(
        "Average Demand",
        f"{filtered_data['peak_demand'].mean():,.0f}"
    )

    col3.metric(
        "Maximum Demand",
        f"{filtered_data['peak_demand'].max():,.0f}"
    )

    best_model = comparison[
        comparison["Model"] != "Persistence Baseline"
    ].sort_values("RMSE").iloc[0]

    col4.metric(
        "Best Model",
        best_model["Model"]
    )

    st.divider()

    # Historical demand
    st.subheader("📈 Historical Electricity Demand")

    demand_chart = px.line(
        filtered_data,
        x="DATE",
        y="peak_demand",
        title=f"Electricity Demand in {selected_year}"
    )

    st.plotly_chart(demand_chart, use_container_width=True)

    # Actual vs predicted
    st.subheader("🎯 Actual vs Predicted Demand")

    prediction_chart = px.line(
        predictions,
        x="DATE",
        y=["Actual", "Predicted"],
        title="Model Prediction Performance"
    )

    st.plotly_chart(
        prediction_chart,
        use_container_width=True
    )

    # Model comparison
    st.subheader("🤖 Machine Learning Model Comparison")

    model_chart = px.bar(
        comparison,
        x="Model",
        y="RMSE",
        color="Model",
        title="RMSE Comparison"
    )

    st.plotly_chart(
        model_chart,
        use_container_width=True
    )

    # Weather analysis
    st.subheader("🌦️ Weather and Electricity Demand")

    weather_feature = st.selectbox(
        "Select Weather Feature",
        [
            "feelslike",
            "precip",
            "windspeed",
            "visibility",
            "solarenergy",
            "uvindex",
            "relative_humidity"
        ]
    )

    weather_chart = px.scatter(
        filtered_data,
        x=weather_feature,
        y="peak_demand",
        trendline="ols",
        title=f"{weather_feature} vs Electricity Demand"
    )

    st.plotly_chart(
        weather_chart,
        use_container_width=True
    )

    # Data table
    st.subheader("📋 Historical Data")

    st.dataframe(
        filtered_data,
        use_container_width=True
    )

except FileNotFoundError as error:
    st.error(f"Required project file not found: {error}")

except Exception as error:
    st.error(f"Dashboard error: {error}")
