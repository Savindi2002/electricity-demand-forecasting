# ⚡ Electricity Demand Analysis & Forecasting

## 📌 Project Overview

This project analyzes electricity peak demand and its relationship with weather and time-based factors. Exploratory Data Analysis, correlation analysis, regression techniques, and time-series forecasting methods were applied to identify demand patterns and predict future electricity consumption.

The project uses historical electricity demand and weather data containing **1,957 observations across 19 variables**.

## 🎯 Objectives

* Analyze historical electricity peak-demand patterns
* Investigate the relationship between electricity demand and weather conditions
* Identify trends, seasonal patterns, and correlations
* Apply statistical and regression techniques
* Develop ARIMA and SARIMA time-series models
* Forecast future electricity demand
* Compare forecasting models using AIC and RMSE

## 📊 Dataset

The dataset includes electricity demand and weather-related variables such as:

* Peak Demand
* Date
* Day of Week
* Year
* Weekend Indicator
* Feels-like Temperature
* Precipitation
* Wind Speed
* Wind Direction
* Sea Level Pressure
* Visibility
* Solar Energy
* UV Index
* Moon Phase
* Lagging Average
* Previous Difference
* Relative Humidity
* Cyclic Month
* Wind Chill

The dataset contained **no missing values**, and IQR-based analysis identified **no outliers in peak demand**.

## 🔍 Exploratory Data Analysis

The analysis included:

* Dataset structure and descriptive statistics
* Missing-value analysis
* Correlation heatmap
* Pair plots
* Peak-demand time-series visualization
* Distribution analysis
* Boxplot and IQR-based outlier detection
* Monthly and yearly demand analysis
* Moving-average trend analysis

## 📈 Statistical Analysis

### Pearson Correlation

The Pearson correlation between feels-like temperature and peak electricity demand was approximately:

**0.706**

This indicates a strong positive linear relationship between temperature and peak demand.

### Spearman Correlation

The Spearman correlation was approximately:

**0.699**

The similar Pearson and Spearman values indicate a strong increasing relationship between the two variables.

## 🤖 Regression Models

Several regression approaches were explored:

### Simple Linear Regression

Predictor:

* Feels-like Temperature

**R² = 0.499**

### Multiple Linear Regression

Predictors:

* Feels-like Temperature
* Relative Humidity
* Wind Speed

**R² = 0.533**

### Polynomial Regression

A degree-2 polynomial model was used to capture nonlinear relationships between temperature and electricity demand.

**R² = 0.723**

These models demonstrate that weather variables can provide useful information for explaining variations in electricity demand.

## ⏱️ Time-Series Analysis

Time-series analysis was performed to investigate:

* Long-term trends
* Moving averages
* Monthly seasonal variation
* Yearly variation
* Stationarity
* Autocorrelation
* Partial autocorrelation

The Augmented Dickey-Fuller test produced a p-value of approximately **0.004**, providing evidence that the analyzed series was stationary under the test.

## 🔮 Forecasting Models

### ARIMA

An **ARIMA(1,0,1)** model was developed for electricity demand forecasting.

### SARIMA

A **SARIMA(1,0,1)(1,0,1,7)** model was also developed to capture weekly seasonal patterns.

A 30-day future forecasting procedure was implemented using the SARIMA model, including forecast confidence intervals.

## 📊 Model Comparison

| Model  |    RMSE |      AIC |
| ------ | ------: | -------: |
| ARIMA  | 1125.29 | 22190.18 |
| SARIMA | 2255.73 | 22033.38 |

The evaluation shows that **ARIMA achieved a lower RMSE**, while the SARIMA model produced a lower AIC in the reported comparison.

## 🛠️ Technologies & Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* Statsmodels
* Jupyter Notebook

## 🧠 Skills Demonstrated

* Data Cleaning
* Exploratory Data Analysis
* Statistical Analysis
* Correlation Analysis
* Data Visualization
* Feature Analysis
* Linear Regression
* Polynomial Regression
* Time-Series Analysis
* Stationarity Testing
* ARIMA
* SARIMA
* Model Evaluation
* Forecasting

## 📁 Project Structure

```text
Electricity-Demand-Forecasting/
│
├── electricity_demand_analysis.ipynb
├── electricity_weather.csv
├── README.md
└── images/
    ├── correlation_heatmap.png
    ├── demand_trend.png
    ├── regression.png
    └── forecast.png
```

## 🚀 Future Improvements

* Compare additional forecasting models such as Prophet, XGBoost, and LSTM
* Perform systematic hyperparameter optimization
* Improve handling of the time-series frequency/index
* Include additional economic and energy-related variables
* Build an interactive Power BI or Streamlit dashboard
* Deploy the forecasting model as a web application

## 👩‍💻 Author

**Savindi Hewage**

Data Science Undergraduate
Sabaragamuwa University of Sri Lanka

---

⭐ If you find this project useful, feel free to explore the repository and connect with me!
