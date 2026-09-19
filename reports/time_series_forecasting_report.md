# Time-Series Forecasting with ARIMA

## Dataset

Official starter dataset:

`data/raw/daily-demand-series.csv`

- Observations: 30
- Date column: `date`
- Demand column: `demand`
- Start date: 2026-06-01
- End date: 2026-06-30

## Time-Series Decomposition

The daily demand series was decomposed into:

- Observed demand
- Trend
- Weekly seasonality
- Residual

A seasonal period of 7 days was used to represent weekly demand patterns.

## Stationarity Testing

### Original Series

ADF Statistic: -0.9039

p-value: 0.786695

### First Differenced Series

ADF Statistic: -2.2336

p-value: 0.194269

## Train / Test Design

The dataset contains only 30 daily observations.

Therefore:

- Training observations: 15
- Holdout observations: 15

A 15-day holdout period was used because the dataset contains only 30 observations.

## ARIMA Model Selection

Candidate ARIMA configurations were compared using AIC.

Selected model:

**ARIMA(2, 1, 0)**

## Holdout Evaluation

- MAPE: **6.876%**
- RMSE: **14.825**

## 30-Day Future Forecast

The selected ARIMA model was refitted using the complete available historical series.

A 30-day future forecast was generated.

Forecast file:

`reports/30_day_forecast.csv`

## Generated Evidence

- `reports/historical_demand.png`
- `reports/time_series_decomposition.png`
- `reports/holdout_forecast.png`
- `reports/30_day_forecast.png`
- `reports/30_day_forecast.csv`

## Reproducibility

The complete analysis is implemented in:

`notebooks/time_series_forecasting_arima.ipynb`

The official starter dataset is stored in:

`data/raw/daily-demand-series.csv`
