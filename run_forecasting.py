import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

ROOT = Path.cwd()
DATA = ROOT / "data" / "raw" / "daily-demand-series.csv"
NB = ROOT / "notebooks" / "time_series_forecasting_arima.ipynb"
REPORT = ROOT / "reports" / "time_series_forecasting_report.md"

NB.parent.mkdir(parents=True, exist_ok=True)
REPORT.parent.mkdir(parents=True, exist_ok=True)

# ==========================================
# LOAD DATA
# ==========================================
df = pd.read_csv(DATA)

print("Dataset shape:", df.shape)
print("Columns:", list(df.columns))

date_col = "date"
demand_col = "demand"

ts = df[[date_col, demand_col]].copy()

ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
ts[demand_col] = pd.to_numeric(ts[demand_col], errors="coerce")

ts = ts.dropna()
ts = ts.sort_values(date_col)
ts = ts.drop_duplicates(subset=[date_col])
ts = ts.set_index(date_col)[demand_col]

ts = ts.asfreq("D")
ts = ts.interpolate().ffill().bfill()

print("Final time-series length:", len(ts))
print("Date range:", ts.index.min(), "to", ts.index.max())

# ==========================================
# FUNCTIONS
# ==========================================
def adf_result(series):
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "stat": result[0],
        "p": result[1],
        "lags": result[2],
        "obs": result[3]
    }

def mape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100

# ==========================================
# HISTORICAL PLOT
# ==========================================
plt.figure(figsize=(14, 6))
plt.plot(ts.index, ts.values)
plt.title("Historical Daily Demand")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.tight_layout()

historical_plot = ROOT / "reports" / "historical_demand.png"
plt.savefig(historical_plot, dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# DECOMPOSITION
# ==========================================
# Dataset has 30 observations.
# Use weekly seasonality with period=7.
decomp = seasonal_decompose(
    ts,
    model="additive",
    period=7
)

fig, axes = plt.subplots(
    4,
    1,
    figsize=(14, 10),
    sharex=True
)

axes[0].plot(ts)
axes[0].set_title("Observed Demand")

axes[1].plot(decomp.trend)
axes[1].set_title("Trend")

axes[2].plot(decomp.seasonal)
axes[2].set_title("Weekly Seasonality")

axes[3].plot(decomp.resid)
axes[3].set_title("Residual")

plt.tight_layout()

decomp_path = ROOT / "reports" / "time_series_decomposition.png"
plt.savefig(decomp_path, dpi=150, bbox_inches="tight")
plt.close()

# ==========================================
# STATIONARITY
# ==========================================
original_adf = adf_result(ts)

diff = ts.diff().dropna()
diff_adf = adf_result(diff)

print("\nOriginal ADF statistic:", original_adf["stat"])
print("Original p-value:", original_adf["p"])

print("\nDifferenced ADF statistic:", diff_adf["stat"])
print("Differenced p-value:", diff_adf["p"])

# ==========================================
# TRAIN / TEST
# ==========================================
# Only 30 observations are available.
# Therefore use 15 days for training and 15 days for holdout.
test_size = 15

train = ts.iloc[:-test_size]
test = ts.iloc[-test_size:]

print("\nTrain observations:", len(train))
print("Test observations:", len(test))

# ==========================================
# ARIMA MODEL SELECTION
# ==========================================
orders = [
    (0,1,0),
    (0,1,1),
    (1,1,0),
    (1,1,1),
    (1,1,2),
    (2,1,0),
    (2,1,1),
    (2,1,2)
]

model_results = []

for order in orders:

    try:

        model = ARIMA(
            train,
            order=order
        )

        fitted = model.fit()

        model_results.append({
            "order": str(order),
            "AIC": fitted.aic
        })

        print(
            "ARIMA",
            order,
            "AIC =",
            round(fitted.aic, 3)
        )

    except Exception as e:

        print(
            "ARIMA",
            order,
            "failed:",
            str(e)
        )

if not model_results:
    raise RuntimeError(
        "No ARIMA model could be fitted."
    )

model_results_df = pd.DataFrame(
    model_results
).sort_values("AIC")

best_order_str = model_results_df.iloc[0]["order"]

best_order = eval(best_order_str)

print("\nSelected ARIMA model:", best_order)

# ==========================================
# HOLDOUT FORECAST
# ==========================================
test_model = ARIMA(
    train,
    order=best_order
)

test_fit = test_model.fit()

test_forecast = test_fit.forecast(
    steps=test_size
)

test_forecast.index = test.index

rmse = np.sqrt(
    mean_squared_error(
        test,
        test_forecast
    )
)

mape_value = mape(
    test.values,
    test_forecast.values
)

print("\nHoldout MAPE:", round(mape_value, 3), "%")
print("Holdout RMSE:", round(rmse, 3))

# ==========================================
# HOLDOUT PLOT
# ==========================================
plt.figure(figsize=(14, 6))

plt.plot(
    train.index,
    train.values,
    label="Training Data"
)

plt.plot(
    test.index,
    test.values,
    label="Actual Test Data"
)

plt.plot(
    test_forecast.index,
    test_forecast.values,
    label="ARIMA Forecast",
    linewidth=2
)

plt.axvline(
    test.index[0],
    linestyle="--",
    label="Test Start"
)

plt.title("ARIMA Holdout Forecast")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.legend()
plt.tight_layout()

holdout_plot = ROOT / "reports" / "holdout_forecast.png"
plt.savefig(
    holdout_plot,
    dpi=150,
    bbox_inches="tight"
)
plt.close()

# ==========================================
# FINAL MODEL
# ==========================================
final_model = ARIMA(
    ts,
    order=best_order
)

final_fit = final_model.fit()

future = final_fit.forecast(
    steps=30
)

future_index = pd.date_range(
    start=ts.index.max()
    + pd.Timedelta(days=1),
    periods=30,
    freq="D"
)

future.index = future_index

forecast_df = pd.DataFrame({
    "date": future.index,
    "forecast_demand": future.values
})

forecast_csv = (
    ROOT
    / "reports"
    / "30_day_forecast.csv"
)

forecast_df.to_csv(
    forecast_csv,
    index=False
)

print("\n30-day future forecast generated.")

print(forecast_df.head())

# ==========================================
# FUTURE FORECAST PLOT
# ==========================================
plt.figure(figsize=(14, 6))

plt.plot(
    ts.index,
    ts.values,
    label="Historical Demand"
)

plt.plot(
    future.index,
    future.values,
    label="30-Day Forecast",
    linewidth=2
)

plt.axvline(
    ts.index.max(),
    linestyle="--",
    label="Forecast Start"
)

plt.title(
    "30-Day Future Demand Forecast using ARIMA"
)

plt.xlabel("Date")
plt.ylabel("Demand")

plt.legend()
plt.tight_layout()

forecast_plot = (
    ROOT
    / "reports"
    / "30_day_forecast.png"
)

plt.savefig(
    forecast_plot,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# ==========================================
# REPORT
# ==========================================
report = f"""# Time-Series Forecasting with ARIMA

## Dataset

Official starter dataset:

`data/raw/daily-demand-series.csv`

- Observations: {len(ts)}
- Date column: `{date_col}`
- Demand column: `{demand_col}`
- Start date: {ts.index.min().date()}
- End date: {ts.index.max().date()}

## Time-Series Decomposition

The daily demand series was decomposed into:

- Observed demand
- Trend
- Weekly seasonality
- Residual

A seasonal period of 7 days was used to represent weekly demand patterns.

## Stationarity Testing

### Original Series

ADF Statistic: {original_adf["stat"]:.4f}

p-value: {original_adf["p"]:.6f}

### First Differenced Series

ADF Statistic: {diff_adf["stat"]:.4f}

p-value: {diff_adf["p"]:.6f}

## Train / Test Design

The dataset contains only 30 daily observations.

Therefore:

- Training observations: {len(train)}
- Holdout observations: {len(test)}

A 15-day holdout period was used because the dataset contains only 30 observations.

## ARIMA Model Selection

Candidate ARIMA configurations were compared using AIC.

Selected model:

**ARIMA{best_order}**

## Holdout Evaluation

- MAPE: **{mape_value:.3f}%**
- RMSE: **{rmse:.3f}**

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
"""

REPORT.write_text(
    report,
    encoding="utf-8"
)

# ==========================================
# NOTEBOOK
# ==========================================
import nbformat as nbf

cells = []

cells.append(
    nbf.v4.new_markdown_cell(
        "# Time-Series Forecasting with ARIMA\n\n"
        "End-to-end reproducible daily demand forecasting."
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from statsmodels.tsa.seasonal import seasonal_decompose\n"
        "from statsmodels.tsa.stattools import adfuller\n"
        "from statsmodels.tsa.arima.model import ARIMA\n"
        "from sklearn.metrics import mean_squared_error\n"
        "import warnings\n"
        "warnings.filterwarnings('ignore')"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "df = pd.read_csv('../data/raw/daily-demand-series.csv')\n"
        "print('Shape:', df.shape)\n"
        "print('Columns:', df.columns.tolist())"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "date_col = 'date'\n"
        "demand_col = 'demand'\n\n"
        "ts = df[[date_col, demand_col]].copy()\n"
        "ts[date_col] = pd.to_datetime(ts[date_col])\n"
        "ts[demand_col] = pd.to_numeric(ts[demand_col])\n"
        "ts = ts.sort_values(date_col).drop_duplicates(date_col)\n"
        "ts = ts.set_index(date_col)[demand_col].asfreq('D')\n"
        "ts = ts.interpolate().ffill().bfill()\n"
        "ts"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "plt.figure(figsize=(14,6))\n"
        "plt.plot(ts)\n"
        "plt.title('Historical Daily Demand')\n"
        "plt.xlabel('Date')\n"
        "plt.ylabel('Demand')\n"
        "plt.show()"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "decomp = seasonal_decompose(ts, model='additive', period=7)\n"
        "fig = decomp.plot()\n"
        "fig.set_size_inches(14,10)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "adf_original = adfuller(ts)\n"
        "print('Original ADF:', adf_original[0])\n"
        "print('Original p-value:', adf_original[1])\n\n"
        "diff = ts.diff().dropna()\n"
        "adf_diff = adfuller(diff)\n"
        "print('Differenced ADF:', adf_diff[0])\n"
        "print('Differenced p-value:', adf_diff[1])"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "test_size = 15\n"
        "train = ts.iloc[:-test_size]\n"
        "test = ts.iloc[-test_size:]\n"
        "print('Train observations:', len(train))\n"
        "print('Test observations:', len(test))"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        f"orders = {orders!r}\n"
        "results = []\n"
        "for order in orders:\n"
        "    try:\n"
        "        fit = ARIMA(train, order=order).fit()\n"
        "        results.append({'order': order, 'AIC': fit.aic})\n"
        "    except Exception:\n"
        "        pass\n"
        "results = pd.DataFrame(results).sort_values('AIC')\n"
        "results"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "best_order = tuple(results.iloc[0]['order'])\n"
        "print('Selected ARIMA:', best_order)"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "fit = ARIMA(train, order=best_order).fit()\n"
        "pred = fit.forecast(steps=15)\n"
        "pred.index = test.index\n"
        "rmse = np.sqrt(mean_squared_error(test, pred))\n"
        "mask = test.values != 0\n"
        "mape_value = np.mean(np.abs((test.values[mask] - pred.values[mask]) / test.values[mask])) * 100\n"
        "print('MAPE:', mape_value)\n"
        "print('RMSE:', rmse)"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "final_fit = ARIMA(ts, order=best_order).fit()\n"
        "future = final_fit.forecast(steps=30)\n"
        "future.index = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=30, freq='D')\n"
        "forecast = pd.DataFrame({'date': future.index, 'forecast_demand': future.values})\n"
        "forecast"
    )
)

cells.append(
    nbf.v4.new_code_cell(
        "plt.figure(figsize=(14,6))\n"
        "plt.plot(ts, label='Historical Demand')\n"
        "plt.plot(future, label='30-Day Forecast', linewidth=2)\n"
        "plt.axvline(ts.index.max(), linestyle='--', label='Forecast Start')\n"
        "plt.title('30-Day ARIMA Demand Forecast')\n"
        "plt.xlabel('Date')\n"
        "plt.ylabel('Demand')\n"
        "plt.legend()\n"
        "plt.show()"
    )
)

cells.append(
    nbf.v4.new_markdown_cell(
        f"## Evaluation Results\n\n"
        f"Selected model: **ARIMA{best_order}**\n\n"
        f"Holdout MAPE: **{mape_value:.3f}%**\n\n"
        f"Holdout RMSE: **{rmse:.3f}**"
    )
)

nb = nbf.v4.new_notebook()
nb["cells"] = cells

nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3"
    }
}

nbf.write(
    nb,
    NB
)

print("\nNotebook created:", NB)

# ==========================================
# EXECUTE NOTEBOOK
# ==========================================
from nbclient import NotebookClient

print("\nExecuting notebook...")

executed_nb = nbf.read(
    NB,
    as_version=4
)

client = NotebookClient(
    executed_nb,
    timeout=600,
    kernel_name="python3"
)

client.execute()

nbf.write(
    executed_nb,
    NB
)

print("Notebook executed successfully.")

# ==========================================
# GIT
# ==========================================
print("\nGit status:")

os.system("git add .")

commit_result = os.system(
    'git commit -m "Add ARIMA time series forecasting analysis"'
)

if commit_result != 0:
    print("Commit may already exist.")

push_result = os.system(
    "git push origin main"
)

if push_result != 0:
    print("Git push failed. Check GitHub authentication.")

print("\n==========================================")
print("TASK COMPLETED")
print("==========================================")
print("Notebook:")
print(NB)

print("\nDataset:")
print(DATA)

print("\nReport:")
print(REPORT)

print("\n30-Day Forecast:")
print(forecast_csv)

print("\nGitHub:")
print("https://github.com/Maddy67bh/reproducible-research")

print("\n==========================================")
