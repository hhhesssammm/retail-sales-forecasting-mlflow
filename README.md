# Retail Sales Forecasting with MLflow

Course project: predict next week's total sales for each Walmart store and compare regression, tree-based and time-series models with MLflow.

## Data

Download the labeled `train.csv` from [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data) and save it as `data/raw/train.csv`. The original rows are department-level weekly sales. We sum departments by store and week. No competition test data is used because it has no sales labels. Raw data is not redistributed here.

## Setup (Python 3.11)

From the project root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m jupyter lab
```

Open `notebooks/01_eda.ipynb` to inspect the data. EDA plots and summaries are saved in `reports/eda/`.

## Evaluation design

- Forecast one week ahead for each store. At prediction time, the previous week's actual sales are available.
- Use the first weeks for training, the next 13 weeks for validation and the last 13 weeks for testing. Never randomly shuffle dates.
- Use calendar variables and shifted sales history only; current-week sales must not enter the features.
- Compare naive, linear regression, random forest, XGBoost and SARIMAX on identical store/week pairs.
- Select using validation RMSE; use the held-out test only for the selected model's final assessment.
- Track parameters, metrics, plots and fitted models in MLflow. Use SQLite for registry metadata and mlruns for artifacts. Model aliases and legacy stages will be documented in the training phase.

## Project status

Data preparation, exploratory analysis and feature engineering are implemented. Training, registry, actual experiment outputs and final report are pending. Run the notebook on the downloaded data before interpreting results.
