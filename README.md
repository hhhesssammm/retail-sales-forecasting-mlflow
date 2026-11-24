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
```

## Run the pipeline

```powershell
.\.venv\Scripts\python.exe -m src.eda
.\.venv\Scripts\python.exe -m src.train
```

EDA plots and summaries are saved in `reports/eda/`. Each training execution creates a new batch folder in `reports/forecasts/`, containing validation forecasts, model comparison, selected model's test forecasts and a JSON summary. Nothing is overwritten except the latest-batch pointer.

Open the notebooks with:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

- `01_eda.ipynb`: data inspection, plots and features.
- `02_experiments.ipynb`: inspect the results of the latest completed training batch.

## MLflow UI

From the project root, run:

```powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Select the `Retail_Sales_Forecasting` experiment, filter by `batch_id` when multiple batches exist, and compare runs by RMSE (ascending), MAE, WAPE and R². The validation metrics use unprefixed names; final holdout metrics use the `test_` prefix. Each run includes parameters, forecast CSV/PNG, model, feature list and date boundaries.

## Evaluation design

- Forecast one week ahead for each store. At prediction time, the previous week's actual sales are available. This is rolling one-step evaluation, not a 13-week forecast made in advance.
- After 52 warm-up weeks for lag features, use the first eligible weeks for training, the next 13 weeks for validation and the last 13 weeks for testing. Never randomly shuffle dates.
- Calendar and shifted sales history are used. Current-week sales never enter the features. Holiday status must be known in advance.
- Ten runs: naive (last week and last year), two linear regressions, two random forests, two XGBoost regressors and two per-store SARIMAX models.
- The SARIMAX models are the AR(1) and AR(2) special cases, with no MA or seasonal component. Seasonal effects are represented in other models by the annual lag, holidays and calendar features.
- Choose the lowest validation RMSE among converged candidates. Do not use test metrics to choose models. The exact fitted winner is reloaded and evaluated on the held-out weeks without refitting.
- MAE and RMSE are in USD; WAPE is expressed as a percentage; R² is dimensionless.

## Registry and inference

Training registers the validated winner as a new version of `RetailSalesForecaster`, moves it to Staging and sets the `candidate` alias. MLflow 2.22 is pinned because the course asks for legacy stages; aliases are the recommended newer mechanism. A new Staging version archives previous Staging versions but does not change Production.

Inspect the validation and test results before promoting a candidate. For example, to promote version 1 (replace with the version from summary.json):

```powershell
.\.venv\Scripts\python.exe -m src.promote --version 1
```

This sets Production and the `champion` alias and archives older Production versions. To produce next-week forecasts, supply all historical weeks and the known holiday flag:

```powershell
.\.venv\Scripts\python.exe -m src.predict --alias candidate --holiday no
```

Change `--holiday` to `yes` if the upcoming week is a holiday. Use `--alias champion` after promotion. The result is saved as `reports/forecasts/next_week.csv`. This is local deployment preparation, not a hosted service.

## Submission

The local `mlflow.db` contains experiment and registry metadata, and `mlruns/` holds the artifacts. Keep both for the course submission, together with executed notebooks and the chosen batch's output folder. These generated files are excluded from Git by default because of their size and local absolute artifact paths. Use the original machine/location to demonstrate the UI, or rerun the pipeline after moving the project.

To run training and generate the final report in one command:

```powershell
.\.venv\Scripts\python.exe -m src.pipeline
```

To regenerate the report without retraining (for example after promoting a model):

```powershell
.\.venv\Scripts\python.exe -m src.report
```

The self-contained Persian report is written to `reports/forecasts/<batch_id>/final_report.html`. Open in Chrome or Edge and print to PDF on A4, at 100% scale with browser headers and footers disabled. It contains four sections with actual EDA, validation results, test results and registry information; it refuses a mismatched dataset. Review the printed pages and add your own interpretation.

See [submission checklist](docs/submission.md) for the complete delivery and demonstration steps. The final report must use actual experiment results; results are not committed until training has been run on the real dataset.
