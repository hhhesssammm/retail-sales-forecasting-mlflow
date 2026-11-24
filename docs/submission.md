# Submission checklist

## Before training

- Download the original Walmart competition train.csv and place it in data/raw/.
- Use Python 3.11 and install requirements.txt in a virtual environment.
- Run python -m pytest -q. Resolve failures before interpreting model results.
- Run python -m src.eda and execute notebooks/01_eda.ipynb.

## Training and review

- Run python -m src.pipeline to train all ten candidates, register the winner, and generate the Persian HTML report.
- Open the MLflow UI with mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000.
- In Retail_Sales_Forecasting, filter by the latest batch_id, sort RMSE ascending, compare runs and inspect validation artifacts.
- Execute notebooks/02_experiments.ipynb and save its outputs.
- Verify that the candidate is registered with a version and Staging stage. Review errors before deciding whether to promote it.
- Optionally run python -m src.promote --version N (replace N with the reviewed version), then python -m src.report to refresh the stage shown in the report.
- Run python -m src.predict --alias candidate --holiday no (or yes according to the actual upcoming week). After promotion, use the champion alias.

## Deliverables

Include the source code, requirements, README, executed notebooks, the completed batch under reports/forecasts/, reports/eda/, the local mlflow.db and mlruns/ directory. Check the competition data terms before redistributing the original CSV. The database may contain absolute artifact paths, so demonstrate from the original project location or rerun after moving.

Open final_report.html from the completed batch in Chrome or Edge, print to PDF with A4 paper, 100% scale, browser headers and footers off. The four sections are designed to start on separate printed pages. Review the page breaks and all charts before submitting. Add your course details and your own observations if required. Screenshots of the real MLflow comparison and model registry are useful supporting evidence.

Do not publish model artifacts or raw data to a public repository without checking licensing and privacy. Do not submit synthetic test metrics as real results.

## Questions to be able to answer

- Why aggregate department data into store-week sales?
- Why use chronological rather than random splitting?
- Why shift before calculating a moving average?
- Why compare to a naive baseline?
- What does each evaluation metric measure?
- Why select on validation and not test?
- How do experiment, run, artifact, model version, stage and alias differ?
- What must be known before making a next-week forecast?
