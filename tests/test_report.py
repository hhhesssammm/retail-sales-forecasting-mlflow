import pandas as pd

from src.report import build_report


def test_report_uses_actual_batch_values(tmp_path):
    for name in ["eda/sales_vs_time.png", "eda/monthly_seasonality.png",
                 "comparison.png", "selected_test/test_forecast.png"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test image")

    weekly = pd.DataFrame({
        "Date": pd.to_datetime(["2020-01-03", "2020-01-10"]),
        "Weekly_Sales": [100, 200], "IsHoliday": [False, True],
    })
    best = dict(run_name="linear", run_id="test-run", rmse=123.45, mae=100,
                wape=10, r2=0.8)
    summary = dict(
        batch_id="demo", dataset_sha256="test-hash", best_run=best,
        registered_model="RetailSalesForecaster", version="2",
        test_metrics=dict(rmse=150, mae=120, wape=12, r2=0.7),
        splits={"train": dict(start="2020-01-03", end="2020-01-10", rows=2, weeks=2)},
    )
    comparison = pd.DataFrame([
        dict(best, nonconverged_stores=0),
        dict(best, run_name="naive_last_week", rmse=200, nonconverged_stores=0),
    ])
    eda = dict(stores=1, store_weeks=2, first_date="2020-01-03",
               last_date="2020-01-10", eda_training_end="2020-01-10",
               missing_values={"Weekly_Sales": 0}, training_iqr_outliers=0)
    report = build_report(summary, comparison, eda, weekly, tmp_path, "Staging")
    assert report.count("<section>") == 4
    assert report.count("data:image/png;base64,") == 4
    assert "123.45" in report
    assert "test-run" in report
    assert "Staging" in report
    assert "test-hash" in report
