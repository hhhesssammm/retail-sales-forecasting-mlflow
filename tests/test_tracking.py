import mlflow
import pandas as pd
import numpy as np

from src.train import run_experiment
from src.tracking import REGISTERED_MODEL, configure_tracking, promote


def test_tracking_registry_round_trip(tmp_path):
    previous_tracking = mlflow.get_tracking_uri()
    previous_registry = mlflow.get_registry_uri()
    try:
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-03", periods=90, freq="W-FRI")
        raw = pd.DataFrame([
            {"Store": store, "Dept": 1, "Date": date, "IsHoliday": False,
             "Weekly_Sales": 1000 * store + i + rng.normal(0, 10)}
            for store in [1, 2] for i, date in enumerate(dates)
        ])
        path = tmp_path / "train.csv"
        raw.to_csv(path, index=False)
        summary = run_experiment(path, root=tmp_path, candidates=[
            ("naive", "Naive", {"lag": 1}),
            ("linear", "LinearRegression", {"fit_intercept": True}),
        ])
        client = configure_tracking(tmp_path)
        version = client.get_model_version_by_alias(REGISTERED_MODEL, "candidate")
        assert version.version == summary["version"]
        assert version.current_stage == "Staging"
        run = client.get_run(version.run_id)
        assert "rmse" in run.data.metrics
        assert "test_rmse" in run.data.metrics
        promote(client, version.version)
        champion = client.get_model_version_by_alias(REGISTERED_MODEL, "champion")
        assert champion.current_stage == "Production"
        assert (tmp_path / "reports" / "forecasts" / summary["batch_id"] / "comparison.csv").exists()
    finally:
        mlflow.set_tracking_uri(previous_tracking)
        mlflow.set_registry_uri(previous_registry)
