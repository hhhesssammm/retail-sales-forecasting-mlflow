import argparse
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from src.data import DATA_PATH, ROOT, load_sales
from src.evaluation import forecast_metrics, save_comparison, save_forecasts
from src.features import FEATURE_COLUMNS, make_features, split_dates
from src.models import CANDIDATES, SEED, build_model
from src.tracking import REGISTERED_MODEL, set_experiment, stage_candidate


def describe_split(frame):
    return {
        "start": str(frame["Date"].min().date()),
        "end": str(frame["Date"].max().date()),
        "rows": len(frame),
        "weeks": int(frame["Date"].nunique()),
    }


def run_experiment(data_path=DATA_PATH, root=ROOT, candidates=CANDIDATES):
    data_path = Path(data_path)
    frame = make_features(load_sales(data_path))
    train, validation, test = split_dates(frame)
    X_train, y_train = train[FEATURE_COLUMNS], train["Weekly_Sales"]
    X_val, y_val = validation[FEATURE_COLUMNS], validation["Weekly_Sales"]

    client = set_experiment(root)
    batch_id = uuid4().hex[:12]
    output = Path(root) / "reports" / "forecasts" / batch_id
    output.mkdir(parents=True, exist_ok=False)
    splits = {
        name: describe_split(part)
        for name, part in [("train", train), ("validation", validation), ("test", test)]
    }
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    results = []

    for name, family, params in candidates:
        print(f"Training {name}...", flush=True)
        directory = output / name
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags({"batch_id": batch_id, "role": "validation_comparison"})
            mlflow.log_params({
                "model_type": family, "random_seed": SEED, "forecast_horizon_weeks": 1,
                "dataset_sha256": data_hash, "train_rows": len(train),
                "validation_rows": len(validation),
                "train_end": splits["train"]["end"], "validation_end": splits["validation"]["end"],
                **params,
            })
            model = build_model(family, params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            metrics = forecast_metrics(y_val, predictions)
            mlflow.log_metrics(metrics)
            nonconverged = getattr(model, "nonconverged_stores_", [])
            mlflow.log_param("nonconverged_stores", len(nonconverged))
            mlflow.log_dict({"stores": nonconverged}, "convergence.json")
            save_forecasts(validation, predictions, directory, "validation")
            mlflow.log_artifacts(str(directory), artifact_path="validation")
            mlflow.log_dict(splits, "splits.json")
            mlflow.log_dict({"columns": FEATURE_COLUMNS}, "features.json")
            example = X_train.head(3)
            mlflow.sklearn.log_model(
                model, artifact_path="model",
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                code_paths=[str(ROOT / "src")],
            )
            results.append({
                "run_name": name, "model_type": family, "run_id": run.info.run_id,
                "nonconverged_stores": len(nonconverged), **metrics,
            })

    comparison = pd.DataFrame(results).sort_values(["rmse", "run_name"]).reset_index(drop=True)
    comparison.to_csv(output / "comparison.csv", index=False)
    save_comparison(comparison, output / "comparison.png")
    # Non-converged time-series fits remain visible for comparison but are not eligible.
    eligible = comparison[comparison["nonconverged_stores"] == 0]
    if eligible.empty:
        raise RuntimeError("No eligible models")
    best = eligible.iloc[0]
    run_id = best["run_id"]
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    # Verify the logged model reproduces its validation metric before registration.
    reloaded_metrics = forecast_metrics(y_val, model.predict(X_val))
    np.testing.assert_allclose(reloaded_metrics["rmse"], best["rmse"], rtol=1e-8)
    X_test = test[FEATURE_COLUMNS]
    test_predictions = model.predict(X_test)
    test_metrics = forecast_metrics(test["Weekly_Sales"], test_predictions)
    test_directory = output / "selected_test"
    save_forecasts(test, test_predictions, test_directory, "test")

    with mlflow.start_run(run_id=run_id):
        mlflow.set_tag("selected", "true")
        mlflow.log_metrics({f"test_{key}": value for key, value in test_metrics.items()})
        mlflow.log_artifacts(str(test_directory), artifact_path="test")
        mlflow.log_artifact(str(output / "comparison.csv"), artifact_path="comparison")
        mlflow.log_artifact(str(output / "comparison.png"), artifact_path="comparison")

    version = stage_candidate(client, run_id, best["model_type"], best["rmse"])
    registered = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL}/{version}")
    np.testing.assert_allclose(registered.predict(X_test), test_predictions, rtol=1e-8)
    summary = {
        "batch_id": batch_id,
        "dataset_sha256": data_hash,
        "splits": splits,
        "best_run": best.to_dict(),
        "test_metrics": test_metrics,
        "registered_model": REGISTERED_MODEL,
        "version": version,
        "stage": "Staging",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output.parent / "latest.json").write_text(json.dumps({"batch_id": batch_id}), encoding="utf-8")
    print(comparison[["run_name", "rmse", "mae", "wape", "r2"]].to_string(index=False))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the retail forecasting experiment")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    args = parser.parse_args()
    run_experiment(args.data)
