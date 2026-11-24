from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.data import ROOT

EXPERIMENT = "Retail_Sales_Forecasting"
REGISTERED_MODEL = "RetailSalesForecaster"


def configure_tracking(root=ROOT):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifacts = root / "mlruns"
    artifacts.mkdir(exist_ok=True)
    uri = "sqlite:///" + (root / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient()


def set_experiment(root=ROOT):
    client = configure_tracking(root)
    if client.get_experiment_by_name(EXPERIMENT) is None:
        client.create_experiment(
            EXPERIMENT, artifact_location=(Path(root).resolve() / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    return client


def stage_candidate(client, run_id, model_type, rmse):
    version = mlflow.register_model(f"runs:/{run_id}/model", REGISTERED_MODEL)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "model_type", model_type)
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "validation_rmse", str(rmse))
    client.set_model_version_tag(REGISTERED_MODEL, version.version, "test_evaluated", "true")
    # Legacy stages are required in the course. Aliases also show the newer workflow.
    client.transition_model_version_stage(
        REGISTERED_MODEL, version.version, "Staging", archive_existing_versions=True
    )
    client.set_registered_model_alias(REGISTERED_MODEL, "candidate", version.version)
    return str(version.version)


def promote(client, version):
    model = client.get_model_version(REGISTERED_MODEL, str(version))
    if model.current_stage != "Staging" or model.tags.get("test_evaluated") != "true":
        raise ValueError("Only a test-evaluated Staging version can be promoted")
    run = client.get_run(model.run_id)
    if run.info.status != "FINISHED" or "test_rmse" not in run.data.metrics:
        raise ValueError("The candidate has no completed test evaluation")
    client.transition_model_version_stage(
        REGISTERED_MODEL, str(version), "Production", archive_existing_versions=True
    )
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", str(version))
