"""Re-log the already-trained champion to DagsHub MLflow + Model Registry.

DOES NOT RETRAIN. Everything is read from artifacts already on disk:
  - metrics/params:  eval/results/training_metrics.json
  - fitted pipeline: models/model.joblib (preprocessor + XGBoost, one object)
  - plot:            models/feature_importance.png

Credentials are read from the environment ONLY (MLFLOW_TRACKING_URI,
MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD). They are never printed,
never written to disk, and never passed as CLI arguments.

Logs two runs (baseline-logreg metrics for comparison context + the XGBoost
champion with params/metrics/model/preprocessor/plot), registers the champion
as ``churn-xgboost`` and sets the ``champion`` alias (falls back to the
``Production`` stage if the server doesn't support aliases).

Prints a JSON summary with run IDs/URLs — no secrets.
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import mlflow
from mlflow.tracking import MlflowClient

from src import config

REQUIRED_ENV = ["MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD"]


def main() -> int:
    missing = [k for k in REQUIRED_ENV if not os.environ.get(k)]
    if missing:
        print(f"ABORT: missing env vars: {missing} (values are never printed)")
        return 2

    metrics_path = config.EVAL_RESULTS_DIR / "training_metrics.json"
    model_path = config.MODELS_DIR / "model.joblib"
    plot_path = config.MODELS_DIR / "feature_importance.png"
    for p in (metrics_path, model_path, plot_path):
        if not p.exists():
            print(f"ABORT: required artifact missing: {p} — do NOT retrain; investigate.")
            return 3

    results = json.loads(metrics_path.read_text())
    pipeline = joblib.load(model_path)

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    # --- Run 1: baseline metrics-only, for comparison context ---
    base = results["baseline_logreg"]["test"]
    with mlflow.start_run(run_name="baseline-logreg-relog") as base_run:
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_param("relogged_from_local_run", results["baseline_logreg"]["mlflow_run_id"])
        mlflow.log_metrics({f"test_{k}": v for k, v in base.items() if k != "threshold"})
        base_run_id = base_run.info.run_id

    # --- Run 2: champion XGBoost with full artifacts ---
    xgb = results["xgboost"]
    with mlflow.start_run(run_name="xgboost-champion-relog") as champ_run:
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_trials", xgb["n_trials"])
        mlflow.log_param("seed", results["seed"])
        mlflow.log_param("relogged_from_local_run", xgb["mlflow_run_id"])
        mlflow.log_params(xgb["best_params"])
        mlflow.log_metric("best_val_roc_auc", xgb["best_val_roc_auc"])
        mlflow.log_metrics({f"val_{k}": v for k, v in xgb["val"].items() if k != "threshold"})
        mlflow.log_metrics({f"test_{k}": v for k, v in xgb["test"].items() if k != "threshold"})

        mlflow.log_artifact(str(plot_path))
        # preprocessor also logged standalone for inspection
        prep_path = config.MODELS_DIR / "preprocessor.joblib"
        joblib.dump(pipeline.named_steps["prep"], prep_path)
        mlflow.log_artifact(str(prep_path))

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            registered_model_name=config.REGISTERED_MODEL_NAME,
        )
        champ_run_id = champ_run.info.run_id

    # --- Alias/stage the freshly registered version as champion ---
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{config.REGISTERED_MODEL_NAME}'")
    newest = max(versions, key=lambda v: int(v.version))
    alias_mode = None
    try:
        client.set_registered_model_alias(
            config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, newest.version
        )
        alias_mode = f"alias:{config.CHAMPION_ALIAS}"
    except Exception as exc:  # server may not support aliases
        print(f"alias not supported ({type(exc).__name__}); falling back to stage")
        client.transition_model_version_stage(
            config.REGISTERED_MODEL_NAME, newest.version, stage="Production"
        )
        alias_mode = "stage:Production"

    summary = {
        "tracking_host": tracking_uri.split("//")[-1].split("/")[0],
        "experiment": config.EXPERIMENT_NAME,
        "baseline_run_id": base_run_id,
        "champion_run_id": champ_run_id,
        "registered_model": config.REGISTERED_MODEL_NAME,
        "registered_version": newest.version,
        "champion_ref": alias_mode,
        "test_roc_auc": xgb["test"]["roc_auc"],
        "test_pr_auc": xgb["test"]["pr_auc"],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
