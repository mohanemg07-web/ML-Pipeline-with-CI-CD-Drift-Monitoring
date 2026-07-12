"""Training entrypoint: baseline LogReg + Optuna-tuned XGBoost, tracked in MLflow.

Runs entirely against a LOCAL MLflow file store by default (``mlruns/``,
gitignored). Writes the authoritative test-set metrics to
``eval/results/training_metrics.json`` — the only file README numbers may cite.

Re-logging the best run to DagsHub and registering the champion in the Model
Registry is a separate, credential-gated step (see ``src/register_dagshub.py``).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config, data
from src.features import build_preprocessor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _metrics(y_true, y_proba, threshold: float = config.DECISION_THRESHOLD) -> dict:
    """Compute the reported metric set from probabilities."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred)),
        "threshold": threshold,
    }


def _build_pipeline(clf) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def _tune_xgboost(splits: data.DataSplits, n_trials: int, seed: int) -> optuna.Study:
    """Optuna search maximizing validation ROC-AUC."""
    scale_pos_weight = float((splits.y_train == 0).sum() / max((splits.y_train == 1).sum(), 1))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        }
        clf = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        pipe = _build_pipeline(clf)
        pipe.fit(splits.X_train, splits.y_train)
        val_proba = pipe.predict_proba(splits.X_val)[:, 1]
        return roc_auc_score(splits.y_val, val_proba)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study


def _feature_importance_plot(pipe: Pipeline, out_path) -> None:
    prep = pipe.named_steps["prep"]
    clf = pipe.named_steps["clf"]
    names = np.array(prep.get_feature_names_out())
    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(order)), importances[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(names[order][::-1], fontsize=8)
    ax.set_xlabel("XGBoost feature importance (gain-weighted)")
    ax.set_title("Top 20 features — churn classifier")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(n_trials: int = 25, seed: int = config.RANDOM_SEED) -> dict:
    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.SERVING_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri((config.REPO_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(config.EXPERIMENT_NAME)

    df = data.load_raw()
    splits = data.split(df, seed=seed)
    # Combined train+val for final refit; test stays untouched.
    X_fit = pd.concat([splits.X_train, splits.X_val])
    y_fit = pd.concat([splits.y_train, splits.y_val])
    target_balance = float(y_fit.mean())

    # --- Baseline: logistic regression ---
    with mlflow.start_run(run_name="baseline-logreg") as base_run:
        base_pipe = _build_pipeline(
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        )
        base_pipe.fit(X_fit, y_fit)
        base_test_proba = base_pipe.predict_proba(splits.X_test)[:, 1]
        base_test = _metrics(splits.y_test, base_test_proba)
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.log_metrics({f"test_{k}": v for k, v in base_test.items() if k != "threshold"})
        base_run_id = base_run.info.run_id

    # --- XGBoost: Optuna search on validation, refit best on train+val ---
    study = _tune_xgboost(splits, n_trials=n_trials, seed=seed)
    scale_pos_weight = float((splits.y_train == 0).sum() / max((splits.y_train == 1).sum(), 1))

    with mlflow.start_run(run_name="xgboost-optuna") as xgb_run:
        best_params = study.best_params
        final_clf = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        final_pipe = _build_pipeline(final_clf)
        final_pipe.fit(X_fit, y_fit)

        val_proba = final_pipe.predict_proba(splits.X_val)[:, 1]
        test_proba = final_pipe.predict_proba(splits.X_test)[:, 1]
        xgb_val = _metrics(splits.y_val, val_proba)
        xgb_test = _metrics(splits.y_test, test_proba)

        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_params(best_params)
        mlflow.log_metric("best_val_roc_auc", float(study.best_value))
        mlflow.log_metrics({f"val_{k}": v for k, v in xgb_val.items() if k != "threshold"})
        mlflow.log_metrics({f"test_{k}": v for k, v in xgb_test.items() if k != "threshold"})

        fi_path = config.MODELS_DIR / "feature_importance.png"
        _feature_importance_plot(final_pipe, fi_path)
        mlflow.log_artifact(str(fi_path))

        mlflow.sklearn.log_model(final_pipe, artifact_path="model")
        xgb_run_id = xgb_run.info.run_id

    # --- Persist model bundle: local + committed serving fallback ---
    import joblib

    joblib.dump(final_pipe, config.MODELS_DIR / "model.joblib")
    joblib.dump(final_pipe, config.SERVING_MODEL_DIR / "model.joblib")

    gate_passed = xgb_test["roc_auc"] >= config.AUC_GATE
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "python_version": platform.python_version(),
        "dataset": {
            "n_total": int(len(df)),
            **{f"n_{k}": v for k, v in splits.sizes.items()},
            "target_balance_fit": target_balance,
        },
        "baseline_logreg": {"test": base_test, "mlflow_run_id": base_run_id},
        "xgboost": {
            "n_trials": n_trials,
            "best_params": best_params,
            "best_val_roc_auc": float(study.best_value),
            "val": xgb_val,
            "test": xgb_test,
            "mlflow_run_id": xgb_run_id,
        },
        "champion": "xgboost",
        "champion_test_roc_auc": xgb_test["roc_auc"],
        "auc_gate": config.AUC_GATE,
        "gate_passed": bool(gate_passed),
        "mlflow": {
            "tracking_uri": mlflow.get_tracking_uri(),
            "experiment": config.EXPERIMENT_NAME,
            "best_run_id": xgb_run_id,
        },
    }
    out_path = config.EVAL_RESULTS_DIR / "training_metrics.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()
    res = main(n_trials=args.trials, seed=args.seed)
    sys.exit(0 if res["gate_passed"] else 1)
