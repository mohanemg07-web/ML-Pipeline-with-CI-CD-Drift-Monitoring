"""Drift-triggered retraining: data prep, fit, promotion gate, registry
promotion, and post-redeploy verification.

ALL logic lives here as plain functions with injectable I/O so it is testable
outside Airflow; the DAG (``dags/churn_retrain_dag.py``) is orchestration only.
Functions that talk to DagsHub (``log_and_promote``) or Render
(``redeploy_and_verify``) read credentials/URLs from the environment at call
time and are exercised with stubs in tests.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import config, data

logger = logging.getLogger(__name__)

BATCH_EVAL_FRACTION = 0.30  # of the production batch, held out for the gate


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 — 3.10 compat


# --------------------------------------------------------------------------
# Data prep
# --------------------------------------------------------------------------
@dataclass
class RetrainData:
    X_fit: pd.DataFrame          # reference train+val + batch train portion
    y_fit: pd.Series
    X_batch_eval: pd.DataFrame   # drifted holdout for the promotion gate
    y_batch_eval: pd.Series
    X_orig_test: pd.DataFrame    # original test set (no-regression check)
    y_orig_test: pd.Series
    sizes: dict[str, int] | None = None


def prepare_retrain_data(
    reference_raw: pd.DataFrame,
    batch_raw: pd.DataFrame,
    seed: int = config.RANDOM_SEED,
) -> RetrainData:
    """Combine reference fit data with the batch; hold out a drifted eval slice.

    The ORIGINAL test split (same seed as training) stays untouched so the
    retrained model can be checked for regression on the reference
    distribution, and the batch eval slice never enters fitting.
    """
    ref_splits = data.split(reference_raw, seed=seed)

    batch = data.clean(batch_raw)
    Xb = batch[config.FEATURE_COLS]
    yb = (batch[config.TARGET_COL] == config.POSITIVE_LABEL).astype(int)
    Xb_train, Xb_eval, yb_train, yb_eval = train_test_split(
        Xb, yb, test_size=BATCH_EVAL_FRACTION, stratify=yb, random_state=seed
    )

    X_fit = pd.concat([ref_splits.X_train, ref_splits.X_val, Xb_train])
    y_fit = pd.concat([ref_splits.y_train, ref_splits.y_val, yb_train])

    return RetrainData(
        X_fit=X_fit,
        y_fit=y_fit,
        X_batch_eval=Xb_eval,
        y_batch_eval=yb_eval,
        X_orig_test=ref_splits.X_test,
        y_orig_test=ref_splits.y_test,
        sizes={
            "fit": len(X_fit),
            "batch_train": len(Xb_train),
            "batch_eval": len(Xb_eval),
            "orig_test": len(ref_splits.X_test),
        },
    )


# --------------------------------------------------------------------------
# Fit (small Optuna budget; separated from any MLflow/registry I/O)
# --------------------------------------------------------------------------
def fit_retrained(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    n_trials: int = config.RETRAIN_N_TRIALS,
    seed: int = config.RANDOM_SEED,
):
    """Fit a fresh pipeline on combined data. Returns ``(pipeline, best_params)``.

    Internal carve-out validation for the search; final refit on all of
    ``X_fit``. Mirrors training-time architecture (same preprocessor + XGB).
    """
    import optuna
    from xgboost import XGBClassifier

    from src.train import _build_pipeline

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_fit, y_fit, test_size=0.2, stratify=y_fit, random_state=seed
    )
    spw = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 400, step=50),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        }
        clf = XGBClassifier(
            **params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
        pipe = _build_pipeline(clf)
        pipe.fit(X_tr, y_tr)
        return roc_auc_score(y_val, pipe.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    final = _build_pipeline(
        XGBClassifier(
            **study.best_params,
            scale_pos_weight=spw,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=-1,
        )
    )
    final.fit(X_fit, y_fit)
    return final, study.best_params


# --------------------------------------------------------------------------
# Evaluation, gate, smoke test, branch — pure logic
# --------------------------------------------------------------------------
def auc_pair(model, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = model.predict_proba(X)[:, 1]
    return {
        "roc_auc": round(float(roc_auc_score(y, proba)), 6),
        "pr_auc": round(float(average_precision_score(y, proba)), 6),
    }


def choose_branch(drift_detected: bool) -> str:
    """BranchPythonOperator target: task_id to follow."""
    return "retrain" if drift_detected else "no_retrain"


def promotion_gate(
    new_auc: float,
    stale_auc: float,
    tolerance: float = config.PROMOTION_AUC_TOLERANCE,
) -> tuple[bool, str]:
    """New model must not trail the stale champion by more than ``tolerance``
    on the SAME drifted eval slice."""
    if new_auc >= stale_auc - tolerance:
        return True, (
            f"PASS: new_auc={new_auc:.4f} >= stale_auc={stale_auc:.4f} - tol={tolerance}"
        )
    return False, (
        f"FAIL: new_auc={new_auc:.4f} < stale_auc={stale_auc:.4f} - tol={tolerance}"
    )


def smoke_test(model, X_rows: pd.DataFrame) -> tuple[bool, str]:
    """Candidate must produce finite probabilities in [0, 1] for real rows."""
    try:
        proba = model.predict_proba(X_rows)[:, 1]
    except Exception as exc:  # noqa: BLE001 — any scoring failure fails the smoke test
        return False, f"FAIL: predict_proba raised {type(exc).__name__}: {exc}"
    if len(proba) != len(X_rows):
        return False, f"FAIL: {len(proba)} probabilities for {len(X_rows)} rows"
    if not np.all(np.isfinite(proba)):
        return False, "FAIL: non-finite probabilities"
    if proba.min() < 0.0 or proba.max() > 1.0:
        return False, f"FAIL: probabilities outside [0,1] ({proba.min()}, {proba.max()})"
    return True, f"PASS: {len(proba)} probabilities in [{proba.min():.4f}, {proba.max():.4f}]"


# --------------------------------------------------------------------------
# Registry promotion (credential-gated; called only inside the DAG)
# --------------------------------------------------------------------------
def log_and_promote(pipeline, metrics: dict, params: dict) -> dict:
    """Log the retrained run to MLflow, register it, move the champion alias.

    Mirrors register_dagshub.py: alias first, Production-stage fallback if the
    server rejects aliases. Credentials come from the environment (compose
    shell-passthrough); never from files.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"drift-retrain-{_now_iso()[:19]}") as run:
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, int | float)})
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        run_id = run.info.run_id

    client = MlflowClient()
    mv = mlflow.register_model(
        f"runs:/{run_id}/model", config.REGISTERED_MODEL_NAME
    )
    try:
        client.set_registered_model_alias(
            config.REGISTERED_MODEL_NAME, config.CHAMPION_ALIAS, mv.version
        )
        champion_ref = f"alias:{config.CHAMPION_ALIAS}"
    except Exception:  # noqa: BLE001 — server may reject aliases; use stage
        client.transition_model_version_stage(
            config.REGISTERED_MODEL_NAME, mv.version, stage="Production",
            archive_existing_versions=True,
        )
        champion_ref = "stage:Production"
    return {"version": str(mv.version), "run_id": run_id, "champion_ref": champion_ref}


# --------------------------------------------------------------------------
# Redeploy + verification (three honest outcomes)
# --------------------------------------------------------------------------
def redeploy_and_verify(
    expected_version: str,
    hook_url: str,
    base_url: str,
    *,
    post=None,
    fetch_health=None,
    sleep=time.sleep,
    timeout_s: float = 900.0,
    interval_s: float = 15.0,
) -> dict:
    """Fire the Render deploy hook, then poll /health until one of:

    - ``new_model_live``:   reported model_version == expected_version
    - ``fallback_bundled``: the NEW deployment answered with the bundle
      (registry pull failed at boot -> partial failure, reason recorded)
    - ``timeout``:          neither observed within timeout_s

    While Render swaps instances the OLD deployment keeps answering (its
    version differs but its source is registry:*), so a bundled answer can
    only come from the new instance — unless the fleet was already bundled
    before the deploy, which the caller rules out by checking /health first.
    """
    import requests

    if post is None:
        post = lambda url: requests.post(url, timeout=30)  # noqa: E731
    if fetch_health is None:
        def fetch_health():
            return requests.get(f"{base_url}/health", timeout=30).json()

    hook_fired_ts = _now_iso()
    hook_resp = post(hook_url)
    hook_status = getattr(hook_resp, "status_code", None)
    if hook_status is not None and hook_status >= 400:
        return {
            "outcome": "hook_failed",
            "hook_status": hook_status,
            "hook_fired_ts": hook_fired_ts,
            "verified_ts": None,
            "polls": 0,
            "last_health": None,
        }

    deadline = time.monotonic() + timeout_s
    polls = 0
    last_health: dict | None = None
    while time.monotonic() < deadline:
        try:
            last_health = fetch_health()
            polls += 1
            version = str(last_health.get("model_version", ""))
            source = str(last_health.get("model_source", ""))
            if version == str(expected_version):
                return {
                    "outcome": "new_model_live",
                    "hook_status": hook_status,
                    "hook_fired_ts": hook_fired_ts,
                    "verified_ts": _now_iso(),
                    "polls": polls,
                    "last_health": last_health,
                }
            if source.startswith("bundled"):
                return {
                    "outcome": "fallback_bundled",
                    "reason": "new deployment answered with the bundled model: "
                    "registry pull failed or timed out at boot",
                    "hook_status": hook_status,
                    "hook_fired_ts": hook_fired_ts,
                    "verified_ts": _now_iso(),
                    "polls": polls,
                    "last_health": last_health,
                }
        except Exception as exc:  # noqa: BLE001 — service restarting; keep polling
            logger.info("health poll failed (%s); retrying", type(exc).__name__)
        sleep(interval_s)

    return {
        "outcome": "timeout",
        "hook_status": hook_status,
        "hook_fired_ts": hook_fired_ts,
        "verified_ts": None,
        "polls": polls,
        "last_health": last_health,
    }


# --------------------------------------------------------------------------
# Results record
# --------------------------------------------------------------------------
def record_results(payload: dict, path: Path = config.RETRAIN_LOOP_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def stage_timings(
    detection_ts: str,
    retrain_done_ts: str,
    hook_fired_ts: str,
    verified_ts: str | None,
) -> dict:
    """Stage breakdown of the detection->redeploy wall clock (seconds)."""
    def _p(ts: str) -> datetime:
        return datetime.fromisoformat(ts)

    out = {
        "detection_to_retrain_done_s": round(
            (_p(retrain_done_ts) - _p(detection_ts)).total_seconds(), 1
        ),
        "retrain_done_to_hook_fired_s": round(
            (_p(hook_fired_ts) - _p(retrain_done_ts)).total_seconds(), 1
        ),
    }
    if verified_ts:
        out["hook_fired_to_verified_live_s"] = round(
            (_p(verified_ts) - _p(hook_fired_ts)).total_seconds(), 1
        )
        out["total_detection_to_live_s"] = round(
            (_p(verified_ts) - _p(detection_ts)).total_seconds(), 1
        )
    return out
