"""Drift-triggered retraining DAG. Orchestration ONLY — every decision lives
in ``src/`` as a plain tested function; tasks are thin wrappers moving small
dicts through XCom and artifacts through mounted repo paths.

Flow:
  ingest_batch -> validate_batch -> drift_check -> branch_on_drift
    -> no_retrain                                  (PSI trigger not met)
    -> retrain -> evaluate_and_gate -> promote -> redeploy -> record_results

Credentials (MLFLOW_*, RENDER_DEPLOY_HOOK_URL, SERVING_BASE_URL) arrive via
docker-compose shell passthrough; nothing here reads or writes secret files.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

from src import config

STALE_CHAMPION_LOAD_TIMEOUT_S = 60.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017


def _ingest_batch(ti):
    import pandas as pd

    path = config.PRODUCTION_BATCH_PATH
    if not path.exists():
        raise AirflowFailException(
            f"no production batch at {path}; run `python -m src.simulate_drift` first"
        )
    df = pd.read_csv(path)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    ti.xcom_push(key="batch_meta", value=meta)
    ti.xcom_push(key="n_batch", value=int(len(df)))


def _validate_batch():
    import pandas as pd

    from src.validate import validate_dataframe

    df = pd.read_csv(config.PRODUCTION_BATCH_PATH)
    result = validate_dataframe(df, save_suite=False)
    if not result["success"]:
        raise AirflowFailException(f"GE validation failed: {result}")
    return result


def _drift_check(ti):
    import pandas as pd

    from src import data
    from src.drift import run_drift_check

    raw = data.load_raw()
    ref_splits = data.split(raw, seed=config.RANDOM_SEED)
    # reference = the distribution the champion was fitted on (train+val)
    reference = pd.concat([ref_splits.X_train, ref_splits.X_val])

    batch = data.clean(pd.read_csv(config.PRODUCTION_BATCH_PATH))[config.FEATURE_COLS]
    decision = run_drift_check(reference, batch)

    ti.xcom_push(key="detection_ts", value=_now_iso())
    ti.xcom_push(
        key="drift",
        value={
            "psi_by_feature": decision.psi_by_feature,
            "threshold": decision.threshold,
            "min_features": decision.min_features,
            "drifted_features": decision.drifted_features,
            "detected": decision.detected,
        },
    )


def _branch(ti) -> str:
    from src.retrain import choose_branch

    drift = ti.xcom_pull(task_ids="drift_check", key="drift")
    return choose_branch(drift["detected"])


def _retrain(ti):
    import joblib
    import pandas as pd

    from src import data
    from src.model_resolver import load_champion
    from src.retrain import auc_pair, fit_retrained, prepare_retrain_data

    prepared = prepare_retrain_data(
        data.load_raw(), pd.read_csv(config.PRODUCTION_BATCH_PATH)
    )

    # Stale champion = what the registry (and, under Option A, the live
    # service) currently points at.
    import mlflow

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    stale_model, stale_uri = load_champion(timeout_s=STALE_CHAMPION_LOAD_TIMEOUT_S)
    stale = {
        "uri": stale_uri,
        "auc_batch_eval": auc_pair(stale_model, prepared.X_batch_eval, prepared.y_batch_eval),
        "auc_orig_test": auc_pair(stale_model, prepared.X_orig_test, prepared.y_orig_test),
    }

    pipeline, best_params = fit_retrained(prepared.X_fit, prepared.y_fit)
    new = {
        "best_params": best_params,
        "auc_batch_eval": auc_pair(pipeline, prepared.X_batch_eval, prepared.y_batch_eval),
        "auc_orig_test": auc_pair(pipeline, prepared.X_orig_test, prepared.y_orig_test),
    }

    config.RETRAIN_CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, config.RETRAIN_CANDIDATE_PATH)

    ti.xcom_push(key="retrain_done_ts", value=_now_iso())
    ti.xcom_push(key="stale", value=stale)
    ti.xcom_push(key="new", value=new)
    ti.xcom_push(key="sizes", value=prepared.sizes)


def _evaluate_and_gate(ti):
    import joblib
    import pandas as pd

    from src import data
    from src.retrain import promotion_gate, smoke_test

    stale = ti.xcom_pull(task_ids="retrain", key="stale")
    new = ti.xcom_pull(task_ids="retrain", key="new")

    passed, reason = promotion_gate(
        new_auc=new["auc_batch_eval"]["roc_auc"],
        stale_auc=stale["auc_batch_eval"]["roc_auc"],
    )

    candidate = joblib.load(config.RETRAIN_CANDIDATE_PATH)
    rows = data.clean(pd.read_csv(config.PRODUCTION_BATCH_PATH))[config.FEATURE_COLS].head(5)
    smoke_ok, smoke_reason = smoke_test(candidate, rows)

    ti.xcom_push(key="gate", value={
        "auc_gate_passed": passed, "auc_gate_reason": reason,
        "smoke_passed": smoke_ok, "smoke_reason": smoke_reason,
    })
    if not (passed and smoke_ok):
        raise AirflowFailException(f"promotion gate: {reason} | smoke: {smoke_reason}")


def _promote(ti):
    import joblib
    import mlflow

    from src.retrain import log_and_promote

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    stale = ti.xcom_pull(task_ids="retrain", key="stale")
    new = ti.xcom_pull(task_ids="retrain", key="new")
    pipeline = joblib.load(config.RETRAIN_CANDIDATE_PATH)
    promoted = log_and_promote(
        pipeline,
        metrics={
            "batch_eval_roc_auc": new["auc_batch_eval"]["roc_auc"],
            "batch_eval_pr_auc": new["auc_batch_eval"]["pr_auc"],
            "orig_test_roc_auc": new["auc_orig_test"]["roc_auc"],
            "orig_test_pr_auc": new["auc_orig_test"]["pr_auc"],
            "stale_batch_eval_roc_auc": stale["auc_batch_eval"]["roc_auc"],
        },
        params=new["best_params"],
    )
    ti.xcom_push(key="promoted", value=promoted)


def _redeploy(ti):
    from src.retrain import redeploy_and_verify

    hook_url = os.environ.get("RENDER_DEPLOY_HOOK_URL", "")
    base_url = os.environ.get("SERVING_BASE_URL", "").rstrip("/")
    if not hook_url or not base_url:
        raise AirflowFailException(
            "RENDER_DEPLOY_HOOK_URL / SERVING_BASE_URL not set in the environment"
        )
    promoted = ti.xcom_pull(task_ids="promote", key="promoted")
    result = redeploy_and_verify(
        expected_version=promoted["version"], hook_url=hook_url, base_url=base_url
    )
    ti.xcom_push(key="redeploy", value=result)
    if result["outcome"] not in ("new_model_live", "fallback_bundled"):
        raise AirflowFailException(f"redeploy verification: {result['outcome']}")


def _record_results(ti):
    from src.retrain import record_results, stage_timings

    drift = ti.xcom_pull(task_ids="drift_check", key="drift")
    detection_ts = ti.xcom_pull(task_ids="drift_check", key="detection_ts")
    retrain_done_ts = ti.xcom_pull(task_ids="retrain", key="retrain_done_ts")
    stale = ti.xcom_pull(task_ids="retrain", key="stale")
    new = ti.xcom_pull(task_ids="retrain", key="new")
    gate = ti.xcom_pull(task_ids="evaluate_and_gate", key="gate")
    promoted = ti.xcom_pull(task_ids="promote", key="promoted")
    redeploy = ti.xcom_pull(task_ids="redeploy", key="redeploy")

    stale_auc = stale["auc_batch_eval"]["roc_auc"]
    new_auc = new["auc_batch_eval"]["roc_auc"]
    record_results({
        "generated_at": _now_iso(),
        "batch_meta": ti.xcom_pull(task_ids="ingest_batch", key="batch_meta"),
        "sizes": ti.xcom_pull(task_ids="retrain", key="sizes"),
        "drift": drift,
        "stale_champion": stale,
        "retrained": {**new, **(promoted or {})},
        "auc_on_drifted_eval": {
            "stale_champion": stale_auc,
            "retrained": new_auc,
            "recovery_delta": round(new_auc - stale_auc, 6),
        },
        "gate": gate,
        "redeploy": redeploy,
        "timings_s": stage_timings(
            detection_ts=detection_ts,
            retrain_done_ts=retrain_done_ts,
            hook_fired_ts=redeploy["hook_fired_ts"],
            verified_ts=redeploy.get("verified_ts"),
        ),
    })


with DAG(
    dag_id="churn_drift_retrain",
    description="PSI drift check -> gated retrain -> registry promote -> Render redeploy",
    schedule=None,          # triggered manually / by upstream tooling
    start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),  # noqa: UP017
    catchup=False,
    default_args={"retries": 0},
    tags=["mlops", "drift"],
) as dag:
    ingest_batch = PythonOperator(task_id="ingest_batch", python_callable=_ingest_batch)
    validate_batch = PythonOperator(task_id="validate_batch", python_callable=_validate_batch)
    drift_check = PythonOperator(task_id="drift_check", python_callable=_drift_check)
    branch_on_drift = BranchPythonOperator(task_id="branch_on_drift", python_callable=_branch)
    no_retrain = EmptyOperator(task_id="no_retrain")
    retrain = PythonOperator(task_id="retrain", python_callable=_retrain)
    evaluate_and_gate = PythonOperator(
        task_id="evaluate_and_gate", python_callable=_evaluate_and_gate
    )
    promote = PythonOperator(task_id="promote", python_callable=_promote)
    redeploy = PythonOperator(task_id="redeploy", python_callable=_redeploy)
    record_results = PythonOperator(task_id="record_results", python_callable=_record_results)

    ingest_batch >> validate_batch >> drift_check >> branch_on_drift
    branch_on_drift >> no_retrain
    branch_on_drift >> retrain >> evaluate_and_gate >> promote >> redeploy >> record_results
