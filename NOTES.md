# Build & sequencing notes

Operational constraints that are NOT obvious from the code and must be respected
when building or re-running the pipeline.

## Phase ordering: load test (Phase 2) MUST precede Prometheus scraping (Phase 5)

Render free-tier services **spin down after ~15 min of inactivity** and cold-start
on the next request (multi-second first-request latency). The Phase 2 load test
deliberately measures **cold-start latency** (first request after spin-down) and
**warm latency** (steady-state) separately.

Once Phase 5 starts, Prometheus scrapes the live Render `/metrics` endpoint every
15–30 s. That continuous traffic keeps the service **permanently warm**, so cold
starts become **unobservable**. Therefore:

- **Do NOT reorder these phases.** Capture cold-start numbers in Phase 2, before
  Prometheus is pointed at the live URL.
- If Prometheus has already been running and you need a fresh cold-start number,
  stop the Prometheus scrape, wait for Render to spin the service down (~15 min
  idle), then measure.

## Airflow image deps (Phase 3)

The DAG's ML/validation libraries (`xgboost`, `mlflow`, `evidently`,
`great_expectations`, …) are baked into a custom image at build time via
`docker/airflow/Dockerfile` + `requirements-airflow.txt` — the stock
`apache/airflow` image ships none of them. Build with `docker compose build`
before first `up`. This install has NOT yet been verified against the Airflow
2.10.4 constraint set; the first real `docker compose build` in Phase 3 is the
verification point, and any pin adjustments will be recorded here.

## Python 3.10 (local) vs 3.11 (Docker/Render) — pickle crossing versions

Local training ran on Python **3.10.4** (only interpreter on this machine); the
serving container and Airflow image are Python **3.11**. The XGBoost `.ubj`
format is version-safe, but the model bundle is a pickled sklearn Pipeline
(`model.joblib`) whose preprocessor crosses a Python minor version at load time.
Identical pinned sklearn (1.5.2) on both sides makes this low-risk, **but it must
be proven, not assumed**: the Phase 2 container smoke test MUST exercise a real
`/predict` call through the preprocessor inside the 3.11 container — `/health`
alone does not validate the cross-version load.

## Metrics language (canonical, from eval/results/training_metrics.json)

Champion XGBoost: **ROC-AUC 0.8527 / PR-AUC 0.6644** (held-out test, seed 42).
Baseline LogReg: ROC-AUC 0.8495 / PR-AUC 0.6362. README/RESULTS must report both
models and say honestly: the ROC-AUC margin over the linear baseline is small
(~0.003), while the PR-AUC improvement (~0.028) is the operationally relevant
gain for an imbalanced churn target. No spin beyond that.

## MLflow: local file store first, DagsHub second

Phase 1 trains against a local `mlruns/` file store (gitignored). Re-logging the
single best run to DagsHub-hosted MLflow + registering the champion in the Model
Registry is a **separate, credential-gated step**. DagsHub credentials are passed
as exported env vars (`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD`) and
are **never** written to `.env` or any committed file.
