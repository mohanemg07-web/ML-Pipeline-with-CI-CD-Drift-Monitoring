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
before first `up`.

**Constraint conflict found at the first real build (2026-07-13):** the official
Airflow 2.10.4/py3.11 constraint set pins `cryptography==42.0.8`, while
`evidently==0.4.40` requires `cryptography>=43.0.1` (`great-expectations` only
needs `>=3.2`), so pip reported ResolutionImpossible. Fix (in
`docker/airflow/Dockerfile`): download the constraint file, delete only the
`cryptography==` line, install against the edited copy. Rationale: Airflow core
accepts `cryptography>=41.0.0`, so letting it float is safe, whereas downgrading
evidently would break version lockstep with `requirements.txt` (local PSI
numbers were produced with 0.4.40). No requirement pins in
`requirements-airflow.txt` were changed.

With the pin dropped, pip resolved `cryptography` to **46.0.0** and warned that
`pyopenssl 24.3.0` (<45), `msal 1.31.1` (<46) and `gcloud-aio-auth 5.3.2` (<45)
declare tighter ranges. Verified empirically in the built image: `import
OpenSSL` and every DAG dependency (xgboost, sklearn, mlflow, evidently, GE,
optuna, airflow) import cleanly, so the warnings are declared-range noise for
providers this DAG never touches. If a future rebuild does hit a real pyOpenSSL
break, pin `cryptography==44.0.1` in `requirements-airflow.txt` (satisfies
evidently >=43.0.1 AND all three <45/<46 ranges).

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

## Checkpoint 1b: DagsHub registry (completed 2026-07-13)

- Champion re-log run: <https://dagshub.com/mohanemg07-web/ML-Pipeline-with-CI-CD-Drift-Monitoring.mlflow/#/experiments/0/runs/4f8a1bd2f8ad4ebc843aa504ad70887e>
  (baseline comparison run `b8475f52b49a46359cd01009499c8896` in the same experiment).
- Registered model `churn-xgboost` **v1**; `verify_registry` resolved via the
  **alias path** (`models:/churn-xgboost@champion`) — the stage fallback was not
  needed. 5-row prediction comparison local vs registry: identical
  (`max_abs_diff = 0.0`), exit code 0.
- One orphaned `baseline-logreg-relog` run exists on DagsHub from a first attempt
  that crashed printing MLflow's emoji status line to a cp1252 console
  (`UnicodeEncodeError`). Fix: run registry scripts with `PYTHONUTF8=1` on
  Windows. The orphan is harmless (nothing registered from it).

## Checkpoint 2-local: serving container verified (2026-07-13)

- `docker build -f serving/Dockerfile .` on python:3.11-slim, non-root user,
  bundled `serving/model/model.joblib` as the primary model path
  (`MODEL_SOURCE=registry` optionally pulls the champion via
  `src.model_resolver` with a hard 5 s timeout and bundled fallback).
- **3.10 → 3.11 pickle crossing PROVEN** (the risk flagged above): the same
  high-risk payload scores `churn_probability = 0.890204` from the local
  Python 3.10 venv AND from `/predict` inside the 3.11 container — the pickled
  sklearn preprocessor loads and transforms identically across versions.
- Memory under burst (300 singles + 20×100-row batches, 8 threads, container
  capped with `--memory 512m`): peak **155.1 MiB / 512 MiB** at ~137% CPU;
  idle ~152 MiB. p50/p95 single-predict latency 181/307 ms on this machine.
- Serving pins `xgboost-cpu==2.1.3` (same `import xgboost` package, no
  `nvidia-nccl-cu12` payload): image is **825 MB** unpacked vs 1.94 GB with
  regular xgboost. Prediction parity after the swap verified to 12 decimal
  places against the local regular-xgboost venv on two contrasting payloads;
  burst peak RSS 132.9 MiB. Training (requirements.txt) keeps regular xgboost.
- Shadow mode: `CHALLENGER_TRAFFIC_PCT` + `serving/model/challenger.joblib`
  (or `CHALLENGER_MODEL_PATH`); champion is always returned; JSONL comparisons
  to `SHADOW_LOG_PATH` (`/app/shadow/comparisons.jsonl` in-container). Disabled
  => `build_shadow()` returns None and the hot path costs one None-check.

## Checkpoint 3-e2e: drift → retrain → promote → redeploy loop verified (2026-07-13)

Full `churn_drift_retrain` DAG run (`manual__2026-07-13T13:57:48+00:00`) in the
local compose stack against live DagsHub + Render, all nine tasks green:

- **Image build**: first verification of `requirements-airflow.txt` against the
  Airflow 2.10.4 constraints surfaced the cryptography conflict documented
  above; after the single-constraint relaxation the build succeeded and all DAG
  deps import cleanly in-image.
- **Drift** (severity 1.0, n=1500, seed=7): PSI flagged exactly the five
  shifted features — tenure 2.73, MonthlyCharges 1.70, TotalCharges 0.44,
  PaymentMethod 0.37, InternetService 0.35 (threshold 0.2, min 2). The 14
  unshifted features all scored < 0.002 — clean separation, no false positives.
- **Retrain + gate**: stale champion 0.8515 ROC-AUC on the drifted eval slice;
  retrained model 0.8566 (**recovery delta +0.0051**), orig-test regression
  check 0.8517 vs 0.8527 (−0.001, within tolerance). Gate + smoke PASS.
- **Promotion**: registered `churn-xgboost` **v2**, run
  `4f476fb4635e4c19a0a8cec33656e57b`, champion alias moved (alias path again,
  no stage fallback).
- **Redeploy**: hook 200 → `new_model_live` verified in **46.9 s** (1 poll;
  fast because the deploy hook rebuilds nothing — cached image, boot re-pulls
  the new champion). Total detection→live **166.5 s**. Full record in
  `eval/results/retrain_loop.json`.
- **Parity sanity**: live `/health` reports v2 + the new run id; `/predict` on
  a drifted-batch row returned 0.356211 from BOTH the local retrain-candidate
  pipeline and the live endpoint (abs diff 0.0).
- Note: under this simulated covariate-only shift the stale champion barely
  degrades (0.8515 vs its 0.8527 test AUC) — labels didn't move, so the honest
  claim is "the loop detects the shift and ships a slightly better model",
  not "the loop rescued a collapsed model".

## Repo name (changed from earlier plan)

The GitHub repo is **`mohanemg07-web/ML-Pipeline-with-CI-CD-Drift-Monitoring`**
(<https://github.com/mohanemg07-web/ML-Pipeline-with-CI-CD-Drift-Monitoring>),
NOT the earlier planned "End-to-end-MLOps-Pipeline". The DagsHub repo uses the
same name. Use this URL as `origin` at push time; `.env.example` and the README
title are already aligned.
