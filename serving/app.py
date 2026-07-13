"""FastAPI churn-serving app.

Factory pattern (``create_app``) so tests can construct isolated apps with
patched environments; the module-level ``app`` is the uvicorn entrypoint.
Each app instance gets its own Prometheus ``CollectorRegistry`` to avoid
duplicate-metric errors across instances.
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)

from serving.model_loader import load_serving_model
from serving.schemas import BatchRequest, BatchResponse, CustomerFeatures, Prediction
from serving.shadow import build_shadow
from src import config


def create_app() -> FastAPI:
    model, model_info = load_serving_model()
    model_source = model_info["model_source"]
    shadow = build_shadow()

    registry = CollectorRegistry()
    predictions_total = Counter(
        "churn_predictions_total",
        "Rows scored, by endpoint",
        ["endpoint"],
        registry=registry,
    )
    predict_latency = Histogram(
        "churn_predict_latency_seconds",
        "Model scoring latency per request, by endpoint",
        ["endpoint"],
        registry=registry,
    )
    probability_hist = Histogram(
        "churn_prediction_probability",
        "Distribution of served churn probabilities (drift signal)",
        buckets=[i / 10 for i in range(11)],
        registry=registry,
    )
    shadow_rows_total = Counter(
        "churn_shadow_comparisons_total",
        "Rows scored by the shadow challenger",
        registry=registry,
    )

    app = FastAPI(
        title="Churn prediction service",
        version="0.1.0",
        description="XGBoost churn classifier with Prometheus metrics and shadow mode",
    )

    def _score(records: list[CustomerFeatures], endpoint: str) -> list[Prediction]:
        frame = pd.DataFrame(
            [r.model_dump() for r in records], columns=config.FEATURE_COLS
        )
        with predict_latency.labels(endpoint).time():
            proba = model.predict_proba(frame)[:, 1]
        predictions_total.labels(endpoint).inc(len(records))
        for p in proba:
            probability_hist.observe(float(p))
        if shadow is not None and shadow.maybe_compare(frame, proba, endpoint):
            shadow_rows_total.inc(len(records))
        return [
            Prediction(
                churn_probability=round(float(p), 6),
                churn=bool(p >= config.DECISION_THRESHOLD),
                model_source=model_source,
            )
            for p in proba
        ]

    @app.get("/health")
    def health() -> dict:
        # model_version is a verifiable identity: registry version number, or a
        # content hash for the bundle — post-redeploy checks compare against it.
        return {
            "status": "ok",
            **model_info,
            "shadow_enabled": shadow is not None,
        }

    @app.post("/predict", response_model=Prediction)
    def predict(features: CustomerFeatures) -> Prediction:
        return _score([features], endpoint="single")[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest) -> BatchResponse:
        predictions = _score(batch.records, endpoint="batch")
        return BatchResponse(count=len(predictions), predictions=predictions)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
