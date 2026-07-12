"""API tests for the serving app, run against the committed bundled model."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from serving.app import create_app
from serving.schemas import MAX_BATCH_ROWS
from tests.conftest import API_PAYLOAD


@pytest.fixture(scope="module")
def client():
    """One app per module (model loads once); shadow/registry env cleared."""
    mp = pytest.MonkeyPatch()
    for var in ("CHALLENGER_TRAFFIC_PCT", "MODEL_SOURCE", "SHADOW_SEED"):
        mp.delenv(var, raising=False)
    with TestClient(create_app()) as c:
        yield c
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_source"] == "bundled:model.joblib"
    assert body["shadow_enabled"] is False


def test_predict_single(client):
    r = client.post("/predict", json=API_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn"] == (body["churn_probability"] >= 0.5)
    assert body["model_source"] == "bundled:model.joblib"


def test_predict_deterministic(client):
    p1 = client.post("/predict", json=API_PAYLOAD).json()["churn_probability"]
    p2 = client.post("/predict", json=API_PAYLOAD).json()["churn_probability"]
    assert p1 == p2


def test_predict_batch(client):
    loyal = {
        **API_PAYLOAD,
        "tenure": 70,
        "Contract": "Two year",
        "InternetService": "DSL",
        "TotalCharges": 4000.0,
    }
    r = client.post("/predict/batch", json={"records": [API_PAYLOAD, loyal, API_PAYLOAD]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    # identical rows score identically; the loyal profile should score lower
    assert body["predictions"][0] == body["predictions"][2]
    assert (
        body["predictions"][1]["churn_probability"]
        < body["predictions"][0]["churn_probability"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"InternetService": "Dialup"},  # outside CATEGORICAL_DOMAINS
        {"SeniorCitizen": 2},
        {"tenure": -1},
        {"MonthlyCharges": -5.0},
        {"unexpected_field": 1},  # extra="forbid"
    ],
)
def test_predict_rejects_invalid(client, mutation):
    r = client.post("/predict", json={**API_PAYLOAD, **mutation})
    assert r.status_code == 422


def test_predict_rejects_missing_field(client):
    payload = dict(API_PAYLOAD)
    del payload["tenure"]
    assert client.post("/predict", json=payload).status_code == 422


def test_batch_rejects_oversize_and_empty(client):
    too_big = {"records": [API_PAYLOAD] * (MAX_BATCH_ROWS + 1)}
    assert client.post("/predict/batch", json=too_big).status_code == 422
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_metrics_exposed(client):
    client.post("/predict", json=API_PAYLOAD)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert 'churn_predictions_total{endpoint="single"}' in r.text
    assert "churn_predict_latency_seconds" in r.text
    assert "churn_prediction_probability_bucket" in r.text
