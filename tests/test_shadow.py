"""Shadow-mode tests: deterministic sampling, JSONL logging, failure safety,
and an end-to-end app run with the champion doubling as its own challenger."""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from serving.app import create_app
from serving.shadow import ShadowRouter, build_shadow
from src import config
from tests.conftest import API_PAYLOAD


class StubChallenger:
    """predict_proba returns a fixed positive-class probability; counts calls."""

    def __init__(self, value: float = 0.5, fail: bool = False):
        self.value = value
        self.fail = fail
        self.calls = 0

    def predict_proba(self, X):
        self.calls += 1
        if self.fail:
            raise RuntimeError("challenger exploded")
        pos = np.full(len(X), self.value)
        return np.column_stack([1 - pos, pos])


FRAME = pd.DataFrame({"x": [1.0]})
CHAMPION_PROBA = np.array([0.7])


def test_sampling_is_deterministic_with_seed(tmp_path):
    stub = StubChallenger()
    router = ShadowRouter(stub, 30.0, tmp_path / "log.jsonl", rng=random.Random(42))
    fired = [router.maybe_compare(FRAME, CHAMPION_PROBA, "single") for _ in range(200)]

    # maybe_compare draws exactly one uniform per call, so a same-seeded Random
    # reproduces the sampling decisions exactly.
    reference = random.Random(42)
    expected = [reference.uniform(0.0, 100.0) < 30.0 for _ in range(200)]
    assert fired == expected
    assert stub.calls == sum(fired)
    assert 0 < sum(fired) < 200  # sanity: 30% actually samples a strict subset


def test_full_traffic_logs_jsonl(tmp_path):
    log = tmp_path / "log.jsonl"
    router = ShadowRouter(StubChallenger(0.2), 100.0, log, rng=random.Random(0))
    frame = pd.DataFrame({"x": [1.0, 2.0]})
    assert router.maybe_compare(frame, np.array([0.7, 0.1]), "batch") is True

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 2
    assert lines[0]["endpoint"] == "batch"
    assert lines[0]["champion_proba"] == 0.7
    assert lines[0]["challenger_proba"] == 0.2
    assert lines[0]["abs_diff"] == 0.5
    assert lines[1]["row"] == 1


def test_challenger_failure_never_raises(tmp_path):
    router = ShadowRouter(
        StubChallenger(fail=True), 100.0, tmp_path / "log.jsonl", rng=random.Random(0)
    )
    assert router.maybe_compare(FRAME, CHAMPION_PROBA, "single") is False


def test_build_shadow_disabled_paths(tmp_path, monkeypatch):
    # pct = 0 -> disabled regardless of files
    monkeypatch.setenv("CHALLENGER_TRAFFIC_PCT", "0")
    assert build_shadow() is None
    # pct > 0 but no challenger bundle -> disabled, not an error
    monkeypatch.setenv("CHALLENGER_TRAFFIC_PCT", "25")
    monkeypatch.setenv("CHALLENGER_MODEL_PATH", str(tmp_path / "missing.joblib"))
    assert build_shadow() is None


def test_app_shadow_end_to_end(tmp_path, monkeypatch):
    """champion-as-challenger: comparison must log with abs_diff exactly 0."""
    log = tmp_path / "comparisons.jsonl"
    monkeypatch.setenv("CHALLENGER_TRAFFIC_PCT", "100")
    monkeypatch.setenv(
        "CHALLENGER_MODEL_PATH", str(config.SERVING_MODEL_DIR / "model.joblib")
    )
    monkeypatch.setenv("SHADOW_LOG_PATH", str(log))
    monkeypatch.setenv("SHADOW_SEED", "7")
    monkeypatch.delenv("MODEL_SOURCE", raising=False)

    client = TestClient(create_app())
    assert client.get("/health").json()["shadow_enabled"] is True

    r = client.post("/predict", json=API_PAYLOAD)
    assert r.status_code == 200

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["champion_proba"] == r.json()["churn_probability"]
    assert lines[0]["abs_diff"] == 0.0

    assert "churn_shadow_comparisons_total 1.0" in client.get("/metrics").text
