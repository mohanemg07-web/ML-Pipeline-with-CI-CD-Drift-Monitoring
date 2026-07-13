"""Retrain-loop logic tests, all outside Airflow and without network/registry:
branch choice, promotion gate, smoke test, redeploy verification outcomes
(stubbed HTTP), stage timings, and a small real fit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.retrain import (
    auc_pair,
    choose_branch,
    fit_retrained,
    prepare_retrain_data,
    promotion_gate,
    redeploy_and_verify,
    smoke_test,
    stage_timings,
)

# ---------------------------------------------------------------------------
# Branch + gate + smoke: pure logic
# ---------------------------------------------------------------------------


def test_choose_branch():
    assert choose_branch(True) == "retrain"
    assert choose_branch(False) == "no_retrain"


def test_promotion_gate_pass_when_better():
    ok, reason = promotion_gate(new_auc=0.85, stale_auc=0.80)
    assert ok and reason.startswith("PASS")


def test_promotion_gate_pass_within_tolerance():
    ok, _ = promotion_gate(new_auc=0.79, stale_auc=0.80, tolerance=0.02)
    assert ok


def test_promotion_gate_fail_beyond_tolerance():
    ok, reason = promotion_gate(new_auc=0.75, stale_auc=0.80, tolerance=0.02)
    assert not ok and reason.startswith("FAIL")


def test_promotion_gate_boundary_exact():
    ok, _ = promotion_gate(new_auc=0.78, stale_auc=0.80, tolerance=0.02)
    assert ok  # new_auc == stale - tolerance passes (>=)


class _StubModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict_proba(self, X):
        p = self._proba[: len(X)]
        return np.column_stack([1 - p, p])


ROWS = pd.DataFrame({"x": range(5)})


def test_smoke_test_pass():
    ok, reason = smoke_test(_StubModel([0.1, 0.5, 0.9, 0.0, 1.0]), ROWS)
    assert ok and reason.startswith("PASS")


def test_smoke_test_fails_on_nan():
    ok, reason = smoke_test(_StubModel([0.1, np.nan, 0.9, 0.2, 0.3]), ROWS)
    assert not ok and "non-finite" in reason


def test_smoke_test_fails_on_exception():
    class Broken:
        def predict_proba(self, X):
            raise RuntimeError("boom")

    ok, reason = smoke_test(Broken(), ROWS)
    assert not ok and "raised" in reason


def test_smoke_test_fails_on_wrong_length():
    ok, _ = smoke_test(_StubModel([0.1, 0.2]), ROWS)
    assert not ok


# ---------------------------------------------------------------------------
# redeploy_and_verify: the three honest outcomes, no real HTTP
# ---------------------------------------------------------------------------


class _Resp:
    def __init__(self, status_code):
        self.status_code = status_code


def _verify(healths, expected="2", hook_status=200, timeout_s=10.0):
    """Run redeploy_and_verify against a scripted sequence of health payloads."""
    seq = iter(healths)
    last = healths[-1] if healths else {}

    def fetch_health():
        return next(seq, last)

    return redeploy_and_verify(
        expected_version=expected,
        hook_url="http://hook.invalid",
        base_url="http://svc.invalid",
        post=lambda url: _Resp(hook_status),
        fetch_health=fetch_health,
        sleep=lambda s: None,
        timeout_s=timeout_s,
        interval_s=1.0,
    )


OLD = {"model_source": "registry:models:/churn-xgboost@champion", "model_version": "1"}
NEW = {"model_source": "registry:models:/churn-xgboost@champion", "model_version": "2"}
BUNDLED = {"model_source": "bundled:model.joblib", "model_version": "file-sha256:abc"}


def test_verify_new_model_live_after_swap():
    result = _verify([OLD, OLD, NEW])
    assert result["outcome"] == "new_model_live"
    assert result["polls"] == 3
    assert result["verified_ts"] is not None


def test_verify_fallback_bundled_is_partial_failure():
    result = _verify([OLD, BUNDLED])
    assert result["outcome"] == "fallback_bundled"
    assert "registry pull failed" in result["reason"]
    assert result["last_health"] == BUNDLED


def test_verify_timeout_when_old_version_persists():
    result = _verify([OLD], timeout_s=3.0)
    assert result["outcome"] == "timeout"
    assert result["last_health"] == OLD


def test_verify_hook_failure_reported():
    result = _verify([NEW], hook_status=500)
    assert result["outcome"] == "hook_failed"
    assert result["hook_status"] == 500
    assert result["polls"] == 0


def test_stage_timings_breakdown():
    t = stage_timings(
        detection_ts="2026-07-13T10:00:00+00:00",
        retrain_done_ts="2026-07-13T10:04:00+00:00",
        hook_fired_ts="2026-07-13T10:04:30+00:00",
        verified_ts="2026-07-13T10:12:30+00:00",
    )
    assert t["detection_to_retrain_done_s"] == 240.0
    assert t["retrain_done_to_hook_fired_s"] == 30.0
    assert t["hook_fired_to_verified_live_s"] == 480.0
    assert t["total_detection_to_live_s"] == 750.0


def test_stage_timings_without_verification():
    t = stage_timings(
        detection_ts="2026-07-13T10:00:00+00:00",
        retrain_done_ts="2026-07-13T10:04:00+00:00",
        hook_fired_ts="2026-07-13T10:04:30+00:00",
        verified_ts=None,
    )
    assert "total_detection_to_live_s" not in t


# ---------------------------------------------------------------------------
# Data prep + a small real fit (no MLflow, no registry)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prep_ctx():
    from src.simulate_drift import simulate_batch
    from tests.conftest import FIXTURE_SAMPLE

    if not FIXTURE_SAMPLE.exists():
        pytest.skip("no committed sample fixture")
    raw = pd.read_csv(FIXTURE_SAMPLE)
    batch = simulate_batch(raw, n=300, seed=7, severity=1.0)
    return len(raw), prepare_retrain_data(raw, batch, seed=config.RANDOM_SEED)


def test_prepare_retrain_data_shapes(prep_ctx):
    n_raw, prepared = prep_ctx
    s = prepared.sizes
    assert s["batch_train"] + s["batch_eval"] == 300
    # fit = reference train+val (raw minus original test) + batch train portion
    assert s["fit"] == (n_raw - s["orig_test"]) + s["batch_train"]
    # both eval sets are stratified non-degenerate: both classes present
    assert prepared.y_batch_eval.nunique() == 2
    assert prepared.y_orig_test.nunique() == 2


def test_fit_retrained_small_produces_scoring_model(prep_ctx):
    _, prepared = prep_ctx
    pipeline, best_params = fit_retrained(
        prepared.X_fit, prepared.y_fit, n_trials=2, seed=config.RANDOM_SEED
    )
    assert best_params  # search actually ran
    metrics = auc_pair(pipeline, prepared.X_batch_eval, prepared.y_batch_eval)
    assert 0.5 <= metrics["roc_auc"] <= 1.0
    ok, reason = smoke_test(pipeline, prepared.X_batch_eval.head(5))
    assert ok, reason
