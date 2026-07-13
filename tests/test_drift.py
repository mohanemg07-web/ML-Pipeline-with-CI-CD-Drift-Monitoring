"""Drift module tests: pure PSI-threshold logic with synthetic dicts, plus
Evidently integration on synthetic frames (no drift vs strong drift), plus
the simulator's guarantees."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data
from src.drift import compute_psi_scores, evaluate_drift
from src.simulate_drift import DRIFTED_FEATURES, simulate_batch

# ---------------------------------------------------------------------------
# evaluate_drift: pure threshold/counting logic
# ---------------------------------------------------------------------------


def test_evaluate_drift_counts_only_above_threshold():
    scores = {"a": 0.5, "b": 0.25, "c": 0.05, "d": 0.19}
    decision = evaluate_drift(scores, threshold=0.2, min_features=2)
    assert decision.drifted_features == ["a", "b"]  # sorted by PSI desc
    assert decision.detected is True


def test_evaluate_drift_below_min_features_not_detected():
    scores = {"a": 0.9, "b": 0.1, "c": 0.1}
    decision = evaluate_drift(scores, threshold=0.2, min_features=2)
    assert decision.drifted_features == ["a"]
    assert decision.detected is False


def test_evaluate_drift_threshold_is_strict():
    # exactly-at-threshold does NOT count as drifted
    decision = evaluate_drift({"a": 0.2, "b": 0.2}, threshold=0.2, min_features=1)
    assert decision.drifted_features == []
    assert decision.detected is False


def test_evaluate_drift_no_scores():
    decision = evaluate_drift({}, threshold=0.2, min_features=1)
    assert decision.detected is False


# ---------------------------------------------------------------------------
# Evidently PSI integration on synthetic frames
# ---------------------------------------------------------------------------


def _numeric_frame(rng, loc, n=800):
    return pd.DataFrame(
        {
            "num1": rng.normal(loc, 1.0, n),
            "num2": rng.uniform(0, 1, n),
        }
    )


def test_psi_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    ref, cur = _numeric_frame(rng, 0.0), _numeric_frame(rng, 0.0)
    scores, _ = compute_psi_scores(ref, cur, features=["num1", "num2"])
    assert all(v < 0.1 for v in scores.values()), scores
    assert evaluate_drift(scores).detected is False


def test_psi_detects_strong_shift():
    rng = np.random.default_rng(0)
    ref, cur = _numeric_frame(rng, 0.0), _numeric_frame(rng, 2.5)  # big mean shift
    scores, _ = compute_psi_scores(ref, cur, features=["num1", "num2"])
    assert scores["num1"] > config.DRIFT_PSI_THRESHOLD
    assert scores["num2"] < config.DRIFT_PSI_THRESHOLD  # undrifted control


# ---------------------------------------------------------------------------
# Simulator guarantees
# ---------------------------------------------------------------------------


@pytest.fixture
def batch_pair(real_sample_df):
    batch = simulate_batch(real_sample_df, n=400, seed=7, severity=1.0)
    return real_sample_df, batch


def test_simulator_preserves_schema_and_labels(batch_pair):
    raw, batch = batch_pair
    assert list(batch.columns) == list(raw.columns)
    assert set(batch[config.TARGET_COL].unique()) <= {"Yes", "No"}
    assert (batch["tenure"] >= 0).all()
    # values stay inside the GE categorical domains
    for col, domain in config.CATEGORICAL_DOMAINS.items():
        assert set(batch[col].unique()) <= set(domain), col


def test_simulator_severity_zero_is_identity_shaped(real_sample_df):
    batch = simulate_batch(real_sample_df, n=400, seed=7, severity=0.0)
    src = real_sample_df.sample(n=400, random_state=7)
    # tenure / categoricals untouched at severity 0 (TotalCharges is recomputed)
    assert (batch["tenure"].values == src["tenure"].values).all()
    assert (batch["InternetService"].values == src["InternetService"].values).all()


def test_simulator_rejects_bad_severity(real_sample_df):
    with pytest.raises(ValueError):
        simulate_batch(real_sample_df, severity=1.5)


def test_simulated_drift_is_detected_by_psi(real_sample_df):
    """End-to-end a/b: the simulator's shift must trip the Evidently check."""
    batch = simulate_batch(real_sample_df, n=450, seed=7, severity=1.0)
    ref = data.clean(real_sample_df)[config.FEATURE_COLS]
    cur = data.clean(batch)[config.FEATURE_COLS]
    scores, _ = compute_psi_scores(ref, cur)
    decision = evaluate_drift(scores)
    assert decision.detected is True
    # the drifted features must dominate the detections
    assert set(decision.drifted_features) & set(DRIFTED_FEATURES)
