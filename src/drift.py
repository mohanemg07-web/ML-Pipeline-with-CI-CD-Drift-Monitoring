"""Evidently-based drift check: PSI per feature vs the training reference.

Outputs a numeric summary (``eval/results/drift_report.json``) and a full
Evidently HTML report (``monitoring/drift_report.html``, gitignored artifact).
The retrain-trigger decision itself is a plain function over the PSI dict so
it can be unit-tested without Evidently.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import config


@dataclass
class DriftDecision:
    psi_by_feature: dict[str, float]
    threshold: float
    min_features: int
    drifted_features: list[str] = field(default_factory=list)
    detected: bool = False


def compute_psi_scores(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
) -> tuple[dict[str, float], object]:
    """Per-feature PSI via Evidently; returns ``(scores, report)``.

    Both frames must already be model-ready (cleaned numeric types).
    """
    from evidently.metrics import ColumnDriftMetric
    from evidently.report import Report

    features = features or config.FEATURE_COLS
    report = Report(
        metrics=[
            ColumnDriftMetric(
                column_name=c,
                stattest="psi",
                stattest_threshold=config.DRIFT_PSI_THRESHOLD,
            )
            for c in features
        ]
    )
    report.run(reference_data=reference[features], current_data=current[features])

    scores: dict[str, float] = {}
    for metric in report.as_dict()["metrics"]:
        result = metric["result"]
        scores[result["column_name"]] = round(float(result["drift_score"]), 6)
    return scores, report


def evaluate_drift(
    psi_scores: dict[str, float],
    threshold: float = config.DRIFT_PSI_THRESHOLD,
    min_features: int = config.DRIFT_MIN_FEATURES,
) -> DriftDecision:
    """Pure trigger logic: drift detected iff > threshold on >= min_features."""
    drifted = sorted(
        (f for f, v in psi_scores.items() if v > threshold),
        key=lambda f: -psi_scores[f],
    )
    return DriftDecision(
        psi_by_feature=dict(psi_scores),
        threshold=threshold,
        min_features=min_features,
        drifted_features=drifted,
        detected=len(drifted) >= min_features,
    )


def run_drift_check(
    reference: pd.DataFrame,
    batch: pd.DataFrame,
    json_out: Path = config.DRIFT_REPORT_JSON,
    html_out: Path = config.DRIFT_REPORT_HTML,
) -> DriftDecision:
    """Full check: PSI scores -> decision -> JSON summary + HTML report."""
    scores, report = compute_psi_scores(reference, batch)
    decision = evaluate_drift(scores)

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                "n_reference": int(len(reference)),
                "n_batch": int(len(batch)),
                **asdict(decision),
            },
            indent=2,
        )
    )
    html_out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(html_out))
    return decision
