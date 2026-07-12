"""Round-trip verification: load the champion back FROM THE REGISTRY and prove
its predictions match the local model on 5 held-out test rows.

Run in a fresh Python process. Reads credentials from the environment only.
Champion resolution (alias vs stage) is delegated to ``src.model_resolver`` —
the single place that answers that question.
"""
from __future__ import annotations

import json
import os
import sys

import joblib
import mlflow
import numpy as np

from src import config, data
from src.model_resolver import load_champion


def main() -> int:
    missing = [
        k
        for k in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD")
        if not os.environ.get(k)
    ]
    if missing:
        print(f"ABORT: missing env vars: {missing}")
        return 2

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

    # Same seed => identical held-out test split as training time.
    splits = data.split(data.load_raw(), seed=config.RANDOM_SEED)
    rows = splits.X_test.iloc[:5]

    local_model = joblib.load(config.MODELS_DIR / "model.joblib")
    local_proba = local_model.predict_proba(rows)[:, 1]

    try:
        registry_model, used_uri = load_champion()
    except LookupError as exc:
        print(f"ABORT: {exc}")
        return 3

    registry_proba = registry_model.predict_proba(rows)[:, 1]
    identical = bool(np.allclose(local_proba, registry_proba, rtol=0, atol=1e-12))

    print(
        json.dumps(
            {
                "registry_uri_used": used_uri,
                "test_row_indices": [int(i) for i in rows.index],
                "local_proba": [round(float(p), 10) for p in local_proba],
                "registry_proba": [round(float(p), 10) for p in registry_proba],
                "max_abs_diff": float(np.max(np.abs(local_proba - registry_proba))),
                "identical": identical,
            },
            indent=2,
        )
    )
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
