"""Model loading for the serving app.

Primary deployment path: the bundled ``serving/model/model.joblib`` COPYed into
the image at build time. Pulling the champion from the MLflow registry
(``MODEL_SOURCE=registry``) is an optional enhancement — it is bounded by a
hard per-attempt timeout (a hung DagsHub connection can never stall container
startup) and ALWAYS falls back to the bundle on any failure.
"""
from __future__ import annotations

import logging
import os

import joblib

from src import config
from src.model_resolver import load_champion

logger = logging.getLogger(__name__)

BUNDLED_MODEL_PATH = config.SERVING_MODEL_DIR / "model.joblib"
DEFAULT_REGISTRY_TIMEOUT_S = 5.0


def load_serving_model() -> tuple[object, str]:
    """Return ``(model, source_label)``; source is surfaced in /health responses."""
    if os.environ.get("MODEL_SOURCE", "bundled").lower() == "registry":
        timeout_s = float(os.environ.get("REGISTRY_LOAD_TIMEOUT_S", DEFAULT_REGISTRY_TIMEOUT_S))
        try:
            tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            if not tracking_uri:
                raise LookupError("MODEL_SOURCE=registry but MLFLOW_TRACKING_URI is unset")
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)
            model, resolved_uri = load_champion(timeout_s=timeout_s)
            return model, f"registry:{resolved_uri}"
        except Exception as exc:  # noqa: BLE001 — any registry failure -> bundled fallback
            logger.warning("registry load failed (%s); falling back to bundled model", exc)

    model = joblib.load(BUNDLED_MODEL_PATH)
    return model, "bundled:model.joblib"
